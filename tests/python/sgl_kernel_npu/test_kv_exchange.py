giimport unittest

import sgl_kernel_npu  # noqa: F401
import torch
from sgl_kernel_npu.hicache_offload import (
    HOST_LAYOUT_PAGE_FIRST,
    KV_EXCHANGE_DIRECTION_D2H,
    KV_EXCHANGE_DIRECTION_H2D,
    build_kv_exchange_component,
    build_kv_exchange_meta,
    create_hugepage_tensor,
    kv_exchange,
)

PAGE_SIZE = 4
BLOCK_DIM = 64  # Ascend950: 64 AIV cores, cores 62/63 participate


class TestKvExchange(unittest.TestCase):
    """Functional tests for the AIV kv_exchange kernel (page_first host,
    layer_first device), ported from MemFabric acc_offload.

    Layouts mirror the HiCache MLA page_first_kv_split pool:
      host   [page, layer, page_size, 1, width]  (page-first)
      device [layer, page, page_size, 1, width]  (layer-first)
    """

    def _run_exchange(
        self,
        direction: int,
        num_components: int,
        width: int,
        num_pages: int,
        dtype=torch.bfloat16,
        skip_empty_component: bool = False,
    ):
        torch.npu.set_device(0)
        layers = 2

        device_shape = (layers, num_pages, PAGE_SIZE, 1, width)
        host_shape = (num_pages, layers, PAGE_SIZE, 1, width)
        device_size = layers * num_pages * PAGE_SIZE * width

        # Host buffers must be device-visible for the AIV kernel; the
        # hugepage allocator provides exactly that.
        host_k, _, host_k_dev = create_hugepage_tensor(host_shape, dtype)
        host_v, _, host_v_dev = create_hugepage_tensor(host_shape, dtype)

        device_k = torch.zeros(device_shape, dtype=dtype, device="npu")
        device_v = torch.zeros(device_shape, dtype=dtype, device="npu")

        # Distinct per (layer, page) values so wrong strides/offsets fail.
        # Logical values are built in the DEVICE layout [layer, page, ...];
        # the page-first host expectation is a transposed view of the same
        # logical data.
        base = (
            torch.arange(layers * num_pages, dtype=torch.float32)
            .add_(1.0)
            .view(layers, num_pages)
        )
        src_k = torch.empty(device_shape, dtype=dtype)
        src_v = torch.empty(device_shape, dtype=dtype)
        for layer in range(layers):
            for page in range(num_pages):
                val = base[layer][page].item()
                src_k[layer, page] = val
                src_v[layer, page] = val * 2
        src_k_host = src_k.permute(1, 0, 2, 3, 4).contiguous()
        src_v_host = src_v.permute(1, 0, 2, 3, 4).contiguous()
        if direction == KV_EXCHANGE_DIRECTION_H2D:
            host_k.copy_(src_k_host)
            host_v.copy_(src_v_host)
        else:
            device_k.copy_(src_k.to("npu"))
            device_v.copy_(src_v.to("npu"))

        components = [
            build_kv_exchange_component(
                device_base=device_k.data_ptr(),
                host_base=host_k_dev,
                device_layer_pitch=device_k.stride(0) * dtype.itemsize,
                device_page_pitch=device_k.stride(1) * dtype.itemsize,
                host_page_pitch=host_k.stride(0) * dtype.itemsize,
                host_layer_pitch=host_k.stride(1) * dtype.itemsize,
                width=host_k.stride(2) * dtype.itemsize * PAGE_SIZE,
                layer_lo=0,
                layer_hi=layers,
            ),
            build_kv_exchange_component(
                device_base=device_v.data_ptr(),
                host_base=host_v_dev,
                device_layer_pitch=device_v.stride(0) * dtype.itemsize,
                device_page_pitch=device_v.stride(1) * dtype.itemsize,
                host_page_pitch=host_v.stride(0) * dtype.itemsize,
                host_layer_pitch=host_v.stride(1) * dtype.itemsize,
                width=host_v.stride(2) * dtype.itemsize * PAGE_SIZE,
                layer_lo=0,
                layer_hi=layers,
            ),
        ]

        if num_components == 4:
            # Third component transfers nothing (empty layer range) -- the
            # walkers must skip it; the fourth one is live.
            empty_host, _, empty_host_dev = create_hugepage_tensor(host_shape, dtype)
            empty_host.zero_()
            components.append(
                build_kv_exchange_component(
                    device_base=device_k.data_ptr(),
                    host_base=empty_host_dev,
                    device_layer_pitch=device_k.stride(0) * dtype.itemsize,
                    device_page_pitch=device_k.stride(1) * dtype.itemsize,
                    host_page_pitch=empty_host.stride(0) * dtype.itemsize,
                    host_layer_pitch=empty_host.stride(1) * dtype.itemsize,
                    width=empty_host.stride(2) * dtype.itemsize * PAGE_SIZE,
                    layer_lo=1,
                    layer_hi=1,  # empty: lo >= hi
                )
            )
            components.append(components[0])  # k again as the live 4th

        # Identity page mapping: host page p <-> device page p. Indices live
        # on the device (the kernel dereferences them there).
        device_indices = torch.arange(
            num_pages * PAGE_SIZE, dtype=torch.int64, device="npu"
        )
        host_indices = torch.arange(
            num_pages * PAGE_SIZE, dtype=torch.int64, device="npu"
        )

        vals = build_kv_exchange_meta(
            num_pages=num_pages,
            page_size=PAGE_SIZE,
            direction=direction,
            device_indices_ptr=device_indices.data_ptr(),
            host_indices_ptr=host_indices.data_ptr(),
            host_layout_mode=HOST_LAYOUT_PAGE_FIRST,
            components=components,
        )
        meta = torch.tensor(vals, dtype=torch.int64, device="npu")

        kv_exchange(meta, block_dim=BLOCK_DIM)
        torch.npu.synchronize()

        if direction == KV_EXCHANGE_DIRECTION_H2D:
            self.assertTrue(
                torch.equal(device_k, src_k.to("npu")),
                "device k must equal host k after H2D",
            )
            self.assertTrue(
                torch.equal(device_v, src_v.to("npu")),
                "device v must equal host v after H2D",
            )
        else:
            # Pure-CPU compare against the page-first expectation.
            self.assertTrue(
                torch.equal(host_k, src_k_host),
                "host k must equal device k after D2H",
            )
            self.assertTrue(
                torch.equal(host_v, src_v_host),
                "host v must equal device v after D2H",
            )
        if num_components == 4:
            self.assertEqual(
                empty_host.abs().sum().item(),
                0,
                "empty component must transfer nothing",
            )

    def test_h2d_one_component(self):
        self._run_exchange(KV_EXCHANGE_DIRECTION_H2D, 1, width=256, num_pages=8)

    def test_d2h_one_component(self):
        self._run_exchange(KV_EXCHANGE_DIRECTION_D2H, 1, width=256, num_pages=8)

    def test_h2d_two_components(self):
        self._run_exchange(KV_EXCHANGE_DIRECTION_H2D, 2, width=256, num_pages=8)

    def test_d2h_two_components(self):
        self._run_exchange(KV_EXCHANGE_DIRECTION_D2H, 2, width=256, num_pages=8)

    def test_h2d_four_components_with_empty(self):
        self._run_exchange(KV_EXCHANGE_DIRECTION_H2D, 4, width=256, num_pages=8)

    def test_d2h_four_components_with_empty(self):
        self._run_exchange(KV_EXCHANGE_DIRECTION_D2H, 4, width=256, num_pages=8)

    def test_wide_rows_split_across_ub_slots(self):
        # 128KB rows exceed the 88KB UB buffer; the kernel must split them.
        self._run_exchange(
            KV_EXCHANGE_DIRECTION_H2D, 2, width=65536, num_pages=3, dtype=torch.bfloat16
        )

    def test_pages_round_robin_across_cores(self):
        # 70 pages > 64 blockDim: round-robin (page += aivNum) must wrap.
        self._run_exchange(KV_EXCHANGE_DIRECTION_H2D, 2, width=64, num_pages=70)

    def test_meta_validation(self):
        with self.assertRaises(ValueError):
            build_kv_exchange_meta(
                1, PAGE_SIZE, 3, 0, 0, HOST_LAYOUT_PAGE_FIRST, [[0] * 9]
            )
        with self.assertRaises(ValueError):
            build_kv_exchange_meta(1, PAGE_SIZE, KV_EXCHANGE_DIRECTION_H2D, 0, 0, 7,
                                   [[0] * 9])
        with self.assertRaises(ValueError):
            build_kv_exchange_meta(
                1, PAGE_SIZE, KV_EXCHANGE_DIRECTION_H2D, 0, 0,
                HOST_LAYOUT_PAGE_FIRST, [[0] * 9] * 5,
            )


if __name__ == "__main__":
    unittest.main()
