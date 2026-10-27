import unittest

import torch

from sgl_kernel_npu.kvcacheio import (
    HostMemoryMapping,
    empty_host_memory,
    get_device_accessible_ptr,
    initialize_host_memory,
    register_host_memory,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_per_layer_direct_pf_lf,
    uninitialize_host_memory,
)

# host: page-first  (pages, layers, page_size, 1, width)
# device: layer-first (layers, pages, page_size, 1, width)
# The quantized index_k and FP32 scale cover fewer layers than k/v.
PAGES = 8
LAYERS = 4
PS = 128
K_WIDTH = 656
V_WIDTH = 64
IDX_LAYERS = 2
IDX_WIDTH = 128
ROUNDTRIP_ROUNDS = 20
# A3 can round each MF allocation up to 1 GiB. The roundtrip allocates four
# registered blocks, so reserve enough space for that form as well as 2 MiB VMM.
MF_RESERVE_SIZE = 8 * 1024**3


def _unique_like(shape):
    """Deterministic values exactly representable in bf16/fp32.

    Each (page, layer) block has a distinct base value and a per-token ripple,
    so a wrong page, layer, token, or row width changes the compared values.
    """
    pages, layers, ps = shape[0], shape[1], shape[2]
    assert pages * layers * 4 <= 256, "block ids must stay unique and bf16-exact"
    page = torch.arange(pages, dtype=torch.float32).view(pages, 1, 1, 1, 1)
    layer = torch.arange(layers, dtype=torch.float32).view(1, layers, 1, 1, 1)
    token = torch.arange(ps, dtype=torch.float32).view(1, 1, ps, 1, 1)
    vals = ((page * layers + layer) * 4 + token % 4) % 256 - 128
    if len(shape) == 4:
        return vals.expand(*shape, 1).squeeze(-1)
    return vals.expand(shape)


def _host_pool(shape, dtype=torch.bfloat16):
    """Allocate and register one MF host block, then fill it with unique data."""
    buf = empty_host_memory(shape, dtype=dtype)
    buf.copy_(_unique_like(shape).to(dtype))
    mapping = register_host_memory(buf)
    return buf, mapping


def _device_pool(shape, dtype=torch.bfloat16):
    """Create a layer-first device pool from a page-first shape."""
    permute = (1, 0, *range(2, len(shape)))
    return _unique_like(shape).permute(permute).contiguous().to(dtype).to("npu")


def _page_pairs():
    """Return random non-contiguous, page-aligned token-slot mappings.

    The kernel derives page numbers from token slots. Callers must pass the
    original token-slot indices without dividing by page_size themselves.
    """
    offset = torch.arange(PS, dtype=torch.int64).repeat(PAGES)
    dev_pages = torch.randperm(PAGES, dtype=torch.int64)
    host_pages = torch.randperm(PAGES, dtype=torch.int64)
    dev_idx = (dev_pages * PS).repeat_interleave(PS) + offset
    host_idx = (host_pages * PS).repeat_interleave(PS) + offset
    return dev_idx.to("npu"), host_idx.to("npu")


class TestKvExchangeMF(unittest.TestCase):

    def setUp(self):
        torch.npu.set_device(0)
        initialize_host_memory(device_id=0, reserve_size=MF_RESERVE_SIZE)

    def tearDown(self):
        # Do not unregister MF memory while an asynchronously launched AIV
        # kernel can still access it, including when a test fails mid-loop.
        torch.npu.synchronize()
        # MF registered blocks cannot be freed individually. Uninitialize the
        # host-memory entry and clear the Python DVA registry after every case.
        uninitialize_host_memory()

    def _verify_pages(self, host, dev, dev_idx, host_idx, host_layer, dev_layer):
        for p in range(PAGES):
            dev_page = int(dev_idx[p * PS]) // PS
            host_page = int(host_idx[p * PS]) // PS
            torch.testing.assert_close(
                dev[dev_layer, dev_page].reshape(PS, -1).cpu(),
                host[host_page, host_layer].reshape(PS, -1).cpu(),
                rtol=0,
                atol=0,
                msg=(
                    f"mismatch dev_layer={dev_layer} host_layer={host_layer} "
                    f"page={p}"
                ),
            )

    def _verify_all_layers(self, host, dev, dev_idx, host_idx, layers):
        for layer in range(layers):
            self._verify_pages(host, dev, dev_idx, host_idx, layer, layer)

    def _verify_untouched_zero(self, dev, layer, layers, name):
        for untouched in range(layer + 1, layers):
            self.assertEqual(
                int(torch.count_nonzero(dev[untouched])),
                0,
                msg=f"{name}: layer {untouched} polluted by layer {layer}",
            )

    def _run_h2d(self, host_tensors, dev_tensors, layers):
        dev_idx, host_idx = _page_pairs()
        for layer in range(layers):
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=host_tensors,
                dst_ptrs=dev_tensors,
                src_indices=host_idx,
                dst_indices=dev_idx,
                layer_id=layer,
                page_size=PS,
            )
            torch.npu.synchronize()
            for name, host, dev in zip(
                ("k", "v", "index_k", "scale"), host_tensors, dev_tensors
            ):
                self._verify_pages(host, dev, dev_idx, host_idx, layer, layer)
                self._verify_untouched_zero(dev, layer, layers, name)

    def _run_d2h(self, host_tensors, dev_tensors):
        for host in host_tensors:
            host.zero_()
        dev_idx, host_idx = _page_pairs()
        transfer_kv_all_layer_direct_lf_pf(
            src_ptrs=dev_tensors,
            dst_ptrs=host_tensors,
            src_indices=dev_idx,
            dst_indices=host_idx,
            page_size=PS,
        )
        torch.npu.synchronize()
        for host, dev in zip(host_tensors, dev_tensors):
            self._verify_all_layers(host, dev, dev_idx, host_idx, dev.shape[0])

    def test_mf_register_dva_and_unregistered_rejected(self):
        buf = empty_host_memory((4096,), dtype=torch.bfloat16)
        hva = buf.data_ptr()
        mapping = register_host_memory(buf)

        self.assertIsInstance(mapping, HostMemoryMapping)
        self.assertTrue(mapping.owned)
        self.assertEqual(mapping.registered_host_ptr, hva)
        self.assertNotEqual(mapping.device_ptr, 0)
        self.assertEqual(get_device_accessible_ptr(buf, 0), mapping.device_ptr)

        offset = 2048
        view = buf[offset : offset + 16]
        self.assertEqual(
            get_device_accessible_ptr(view, 0),
            mapping.device_ptr + offset * view.element_size(),
        )

        # An MF-allocated but unregistered block must fail in the Python
        # resolver, before an AIV launch or any aclrt fallback.
        unregistered = empty_host_memory((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev = torch.zeros(
            (LAYERS, PAGES, PS, 1, K_WIDTH),
            dtype=torch.bfloat16,
            device="npu",
        )
        dev_idx, host_idx = _page_pairs()
        with self.assertRaisesRegex(RuntimeError, "MF host memory is not registered"):
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[unregistered],
                dst_ptrs=[dev],
                src_indices=host_idx,
                dst_indices=dev_idx,
                layer_id=0,
                page_size=PS,
            )

    def test_kv_copy_h2d(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        dev_k = torch.zeros(
            (LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        dev_v = torch.zeros(
            (LAYERS, PAGES, PS, 1, V_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        self._run_h2d([host_k, host_v], [dev_k, dev_v], LAYERS)

    def test_kv_index_k_copy_h2d(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        dev_k = torch.zeros(
            (LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        dev_v = torch.zeros(
            (LAYERS, PAGES, PS, 1, V_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        dev_idx_k = torch.zeros(
            (IDX_LAYERS, PAGES, PS, 1, IDX_WIDTH),
            dtype=torch.bfloat16,
            device="npu",
        )
        self._run_h2d([host_k, host_v], [dev_k, dev_v], LAYERS)
        self._run_h2d([host_idx_k], [dev_idx_k], IDX_LAYERS)

    def test_k_copy_h2d(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_k = torch.zeros(
            (LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        self._run_h2d([host_k], [dev_k], LAYERS)

    def test_scale_copy_h2d(self):
        host_scale, _ = _host_pool(
            (PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32
        )
        dev_scale = torch.zeros(
            (IDX_LAYERS, PAGES, PS, 1), dtype=torch.float32, device="npu"
        )
        self._run_h2d([host_scale], [dev_scale], IDX_LAYERS)

    def test_kv_copy_d2h(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        dev_k = _device_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_v = _device_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        self._run_d2h([host_k, host_v], [dev_k, dev_v])

    def test_kv_index_k_copy_d2h(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        dev_k = _device_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_v = _device_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        dev_idx_k = _device_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        self._run_d2h(
            [host_k, host_v, host_idx_k], [dev_k, dev_v, dev_idx_k]
        )

    def test_k_copy_d2h(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_k = _device_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        self._run_d2h([host_k], [dev_k])

    def test_scale_copy_d2h(self):
        host_scale, _ = _host_pool(
            (PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32
        )
        dev_scale = _device_pool(
            (PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32
        )
        self._run_d2h([host_scale], [dev_scale])

    def test_kv_index_k_scale_roundtrip(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        host_scale, _ = _host_pool(
            (PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32
        )
        dev_k = _device_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_v = _device_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        dev_idx_k = _device_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        dev_scale = _device_pool(
            (PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32
        )
        expected = {
            "k": dev_k.clone(),
            "v": dev_v.clone(),
            "index_k": dev_idx_k.clone(),
            "scale": dev_scale.clone(),
        }

        for round_idx in range(ROUNDTRIP_ROUNDS):
            dev_idx, host_idx = _page_pairs()
            for host in (host_k, host_v, host_idx_k, host_scale):
                host.zero_()

            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=[dev_k, dev_v],
                dst_ptrs=[host_k, host_v],
                src_indices=dev_idx,
                dst_indices=host_idx,
                page_size=PS,
            )
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=[dev_idx_k, dev_scale],
                dst_ptrs=[host_idx_k, host_scale],
                src_indices=dev_idx,
                dst_indices=host_idx,
                page_size=PS,
            )
            torch.npu.synchronize()

            if round_idx == 0:
                self._verify_all_layers(
                    host_k, expected["k"], dev_idx, host_idx, LAYERS
                )
                self._verify_all_layers(
                    host_v, expected["v"], dev_idx, host_idx, LAYERS
                )
                self._verify_all_layers(
                    host_idx_k,
                    expected["index_k"],
                    dev_idx,
                    host_idx,
                    IDX_LAYERS,
                )
                self._verify_all_layers(
                    host_scale,
                    expected["scale"],
                    dev_idx,
                    host_idx,
                    IDX_LAYERS,
                )

            dev_k.zero_()
            dev_v.zero_()
            dev_idx_k.zero_()
            dev_scale.zero_()

            # Do not synchronize between layer launches: this also exercises
            # asynchronous H2D submission through automatically resolved DVAs.
            for layer in range(LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[host_k, host_v],
                    dst_ptrs=[dev_k, dev_v],
                    src_indices=host_idx,
                    dst_indices=dev_idx,
                    layer_id=layer,
                    page_size=PS,
                )
            for idx_layer in range(IDX_LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[host_idx_k, host_scale],
                    dst_ptrs=[dev_idx_k, dev_scale],
                    src_indices=host_idx,
                    dst_indices=dev_idx,
                    layer_id=idx_layer,
                    page_size=PS,
                )
            torch.npu.synchronize()

            for name, actual in (
                ("k", dev_k),
                ("v", dev_v),
                ("index_k", dev_idx_k),
                ("scale", dev_scale),
            ):
                self.assertTrue(
                    torch.equal(actual, expected[name]),
                    msg=f"{name} roundtrip mismatch at round {round_idx + 1}",
                )


if __name__ == "__main__":
    unittest.main()
