import unittest

import torch

from sgl_kernel_npu.kvcacheio import (
    HostMemoryMapping,
    empty_host_memory,
    get_device_accessible_ptr,
    initialize_host_memory,
    register_host_memory,
    uninitialize_host_memory,
)
from sgl_kernel_npu.operators.kv_exchange import (
    transfer_kv_all_layer_kv_split_lf_pf,
    transfer_kv_per_layer_kv_split_pf_lf,
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


def _page_pairs(to_npu=True):
    """Return random non-contiguous, page-aligned token-slot mappings.

    The kernel derives page numbers from token slots. Callers must pass the
    original token-slot indices without dividing by page_size themselves.
    """
    offset = torch.arange(PS, dtype=torch.int64).repeat(PAGES)
    dev_pages = torch.randperm(PAGES, dtype=torch.int64)
    host_pages = torch.randperm(PAGES, dtype=torch.int64)
    dev_idx = (dev_pages * PS).repeat_interleave(PS) + offset
    host_idx = (host_pages * PS).repeat_interleave(PS) + offset
    if to_npu:
        return dev_idx.to("npu"), host_idx.to("npu")
    return dev_idx, host_idx


class TestKvExchangeMf(unittest.TestCase):

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

    def _verify_all_layers(self, host, dev, dev_idx, host_idx, layers, host_layer_start=0):
        for layer in range(layers):
            self._verify_pages(
                host, dev, dev_idx, host_idx, host_layer_start + layer, layer
            )

    def _verify_untouched_zero(self, dev, layer, layers, name):
        for untouched in range(layer + 1, layers):
            self.assertEqual(
                int(torch.count_nonzero(dev[untouched])),
                0,
                msg=f"{name}: layer {untouched} polluted by layer {layer}",
            )

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
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev,
                host_k=unregistered,
                page_size=PS,
                layer_id=0,
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
        for layer in range(LAYERS):
            dev_idx, host_idx = _page_pairs()
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                device_v=dev_v,
                host_v=host_v,
                page_size=PS,
                layer_id=layer,
            )
            torch.npu.synchronize()
            self._verify_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)
            self._verify_pages(host_v, dev_v, dev_idx, host_idx, layer, layer)
            self._verify_untouched_zero(dev_k, layer, LAYERS, "k")
            self._verify_untouched_zero(dev_v, layer, LAYERS, "v")

    def test_kv_layer_offset_h2d(self):
        # The device layer id and the host layer id are independent: device
        # layer 1 is restored from host layer 3 of the full host pool.
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_k = torch.zeros(
            (LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        dev_idx, host_idx = _page_pairs()
        transfer_kv_per_layer_kv_split_pf_lf(
            device_indices=dev_idx,
            host_indices=host_idx,
            device_k=dev_k,
            host_k=host_k,
            page_size=PS,
            layer_id=1,
            host_layer_id=3,
        )
        torch.npu.synchronize()
        self._verify_pages(host_k, dev_k, dev_idx, host_idx, 3, 1)
        for other in (0, 2, 3):
            self.assertEqual(
                int(torch.count_nonzero(dev_k[other])),
                0,
                msg=f"k: device layer {other} polluted",
            )

    def test_kv_index_k_scale_h2d(self):
        # k/v/index_k/scale ride in one launch; layers >= IDX_LAYERS skip the
        # index components via a negative index_k_layer_id.
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        host_scale, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32)
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
        # Device scale stays 4-D (index_layer, page, page_size, 1): the
        # interface must not require a padded 5-D form.
        dev_scale = torch.zeros(
            (IDX_LAYERS, PAGES, PS, 1), dtype=torch.float32, device="npu"
        )
        for layer in range(LAYERS):
            dev_idx, host_idx = _page_pairs()
            index_k_layer_id = layer if layer < IDX_LAYERS else -1
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                device_v=dev_v,
                host_v=host_v,
                device_index_k=dev_idx_k,
                host_index_k=host_idx_k,
                device_index_k_scale=dev_scale,
                host_index_k_scale=host_scale,
                page_size=PS,
                layer_id=layer,
                index_k_layer_id=index_k_layer_id,
            )
            torch.npu.synchronize()
            self._verify_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)
            self._verify_pages(host_v, dev_v, dev_idx, host_idx, layer, layer)
            self._verify_untouched_zero(dev_k, layer, LAYERS, "k")
            if index_k_layer_id >= 0:
                self._verify_pages(
                    host_idx_k, dev_idx_k, dev_idx, host_idx, layer, layer
                )
                self._verify_pages(
                    host_scale, dev_scale, dev_idx, host_idx, layer, layer
                )
            else:
                self.assertEqual(int(torch.count_nonzero(dev_idx_k)), 0)
                self.assertEqual(int(torch.count_nonzero(dev_scale)), 0)

    def test_empty_v_and_empty_indices_h2d(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_k = torch.zeros(
            (LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        empty_dev_idx = torch.empty(0, dtype=torch.int64, device="npu")
        empty_host_idx = torch.empty(0, dtype=torch.int64, device="npu")
        # Empty indices are a no-op even with all components wired.
        transfer_kv_per_layer_kv_split_pf_lf(
            device_indices=empty_dev_idx,
            host_indices=empty_host_idx,
            device_k=dev_k,
            host_k=host_k,
            page_size=PS,
            layer_id=0,
        )
        torch.npu.synchronize()
        self.assertEqual(int(torch.count_nonzero(dev_k)), 0)

        dev_idx, host_idx = _page_pairs()
        # V omitted on both sides is skipped, not an error.
        transfer_kv_per_layer_kv_split_pf_lf(
            device_indices=dev_idx,
            host_indices=host_idx,
            device_k=dev_k,
            host_k=host_k,
            page_size=PS,
            layer_id=0,
        )
        torch.npu.synchronize()
        self._verify_pages(host_k, dev_k, dev_idx, host_idx, 0, 0)

    def test_cpu_indices_h2d(self):
        # CPU indices take the pinned-staging non-blocking upload path.
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_k = torch.zeros(
            (LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        for layer in range(LAYERS):
            dev_idx, host_idx = _page_pairs(to_npu=False)
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                page_size=PS,
                layer_id=layer,
            )
        torch.npu.synchronize()
        for layer in range(LAYERS):
            self._verify_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)

    def test_kv_copy_d2h(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        dev_k = _device_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_v = _device_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_k.zero_()
        host_v.zero_()
        dev_idx, host_idx = _page_pairs()
        transfer_kv_all_layer_kv_split_lf_pf(
            device_indices=dev_idx,
            host_indices=host_idx,
            device_k=dev_k,
            host_k=host_k,
            device_v=dev_v,
            host_v=host_v,
            page_size=PS,
        )
        torch.npu.synchronize()
        self._verify_all_layers(host_k, dev_k, dev_idx, host_idx, LAYERS)
        self._verify_all_layers(host_v, dev_v, dev_idx, host_idx, LAYERS)

    def test_kv_d2h_host_layer_start(self):
        # A 2-layer device pool lands at host layers [2, 4) of the full host
        # pool; host layers [0, 2) stay untouched.
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_layers = 2
        dev_k = _device_pool((PAGES, dev_layers, PS, 1, K_WIDTH))
        host_k.zero_()
        dev_idx, host_idx = _page_pairs()
        transfer_kv_all_layer_kv_split_lf_pf(
            device_indices=dev_idx,
            host_indices=host_idx,
            device_k=dev_k,
            host_k=host_k,
            page_size=PS,
            host_layer_start=2,
        )
        torch.npu.synchronize()
        self._verify_all_layers(host_k, dev_k, dev_idx, host_idx, dev_layers, host_layer_start=2)
        for layer in (0, 1):
            self.assertEqual(
                int(torch.count_nonzero(host_k[:, layer])),
                0,
                msg=f"host layer {layer} polluted",
            )

    def test_kv_index_k_scale_d2h(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        host_scale, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32)
        dev_k = _device_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_v = _device_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        dev_idx_k = _device_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        dev_scale = _device_pool((PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32)
        for host in (host_k, host_v, host_idx_k, host_scale):
            host.zero_()
        dev_idx, host_idx = _page_pairs()
        transfer_kv_all_layer_kv_split_lf_pf(
            device_indices=dev_idx,
            host_indices=host_idx,
            device_k=dev_k,
            host_k=host_k,
            device_v=dev_v,
            host_v=host_v,
            device_index_k=dev_idx_k,
            host_index_k=host_idx_k,
            device_index_k_scale=dev_scale,
            host_index_k_scale=host_scale,
            page_size=PS,
        )
        torch.npu.synchronize()
        self._verify_all_layers(host_k, dev_k, dev_idx, host_idx, LAYERS)
        self._verify_all_layers(host_v, dev_v, dev_idx, host_idx, LAYERS)
        self._verify_all_layers(host_idx_k, dev_idx_k, dev_idx, host_idx, IDX_LAYERS)
        self._verify_all_layers(host_scale, dev_scale, dev_idx, host_idx, IDX_LAYERS)

    def test_kv_index_k_scale_roundtrip(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        host_scale, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32)
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
        dev_scale = torch.zeros(
            (IDX_LAYERS, PAGES, PS, 1), dtype=torch.float32, device="npu"
        )
        # Independent zeroed host outputs: the D2H leg restores into them.
        out_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        out_v, _ = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        out_idx_k, _ = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        out_scale, _ = _host_pool(
            (PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32
        )

        for round_idx in range(ROUNDTRIP_ROUNDS):
            dev_idx, host_idx = _page_pairs()

            # H2D: all components in merged per-layer launches, no
            # synchronization between layer submissions.
            for layer in range(LAYERS):
                transfer_kv_per_layer_kv_split_pf_lf(
                    device_indices=dev_idx,
                    host_indices=host_idx,
                    device_k=dev_k,
                    host_k=host_k,
                    device_v=dev_v,
                    host_v=host_v,
                    device_index_k=dev_idx_k,
                    host_index_k=host_idx_k,
                    device_index_k_scale=dev_scale,
                    host_index_k_scale=host_scale,
                    page_size=PS,
                    layer_id=layer,
                    index_k_layer_id=layer if layer < IDX_LAYERS else -1,
                )
            torch.npu.synchronize()

            for out in (out_k, out_v, out_idx_k, out_scale):
                out.zero_()
            # D2H: every component of both pools in one launch per pool.
            transfer_kv_all_layer_kv_split_lf_pf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=out_k,
                device_v=dev_v,
                host_v=out_v,
                page_size=PS,
            )
            transfer_kv_all_layer_kv_split_lf_pf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_idx_k,
                host_k=out_idx_k,
                device_index_k_scale=dev_scale,
                host_index_k_scale=out_scale,
                page_size=PS,
            )
            torch.npu.synchronize()

            for name, actual, expected in (
                ("k", out_k, host_k),
                ("v", out_v, host_v),
                ("index_k", out_idx_k, host_idx_k),
                ("scale", out_scale, host_scale),
            ):
                self.assertTrue(
                    torch.equal(actual, expected),
                    msg=f"{name} roundtrip mismatch at round {round_idx + 1}",
                )

    def test_error_branches(self):
        host_k, _ = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_k = torch.zeros(
            (LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu"
        )
        dev_idx, host_idx = _page_pairs()

        # One-sided V is a component mismatch.
        with self.assertRaisesRegex(ValueError, "component v"):
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                device_v=dev_k,
                host_v=None,
                page_size=PS,
                layer_id=0,
            )

        # Scale cannot travel without a paired index K.
        host_scale, _ = _host_pool(
            (PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32
        )
        dev_scale = torch.zeros(
            (IDX_LAYERS, PAGES, PS, 1), dtype=torch.float32, device="npu"
        )
        with self.assertRaisesRegex(ValueError, "index_k_scale"):
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                device_index_k_scale=dev_scale,
                host_index_k_scale=host_scale,
                page_size=PS,
                layer_id=0,
            )

        # Mismatched indices lengths fail before any launch.
        with self.assertRaisesRegex(ValueError, "same length"):
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx[:-PS],
                device_k=dev_k,
                host_k=host_k,
                page_size=PS,
                layer_id=0,
            )

        # Device layer id out of range for the K/V pool.
        with self.assertRaisesRegex(RuntimeError, "device layer id"):
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                page_size=PS,
                layer_id=LAYERS,
            )

        # Host layer id out of range for the full host pool.
        with self.assertRaisesRegex(RuntimeError, "host layer id"):
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                page_size=PS,
                layer_id=0,
                host_layer_id=LAYERS,
            )

        # All-layer: host layer range exceeding the host pool.
        with self.assertRaisesRegex(RuntimeError, "host layer range"):
            transfer_kv_all_layer_kv_split_lf_pf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                page_size=PS,
                host_layer_start=2,
            )

        # Indices not divisible by page_size.
        with self.assertRaisesRegex(RuntimeError, "divisible by page_size"):
            transfer_kv_per_layer_kv_split_pf_lf(
                device_indices=dev_idx,
                host_indices=host_idx,
                device_k=dev_k,
                host_k=host_k,
                page_size=PS + 1,
                layer_id=0,
            )


if __name__ == "__main__":
    unittest.main()
