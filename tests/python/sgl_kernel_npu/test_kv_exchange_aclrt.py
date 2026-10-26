import unittest

import torch

from sgl_kernel_npu.kvcacheio import (
    register_host_memory,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_per_layer_direct_pf_lf,
    unregister_host_memory,
)

# Case matrix mirrors test_transfer_kv_dim_exchange.py (kv / kv+index_k / k /
# scale x D2H/H2D), adapted to the production entry points: H2D = per-layer
# loop (mla.py load_to_device_per_layer), D2H = all-layer single call (mla.py
# backup_from_device_all_layer). Verification is element-wise per (page,
# layer) block -- stronger than dim_exchange's sum-based kv checks.
#
# Geometry mirrors sglang pool_host/mla.py page_first_kv_split (MLA host pool):
# host (pages, layers, page_size, 1, width) page-first, device (layers, pages,
# page_size, 1, width) layer-first. The quantized-Indexer k and its FP32 scale
# cover a subset of layers, like dim_exchange's index_k/scale components.
PAGES = 8
LAYERS = 4
PS = 128  # page_size
K_WIDTH = 656  # kv_lora_rank
V_WIDTH = 64  # qk_rope_head_dim
IDX_LAYERS = 2  # indexer-style components cover fewer layers than the pool
IDX_WIDTH = 128  # quantized-Indexer k width

# name -> (host shape (page-first), dtype, component layer count)
POOL_SPECS = {
    "k": ((PAGES, LAYERS, PS, 1, K_WIDTH), torch.bfloat16, LAYERS),
    "v": ((PAGES, LAYERS, PS, 1, V_WIDTH), torch.bfloat16, LAYERS),
    "index_k": ((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH), torch.bfloat16, IDX_LAYERS),
    "scale": ((PAGES, IDX_LAYERS, PS, 1, 1), torch.float32, IDX_LAYERS),
}


def _unique_like(shape):
    """Deterministic per-position values, exactly representable in bf16/fp32.

    value(page, layer, token) = ((page * layers + layer) * 4 + token % 4) % 256
    - 128: unique per (page, layer) block (PAGES * LAYERS * 4 <= 256) with a
    per-token ripple. Any misdirected write -- wrong page, layer, token or an
    overrun row width -- changes values at the affected positions, so the
    element-wise comparisons catch it; iid random data could hide a wrong
    write that a later transfer overwrites (ref:
    test_transfer_kv_dim_exchange.py's per-(layer, page) scale test).
    """
    pages, layers, ps = shape[0], shape[1], shape[2]
    assert pages * layers * 4 <= 256, "block ids must stay unique and bf16-exact"
    page = torch.arange(pages, dtype=torch.float32).view(pages, 1, 1, 1, 1)
    layer = torch.arange(layers, dtype=torch.float32).view(1, layers, 1, 1, 1)
    token = torch.arange(ps, dtype=torch.float32).view(1, 1, ps, 1, 1)
    vals = ((page * layers + layer) * 4 + token % 4) % 256 - 128
    return vals.expand(shape)


def _host_pool(shape, dtype=torch.bfloat16):
    """Pinned + registered host pool filled with deterministic unique values;
    returns (tensor, HostMemoryMapping)."""
    buf = _unique_like(shape).to(dtype).pin_memory().contiguous()
    mapping = register_host_memory(buf)
    return buf, mapping


def _page_pairs():
    """Page-aligned, collision-free page-pair indices.

    The kernel transfers WHOLE (page, layer) blocks: it derives both page
    numbers from the first token of each transfer slot (idx[p*PS] // PS). Slot
    indices must therefore be page-aligned token indices with a distinct page
    per slot -- exactly the production contract (hicache token indices are
    page*page_size + offset, grouped page by page). Random token permutations
    would alias pages across slots and corrupt the comparison.
    """
    offset = torch.arange(PS, dtype=torch.int64).repeat(PAGES)
    dev_pages = torch.randperm(PAGES, dtype=torch.int64)
    host_pages = torch.randperm(PAGES, dtype=torch.int64)
    dev_idx = (dev_pages * PS).repeat_interleave(PS) + offset
    host_idx = (host_pages * PS).repeat_interleave(PS) + offset
    return dev_idx.to("npu"), host_idx.to("npu")


class TestKvExchangeAclrt(unittest.TestCase):

    def _check_pages(self, host, dev, dev_idx, host_idx, host_layer, dev_layer):
        """host[host_page, host_layer] == dev[dev_layer, dev_page] per page pair."""
        for p in range(PAGES):
            dev_page = int(dev_idx[p * PS]) // PS
            host_page = int(host_idx[p * PS]) // PS
            torch.testing.assert_close(
                dev[dev_layer, dev_page, :, 0, :].cpu(),
                host[host_page, host_layer, :, 0, :].cpu(),
                msg=f"mismatch dev_layer={dev_layer} host_layer={host_layer} page={p}",
            )

    def _run_case(self, direction, names):
        """Build + transfer + verify one matrix cell.

        direction: 'H2D' -> per-layer loop through
        transfer_kv_per_layer_direct_pf_lf; 'D2H' -> single all-layer call
        through transfer_kv_all_layer_direct_lf_pf. Components with different
        layer counts (k/v vs index_k/scale) are transferred in separate
        per-layer loops, mirroring mla.py's mapped layer ids.
        """
        torch.npu.set_device(0)
        comps = []
        for name in names:
            shape, dtype, n_layers = POOL_SPECS[name]
            host, mapping = _host_pool(shape, dtype)
            if direction == "D2H":
                # Host is the destination: zero it; the device source carries
                # the same unique values (page-first permuted to layer-first).
                host.zero_()
                dev = (
                    _unique_like(shape).permute(1, 0, 2, 3, 4)
                    .contiguous().to(dtype).to("npu")
                )
            else:
                # Host is the source (unique values); device starts zeroed so
                # not-yet-transferred regions can be checked to stay zero.
                dev = torch.zeros(
                    (n_layers, shape[0]) + shape[2:], dtype=dtype, device="npu"
                )
            comps.append((name, host, mapping, dev, n_layers))
        dev_idx, host_idx = _page_pairs()
        try:
            if direction == "D2H":
                transfer_kv_all_layer_direct_lf_pf(
                    [dev for _, _, _, dev, _ in comps],
                    [host for _, host, _, _, _ in comps],
                    dev_idx,
                    host_idx,
                    PS,
                    host_mappings=[m for _, _, m, _, _ in comps],
                )
                torch.npu.synchronize()
                for name, host, _, dev, n_layers in comps:
                    for layer in range(n_layers):
                        self._check_pages(host, dev, dev_idx, host_idx, layer, layer)
            else:
                for group_layers in sorted({c[4] for c in comps}, reverse=True):
                    members = [c for c in comps if c[4] == group_layers]
                    for layer in range(group_layers):
                        transfer_kv_per_layer_direct_pf_lf(
                            [host for _, host, _, _, _ in members],
                            [dev for _, _, _, dev, _ in members],
                            host_idx,
                            dev_idx,
                            layer,
                            PS,
                            host_mappings=[m for _, _, m, _, _ in members],
                        )
                        torch.npu.synchronize()
                        # Verify right after each layer: a write aimed at this
                        # layer but landing on a not-yet-transferred one would
                        # be overwritten (and masked) by the next call if only
                        # the final state were checked.
                        for name, host, _, dev, _ in members:
                            self._check_pages(host, dev, dev_idx, host_idx, layer, layer)
                            for untouched in range(layer + 1, group_layers):
                                self.assertEqual(
                                    int(torch.count_nonzero(dev[untouched])),
                                    0,
                                    msg=f"{name}: layer {untouched} polluted by "
                                    f"layer {layer}'s transfer",
                                )
        finally:
            for _, _, mapping, _, _ in comps:
                unregister_host_memory(mapping)

    def test_kv_copy_d2h(self):
        """k + v, all-layer D2H (dim_exchange test_kv_copy_d2h)."""
        self._run_case("D2H", ["k", "v"])

    def test_kv_copy_h2d(self):
        """k + v, per-layer H2D (dim_exchange test_kv_copy_h2d)."""
        self._run_case("H2D", ["k", "v"])

    def test_kv_index_k_copy_d2h(self):
        """k + v + index_k, all-layer D2H (dim_exchange test_kv_index_k_copy_d2h)."""
        self._run_case("D2H", ["k", "v", "index_k"])

    def test_kv_index_k_copy_h2d(self):
        """k + v + index_k (subset layers), per-layer H2D
        (dim_exchange test_kv_index_k_copy_h2d, plus layer-count mapping)."""
        self._run_case("H2D", ["k", "v", "index_k"])

    def test_k_copy_d2h(self):
        """k only, all-layer D2H (dim_exchange test_k_copy_d2h)."""
        self._run_case("D2H", ["k"])

    def test_k_copy_h2d(self):
        """k only, per-layer H2D (dim_exchange test_k_copy_h2d)."""
        self._run_case("H2D", ["k"])

    def test_scale_copy_d2h(self):
        """FP32 scale only, all-layer D2H (dim_exchange test_scale_copy_d2h)."""
        self._run_case("D2H", ["scale"])

    def test_scale_copy_h2d(self):
        """FP32 scale only, per-layer H2D (dim_exchange test_scale_copy_h2d)."""
        self._run_case("H2D", ["scale"])


if __name__ == "__main__":
    unittest.main()
