import unittest

import torch

from sgl_kernel_npu.kvcacheio import (
    aclrt_host_register,
    aclrt_host_unregister,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_per_layer_direct_pf_lf,
)

# Geometry mirrors sglang pool_host/mla.py page_first_kv_split (MLA host pool):
# host (pages, layers, page_size, 1, width) page-first, device (layers, pages,
# page_size, 1, width) layer-first. k/v are separate pools with different
# widths (kv_lora_rank vs qk_rope_head_dim); the quantized-Indexer k covers a
# subset of layers and the FP32 scale is a 4-D cache padded to 5-D on call.
PAGES = 8
LAYERS = 4
PS = 128  # page_size
K_WIDTH = 656  # kv_lora_rank
V_WIDTH = 64  # qk_rope_head_dim
IDX_LAYERS = 2  # indexer-style components cover fewer layers than the pool
IDX_WIDTH = 128  # quantized-Indexer k width
NUM_TOKENS = PAGES * PS


def _host_pool(shape, dtype=torch.bfloat16):
    """Pinned + registered host pool; returns (tensor, holder)."""
    buf = torch.randn(shape).to(dtype).pin_memory().contiguous()
    aclrt_host_register(buf)
    return buf


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

    def test_h2d_per_layer_kv(self):
        torch.npu.set_device(0)
        host_k = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        dev_k = torch.zeros((LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_v = torch.zeros((LAYERS, PAGES, PS, 1, V_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_idx, host_idx = _page_pairs()
        try:
            for layer in range(LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    [host_k, host_v], [dev_k, dev_v], host_idx, dev_idx, layer, PS
                )
            torch.npu.synchronize()
            for layer in range(LAYERS):
                self._check_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)
                self._check_pages(host_v, dev_v, dev_idx, host_idx, layer, layer)
        finally:
            aclrt_host_unregister(host_k)
            aclrt_host_unregister(host_v)

    def test_h2d_indexer_layer_mapping(self):
        """Indexer components transfer under their own (mapped) layer id."""
        torch.npu.set_device(0)
        host_k = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_k = torch.zeros((LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu")
        host_idx_k = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        dev_idx_k = torch.zeros((IDX_LAYERS, PAGES, PS, 1, IDX_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_idx, host_idx = _page_pairs()
        try:
            for layer in range(LAYERS):
                transfer_kv_per_layer_direct_pf_lf([host_k], [dev_k], host_idx, dev_idx, layer, PS)
            # _indexer_slot_range_for_layer-style mapping: pool layer -> indexer layer.
            for idx_layer in range(IDX_LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    [host_idx_k], [dev_idx_k], host_idx, dev_idx, idx_layer, PS
                )
            torch.npu.synchronize()
            for layer in range(LAYERS):
                self._check_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)
            for idx_layer in range(IDX_LAYERS):
                self._check_pages(host_idx_k, dev_idx_k, dev_idx, host_idx, idx_layer, idx_layer)
        finally:
            aclrt_host_unregister(host_k)
            aclrt_host_unregister(host_idx_k)

    def test_d2h_all_layer_kv_index_scales(self):
        """All-layer D2H with mixed-width and fp32-scale components.

        Mirrors mla.py `_transfer_aclrt_aiv`: the FP32 scale caches are 4-D and
        get a trailing singleton padded on the call (5-D op requirement).
        """
        torch.npu.set_device(0)
        host_k = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        host_scale = _host_pool((PAGES, IDX_LAYERS, PS, 1), dtype=torch.float32)
        host_k.zero_()
        host_v.zero_()
        host_idx_k.zero_()
        host_scale.zero_()

        dev_k = torch.randn((LAYERS, PAGES, PS, 1, K_WIDTH), device="npu").to(torch.bfloat16)
        dev_v = torch.randn((LAYERS, PAGES, PS, 1, V_WIDTH), device="npu").to(torch.bfloat16)
        dev_idx_k = torch.randn((IDX_LAYERS, PAGES, PS, 1, IDX_WIDTH), device="npu").to(torch.bfloat16)
        dev_scale = torch.randn((IDX_LAYERS, PAGES, PS, 1), device="npu", dtype=torch.float32)
        dev_idx, host_idx = _page_pairs()
        try:
            transfer_kv_all_layer_direct_lf_pf(
                [dev_k, dev_v, dev_idx_k, dev_scale.unsqueeze(-1)],
                [host_k, host_v, host_idx_k, host_scale.unsqueeze(-1)],
                dev_idx,
                host_idx,
                PS,
            )
            torch.npu.synchronize()
            for layer in range(LAYERS):
                self._check_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)
                self._check_pages(host_v, dev_v, dev_idx, host_idx, layer, layer)
            for idx_layer in range(IDX_LAYERS):
                self._check_pages(host_idx_k, dev_idx_k, dev_idx, host_idx, idx_layer, idx_layer)
                for p in range(PAGES):
                    dev_page = int(dev_idx[p * PS]) // PS
                    host_page = int(host_idx[p * PS]) // PS
                    torch.testing.assert_close(
                        host_scale[host_page, idx_layer, :, 0].cpu(),
                        dev_scale[idx_layer, dev_page, :, 0].cpu(),
                        msg=f"scale mismatch idx_layer={idx_layer} page={p}",
                    )
        finally:
            aclrt_host_unregister(host_k)
            aclrt_host_unregister(host_v)
            aclrt_host_unregister(host_idx_k)
            aclrt_host_unregister(host_scale)

    def test_wide_row_split(self):
        """Row width above the 88KB UB budget must still transfer correctly."""
        torch.npu.set_device(0)
        wide = 65536  # bf16 -> 128KB rows
        host = _host_pool((PAGES, 1, PS, 1, wide))
        host.zero_()
        dev = torch.randn((1, PAGES, PS, 1, wide), device="npu").to(torch.bfloat16)
        dev_idx, host_idx = _page_pairs()
        try:
            transfer_kv_per_layer_direct_pf_lf([host], [dev], host_idx, dev_idx, 0, PS)
            torch.npu.synchronize()
            for p in range(PAGES):
                dev_page = int(dev_idx[p * PS]) // PS
                host_page = int(host_idx[p * PS]) // PS
                torch.testing.assert_close(
                    dev[0, dev_page, :, 0, :].cpu(),
                    host[host_page, 0, :, 0, :].cpu(),
                    msg=f"wide-row mismatch page={p}",
                )
        finally:
            aclrt_host_unregister(host)

    def test_unregistered_host_raises(self):
        torch.npu.set_device(0)
        host = torch.zeros((PAGES, LAYERS, PS, 1, K_WIDTH), dtype=torch.bfloat16).pin_memory()
        dev = torch.zeros((LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_idx, host_idx = _page_pairs()
        with self.assertRaises(RuntimeError):
            transfer_kv_per_layer_direct_pf_lf([host], [dev], host_idx, dev_idx, 0, PS)

    def test_too_many_components_raises(self):
        torch.npu.set_device(0)
        host = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev = torch.zeros((LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_idx, host_idx = _page_pairs()
        try:
            with self.assertRaises(RuntimeError):
                transfer_kv_per_layer_direct_pf_lf(
                    [host] * 5, [dev] * 5, host_idx, dev_idx, 0, PS
                )
        finally:
            aclrt_host_unregister(host)


if __name__ == "__main__":
    unittest.main()
