import unittest

import torch

from sgl_kernel_npu.kvcacheio import (
    register_host_memory,
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_per_layer_direct_pf_lf,
    unregister_host_memory,
)

# Case matrix mirrors test_transfer_kv_dim_exchange.py (kv / kv+index_k / k /
# scale x D2H/H2D). Every call spells out formal=actual keyword arguments, so
# the operator signature reads like the dim_exchange calls. H2D goes through
# transfer_kv_per_layer_direct_pf_lf (mla.py load_to_device_per_layer), D2H
# through transfer_kv_all_layer_direct_lf_pf (mla.py
# backup_from_device_all_layer).
#
# Layouts (MLA host pool, page_first_kv_split):
#   host   page-first  (pages, layers, page_size, 1, width)  pinned+registered
#   device layer-first (layers, pages, page_size, 1, width)
# The quantized-Indexer k and its FP32 scale cover a subset of layers.
PAGES = 8
LAYERS = 4
PS = 128  # page_size
K_WIDTH = 656  # kv_lora_rank
V_WIDTH = 64  # qk_rope_head_dim
IDX_LAYERS = 2  # indexer-style components cover fewer layers than the pool
IDX_WIDTH = 128  # quantized-Indexer k width


def _unique_like(shape):
    """Deterministic per-position values, exactly representable in bf16/fp32.

    value(page, layer, token) = ((page * layers + layer) * 4 + token % 4) % 256
    - 128: unique per (page, layer) block (PAGES * LAYERS * 4 <= 256) with a
    per-token ripple. Any misdirected write -- wrong page, layer, token or an
    overrun row width -- changes values at the affected positions, so the
    element-wise comparisons catch it (ref: dim_exchange's per-(layer, page)
    scale test philosophy).
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
    indices must be page-aligned token indices with a distinct page per slot
    (hicache token indices are page*page_size + offset, grouped page by page).
    """
    offset = torch.arange(PS, dtype=torch.int64).repeat(PAGES)
    dev_pages = torch.randperm(PAGES, dtype=torch.int64)
    host_pages = torch.randperm(PAGES, dtype=torch.int64)
    dev_idx = (dev_pages * PS).repeat_interleave(PS) + offset
    host_idx = (host_pages * PS).repeat_interleave(PS) + offset
    return dev_idx.to("npu"), host_idx.to("npu")


class TestKvExchangeAclrt(unittest.TestCase):

    def _verify_pages(self, host, dev, dev_idx, host_idx, host_layer, dev_layer):
        """Element-wise: host[host_page, host_layer] == dev[dev_layer, dev_page]."""
        for p in range(PAGES):
            dev_page = int(dev_idx[p * PS]) // PS
            host_page = int(host_idx[p * PS]) // PS
            torch.testing.assert_close(
                dev[dev_layer, dev_page, :, 0, :].cpu(),
                host[host_page, host_layer, :, 0, :].cpu(),
                msg=f"mismatch dev_layer={dev_layer} host_layer={host_layer} page={p}",
            )

    def _verify_all_layers(self, host, dev, dev_idx, host_idx, layers):
        for layer in range(layers):
            self._verify_pages(host, dev, dev_idx, host_idx, layer, layer)

    def _verify_untouched_zero(self, dev, layer, name):
        """Layers after `layer` must still be all zero during an H2D loop."""
        for untouched in range(layer + 1, LAYERS):
            self.assertEqual(
                int(torch.count_nonzero(dev[untouched])),
                0,
                msg=f"{name}: layer {untouched} polluted by layer {layer}'s transfer",
            )

    # ------------------- H2D: per-layer load (mla.py) -------------------
    #
    # transfer_kv_per_layer_direct_pf_lf(
    #     src_ptrs,       # host page-first pools (k/v/index_k/scale lists)
    #     dst_ptrs,       # device layer-first pools
    #     src_indices,    # host token indices,  page*PS + slot
    #     dst_indices,    # device token indices (same token count)
    #     layer_id,       # which layer this call transfers
    #     page_size,      # PS
    #     host_mappings,  # per-component HostMemoryMapping (host DVA bases)
    # )

    def test_kv_copy_h2d(self):
        """k + v, per-layer H2D (dim_exchange test_kv_copy_h2d)."""
        torch.npu.set_device(0)
        host_k, mk = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, mv = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        dev_k = torch.zeros((LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_v = torch.zeros((LAYERS, PAGES, PS, 1, V_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_idx, host_idx = _page_pairs()
        try:
            for layer in range(LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[host_k, host_v],
                    dst_ptrs=[dev_k, dev_v],
                    src_indices=host_idx,
                    dst_indices=dev_idx,
                    layer_id=layer,
                    page_size=PS,
                    host_mappings=[mk, mv],
                )
                torch.npu.synchronize()
                # Verify right after each layer: a write aimed at this layer
                # but landing on a not-yet-transferred one would be masked by
                # the next call if only the final state were checked.
                self._verify_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)
                self._verify_pages(host_v, dev_v, dev_idx, host_idx, layer, layer)
                self._verify_untouched_zero(dev_k, layer, "dev_k")
                self._verify_untouched_zero(dev_v, layer, "dev_v")
        finally:
            unregister_host_memory(mk)
            unregister_host_memory(mv)

    def test_kv_index_k_copy_h2d(self):
        """k + v + index_k, per-layer H2D (dim_exchange test_kv_index_k_copy_h2d).

        index_k covers IDX_LAYERS (< LAYERS) layers, so it is transferred in
        its own loop under its own layer ids -- the mla.py mapped-layer-id
        semantics.
        """
        torch.npu.set_device(0)
        host_k, mk = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, mv = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k, mik = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        dev_k = torch.zeros((LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_v = torch.zeros((LAYERS, PAGES, PS, 1, V_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_idx_k = torch.zeros((IDX_LAYERS, PAGES, PS, 1, IDX_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_idx, host_idx = _page_pairs()
        try:
            for layer in range(LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[host_k, host_v],
                    dst_ptrs=[dev_k, dev_v],
                    src_indices=host_idx,
                    dst_indices=dev_idx,
                    layer_id=layer,
                    page_size=PS,
                    host_mappings=[mk, mv],
                )
                torch.npu.synchronize()
                self._verify_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)
                self._verify_pages(host_v, dev_v, dev_idx, host_idx, layer, layer)
                self._verify_untouched_zero(dev_k, layer, "dev_k")
                self._verify_untouched_zero(dev_v, layer, "dev_v")
            for idx_layer in range(IDX_LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[host_idx_k],
                    dst_ptrs=[dev_idx_k],
                    src_indices=host_idx,
                    dst_indices=dev_idx,
                    layer_id=idx_layer,  # index_k's own (smaller) layer id space
                    page_size=PS,
                    host_mappings=[mik],
                )
                torch.npu.synchronize()
                self._verify_pages(host_idx_k, dev_idx_k, dev_idx, host_idx, idx_layer, idx_layer)
        finally:
            unregister_host_memory(mk)
            unregister_host_memory(mv)
            unregister_host_memory(mik)

    def test_k_copy_h2d(self):
        """k only, per-layer H2D (dim_exchange test_k_copy_h2d)."""
        torch.npu.set_device(0)
        host_k, mk = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        dev_k = torch.zeros((LAYERS, PAGES, PS, 1, K_WIDTH), dtype=torch.bfloat16, device="npu")
        dev_idx, host_idx = _page_pairs()
        try:
            for layer in range(LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[host_k],
                    dst_ptrs=[dev_k],
                    src_indices=host_idx,
                    dst_indices=dev_idx,
                    layer_id=layer,
                    page_size=PS,
                    host_mappings=[mk],
                )
                torch.npu.synchronize()
                self._verify_pages(host_k, dev_k, dev_idx, host_idx, layer, layer)
                self._verify_untouched_zero(dev_k, layer, "dev_k")
        finally:
            unregister_host_memory(mk)

    def test_scale_copy_h2d(self):
        """FP32 scale only, per-layer H2D (dim_exchange test_scale_copy_h2d)."""
        torch.npu.set_device(0)
        host_scale, msc = _host_pool((PAGES, IDX_LAYERS, PS, 1, 1), dtype=torch.float32)
        dev_scale = torch.zeros((IDX_LAYERS, PAGES, PS, 1, 1), dtype=torch.float32, device="npu")
        dev_idx, host_idx = _page_pairs()
        try:
            for idx_layer in range(IDX_LAYERS):
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[host_scale],
                    dst_ptrs=[dev_scale],
                    src_indices=host_idx,
                    dst_indices=dev_idx,
                    layer_id=idx_layer,
                    page_size=PS,
                    host_mappings=[msc],
                )
                torch.npu.synchronize()
                self._verify_pages(host_scale, dev_scale, dev_idx, host_idx, idx_layer, idx_layer)
        finally:
            unregister_host_memory(msc)

    # ------------- D2H: all-layer single-call backup (mla.py) -------------
    #
    # transfer_kv_all_layer_direct_lf_pf(
    #     src_ptrs,       # device layer-first pools
    #     dst_ptrs,       # host page-first pools
    #     src_indices,    # device token indices
    #     dst_indices,    # host token indices
    #     page_size,      # PS (no layer_id: every component's dim(0) defines
    #                     #   its transferred layer range)
    #     host_mappings,  # per-component HostMemoryMapping (host DVA bases)
    # )

    def test_kv_copy_d2h(self):
        """k + v, all-layer D2H (dim_exchange test_kv_copy_d2h)."""
        torch.npu.set_device(0)
        host_k, mk = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, mv = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_k.zero_()  # host is the destination; start from zero
        host_v.zero_()
        dev_k = (
            _unique_like((PAGES, LAYERS, PS, 1, K_WIDTH)).permute(1, 0, 2, 3, 4)
            .contiguous().to(torch.bfloat16).to("npu")
        )
        dev_v = (
            _unique_like((PAGES, LAYERS, PS, 1, V_WIDTH)).permute(1, 0, 2, 3, 4)
            .contiguous().to(torch.bfloat16).to("npu")
        )
        dev_idx, host_idx = _page_pairs()
        try:
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=[dev_k, dev_v],
                dst_ptrs=[host_k, host_v],
                src_indices=dev_idx,
                dst_indices=host_idx,
                page_size=PS,
                host_mappings=[mk, mv],
            )
            torch.npu.synchronize()
            self._verify_all_layers(host_k, dev_k, dev_idx, host_idx, LAYERS)
            self._verify_all_layers(host_v, dev_v, dev_idx, host_idx, LAYERS)
        finally:
            unregister_host_memory(mk)
            unregister_host_memory(mv)

    def test_kv_index_k_copy_d2h(self):
        """k + v + index_k, all-layer D2H (dim_exchange test_kv_index_k_copy_d2h).

        One call carries components with different layer counts (k/v: LAYERS,
        index_k: IDX_LAYERS); each component's dim(0) defines its own layer
        range.
        """
        torch.npu.set_device(0)
        host_k, mk = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_v, mv = _host_pool((PAGES, LAYERS, PS, 1, V_WIDTH))
        host_idx_k, mik = _host_pool((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH))
        host_k.zero_()
        host_v.zero_()
        host_idx_k.zero_()
        dev_k = (
            _unique_like((PAGES, LAYERS, PS, 1, K_WIDTH)).permute(1, 0, 2, 3, 4)
            .contiguous().to(torch.bfloat16).to("npu")
        )
        dev_v = (
            _unique_like((PAGES, LAYERS, PS, 1, V_WIDTH)).permute(1, 0, 2, 3, 4)
            .contiguous().to(torch.bfloat16).to("npu")
        )
        dev_idx_k = (
            _unique_like((PAGES, IDX_LAYERS, PS, 1, IDX_WIDTH)).permute(1, 0, 2, 3, 4)
            .contiguous().to(torch.bfloat16).to("npu")
        )
        dev_idx, host_idx = _page_pairs()
        try:
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=[dev_k, dev_v, dev_idx_k],
                dst_ptrs=[host_k, host_v, host_idx_k],
                src_indices=dev_idx,
                dst_indices=host_idx,
                page_size=PS,
                host_mappings=[mk, mv, mik],
            )
            torch.npu.synchronize()
            self._verify_all_layers(host_k, dev_k, dev_idx, host_idx, LAYERS)
            self._verify_all_layers(host_v, dev_v, dev_idx, host_idx, LAYERS)
            self._verify_all_layers(host_idx_k, dev_idx_k, dev_idx, host_idx, IDX_LAYERS)
        finally:
            unregister_host_memory(mk)
            unregister_host_memory(mv)
            unregister_host_memory(mik)

    def test_k_copy_d2h(self):
        """k only, all-layer D2H (dim_exchange test_k_copy_d2h)."""
        torch.npu.set_device(0)
        host_k, mk = _host_pool((PAGES, LAYERS, PS, 1, K_WIDTH))
        host_k.zero_()
        dev_k = (
            _unique_like((PAGES, LAYERS, PS, 1, K_WIDTH)).permute(1, 0, 2, 3, 4)
            .contiguous().to(torch.bfloat16).to("npu")
        )
        dev_idx, host_idx = _page_pairs()
        try:
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=[dev_k],
                dst_ptrs=[host_k],
                src_indices=dev_idx,
                dst_indices=host_idx,
                page_size=PS,
                host_mappings=[mk],
            )
            torch.npu.synchronize()
            self._verify_all_layers(host_k, dev_k, dev_idx, host_idx, LAYERS)
        finally:
            unregister_host_memory(mk)

    def test_scale_copy_d2h(self):
        """FP32 scale only, all-layer D2H (dim_exchange test_scale_copy_d2h)."""
        torch.npu.set_device(0)
        host_scale, msc = _host_pool((PAGES, IDX_LAYERS, PS, 1, 1), dtype=torch.float32)
        host_scale.zero_()
        dev_scale = (
            _unique_like((PAGES, IDX_LAYERS, PS, 1, 1)).permute(1, 0, 2, 3, 4)
            .contiguous().to(torch.float32).to("npu")
        )
        dev_idx, host_idx = _page_pairs()
        try:
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=[dev_scale],
                dst_ptrs=[host_scale],
                src_indices=dev_idx,
                dst_indices=host_idx,
                page_size=PS,
                host_mappings=[msc],
            )
            torch.npu.synchronize()
            self._verify_all_layers(host_scale, dev_scale, dev_idx, host_idx, IDX_LAYERS)
        finally:
            unregister_host_memory(msc)


if __name__ == "__main__":
    unittest.main()
