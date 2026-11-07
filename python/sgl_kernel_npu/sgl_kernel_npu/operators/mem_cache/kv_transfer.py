"""Public split-KV HiCache operators and MemFabric implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_memfabric_fn = None
_reference_fn = None
_memfabric_range_fn = None
_reference_range_fn = None


@register_impl(
    "mem_cache.transfer_kv_per_layer_kv_split_pf_lf",
    "aiv",
)
@register_impl(
    "mem_cache.transfer_kv_all_layer_kv_split_lf_pf",
    "aiv",
)
@register_impl(
    "mem_cache.transfer_kv_per_layer_kv_split",
    "aiv",
)
@register_impl(
    "mem_cache.transfer_kv_all_layer_kv_split",
    "aiv",
)
@register_impl(
    "mem_cache.transfer_kv_all_layer_kv_split_lf_pf",
    "auto",
)
@register_impl(
    "mem_cache.transfer_kv_all_layer_kv_split",
    "auto",
)
@register_impl("mem_cache.transfer_kv_per_layer_kv_split_pf_lf", "auto")
@register_impl("mem_cache.transfer_kv_per_layer_kv_split", "auto")
def _aiv_impl(*args, **kwargs):
    global _memfabric_fn
    if _memfabric_fn is None:
        from ...implementation.memfabric.mem_cache.kv_transfer import transfer_kv

        _memfabric_fn = transfer_kv
    return _memfabric_fn(*args, **kwargs)


@register_impl("mem_cache.transfer_kv_per_layer_kv_split_pf_lf", "torch_native")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split_lf_pf", "torch_native")
@register_impl("mem_cache.transfer_kv_per_layer_kv_split", "torch_native")
@register_impl("mem_cache.transfer_kv_all_layer_kv_split", "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _reference_fn
    if _reference_fn is None:
        from ...implementation.torch_native.mem_cache.kv_transfer import transfer_kv

        _reference_fn = transfer_kv
    return _reference_fn(*args, **kwargs)


@register_impl("mem_cache.transfer_kv_range_kv_split_pf_lf", "aiv")
@register_impl("mem_cache.transfer_kv_range_kv_split_pf_lf", "auto")
@register_impl("mem_cache.transfer_kv_range_kv_split", "aiv")
@register_impl("mem_cache.transfer_kv_range_kv_split", "auto")
def _aiv_range_impl(*args, **kwargs):
    global _memfabric_range_fn
    if _memfabric_range_fn is None:
        from ...implementation.memfabric.mem_cache.kv_transfer import transfer_kv_range

        _memfabric_range_fn = transfer_kv_range
    return _memfabric_range_fn(*args, **kwargs)


@register_impl("mem_cache.transfer_kv_range_kv_split_pf_lf", "torch_native")
@register_impl("mem_cache.transfer_kv_range_kv_split", "torch_native")
def _torch_native_range_impl(*args, **kwargs):
    global _reference_range_fn
    if _reference_range_fn is None:
        from ...implementation.torch_native.mem_cache.kv_transfer import (
            transfer_kv_range,
        )

        _reference_range_fn = transfer_kv_range
    return _reference_range_fn(*args, **kwargs)


def _index_layers(device_layer_id, host_layer_id):
    if device_layer_id is None and host_layer_id is None:
        return (0, 0, 0)
    if device_layer_id is None or host_layer_id is None:
        raise ValueError("Device and Host Indexer layer IDs must be supplied together")
    return (device_layer_id, host_layer_id, 1)


@register_operator("mem_cache.transfer_kv_per_layer_kv_split_pf_lf")
def transfer_kv_per_layer_kv_split_pf_lf(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    device_layer_id,
    host_layer_id,
    device_index_layer_id=None,
    host_index_layer_id=None,
    impl="auto",
):
    """H2D one KV layer and its optional Indexer slot.

    Device K/V: [L,P,T,H,D]; Host storage: [P,L,T,H,D].
    Indices are int64 complete-page token slots on CPU or this NPU.
    Optional Indexer/scale tensors are copied only when both operands exist.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto selects aiv; explicit impl can override it.
    """
    return get_impl_by_name("mem_cache.transfer_kv_per_layer_kv_split_pf_lf", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout="page_first",
        direction=1,
        kv_layers=(device_layer_id, host_layer_id, 1),
        index_layers=_index_layers(device_index_layer_id, host_index_layer_id),
    )


@register_operator("mem_cache.transfer_kv_all_layer_kv_split_lf_pf")
def transfer_kv_all_layer_kv_split_lf_pf(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layer_start=0,
    impl="auto",
):
    """D2H every Device KV layer into Host layers starting at host_layer_start.

    Device K/V: [L,P,T,H,D]; Host storage: [P,L,T,H,D].
    Indices are int64 complete-page token slots on CPU or this NPU.
    Optional Indexer/scale tensors are copied only when both operands exist.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto selects aiv; explicit impl can override it.
    """
    return get_impl_by_name("mem_cache.transfer_kv_all_layer_kv_split_lf_pf", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout="page_first",
        direction=2,
        kv_layers=(0, host_layer_start, None),
        index_layers=(0, 0, None),
    )


@register_operator("mem_cache.transfer_kv_per_layer_kv_split")
def transfer_kv_per_layer_kv_split(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    device_layer_id,
    host_layer_id,
    device_index_layer_id=None,
    host_index_layer_id=None,
    impl="auto",
):
    """H2D one KV layer and its optional Indexer slot.

    Device K/V: [L,P,T,H,D]; Host storage: [L,P,T,H,D].
    Indices are int64 complete-page token slots on CPU or this NPU.
    Optional Indexer/scale tensors are copied only when both operands exist.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto selects aiv; explicit impl can override it.
    """
    return get_impl_by_name("mem_cache.transfer_kv_per_layer_kv_split", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout="layer_first",
        direction=1,
        kv_layers=(device_layer_id, host_layer_id, 1),
        index_layers=_index_layers(device_index_layer_id, host_index_layer_id),
    )


@register_operator("mem_cache.transfer_kv_all_layer_kv_split")
def transfer_kv_all_layer_kv_split(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layer_start=0,
    impl="auto",
):
    """D2H every Device KV layer into Host layers starting at host_layer_start.

    Device K/V: [L,P,T,H,D]; Host storage: [L,P,T,H,D].
    Indices are int64 complete-page token slots on CPU or this NPU.
    Optional Indexer/scale tensors are copied only when both operands exist.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto selects aiv; explicit impl can override it.
    """
    return get_impl_by_name("mem_cache.transfer_kv_all_layer_kv_split", impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout="layer_first",
        direction=2,
        kv_layers=(0, host_layer_start, None),
        index_layers=(0, 0, None),
    )


def _range_submit(
    op_name,
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k,
    host_index_k,
    device_index_k_scale,
    host_index_k_scale,
    page_size,
    host_layout,
    num_ctrl_layers,
    iter_table,
    pool_iter_counts,
    event_pool_id,
    event_first_index,
    impl,
):
    return get_impl_by_name(op_name, impl)(
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        host_layout=host_layout,
        direction=1,
        num_ctrl_layers=num_ctrl_layers,
        iter_table=iter_table,
        pool_iter_counts=pool_iter_counts,
        event_pool_id=event_pool_id,
        event_first_index=event_first_index,
    )


@register_operator("mem_cache.transfer_kv_range_kv_split_pf_lf")
def transfer_kv_range_kv_split_pf_lf(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    num_ctrl_layers,
    iter_table,
    pool_iter_counts,
    event_pool_id=-1,
    event_first_index=0,
    impl="auto",
):
    """Fused H2D of every pool layer, one kernel launch per layer, one call.

    Per-pool lists hold FULL component buffers: Device K/V [L,P,T,H,D]; Host
    storage [P,L,T,H,D]. iter_table rows [ctrl, device_layer, host_layer,
    index_slot] (sorted by ctrl, pool_iter_counts rows per pool) select each
    launch's layer; slot < 0 skips the Indexer pair for that layer. The
    implementation records one kv_exchange event-pool event per controller
    layer when event_pool_id >= 0. Optional components are None or empty.
    Mutates the destination, returns None, and enqueues on the current stream.
    auto selects aiv; explicit impl can override it.
    """
    return _range_submit(
        "mem_cache.transfer_kv_range_kv_split_pf_lf",
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        "page_first",
        num_ctrl_layers,
        iter_table,
        pool_iter_counts,
        event_pool_id,
        event_first_index,
        impl,
    )


@register_operator("mem_cache.transfer_kv_range_kv_split")
def transfer_kv_range_kv_split(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    num_ctrl_layers,
    iter_table,
    pool_iter_counts,
    event_pool_id=-1,
    event_first_index=0,
    impl="auto",
):
    """Fused H2D of every pool layer for layer-first Host storage [L,P,T,H,D].

    See transfer_kv_range_kv_split_pf_lf for the iter_table and component
    contract; only the Host layout differs.
    """
    return _range_submit(
        "mem_cache.transfer_kv_range_kv_split",
        device_indices,
        host_indices,
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        page_size,
        "layer_first",
        num_ctrl_layers,
        iter_table,
        pool_iter_counts,
        event_pool_id,
        event_first_index,
        impl,
    )
