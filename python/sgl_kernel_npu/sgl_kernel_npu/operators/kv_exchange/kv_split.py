"""MF+AIV kv_split transfer operators.

MLA HiCache L2<->L1 IO between a host ``page_first_kv_split`` pool
(pages, layers, page_size, 1, width) and a device ``layer_first_kv_split``
pool (layers, pages, page_size, 1, width), carried by the AIV kv_exchange
kernel over registered MemFabric host blocks.  The caller hands over the
FULL host pool components plus explicit layer ids; the layer offsets are
folded into the device base pointer and host DVA inside the C++ adapter,
so the framework never builds host layer slices itself.
"""

import torch

from ...host_memory import resolve_host_dva
from ..utils.registry import get_impl_by_name, register_impl, register_operator

# Pinned index-staging buffers kept alive until their async H2D copy retires.
_PINNED_INFLIGHT = []


def _is_empty(tensor):
    return tensor is None or tensor.numel() == 0


def _pair_error(name, device_tensor, host_tensor):
    return ValueError(
        f"component {name}: device/host operands must both be present and "
        f"non-empty or both empty, got device="
        f"{None if device_tensor is None else tuple(device_tensor.shape)}, "
        f"host={None if host_tensor is None else tuple(host_tensor.shape)}"
    )


def _validate_pair(name, device_tensor, host_tensor):
    """Require a fully present or fully absent component pair."""
    if _is_empty(device_tensor) != _is_empty(host_tensor):
        raise _pair_error(name, device_tensor, host_tensor)
    return not _is_empty(device_tensor)


def _prune_pinned_staging():
    _PINNED_INFLIGHT[:] = [
        entry for entry in _PINNED_INFLIGHT if not entry[1].query()
    ]


def _upload_indices(indices, device):
    """Upload CPU indices through pinned staging; NPU indices pass through.

    A pageable ``.to(device)`` completes with an aclrtStreamSynchronize that
    drains everything queued on the stream, so CPU indices are staged through
    pinned memory instead; an event keeps the pinned copy alive until the
    consumer copy retires (Event.query() never synchronizes).
    """
    if indices.device.type != "cpu":
        return indices
    pinned = indices.pin_memory()
    out = torch.empty(pinned.shape, dtype=pinned.dtype, device=device)
    out.copy_(pinned, non_blocking=True)
    done = torch.npu.Event()
    done.record()
    _PINNED_INFLIGHT.append((pinned, done))
    _prune_pinned_staging()
    return out


def _upload_and_check_indices(device_indices, host_indices, device):
    if device_indices.numel() != host_indices.numel():
        raise ValueError(
            "device_indices and host_indices must have the same length, got "
            f"{device_indices.numel()} and {host_indices.numel()}"
        )
    if device_indices.numel() == 0:
        return None
    return (
        _upload_indices(device_indices, device),
        _upload_indices(host_indices, device),
    )


@register_impl("kv_exchange.transfer_kv_per_layer_kv_split_pf_lf", "auto")
@register_impl("kv_exchange.transfer_kv_per_layer_kv_split_pf_lf", "ascendc")
def _ascendc_per_layer_impl(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v=None,
    host_v=None,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    layer_id,
    host_layer_id=None,
    index_k_layer_id=-1,
):
    if host_layer_id is None:
        host_layer_id = layer_id

    device_components = []
    host_components = []
    device_layer_ids = []
    host_layer_ids = []
    host_dvas = []

    def _add(device_tensor, host_tensor, device_layer, host_layer):
        device_components.append(device_tensor)
        host_components.append(host_tensor)
        device_layer_ids.append(device_layer)
        host_layer_ids.append(host_layer)
        host_dvas.append(resolve_host_dva(host_tensor))

    if not _validate_pair("k", device_k, host_k):
        raise ValueError("k components are required and must be non-empty")
    _add(device_k, host_k, layer_id, host_layer_id)
    if _validate_pair("v", device_v, host_v):
        _add(device_v, host_v, layer_id, host_layer_id)

    # Provided index components must be paired regardless of the layer gate:
    # index scale only ever rides along a paired index K.  The per-layer
    # index_k_layer_id then selects the paired components; a negative value
    # means the current device layer is not an indexer layer, so the index
    # components are skipped and the K/V transfer is unaffected.
    index_paired = _validate_pair("index_k", device_index_k, host_index_k)
    scale_paired = _validate_pair(
        "index_k_scale", device_index_k_scale, host_index_k_scale
    )
    if not index_paired and scale_paired:
        raise ValueError(
            "index_k_scale can only be transferred together with a paired "
            "index_k component"
        )
    if index_paired and index_k_layer_id >= 0:
        _add(device_index_k, host_index_k, index_k_layer_id, index_k_layer_id)
        if scale_paired:
            _add(
                device_index_k_scale,
                host_index_k_scale,
                index_k_layer_id,
                index_k_layer_id,
            )

    indices = _upload_and_check_indices(
        device_indices, host_indices, device_k.device
    )
    if indices is None:
        return
    device_indices, host_indices = indices

    torch.ops.npu.transfer_kv_per_layer_kv_split_pf_lf(
        device_indices,
        host_indices,
        device_components,
        host_components,
        host_dvas,
        device_layer_ids,
        host_layer_ids,
        page_size,
    )


@register_impl("kv_exchange.transfer_kv_all_layer_kv_split_lf_pf", "auto")
@register_impl("kv_exchange.transfer_kv_all_layer_kv_split_lf_pf", "ascendc")
def _ascendc_all_layer_impl(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v=None,
    host_v=None,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layer_start=0,
):
    device_components = []
    host_components = []
    host_dvas = []

    def _add(device_tensor, host_tensor):
        device_components.append(device_tensor)
        host_components.append(host_tensor)
        host_dvas.append(resolve_host_dva(host_tensor))

    if not _validate_pair("k", device_k, host_k):
        raise ValueError("k components are required and must be non-empty")
    _add(device_k, host_k)
    if _validate_pair("v", device_v, host_v):
        _add(device_v, host_v)

    # Index components ride along only when both sides are paired; the layer
    # count is the device operand's dim(0), starting at host_layer_start.
    if _validate_pair("index_k", device_index_k, host_index_k):
        _add(device_index_k, host_index_k)
        if _validate_pair("index_k_scale", device_index_k_scale, host_index_k_scale):
            _add(device_index_k_scale, host_index_k_scale)
    elif _validate_pair("index_k_scale", device_index_k_scale, host_index_k_scale):
        raise ValueError(
            "index_k_scale can only be transferred together with a paired "
            "index_k component"
        )

    indices = _upload_and_check_indices(
        device_indices, host_indices, device_k.device
    )
    if indices is None:
        return
    device_indices, host_indices = indices

    torch.ops.npu.transfer_kv_all_layer_kv_split_lf_pf(
        device_indices,
        host_indices,
        device_components,
        host_components,
        host_dvas,
        page_size,
        host_layer_start,
    )


@register_operator("kv_exchange.transfer_kv_per_layer_kv_split_pf_lf")
def transfer_kv_per_layer_kv_split_pf_lf(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v=None,
    host_v=None,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    layer_id,
    host_layer_id=None,
    index_k_layer_id=-1,
    impl="auto",
):
    """H2D one layer: host page_first_kv_split -> device layer_first_kv_split.

    Receives the FULL host pool components, not framework-cut host layer
    views: the device layer offset (``layer_id``) and the host layer offset
    (``host_layer_id``, defaulting to ``layer_id``) are folded into the base
    addresses inside the C++ adapter.  ``index_k_layer_id`` selects the
    device/host index K layers and their FP32 scales together; a negative
    value skips the index components.  Indices stay token-slot numbered.

    Args:
        device_indices: device token-slot indices (NPU, or CPU uploaded via
            pinned staging with non-blocking copies)
        host_indices: host token-slot indices (NPU, or CPU uploaded likewise)
        device_k: device K pool [layer, page, page_size, 1, width]
        host_k: full host K pool [page, layer, page_size, 1, width] (MF
            registered memory)
        device_v / host_v: optional V pools, both sides empty to skip
        device_index_k / host_index_k: optional indexer K pools
        device_index_k_scale / host_index_k_scale: optional FP32 scales
            ([index_layer, page, page_size, 1] device / page-first host)
        page_size: token slots per page
        layer_id: device K/V layer id
        host_layer_id: layer id inside the full host K/V pool
        index_k_layer_id: index K layer id; negative skips index components
        impl: implementation selector, "auto" resolves to ascendc
    """
    impl_fn = get_impl_by_name(
        "kv_exchange.transfer_kv_per_layer_kv_split_pf_lf", impl
    )
    return impl_fn(
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
        layer_id=layer_id,
        host_layer_id=host_layer_id,
        index_k_layer_id=index_k_layer_id,
    )


@register_operator("kv_exchange.transfer_kv_all_layer_kv_split_lf_pf")
def transfer_kv_all_layer_kv_split_lf_pf(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v=None,
    host_v=None,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layer_start=0,
    impl="auto",
):
    """D2H all layers: device layer_first_kv_split -> host page_first_kv_split.

    Receives the FULL host pool components together with ``host_layer_start``,
    the host layer id of the device pool's layer 0 inside the full host K/V
    pool; the per-component transferred layer count is the device operand's
    dim(0).  Indices stay token-slot numbered.

    Args:
        device_indices: device token-slot indices (NPU, or CPU uploaded via
            pinned staging with non-blocking copies)
        host_indices: host token-slot indices (NPU, or CPU uploaded likewise)
        device_k: device K pool [layer, page, page_size, 1, width]
        host_k: full host K pool [page, layer, page_size, 1, width] (MF
            registered memory)
        device_v / host_v: optional V pools, both sides empty to skip
        device_index_k / host_index_k: optional indexer K pools
        device_index_k_scale / host_index_k_scale: optional FP32 scales
            ([index_layer, page, page_size, 1] device / page-first host)
        page_size: token slots per page
        host_layer_start: host layer id of device layer 0
        impl: implementation selector, "auto" resolves to ascendc
    """
    impl_fn = get_impl_by_name(
        "kv_exchange.transfer_kv_all_layer_kv_split_lf_pf", impl
    )
    return impl_fn(
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
        host_layer_start=host_layer_start,
    )
