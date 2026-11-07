"""MemFabric KV transfers on the caller's current NPU stream.

The caller owns initialized MemFabric resources and Host storage. Input
lifetime rides on the caching allocator's stream-ordered reuse (the C++
adapter record_stream's every NPU input). The AIV kv_exchange kernel
receives the whole transfer description inside its launch parameter block,
so no pinned staging, meta upload or per-submission event remains on the
submission path. NPU graph capture is not supported by this AIV path.
"""

from threading import Lock

import torch

_submit_lock = Lock()


def _check_component(device, host, page_size, index_device):
    if device.device != index_device or host.device.type != "cpu":
        raise ValueError("KV components must be on the index NPU and CPU respectively")
    # _layer_views already checked dimensions and selected equal layer counts.
    if device.dtype != host.dtype or device.shape[0] <= 0:
        raise ValueError(
            "KV component dtypes must match and layer count must be positive"
        )
    if device.shape[2:] != host.shape[2:] or device.shape[2] != page_size:
        raise ValueError("KV component page shapes must match")
    for tensor in (device, host):
        if (
            tensor.stride(4) != 1
            or tensor.stride(3) != tensor.shape[4]
            or tensor.stride(2) != tensor.shape[3] * tensor.shape[4]
        ):
            raise ValueError("each (page,layer) payload must be contiguous")
    page_bytes = page_size * device.shape[3] * device.shape[4] * device.element_size()
    if page_bytes <= 0:
        raise ValueError("KV component page byte count must be positive")


def _layer_views(dev, host, host_layout, layers):
    if dev.ndim != 5 or host.ndim != 5:
        raise ValueError(
            "KV components must be five-dimensional (Device scales may be four-dimensional)"
        )
    if host_layout == "layer_first":
        host = host.transpose(0, 1)
    device_start, host_start, count = layers
    count = dev.shape[0] - device_start if count is None else count
    if (
        device_start < 0
        or host_start < 0
        or count < 0
        or device_start + count > dev.shape[0]
        or host_start + count > host.shape[1]
    ):
        raise ValueError("KV layer selection exceeds Device or Host storage")
    if device_start != 0 or count != dev.shape[0]:
        dev = dev.narrow(0, device_start, count)
    if host_start != 0 or count != host.shape[1]:
        host = host.narrow(1, host_start, count)
    return dev, host


def select_components(
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k,
    host_index_k,
    device_index_k_scale,
    host_index_k_scale,
    host_layout,
    kv_layers,
    index_layers,
):
    """Select the active K/V/Indexer/scale views."""
    if host_layout not in ("page_first", "layer_first"):
        raise ValueError("host_layout must be page_first or layer_first")
    if device_k is None or host_k is None or not device_k.numel() or not host_k.numel():
        raise ValueError("K buffers must be nonempty")
    if device_v is None or host_v is None:
        raise ValueError(
            "V buffers must be tensors; use an empty Device V for no V payload"
        )
    components = [_layer_views(device_k, host_k, host_layout, kv_layers)]
    if device_v.numel():
        if not host_v.numel():
            raise ValueError("nonempty Device V requires a nonempty Host V")
        components.append(_layer_views(device_v, host_v, host_layout, kv_layers))
    if index_layers[2] != 0:
        for dev, host, is_scale in (
            (device_index_k, host_index_k, False),
            (device_index_k_scale, host_index_k_scale, True),
        ):
            if dev is not None and host is not None:
                if dev.numel() or host.numel():
                    if not dev.numel() or not host.numel():
                        raise ValueError(
                            "paired Indexer components must both be nonempty"
                        )
                    if is_scale and dev.ndim == 4:
                        dev = dev.unsqueeze(-1)
                    components.append(
                        _layer_views(dev, host, host_layout, index_layers)
                    )
                    continue
    return components


def _resolve_host_dva(host):
    # Mirrors the mf_aiv host_memory contract, device-validated there: the
    # registration table (host_register + host_get_device_pointer at
    # allocation time, interior views by byte offset) is the ONLY DVA source;
    # unregistered MF memory is an error with no HVA/get_dva fallback.
    from sgl_kernel_npu.host_memory import resolve_host_dva

    return resolve_host_dva(host)


def _submit_aiv(components, di, hi, page_size, direction, host_layout):
    # Host tensors have PF-shaped metadata views even when their storage is LF.
    # Actual page/layer strides preserve the original physical layout.
    # The C++ kv_exchange adapter packs the component/pitch/width/layer
    # description into the kernel launch parameter block and record_stream's
    # every NPU input, so nothing else outlives the call on the host side.
    host_dvas = [_resolve_host_dva(host) for _, host in components]
    torch.ops.npu.kv_exchange(
        di,
        hi,
        [dev for dev, _ in components],
        [host for _, host in components],
        host_dvas,
        page_size,
        direction,
        int(host_layout == "layer_first"),
    )


def transfer_kv(
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
    host_layout,
    direction,
    kv_layers,
    index_layers,
):
    """Submit one selected pool/range without synchronizing the caller's stream."""
    if page_size <= 0 or device_indices.ndim != 1 or host_indices.ndim != 1:
        raise ValueError("expected positive page_size and 1-D token indices")
    if device_indices.dtype != torch.int64 or host_indices.dtype != torch.int64:
        raise ValueError("KV indices must have dtype int64")
    if (
        device_indices.numel() != host_indices.numel()
        or device_indices.numel() % page_size
    ):
        raise ValueError("indices must describe equal numbers of complete pages")
    if direction not in (1, 2):
        raise ValueError("expected H2D (1) or D2H (2)")
    components = select_components(
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        host_layout,
        kv_layers,
        index_layers,
    )
    device = device_k.device
    if device.type != "npu":
        raise ValueError("KV Device buffers must be NPU tensors")
    if device_indices.device != device or host_indices.device != device:
        raise ValueError("KV indices must be on the KV buffer's NPU")
    for dev, host in components:
        _check_component(dev, host, page_size, device)
    if not device_indices.numel():
        return
    # Host addresses resolve through the registered DVA table (MF or aclrt).
    # Input lifetime needs no Python-side event bookkeeping: the C++ adapter
    # record_stream's every NPU input, so the caching allocator's
    # stream-ordered reuse defers any freed slot until this submission
    # retires, and the host component views are persistent pool buffers.

    with _submit_lock, torch.npu.device(device):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("AIV KV transfer does not support NPU graph capture")
        di = device_indices.contiguous()
        hi = host_indices.contiguous()
        _submit_aiv(components, di, hi, page_size, direction, host_layout)  # fmt: skip


def _range_host_view(host, host_layout):
    """PF-shaped metadata view of one full host component buffer."""
    if host_layout == "layer_first":
        return host.transpose(0, 1)
    return host


def _range_dva(view):
    if not view.numel():
        return 0
    return _resolve_host_dva(view)


def transfer_kv_range(
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
    host_layout,
    direction,
    num_ctrl_layers,
    iter_table,
    pool_iter_counts,
    event_pool_id=-1,
    event_first_index=0,
):
    """Submit every pool's per-layer H2D program in one call.

    The per-pool lists hold FULL component buffers (no per-layer slicing);
    iter_table rows [ctrl, device_layer, host_layer, index_slot] select one
    layer per launch inside the C++ loop, which also records one event per
    controller layer from the kv_exchange event pool. Same component contract
    as transfer_kv: paired components only, K mandatory, V optional, the
    Indexer pair optional per pool and per iteration (slot < 0).
    """
    if page_size <= 0 or device_indices.ndim != 1 or host_indices.ndim != 1:
        raise ValueError("expected positive page_size and 1-D token indices")
    if device_indices.dtype != torch.int64 or host_indices.dtype != torch.int64:
        raise ValueError("KV indices must have dtype int64")
    if (
        device_indices.numel() != host_indices.numel()
        or device_indices.numel() % page_size
    ):
        raise ValueError("indices must describe equal numbers of complete pages")
    if direction != 1:
        raise ValueError("the fused range submission is H2D only (direction 1)")
    if host_layout not in ("page_first", "layer_first"):
        raise ValueError("host_layout must be page_first or layer_first")
    num_pools = len(pool_iter_counts)
    if not num_pools or any(
        x is None or len(x) != num_pools for x in (device_k, host_k, device_v, host_v)
    ):
        raise ValueError("per-pool component lists must match pool_iter_counts")
    # Absent optional components arrive as None lists; normalize to per-pool
    # None entries (handled as empty placeholders below).
    device_index_k = [None] * num_pools if device_index_k is None else device_index_k
    host_index_k = [None] * num_pools if host_index_k is None else host_index_k
    device_index_k_scale = (
        [None] * num_pools if device_index_k_scale is None else device_index_k_scale
    )
    host_index_k_scale = (
        [None] * num_pools if host_index_k_scale is None else host_index_k_scale
    )
    if any(
        len(x) != num_pools
        for x in (
            device_index_k,
            host_index_k,
            device_index_k_scale,
            host_index_k_scale,
        )
    ):
        raise ValueError("per-pool component lists must match pool_iter_counts")
    if not device_indices.numel():
        # Nothing to transfer, but events still fire per controller layer.
        if event_pool_id >= 0:
            for layer in range(num_ctrl_layers):
                torch.ops.npu.kv_exchange_event_pool_record(
                    event_pool_id, event_first_index + layer
                )
        return
    device = device_k[0].device
    if device.type != "npu":
        raise ValueError("KV Device buffers must be NPU tensors")
    if device_indices.device != device or host_indices.device != device:
        raise ValueError("KV indices must be on the KV buffer's NPU")

    # Normalize per-pool buffers: PF-shaped host views, 5-D Indexer scales.
    host_dvas = []
    norm = dict(
        dev_k=[], host_k=[], dev_v=[], host_v=[],
        dev_ik=[], host_ik=[], dev_is=[], host_is=[],
    )
    empty = torch.empty(0)
    for pool in range(num_pools):
        for key, dev_buf, host_buf in (
            ("k", device_k[pool], host_k[pool]),
            ("v", device_v[pool], host_v[pool]),
            ("ik", device_index_k[pool], host_index_k[pool]),
            ("is", device_index_k_scale[pool], host_index_k_scale[pool]),
        ):
            if dev_buf is None or host_buf is None:
                # None marks an absent optional component (e.g. a pool
                # without V); the empty placeholder keeps it out of every
                # launch, while a half-supplied pair stays an error.
                if dev_buf is not None or host_buf is not None:
                    raise ValueError(
                        "paired range components must both be supplied or omitted"
                    )
                dev_buf = host_buf = empty
            if dev_buf.numel() and host_buf.numel():
                if dev_buf.ndim != 5 or host_buf.ndim != 5:
                    raise ValueError(
                        "range components must be five-dimensional (pad Indexer scales)"
                    )
                if dev_buf.dtype != host_buf.dtype or dev_buf.shape[2:] != host_buf.shape[2:]:
                    raise ValueError("range component dtypes and page shapes must match")
                host_view = _range_host_view(host_buf, host_layout)
                norm["dev_" + key].append(dev_buf)
                norm["host_" + key].append(host_view)
                host_dvas.append(_range_dva(host_view))
            else:
                if dev_buf.numel() != host_buf.numel():
                    raise ValueError("paired range components must both be empty or nonempty")
                norm["dev_" + key].append(dev_buf)
                norm["host_" + key].append(host_buf)
                host_dvas.append(0)
    if not norm["dev_k"][0].numel():
        raise ValueError("range pool 0: K buffers must be nonempty")
    if len(iter_table) != 4 * sum(pool_iter_counts):
        raise ValueError("iter_table must hold 4 entries per pool iteration")

    # Host addresses resolve through the registered DVA table (MF or aclrt).
    # Input lifetime needs no Python-side event bookkeeping: the C++ adapter
    # record_stream's every NPU input once for all launches.
    with _submit_lock, torch.npu.device(device):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("AIV KV transfer does not support NPU graph capture")
        torch.ops.npu.kv_exchange_range(
            device_indices.contiguous(),
            host_indices.contiguous(),
            norm["dev_k"], norm["host_k"],
            norm["dev_v"], norm["host_v"],
            norm["dev_ik"], norm["host_ik"],
            norm["dev_is"], norm["host_is"],
            host_dvas,
            page_size,
            direction,
            int(host_layout == "layer_first"),
            num_ctrl_layers,
            list(iter_table),
            list(pool_iter_counts),
            event_pool_id,
            event_first_index,
        )
