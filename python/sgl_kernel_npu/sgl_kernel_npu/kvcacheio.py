from enum import Enum
from typing import List, Optional, Sequence

import torch

# HostMemoryMapping and the aclrt registration itself come from host_memory,
# which mirrors PR #39548 srt/hardware_backend/npu/host_memory.py call-for-call:
# the aclrt calls go through the `acl.rt` python binding, never torch.ops.npu.
from sgl_kernel_npu.host_memory import HostMemoryMapping
import sgl_kernel_npu.host_memory as _hm

_PAGE_SIZE = 4096

# Process-level table of owned HostMemoryMappings, keyed by the host range the
# caller registered: (host_ptr, nbytes, mapping), registration order. Pure
# python like PR #39548 -- it backs get_device_accessible_ptr and lets the
# transfer ops resolve host pool DVAs without the caller carrying mappings.
_G_MAPPINGS: List[tuple] = []


def _lookup_mapping(host_ptr: int):
    """Return ``(mapping, reg_ptr)`` for the registered range holding
    ``host_ptr``, or ``(None, 0)`` when the pointer is not registered here."""
    if host_ptr:
        for reg_ptr, nbytes, mapping in reversed(_G_MAPPINGS):
            if reg_ptr <= host_ptr < reg_ptr + nbytes:
                return mapping, reg_ptr
    return None, 0


class TransferDirection(Enum):
    H2D = 1
    D2H = 2


class TransferFlag(Enum):
    FAST2D = 2


def transfer_state_per_layer_direct_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    flags: TransferFlag = TransferFlag.FAST2D,
) -> None:
    torch.ops.npu.transfer_state_per_layer_direct_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        flags.value,
    )


def transfer_state_all_layer_direct_lf_pf(
    device_states: Sequence[torch.Tensor],
    host_states: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    flags: TransferFlag = TransferFlag.FAST2D,
) -> None:
    torch.ops.npu.transfer_state_all_layer_direct_lf_pf(
        list(device_states),
        list(host_states),
        device_indices,
        host_indices,
        flags.value,
    )


def transfer_kv_dim_exchange(
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    device_k: torch.Tensor,
    host_k: torch.Tensor,
    device_v: torch.Tensor,
    host_v: torch.Tensor,
    device_index_k: Optional[torch.Tensor] = None,
    host_index_k: Optional[torch.Tensor] = None,
    device_index_k_scale: Optional[torch.Tensor] = None,
    host_index_k_scale: Optional[torch.Tensor] = None,
    page_size: int = 128,
    direction: TransferDirection = TransferDirection.H2D,
    flags: TransferFlag = TransferFlag.FAST2D,
):
    """
    In the L1 and L2 radix cache scenarios, perform batch copy of KV data between the device and the host.

    Args:
        device_indices: token indices in device
        host_indices: token indices in host
        device_k: k_buffer in device
        host_k: k_buffer in host
        device_v: v_buffer in device
        host_v: v_buffer in host
        device_index_k: index_k_buffer in device
        host_index_k: index_k_buffer in host
        device_index_k_scale: per-token FP32 scale of the quantized Indexer in device
        host_index_k_scale: per-token FP32 scale of the quantized Indexer in host
        page_size: page size
        direction: only support H2D and D2H.
        flags: only FAST2D is supported, which indicates 2D data transfer via calling aclrtMemcpy2dAsync.
    """
    torch.ops.npu.transfer_kv_dim_exchange(
        device_k,
        host_k,
        device_v,
        host_v,
        device_indices,
        host_indices,
        page_size,
        direction.value,
        flags.value,
    )
    if device_index_k is not None and host_index_k is not None:
        torch.ops.npu.transfer_kv_dim_exchange(
            device_index_k,
            host_index_k,
            torch.empty(0),
            torch.empty(0),
            device_indices,
            host_indices,
            page_size,
            direction.value,
            flags.value,
        )
    if device_index_k_scale is not None and host_index_k_scale is not None:
        # Device scale cache is 4-D (layers, pages, page_size, 1) while the host
        # cache is 5-D (pages, layers, page_size, 1, 1); the kernel requires both
        # operands to be 5-D, so pad the device operand with a trailing singleton.
        if device_index_k_scale.dim() == 4:
            device_index_k_scale = device_index_k_scale.unsqueeze(-1)
        torch.ops.npu.transfer_kv_dim_exchange(
            device_index_k_scale,
            host_index_k_scale,
            torch.empty(0),
            torch.empty(0),
            device_indices,
            host_indices,
            page_size,
            direction.value,
            flags.value,
        )


def transfer_mamba_state(
    device_buf: torch.Tensor,
    host_buf: torch.Tensor,
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
    direction: TransferDirection = TransferDirection.H2D,
):
    """
    Transfer Mamba/SSM state between device (layer-first) and host (page-first).

    Device buffer layout: [num_layers, device_size, *state_shape]  (layer-first)
    Host buffer layout:   [host_size, num_layers, 1, *state_shape] (page-first)

    Uses aclrtMemcpy2dAsync for efficient 2D strided copy, transferring all
    layers for each slot index in a single 2D copy call.

    Args:
        device_buf: device Mamba state buffer [num_layers, size, *shape]
        host_buf: host Mamba state buffer [size, num_layers, 1, *shape]
        device_indices: slot indices in device buffer
        host_indices: slot indices in host buffer
        direction: H2D (host→device) or D2H (device→host)
    """
    torch.ops.npu.transfer_mamba_state(
        device_buf,
        host_buf,
        device_indices,
        host_indices,
        direction.value,
    )


def register_host_memory(tensor: torch.Tensor) -> HostMemoryMapping:
    """
    Map a host (CPU) tensor into the device address space and return its
    HostMemoryMapping. Pure python, PR #39548 style: the aclrt calls go
    through host_memory.py's ``acl.rt`` binding and the mapping table is kept
    here in python -- no torch.ops.npu in the registration path at all.

    1. page-align the range down/up (the returned device_ptr carries the inner
       offset so kernels can use it directly);
    2. borrow an existing device mapping (aclrtHostGetDevicePointer) when one
       is already present -- borrowed mappings are never unregistered here;
    3. otherwise aclrtHostRegister(ACL_HOST_REGISTER_MAPPED); on failure the
       borrow lookup runs once more (another component may have registered
       concurrently).

    ``tensor`` must be kept alive until the mapping is released via
    unregister_host_memory.
    """
    host_ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    if not tensor.is_cpu:
        raise RuntimeError("register_host_memory: tensor must be host (CPU) memory")
    if nbytes <= 0:
        raise RuntimeError("register_host_memory: tensor must be non-empty")
    mapping = _hm.register_host_memory(host_ptr, nbytes)
    if mapping is None:
        raise RuntimeError(
            f"aclrtHostRegister failed for {hex(host_ptr)} ({nbytes} bytes); "
            "host memory is not device-mapped and transfers would read stale data"
        )
    if mapping.owned:
        _G_MAPPINGS.append((host_ptr, nbytes, mapping))
    return mapping


def unregister_host_memory(mapping: HostMemoryMapping) -> None:
    """Release a mapping returned by register_host_memory.

    Borrowed mappings (``owned=False``) are left untouched, mirroring PR
    #39548's unregister_host_memory ownership rule.
    """
    if mapping.owned:
        for i, (_, _, m) in enumerate(_G_MAPPINGS):
            if m is mapping:
                del _G_MAPPINGS[i]
                break
        else:
            raise RuntimeError(
                "unregister_host_memory: mapping for "
                f"{hex(mapping.registered_host_ptr)} is not registered"
            )
    _hm.unregister_host_memory(mapping)


def aclrt_host_register(tensor: torch.Tensor) -> HostMemoryMapping:
    """Backward-compatible alias of register_host_memory."""
    return register_host_memory(tensor)


def aclrt_host_unregister(mapping: HostMemoryMapping) -> None:
    """Backward-compatible alias of unregister_host_memory."""
    unregister_host_memory(mapping)


def get_device_accessible_ptr(tensor: torch.Tensor, device_index: int) -> int:
    """Return the address a kernel on ``device_index`` must use for ``tensor``.

    Mirrors sgl_kernel.kvcacheio.get_device_accessible_ptr (CUDA): registered
    host memory resolves to its DVA, everything else keeps its raw address.
    Resolved through the python mapping table (see _G_MAPPINGS).
    """
    host_ptr = tensor.data_ptr() if tensor.is_cpu else 0
    mapping, reg_ptr = _lookup_mapping(host_ptr)
    if mapping is not None:
        return mapping.device_ptr + (host_ptr - reg_ptr)
    return tensor.data_ptr()


def set_host_dva_mode(mode: str) -> None:
    """Select how the kv_exchange transfer ops resolve host pool pointers.

    "aclrt" (default) resolves through the aclrtHostRegister table populated
    by register_host_memory; "passthrough" hands host pointers to the kernel
    unchanged (they must already be device-accessible). This is the swap point
    for a future memfabric DVA provider — transfer ops and the AIV kernel are
    provider-agnostic.
    """
    torch.ops.npu.aclrt_set_host_dva_mode(mode)


def _resolve_host_dvas(host_tensors, host_mappings) -> List[int]:
    """Explicit per-component DVA (PR #39548 style): the HostMemoryMapping
    (or device_ptr int) if the caller passes one, otherwise the host tensor is
    resolved through the python mapping table; 0 = unregistered, the op raises
    a clear error for it."""
    out = []
    for i, tensor in enumerate(host_tensors):
        m = host_mappings[i] if host_mappings and i < len(host_mappings) else None
        if m is None:
            ptr = tensor.data_ptr() if tensor.is_cpu else 0
            mapping, reg_ptr = _lookup_mapping(ptr)
            out.append(
                0 if mapping is None else mapping.device_ptr + (ptr - reg_ptr)
            )
        elif isinstance(m, int):
            out.append(m)
        else:
            out.append(m.device_ptr)
    return out


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
    host_mappings: Optional[Sequence] = None,
):
    """
    H2D, one layer: registered host pool (page-first, 5-D) -> device pool
    (layer-first, 5-D). src_ptrs/dst_ptrs are matching component lists
    (k/v/index_k/index_k_scale); host-side tensors must be registered via
    register_host_memory. ``host_mappings`` optionally carries the
    HostMemoryMapping (or device_ptr int) per component -- PR #39548 style
    explicit mapping; None entries or omitting the argument resolve the DVA
    through the python mapping table instead. Signature mirrors
    sgl_kernel (CUDA).
    """
    torch.ops.npu.transfer_kv_per_layer_direct_pf_lf(
        list(src_ptrs), list(dst_ptrs), src_indices, dst_indices, layer_id, page_size,
        _resolve_host_dvas(src_ptrs, host_mappings),
    )


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
    host_mappings: Optional[Sequence] = None,
):
    """
    D2H, all layers: device pool (layer-first, 5-D) -> registered host pool
    (page-first, 5-D). Per-component layer ranges follow the device buffer's
    layer count, so components covering fewer layers (indexer k / scales) need
    no extra mapping on the caller side. ``host_mappings`` follows the same
    contract as in transfer_kv_per_layer_direct_pf_lf. Signature mirrors
    sgl_kernel (CUDA).
    """
    torch.ops.npu.transfer_kv_all_layer_direct_lf_pf(
        list(src_ptrs), list(dst_ptrs), src_indices, dst_indices, page_size,
        _resolve_host_dvas(dst_ptrs, host_mappings),
    )
