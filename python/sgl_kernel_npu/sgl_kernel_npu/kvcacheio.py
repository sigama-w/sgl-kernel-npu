from enum import Enum
from typing import List, Optional, Sequence

import torch

from sgl_kernel_npu.host_memory import HostMemoryMapping
import sgl_kernel_npu.host_memory as _hm

# Registered MF blocks: (base_hva, nbytes, HostMemoryMapping). The table also
# resolves interior tensor views to DVA + byte offset.
_G_MAPPINGS: List[tuple] = []


def _lookup_mapping(host_ptr: int):
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


def initialize_host_memory(device_id: int, reserve_size: int) -> None:
    """Initialize the MemFabric host-memory provider used by these wrappers."""
    _hm.initialize_host_memory(device_id, reserve_size)


def empty_host_memory(shape, dtype=torch.bfloat16) -> torch.Tensor:
    """Allocate a host tensor that can be registered with MemFabric."""
    return _hm.empty_host_memory(shape, dtype=dtype)


def uninitialize_host_memory() -> None:
    """Release all MF blocks and invalidate every Python-side DVA mapping."""
    try:
        _hm.uninitialize_host_memory()
    finally:
        _G_MAPPINGS.clear()


def register_host_memory(tensor: torch.Tensor) -> HostMemoryMapping:
    """Register an MF-allocated CPU tensor and record its DVA mapping."""
    if not tensor.is_cpu:
        raise RuntimeError("register_host_memory: tensor must be host (CPU) memory")
    host_ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes <= 0:
        raise RuntimeError("register_host_memory: tensor must be non-empty")

    mapping = _hm.register_host_memory(host_ptr, nbytes)
    _G_MAPPINGS.append((host_ptr, nbytes, mapping))
    return mapping


def get_device_accessible_ptr(tensor: torch.Tensor, device_index: int) -> int:
    """Resolve a registered host tensor or interior view to its device address."""
    if not tensor.is_cpu:
        return tensor.data_ptr()
    host_ptr = tensor.data_ptr()
    mapping, reg_ptr = _lookup_mapping(host_ptr)
    if mapping is not None:
        return mapping.device_ptr + (host_ptr - reg_ptr)
    return host_ptr


def _resolve_host_dvas(host_tensors, host_mappings) -> List[int]:
    """Resolve every host component without invoking an aclrt fallback."""
    out = []
    for i, tensor in enumerate(host_tensors):
        explicit = (
            host_mappings[i]
            if host_mappings is not None and i < len(host_mappings)
            else None
        )
        if explicit is not None:
            dva = explicit if isinstance(explicit, int) else explicit.device_ptr
        else:
            host_ptr = tensor.data_ptr() if tensor.is_cpu else 0
            mapping, reg_ptr = _lookup_mapping(host_ptr)
            if mapping is None:
                raise RuntimeError(
                    "MF host memory is not registered; call register_host_memory "
                    "with an empty_host_memory tensor before transfer"
                )
            dva = mapping.device_ptr + (host_ptr - reg_ptr)
        if not dva:
            raise RuntimeError("MF host DVA must be non-zero")
        out.append(int(dva))
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
    """H2D one layer, automatically resolving registered MF host DVAs."""
    torch.ops.npu.transfer_kv_per_layer_direct_pf_lf(
        list(src_ptrs),
        list(dst_ptrs),
        src_indices,
        dst_indices,
        layer_id,
        page_size,
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
    """D2H all layers, automatically resolving registered MF host DVAs."""
    torch.ops.npu.transfer_kv_all_layer_direct_lf_pf(
        list(src_ptrs),
        list(dst_ptrs),
        src_indices,
        dst_indices,
        page_size,
        _resolve_host_dvas(dst_ptrs, host_mappings),
    )
