from enum import Enum
from typing import List, Optional, Sequence

import torch


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


def aclrt_host_register(tensor: torch.Tensor) -> int:
    """
    Register host (CPU) memory with aclrtHostRegister(ACL_HOST_REGISTER_MAPPED)
    and return the device-visible address (DVA) an AIV kernel must use to
    dereference it. ``tensor`` must be kept alive until aclrt_host_unregister
    is called on it.
    """
    return torch.ops.npu.aclrt_host_register(tensor)


def aclrt_host_unregister(tensor: torch.Tensor) -> None:
    """Undo aclrt_host_register for memory registered via aclrt_host_register."""
    torch.ops.npu.aclrt_host_unregister(tensor)


def get_device_accessible_ptr(tensor: torch.Tensor, device_index: int) -> int:
    """Return the address a kernel on ``device_index`` must use for ``tensor``.

    Mirrors sgl_kernel.kvcacheio.get_device_accessible_ptr (CUDA): registered
    host memory resolves to its DVA, everything else keeps its raw address.
    """
    return torch.ops.npu.aclrt_get_device_accessible_ptr(tensor, device_index)


def set_host_dva_mode(mode: str) -> None:
    """Select how the kv_exchange transfer ops resolve host pool pointers.

    "aclrt" (default) resolves through the aclrtHostRegister table populated
    by aclrt_host_register; "passthrough" hands host pointers to the kernel
    unchanged (they must already be device-accessible). This is the swap point
    for a future memfabric DVA provider — transfer ops and the AIV kernel are
    provider-agnostic.
    """
    torch.ops.npu.aclrt_set_host_dva_mode(mode)


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
):
    """
    H2D, one layer: registered host pool (page-first, 5-D) -> device pool
    (layer-first, 5-D). src_ptrs/dst_ptrs are matching component lists
    (k/v/index_k/index_k_scale); host-side tensors must be registered via
    aclrt_host_register. Signature mirrors sgl_kernel (CUDA).
    """
    torch.ops.npu.transfer_kv_per_layer_direct_pf_lf(
        list(src_ptrs), list(dst_ptrs), src_indices, dst_indices, layer_id, page_size
    )


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    """
    D2H, all layers: device pool (layer-first, 5-D) -> registered host pool
    (page-first, 5-D). Per-component layer ranges follow the device buffer's
    layer count, so components covering fewer layers (indexer k / scales) need
    no extra mapping on the caller side. Signature mirrors sgl_kernel (CUDA).
    """
    torch.ops.npu.transfer_kv_all_layer_direct_lf_pf(
        list(src_ptrs), list(dst_ptrs), src_indices, dst_indices, page_size
    )
