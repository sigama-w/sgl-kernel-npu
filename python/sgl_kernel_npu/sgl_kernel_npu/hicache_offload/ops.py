"""Python wrappers for the HiCache offload operators (kv_exchange and the
hugepage host allocator), ported from MemFabric acc_offload."""

import ctypes
import logging
import os
from math import prod
from typing import List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# meta layout (int64 values, see kv_exchange_kernel.cpp):
#   header: 8 values, per component: 9 values, at most 4 components.
#   header[7] = start_dim override: cores [start_dim, block_dim) participate;
#   0 keeps the MemFabric default (62 -> 2 active AIV cores on Ascend950).
KV_EXCHANGE_META_HEADER = 8
KV_EXCHANGE_META_STRIDE = 9
KV_EXCHANGE_MAX_COMPONENTS = 4
KV_EXCHANGE_META_NUMEL = KV_EXCHANGE_META_HEADER + KV_EXCHANGE_META_STRIDE * KV_EXCHANGE_MAX_COMPONENTS

KV_EXCHANGE_DIRECTION_H2D = 1
KV_EXCHANGE_DIRECTION_D2H = 2

HOST_LAYOUT_PAGE_FIRST = 0
HOST_LAYOUT_LAYER_FIRST = 1

# Number of AIV cores on Ascend950; only cores >= 62 participate in the
# kernel (KV_EXCHANGE_START_DIM).
KV_EXCHANGE_BLOCK_DIM_A5 = 64


def _ctype_for_dtype(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return ctypes.c_uint16
    if dtype == torch.float32:
        return ctypes.c_float
    if dtype == torch.float64:
        return ctypes.c_double
    if dtype == torch.int8:
        return ctypes.c_int8
    if dtype == torch.uint8:
        return ctypes.c_uint8
    if dtype == torch.int16:
        return ctypes.c_int16
    if dtype == torch.int32:
        return ctypes.c_int32
    if dtype == torch.int64:
        return ctypes.c_int64
    if dtype == torch.bool:
        return ctypes.c_bool
    raise TypeError(f"unsupported hugepage tensor dtype: {dtype}")


def hugepage_mode() -> str:
    """Return the active SGLANG_HICACHE_HUGEPAGE_MODE (auto/1g/2m/off)."""
    return os.environ.get("SGLANG_HICACHE_HUGEPAGE_MODE", "").lower() or "auto"


def create_hugepage_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    device_id: int = 0,
) -> Tuple[torch.Tensor, int, int]:
    """Allocate hugepage-backed host memory and register it to the NPU device.

    Returns ``(host_tensor, host_ptr, dev_ptr)``. ``host_tensor`` is a CPU
    tensor backed by the registered hugepage allocation. ``dev_ptr`` is the
    device-visible address (DVA) that the kv_exchange kernel dereferences
    through the per-component ``host_base`` field.

    The backing page size actually used (1GB/2MB/4KB, per
    SGLANG_HICACHE_HUGEPAGE_MODE) is logged once per distinct value.
    """
    shape_tuple = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape_tuple):
        raise ValueError(f"shape dimensions must be non-negative, got {shape_tuple}")

    numel = int(prod(shape_tuple))
    elem_size = torch.empty((), dtype=dtype).element_size()
    size = numel * elem_size
    if size <= 0:
        raise ValueError(f"hugepage tensor size must be positive, got shape={shape_tuple}")

    host_ptr, dev_ptr, page_size = torch.ops.npu.hugepage_host_alloc(size, device_id)
    logger.info(
        "hugepage_host_alloc: %d bytes on device %d, mode=%s, backing page=%d",
        size,
        device_id,
        hugepage_mode(),
        page_size,
    )
    buffer_type = _ctype_for_dtype(dtype) * numel
    buffer = buffer_type.from_address(host_ptr)
    tensor = torch.frombuffer(buffer, dtype=dtype).view(shape_tuple)
    if tensor.element_size() != elem_size:
        raise RuntimeError("hugepage tensor element size mismatch")
    return tensor, int(host_ptr), int(dev_ptr)


def free_hugepage(host_ptr: int, device_id: int = 0) -> None:
    """Unregister and release one hugepage allocation by host pointer."""
    torch.ops.npu.hugepage_host_free(int(host_ptr), device_id)


def free_hugepage_all(device_id: int = 0) -> None:
    """Unregister and release every hugepage allocation of this process."""
    torch.ops.npu.hugepage_host_free_all(device_id)


def build_kv_exchange_component(
    device_base: int,
    host_base: int,
    device_layer_pitch: int,
    device_page_pitch: int,
    host_page_pitch: int,
    host_layer_pitch: int,
    width: int,
    layer_lo: int,
    layer_hi: int,
) -> List[int]:
    """Pack one component's 9 meta values (k/v/index_k/index_k_scale order)."""
    return [
        int(device_base),
        int(host_base),
        int(device_layer_pitch),
        int(device_page_pitch),
        int(host_page_pitch),
        int(host_layer_pitch),
        int(width),
        int(layer_lo),
        int(layer_hi),
    ]


def build_kv_exchange_meta(
    num_pages: int,
    page_size: int,
    direction: int,
    device_indices_ptr: int,
    host_indices_ptr: int,
    host_layout_mode: int,
    components: Sequence[Sequence[int]],
    start_dim: Optional[int] = None,
) -> List[int]:
    """Assemble the full 44-value meta vector consumed by ``kv_exchange``.

    ``components`` is a sequence of :func:`build_kv_exchange_component`
    outputs; at most 4 are accepted. Padding values are appended so the meta
    tensor always has ``KV_EXCHANGE_META_NUMEL`` elements.

    ``start_dim`` overrides which AIV cores participate (cores
    ``[start_dim, block_dim)``; active cores = ``block_dim - start_dim``).
    ``None`` reads ``SGLANG_KV_EXCHANGE_START_DIM``; the effective 0 keeps the
    MemFabric default of 62 (2 active cores on Ascend950).
    """
    if not 0 < len(components) <= KV_EXCHANGE_MAX_COMPONENTS:
        raise ValueError(f"expected 1..{KV_EXCHANGE_MAX_COMPONENTS} components, got {len(components)}")
    if direction not in (KV_EXCHANGE_DIRECTION_H2D, KV_EXCHANGE_DIRECTION_D2H):
        raise ValueError(f"invalid direction: {direction}")
    if host_layout_mode not in (HOST_LAYOUT_PAGE_FIRST, HOST_LAYOUT_LAYER_FIRST):
        raise ValueError(f"invalid host_layout_mode: {host_layout_mode}")
    for comp in components:
        if len(comp) != KV_EXCHANGE_META_STRIDE:
            raise ValueError(f"each component must have {KV_EXCHANGE_META_STRIDE} values, got {len(comp)}")

    if start_dim is None:
        start_dim = int(os.environ.get("SGLANG_KV_EXCHANGE_START_DIM", "0") or 0)

    meta = [
        len(components),
        int(num_pages),
        int(page_size),
        int(direction),
        int(device_indices_ptr),
        int(host_indices_ptr),
        int(host_layout_mode),
        int(start_dim),
    ]
    for comp in components:
        meta.extend(int(v) for v in comp)
    meta.extend([0] * (KV_EXCHANGE_META_NUMEL - len(meta)))
    return meta


def kv_exchange(meta: torch.Tensor, block_dim: int = KV_EXCHANGE_BLOCK_DIM_A5) -> None:
    """Launch the AIV kv_exchange kernel with a device-resident meta tensor.

    ``meta`` must be a contiguous int64 NPU tensor of at least
    ``KV_EXCHANGE_META_NUMEL`` elements (see ``build_kv_exchange_meta`` for
    the layout). All buffers referenced through ``meta`` must stay alive
    until work on the current NPU stream completes.
    """
    torch.ops.npu.kv_exchange(meta, block_dim)
