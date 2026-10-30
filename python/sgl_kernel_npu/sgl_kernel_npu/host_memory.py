"""MemFabric host-memory allocation and device-address registration."""

from typing import List, NamedTuple, Optional, Tuple

import torch


class HostMemoryMapping(NamedTuple):
    """Device-visible mapping of one registered MemFabric host block."""

    device_ptr: int
    registered_host_ptr: int
    owned: bool


# Registered MF blocks: (base_hva, nbytes, HostMemoryMapping). The table also
# resolves interior tensor views to DVA + byte offset.
_G_MAPPINGS: List[Tuple[int, int, HostMemoryMapping]] = []


def _lookup_mapping(host_ptr: int) -> Tuple[Optional[HostMemoryMapping], int]:
    if host_ptr:
        for reg_ptr, nbytes, mapping in reversed(_G_MAPPINGS):
            if reg_ptr <= host_ptr < reg_ptr + nbytes:
                return mapping, reg_ptr
    return None, 0


def _load_offload():
    from memfabric_hybrid import offload

    return offload


def initialize_host_memory(device_id: int, reserve_size: int) -> None:
    """Initialize MemFabric's LOCAL host-memory entry for one NPU device."""
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id}")
    if reserve_size <= 0:
        raise ValueError(f"reserve_size must be positive, got {reserve_size}")

    offload = _load_offload()
    config = offload.OffloadConfig()
    config.device_id = device_id
    config.reserve_size = reserve_size
    config.scene = offload.Scene.LOCAL
    config.flags = offload.OFFLOAD_FLAG_HOST_MEM
    ret = offload.initialize(config)
    if ret != 0:
        raise RuntimeError(
            f"MF offload.initialize(OFFLOAD_FLAG_HOST_MEM) failed: ret={ret}, "
            f"device_id={device_id}, reserve_size={reserve_size}"
        )


def empty_host_memory(shape, dtype=torch.bfloat16) -> torch.Tensor:
    """Allocate a CPU tensor from MemFabric's host-memory entry."""
    return _load_offload().empty(tuple(shape), dtype=dtype)


def _register_host_block(host_ptr: int, nbytes: int) -> HostMemoryMapping:
    """Register one MF-allocated block and return its kernel-visible DVA.

    MemFabric only accepts the base pointer returned by offload.malloc/empty.
    Registered blocks remain alive until uninitialize_host_memory().
    """
    if host_ptr <= 0 or nbytes <= 0:
        raise ValueError(
            f"host_ptr and nbytes must be positive, got {host_ptr}, {nbytes}"
        )

    offload = _load_offload()
    ret = offload.host_register(host_ptr)
    if ret != 0:
        raise RuntimeError(
            f"MF host_register failed for {hex(host_ptr)} "
            f"({nbytes} bytes): ret={ret}; allocate the tensor with "
            "empty_host_memory before registering it"
        )
    device_ptr = offload.host_get_device_pointer(host_ptr)
    if not device_ptr:
        raise RuntimeError(
            f"MF host_get_device_pointer failed for registered host block {hex(host_ptr)}"
        )
    return HostMemoryMapping(
        device_ptr=int(device_ptr), registered_host_ptr=int(host_ptr), owned=True
    )


def register_host_memory(tensor: torch.Tensor) -> HostMemoryMapping:
    """Register an MF-allocated CPU tensor and record its DVA mapping."""
    if not tensor.is_cpu:
        raise RuntimeError("register_host_memory: tensor must be host (CPU) memory")
    host_ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes <= 0:
        raise RuntimeError("register_host_memory: tensor must be non-empty")

    mapping = _register_host_block(host_ptr, nbytes)
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


def resolve_host_dva(tensor: torch.Tensor) -> int:
    """Resolve a registered MF host tensor (or interior view) to its DVA.

    Unlike get_device_accessible_ptr, unregistered CPU memory is an error:
    AIV kv_exchange kernels can only de-reference host VAs mapped through the
    MemFabric offload entity, so there is no fallback.
    """
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
    return int(dva)


def uninitialize_host_memory() -> None:
    """Release all MF blocks and invalidate every Python-side DVA mapping."""
    try:
        _load_offload().uninitialize()
    finally:
        _G_MAPPINGS.clear()
