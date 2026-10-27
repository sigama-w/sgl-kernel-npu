"""MemFabric host-memory allocation and device-address registration."""

from typing import NamedTuple

import torch


class HostMemoryMapping(NamedTuple):
    """Device-visible mapping of one registered MemFabric host block."""

    device_ptr: int
    registered_host_ptr: int
    owned: bool


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


def register_host_memory(host_ptr: int, nbytes: int) -> HostMemoryMapping:
    """Register an MF-allocated block and return its kernel-visible DVA.

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


def uninitialize_host_memory() -> None:
    """Release all allocations and registrations owned by this MF entry."""
    _load_offload().uninitialize()
