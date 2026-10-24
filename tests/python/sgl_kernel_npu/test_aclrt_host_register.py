import unittest

import torch

from sgl_kernel_npu.kvcacheio import (
    aclrt_host_register,
    aclrt_host_unregister,
    get_device_accessible_ptr,
    set_host_dva_mode,
)

REG_BYTES = 4 * 1024 * 1024


def _aligned_pinned(numel: int, dtype=torch.bfloat16) -> torch.Tensor:
    buf = torch.zeros(numel, dtype=dtype).pin_memory()
    assert buf.data_ptr() % 4096 == 0, f"pinned buffer not 4K aligned: {hex(buf.data_ptr())}"
    return buf


class TestAclrtHostRegister(unittest.TestCase):

    def test_register_returns_dva_and_resolves(self):
        torch.npu.set_device(0)
        buf = _aligned_pinned(REG_BYTES // 2)
        hva = buf.data_ptr()
        dva = aclrt_host_register(buf)
        self.assertIsInstance(dva, int)
        self.assertNotEqual(dva, 0)
        self.assertNotEqual(dva, hva)
        # Full-tensor lookup resolves to the DVA.
        self.assertEqual(get_device_accessible_ptr(buf, 0), dva)
        # A view inside the registered range resolves into the same range.
        off = 4096 // 2  # elements (bf16), keep 4K alignment
        view = buf[off:off + 16]
        self.assertEqual(get_device_accessible_ptr(view, 0), dva + off * view.element_size())
        aclrt_host_unregister(buf)
        # After unregistration the raw address is returned again.
        self.assertEqual(get_device_accessible_ptr(buf, 0), buf.data_ptr())

    def test_unregistered_cpu_returns_hva(self):
        torch.npu.set_device(0)
        buf = _aligned_pinned(1024)
        self.assertEqual(get_device_accessible_ptr(buf, 0), buf.data_ptr())

    def test_device_tensor_returns_data_ptr(self):
        torch.npu.set_device(0)
        dev = torch.zeros(64, dtype=torch.float32, device="npu")
        self.assertEqual(get_device_accessible_ptr(dev, 0), dev.data_ptr())

    def test_multiple_ranges(self):
        torch.npu.set_device(0)
        a = _aligned_pinned(REG_BYTES // 2)
        b = _aligned_pinned(REG_BYTES // 2)
        dva_a = aclrt_host_register(a)
        dva_b = aclrt_host_register(b)
        self.assertNotEqual(dva_a, dva_b)
        self.assertEqual(get_device_accessible_ptr(a, 0), dva_a)
        self.assertEqual(get_device_accessible_ptr(b, 0), dva_b)
        aclrt_host_unregister(a)
        self.assertEqual(get_device_accessible_ptr(a, 0), a.data_ptr())
        self.assertEqual(get_device_accessible_ptr(b, 0), dva_b)
        aclrt_host_unregister(b)

    def test_double_register_and_unregistered_free_raise(self):
        torch.npu.set_device(0)
        buf = _aligned_pinned(1024 * 1024)
        aclrt_host_register(buf)
        with self.assertRaises(RuntimeError):
            aclrt_host_register(buf)
        aclrt_host_unregister(buf)
        with self.assertRaises(RuntimeError):
            aclrt_host_unregister(buf)
        # The range is gone: re-registering must succeed.
        dva = aclrt_host_register(buf)
        self.assertNotEqual(dva, 0)
        aclrt_host_unregister(buf)

    def test_non_cpu_rejected(self):
        torch.npu.set_device(0)
        dev = torch.zeros(16, dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            aclrt_host_register(dev)

    def test_host_dva_mode_switch(self):
        # The provider mode is the seam for swapping in a future memfabric
        # resolver: "aclrt" (default) consults the registry, "passthrough"
        # bypasses it for already-device-accessible host pointers.
        torch.npu.set_device(0)
        buf = _aligned_pinned(REG_BYTES // 2)
        dva = aclrt_host_register(buf)
        self.assertEqual(get_device_accessible_ptr(buf, 0), dva)
        with self.assertRaises(RuntimeError):
            set_host_dva_mode("invalid-mode")
        set_host_dva_mode("passthrough")
        try:
            # The registry is bypassed: the raw host address comes back even
            # for a registered range.
            self.assertEqual(get_device_accessible_ptr(buf, 0), buf.data_ptr())
        finally:
            set_host_dva_mode("aclrt")
        self.assertEqual(get_device_accessible_ptr(buf, 0), dva)
        aclrt_host_unregister(buf)


if __name__ == "__main__":
    unittest.main()
