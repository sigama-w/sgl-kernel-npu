import unittest

import torch

from sgl_kernel_npu.kvcacheio import (
    HostMemoryMapping,
    get_device_accessible_ptr,
    register_host_memory,
    set_host_dva_mode,
    unregister_host_memory,
)

REG_BYTES = 4 * 1024 * 1024


def _aligned_pinned(numel: int, dtype=torch.bfloat16) -> torch.Tensor:
    buf = torch.zeros(numel, dtype=dtype).pin_memory()
    assert buf.data_ptr() % 4096 == 0, f"pinned buffer not 4K aligned: {hex(buf.data_ptr())}"
    return buf


class TestAclrtHostRegister(unittest.TestCase):

    def test_register_returns_mapping_and_resolves(self):
        torch.npu.set_device(0)
        buf = _aligned_pinned(REG_BYTES // 2)
        hva = buf.data_ptr()
        mapping = register_host_memory(buf)
        self.assertIsInstance(mapping, HostMemoryMapping)
        self.assertTrue(mapping.owned)
        self.assertIsInstance(mapping.device_ptr, int)
        self.assertNotEqual(mapping.device_ptr, 0)
        self.assertNotEqual(mapping.device_ptr, hva)
        # The registered base is page-aligned and at or below the tensor.
        self.assertLessEqual(mapping.registered_host_ptr, hva)
        self.assertLessEqual(hva - mapping.registered_host_ptr, 4096)
        # Full-tensor lookup resolves to the mapping's device_ptr.
        self.assertEqual(get_device_accessible_ptr(buf, 0), mapping.device_ptr)
        # A view inside the registered range resolves into the same range.
        off = 4096 // 2  # elements (bf16), keep 4K alignment
        view = buf[off:off + 16]
        self.assertEqual(
            get_device_accessible_ptr(view, 0),
            mapping.device_ptr + off * view.element_size(),
        )
        unregister_host_memory(mapping)
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
        mapping_a = register_host_memory(a)
        mapping_b = register_host_memory(b)
        self.assertNotEqual(mapping_a.device_ptr, mapping_b.device_ptr)
        self.assertEqual(get_device_accessible_ptr(a, 0), mapping_a.device_ptr)
        self.assertEqual(get_device_accessible_ptr(b, 0), mapping_b.device_ptr)
        unregister_host_memory(mapping_a)
        self.assertEqual(get_device_accessible_ptr(a, 0), a.data_ptr())
        self.assertEqual(get_device_accessible_ptr(b, 0), mapping_b.device_ptr)
        unregister_host_memory(mapping_b)

    def test_duplicate_register_borrows_and_owned_release(self):
        # PR #39548 semantics: re-registering an already-mapped range borrows
        # the existing mapping (owned=False) instead of failing, and only the
        # owner releases the runtime registration.
        torch.npu.set_device(0)
        buf = _aligned_pinned(1024 * 1024)
        owner = register_host_memory(buf)
        self.assertTrue(owner.owned)
        borrower = register_host_memory(buf)
        self.assertFalse(borrower.owned)
        self.assertEqual(borrower.device_ptr, owner.device_ptr)
        self.assertEqual(borrower.registered_host_ptr, owner.registered_host_ptr)
        # Borrower release is a no-op: the mapping stays live.
        unregister_host_memory(borrower)
        self.assertEqual(get_device_accessible_ptr(buf, 0), owner.device_ptr)
        # Owner release frees it; a stale borrower release afterwards is a
        # silent no-op too (owned=False).
        unregister_host_memory(owner)
        self.assertEqual(get_device_accessible_ptr(buf, 0), buf.data_ptr())
        unregister_host_memory(borrower)
        # The range is gone: re-registering must succeed with a fresh mapping.
        again = register_host_memory(buf)
        self.assertTrue(again.owned)
        self.assertNotEqual(again.device_ptr, 0)
        unregister_host_memory(again)

    def test_unregistered_owned_free_raises(self):
        torch.npu.set_device(0)
        buf = _aligned_pinned(1024)
        stale = HostMemoryMapping(buf.data_ptr(), buf.data_ptr(), True)
        with self.assertRaises(RuntimeError):
            unregister_host_memory(stale)

    def test_non_cpu_rejected(self):
        torch.npu.set_device(0)
        dev = torch.zeros(16, dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            register_host_memory(dev)

    def test_host_dva_mode_switch(self):
        # The provider mode is the seam for swapping in a future memfabric
        # resolver: "aclrt" (default) consults the registry, "passthrough"
        # bypasses it for already-device-accessible host pointers.
        torch.npu.set_device(0)
        buf = _aligned_pinned(REG_BYTES // 2)
        mapping = register_host_memory(buf)
        self.assertEqual(get_device_accessible_ptr(buf, 0), mapping.device_ptr)
        with self.assertRaises(RuntimeError):
            set_host_dva_mode("invalid-mode")
        set_host_dva_mode("passthrough")
        try:
            # The registry is bypassed: the raw host address comes back even
            # for a registered range.
            self.assertEqual(get_device_accessible_ptr(buf, 0), buf.data_ptr())
        finally:
            set_host_dva_mode("aclrt")
        self.assertEqual(get_device_accessible_ptr(buf, 0), mapping.device_ptr)
        unregister_host_memory(mapping)


if __name__ == "__main__":
    unittest.main()
