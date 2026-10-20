import os
import unittest

import sgl_kernel_npu  # noqa: F401
import torch
from sgl_kernel_npu.hicache_offload import (
    create_hugepage_tensor,
    free_hugepage,
    free_hugepage_all,
    hugepage_mode,
)


class TestHugepageHostAllocator(unittest.TestCase):
    def test_roundtrip_and_registration(self):
        tensor, host_ptr, dev_ptr = create_hugepage_tensor((16, 64), torch.bfloat16)
        self.assertNotEqual(host_ptr, 0)
        self.assertNotEqual(dev_ptr, 0)
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(tuple(tensor.shape), (16, 64))

        # Host-side write/readback through the torch view.
        tensor.fill_(3.5)
        self.assertAlmostEqual(tensor.float().mean().item(), 3.5, places=5)

        free_hugepage(host_ptr)

    def test_multiple_allocations_are_distinct(self):
        t1, p1, d1 = create_hugepage_tensor((1024,), torch.float32)
        t2, p2, d2 = create_hugepage_tensor((1024,), torch.float32)
        self.assertNotEqual(p1, p2)
        self.assertNotEqual(d1, d2)
        t1.fill_(1.0)
        t2.fill_(2.0)
        self.assertAlmostEqual(t1.mean().item(), 1.0, places=5)
        self.assertAlmostEqual(t2.mean().item(), 2.0, places=5)
        free_hugepage_all()

    def test_mode_switch_off_uses_4kb(self):
        # A/B switch: mode=off must back the allocation with regular 4KB
        # pages, reported via the third tuple element.
        old = os.environ.get("SGLANG_HICACHE_HUGEPAGE_MODE")
        os.environ["SGLANG_HICACHE_HUGEPAGE_MODE"] = "off"
        try:
            self.assertEqual(hugepage_mode(), "off")
            host_ptr, dev_ptr, page_size = torch.ops.npu.hugepage_host_alloc(1 << 20, 0)
            self.assertEqual(page_size, 4096)
            free_hugepage(host_ptr)

            # Strict 1g mode must fail loudly when 1GB hugepages are absent
            # (this test never asserts which side it lands on, only that the
            # mode is honored: auto falls back, 1g either succeeds with 1GB
            # or raises).
            os.environ["SGLANG_HICACHE_HUGEPAGE_MODE"] = "1g"
            try:
                host_ptr, dev_ptr, page_size = torch.ops.npu.hugepage_host_alloc(1 << 20, 0)
                self.assertEqual(page_size, 1 << 30)
                free_hugepage(host_ptr)
            except RuntimeError:
                pass  # 1GB hugepages not configured on this machine
        finally:
            if old is None:
                os.environ.pop("SGLANG_HICACHE_HUGEPAGE_MODE", None)
            else:
                os.environ["SGLANG_HICACHE_HUGEPAGE_MODE"] = old
            free_hugepage_all()

    def test_invalid_args(self):
        with self.assertRaisesRegex(RuntimeError, "num_bytes must be positive"):
            torch.ops.npu.hugepage_host_alloc(0, 0)
        with self.assertRaisesRegex(RuntimeError, "device_id must be non-negative"):
            torch.ops.npu.hugepage_host_alloc(64, -1)
        with self.assertRaises(RuntimeError):
            torch.ops.npu.hugepage_host_free(0, 0)


if __name__ == "__main__":
    unittest.main()
