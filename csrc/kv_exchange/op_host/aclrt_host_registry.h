/*
 * Process-level registry of aclrtHostRegister-ed host ranges.
 *
 * The aclrtHostRegister(4-arg, ACL_HOST_REGISTER_MAPPED) call itself lives in
 * python (host_memory.py, PR #39548's `acl.rt` binding path); the
 * aclrt_record_host_mapping op publishes each mapping {hva, size, dva} here.
 * Transfer ops call ResolveHostDva() to translate a registered host tensor
 * (e.g. an L2 KV cache pool) into the device-visible address the AIV kernel
 * dereferences.
 *
 * There is deliberately no HAL fallback: for host pointers never published
 * here this fails loudly instead of silently taking a slow path.
 */

#ifndef SGL_KERNEL_NPU_KV_EXCHANGE_ACLRT_HOST_REGISTRY_H
#define SGL_KERNEL_NPU_KV_EXCHANGE_ACLRT_HOST_REGISTRY_H

#include <cstdint>
#include <string>

namespace sglang {
namespace npu_kernel {

// Translate a registered host address range into its device-visible address.
// Throws c10::Error when [host_ptr, host_ptr + bytes) is not fully contained
// in a single registered range.
int64_t ResolveHostDva(const void *host_ptr, uint64_t bytes);

// Lenient lookup used by get_device_accessible_ptr: returns the DVA of the
// registered range containing `host_ptr` plus the intra-range offset, or 0
// when the pointer is not registered (caller falls back to the raw address).
int64_t LookupHostDva(const void *host_ptr);

// The host-address provider is selectable via aclrt_set_host_dva_mode()
// (declared in sgl_kenel_npu_ops.h):
//   "aclrt"       (default) ResolveHostDva/LookupHostDva resolve through the
//                 process aclrtHostRegister table populated by
//                 aclrt_record_host_mapping()
//   "passthrough" host pointers are already device-accessible; the registry
//                 is bypassed
// This is the seam where a future provider (e.g. a memfabric DVA resolver)
// plugs in: transfer ops and the AIV kernel never see the backend.

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_KV_EXCHANGE_ACLRT_HOST_REGISTRY_H
