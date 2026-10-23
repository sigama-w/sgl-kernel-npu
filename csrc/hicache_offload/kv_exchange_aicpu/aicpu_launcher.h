/*
 * Host-side launcher for the kv_exchange AICPU device kernel.
 *
 * Loads the kernel via aclrtBinaryLoadFromFile(json) + aclrtBinaryGetFunction
 * (the same raw-handle path memfabric_hybrid uses, bypassing the op_proto /
 * op_tiling framework), then launches it on the current NPU stream with
 * aclrtLaunchKernelWithConfig.
 *
 * Also exposes aclrtHostRegister / aclrtHostUnregister wrappers so callers
 * can obtain the device-visible address (DVA) of pinned host KV buffers.
 */

#ifndef SGL_KERNEL_NPU_HICACHE_OFFLOAD_KV_EXCHANGE_AICPU_LAUNCHER_H
#define SGL_KERNEL_NPU_HICACHE_OFFLOAD_KV_EXCHANGE_AICPU_LAUNCHER_H

#include <ATen/ATen.h>

namespace sglang {
namespace npu_kernel {

// Launch the KvExchangeAicpu device kernel.
// meta: CPU int64 tensor with exactly 44 slots; slot contents are defined in
// csrc/hicache_offload/kv_exchange_aicpu/op_kernel/kv_exchange_aicpu_kernel.cc.
// All addresses inside meta (device bases, registered host bases/DVAs, index
// pointers) must already be device-visible addresses.
void kv_exchange_aicpu(const at::Tensor &meta);

// Register host memory with aclrtHostRegister and return the mapped DVA.
// buf must be a CPU (ideally pinned, page-aligned) tensor kept alive until
// aclrt_host_unregister is called on it.
int64_t aclrt_host_register(const at::Tensor &buf);

// Undo aclrt_host_register for memory that was registered via
// aclrt_host_register(buf).
void aclrt_host_unregister(const at::Tensor &buf);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_HICACHE_OFFLOAD_KV_EXCHANGE_AICPU_LAUNCHER_H
