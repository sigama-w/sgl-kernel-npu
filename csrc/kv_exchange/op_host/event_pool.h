/*!
 * \file event_pool.h
 * \brief Process-wide aclrtEvent pool for fused per-layer H2D submissions
 *
 * The fused range submission (kv_exchange_submit_range) records one event per
 * controller layer inside the C++ call, so the Python side hands out pool
 * slots instead of torch.npu.Event objects: the forward stream waits on a
 * slot (kv_exchange_event_pool_wait) and the producer-reuse assertion polls
 * a slot (kv_exchange_event_pool_query). Pools live for the process
 * lifetime; slots are addressed by (pool_id, index).
 */

#ifndef SGL_KERNEL_NPU_KV_EXCHANGE_EVENT_POOL_H_
#define SGL_KERNEL_NPU_KV_EXCHANGE_EVENT_POOL_H_

#include <cstdint>

#include "defines.h"

namespace sglang {
namespace npu_kernel {

// Create `count` aclrtEvents and return their pool id. Events are created
// in the caller's current NPU context.
HOST_API int64_t kv_exchange_event_pool_create(int64_t count);

// Record pool slot `index` on the caller's current NPU stream.
HOST_API void kv_exchange_event_pool_record(int64_t pool_id, int64_t index);

// Make the caller's current NPU stream wait on pool slot `index`.
HOST_API void kv_exchange_event_pool_wait(int64_t pool_id, int64_t index);

// Query pool slot `index` completion without blocking.
HOST_API bool kv_exchange_event_pool_query(int64_t pool_id, int64_t index);

// Destroy one pool (never called in production; pools are process-lifetime).
HOST_API void kv_exchange_event_pool_destroy(int64_t pool_id);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_KV_EXCHANGE_EVENT_POOL_H_
