// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_host/event_pool.h"

#include <mutex>
#include <unordered_map>
#include <vector>

#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

namespace {

std::mutex g_event_pool_mutex;
// pool id -> (slot index -> aclrtEvent)
std::unordered_map<int64_t, std::vector<aclrtEvent>> g_event_pools;
int64_t g_next_event_pool_id = 0;

aclrtEvent LookupEvent(int64_t pool_id, int64_t index)
{
    const std::lock_guard<std::mutex> lock(g_event_pool_mutex);
    const auto pool = g_event_pools.find(pool_id);
    TORCH_CHECK(pool != g_event_pools.end(), "unknown kv_exchange event pool id ", pool_id);
    TORCH_CHECK(index >= 0 && index < static_cast<int64_t>(pool->second.size()),
                "kv_exchange event pool ", pool_id, " index ", index, " out of range");
    return pool->second[static_cast<size_t>(index)];
}

}  // namespace

int64_t kv_exchange_event_pool_create(int64_t count)
{
    TORCH_CHECK(count > 0, "kv_exchange event pool count must be positive, got ", count);
    std::vector<aclrtEvent> events(static_cast<size_t>(count), nullptr);
    for (auto &event : events) {
        const aclError status = aclrtCreateEvent(&event);
        TORCH_CHECK(status == ACL_SUCCESS && event != nullptr,
                    "aclrtCreateEvent failed with status ", static_cast<int64_t>(status));
    }
    const std::lock_guard<std::mutex> lock(g_event_pool_mutex);
    const int64_t pool_id = g_next_event_pool_id++;
    g_event_pools.emplace(pool_id, std::move(events));
    return pool_id;
}

void kv_exchange_event_pool_record(int64_t pool_id, int64_t index)
{
    const aclrtEvent event = LookupEvent(pool_id, index);
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const aclError status = aclrtRecordEvent(event, stream);
    TORCH_CHECK(status == ACL_SUCCESS, "aclrtRecordEvent failed with status ", static_cast<int64_t>(status));
}

void kv_exchange_event_pool_wait(int64_t pool_id, int64_t index)
{
    const aclrtEvent event = LookupEvent(pool_id, index);
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const aclError status = aclrtStreamWaitEvent(stream, event);
    TORCH_CHECK(status == ACL_SUCCESS, "aclrtStreamWaitEvent failed with status ", static_cast<int64_t>(status));
}

bool kv_exchange_event_pool_query(int64_t pool_id, int64_t index)
{
    const aclrtEvent event = LookupEvent(pool_id, index);
    aclrtBool done = ACL_FALSE;
    const aclError status = aclrtEventQuery(event, &done);
    TORCH_CHECK(status == ACL_SUCCESS, "aclrtEventQuery failed with status ", static_cast<int64_t>(status));
    return done == ACL_TRUE;
}

void kv_exchange_event_pool_destroy(int64_t pool_id)
{
    std::vector<aclrtEvent> events;
    {
        const std::lock_guard<std::mutex> lock(g_event_pool_mutex);
        const auto pool = g_event_pools.find(pool_id);
        if (pool == g_event_pools.end()) {
            return;
        }
        events = std::move(pool->second);
        g_event_pools.erase(pool);
    }
    for (auto event : events) {
        if (event != nullptr) {
            const aclError status = aclrtDestroyEvent(event);
            TORCH_CHECK(status == ACL_SUCCESS, "aclrtDestroyEvent failed with status ",
                        static_cast<int64_t>(status));
        }
    }
}

}  // namespace npu_kernel
}  // namespace sglang
