// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "aclrt_host_registry.h"

#include "defines.h"
#include "torch_helper.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace sglang {
namespace npu_kernel {

namespace {

// Host-address provider mode (see SetHostDvaMode in aclrt_host_registry.h).
// false = "aclrt" (default, registry lookup); true = "passthrough".
std::atomic<bool> g_passthrough{false};

struct HostRange {
    uintptr_t hva = 0;  // page-aligned host base passed to aclrtHostRegister
    uint64_t size = 0;  // page-aligned span actually registered
    uintptr_t dva = 0;  // device-visible address of the registered base
};

std::mutex g_ranges_mu;
std::vector<HostRange> g_ranges;  // sorted by hva, non-overlapping

std::vector<HostRange>::iterator FindByHva(uintptr_t hva)
{
    for (auto it = g_ranges.begin(); it != g_ranges.end(); ++it) {
        if (it->hva == hva) {
            return it;
        }
    }
    return g_ranges.end();
}

}  // namespace

int64_t LookupHostDva(const void *host_ptr)
{
    if (g_passthrough.load(std::memory_order_relaxed)) {
        return 0;  // caller falls back to the raw (already accessible) address
    }
    const uintptr_t ptr = reinterpret_cast<uintptr_t>(host_ptr);
    std::lock_guard<std::mutex> lock(g_ranges_mu);
    auto it = std::upper_bound(g_ranges.begin(), g_ranges.end(), ptr,
                               [](uintptr_t value, const HostRange &range) { return value < range.hva; });
    if (it == g_ranges.begin()) {
        return 0;
    }
    --it;
    if (ptr < it->hva || ptr >= it->hva + it->size) {
        return 0;
    }
    return static_cast<int64_t>(it->dva + (ptr - it->hva));
}

int64_t ResolveHostDva(const void *host_ptr, uint64_t bytes)
{
    if (g_passthrough.load(std::memory_order_relaxed)) {
        // The active provider guarantees host pointers are already
        // device-dereferenceable; hand them to the kernel unchanged.
        return static_cast<int64_t>(reinterpret_cast<uintptr_t>(host_ptr));
    }
    const int64_t dva = LookupHostDva(host_ptr);
    TORCH_CHECK(dva != 0, "host address ", host_ptr, " is not registered with aclrtHostRegister; "
               "allocate host pools via sglang pool_host alloc_with_host_register (NPU) so the "
               "AIV kv_exchange kernel can dereference them");
    const uintptr_t ptr = reinterpret_cast<uintptr_t>(host_ptr);
    std::lock_guard<std::mutex> lock(g_ranges_mu);
    auto it = std::upper_bound(g_ranges.begin(), g_ranges.end(), ptr,
                               [](uintptr_t value, const HostRange &range) { return value < range.hva; });
    --it;  // LookupHostDva succeeded, so the containing range exists
    TORCH_CHECK(ptr + bytes <= it->hva + it->size, "host range [", host_ptr, ", +", bytes,
                " bytes) crosses the boundary of the registered range [", reinterpret_cast<void *>(it->hva), ", +",
                it->size, " bytes); register host pools in chunks that are multiples of the copy granularity");
    return dva;
}

// Pure bookkeeping: the aclrtHostRegister(ACL_HOST_REGISTER_MAPPED) call itself
// lives in python (host_memory.py, PR #39548's `acl.rt` binding path). This op
// only publishes the page-aligned mapping {hva, span, base_dva} into the
// process registry so the transfer ops' DVA resolver can find it. `hva` must be
// the page-aligned registered base, `nbytes` the page-aligned span, `dva` the
// device-visible address of `hva` (device_ptr minus the tensor's inner offset).
HOST_API void aclrt_record_host_mapping(int64_t hva, int64_t nbytes, int64_t dva)
{
    TORCH_CHECK(hva > 0 && hva % 4096 == 0, "aclrt_record_host_mapping: hva must be a 4K-aligned registered base, got ",
                hva);
    TORCH_CHECK(nbytes > 0 && nbytes % 4096 == 0,
                "aclrt_record_host_mapping: nbytes must be a positive 4K multiple, got ", nbytes);
    TORCH_CHECK(dva > 0, "aclrt_record_host_mapping: dva must be non-zero, got ", dva);
    HostRange range{static_cast<uintptr_t>(hva), static_cast<uint64_t>(nbytes), static_cast<uintptr_t>(dva)};
    std::lock_guard<std::mutex> lock(g_ranges_mu);
    auto it = FindByHva(range.hva);
    if (it != g_ranges.end()) {
        *it = range;  // re-registration of the same base replaces the entry
        return;
    }
    g_ranges.insert(std::upper_bound(g_ranges.begin(), g_ranges.end(), range,
                                     [](const HostRange &a, const HostRange &b) { return a.hva < b.hva; }),
                    range);
}

HOST_API void aclrt_forget_host_mapping(int64_t hva)
{
    std::lock_guard<std::mutex> lock(g_ranges_mu);
    auto it = FindByHva(static_cast<uintptr_t>(hva));
    TORCH_CHECK(it != g_ranges.end(), "aclrt_forget_host_mapping: address ",
                reinterpret_cast<void *>(hva), " is not registered");
    g_ranges.erase(it);
}

HOST_API int64_t aclrt_get_device_accessible_ptr(const at::Tensor &buf, int64_t device_index)
{
    (void)device_index;  // kept for signature parity with sgl_kernel.kvcacheio (CUDA)
    if (buf.is_cpu() && buf.data_ptr() != nullptr) {
        const int64_t dva = LookupHostDva(buf.data_ptr());
        if (dva != 0) {
            return dva;
        }
    }
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(buf.data_ptr()));
}

HOST_API void aclrt_set_host_dva_mode(const std::string &mode)
{
    if (mode == "aclrt") {
        g_passthrough.store(false, std::memory_order_relaxed);
    } else if (mode == "passthrough") {
        g_passthrough.store(true, std::memory_order_relaxed);
    } else {
        TORCH_CHECK(false, "aclrt_set_host_dva_mode: unknown mode '", mode,
                    "' (expected 'aclrt' or 'passthrough')");
    }
}

}  // namespace npu_kernel
}  // namespace sglang
