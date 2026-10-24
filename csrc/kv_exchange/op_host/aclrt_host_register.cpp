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

#include <acl/acl.h>

#include <dlfcn.h>

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

// int32_t aclrtHostRegister(void *ptr, uint64_t size, int32_t type, void **devPtr)
using aclrtHostRegisterFunc = int32_t (*)(void *, uint64_t, int32_t, void **);
// int32_t aclrtHostUnregister(void *ptr)
using aclrtHostUnregisterFunc = int32_t (*)(void *);

// ACL_HOST_REGISTER_MAPPED comes from <acl/acl_rt.h> (via acl/acl.h). It is an
// enum constant (=0 on CANN 9.1+), NOT a macro -- never #define it here, that
// would shadow the header value with a wrong flag and fail with ret=207000.

struct HostRange {
    uintptr_t hva = 0;
    uint64_t size = 0;
    uintptr_t dva = 0;
};

std::mutex g_ranges_mu;
std::vector<HostRange> g_ranges;  // sorted by hva, non-overlapping

aclrtHostRegisterFunc GetAclHostRegister()
{
    static aclrtHostRegisterFunc fn = reinterpret_cast<aclrtHostRegisterFunc>(dlsym(RTLD_DEFAULT, "aclrtHostRegister"));
    TORCH_CHECK(fn != nullptr, "aclrtHostRegister is not available in this CANN runtime; "
                               "host registration for the AIV kv_exchange backend cannot proceed");
    return fn;
}

aclrtHostUnregisterFunc GetAclHostUnregister()
{
    static aclrtHostUnregisterFunc fn =
        reinterpret_cast<aclrtHostUnregisterFunc>(dlsym(RTLD_DEFAULT, "aclrtHostUnregister"));
    TORCH_CHECK(fn != nullptr, "aclrtHostUnregister is not available in this CANN runtime");
    return fn;
}

void CheckRegistrableCpuBuf(const at::Tensor &buf)
{
    TORCH_CHECK(buf.is_cpu(), "aclrt_host_register: buf must be host (CPU) memory");
    TORCH_CHECK(buf.nbytes() > 0, "aclrt_host_register: buf must be non-empty");
    TORCH_CHECK(buf.data_ptr() != nullptr, "aclrt_host_register: buf must have a valid data_ptr");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(buf.data_ptr()) % 4096 == 0,
                "aclrt_host_register: host address must be 4K aligned, got ", buf.data_ptr());
}

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

HOST_API int64_t aclrt_host_register(const at::Tensor &buf)
{
    CheckRegistrableCpuBuf(buf);
    void *host_ptr = buf.data_ptr();
    const uint64_t bytes = static_cast<uint64_t>(buf.nbytes());

    void *dev_ptr = nullptr;
    const int32_t ret = GetAclHostRegister()(host_ptr, bytes, ACL_HOST_REGISTER_MAPPED, &dev_ptr);
    TORCH_CHECK(ret == 0 && dev_ptr != nullptr, "aclrtHostRegister failed, ret=", ret, " ptr=", host_ptr,
                " bytes=", bytes, " (type=ACL_HOST_REGISTER_MAPPED=", static_cast<int32_t>(ACL_HOST_REGISTER_MAPPED),
                ")");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(dev_ptr) != reinterpret_cast<uintptr_t>(host_ptr),
                "aclrtHostRegister returned the host address unchanged; the mapping is not usable by AIV kernels");

    {
        std::lock_guard<std::mutex> lock(g_ranges_mu);
        TORCH_CHECK(FindByHva(reinterpret_cast<uintptr_t>(host_ptr)) == g_ranges.end(),
                    "aclrt_host_register: address ", host_ptr, " is already registered");
        HostRange range{reinterpret_cast<uintptr_t>(host_ptr), bytes, reinterpret_cast<uintptr_t>(dev_ptr)};
        g_ranges.insert(std::upper_bound(g_ranges.begin(), g_ranges.end(), range,
                                         [](const HostRange &a, const HostRange &b) { return a.hva < b.hva; }),
                        range);
    }
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(dev_ptr));
}

HOST_API void aclrt_host_unregister(const at::Tensor &buf)
{
    TORCH_CHECK(buf.is_cpu(), "aclrt_host_unregister: buf must be host (CPU) memory");
    void *host_ptr = buf.data_ptr();
    {
        std::lock_guard<std::mutex> lock(g_ranges_mu);
        auto it = FindByHva(reinterpret_cast<uintptr_t>(host_ptr));
        TORCH_CHECK(it != g_ranges.end(), "aclrt_host_unregister: address ", host_ptr, " is not registered");
        g_ranges.erase(it);
    }
    const int32_t ret = GetAclHostUnregister()(host_ptr);
    TORCH_CHECK(ret == 0, "aclrtHostUnregister failed, ret=", ret, " ptr=", host_ptr);
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
