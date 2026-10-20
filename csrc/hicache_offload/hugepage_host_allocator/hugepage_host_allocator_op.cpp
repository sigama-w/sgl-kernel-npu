// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hugepage_host_allocator.h"

#include "torch_helper.h"

#include <cstdint>
#include <tuple>

namespace sglang {
namespace npu_kernel {

/*
 * Allocates hugepage-backed host memory and registers it with the NPU
 * driver. Returns ``(host_ptr, dev_ptr, page_size)`` where ``dev_ptr`` is the
 * device-visible address (DVA) that AIV device kernels can dereference
 * directly and ``page_size`` is the backing page actually used (1GB/2MB
 * hugepage or 4KB), which lets callers verify the hugepage configuration
 * took effect. Allocation order follows SGLANG_HICACHE_HUGEPAGE_MODE:
 * auto/unset = 1GB -> 2MB -> 4KB fallback; 1g/2m/off pin a single page size
 * and fail if unavailable.
 *
 * The returned pointer stays valid until :func:`hugepage_host_free` (or
 * ``hugepage_host_free_all``) is called with the same address.
 */
std::tuple<int64_t, int64_t, int64_t> hugepage_host_alloc(int64_t num_bytes, int64_t device_id)
{
    TORCH_CHECK(num_bytes > 0, "hugepage_host_alloc: num_bytes must be positive, got ", num_bytes);
    TORCH_CHECK(device_id >= 0, "hugepage_host_alloc: device_id must be non-negative, got ", device_id);

    uint64_t host_ptr = 0;
    uint64_t dev_ptr = 0;
    uint64_t page_size = 0;
    int ret = hp_alloc(static_cast<int>(device_id), static_cast<uint64_t>(num_bytes), &host_ptr, &dev_ptr,
                       &page_size);
    TORCH_CHECK(ret == 0, "hugepage_host_alloc failed, ret=", ret);
    return {static_cast<int64_t>(host_ptr), static_cast<int64_t>(dev_ptr), static_cast<int64_t>(page_size)};
}

/*
 * Unregisters and releases the allocation identified by ``host_ptr``.
 */
void hugepage_host_free(int64_t host_ptr, int64_t device_id)
{
    TORCH_CHECK(host_ptr != 0, "hugepage_host_free: host_ptr must be non-zero");
    TORCH_CHECK(device_id >= 0, "hugepage_host_free: device_id must be non-negative, got ", device_id);
    int ret = hp_free(reinterpret_cast<void *>(static_cast<uintptr_t>(host_ptr)), static_cast<int>(device_id));
    TORCH_CHECK(ret == 0, "hugepage_host_free failed, ret=", ret);
}

/*
 * Unregisters and releases every allocation created by this process on
 * ``device_id``.
 */
void hugepage_host_free_all(int64_t device_id)
{
    TORCH_CHECK(device_id >= 0, "hugepage_host_free_all: device_id must be non-negative, got ", device_id);
    int ret = hp_free_all(static_cast<int>(device_id));
    TORCH_CHECK(ret == 0, "hugepage_host_free_all failed, ret=", ret);
}

}  // namespace npu_kernel
}  // namespace sglang
