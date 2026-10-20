// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_HUGEPAGE_HOST_ALLOCATOR_H
#define SGL_KERNEL_NPU_HUGEPAGE_HOST_ALLOCATOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hugepage-backed host memory allocator for the HiCache L1 <-> L2 offload
// path. Allocation order: 1GB hugepage (MAP_HUGETLB | MAP_HUGE_1GB), then
// 2MB hugepage, then regular 4KB pages (with a warning). Every allocation is
// registered with the NPU driver via halHostRegister so that AIV device
// kernels can dereference the returned device-visible address (DVA) directly,
// replacing the MemFabric DRAM_MAP_HOST_VA allocator on the sglang side.
//
// A/B switch: SGLANG_HICACHE_HUGEPAGE_MODE controls the candidate order --
//   unset / auto : 1GB -> 2MB -> 4KB (production default)
//   1g           : 1GB only (fails if 1GB hugepages are not configured)
//   2m           : 2MB only (fails if 2MB hugepages are not configured)
//   off          : 4KB pages only (benchmark baseline)
// Non-auto modes fail loudly instead of silently falling back, so an A/B
// comparison never mixes page sizes.

// Allocates `size` bytes of host memory (rounded up to the page size actually
// used), registers it for `device_id`, and returns the host address, the
// device-visible address (DVA), and the page size actually used. Returns 0
// on success.
int hp_alloc(int device_id, uint64_t size, uint64_t *out_host_ptr, uint64_t *out_dev_ptr,
             uint64_t *out_page_size);

// Unregisters and releases the allocation identified by `host_ptr`.
// Returns 0 on success, -2 if `host_ptr` is unknown.
int hp_free(void *host_ptr, int device_id);

// Unregisters and releases every allocation created by this process.
// Returns 0 on success.
int hp_free_all(int device_id);

#ifdef __cplusplus
}
#endif

#endif  // SGL_KERNEL_NPU_HUGEPAGE_HOST_ALLOCATOR_H
