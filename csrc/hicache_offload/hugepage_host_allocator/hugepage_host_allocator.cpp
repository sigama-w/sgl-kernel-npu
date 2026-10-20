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

#include <acl/acl.h>
#include <driver/ascend_hal_define.h>

#include <sys/mman.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

extern "C" {
drvError_t halHostRegister(void *srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void **dstPtr);
drvError_t halHostUnregister(void *srcPtr, UINT32 devid);
}

namespace {

constexpr uint64_t k1GB = 1ULL << 30;
constexpr uint64_t k2MB = 2ULL << 20;
constexpr uint64_t k4KB = 4ULL << 10;
// MAP_HUGE_1GB encoding: log2(page size) << 26 (kernel uapi convention).
constexpr int kMapHugeShift = 26;

struct HpEntry {
    void *host_ptr = nullptr;
    void *dev_ptr = nullptr;
    uint64_t size = 0;      // requested bytes
    uint64_t map_size = 0;  // mmap length (hugepage aligned)
    uint64_t page_size = 0;
    int device_id = -1;
    bool is_registered = false;
};

std::mutex g_mu;
std::unordered_map<void *, HpEntry> g_entries;
std::unordered_set<int> g_acl_initialized_devices;

bool HasPrefix(const std::string &s, const std::string &prefix)
{
    return s.rfind(prefix, 0) == 0;
}

// Same SOC detection as shm_allocator: Ascend910_93 supports shared virtual
// memory mapping, everything else (including Ascend950) uses the PCIe
// translation-threshold host mapping.
uint32_t GetRegisterFlag()
{
    const char *soc = std::getenv("SOC_VERSION");
    if (soc == nullptr) {
        soc = std::getenv("ASCEND_SOC_VERSION");
    }
    const bool svm = soc != nullptr &&
                     (HasPrefix(std::string(soc), "Ascend910_93") || HasPrefix(std::string(soc), "910_93"));
    return svm ? HOST_SVM_MAP_DEV : HOST_MEM_MAP_DEV_PCIE_TH;
}

int EnsureAclDevice(int device_id)
{
    if (g_acl_initialized_devices.find(device_id) != g_acl_initialized_devices.end()) {
        return 0;
    }

    int32_t current_device = -1;
    aclError ret = aclrtGetDevice(&current_device);
    if (ret == ACL_ERROR_NONE) {
        if (current_device == device_id) {
            g_acl_initialized_devices.insert(device_id);
            return 0;
        }
        ret = aclrtSetDevice(device_id);
        if (ret != ACL_ERROR_NONE) {
            return -2;
        }
        g_acl_initialized_devices.insert(device_id);
        return 0;
    }

    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return -1;
    }
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_ERROR_NONE) {
        return -2;
    }
    g_acl_initialized_devices.insert(device_id);
    return 0;
}

// Round `size` up to a multiple of `page_size`.
uint64_t RoundUp(uint64_t size, uint64_t page_size)
{
    return (size + page_size - 1) / page_size * page_size;
}

// A/B switch: pick the mmap candidates from SGLANG_HICACHE_HUGEPAGE_MODE.
// Non-auto modes contain a single candidate so a missing hugepage pool fails
// loudly instead of silently degrading (an A/B run must never mix page
// sizes). Returns the number of candidates written to the out arrays.
int SelectCandidates(const char *mode, uint64_t *out_pages, int *out_flags)
{
    const std::string m = mode != nullptr && mode[0] != '\0' ? mode : "auto";
    if (m == "off") {
        out_pages[0] = k4KB;
        out_flags[0] = 0;
        return 1;
    }
    if (m == "1g") {
        out_pages[0] = k1GB;
        out_flags[0] = MAP_HUGETLB | (30U << kMapHugeShift);
        return 1;
    }
    if (m == "2m") {
        out_pages[0] = k2MB;
        out_flags[0] = MAP_HUGETLB | (21U << kMapHugeShift);
        return 1;
    }
    // auto (default): 1GB -> 2MB -> 4KB.
    out_pages[0] = k1GB;
    out_flags[0] = MAP_HUGETLB | (30U << kMapHugeShift);
    out_pages[1] = k2MB;
    out_flags[1] = MAP_HUGETLB | (21U << kMapHugeShift);
    out_pages[2] = k4KB;
    out_flags[2] = 0;
    return 3;
}

int MmapHostMemory(uint64_t size, void **host_ptr, uint64_t *map_size, uint64_t *page_size)
{
    uint64_t candPages[3];
    int candFlags[3];
    const int numCandidates =
        SelectCandidates(std::getenv("SGLANG_HICACHE_HUGEPAGE_MODE"), candPages, candFlags);

    for (int i = 0; i < numCandidates; ++i) {
        const uint64_t rounded = RoundUp(size, candPages[i]);
        void *ptr = mmap(nullptr, rounded, PROT_READ | PROT_WRITE,
                         candFlags[i] | MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (ptr != MAP_FAILED) {
            // The 4KB fallback is silent: operators configure hugepages via
            // /proc/sys/vm/nr_hugepages; hugepage-backed mapping is the whole
            // point of this allocator (DMA efficiency on the offload path).
            *host_ptr = ptr;
            *map_size = rounded;
            *page_size = candPages[i];
            return 0;
        }
    }
    return -1;
}

}  // namespace

extern "C" {

int hp_alloc(int device_id, uint64_t size, uint64_t *out_host_ptr, uint64_t *out_dev_ptr,
             uint64_t *out_page_size)
{
    if (out_host_ptr == nullptr || out_dev_ptr == nullptr || size == 0) {
        return -1;
    }

    void *host_ptr = nullptr;
    uint64_t map_size = 0;
    uint64_t page_size = 0;
    if (MmapHostMemory(size, &host_ptr, &map_size, &page_size) != 0) {
        return -3;
    }

    // Materialize the pages so halHostRegister sees a fully backed mapping.
    std::memset(host_ptr, 0, map_size);

    std::lock_guard<std::mutex> lock(g_mu);

    int ret = EnsureAclDevice(device_id);
    if (ret != 0) {
        (void)munmap(host_ptr, map_size);
        return ret;
    }

    void *dev_ptr = nullptr;
    if (halHostRegister(host_ptr, map_size, GetRegisterFlag(), static_cast<UINT32>(device_id), &dev_ptr) !=
        DRV_ERROR_NONE) {
        (void)munmap(host_ptr, map_size);
        return -4;
    }

    HpEntry entry;
    entry.host_ptr = host_ptr;
    entry.dev_ptr = dev_ptr;
    entry.size = size;
    entry.map_size = map_size;
    entry.page_size = page_size;
    entry.device_id = device_id;
    entry.is_registered = true;
    g_entries.emplace(host_ptr, entry);

    *out_host_ptr = reinterpret_cast<uint64_t>(host_ptr);
    *out_dev_ptr = reinterpret_cast<uint64_t>(dev_ptr);
    if (out_page_size != nullptr) {
        *out_page_size = page_size;
    }
    return 0;
}

int hp_free(void *host_ptr, int device_id)
{
    if (host_ptr == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_entries.find(host_ptr);
    if (it == g_entries.end()) {
        return -2;
    }
    if (it->second.is_registered &&
        halHostUnregister(it->second.host_ptr, static_cast<UINT32>(device_id)) != DRV_ERROR_NONE) {
        return -3;
    }
    (void)munmap(it->second.host_ptr, it->second.map_size);
    g_entries.erase(it);
    return 0;
}

int hp_free_all(int device_id)
{
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto &kv : g_entries) {
        if (kv.second.is_registered) {
            (void)halHostUnregister(kv.second.host_ptr, static_cast<UINT32>(device_id));
        }
        (void)munmap(kv.second.host_ptr, kv.second.map_size);
    }
    g_entries.clear();
    g_acl_initialized_devices.erase(device_id);
    return 0;
}

}  // extern "C"
