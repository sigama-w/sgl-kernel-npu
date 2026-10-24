/*
 * Host-side launcher for the kv_exchange AICPU device kernel.
 *
 * The six aclrt binary/launch entry points are resolved with dlsym at runtime
 * (they are absent from older CANN headers) and the kernel json is loaded
 * lazily once per process. The launch sequence mirrors
 * memfabric_hybrid device_urma_transport_manager.cpp:
 *   AclrtKernelArgsInit -> AclrtKernelArgsAppend(param by value)
 *   -> AclrtKernelArgsFinalize -> AclrtLaunchKernelWithConfig(TIMEOUT attr)
 */

#include "aicpu_launcher.h"

#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"

#include <acl/acl.h>
#include <driver/ascend_hal_define.h>

namespace sglang {
namespace npu_kernel {

namespace {

using aclrtBinHandle = void *;
using aclrtFuncHandle = void *;
using aclrtArgsHandle = void *;
using aclrtParamHandle = void *;

// ---- aclrt binary load / launch types (mirrors mf dl_acl_api.h) ----
enum class aclrtBinaryLoadOptionType : int32_t {
    ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD = 1,
    ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC = 2,
    ACL_RT_BINARY_LOAD_OPT_MAGIC = 2,
    ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE = 3,
};

union aclrtBinaryLoadOptionValue {
    uint32_t isLazyLoad;
    uint32_t magic;
    int32_t cpuKernelMode;
    uint32_t rsv[4];
};

struct aclrtBinaryLoadOption {
    aclrtBinaryLoadOptionType type;
    aclrtBinaryLoadOptionValue value;
};

struct aclrtBinaryLoadOptions {
    aclrtBinaryLoadOption *options;
    size_t numOpt;
};

enum aclrtLaunchKernelAttrId {
    ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE = 1,
    ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE = 2,
    ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE = 3,
    ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET,
    ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH,
    ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP,
    ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT,
    ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US = 8,
};

struct aclrtTimeoutUs {
    uint32_t timeoutLow;
    uint32_t timeoutHigh;
};

union aclrtLaunchKernelAttrValue {
    uint8_t schemMode;
    uint32_t localMemorySize;
    uint32_t engineType;
    uint32_t numBlocksOffset;
    uint8_t isBlockTaskPrefetch;
    uint8_t isDataDump;
    uint16_t timeout;
    aclrtTimeoutUs timeoutUs;
    uint32_t rsv[4];
};

struct aclrtLaunchKernelAttr {
    aclrtLaunchKernelAttrId id;
    aclrtLaunchKernelAttrValue value;
};

struct aclrtLaunchKernelCfg {
    aclrtLaunchKernelAttr *attrs;
    size_t numAttrs;
};

using aclrtBinaryLoadFromFileFunc = int32_t (*)(const char *, aclrtBinaryLoadOptions *, aclrtBinHandle *);
using aclrtBinaryGetFunctionFunc = int32_t (*)(const aclrtBinHandle, const char *, aclrtFuncHandle *);
using aclrtKernelArgsInitFunc = int32_t (*)(aclrtFuncHandle, aclrtArgsHandle *);
using aclrtKernelArgsAppendFunc = int32_t (*)(aclrtArgsHandle, void *, size_t, aclrtParamHandle *);
using aclrtKernelArgsFinalizeFunc = int32_t (*)(aclrtArgsHandle);
using aclrtLaunchKernelWithConfigFunc = int32_t (*)(aclrtFuncHandle, uint32_t, void *, aclrtLaunchKernelCfg *,
                                                    aclrtArgsHandle, void *);

// Must match struct KvExchangeAicpuParam in kv_exchange_aicpu_kernel.cc.
constexpr size_t kMetaSlots = 44;
struct KvExchangeAicpuParam {
    int64_t meta[kMetaSlots];
};

constexpr const char *kFuncName = "KvExchangeAicpu";
constexpr const char *kJsonRelativePath = "/opp/vendors/cust/op_impl/aicpu/config/libcann_kv_exchange_kernel.json";
constexpr uint32_t kBlockDim = 1;
constexpr uint16_t kLaunchTimeoutSec = 60;

struct AicpuLaunchApi {
    aclrtBinaryLoadFromFileFunc binaryLoadFromFile = nullptr;
    aclrtBinaryGetFunctionFunc binaryGetFunction = nullptr;
    aclrtKernelArgsInitFunc kernelArgsInit = nullptr;
    aclrtKernelArgsAppendFunc kernelArgsAppend = nullptr;
    aclrtKernelArgsFinalizeFunc kernelArgsFinalize = nullptr;
    aclrtLaunchKernelWithConfigFunc launchKernelWithConfig = nullptr;
};

void *DlsymRequired(const char *name)
{
    void *sym = dlsym(RTLD_DEFAULT, name);
    TORCH_CHECK(sym != nullptr, "kv_exchange_aicpu: dlsym(", name, ") failed: ", dlerror(),
                ". libascendcl must be loaded (import torch_npu first).");
    return sym;
}

const AicpuLaunchApi &GetApi()
{
    static AicpuLaunchApi api = []() {
        AicpuLaunchApi a{};
        a.binaryLoadFromFile = reinterpret_cast<aclrtBinaryLoadFromFileFunc>(DlsymRequired("aclrtBinaryLoadFromFile"));
        a.binaryGetFunction = reinterpret_cast<aclrtBinaryGetFunctionFunc>(DlsymRequired("aclrtBinaryGetFunction"));
        a.kernelArgsInit = reinterpret_cast<aclrtKernelArgsInitFunc>(DlsymRequired("aclrtKernelArgsInit"));
        a.kernelArgsAppend = reinterpret_cast<aclrtKernelArgsAppendFunc>(DlsymRequired("aclrtKernelArgsAppend"));
        a.kernelArgsFinalize = reinterpret_cast<aclrtKernelArgsFinalizeFunc>(DlsymRequired("aclrtKernelArgsFinalize"));
        a.launchKernelWithConfig =
            reinterpret_cast<aclrtLaunchKernelWithConfigFunc>(DlsymRequired("aclrtLaunchKernelWithConfig"));
        return a;
    }();
    return api;
}

std::string FindKernelJsonPath()
{
    if (const char *env = std::getenv("SGL_KV_EXCHANGE_AICPU_JSON")) {
        if (env[0] != '\0') {
            return env;
        }
    }
    std::vector<std::string> roots;
    if (const char *home = std::getenv("ASCEND_HOME_PATH")) {
        if (home[0] != '\0') {
            roots.emplace_back(home);
        }
    }
    roots.emplace_back("/usr/local/Ascend/ascend-toolkit/latest");
    roots.emplace_back("/usr/local/Ascend/cann");
    for (const auto &root : roots) {
        std::string candidate = root + kJsonRelativePath;
        if (access(candidate.c_str(), R_OK) == 0) {
            return candidate;
        }
    }
    return {};
}

// Loads the kernel json (once) and resolves the KvExchangeAicpu function handle.
aclrtFuncHandle GetKernelFunc()
{
    static aclrtFuncHandle funcHandle = nullptr;
    static std::once_flag once;
    std::call_once(once, []() {
        const auto &api = GetApi();
        std::string jsonPath = FindKernelJsonPath();
        TORCH_CHECK(!jsonPath.empty(),
                    "kv_exchange_aicpu: kernel json not found. Install libcann_kv_exchange_kernel.{json,so} under "
                    "<ASCEND_HOME_PATH>/opp/vendors/cust/op_impl/aicpu/{config,kernel} or set "
                    "SGL_KV_EXCHANGE_AICPU_JSON.");
        char resolved[4096] = {};
        TORCH_CHECK(realpath(jsonPath.c_str(), resolved) != nullptr, "kv_exchange_aicpu: realpath failed for ",
                    jsonPath);

        aclrtBinaryLoadOption loadOption{};
        loadOption.type = aclrtBinaryLoadOptionType::ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
        loadOption.value.cpuKernelMode = 0;
        aclrtBinaryLoadOptions loadOptions{};
        loadOptions.options = &loadOption;
        loadOptions.numOpt = 1;

        aclrtBinHandle binHandle = nullptr;
        int32_t ret = api.binaryLoadFromFile(resolved, &loadOptions, &binHandle);
        TORCH_CHECK(ret == 0, "kv_exchange_aicpu: aclrtBinaryLoadFromFile failed, path=", resolved, " ret=", ret);
        ret = api.binaryGetFunction(binHandle, kFuncName, &funcHandle);
        TORCH_CHECK(ret == 0 && funcHandle != nullptr, "kv_exchange_aicpu: aclrtBinaryGetFunction(", kFuncName,
                    ") failed, ret=", ret);
    });
    return funcHandle;
}

}  // namespace

void kv_exchange_aicpu(const at::Tensor &meta)
{
    TORCH_CHECK(meta.is_cpu(), "kv_exchange_aicpu: meta must be a CPU int64 tensor");
    TORCH_CHECK(meta.scalar_type() == at::kLong, "kv_exchange_aicpu: meta must be int64");
    TORCH_CHECK(meta.numel() == static_cast<int64_t>(kMetaSlots), "kv_exchange_aicpu: meta must have ",
                kMetaSlots, " elements, got ", meta.numel());

    const auto &api = GetApi();
    aclrtFuncHandle funcHandle = GetKernelFunc();

    KvExchangeAicpuParam param{};
    std::memcpy(param.meta, meta.data_ptr<int64_t>(), sizeof(param.meta));

    aclrtArgsHandle argsHandle = nullptr;
    int32_t ret = api.kernelArgsInit(funcHandle, &argsHandle);
    TORCH_CHECK(ret == 0, "kv_exchange_aicpu: AclrtKernelArgsInit failed, ret=", ret);

    aclrtParamHandle paramHandle = nullptr;
    ret = api.kernelArgsAppend(argsHandle, &param, sizeof(param), &paramHandle);
    if (ret != 0) {
        TORCH_CHECK(false, "kv_exchange_aicpu: AclrtKernelArgsAppend failed, ret=", ret);
    }
    ret = api.kernelArgsFinalize(argsHandle);
    TORCH_CHECK(ret == 0, "kv_exchange_aicpu: AclrtKernelArgsFinalize failed, ret=", ret);

    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = kLaunchTimeoutSec;
    aclrtLaunchKernelCfg cfg{};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    TORCH_CHECK(stream != nullptr, "kv_exchange_aicpu: no current NPU stream");
    ret = api.launchKernelWithConfig(funcHandle, kBlockDim, stream, &cfg, argsHandle, nullptr);
    TORCH_CHECK(ret == 0, "kv_exchange_aicpu: AclrtLaunchKernelWithConfig failed, ret=", ret);
}

// Host registration, mirroring LMCache-Ascend managed_mem.cpp:
//   primary  : aclrtHostRegister(ptr, size, ACL_HOST_REGISTER_MAPPED, &devPtr)
//              (driver >= 25.0 pins the pages internally)
//   fallback : halHostRegister(ptr, size, flag, devid, &devPtr) + explicit mlock
// NOTE: aclrtHostRegister takes FOUR arguments -- omitting the register type
// misaligns parameters and fails with ret=207000.
extern "C" {
drvError_t halHostRegister(void *srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void **dstPtr);
drvError_t halHostUnregister(void *srcPtr, UINT32 devid);
}

using aclrtHostRegisterFunc = int32_t (*)(void *, uint64_t, int32_t, void **);
using aclrtHostUnregisterFunc = int32_t (*)(void *);

#ifndef ACL_HOST_REGISTER_MAPPED
#define ACL_HOST_REGISTER_MAPPED 1
#endif

namespace {

struct HostRegApi {
    aclrtHostRegisterFunc aclRegister = nullptr;
    aclrtHostUnregisterFunc aclUnregister = nullptr;
};

const HostRegApi &GetHostRegApi()
{
    static HostRegApi api = []() {
        HostRegApi a{};
        // Optional: fall back to HAL if the aclrt entry is absent/fails.
        a.aclRegister = reinterpret_cast<aclrtHostRegisterFunc>(dlsym(RTLD_DEFAULT, "aclrtHostRegister"));
        a.aclUnregister = reinterpret_cast<aclrtHostUnregisterFunc>(dlsym(RTLD_DEFAULT, "aclrtHostUnregister"));
        return a;
    }();
    return api;
}

std::mutex g_hal_reg_mu;
std::unordered_set<void *> g_hal_registered;  // host ptrs registered via HAL

UINT32 HostRegisterFlag()
{
    // Mirror shm_allocator's SoC detection: A3 (Ascend910_93) uses SVM map,
    // everything else (incl. A5/Ascend950) uses PCIe throughput map.
    const char *soc = std::getenv("SOC_VERSION");
    if (soc == nullptr) {
        soc = std::getenv("ASCEND_SOC_VERSION");
    }
    if (soc != nullptr) {
        const std::string soc_name(soc);
        if (soc_name.rfind("Ascend910_93", 0) == 0 || soc_name.rfind("910_93", 0) == 0) {
            return HOST_SVM_MAP_DEV;
        }
    }
    return HOST_MEM_MAP_DEV_PCIE_TH;
}

int32_t CurrentDeviceId()
{
    int32_t device_id = -1;
    const aclError ret = aclrtGetDevice(&device_id);
    TORCH_CHECK(ret == ACL_ERROR_NONE && device_id >= 0,
                "kv_exchange_aicpu: aclrtGetDevice failed, ret=", static_cast<int32_t>(ret));
    return device_id;
}

}  // namespace

int64_t aclrt_host_register(const at::Tensor &buf)
{
    TORCH_CHECK(buf.is_cpu(), "aclrt_host_register: buf must be host (CPU) memory");
    TORCH_CHECK(buf.nbytes() > 0, "aclrt_host_register: buf must be non-empty");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(buf.data_ptr()) % 4096 == 0,
                "aclrt_host_register: host address must be 4K aligned, got ", buf.data_ptr());
    void *devPtr = nullptr;

    // Primary path: ACL runtime API (driver pins pages internally).
    const auto &api = GetHostRegApi();
    if (api.aclRegister != nullptr) {
        const int32_t ret = api.aclRegister(buf.data_ptr(), static_cast<uint64_t>(buf.nbytes()),
                                            ACL_HOST_REGISTER_MAPPED, &devPtr);
        if (ret == 0) {
            fprintf(stderr, "[kv_exchange_aicpu] host register via aclrtHostRegister OK ptr=%p bytes=%zu dva=%p\n",
                    buf.data_ptr(), static_cast<size_t>(buf.nbytes()), devPtr);
            return static_cast<int64_t>(reinterpret_cast<uintptr_t>(devPtr));
        }
        // ACL path failed (likely driver < 25.0); fall through to HAL.
        fprintf(stderr, "[kv_exchange_aicpu] aclrtHostRegister FAILED ret=%d, falling back to halHostRegister\n", ret);
    }

    // Fallback path: driver HAL + explicit mlock (driver < 25.0).
    void *host_ptr = buf.data_ptr();
    TORCH_CHECK(mlock(host_ptr, buf.nbytes()) == 0,
                "aclrt_host_register: mlock failed, errno=", errno);
    const drvError_t ret = halHostRegister(host_ptr, static_cast<UINT64>(buf.nbytes()), HostRegisterFlag(),
                                           static_cast<UINT32>(CurrentDeviceId()), &devPtr);
    if (ret != DRV_ERROR_NONE) {
        munlock(host_ptr, buf.nbytes());
        TORCH_CHECK(false, "aclrt_host_register: both aclrtHostRegister and halHostRegister failed, hal ret=",
                    static_cast<int32_t>(ret), " ptr=", host_ptr, " bytes=", buf.nbytes());
    }
    {
        std::lock_guard<std::mutex> lock(g_hal_reg_mu);
        g_hal_registered.insert(host_ptr);
    }
    fprintf(stderr, "[kv_exchange_aicpu] host register via halHostRegister OK ptr=%p bytes=%zu dva=%p\n",
            host_ptr, static_cast<size_t>(buf.nbytes()), devPtr);
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(devPtr));
}

void aclrt_host_unregister(const at::Tensor &buf)
{
    TORCH_CHECK(buf.is_cpu(), "aclrt_host_unregister: buf must be host (CPU) memory");
    void *host_ptr = buf.data_ptr();
    bool via_hal = false;
    {
        std::lock_guard<std::mutex> lock(g_hal_reg_mu);
        via_hal = g_hal_registered.erase(host_ptr) > 0;
    }
    if (via_hal) {
        const drvError_t ret = halHostUnregister(host_ptr, static_cast<UINT32>(CurrentDeviceId()));
        munlock(host_ptr, buf.nbytes());
        TORCH_CHECK(ret == DRV_ERROR_NONE, "aclrt_host_unregister: halHostUnregister failed, ret=",
                    static_cast<int32_t>(ret));
        return;
    }
    const auto &api = GetHostRegApi();
    TORCH_CHECK(api.aclUnregister != nullptr, "aclrt_host_unregister: aclrtHostUnregister not available");
    const int32_t ret = api.aclUnregister(host_ptr);
    TORCH_CHECK(ret == 0, "aclrt_host_unregister: aclrtHostUnregister failed, ret=", ret);
}

}  // namespace npu_kernel
}  // namespace sglang
