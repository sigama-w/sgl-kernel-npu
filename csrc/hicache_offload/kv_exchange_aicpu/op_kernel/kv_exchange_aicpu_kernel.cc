/*
 * AICPU device-side kernel for sgl-kernel-npu KV exchange (hicache offload).
 *
 * Moves KV pages between device HBM and host-pinned memory that has been
 * registered with aclrtHostRegister (device-side DVA). Both sides of every
 * copy are device-visible addresses, so the AICPU core can memcpy directly.
 *
 * The kernel mirrors the memfabric_hybrid AICPU kernel style: a bare
 * extern "C" function with no external linkage dependencies, cross-compiled
 * as a shared library and loaded via aclrtBinaryLoadFromFile(json).
 *
 * Launch contract (host side, see aicpu_launcher.cpp):
 *   - single parameter: KvExchangeAicpuParam* appended by AclrtKernelArgsAppend
 *   - blockDim = 1 (all work is done in one AICPU core)
 *
 * meta layout (44 int64 slots, by value inside the param struct):
 *   [0] num_components        1..4 (e.g. K and V count as 2)
 *   [1] num_indices           number of token slots to copy
 *   [2] page_size             tokens per page
 *   [3] direction             0 = probe, 1 = H2D, 2 = D2H
 *   [4] device_indices_ptr    DVA of int64 token indices into device cache
 *   [5] host_indices_ptr      DVA of int64 token indices into host cache
 *   [6] host_layout_mode      0 = page_first, 1 = layer_first
 *   [7] probe only: device int32 status buffer (>=16 slots)
 *
 *   PROBE MODE (direction=0): answers "which address spaces can this AICPU
 *   stars sandbox dereference?" by touching one address per stage and writing
 *   a stage magic BEFORE the access plus the observed value AFTER it. A crash
 *   therefore leaves the highest magic without its companion value:
 *     status[0]=M+0 always (kernel running, meta readable)
 *     status[1]=M+1 then status[9]=device_indices[0]      (device mem read)
 *     status[2]=M+2 then status[10]=device_base[0] byte   (device data read)
 *     status[3]=M+3 then status[11]=host_indices[0]       (host HVA read)
 *     status[4]=M+4 then status[12]=host_base[0] byte     (host data read)
 *     status[5]=M+5 all probes completed
 *
 *   per component c (base = 8 + 9*c):
 *   +0 device_base            DVA of the device KV tensor for this component
 *   +1 host_base              DVA of the registered host KV tensor
 *   +2 device_layer_pitch     bytes between consecutive layers (device)
 *   +3 device_page_pitch      bytes between consecutive pages (device)
 *   +4 host_page_pitch        bytes between consecutive pages (host)
 *   +5 host_layer_pitch       bytes between consecutive layers (host)
 *   +6 width                  bytes of one full page row (page_size tokens)
 *   +7 layer_lo               first layer owned by this component (inclusive)
 *   +8 layer_hi               last layer owned by this component (exclusive)
 */

#include <cstdint>
#include <cstring>

#include "dlog_pub.h" // device-side slog; output lands in host plog files

namespace {

constexpr int kMetaSlots = 44;
constexpr int64_t kMaxComponents = 4;
constexpr int64_t kMaxLayers = 1LL << 20;

// meta header slot offsets
constexpr int kMetaNumComponents = 0;
constexpr int kMetaNumIndices = 1;
constexpr int kMetaPageSize = 2;
constexpr int kMetaDirection = 3;
constexpr int kMetaDeviceIndicesPtr = 4;
constexpr int kMetaHostIndicesPtr = 5;
constexpr int kMetaHostLayoutMode = 6;

// per-component slot offsets
constexpr int kCompStride = 9;
constexpr int kCompDeviceBase = 0;
constexpr int kCompHostBase = 1;
constexpr int kCompDeviceLayerPitch = 2;
constexpr int kCompDevicePagePitch = 3;
constexpr int kCompHostPagePitch = 4;
constexpr int kCompHostLayerPitch = 5;
constexpr int kCompWidth = 6;
constexpr int kCompLayerLo = 7;
constexpr int kCompLayerHi = 8;

constexpr int64_t kDirectionProbe = 0;
constexpr int64_t kDirectionH2D = 1;
constexpr int64_t kDirectionD2H = 2;
constexpr int64_t kLayoutPageFirst = 0;
constexpr int64_t kLayoutLayerFirst = 1;

constexpr int kMetaProbeStatusPtr = 7;
constexpr int32_t kProbeMagicBase = 0x41435030; // "ACP0"

// error codes (kept non-zero and coarse; host side only needs pass/fail)
constexpr int32_t kRetOk = 0;
constexpr int32_t kRetInvalidParam = 1;
constexpr int32_t kRetCopyFailed = 2;

} // namespace

struct KvExchangeAicpuParam {
    int64_t meta[kMetaSlots];
};

extern "C" int32_t KvExchangeAicpu(KvExchangeAicpuParam *param)
{
    if (param == nullptr) {
        return kRetInvalidParam;
    }
    const int64_t *meta = param->meta;

    if (meta[kMetaDirection] == kDirectionProbe) {
        // Probe mode: touch one address per stage. Write the stage magic
        // BEFORE the access and the observed value AFTER it, so a crash
        // leaves the highest magic without its companion value.
        // meta[kMetaPageSize] is reused as the stage limit (0..4, default 4):
        // 0 = touch nothing (pure return, proves the kernel executes at all),
        // 1-2 only touch device memory, 3-4 touch host memory.
        // Every stage also dlog-prints BEFORE the access: unlike the status
        // buffer, the host plog survives an AICPU crash, so the last printed
        // stage identifies the killer even when the device is poisoned.
        volatile int32_t *status = reinterpret_cast<volatile int32_t *>(meta[kMetaProbeStatusPtr]);
        const int64_t stageLimit = meta[kMetaPageSize] >= 0 && meta[kMetaPageSize] <= 4 ? meta[kMetaPageSize] : 4;
        const int64_t *devIdx = reinterpret_cast<const int64_t *>(meta[kMetaDeviceIndicesPtr]);
        const int64_t *hostIdx = reinterpret_cast<const int64_t *>(meta[kMetaHostIndicesPtr]);
        const int64_t *comp = meta + 8;
        const uint8_t *devBase = reinterpret_cast<const uint8_t *>(comp[0]);
        const uint8_t *hostBase = reinterpret_cast<const uint8_t *>(comp[1]);

        dlog_error(static_cast<int32_t>(APP),
                   "[KVX] probe S0 enter param=%p status=%p stageLimit=%d nIdx=%d devIdx=%p hostIdx=%p "
                   "devBase=%p hostBase=%p meta4=%lx meta5=%lx meta8=%lx meta9=%lx",
                   static_cast<const void *>(param), static_cast<const void *>(const_cast<int32_t *>(status)),
                   static_cast<int>(stageLimit), static_cast<int>(meta[kMetaNumIndices]),
                   static_cast<const void *>(devIdx), static_cast<const void *>(hostIdx),
                   static_cast<const void *>(devBase), static_cast<const void *>(hostBase),
                   meta[kMetaDeviceIndicesPtr], meta[kMetaHostIndicesPtr], comp[0], comp[1]);
        status[0] = kProbeMagicBase + 0; // kernel running, param/meta readable
        status[8] = static_cast<int32_t>(meta[kMetaNumIndices]);
        if (stageLimit >= 1) {
            dlog_error(static_cast<int32_t>(APP), "[KVX] probe S1 enter devIdx=%p", static_cast<const void *>(devIdx));
            status[1] = kProbeMagicBase + 1; // about to read device memory
            status[9] = static_cast<int32_t>(devIdx[0]);
            dlog_error(static_cast<int32_t>(APP), "[KVX] probe S1 done devIdx0=%lx", devIdx[0]);
        }
        if (stageLimit >= 2) {
            dlog_error(static_cast<int32_t>(APP), "[KVX] probe S2 enter devBase=%p", static_cast<const void *>(devBase));
            status[2] = kProbeMagicBase + 2; // about to read device data
            status[10] = static_cast<int32_t>(devBase[0]);
            dlog_error(static_cast<int32_t>(APP), "[KVX] probe S2 done devBase0=%x", status[10]);
        }
        if (stageLimit >= 3) {
            dlog_error(static_cast<int32_t>(APP), "[KVX] probe S3 enter hostIdx=%p", static_cast<const void *>(hostIdx));
            status[3] = kProbeMagicBase + 3; // about to read host HVA
            status[11] = static_cast<int32_t>(hostIdx[0]);
            dlog_error(static_cast<int32_t>(APP), "[KVX] probe S3 done hostIdx0=%lx", hostIdx[0]);
        }
        if (stageLimit >= 4) {
            dlog_error(static_cast<int32_t>(APP), "[KVX] probe S4 enter hostBase=%p", static_cast<const void *>(hostBase));
            status[4] = kProbeMagicBase + 4; // about to read host data
            status[12] = static_cast<int32_t>(hostBase[0]);
            dlog_error(static_cast<int32_t>(APP), "[KVX] probe S4 done hostBase0=%x", status[12]);
        }
        status[5] = kProbeMagicBase + 5; // requested probes completed
        dlog_error(static_cast<int32_t>(APP), "[KVX] probe all stages done");
        return kRetOk;
    }

    const int64_t numComponents = meta[kMetaNumComponents];
    const int64_t numIndices = meta[kMetaNumIndices];
    const int64_t pageSize = meta[kMetaPageSize];
    const int64_t direction = meta[kMetaDirection];
    const int64_t hostLayoutMode = meta[kMetaHostLayoutMode];

    if (numComponents <= 0 || numComponents > kMaxComponents) {
        return kRetInvalidParam;
    }
    if (numIndices < 0) {
        return kRetInvalidParam;
    }
    if (pageSize <= 0) {
        return kRetInvalidParam;
    }
    if (direction != kDirectionH2D && direction != kDirectionD2H) {
        return kRetInvalidParam;
    }
    if (hostLayoutMode != kLayoutPageFirst && hostLayoutMode != kLayoutLayerFirst) {
        return kRetInvalidParam;
    }

    const int64_t *deviceIndices = reinterpret_cast<const int64_t *>(meta[kMetaDeviceIndicesPtr]);
    const int64_t *hostIndices = reinterpret_cast<const int64_t *>(meta[kMetaHostIndicesPtr]);
    if (numIndices > 0 && (deviceIndices == nullptr || hostIndices == nullptr)) {
        return kRetInvalidParam;
    }

    // per-component sanity + precomputed fields
    uint8_t *deviceBase[kMaxComponents] = {};
    uint8_t *hostBase[kMaxComponents] = {};
    int64_t deviceLayerPitch[kMaxComponents] = {};
    int64_t devicePagePitch[kMaxComponents] = {};
    int64_t hostPagePitch[kMaxComponents] = {};
    int64_t hostLayerPitch[kMaxComponents] = {};
    int64_t tokenBytes[kMaxComponents] = {};
    int64_t layerLo[kMaxComponents] = {};
    int64_t layerHi[kMaxComponents] = {};

    for (int64_t c = 0; c < numComponents; ++c) {
        const int64_t *comp = meta + 8 + c * kCompStride;
        const int64_t width = comp[kCompWidth];
        layerLo[c] = comp[kCompLayerLo];
        layerHi[c] = comp[kCompLayerHi];
        if (width <= 0 || width % pageSize != 0) {
            return kRetInvalidParam;
        }
        if (layerLo[c] < 0 || layerHi[c] < layerLo[c] || layerHi[c] > kMaxLayers) {
            return kRetInvalidParam;
        }
        deviceBase[c] = reinterpret_cast<uint8_t *>(comp[kCompDeviceBase]);
        hostBase[c] = reinterpret_cast<uint8_t *>(comp[kCompHostBase]);
        deviceLayerPitch[c] = comp[kCompDeviceLayerPitch];
        devicePagePitch[c] = comp[kCompDevicePagePitch];
        hostPagePitch[c] = comp[kCompHostPagePitch];
        hostLayerPitch[c] = comp[kCompHostLayerPitch];
        tokenBytes[c] = width / pageSize;
        if (deviceBase[c] == nullptr || hostBase[c] == nullptr) {
            return kRetInvalidParam;
        }
    }

    for (int64_t i = 0; i < numIndices; ++i) {
        const int64_t devTok = deviceIndices[i];
        const int64_t hostTok = hostIndices[i];
        if (devTok < 0 || hostTok < 0) {
            return kRetInvalidParam;
        }
        const int64_t devPage = devTok / pageSize;
        const int64_t devIntra = devTok % pageSize;
        const int64_t hostPage = hostTok / pageSize;
        const int64_t hostIntra = hostTok % pageSize;

        for (int64_t c = 0; c < numComponents; ++c) {
            const int64_t devTokenOffset = devPage * devicePagePitch[c] + devIntra * tokenBytes[c];

            for (int64_t layer = layerLo[c]; layer < layerHi[c]; ++layer) {
                uint8_t *devAddr = deviceBase[c] + layer * deviceLayerPitch[c] + devTokenOffset;
                uint8_t *hostAddr;
                if (hostLayoutMode == kLayoutPageFirst) {
                    hostAddr = hostBase[c] + hostPage * hostPagePitch[c] + layer * hostLayerPitch[c] +
                               hostIntra * tokenBytes[c];
                } else {
                    hostAddr = hostBase[c] + layer * hostLayerPitch[c] + hostPage * hostPagePitch[c] +
                               hostIntra * tokenBytes[c];
                }
                uint8_t *dst = direction == kDirectionH2D ? devAddr : hostAddr;
                const uint8_t *src = direction == kDirectionH2D ? hostAddr : devAddr;
                if (memcpy(dst, src, static_cast<size_t>(tokenBytes[c])) != dst) {
                    return kRetCopyFailed;
                }
            }
        }
    }
    return kRetOk;
}
