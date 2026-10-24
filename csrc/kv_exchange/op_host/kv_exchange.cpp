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

#include "aclrtlaunch_kv_exchange.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr int64_t kMetaHeader = 8;
constexpr int64_t kMetaStride = 9;
constexpr int64_t kMaxComponents = 4;
constexpr int64_t kMinMetaNumel = kMetaHeader + kMetaStride * kMaxComponents;

constexpr int64_t kDirectionH2D = 1;
constexpr int64_t kDirectionD2H = 2;
constexpr int64_t kHostLayoutPageFirst = 0;

// AIV core count differs per SoC family (arch35/Ascend950: 64, arch22: 48);
// SGLANG_KV_EXCHANGE_BLOCK_DIM overrides for perf sweeps.
int64_t ResolveBlockDim()
{
    if (const char *env = std::getenv("SGLANG_KV_EXCHANGE_BLOCK_DIM")) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed > 0) {
            return static_cast<int64_t>(parsed);
        }
    }
#ifdef SGL_KERNEL_ARCH_35
    return 64;
#else
    return 48;
#endif
}

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

// The kernel walks (page, layer) blocks addressed as
// base + layer*layer_pitch + page*page_pitch and copies `width` contiguous
// bytes from each block, so only the trailing (page_size, 1, width) dims must
// be packed; page/layer dims may be strided views (e.g. host layer slices).
bool TrailingDimsPacked(const at::Tensor &t)
{
    int64_t expect = 1;
    for (int64_t i = t.dim() - 1; i >= 2; --i) {
        if (t.stride(i) != expect) {
            return false;
        }
        expect *= t.size(i);
    }
    return true;
}

void CheckComponentPair(const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t page_size, int64_t which)
{
    TORCH_CHECK(device_buf.scalar_type() == host_buf.scalar_type(), "component ", which,
                ": device and host dtypes must match, got ", device_buf.scalar_type(), " vs ",
                host_buf.scalar_type());
    TORCH_CHECK(device_buf.dim() == 5 && host_buf.dim() == 5, "component ", which,
                ": buffers must be 5-D -- device (layers, pages, page_size, 1, width), host "
                "(pages, layers, page_size, 1, width), got device ", device_buf.dim(), "-D / host ",
                host_buf.dim(), "-D");
    TORCH_CHECK(device_buf.size(2) == page_size, "component ", which, ": device dim(2) must equal page_size");
    TORCH_CHECK(host_buf.size(2) == page_size, "component ", which, ": host dim(2) must equal page_size");
    TORCH_CHECK(device_buf.size(1) == host_buf.size(0), "component ", which,
                ": device page count (dim 1) must equal host page count (dim 0)");
    TORCH_CHECK(device_buf.size(4) * device_buf.element_size() == host_buf.size(4) * host_buf.element_size(),
                "component ", which, ": device and host row widths (last dim in bytes) must match");
    TORCH_CHECK(device_buf.size(3) == host_buf.size(3), "component ", which,
                ": device and host singleton dims (dim 3) must match");
    TORCH_CHECK(TrailingDimsPacked(device_buf) && TrailingDimsPacked(host_buf), "component ", which,
                ": trailing (page_size, 1, width) dims must be packed (contiguous within a "
                "(page, layer) block)");
}

// meta layout (see op_kernel/kv_exchange_kernel.cpp):
//   [0] num_components [1] num_pages [2] page_size [3] direction
//   [4] device_indices_ptr [5] host_indices_ptr [6] host_layout_mode [7] start_dim
//   per component c at 8 + 9*c: dev_base, host_base(DVA), dev_layer_pitch,
//   dev_page_pitch, host_page_pitch, host_layer_pitch, width, layer_lo, layer_hi
void FillComponent(std::vector<int64_t> &meta, int64_t c, const at::Tensor &device_buf, const at::Tensor &host_buf,
                   int64_t layer_lo, int64_t layer_hi)
{
    const int64_t esz = device_buf.element_size();
    const int64_t o = kMetaHeader + kMetaStride * c;
    meta[o + 0] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_buf.data_ptr()));
    meta[o + 1] = ResolveHostDva(host_buf.data_ptr(), host_buf.nbytes());
    meta[o + 2] = device_buf.stride(0) * esz;  // device_layer_pitch
    meta[o + 3] = device_buf.stride(1) * esz;  // device_page_pitch
    meta[o + 4] = host_buf.stride(0) * esz;    // host_page_pitch
    meta[o + 5] = host_buf.stride(1) * esz;    // host_layer_pitch
    // The kernel copies this many contiguous bytes per (page, layer) block:
    // the packed extent of the trailing dims, matching the Memfabric
    // kv_exchange meta builder (prod(shape[2:]) * itemsize).
    int64_t width = esz;
    for (int64_t i = 2; i < host_buf.dim(); ++i) {
        width *= host_buf.size(i);
    }
    meta[o + 6] = width;
    meta[o + 7] = layer_lo;
    meta[o + 8] = layer_hi;
}

void LaunchKvExchange(std::vector<int64_t> &meta, const at::Tensor &device_ref)
{
    const auto npuStream = c10_npu::getCurrentNPUStream();
    at::Tensor meta_dev = at::empty({kMinMetaNumel},
                                    at::TensorOptions().dtype(at::kLong).device(device_ref.device()));
    meta_dev.copy_(at::from_blob(static_cast<void *>(meta.data()), {kMinMetaNumel}, at::TensorOptions().dtype(at::kLong)));
    meta_dev.record_stream(npuStream);
    EXEC_KERNEL_CMD(kv_exchange, static_cast<uint32_t>(ResolveBlockDim()), meta_dev);
}

}  // namespace

/*
 * Low-level launch: `meta` is a device-resident int64 tensor laid out per
 * op_kernel/kv_exchange_kernel.cpp. Kept exposed for tests and debugging;
 * production callers use the transfer_kv_* wrappers below, which build the
 * meta from plain tensors.
 */
HOST_API void kv_exchange(const at::Tensor &meta, int64_t block_dim)
{
    CheckNpuTensor(meta, "meta");
    TORCH_CHECK(meta.scalar_type() == at::kLong, "meta must be int64, got ", meta.scalar_type());
    TORCH_CHECK(meta.dim() == 1, "meta must be 1-D, got ", meta.dim());
    TORCH_CHECK(meta.is_contiguous(), "meta must be contiguous");
    TORCH_CHECK(meta.numel() >= kMinMetaNumel, "meta must have at least ", kMinMetaNumel, " elements, got ",
                meta.numel());
    TORCH_CHECK(block_dim > 0, "block_dim must be positive, got ", block_dim);

    const auto npuStream = c10_npu::getCurrentNPUStream();
    meta.record_stream(npuStream);

    EXEC_KERNEL_CMD(kv_exchange, static_cast<uint32_t>(block_dim), meta);
}

/*
 * H2D, one layer: host (page-first, 5-D) -> device (layer-first, 5-D).
 * src_ptrs are HOST pool components (k/v/index_k/...), dst_ptrs the matching
 * DEVICE components. Host memory must be registered via aclrt_host_register;
 * the DVA resolution happens here, not on the caller side. Signatures mirror
 * sgl_kernel.kvcacheio (CUDA).
 */
HOST_API void transfer_kv_per_layer_direct_pf_lf(at::TensorList src_ptrs, at::TensorList dst_ptrs,
                                                 const at::Tensor &src_indices, const at::Tensor &dst_indices,
                                                 int64_t layer_id, int64_t page_size)
{
    TORCH_CHECK(!src_ptrs.empty() && src_ptrs.size() == dst_ptrs.size(),
                "src_ptrs and dst_ptrs must be non-empty lists of the same length");
    TORCH_CHECK(src_ptrs.size() <= kMaxComponents, "at most ", kMaxComponents, " components are supported, got ",
                src_ptrs.size());
    TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);
    TORCH_CHECK(layer_id >= 0, "layer_id must be non-negative, got ", layer_id);

    at::Tensor device_indices = dst_indices;
    at::Tensor host_indices = src_indices;
    if (device_indices.is_cpu()) {
        device_indices = device_indices.to(dst_ptrs[0].device());
    }
    if (host_indices.is_cpu()) {
        host_indices = host_indices.to(dst_ptrs[0].device());
    }
    CheckNpuTensor(device_indices, "dst_indices");
    CheckNpuTensor(host_indices, "src_indices");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "src/dst indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "indices size must be divisible by page_size");

    const int64_t num_pages = device_indices.numel() / page_size;
    if (num_pages == 0) {
        return;
    }

    std::vector<int64_t> meta(kMinMetaNumel, 0);
    meta[0] = static_cast<int64_t>(src_ptrs.size());
    meta[1] = num_pages;
    meta[2] = page_size;
    meta[3] = kDirectionH2D;
    meta[4] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_indices.data_ptr()));
    meta[5] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(host_indices.data_ptr()));
    meta[6] = kHostLayoutPageFirst;
    meta[7] = 0;  // start_dim: all launched cores participate

    for (size_t c = 0; c < src_ptrs.size(); ++c) {
        CheckComponentPair(dst_ptrs[c], src_ptrs[c], page_size, static_cast<int64_t>(c));
        TORCH_CHECK(layer_id < dst_ptrs[c].size(0), "layer_id ", layer_id, " out of range for component ", c,
                    " with ", dst_ptrs[c].size(0), " device layers");
        TORCH_CHECK(layer_id < src_ptrs[c].size(1), "layer_id ", layer_id, " out of range for component ", c,
                    " with ", src_ptrs[c].size(1), " host layers");
        FillComponent(meta, static_cast<int64_t>(c), dst_ptrs[c], src_ptrs[c], layer_id, layer_id + 1);
    }

    LaunchKvExchange(meta, dst_ptrs[0]);
    for (const auto &t : dst_ptrs) {
        t.record_stream(c10_npu::getCurrentNPUStream());
    }
    device_indices.record_stream(c10_npu::getCurrentNPUStream());
    host_indices.record_stream(c10_npu::getCurrentNPUStream());
}

/*
 * D2H, all layers: device (layer-first, 5-D) -> host (page-first, 5-D).
 * The per-component layer range is the device buffer's dim(0), so components
 * covering fewer layers than the KV pool (e.g. indexer k / fp32 scales) are
 * handled without extra mapping on the caller side.
 */
HOST_API void transfer_kv_all_layer_direct_lf_pf(at::TensorList src_ptrs, at::TensorList dst_ptrs,
                                                 const at::Tensor &src_indices, const at::Tensor &dst_indices,
                                                 int64_t page_size)
{
    TORCH_CHECK(!src_ptrs.empty() && src_ptrs.size() == dst_ptrs.size(),
                "src_ptrs and dst_ptrs must be non-empty lists of the same length");
    TORCH_CHECK(src_ptrs.size() <= kMaxComponents, "at most ", kMaxComponents, " components are supported, got ",
                src_ptrs.size());
    TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);

    at::Tensor device_indices = src_indices;
    at::Tensor host_indices = dst_indices;
    if (device_indices.is_cpu()) {
        device_indices = device_indices.to(src_ptrs[0].device());
    }
    if (host_indices.is_cpu()) {
        host_indices = host_indices.to(src_ptrs[0].device());
    }
    CheckNpuTensor(device_indices, "src_indices");
    CheckNpuTensor(host_indices, "dst_indices");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "src/dst indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "indices size must be divisible by page_size");

    const int64_t num_pages = device_indices.numel() / page_size;
    if (num_pages == 0) {
        return;
    }

    std::vector<int64_t> meta(kMinMetaNumel, 0);
    meta[0] = static_cast<int64_t>(src_ptrs.size());
    meta[1] = num_pages;
    meta[2] = page_size;
    meta[3] = kDirectionD2H;
    meta[4] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_indices.data_ptr()));
    meta[5] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(host_indices.data_ptr()));
    meta[6] = kHostLayoutPageFirst;
    meta[7] = 0;  // start_dim: all launched cores participate

    for (size_t c = 0; c < src_ptrs.size(); ++c) {
        CheckComponentPair(src_ptrs[c], dst_ptrs[c], page_size, static_cast<int64_t>(c));
        TORCH_CHECK(src_ptrs[c].size(0) <= dst_ptrs[c].size(1), "component ", c, ": device layer count ",
                    src_ptrs[c].size(0), " exceeds host layer count ", dst_ptrs[c].size(1));
        FillComponent(meta, static_cast<int64_t>(c), src_ptrs[c], dst_ptrs[c], 0, src_ptrs[c].size(0));
    }

    LaunchKvExchange(meta, src_ptrs[0]);
    for (const auto &t : src_ptrs) {
        t.record_stream(c10_npu::getCurrentNPUStream());
    }
    device_indices.record_stream(c10_npu::getCurrentNPUStream());
    host_indices.record_stream(c10_npu::getCurrentNPUStream());
}

}  // namespace npu_kernel
}  // namespace sglang
