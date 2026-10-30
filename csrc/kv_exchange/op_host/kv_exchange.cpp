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

#include "kv_exchange.h"

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
    // 4-D (layers, pages, page_size, 1) is accepted alongside the canonical
    // 5-D form: the fp32 Indexer scale caches carry no trailing width dim, so
    // the caller can hand them over without padding a singleton (width
    // resolves to 1 below, matching the padded form).
    TORCH_CHECK((device_buf.dim() == 5 || device_buf.dim() == 4) && (host_buf.dim() == 5 || host_buf.dim() == 4),
                "component ", which,
                ": buffers must be 4-D or 5-D -- device (layers, pages, page_size, [1,] width), host "
                "(pages, layers, page_size, [1,] width), got device ", device_buf.dim(), "-D / host ",
                host_buf.dim(), "-D");
    TORCH_CHECK(device_buf.size(2) == page_size, "component ", which, ": device dim(2) must equal page_size");
    TORCH_CHECK(host_buf.size(2) == page_size, "component ", which, ": host dim(2) must equal page_size");
    // Pool page counts deliberately differ in production and are NOT checked:
    // the host pool is sized by --hicache-ratio (typically 2x the device pool)
    // and the H2D device side is a staging buffer sized to the batch, so
    // host pages >= device pages never holds an equality. The kernel only
    // dereferences pages listed in the token indices; index range safety is
    // the caller's contract (indices are generated per-pool-capacity).
    TORCH_CHECK(device_buf.size(3) == host_buf.size(3), "component ", which,
                ": device and host singleton dims (dim 3) must match");
    TORCH_CHECK((device_buf.dim() == host_buf.dim()) && (device_buf.dim() == 4 ||
                device_buf.size(4) * device_buf.element_size() == host_buf.size(4) * host_buf.element_size()),
                "component ", which, ": device and host packed row widths must match");
    TORCH_CHECK(TrailingDimsPacked(device_buf) && TrailingDimsPacked(host_buf), "component ", which,
                ": trailing (page_size, 1, width) dims must be packed (contiguous within a "
                "(page, layer) block)");
}

// meta layout (see op_kernel/kv_exchange_kernel.cpp):
//   [0] num_components [1] num_pages [2] page_size [3] direction
//   [4] device_indices_ptr [5] host_indices_ptr [6] host_layout_mode [7] start_dim
//   per component c at 8 + 9*c: dev_base, host_base(DVA), dev_layer_pitch,
//   dev_page_pitch, host_page_pitch, host_layer_pitch, width, layer_lo, layer_hi
//
// The device layer offset is folded into dev_base and the host layer offset
// into the host DVA, so the meta's layer range is normalized to start at 0
// and the device kernel's addressing format stays untouched. Components with
// layer_lo >= layer_hi (or width <= 0) transfer nothing.
void FillComponent(std::vector<int64_t> &meta, int64_t c, const at::Tensor &device_buf,
                   const at::Tensor &host_buf, int64_t device_layer_offset, int64_t host_dva,
                   int64_t host_layer_offset, int64_t layer_lo, int64_t layer_hi)
{
    const int64_t esz = device_buf.element_size();
    const int64_t o = kMetaHeader + kMetaStride * c;
    meta[o + 0] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_buf.data_ptr())) +
                  device_layer_offset * device_buf.stride(0) * esz;
    meta[o + 1] = host_dva + host_layer_offset * host_buf.stride(1) * esz;
    meta[o + 2] = device_buf.stride(0) * esz;  // device_layer_pitch
    meta[o + 3] = device_buf.stride(1) * esz;  // device_page_pitch
    meta[o + 4] = host_buf.stride(0) * esz;     // host_page_pitch
    meta[o + 5] = host_buf.stride(1) * esz;     // host_layer_pitch
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
    const uint32_t block_dim_u32 = ResolveBlockDim();
    EXEC_KERNEL_CMD(kv_exchange, block_dim_u32, meta_dev);
}

struct ComponentArgs {
    at::TensorList device_components;
    at::TensorList host_components;
    const std::vector<int64_t> &host_dvas;

    int64_t size() const { return static_cast<int64_t>(device_components.size()); }
};

void CheckCommonArgs(const ComponentArgs &args, const at::Tensor &device_indices,
                     const at::Tensor &host_indices, int64_t page_size)
{
    TORCH_CHECK(!args.device_components.empty() && args.device_components.size() == args.host_components.size(),
                "device_components and host_components must be non-empty lists of the same length");
    TORCH_CHECK(args.device_components.size() <= kMaxComponents, "at most ", kMaxComponents,
                " components are supported, got ", args.device_components.size());
    TORCH_CHECK(args.host_dvas.size() == args.device_components.size(), "host_dvas must match the component count, got ",
                args.host_dvas.size(), " vs ", args.device_components.size());
    TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);

    CheckNpuTensor(device_indices, "device_indices");
    CheckNpuTensor(host_indices, "host_indices");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device/host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "indices size must be divisible by page_size");
    for (int64_t c = 0; c < args.size(); ++c) {
        CheckNpuTensor(args.device_components[c], "device_components");
    }
}

// Upload is the caller's (python) job; only the async-use lifetime is
// recorded here, on the stream the kernel launches on.
void RecordStreamLifetime(const ComponentArgs &args, const at::Tensor &device_indices,
                          const at::Tensor &host_indices)
{
    const auto npuStream = c10_npu::getCurrentNPUStream();
    for (const auto &t : args.device_components) {
        t.record_stream(npuStream);
    }
    device_indices.record_stream(npuStream);
    host_indices.record_stream(npuStream);
}

}  // namespace

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

    // EXEC_KERNEL_CMD substitutes blockdim into a lambda capture list, so it
    // must be a plain identifier, not a cast expression.
    const uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
    EXEC_KERNEL_CMD(kv_exchange, block_dim_u32, meta);
}

/*
 * H2D, one layer: host page_first_kv_split -> device layer_first_kv_split.
 * The python layer hands over full host pool components with their MF DVAs;
 * the per-component device and host layer ids may differ (e.g. an MTP draft
 * pool's layer 0 against the packed host layer dim), so both offsets are
 * folded into the base addresses and the meta layer range is [0, 1).
 */
HOST_API void transfer_kv_per_layer_kv_split_pf_lf(const at::Tensor &device_indices, const at::Tensor &host_indices,
                                                   at::TensorList device_components, at::TensorList host_components,
                                                   const std::vector<int64_t> &host_dvas,
                                                   const std::vector<int64_t> &device_layer_ids,
                                                   const std::vector<int64_t> &host_layer_ids, int64_t page_size)
{
    const ComponentArgs args{device_components, host_components, host_dvas};
    CheckCommonArgs(args, device_indices, host_indices, page_size);
    TORCH_CHECK(device_layer_ids.size() == args.host_components.size(), "device_layer_ids must match the component count, got ",
                device_layer_ids.size(), " vs ", args.host_components.size());
    TORCH_CHECK(host_layer_ids.size() == args.host_components.size(), "host_layer_ids must match the component count, got ",
                host_layer_ids.size(), " vs ", args.host_components.size());

    const int64_t num_pages = device_indices.numel() / page_size;
    if (num_pages == 0) {
        return;
    }

    std::vector<int64_t> meta(kMinMetaNumel, 0);
    meta[0] = args.size();
    meta[1] = num_pages;
    meta[2] = page_size;
    meta[3] = kDirectionH2D;
    meta[4] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_indices.data_ptr()));
    meta[5] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(host_indices.data_ptr()));
    meta[6] = kHostLayoutPageFirst;
    meta[7] = 0;  // start_dim: all launched cores participate

    for (int64_t c = 0; c < args.size(); ++c) {
        const at::Tensor &device_buf = args.device_components[c];
        const at::Tensor &host_buf = args.host_components[c];
        const int64_t layer_id = device_layer_ids[c];
        const int64_t host_layer_id = host_layer_ids[c];
        CheckComponentPair(device_buf, host_buf, page_size, c);
        TORCH_CHECK(layer_id >= 0 && layer_id < device_buf.size(0), "component ", c, ": device layer id ", layer_id,
                    " out of range for ", device_buf.size(0), " device layers");
        TORCH_CHECK(host_layer_id >= 0 && host_layer_id < host_buf.size(1), "component ", c, ": host layer id ",
                    host_layer_id, " out of range for ", host_buf.size(1), " host layers");
        TORCH_CHECK(host_dvas[c] != 0, "component ", c, ": MF host DVA must be non-zero");
        FillComponent(meta, c, device_buf, host_buf, layer_id, host_dvas[c], host_layer_id, 0, 1);
    }

    LaunchKvExchange(meta, args.device_components[0]);
    RecordStreamLifetime(args, device_indices, host_indices);
}

/*
 * D2H, all layers: device layer_first_kv_split -> host page_first_kv_split.
 * Each component transfers its own device dim(0) layers; the host layers land
 * at [host_layer_start, host_layer_start + device layers) of the full host
 * pool, so the host offset is folded into the DVA and the meta layer range is
 * [0, device layers).
 */
HOST_API void transfer_kv_all_layer_kv_split_lf_pf(const at::Tensor &device_indices, const at::Tensor &host_indices,
                                                   at::TensorList device_components, at::TensorList host_components,
                                                   const std::vector<int64_t> &host_dvas, int64_t page_size,
                                                   int64_t host_layer_start)
{
    const ComponentArgs args{device_components, host_components, host_dvas};
    CheckCommonArgs(args, device_indices, host_indices, page_size);
    TORCH_CHECK(host_layer_start >= 0, "host_layer_start must be non-negative, got ", host_layer_start);

    const int64_t num_pages = device_indices.numel() / page_size;
    if (num_pages == 0) {
        return;
    }

    std::vector<int64_t> meta(kMinMetaNumel, 0);
    meta[0] = args.size();
    meta[1] = num_pages;
    meta[2] = page_size;
    meta[3] = kDirectionD2H;
    meta[4] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_indices.data_ptr()));
    meta[5] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(host_indices.data_ptr()));
    meta[6] = kHostLayoutPageFirst;
    meta[7] = 0;  // start_dim: all launched cores participate

    for (int64_t c = 0; c < args.size(); ++c) {
        const at::Tensor &device_buf = args.device_components[c];
        const at::Tensor &host_buf = args.host_components[c];
        const int64_t device_layers = device_buf.size(0);
        CheckComponentPair(device_buf, host_buf, page_size, c);
        TORCH_CHECK(host_layer_start + device_layers <= host_buf.size(1), "component ", c, ": host layer range [",
                    host_layer_start, ", ", host_layer_start + device_layers, ") exceeds host layer count ",
                    host_buf.size(1));
        TORCH_CHECK(host_dvas[c] != 0, "component ", c, ": MF host DVA must be non-zero");
        FillComponent(meta, c, device_buf, host_buf, 0, host_dvas[c], host_layer_start, 0, device_layers);
    }

    LaunchKvExchange(meta, args.device_components[0]);
    RecordStreamLifetime(args, device_indices, host_indices);
}

}  // namespace npu_kernel
}  // namespace sglang
