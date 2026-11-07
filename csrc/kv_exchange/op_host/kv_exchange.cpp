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

#include "event_pool.h"

#include "aclrtlaunch_kv_exchange.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr int64_t kMaxComponents = 4;

constexpr int64_t kDirectionH2D = 1;
constexpr int64_t kDirectionD2H = 2;
constexpr int64_t kHostLayoutPageFirst = 0;
constexpr int64_t kHostLayoutLayerFirst = 1;

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

// First participating AIV core index. The default (-1) selects the native
// acc_offload kv_exchange occupancy: only the last 2 AIV cores copy, leaving
// the rest of the cores free for concurrent compute on the main stream, which
// is what lets the per-layer H2D overlap attention. Explicit values (e.g. 0
// for all cores) override for perf sweeps.
int64_t ResolveStartDim()
{
    if (const char *env = std::getenv("SGLANG_KV_EXCHANGE_START_DIM")) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed >= 0) {
            return static_cast<int64_t>(parsed);
        }
    }
    return -1;
}

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

// The kernel walks (page, layer) blocks addressed as
// base + layer*layer_pitch + page*page_pitch and copies `width` contiguous
// bytes from each block, so only the trailing (page_size, 1, width) dims must
// be packed; page/layer dims may be strided views (e.g. transposed host
// layer-first pools).
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

// Mirrors the Python _check_component contract: matching dtypes, page shapes
// and packed trailing dims. Pool page counts deliberately differ in
// production (the host pool is sized by --hicache-ratio) and are NOT checked;
// the kernel only dereferences pages listed in the token indices.
void CheckComponentPair(const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t page_size, int64_t which)
{
    TORCH_CHECK(device_buf.scalar_type() == host_buf.scalar_type(), "component ", which,
                ": device and host dtypes must match, got ", device_buf.scalar_type(), " vs ", host_buf.scalar_type());
    // 4-D (layers, pages, page_size, 1) is accepted alongside the canonical
    // 5-D form: the fp32 Indexer scale caches carry no trailing width dim, so
    // the caller can hand them over without padding a singleton (width
    // resolves to 1 below, matching the padded form).
    TORCH_CHECK((device_buf.dim() == 5 || device_buf.dim() == 4) && (host_buf.dim() == 5 || host_buf.dim() == 4),
                "component ", which,
                ": buffers must be 4-D or 5-D -- device (layers, pages, page_size, [1,] width), host "
                "(pages, layers, page_size, [1,] width), got device ",
                device_buf.dim(), "-D / host ", host_buf.dim(), "-D");
    TORCH_CHECK(device_buf.size(2) == page_size, "component ", which, ": device dim(2) must equal page_size");
    TORCH_CHECK(host_buf.size(2) == page_size, "component ", which, ": host dim(2) must equal page_size");
    TORCH_CHECK(device_buf.size(3) == host_buf.size(3), "component ", which,
                ": device and host singleton dims (dim 3) must match");
    TORCH_CHECK((device_buf.dim() == host_buf.dim()) &&
                    (device_buf.dim() == 4 ||
                     device_buf.size(4) * device_buf.element_size() == host_buf.size(4) * host_buf.element_size()),
                "component ", which, ": device and host packed row widths must match");
    TORCH_CHECK(TrailingDimsPacked(device_buf) && TrailingDimsPacked(host_buf), "component ", which,
                ": trailing (page_size, 1, width) dims must be packed (contiguous within a "
                "(page, layer) block)");
}

// Transfer-description layout (see op_kernel/kv_exchange_kernel.cpp): the
// production kv_exchange kernel takes the description as flat launch
// parameters -- 8 header scalars plus 4 x 9 component scalars in the same
// field order as the legacy device-resident meta slots.
//
// The component views arrive with their layer offsets already folded into
// their base addresses (device view slicing / host DVA of the sliced view),
// so the layer range starts at 0 and the device kernel's addressing format
// stays untouched. Components with layer_lo >= layer_hi (or width <= 0)
// transfer nothing.
struct ComponentParams {
    int64_t fields[9];
};

#define KV_COMPONENT_ARGS(c)                                                                   \
    comps[c].fields[0], comps[c].fields[1], comps[c].fields[2], comps[c].fields[3],            \
        comps[c].fields[4], comps[c].fields[5], comps[c].fields[6], comps[c].fields[7],         \
        comps[c].fields[8]

void FillComponent(ComponentParams &cp, const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t host_dva,
                   int64_t layer_lo, int64_t layer_hi)
{
    const int64_t esz = device_buf.element_size();
    cp.fields[0] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_buf.data_ptr()));
    cp.fields[1] = host_dva;
    cp.fields[2] = device_buf.stride(0) * esz;  // device_layer_pitch
    cp.fields[3] = device_buf.stride(1) * esz;  // device_page_pitch
    cp.fields[4] = host_buf.stride(0) * esz;    // host_page_pitch
    cp.fields[5] = host_buf.stride(1) * esz;    // host_layer_pitch
    // The kernel copies this many contiguous bytes per (page, layer) block:
    // the packed extent of the trailing dims (prod(shape[2:]) * itemsize).
    int64_t width = esz;
    for (int64_t i = 2; i < host_buf.dim(); ++i) {
        width *= host_buf.size(i);
    }
    cp.fields[6] = width;
    cp.fields[7] = layer_lo;
    cp.fields[8] = layer_hi;
}

// Per-pool component geometry for the fused range submission: full-tensor
// bases and pitches, so each iteration only folds its layer offsets in.
struct RangeComponent {
    bool active = false;
    int64_t dev_base = 0;         // device buffer base address
    int64_t dev_layer_pitch = 0;  // bytes
    int64_t dev_page_pitch = 0;   // bytes
    int64_t host_dva = 0;
    int64_t host_page_pitch = 0;  // bytes
    int64_t host_layer_pitch = 0;  // bytes
    int64_t width = 0;            // packed bytes of one (page, layer) row
};

// Build the geometry from one paired full view. Empty (numel == 0) views
// stay inactive and never join a launch.
void FillRangeComponent(RangeComponent &rc, const at::Tensor &device_buf, const at::Tensor &host_buf, int64_t dva,
                        int64_t page_size, int64_t pool, const char *name)
{
    if (!device_buf.defined() || !host_buf.defined() || !device_buf.numel() || !host_buf.numel()) {
        return;
    }
    CheckNpuTensor(device_buf, "range pool component");
    TORCH_CHECK(host_buf.device().type() == at::kCPU, "range pool ", pool, ": host ", name,
                " component must be a CPU (MemFabric-mapped) tensor");
    CheckComponentPair(device_buf, host_buf, page_size, pool);
    TORCH_CHECK(dva != 0, "range pool ", pool, ": host ", name, " component DVA must be non-zero");
    const int64_t esz = device_buf.element_size();
    rc.dev_base = static_cast<int64_t>(reinterpret_cast<uintptr_t>(device_buf.data_ptr()));
    rc.dev_layer_pitch = device_buf.stride(0) * esz;
    rc.dev_page_pitch = device_buf.stride(1) * esz;
    rc.host_dva = dva;
    rc.host_page_pitch = host_buf.stride(0) * esz;
    rc.host_layer_pitch = host_buf.stride(1) * esz;
    int64_t width = esz;
    for (int64_t i = 2; i < host_buf.dim(); ++i) {
        width *= host_buf.size(i);
    }
    rc.width = width;
    rc.active = true;
}

// Pack one (device_layer, host_layer) iteration of a component into the
// launch parameter block: same field layout as FillComponent with the layer
// offsets folded into the bases instead of the views.
void PackRangeComponent(ComponentParams &cp, const RangeComponent &rc, int64_t device_layer, int64_t host_layer)
{
    cp.fields[0] = rc.dev_base + device_layer * rc.dev_layer_pitch;
    cp.fields[1] = rc.host_dva + host_layer * rc.host_layer_pitch;
    cp.fields[2] = rc.dev_layer_pitch;
    cp.fields[3] = rc.dev_page_pitch;
    cp.fields[4] = rc.host_page_pitch;
    cp.fields[5] = rc.host_layer_pitch;
    cp.fields[6] = rc.width;
    cp.fields[7] = 0;
    cp.fields[8] = 1;
}

// Row offset of pool p's iterations inside the flat iter_table.
int64_t RowOffset(const std::vector<int64_t> &pool_iter_counts, int64_t pool)
{
    int64_t offset = 0;
    for (int64_t i = 0; i < pool; ++i) {
        offset += pool_iter_counts[static_cast<size_t>(i)];
    }
    return offset;
}

// Index uploads are the caller's (python) job; only the async-use lifetime
// is recorded here, on the stream the kernel launches on.
void RecordStreamLifetime(at::TensorList device_components, const at::Tensor &device_indices,
                          const at::Tensor &host_indices)
{
    const auto npuStream = c10_npu::getCurrentNPUStream();
    for (const auto &t : device_components) {
        t.record_stream(npuStream);
    }
    device_indices.record_stream(npuStream);
    host_indices.record_stream(npuStream);
}

}  // namespace

/*
 * One AIV kv_exchange submission from the split-KV component list.
 *
 * The Python layer (implementation/memfabric/mem_cache/kv_transfer.py)
 * selects the paired component views, uploads the token indices and resolves
 * each host view's MF DVA; this adapter packs the same component/pitch/
 * width/layer description into the kernel launch parameter block, so the
 * description never round-trips through device memory: no device-resident
 * meta, no pinned host staging and no separate H2D meta upload.
 */
HOST_API void kv_exchange_submit(const at::Tensor &device_indices, const at::Tensor &host_indices,
                                 at::TensorList device_components, at::TensorList host_components,
                                 const std::vector<int64_t> &host_dvas, int64_t page_size, int64_t direction,
                                 int64_t host_layout_mode)
{
    TORCH_CHECK(direction == kDirectionH2D || direction == kDirectionD2H, "direction must be H2D (1) or D2H (2), got ",
                direction);
    TORCH_CHECK(host_layout_mode == kHostLayoutPageFirst || host_layout_mode == kHostLayoutLayerFirst,
                "host_layout_mode must be page_first (0) or layer_first (1), got ", host_layout_mode);
    TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);
    TORCH_CHECK(!device_components.empty() && device_components.size() == host_components.size(),
                "device_components and host_components must be non-empty lists of the same length");
    TORCH_CHECK(device_components.size() <= kMaxComponents, "at most ", kMaxComponents,
                " components are supported, got ", device_components.size());
    TORCH_CHECK(host_dvas.size() == device_components.size(), "host_dvas must match the component count, got ",
                host_dvas.size(), " vs ", device_components.size());

    CheckNpuTensor(device_indices, "device_indices");
    CheckNpuTensor(host_indices, "host_indices");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device/host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "indices size must be divisible by page_size");

    const int64_t num_pages = device_indices.numel() / page_size;
    if (num_pages == 0) {
        // Nothing to transfer.
        return;
    }

    ComponentParams comps[kMaxComponents] = {};
    for (int64_t c = 0; c < static_cast<int64_t>(device_components.size()); ++c) {
        const at::Tensor &device_buf = device_components[c];
        const at::Tensor &host_buf = host_components[c];
        CheckNpuTensor(device_buf, "device_components");
        TORCH_CHECK(host_buf.device().type() == at::kCPU, "component ", c,
                    ": host components must be CPU (MemFabric-mapped) tensors");
        CheckComponentPair(device_buf, host_buf, page_size, c);
        TORCH_CHECK(host_dvas[static_cast<size_t>(c)] != 0, "component ", c, ": MF host DVA must be non-zero");
        // Layer offsets are already folded into the component views; the
        // whole view (all its layers) is transferred.
        FillComponent(comps[c], device_buf, host_buf, host_dvas[static_cast<size_t>(c)], 0, device_buf.size(0));
    }

    // start_dim: negative (the default) reserves the last 2 AIV cores to
    // match the native acc_offload kv_exchange occupancy; SGLANG_KV_
    // EXCHANGE_START_DIM=0 re-engages all launched cores for perf sweeps.
    const int64_t start_dim = ResolveStartDim();
    // EXEC_KERNEL_CMD requires plain identifier arguments (blockdim goes into
    // a lambda capture list, and the rest must bind to ConvertTypes' lvalue
    // references), so hoist every runtime value into named locals.
    const int64_t num_components = static_cast<int64_t>(device_components.size());
    const int64_t num_pages_local = num_pages;
    const int64_t page_size_local = page_size;
    const int64_t direction_local = direction;
    const int64_t host_layout_local = host_layout_mode;
    const uint32_t block_dim_u32 = ResolveBlockDim();
    EXEC_KERNEL_CMD(kv_exchange, block_dim_u32, device_indices, host_indices, num_components, num_pages_local,
                    page_size_local, direction_local, host_layout_local, start_dim, KV_COMPONENT_ARGS(0),
                    KV_COMPONENT_ARGS(1), KV_COMPONENT_ARGS(2), KV_COMPONENT_ARGS(3));
    RecordStreamLifetime(device_components, device_indices, host_indices);
}

/*
 * Fused per-layer H2D submission (see kv_exchange.h). One Python call covers
 * every controller layer of every pool: the per-layer "launch kv_exchange,
 * then record the layer-done event" sequence runs as a C++ loop, so the
 * per-layer Python dispatch overhead disappears while the per-layer event
 * granularity -- and therefore the implicit forward/transfer pipeline --
 * stays bit-identical to the Python loop.
 */
HOST_API void kv_exchange_submit_range(
    const at::Tensor &device_indices, const at::Tensor &host_indices, at::TensorList device_k,
    at::TensorList host_k, at::TensorList device_v, at::TensorList host_v, at::TensorList device_index_k,
    at::TensorList host_index_k, at::TensorList device_index_scale, at::TensorList host_index_scale,
    const std::vector<int64_t> &host_dvas, int64_t page_size, int64_t direction, int64_t host_layout_mode,
    int64_t num_ctrl_layers, const std::vector<int64_t> &iter_table,
    const std::vector<int64_t> &pool_iter_counts, int64_t event_pool_id, int64_t event_first_index)
{
    constexpr int64_t kMaxPools = 4;
    TORCH_CHECK(direction == kDirectionH2D, "range submission is H2D only, got direction ", direction);
    TORCH_CHECK(host_layout_mode == kHostLayoutPageFirst || host_layout_mode == kHostLayoutLayerFirst,
                "host_layout_mode must be page_first (0) or layer_first (1), got ", host_layout_mode);
    TORCH_CHECK(page_size > 0, "page_size must be positive, got ", page_size);
    TORCH_CHECK(num_ctrl_layers > 0, "num_ctrl_layers must be positive, got ", num_ctrl_layers);
    const int64_t num_pools = static_cast<int64_t>(pool_iter_counts.size());
    TORCH_CHECK(num_pools > 0 && num_pools <= kMaxPools, "pool count must be in [1, ", kMaxPools, "], got ",
                num_pools);
    TORCH_CHECK(device_k.size() == host_k.size() && static_cast<int64_t>(device_k.size()) == num_pools,
                "device_k/host_k must hold one full buffer per pool");
    TORCH_CHECK(device_v.size() == host_v.size() && static_cast<int64_t>(device_v.size()) == num_pools,
                "device_v/host_v must hold one buffer per pool");
    TORCH_CHECK(device_index_k.size() == host_index_k.size() && static_cast<int64_t>(device_index_k.size()) ==
                                                                             num_pools,
                "device_index_k/host_index_k must hold one buffer per pool");
    TORCH_CHECK(device_index_scale.size() == host_index_scale.size() &&
                    static_cast<int64_t>(device_index_scale.size()) == num_pools,
                "device_index_scale/host_index_scale must hold one buffer per pool");
    TORCH_CHECK(host_dvas.size() == static_cast<size_t>(4 * num_pools),
                "host_dvas must hold 4 DVAs (k, v, index_k, index_scale) per pool, got ", host_dvas.size());
    int64_t total_iters = 0;
    for (int64_t p = 0; p < num_pools; ++p) {
        TORCH_CHECK(pool_iter_counts[static_cast<size_t>(p)] >= 0, "pool ", p,
                    ": iteration count must be nonnegative");
        total_iters += pool_iter_counts[static_cast<size_t>(p)];
    }
    TORCH_CHECK(static_cast<int64_t>(iter_table.size()) == 4 * total_iters,
                "iter_table must hold 4 entries per pool iteration, got ", iter_table.size(), " for ",
                total_iters, " iterations");
    TORCH_CHECK(event_pool_id < 0 || event_first_index >= 0, "event_first_index must be nonnegative");

    CheckNpuTensor(device_indices, "device_indices");
    CheckNpuTensor(host_indices, "host_indices");
    TORCH_CHECK(device_indices.numel() == host_indices.numel(), "device/host indices must have the same length");
    TORCH_CHECK(device_indices.numel() % page_size == 0, "indices size must be divisible by page_size");
    const int64_t num_pages = device_indices.numel() / page_size;

    // Resolve the per-pool component geometry once; per-iteration packing then
    // only folds layer offsets into the bases.
    RangeComponent comps[kMaxPools][kMaxComponents] = {};
    for (int64_t p = 0; p < num_pools; ++p) {
        const size_t sp = static_cast<size_t>(p);
        const int64_t dv = host_dvas[static_cast<size_t>(4 * p + 1)];
        const int64_t dik = host_dvas[static_cast<size_t>(4 * p + 2)];
        const int64_t dis = host_dvas[static_cast<size_t>(4 * p + 3)];
        FillRangeComponent(comps[p][0], device_k[sp], host_k[sp], host_dvas[static_cast<size_t>(4 * p)], page_size,
                           p, "k");
        TORCH_CHECK(comps[p][0].active, "range pool ", p, ": K component must be nonempty");
        FillRangeComponent(comps[p][1], device_v[sp], host_v[sp], dv, page_size, p, "v");
        FillRangeComponent(comps[p][2], device_index_k[sp], host_index_k[sp], dik, page_size, p, "index_k");
        FillRangeComponent(comps[p][3], device_index_scale[sp], host_index_scale[sp], dis, page_size, p,
                           "index_scale");
        // Paired Indexer components must arrive together, mirroring the
        // per-layer select_components contract.
        TORCH_CHECK(comps[p][2].active == comps[p][3].active, "range pool ", p,
                    ": index_k and index_scale must be supplied together");
    }

    // Validate the iteration table against the pool geometries up front so
    // the launch loop below cannot produce out-of-range addresses.
    for (int64_t p = 0; p < num_pools; ++p) {
        const int64_t count = pool_iter_counts[static_cast<size_t>(p)];
        const int64_t offset = RowOffset(pool_iter_counts, p);
        int64_t prev_ctrl = -1;
        for (int64_t i = 0; i < count; ++i) {
            const int64_t *row = &iter_table[static_cast<size_t>(4 * (offset + i))];
            const int64_t ctrl = row[0], dev_layer = row[1], host_layer = row[2], slot = row[3];
            TORCH_CHECK(ctrl > prev_ctrl, "range pool ", p, ": iteration rows must be sorted by ctrl layer");
            prev_ctrl = ctrl;
            TORCH_CHECK(ctrl >= 0 && ctrl < num_ctrl_layers, "range pool ", p, ": ctrl layer ", ctrl,
                        " out of [0, ", num_ctrl_layers, ")");
            TORCH_CHECK(dev_layer >= 0 && dev_layer < static_cast<int64_t>(device_k[static_cast<size_t>(p)].size(0)),
                        "range pool ", p, ": device layer ", dev_layer, " out of range");
            TORCH_CHECK(host_layer >= 0 && host_layer < static_cast<int64_t>(host_k[static_cast<size_t>(p)].size(1)),
                        "range pool ", p, ": host layer ", host_layer, " out of range");
            if (slot >= 0) {
                TORCH_CHECK(comps[p][2].active && comps[p][3].active, "range pool ", p,
                            ": index slot ", slot, " supplied without index components");
                TORCH_CHECK(slot < static_cast<int64_t>(device_index_k[static_cast<size_t>(p)].size(0)),
                            "range pool ", p, ": index slot ", slot, " out of range");
                TORCH_CHECK(slot < static_cast<int64_t>(host_index_k[static_cast<size_t>(p)].size(1)),
                            "range pool ", p, ": index host slot ", slot, " out of range");
            }
        }
    }

    // Hoisted launch parameters (EXEC_KERNEL_CMD needs named lvalues).
    const int64_t num_pages_local = num_pages;
    const int64_t page_size_local = page_size;
    const int64_t direction_local = direction;
    const int64_t host_layout_local = host_layout_mode;
    const int64_t start_dim = ResolveStartDim();
    const uint32_t block_dim_u32 = ResolveBlockDim();
    const auto npuStream = c10_npu::getCurrentNPUStream();

    // One cursor per pool over its (ctrl-sorted) iteration rows.
    int64_t cursor[kMaxPools] = {0};
    for (int64_t ctrl = 0; ctrl < num_ctrl_layers; ++ctrl) {
        if (num_pages > 0) {
            for (int64_t p = 0; p < num_pools; ++p) {
                const int64_t offset = RowOffset(pool_iter_counts, p);
                const int64_t count = pool_iter_counts[static_cast<size_t>(p)];
                if (cursor[p] >= count) {
                    continue;
                }
                const int64_t *row = &iter_table[static_cast<size_t>(4 * (offset + cursor[p]))];
                if (row[0] != ctrl) {
                    continue;
                }
                ++cursor[p];
                // Same component contract as the per-layer path: K/V always
                // ride along, the Indexer pair only for iterations with a
                // slot. Inactive components never join the launch.
                ComponentParams launch[kMaxComponents] = {};
                int64_t active = 0;
                for (int64_t c = 0; c < kMaxComponents; ++c) {
                    if (!comps[p][c].active || (c >= 2 && row[3] < 0)) {
                        continue;
                    }
                    // Index components select their row by the iteration's
                    // index slot; K/V select it by the device/host layers.
                    const int64_t layer = (c >= 2) ? row[3] : row[1];
                    const int64_t host_layer = (c >= 2) ? row[3] : row[2];
                    PackRangeComponent(launch[active], comps[p][c], layer, host_layer);
                    ++active;
                }
                const int64_t launch_components = active;
                EXEC_KERNEL_CMD(kv_exchange, block_dim_u32, device_indices, host_indices, launch_components,
                                num_pages_local, page_size_local, direction_local, host_layout_local, start_dim,
                                launch[0].fields[0], launch[0].fields[1], launch[0].fields[2], launch[0].fields[3],
                                launch[0].fields[4], launch[0].fields[5], launch[0].fields[6], launch[0].fields[7],
                                launch[0].fields[8], launch[1].fields[0], launch[1].fields[1], launch[1].fields[2],
                                launch[1].fields[3], launch[1].fields[4], launch[1].fields[5], launch[1].fields[6],
                                launch[1].fields[7], launch[1].fields[8], launch[2].fields[0], launch[2].fields[1],
                                launch[2].fields[2], launch[2].fields[3], launch[2].fields[4], launch[2].fields[5],
                                launch[2].fields[6], launch[2].fields[7], launch[2].fields[8], launch[3].fields[0],
                                launch[3].fields[1], launch[3].fields[2], launch[3].fields[3], launch[3].fields[4],
                                launch[3].fields[5], launch[3].fields[6], launch[3].fields[7], launch[3].fields[8]);
            }
        }
        // Events fire for every controller layer -- including layers no pool
        // covers -- mirroring the Python loop's per-layer on_layer_done.
        if (event_pool_id >= 0) {
            kv_exchange_event_pool_record(event_pool_id, event_first_index + ctrl);
        }
    }

    // One record_stream pass covers every launch above: all launches share
    // the current stream, and stream-ordered reuse only needs one mark.
    for (int64_t p = 0; p < num_pools; ++p) {
        const size_t sp = static_cast<size_t>(p);
        for (const auto *list : {&device_k, &device_v, &device_index_k, &device_index_scale}) {
            (*list)[sp].record_stream(npuStream);
        }
    }
    device_indices.record_stream(npuStream);
    host_indices.record_stream(npuStream);
}

}  // namespace npu_kernel
}  // namespace sglang
