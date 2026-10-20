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

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_kv_exchange.h"

#include <cstdint>

namespace sglang {
namespace npu_kernel {

namespace {

constexpr int64_t kMetaHeader = 8;
constexpr int64_t kMetaStride = 9;
constexpr int64_t kMaxComponents = 4;
constexpr int64_t kMinMetaNumel = kMetaHeader + kMetaStride * kMaxComponents;

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

}  // namespace

/*
 * HiCache L1 <-> L2 KV cache exchange driven entirely by device-resident
 * metadata. `meta` is an int64 device tensor laid out as:
 *
 *   [0] num_components (<= 4, k/v/index_k/index_k_scale order)
 *   [1] num_pages      [2] page_size
 *   [3] direction: 1 = H2D, 2 = D2H
 *   [4] device_indices_ptr (int64 token indices, device memory)
 *   [5] host_indices_ptr   (int64 token indices, device memory)
 *   [6] host_layout_mode: 0 = page_first, 1 = layer_first
 *   [7] reserved (must be 0)
 *   per component c (9 values starting at 8 + 9*c):
 *   [0] device_base  [1] host_base (device-visible host address)
 *   [2] device_layer_pitch  [3] device_page_pitch   (bytes)
 *   [4] host_page_pitch     [5] host_layer_pitch    (bytes)
 *   [6] width (bytes of one (page, layer) row)  [7] layer_lo  [8] layer_hi
 *
 * The host page addresses must be device-visible (registered host memory).
 * The caller owns all buffers referenced by `meta` and must keep them alive
 * until work on the current NPU stream completes.
 *
 * `block_dim` is the number of AIV cores launched; on Ascend950 the device
 * exposes 64 AIV cores and only cores >= 62 participate (see
 * KV_EXCHANGE_START_DIM in the kernel), so callers pass 64 there.
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

    auto npuStream = c10_npu::getCurrentNPUStream();
    meta.record_stream(npuStream);

    uint32_t blockDimU32 = static_cast<uint32_t>(block_dim);
    EXEC_KERNEL_CMD(kv_exchange, blockDimU32, meta);
}

}  // namespace npu_kernel
}  // namespace sglang
