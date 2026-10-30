/*!
 * \file kv_exchange.h
 * \brief MF+AIV kv_split transfer host-side declarations
 *
 * MLA HiCache L2<->L1 IO between a host page_first_kv_split pool
 * (pages, layers, page_size, 1, width) and a device layer_first_kv_split
 * pool (layers, pages, page_size, 1, width). The host components are full
 * pool tensors whose base addresses arrive together with their MF DVAs;
 * layer offsets are folded into the base addresses here, and the 44-slot
 * kv_exchange meta is built with a zero-based layer range so the device
 * kernel's addressing format is unchanged.
 */

#ifndef SGL_KERNEL_NPU_KV_EXCHANGE_HOST_H_
#define SGL_KERNEL_NPU_KV_EXCHANGE_HOST_H_

#include <ATen/ATen.h>
#include <vector>

#include "defines.h"

namespace sglang {
namespace npu_kernel {

// Low-level launch: `meta` is a device-resident int64 tensor laid out per
// op_kernel/kv_exchange_kernel.cpp. Exposed for tests and debugging; the
// production transfer ops below build the meta from plain tensors.
HOST_API void kv_exchange(const at::Tensor &meta, int64_t block_dim);

// H2D, one layer: host page_first_kv_split -> device layer_first_kv_split.
// device_components[i] / host_components[i] are paired full-pool tensors;
// host_dvas[i] is the MF device-visible address of host_components[i]'s
// base; device_layer_ids[i] / host_layer_ids[i] select each component's
// single layer on the device and host sides independently.
HOST_API void transfer_kv_per_layer_kv_split_pf_lf(
    const at::Tensor &device_indices, const at::Tensor &host_indices,
    at::TensorList device_components, at::TensorList host_components,
    const std::vector<int64_t> &host_dvas, const std::vector<int64_t> &device_layer_ids,
    const std::vector<int64_t> &host_layer_ids, int64_t page_size);

// D2H, all layers: device layer_first_kv_split -> host page_first_kv_split.
// Each component transfers its own device dim(0) layers, starting at host
// layer host_layer_start of the full host pool.
HOST_API void transfer_kv_all_layer_kv_split_lf_pf(
    const at::Tensor &device_indices, const at::Tensor &host_indices,
    at::TensorList device_components, at::TensorList host_components,
    const std::vector<int64_t> &host_dvas, int64_t page_size, int64_t host_layer_start);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // SGL_KERNEL_NPU_KV_EXCHANGE_HOST_H_
