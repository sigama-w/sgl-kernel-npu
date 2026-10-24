#!/bin/bash
# Build and install the kv_exchange AICPU device kernel (sgl-kernel-npu).
#
# Usage (on the NPU device, needs ASCEND_HOME_PATH):
#   bash scripts/build_kv_exchange_aicpu_run.sh            # build only
#   bash scripts/build_kv_exchange_aicpu_run.sh install    # build + install into CANN cust dirs
#
# Artifacts (in this script's build dir: csrc/hicache_offload/kv_exchange_aicpu/op_kernel/build):
#   libcann_kv_exchange_kernel.so      aarch64 device .so (zero-link)
#   libcann_kv_exchange_kernel.json    kernel registration json
#   cann-kv-exchange-aicpu.tar.gz      packaged tar (aicpu_kernels_device/ root)
#
# Install layout expected by the aclrtBinaryLoadFromFile loader:
#   $ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/config/libcann_kv_exchange_kernel.json
#   $ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/kernel/libcann_kv_exchange_kernel.so
# The .so is additionally copied into kernel/aicpu_kernels_device/ to cover either
# kernelSo resolution convention.
#
# Alternative deployment without touching CANN dirs: put the .so and .json in any
# readable directory and export SGL_KV_EXCHANGE_AICPU_JSON=/path/to/libcann_kv_exchange_kernel.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="${SCRIPT_DIR}/../csrc/hicache_offload/kv_exchange_aicpu/op_kernel"
BUILD_DIR="${OP_DIR}/build"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
fi
CROSS_GXX="${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++"

echo "=== kv_exchange AICPU kernel build ==="
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "CROSS_GXX=${CROSS_GXX}"

if [[ ! -x "${CROSS_GXX}" ]]; then
    echo "ERROR: aarch64 cross compiler not found at ${CROSS_GXX}" >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"

echo "---- compile libcann_kv_exchange_kernel.so ----"
DLOG_FLAGS=""
if [ "${SGL_KVX_DLOG}" = "1" ]; then
    DLOG_FLAGS="-DSGL_KVX_ENABLE_DLOG -I${ASCEND_HOME_PATH}/pkg_inc/base"
fi
"${CROSS_GXX}" -shared -O2 -std=c++17 -fPIC \
    ${DLOG_FLAGS} \
    -o "${BUILD_DIR}/libcann_kv_exchange_kernel.so" \
    "${OP_DIR}/kv_exchange_aicpu_kernel.cc"

echo "---- stage json + tar.gz ----"
cp "${OP_DIR}/libcann_kv_exchange_kernel.json" "${BUILD_DIR}/libcann_kv_exchange_kernel.json"

STAGING="${BUILD_DIR}/tar_staging/aicpu_kernels_device"
rm -rf "${BUILD_DIR}/tar_staging"
mkdir -p "${STAGING}"
cp "${BUILD_DIR}/libcann_kv_exchange_kernel.so" "${STAGING}/"
tar czf "${BUILD_DIR}/cann-kv-exchange-aicpu.tar.gz" -C "${BUILD_DIR}/tar_staging" aicpu_kernels_device

echo "---- build done ----"
ls -l "${BUILD_DIR}"

if [[ "${1:-}" == "install" ]]; then
    CUST_DIR="${ASCEND_HOME_PATH}/opp/vendors/cust/op_impl/aicpu"
    echo "=== install into ${CUST_DIR} ==="
    mkdir -p "${CUST_DIR}/config" "${CUST_DIR}/kernel/aicpu_kernels_device"
    cp "${BUILD_DIR}/libcann_kv_exchange_kernel.json" "${CUST_DIR}/config/"
    cp "${BUILD_DIR}/libcann_kv_exchange_kernel.so" "${CUST_DIR}/kernel/"
    cp "${BUILD_DIR}/libcann_kv_exchange_kernel.so" "${CUST_DIR}/kernel/aicpu_kernels_device/"
    echo "installed:"
    ls -l "${CUST_DIR}/config/libcann_kv_exchange_kernel.json" \
        "${CUST_DIR}/kernel/libcann_kv_exchange_kernel.so" \
        "${CUST_DIR}/kernel/aicpu_kernels_device/libcann_kv_exchange_kernel.so"
    echo "NOTE: if the loader still cannot resolve kernelSo, fall back to:"
    echo "  export SGL_KV_EXCHANGE_AICPU_JSON=${BUILD_DIR}/libcann_kv_exchange_kernel.json"
fi
