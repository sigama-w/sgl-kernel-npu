# AICPU device kernel for kv_exchange: aarch64 cross-compile + tar.gz packaging.
# Mirrors memfabric_hybrid src/hybm/ops/hybm_kernel/CMakeLists.txt.
#
# Build (on the NPU device or in an ASCEND_HOME_PATH env):
#   mkdir build && cd build
#   cmake .. -DASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
#   make cann_kv_exchange_kernel
# Artifacts:
#   libcann_kv_exchange_kernel.so      raw device .so
#   libcann_kv_exchange_kernel.json    kernel registration json
#   cann-kv-exchange-aicpu.tar.gz      install package (aicpu_kernels_device/ root)
#
# Manual install without the .run flow (both files under the same dir):
#   cp libcann_kv_exchange_kernel.so  $ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/kernel/
#   cp libcann_kv_exchange_kernel.json $ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/config/
cmake_minimum_required(VERSION 3.16)
project(kv_exchange_aicpu_kernel CXX)

set(AICPU_COMPILER "${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++")

set(AICPU_SO_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libcann_kv_exchange_kernel.so")
set(AICPU_JSON_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libcann_kv_exchange_kernel.json")
set(AICPU_TAR_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cann-kv-exchange-aicpu.tar.gz")

set(KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kv_exchange_aicpu_kernel.cc")

# ---- Copy static .json ----
add_custom_command(
    OUTPUT ${AICPU_JSON_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/libcann_kv_exchange_kernel.json
            ${AICPU_JSON_OUTPUT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/libcann_kv_exchange_kernel.json
)

# ---- Cross-compile .so (zero-link, like mf: no DT_NEEDED deps) ----
add_custom_command(
    OUTPUT ${AICPU_SO_OUTPUT}
    COMMAND ${AICPU_COMPILER}
        -shared -O2 -std=c++17 -fPIC
        -o ${AICPU_SO_OUTPUT}
        ${KERNEL_SRC}
    DEPENDS ${KERNEL_SRC}
    COMMENT "Compiling kv_exchange AICPU kernel with aarch64 cross-compiler"
)

# ---- Package .so into tar.gz (AICPUKernel install format) ----
set(AICPU_TAR_ROOT "aicpu_kernels_device")
set(AICPU_STAGING_DIR "${CMAKE_CURRENT_BINARY_DIR}/tar_staging")

add_custom_command(
    OUTPUT ${AICPU_TAR_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${AICPU_STAGING_DIR}/${AICPU_TAR_ROOT}"
    COMMAND ${CMAKE_COMMAND} -E copy "${AICPU_SO_OUTPUT}" "${AICPU_STAGING_DIR}/${AICPU_TAR_ROOT}/"
    COMMAND tar czf "${AICPU_TAR_OUTPUT}" -C "${AICPU_STAGING_DIR}" "${AICPU_TAR_ROOT}"
    DEPENDS ${AICPU_SO_OUTPUT}
    COMMENT "Packaging kv_exchange AICPU kernel into tar.gz"
)

add_custom_target(cann_kv_exchange_kernel ALL
    DEPENDS ${AICPU_SO_OUTPUT} ${AICPU_JSON_OUTPUT} ${AICPU_TAR_OUTPUT}
)
