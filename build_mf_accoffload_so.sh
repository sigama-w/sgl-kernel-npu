#!/bin/bash
# Build mf's libmf_hybm_accoffload.so manually on an A5 device.
# The official install path (mf script/run_pkg_maker/install.sh try_install_extend)
# builds this .so at wheel-install time; if bisheng was missing it silently leaves
# a 0-byte placeholder, which is exactly the broken state on the current device.
#
# Usage (run on the NPU device):
#   bash build_mf_accoffload_so.sh <operators_dir> <launch_cpp> <pkg_lib_dir>
#     <operators_dir> : mf src/acc_offload/csrc/operators/  (kernel + config_ascendc.cmake)
#     <launch_cpp>    : mf src/acc_offload/csrc/launch/acc_offload_operators_launch.cpp
#     <pkg_lib_dir>   : directory that receives libmf_hybm_accoffload.so
#                       (site-packages/memfabric_hybrid/lib of a shadow copy)
set -e

SRC_OPERATORS=$1
LAUNCH_CPP=$2
OUT_DIR=$3
if [ -z "${SRC_OPERATORS}" ] || [ -z "${LAUNCH_CPP}" ] || [ -z "${OUT_DIR}" ]; then
    echo "usage: bash build_mf_accoffload_so.sh <operators_dir> <launch_cpp> <out_dir>"
    exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
if ! command -v bisheng >/dev/null 2>&1; then
    echo "ERROR: bisheng not found after sourcing CANN set_env.sh (need CANN >= 8.3.RC1)"
    exit 1
fi

ASCEND_VERSION=A5
mf_soc_version=ascend950pr_9599
ascend_home=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}

python_bin=python3
torch_dir=$(${python_bin} -c "import torch; import os; print(os.path.dirname(torch.__file__))")
torch_npu_dir=$(${python_bin} -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))")

abi_flag="-D_GLIBCXX_USE_CXX11_ABI=0"
abi_val=$(${python_bin} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" 2>/dev/null)
if [ "${abi_val}" == "1" ]; then
    abi_flag="-D_GLIBCXX_USE_CXX11_ABI=1"
fi

# Step 1: AscendC kernels -> libmf_accoffload_kernel.a (same as mf install.sh L366-402)
rm -rf build_cmake && mkdir build_cmake
cmake -S "${SRC_OPERATORS}" -B build_cmake \
    -DSOC_VERSION=${mf_soc_version} \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DASCEND_HOME_PATH="${ascend_home}"
cmake --build build_cmake -j$(nproc 2>/dev/null || echo 8)

KERN_A=$(find build_cmake -name 'libmf_accoffload_kernel.a' -print -quit)
if [ -z "${KERN_A}" ] || [ ! -f "${KERN_A}" ]; then
    echo "ERROR: libmf_accoffload_kernel.a not found after cmake build"
    exit 1
fi

# Step 2: torch_npu adapter object (same as mf install.sh L420-438)
g++ -c -fPIC -std=c++17 -O3 -fstack-protector-strong \
    -Wno-unused-parameter -Wno-unused-function -Wunused-value -Wcast-align \
    -Wcast-qual -Wwrite-strings -Wsign-compare -Wextra \
    -fvisibility-inlines-hidden -ftrapv \
    ${abi_flag} \
    -DACC_SOC_VERSION_${ASCEND_VERSION} \
    -isystem ${ascend_home}/include \
    -isystem ${ascend_home}/include/experiment/runtime/runtime/ \
    -isystem ${torch_dir}/include \
    -isystem ${torch_dir}/include/torch/csrc/api/include \
    -isystem ${torch_dir}/include/torch/csrc/utils \
    -isystem ${torch_dir}/include/c10/util \
    -isystem ${torch_dir}/include/c10/core \
    -isystem ${torch_dir}/include/ATen \
    -isystem ${torch_dir}/include/ATen/detail \
    -isystem ${torch_npu_dir}/include \
    -isystem ${torch_npu_dir}/include/torch_npu/csrc/aten \
    -isystem ${torch_npu_dir}/include/torch_npu/csrc/core/npu \
    "${LAUNCH_CPP}" -o acc_offload_operators_launch.o

# Step 3: link the final .so (same as mf install.sh L446-453)
g++ -shared -fPIC -o libmf_hybm_accoffload.so \
    acc_offload_operators_launch.o \
    -Wl,--whole-archive "${KERN_A}" -Wl,--no-whole-archive \
    -L${torch_dir}/lib -ltorch -lc10 -ltorch_python \
    -L${torch_npu_dir}/lib -ltorch_npu \
    -L${ascend_home}/lib64 -lopapi \
    -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now \
    -Wl,-rpath,\$ORIGIN -Wl,-rpath,${torch_dir}/lib -Wl,-rpath,${torch_npu_dir}/lib -Wl,-rpath,${ascend_home}/lib64

mkdir -p "${OUT_DIR}"
cp -v libmf_hybm_accoffload.so "${OUT_DIR}/"
rm -rf build_cmake acc_offload_operators_launch.o libmf_hybm_accoffload.so
echo "DONE: ${OUT_DIR}/libmf_hybm_accoffload.so"
