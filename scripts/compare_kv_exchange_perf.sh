#!/bin/bash
# ============================================================================
# compare_kv_exchange_perf.sh
# 对比迁移版 kv_exchange kernel (sgl-kernel-npu) 与 mf 原生 offload 算子的
# L1<->L2 KV 搬运带宽。两个 harness 各自用原生参数表达同一真实推理场景，
# 带宽自带 payload 归一化，可直接对比。
#
# 传输方式：本文件通过 scp 传到设备，禁止聊天粘贴（会损坏 $ 等字符）。
#
# 用法（均在设备上执行，默认参数即真实推理场景）：
#   bash compare_kv_exchange_perf.sh
#
#   大批量抢占场景（32768 token，批量换出）：
#   bash compare_kv_exchange_perf.sh -n 256
#
#   NUMA 归一化（绑到 mf 日志显示的 node127 CPU 组，排除亲和干扰）：
#   bash compare_kv_exchange_perf.sh -T 288-383
#
#   只跑我们的算子 / 核数扫参：
#   bash compare_kv_exchange_perf.sh --skip-mf -d 62,60,56,48,32,16,0
#
# 参数含义（-h 查看）：
#   -p POOL_PAGES  L2 池大小（页）。页=128 token。默认 32768 页 = 4M token，
#                  对应 sglang HiCache 典型 L2 容量（数百 GB 级 host 池的页数规模）
#   -n PAGES       单次换入/换出的页数 = 一次请求级操作的 token 数/128。
#                  默认 32 页 = 4096 token（典型请求被抢占时的 KV 量）
#   -l LAYERS      模型 KV 层数。默认 61（DSV3/GLM 满层全量换入换出）
#   -w WIDTH       每 token 每组件元素数。默认 576（MLA: kv_lora 512 + rope 64）
#   -s PAGE_SIZE   每页 token 数。默认 128（sglang 默认 page size）
#   -m MODES       大页模式（auto/1g/2m/off）。默认 2m（部署推荐档）
#   -d START_DIMS  参与搬运的 AIV 核数扫参（62=仅2核, 0=全部64核）
#   -i ITERS       计时迭代数，默认 20
#   -T CPUS        taskset CPU 绑定（NUMA 归一化实验用，如 288-383）
#   -W WORLD_SIZE  mf 用例 rank 数，默认 1（与单卡 bench 严格可比）
#   --skip-ours    只跑 mf
#   --skip-mf      只跑我们的算子
# ============================================================================
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OUR_BENCH="${REPO_ROOT}/benchmark/hicache_offload/bench_kv_exchange.py"
MF_EXAMPLE_DIR="${MF_EXAMPLE_DIR:-/mnt/share/w00937173/offload/kv_exchange_offload}"
MF_SHADOW="${MF_SHADOW_DIR:-/mnt/share/w00937173/mf_shadow}"

POOL_PAGES=32768
PAGES=32
PAGE_SIZE=128
WIDTH=576
LAYERS=61
DTYPE=bfloat16
MODES=2m
START_DIMS=""
ITERS=20
TASKSET_CPUS=""
WORLD_SIZE=1
RUN_OURS=1
RUN_MF=1
MF_K_DIM=512
MF_V_DIM=64

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        -p) POOL_PAGES=$2; shift 2 ;;
        -n) PAGES=$2; shift 2 ;;
        -l) LAYERS=$2; shift 2 ;;
        -w) WIDTH=$2; shift 2 ;;
        -s) PAGE_SIZE=$2; shift 2 ;;
        -m) MODES=$2; shift 2 ;;
        -d) START_DIMS=$2; shift 2 ;;
        -i) ITERS=$2; shift 2 ;;
        -T) TASKSET_CPUS=$2; shift 2 ;;
        -W) WORLD_SIZE=$2; shift 2 ;;
        --skip-ours) RUN_OURS=0; shift ;;
        --skip-mf) RUN_MF=0; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1 (-h for help)"; exit 1 ;;
    esac
done

case "${DTYPE}" in
    float32) ELEM=4 ;;
    *)       ELEM=2 ;;
esac

TOKENS=$(( PAGES * PAGE_SIZE ))
# 我们 bench 的单次 payload：pages x page_size x layers x 2组件 x width x elem
OUR_PAYLOAD_MB=$(( TOKENS * LAYERS * 2 * WIDTH * ELEM / 1000000 ))
# mf 用例按 token 数匹配同一场景（其 k+v=576/token 原生表示），换算成 --data_mibs
MF_TOKEN_BYTES=$(( LAYERS * (MF_K_DIM + MF_V_DIM) * ELEM ))
MF_MIBS=$(( TOKENS * MF_TOKEN_BYTES / 1048576 ))
[ "${MF_MIBS}" -lt 1 ] && MF_MIBS=1

echo "================ 场景 ================"
echo "  单次传输 token 数 : ${TOKENS} (=${PAGES} 页 x ${PAGE_SIZE} token/页)"
echo "  模型形状          : ${LAYERS} 层 x 每token 2x${WIDTH} elem (我们) / ${MF_K_DIM}+${MF_V_DIM} elem (mf), ${DTYPE}"
echo "  我们单次 payload  : ~${OUR_PAYLOAD_MB} MB"
echo "  mf 单次 payload   : ~${MF_MIBS} MiB (同 token 数, mf 原生表示)"
echo "  大页模式          : ${MODES};  迭代 ${ITERS};  taskset: ${TASKSET_CPUS:-none}"
echo

RUNNER=()
if [ -n "${TASKSET_CPUS}" ]; then
    RUNNER=(taskset -c "${TASKSET_CPUS}")
fi

if [ "${RUN_OURS}" -eq 1 ]; then
    if [ ! -f "${OUR_BENCH}" ]; then
        echo "ERROR: bench not found: ${OUR_BENCH}"; exit 1
    fi
    echo "================ 我们的 kv_exchange kernel ================"
    OUR_ARGS=(--pool-pages "${POOL_PAGES}" --pages "${PAGES}" --page-size "${PAGE_SIZE}" \
              --width "${WIDTH}" --layers "${LAYERS}" --dtype "${DTYPE}" \
              --modes "${MODES}" --iters "${ITERS}" --directions h2d,d2h)
    [ -n "${START_DIMS}" ] && OUR_ARGS+=(--start-dims "${START_DIMS}")
    ( cd "${REPO_ROOT}" && \
      PYTHONPATH="${REPO_ROOT}/python/sgl_kernel_npu" \
      "${RUNNER[@]}" python3 "${OUR_BENCH}" "${OUR_ARGS[@]}" )
    echo "  (注: 本输出为 GB/s 十进制; x0.931 换算 GiB/s 后与 mf 直接对比)"
    echo
fi

if [ "${RUN_MF}" -eq 1 ]; then
    if [ ! -f "${MF_EXAMPLE_DIR}/kv_exchange_offload.py" ]; then
        echo "ERROR: mf example not found: ${MF_EXAMPLE_DIR}/kv_exchange_offload.py"
        echo "       可用 MF_EXAMPLE_DIR=... 环境变量指定目录"; exit 1
    fi
    echo "================ mf 原生 offload 算子 (AIV) ================"
    MF_PYP=""
    [ -d "${MF_SHADOW}/memfabric_hybrid" ] && MF_PYP="${MF_SHADOW}"
    ( cd "${MF_EXAMPLE_DIR}" && \
      PYTHONPATH="${MF_PYP}" \
      "${RUNNER[@]}" python3 kv_exchange_offload.py \
          --mode aiv --world_size "${WORLD_SIZE}" \
          --data_mibs "${MF_MIBS}" --num_layers "${LAYERS}" \
          --k_dim "${MF_K_DIM}" --v_dim "${MF_V_DIM}" )
    echo "  (注: 本输出为 GiB/s; page_first 行 = sglang 真实 host 布局, 是对比关键行)"
    echo
fi

echo "================ 判读 ================"
echo "  1. 对比 page_first 带宽 (sglang 真实布局): 我们(GB/s x0.931=GiB/s) vs mf aiv"
echo "  2. -T 跑出的第二组用于 NUMA 归因: 与第一组差异大 -> mf 慢因是亲和绑定"
echo "  3. -d 扫参曲线 -> AIV 核并行度上限 (62=2核 为 mf 原语义)"
