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
OURS_LOG=/tmp/cmp_ours.log
MF_LOG=/tmp/cmp_mf.log
rm -f "${OURS_LOG}" "${MF_LOG}"

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

# bench 的 host/device 池缓冲都按整池分配: pool_bytes = POOL x LAYERS x PAGE_SIZE x 2组件 x WIDTH x elem。
# 大层数 + 大池会超过物理内存/HBM, 按预算钳制 (带宽只取决于传输页数, 池大小不影响结果,
# 只影响索引散布范围)。预算同时约束 host DRAM 和 device HBM 两侧 (缓冲等大), 留足运行时余量。
HOST_BUDGET_GB=${HOST_BUDGET_GB:-4}
PAGE_HOST_BYTES=$(( LAYERS * PAGE_SIZE * 2 * WIDTH * ELEM ))
MAX_POOL=$(( HOST_BUDGET_GB * 1073741824 / PAGE_HOST_BYTES ))
if [ "${POOL_PAGES}" -gt "${MAX_POOL}" ]; then
    echo "[note] 请求的池 ${POOL_PAGES} 页需要 $(( POOL_PAGES * PAGE_HOST_BYTES / 1073741824 )) GB host 内存,"
    echo "       超出预算 ${HOST_BUDGET_GB}GB (HOST_BUDGET_GB 可调), 已钳制到 ${MAX_POOL} 页。"
    echo "       传输带宽只由 -n 传输页数决定, 此钳制不影响对比结论。"
    POOL_PAGES=${MAX_POOL}
fi

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
      "${RUNNER[@]}" python3 "${OUR_BENCH}" "${OUR_ARGS[@]}" ) 2>&1 | tee "${OURS_LOG}"
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
          --k_dim "${MF_K_DIM}" --v_dim "${MF_V_DIM}" ) 2>&1 | tee "${MF_LOG}"
    echo
fi

# ---------------- 自动汇总：从两份日志提取带宽数字并对比 ----------------
extract_ours() {  # 第一个 (默认核数) 的指定方向 GB/s
    awk -v d="$1" '$0 ~ "dir="d && $0 ~ /GB\/s/ {for(i=1;i<=NF;i++) if($i=="GB/s"){print $(i-1); exit}}' "$2"
}
extract_mf() {    # 指定布局的 GiB/s
    awk -v l="$1" '$0 ~ "layout:"l {f=1; next} f && /bandwidth:/ {for(i=1;i<=NF;i++) if($i=="GiB/s"){print $(i-1); exit}}' "$2"
}

echo "================ 性能对比总结 ================"
echo "  场景: ${TOKENS} token x ${LAYERS} 层 | ${MODES} 大页 | 核数(我们)=mf原语义2核 | taskset=${TASKSET_CPUS:-none}"
if [ "${RUN_OURS}" -eq 1 ]; then
    echo "  [迁移版 kv_exchange]  (GB/s, 十进制)"
    O_H2D=$(extract_ours H2D "${OURS_LOG}")
    O_D2H=$(extract_ours D2H "${OURS_LOG}")
    [ -n "${O_H2D}" ] && echo "    H2D(换入): ${O_H2D} GB/s"
    [ -n "${O_D2H}" ] && echo "    D2H(换出): ${O_D2H} GB/s"
    [ -z "${O_H2D}" ] && echo "    (未解析到结果, 见上方日志)"
    N_SWEEP=$(grep -c 'GB/s' "${OURS_LOG}" 2>/dev/null || true)
    [ "${N_SWEEP:-0}" -gt 2 ] && echo "    (含 -d 扫参共 ${N_SWEEP} 行, 上表取默认核数首行; 扫参全表见上方)"
fi
if [ "${RUN_MF}" -eq 1 ]; then
    echo "  [mf 原生 offload]  (GiB/s)"
    M_PF=$(extract_mf page_first "${MF_LOG}")
    M_LF=$(extract_mf layer_first "${MF_LOG}")
    [ -n "${M_PF}" ] && echo "    page_first(H2D, HiCache真实布局): ${M_PF} GiB/s"
    [ -n "${M_LF}" ] && echo "    layer_first(H2D, 参考): ${M_LF} GiB/s"
    [ -z "${M_PF}" ] && echo "    (未解析到结果, 见上方日志)"
fi
if [ -n "${O_H2D:-}" ] && [ -n "${M_PF:-}" ]; then
    awk -v o="${O_H2D}" -v m="${M_PF}" 'BEGIN{
        ogi=o/1.024
        printf "  [结论] page_first H2D: 我们 %.2f GiB/s vs mf %.2f GiB/s -> 加速比 %.1fx\n", ogi, m, ogi/m
        if (ogi >= m*0.9)      print "         判定: 迁移版 >= mf 原生, 迁移无性能损失"
        else                   print "         判定: 迁移版低于 mf 原生, 需排查"
    }'
fi
echo
