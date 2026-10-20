#!/bin/bash
# HiCache offload 单算子大页 A/B 实测 runbook（NPU 机器上执行）。
#
# 用法（仓库根目录）:
#   bash scripts/bench_hicache_offload.sh                # 检查 + 单测 + auto/off 对比
#   sudo bash scripts/bench_hicache_offload.sh --setup   # 先自动配置大页再跑
#
# 产物:
#   1. test_kv_exchange / test_hugepage_host_allocator 正确性
#   2. kv_exchange 在 hugepage(auto/1g/2m) vs 4K(off) 下的 H2D/D2H 带宽对比

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/python/sgl_kernel_npu:$PYTHONPATH"

echo "==== [1/4] 当前大页配置 ===="
grep -i 'HugePages_Total\|HugePages_Free\|Hugepagesize' /proc/meminfo || true
cat /proc/sys/vm/nr_hugepages 2>/dev/null || true
ls /proc/sys/vm/nr_hugepages_1G 2>/dev/null && cat /proc/sys/vm/nr_hugepages_1G || echo "(无 1G 大页运行时入口，需内核 cmdline 配置 default_hugepagesz=1G)"

if [ "$1" = "--setup" ]; then
    echo "==== [0/4] 配置 2M 大页池 (32000 页 = 62.5GB) ===="
    sysctl -w vm.nr_hugepages=32000
    echo "1G 大页通常需要内核启动参数: default_hugepagesz=1G hugepagesz=1G hugepages=64"
    echo "若要 1G 模式，请先在 /etc/default/grub 或 bootloader 配置后重启，再重跑本脚本"
fi

echo "==== [2/4] 单算子正确性 ===="
python -m pytest "$ROOT/tests/python/sgl_kernel_npu/test_hugepage_host_allocator.py" \
    "$ROOT/tests/python/sgl_kernel_npu/test_kv_exchange.py" -v

echo "==== [3/4] 大页 A/B 对比 (auto vs off) ===="
python "$ROOT/benchmark/hicache_offload/bench_kv_exchange.py" --modes auto,off

echo "==== [4/4] 严格 1G vs off（需已配置 1G 大页，未配置会显式报错） ===="
python "$ROOT/benchmark/hicache_offload/bench_kv_exchange.py" --modes 1g,off --pool-pages 512
echo "完成。对比第 3/4 步输出的 GB/s 汇总表即可评估大页收益。"
