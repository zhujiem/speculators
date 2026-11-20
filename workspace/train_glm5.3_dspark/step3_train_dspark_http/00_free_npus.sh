#!/usr/bin/env bash
# 00_free_npus.sh -- 停训练进程, 然后验证训练节点 NPU 空闲。
# 不动 vllm 服务 / LB / HTTP server (由 producer 目录管理)。
# Usage: bash 00_free_npus.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

echo "=== 1. stop training on ${TRAIN_IP} (if any) ==="
n=$(train_procs)
if [ "${n:-0}" -gt 0 ]; then
    # TERM first (graceful checkpoint), then KILL after 20s
    train_run "pkill -TERM -f 'step3_train_[d]spark'; pkill -TERM -f 'speculators.trai[n]'" || true
    echo "sent SIGTERM to ${n} training procs; waiting 20s for graceful exit..."
    sleep 20
    n=$(train_procs)
    if [ "${n:-0}" -gt 0 ]; then
        echo "still ${n} alive -> SIGKILL"
        train_run "pkill -9 -f 'step3_train_[d]spark'; pkill -9 -f 'speculators.trai[n]'; pkill -9 -f 'torchru[n]'" || true
        sleep 3
    fi
else
    echo "no training procs on ${TRAIN_IP}"
fi

echo "=== 2. verify training NPU idle ==="
n=$(train_run "npu-smi info | grep -c 'VLLMWorker\|python3'" 2>/dev/null | tr -d '\r')
if [ "${n:-1}" -ne 0 ]; then
    echo "WARN: ${TRAIN_IP} still shows ${n} NPU processes"
    exit 1
fi
echo "OK: ${TRAIN_IP} training NPU idle"
n=$(train_procs)
echo "train procs: ${n:-0}"
echo "=== training node freed ==="
