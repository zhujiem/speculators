#!/usr/bin/env bash
# 04_stop_training.sh -- 停止 s58 训练 (优雅退出优先, 60s 后强杀)。
# 不停 vllm 服务 / LB / HTTP server (由 producer 目录管理)。
# Usage: bash 04_stop_training.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

echo "=== stopping training on ${TRAIN_IP} ==="
n=$(train_procs)
if [ "${n:-0}" -gt 0 ]; then
    echo "sending SIGTERM to ${n} procs (graceful checkpoint attempt)..."
    train_run "pkill -TERM -f 'step3_train_[d]spark'; pkill -TERM -f 'speculators.trai[n]'" || true
    for i in $(seq 1 12); do
        sleep 5
        n=$(train_procs)
        [ "${n:-0}" -eq 0 ] && break
    done
    if [ "${n:-0}" -gt 0 ]; then
        echo "still ${n} alive -> SIGKILL"
        train_run "pkill -9 -f 'step3_train_[d]spark'; pkill -9 -f 'speculators.trai[n]'; pkill -9 -f 'torchru[n]'" || true
        sleep 3
    fi
    echo "training stopped (last step: $(train_step))"
else
    echo "no training running"
fi

n=$(train_procs)
echo "train procs: ${n:-0} (last step: $(train_step))"
echo "(vllm/LB/HTTP server 未动, 由 producer 目录管理)"
