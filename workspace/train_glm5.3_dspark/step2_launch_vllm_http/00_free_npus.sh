#!/usr/bin/env bash
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

echo "=== 2. stop LB on ${LB_IP} (if any) ==="
in_container "${LB_IP}" "pkill -f 'lb_proxy.py'" 2>/dev/null && echo "LB stopped" || echo "no LB running"

echo "=== 3. stop vllm services on ${INFER_IPS[*]} ==="
for ip in "${INFER_IPS[@]}"; do
    # 容器内杀 python/launch_vllm 进程
    in_container "${ip}" "pkill -9 -f 'launch_vllm.py' 2>/dev/null; pkill -9 -f 'python.*vllm' 2>/dev/null; true"
    # 宿主机杀 VLLM::Worker 进程 (进程名不含 python, docker exec 内 pkill 匹配不到)
    remote "${ip}" "pkill -9 -f 'VLLM::' 2>/dev/null; pkill -9 -f 'vllm.entrypoints' 2>/dev/null; true"
    echo "kill signal sent on ${ip}"
done
sleep 8

echo "=== 4. verify NPU idle ==="
fail=0
for ip in "${INFER_IPS[@]}"; do
    n=$(remote "${ip}" "npu-smi info | grep -c 'VLLMWorker\|launch_vllm'" 2>/dev/null | tr -d '\r')
    if [ "${n:-1}" -ne 0 ]; then
        echo "WARN: ${ip} still shows ${n} NPU processes"
        fail=1
    else
        echo "OK: ${ip} NPU idle"
    fi
done
n=$(train_procs)
echo "train procs on ${TRAIN_IP}: ${n:-0}"

if [ "$fail" -ne 0 ]; then
    echo "NOT fully idle -- inspect npu-smi on the WARN nodes" >&2
    exit 1
fi
echo "=== all nodes freed ==="
