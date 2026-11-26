#!/usr/bin/env bash
# 08_stop_all.sh -- 停止整套部署 (顺序: 训练 -> LB -> vllm 服务)。
# HTTP server (s58:9020) 独立运行, 默认保留 (重启 vllm 不影响)。
# Usage: bash 08_stop_all.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

echo "=== 1. stop training on ${TRAIN_IP} ==="
n=$(train_procs)
if [ "${n:-0}" -gt 0 ]; then
    echo "sending SIGTERM to ${n} procs (graceful checkpoint attempt)..."
    train_run "pkill -TERM -f 'step[23]_train_[d]spark'; pkill -TERM -f 'speculators.trai[n]'" || true
    # give the graceful path up to 60s (torchrun may escalate sooner)
    for i in $(seq 1 12); do
        sleep 5
        n=$(train_procs)
        [ "${n:-0}" -eq 0 ] && break
    done
    if [ "${n:-0}" -gt 0 ]; then
        echo "still ${n} alive -> SIGKILL"
        train_run "pkill -9 -f 'step[23]_train_[d]spark'; pkill -9 -f 'speculators.trai[n]'; pkill -9 -f 'torchru[n]'" || true
        sleep 3
    fi
    echo "training stopped (last step: $(train_step))"
else
    echo "no training running"
fi

echo "=== 2. stop LB on ${LB_IP} ==="
in_container "${LB_IP}" "pkill -f 'lb_proxy.py'" 2>/dev/null && echo "LB stopped" || echo "no LB running"

echo "=== 3. stop vllm services ==="
for ip in "${INFER_IPS[@]}"; do
    # 容器内杀 python/launch_vllm 进程
    in_container "${ip}" "pkill -9 -f 'launch_vllm.py' 2>/dev/null; pkill -9 -f 'python.*vllm' 2>/dev/null; true"
    # 宿主机杀 VLLM::Worker 进程 (进程名不含 python)
    remote "${ip}" "pkill -9 -f 'VLLM::' 2>/dev/null; pkill -9 -f 'vllm.entrypoints' 2>/dev/null; true"
    echo "kill signal sent on ${ip}"
done
sleep 8

echo "=== 4. final check ==="
fail=0
for ip in "${INFER_IPS[@]}"; do
    n=$(remote "${ip}" "npu-smi info | grep -c 'VLLMWorker\|launch_vllm'" 2>/dev/null | tr -d '\r')
    if [ "${n:-1}" -ne 0 ]; then
        echo "WARN: ${ip} still has ${n} NPU processes"
        fail=1
    else
        echo "OK: ${ip} idle"
    fi
done
n=$(train_procs)
echo "train procs: ${n:-0} (last step: $(train_step))"
[ "$fail" -ne 0 ] && { echo "NOT fully stopped" >&2; exit 1; }
echo "=== all stopped ==="
