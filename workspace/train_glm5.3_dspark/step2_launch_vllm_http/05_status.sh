#!/usr/bin/env bash
# 05_status.sh -- 全景状态: 推理服务 / LB / HTTP server / 训练 / NPU / 内存。
# Usage: bash 06_status.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

line() { printf '%s\n' "----------------------------------------"; }

echo "=========== dspark 2-node deployment status ($(date '+%F %T')) ==========="

# --- inference services ---------------------------------------------------------
line; echo "[infer services]"
for ip in "${INFER_IPS[@]}"; do
    up=$(remote "${ip}" "docker inspect -f '{{.State.Running}}' ${CONTAINER_NAME}" 2>/dev/null | tr -d '\r')
    if curl -sf --max-time 4 "http://${ip}:${SVC_PORT}/v1/models" >/dev/null 2>&1; then
        svc="READY"
    else
        svc="DOWN"
    fi
    npu=$(remote "${ip}" "npu-smi info | grep -c 'VLLMWorker'" 2>/dev/null | tr -d '\r')
    mem=$(remote "${ip}" "free -g | awk 'NR==2{printf \"%d/%d\", \$3, \$2}'" 2>/dev/null | tr -d '\r')
    echo "  ${ip}: container=${up:-?} service=${svc} vllm_workers=${npu:-?} mem(used/total)=${mem:-?}G"
done

# --- LB --------------------------------------------------------------------------
line; echo "[LB]"
if curl -sf --max-time 4 "http://${LB_IP}:${LB_PORT}/v1/models" >/dev/null 2>&1; then
    echo "  http://${LB_IP}:${LB_PORT} READY (backends: $(lb_backend_list))"
else
    echo "  http://${LB_IP}:${LB_PORT} DOWN"

fi

# --- LB full stats (from openresty.log) ---
lb_stats=$(grep "stats " "$LOG_DIR/openresty.log" 2>/dev/null | grep "10.100.1.58:8008" | tail -1)
if [ -n "$lb_stats" ]; then
    echo "  served:      $(echo "$lb_stats" | grep -oE "[0-9.]+:8008:[0-9]+/[0-9]+" | tr '\n' ' ')"
    echo "  inflight:    $(echo "$lb_stats" | grep -oE "inflight=\[[^]]*\]")"
    echo "  inflight_requests: $(echo "$lb_stats" | grep -oE "inflight_requests=[0-9]+")"
    echo "  inflight_batches:  $(echo "$lb_stats" | grep -oE "inflight_batches=[0-9]+")"
    echo "  queued_batches:    $(echo "$lb_stats" | grep -oE "queued_batches=[0-9]+")"
    echo "  sem_locked:  $(echo "$lb_stats" | grep -oE "sem_locked=[A-Za-z]+")"
    echo "  queued_total: $(echo "$lb_stats" | grep -oE "queued_total=[0-9]+")"
    echo "  queue_timeouts: $(echo "$lb_stats" | grep -oE "queue_timeouts=[0-9]+")"
    echo "  affinity_requests: $(echo "$lb_stats" | grep -oE "affinity_requests=[0-9]+")"
    echo "  batch_routes: $(echo "$lb_stats" | grep -oE "batch_routes=[0-9]+")"
    echo "  (as of: $(echo "$lb_stats" | awk '{print $1, $2}'))"
else
    echo "  no stats available"
fi

# --- HTTP server ---------------------------------------------------------------
line; echo "[http]"
if timeout 2 bash -c "</dev/tcp/${HTTP_SERVER_IP}/${HTTP_SERVER_PORT}" 2>/dev/null; then
    echo "  server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} reachable"
else
    echo "  server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} NOT reachable"
fi

# --- training ----------------------------------------------------------------------
line; echo "[training on ${TRAIN_IP}]"
n=$(train_procs)
if [ "${n:-0}" -gt 0 ]; then
    step=$(train_step)
    loss=$(train_run "grep -oE 'train/loss=[0-9.]+' '${TRAIN_LOG}' 2>/dev/null | tail -1" 2>/dev/null | tr -d '\r')
    alen=$(train_run "grep -oE 'train/accept_len=[0-9.]+' '${TRAIN_LOG}' 2>/dev/null | tail -1" 2>/dev/null | tr -d '\r')
    echo "  RUNNING: procs=${n} step=${step:-?} ${loss:-} ${alen:-}"
    echo "  log: ${TRAIN_LOG}"
else
    echo "  NOT running"
    [ -f "$TRAIN_LOG" ] && echo "  last log: $(train_run "tail -1 '${TRAIN_LOG}'" 2>/dev/null | tr -d '\r')"
fi
echo "  host mem: $(train_run "free -g | awk 'NR==2{printf \"%dG used / %dG total, avail %dG\", \$3, \$2, \$7}'" 2>/dev/null | tr -d '\r')"

# --- latest checkpoint ---------------------------------------------------------------
line; echo "[latest checkpoints]"
train_run "ls -t /mnt/cephfs/hw/zhujieming/runs/glm5.3_dspark_32k_v1/checkpoints2/ 2>/dev/null | head -3" 2>/dev/null | tr -d '\r' || true
echo "(full listing: /mnt/cephfs/hw/zhujieming/runs/glm5.3_dspark_32k_v1/checkpoints2/)"
line