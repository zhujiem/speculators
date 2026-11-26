#!/usr/bin/env bash
# 启动前硬性校验所有推理节点 NPU 空闲。
# Usage: bash 02_start_vllm_instances.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

mkdir -p "$LOG_DIR"

# --- hard guard: every inference node must have 0 NPU worker processes -------
for ip in "${INFER_IPS[@]}"; do
    n=$(remote "$ip" "npu-smi info | grep -c 'VLLMWorker\|python3'" 2>/dev/null)
    if [ "${n:-1}" -ne 0 ]; then
        echo "REFUSING: $ip still has $n NPU worker processes. Run 00_free_npus.sh first." >&2
        exit 1
    fi
done
echo "=== all inference nodes NPU-idle ==="

# --- HTTP server must be up (run 01_start_http_server.sh first) ----------------
if ! remote "${HTTP_SERVER_IP}" "timeout 2 bash -c '</dev/tcp/${HTTP_SERVER_IP}/${HTTP_SERVER_PORT}'" 2>/dev/null; then
    echo "REFUSING: HTTP server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} not reachable." >&2
    echo "  Run: bash 01_start_http_server.sh first." >&2
    exit 1
fi
echo "=== HTTP server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} reachable ==="

SVC_SCRIPT="${SVC_SCRIPT:-$BASE_DIR/launch_vllm_glm5.3.sh}"

launch() { # ip
    local ip="$1"
    echo "[$(date '+%T')] launching service on $ip"
    # $1 to launch_vllm_glm5.3.sh = HTTP server IP (s58, http kv store)
    remote "$ip" "docker exec -d $CONTAINER_NAME bash -c \
        'bash $SVC_SCRIPT $HTTP_SERVER_IP > $LOG_DIR/vllm_${ip}.log 2>&1'" 2>/dev/null &
}

# staggered launch (cephfs read contention on cold weights)
i=0
for ip in "${INFER_IPS[@]}"; do
    launch "$ip"
    i=$((i+1))
    [ $i -lt ${#INFER_IPS[@]} ] && sleep "$STAGGER_S"
done

# --- readiness wait ------------------------------------------------------------
echo "=== waiting for readiness (timeout ${SVC_READY_TIMEOUT}s) ==="
deadline=$((SECONDS + SVC_READY_TIMEOUT))
declare -A ready
total=${#INFER_IPS[@]}
while [ "$SECONDS" -lt "$deadline" ]; do
    n_ready=0
    for ip in "${INFER_IPS[@]}"; do
        key="$ip:$SVC_PORT"
        [ "${ready[$key]:-0}" = "1" ] && { n_ready=$((n_ready+1)); continue; }
        id=$(curl -sf --max-time 5 "http://${ip}:${SVC_PORT}/v1/models" 2>/dev/null \
            | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
        if [ "$id" = "$MODEL_NAME" ]; then
            ready[$key]=1; n_ready=$((n_ready+1))
            echo "[$(date '+%T')] READY $key"
        fi
    done
    [ "$n_ready" -eq "$total" ] && { echo "=== all $total services READY ==="; exit 0; }
    sleep 20
done
echo "TIMEOUT: only ${#ready[@]}/$total services ready" >&2
exit 1
