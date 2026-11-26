#!/usr/bin/env bash
# 01_start_http_server.sh -- 启动 HTTP key-value server (hs_http_server.py)。
# vllm 与训练的隐状态传输都依赖此服务, 必须在 02_start_vllm_instances 之前运行。
# 幂等: 已在监听则跳过。
# Usage: bash 01_start_http_server.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

mkdir -p "$LOG_DIR"

# --- idempotent: skip if already listening ------------------------------------
if remote "${HTTP_SERVER_IP}" "timeout 2 bash -c '</dev/tcp/${HTTP_SERVER_IP}/${HTTP_SERVER_PORT}'" 2>/dev/null; then
    echo "[http] server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} already running"
    exit 0
fi

echo "[http] starting hs_http_server on ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT}"
remote "${HTTP_SERVER_IP}" "docker exec -d ${CONTAINER_NAME} bash -c '
    python3 /mnt/cephfs/hw/zhujieming/speculators/hs_connectors/scripts/hs_http_server.py \
        --port ${HTTP_SERVER_PORT} \
        --max-buffer-gib ${HTTP_SERVER_MAX_BUFFER_GIB:-64} \
        > ${LOG_DIR}/hs_http_server.log 2>&1'"

# --- wait for readiness (max 15s) ----------------------------------------------
for _i in $(seq 1 15); do
    if remote "${HTTP_SERVER_IP}" "timeout 2 bash -c '</dev/tcp/${HTTP_SERVER_IP}/${HTTP_SERVER_PORT}'" 2>/dev/null; then
        echo "[http] server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} READY"
        echo "  max_buffer=${HTTP_SERVER_MAX_BUFFER_GIB:-64} GiB  read-once GET"
        exit 0
    fi
    sleep 1
done

echo "FATAL: hs_http_server failed to start on ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT}" >&2
echo "  check: ${LOG_DIR}/hs_http_server.log" >&2
exit 1
