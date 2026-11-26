#!/usr/bin/env bash
# 01_start_mooncake.sh -- 在 s67 容器内后台启动 mooncake master (:50051)。
# 必须在 02_start_vllm_instances.sh 之前运行 (vllm 服务启动时连接 master)。
# 幂等: 端口已监听则直接跳过。master 独立于 vllm 服务存活。
# Usage: bash 01_start_mooncake.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

mkdir -p "$LOG_DIR"

mc_ok() {
    remote "${MOONCAKE_MASTER_IP}" "timeout 2 bash -c '</dev/tcp/${MOONCAKE_MASTER_IP}/${MOONCAKE_MASTER_PORT}'" 2>/dev/null
}

if mc_ok; then
    echo "mooncake master already running on ${MOONCAKE_MASTER_IP}:${MOONCAKE_MASTER_PORT}"
    exit 0
fi

echo "[$(date '+%T')] starting mooncake master on ${MOONCAKE_MASTER_IP}:${MOONCAKE_MASTER_PORT} ..."
remote "${MOONCAKE_MASTER_IP}" "docker exec -d ${CONTAINER_NAME} bash -c \
    'nohup mooncake_master --rpc_port ${MOONCAKE_MASTER_PORT} --metrics_port 9003 \
     --rpc_thread_num 8 --enable_disk_eviction=false --logtostderr=true \
     >> ${LOG_DIR}/mooncake_master.log 2>&1'"

deadline=$((SECONDS + 30))
while [ "$SECONDS" -lt "$deadline" ]; do
    if mc_ok; then
        echo "mooncake master READY: ${MOONCAKE_MASTER_IP}:${MOONCAKE_MASTER_PORT}"
        echo "log: ${LOG_DIR}/mooncake_master.log"
        exit 0
    fi
    sleep 2
done
echo "mooncake master failed to come up -- see ${LOG_DIR}/mooncake_master.log" >&2
exit 1