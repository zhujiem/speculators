#!/usr/bin/env bash
# Usage: bash 03_start_lb.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

mkdir -p "$LOG_DIR"

BACKENDS=$(lb_backend_list)
echo "backends: $BACKENDS"

# all backends healthy?
for b in $BACKENDS; do
    curl -sf --max-time 5 "$b/v1/models" >/dev/null \
        || { echo "backend $b not ready -- run 02_start_vllm.sh first" >&2; exit 1; }
done

if curl -sf --max-time 3 "http://${LB_IP}:${LB_PORT}/v1/models" >/dev/null 2>&1; then
    echo "LB already running on ${LB_IP}:${LB_PORT}"
else
    remote "$LB_IP" "docker exec -d $CONTAINER_NAME bash -c \
        'nohup python3 $BASE_DIR/openresty.py $BACKENDS --port $LB_PORT --max-inflight-batches 4 >> $LOG_DIR/openresty.log 2>&1'" 2>/dev/null
    sleep 2
fi

deadline=$((SECONDS + 30))
while [ "$SECONDS" -lt "$deadline" ]; do
    id=$(curl -sf --max-time 5 "http://${LB_IP}:${LB_PORT}/v1/models" 2>/dev/null \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
    if [ "$id" = "$MODEL_NAME" ]; then
        echo "LB READY: http://${LB_IP}:${LB_PORT}/v1 (model=$id)"
        echo "client endpoint: http://${LB_IP}:${LB_PORT}/v1"
        exit 0
    fi
    sleep 2
done
echo "LB failed to come up -- see $LOG_DIR/openresty.log" >&2
exit 1