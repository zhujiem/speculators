#!/usr/bin/env bash
# 02_start_training.sh -- 在训练容器内后台启动正式训练。
# 前置: producer 目录 vllm 服务 + LB :8100 已就绪。
# Usage: bash 02_start_training.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

mkdir -p "$LOG_DIR"

# Use a timestamped log file so restarts never overwrite previous runs;
# keep a "latest" symlink for convenience (tail -f $LOG_DIR/latest.log)
_TRAIN_TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/train_${_TRAIN_TS}.log"
ln -sfn "$TRAIN_LOG" "$LOG_DIR/latest.log"

# --- preflight: LB must answer -----------------------------------------------
if ! curl -sf --max-time 5 "http://${LB_IP}:${LB_PORT}/v1/models" >/dev/null 2>&1; then
    echo "REFUSING: LB ${LB_IP}:${LB_PORT} not answering. Run 02+03 first." >&2
    exit 1
fi
echo "preflight OK: LB answering on ${LB_IP}:${LB_PORT}"

# --- refuse if training already running ---------------------------------------
n=$(train_procs)
if [ "${n:-0}" -gt 0 ]; then
    echo "REFUSING: ${n} training procs already running on ${TRAIN_IP} (step=$(train_step))" >&2
    exit 1
fi

# --- launch inside the trainer container (background, detached) ---------------
train_run "docker exec -d ${TRAINER_CONTAINER} bash -c \
    'cd ${BASE_DIR} && bash ${TRAIN_SCRIPT:-train_dspark_glm5.3.sh} ${MASTER_IP:-10.100.1.67} > ${TRAIN_LOG} 2>&1'"
echo "training launched in container ${TRAINER_CONTAINER} on ${TRAIN_IP}"

# --- verify it came up ---------------------------------------------------------
sleep 15
n=$(train_procs)
if [ "${n:-0}" -gt 0 ]; then
    echo "OK: ${n} training procs alive"
    echo "log: ${TRAIN_LOG}"
    echo "watch: train_run tail -f  (or) bash 03_status.sh"
else
    echo "WARN: no training procs after 15s -- check log tail:" >&2
    train_run "tail -5 '${TRAIN_LOG}'" || true
    exit 1
fi
