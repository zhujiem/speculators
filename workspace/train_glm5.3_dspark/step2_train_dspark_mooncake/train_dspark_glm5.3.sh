#!/bin/bash
# train_dspark_glm5.3.sh -- GLM-5.3 训练本体 (由 05_start_training.sh 在 s58 的
# vllm_ascend_v0.30.0 容器内后台运行; 容器内有 torchrun + datasets + http)。
set -euo pipefail

MASTER_IP=${1:?usage: train_dspark_glm5.3.sh <MASTER_IP>}

MODEL="/mnt/cephfs/weight/GLM-5.3"
RESUME_CKPT="/mnt/cephfs/hw/checkpoints/epoch0_step15000_20261110"
OUTPUT_DIR="/mnt/cephfs/hw/runs/glm5.3_dspark_32k_v1"
DATA_PATH="/mnt/cephfs/hw/spectrain/data/train_1110_32k_v2"
VLLM_ENDPOINT="http://${MASTER_IP}:8100/v1"
MOONCAKE_MASTER="http://${MASTER_IP}:50051"
TRAIN_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
NUM_TRAIN_NPUS=16

MASK_TOKEN_ID=154856
PACK_SEQ_LENGTH=32768
TARGET_LAYER_IDS="1 20 38 56 75"

export TMPDIR="/tmp"
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT_DIR}/cache/torchinductor"
export HF_HOME="${OUTPUT_DIR}/cache/hf"
export HCCL_NPU_SOCKET_PORT_RANGE=26000-27000
export ASCEND_RT_VISIBLE_DEVICES="$TRAIN_DEVICES"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64
export ASCEND_AUTO_CONNECT=1
export ASCEND_ENABLE_USE_FABRIC_MEM=1
export ASCEND_TRANSFER_TIMEOUT=31000
export MALLOC_ARENA_MAX=2 # avoid OOM
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

echo "=== Training starts. ==="
echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') Training starts."
torchrun --standalone \
    --nproc_per_node "$NUM_TRAIN_NPUS" \
    -m speculators.train \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --hidden-states-backend mooncake \
    --mooncake-master "$MOONCAKE_MASTER" \
    --mooncake-metadata-server P2PHANDSHAKE \
    --mooncake-protocol ascend \
    --mooncake-global-segment-gib 0 \
    --mooncake-local-buffer-gib 4 \
    --on-missing generate \
    --request-timeout 900 \
    --max-retries 5 \
    --generation-validation-retries 1 \
    --max-consecutive-generation-failures 10 \
    --dataloader streaming \
    --max-queued-batches 8 \
    --num-workers 1 \
    --prefetch-factor 2 \
    --max-transfer-workers 4 \
    --verifier-name-or-path "$MODEL" \
    --from-pretrained "$RESUME_CKPT" \
    --trust-remote-code \
    --data-path "$DATA_PATH" \
    --train-data-ratio 0.999 \
    --epochs 5 \
    --lr 1.0e-4 \
    --optimizer muon \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --total-seq-len "$PACK_SEQ_LENGTH" \
    --speculator-type "dspark" \
    --block-size 8 \
    --max-anchors 1024 \
    --target-layer-ids $TARGET_LAYER_IDS \
    --draft-attn-impl sdpa \
    --markov-rank 256 \
    --markov-head-type "vanilla" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --confidence-head-alpha 1.0 \
    --mask-token-id $MASK_TOKEN_ID \
    --scheduler-type cosine \
    --scheduler-warmup-steps 100 \
    --weight-decay 0.0 \
    --dflash-decay-gamma 4.0 \
    --gradient-checkpointing \
    --checkpoint-freq 0.005 \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$OUTPUT_DIR/logs" \
    --log-freq 1 \
    --logger tensorboard &

TORCHRUN_PID=$!

cleanup() {
    echo "Stopping training..."
    if [[ -n "${TORCHRUN_PID:-}" ]] && kill -0 "$TORCHRUN_PID" 2>/dev/null; then
        kill -TERM "$TORCHRUN_PID" 2>/dev/null || true
        wait "$TORCHRUN_PID" 2>/dev/null || true
    fi
}

trap cleanup INT TERM EXIT

wait "$TORCHRUN_PID"

echo "Checkpoints saved to $OUTPUT_DIR/checkpoints2/"
echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') Training done."
