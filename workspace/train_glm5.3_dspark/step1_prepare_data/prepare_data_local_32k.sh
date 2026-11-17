#!/bin/bash
set -euo pipefail

MODEL="/mnt/cephfs/weight/GLM-5.3-w4a8-spec"
DATA_PATH="/mnt/cephfs/hw/spectrain/data/train_normal.jsonl"
OUTPUT_DIR="/mnt/cephfs/hw/spectrain/data/train_1110_32k_v2"
SEQ_LENGTH=32768
MIN_VALID_TOKENS=128
OVERWRITE=0

export TMPDIR="/tmp"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OVERWRITE_FLAG=""
if [[ "$OVERWRITE" -eq 1 ]]; then OVERWRITE_FLAG="--overwrite"; fi

echo "=== Step 2: preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATA_PATH" \
    --render-endpoint local \
    --seed 4242 \
    --trust-remote-code \
    --output "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH" \
    --minimum-valid-tokens "$MIN_VALID_TOKENS" \
    --num-preprocessing-workers 480 \
    --max-samples 500000 \
    $OVERWRITE_FLAG  

echo "All done. Prepared data at $OUTPUT_DIR"
