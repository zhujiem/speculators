#!/bin/bash
set -euo pipefail

MODEL="/mnt/data1/GLM-5.3-BF16-w4a8c8/"
DATASET="/mnt/data/hw/spectrain/data/train_10k.jsonl"
OUTPUT_DIR="/mnt/data/hw/spectrain/output/glm5.3_dspark_32k_v1"
SEQ_LENGTH=16384 # maximal tokens of prompt + answser
MIN_VALID_TOKENS=128  # minimal answer tokens

export TMPDIR="/tmp"
export TORCHINDUCTOR_CACHE_DIR="${OUTPUT_DIR}/cache/torchinductor"
export HF_HOME="${OUTPUT_DIR}/cache/hf"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

echo "=== Step 2: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --render-endpoint "http://127.0.0.1:8008" \
    --output "${OUTPUT_DIR}/data" \
    --seq-length $SEQ_LENGTH \
    --minimum-valid-tokens $MIN_VALID_TOKENS
    --num-preprocessing-workers 16 \
    --trust-remote-code 

echo "All done. Prepared data at ${OUTPUT_DIR}/data"

