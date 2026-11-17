#!/bin/bash

START_TS=$(date +%s)

DATADIR=/mnt/cephfs/hw/specbench/agentbench_data
RESULT_DIR=/mnt/cephfs/hw/specbench/result
BASE_URL=http://127.0.0.1:8001
MODEL=glm-5.3
TS=$(date +%Y%m%d_%H%M%S)

#for ds in agentbench_64k_128k agentbench_128k_256k; do
for ds in gsm8k_prompt agentbench_0k_8k agentbench_8k_16k agentbench_16k_32k agentbench_32k_64k agentbench_64k_128k agentbench_128k_256k; do
  echo "======== Running bench: dataset=${ds} ========"
  vllm bench serve \
    --model "${MODEL}" \
    --tokenizer "/mnt/cephfs/weight/GLM-5.3-w4a8-spec/" \
    --dataset-name custom \
    --max-concurrency 1 \
    --base-url "${BASE_URL}" \
    --save-result \
    --save-detailed \
    --result-dir "${RESULT_DIR}" \
    --result-filename "dspark_result_${MODEL}_${ds}_${TS}.json" \
    --dataset-path "${DATADIR}/${ds}.jsonl" \
    --trust-remote-code \
    --num-prompts 50 \
    --no-oversample \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --skip-chat-template \
    --disable-shuffle \
    --custom-output-len 256 \
    --seed 42
  echo "======== Finished: dataset=${ds} ========"
done

END_TS=$(date +%s)
TOTAL=$((END_TS - START_TS))
printf "All datasets done. Total time: %dh %02dm %02ds (%ds)\n" \
  $((TOTAL/3600)) $(((TOTAL%3600)/60)) $((TOTAL%60)) "${TOTAL}"
