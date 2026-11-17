#!/bin/bash
set -euo pipefail

MODEL_PATH=/mnt/cephfs/weight/GLM-5.3-w4a8-spec/

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve $MODEL_PATH \
--host 0.0.0.0 \
--port 8001 \
--api-server-count 1 \
--data-parallel-size 1 \
--enable-expert-parallel \
--tensor-parallel-size 16 \
--seed 1024 \
--served-model-name glm-5.3 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--max-num-seqs 16 \
--max-model-len 204800 \
--max-num-batched-tokens 16384 \
--enable-chunked-prefill \
--no-enable-prefix-caching \
--safetensors-load-strategy prefetch \
--async-scheduling \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--no-enable-prefix-caching \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_dsa_cp": false, "enable_sparse_sfa_c8": false, "enable_sparse_li_c8": false, "enable_balance_scheduling": true,"multistream_overlap_shared_expert":true,"enable_flashcomm1": true}' \
--speculative-config '{"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager":true}'
