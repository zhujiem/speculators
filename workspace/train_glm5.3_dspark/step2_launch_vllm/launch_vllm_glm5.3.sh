#!/bin/bash
# launch_vllm_glm5.3.sh -- 单节点 vllm 抽取服务启动器 (由 02_start_vllm_services.sh 在各推理
# 节点容器内调用; hidden-states 走 HTTP backend)
# usage: bash launch_vllm_glm5.3.sh <http_server_ip>
set -u

NODE0_IP=${1:?}
LOCAL_IP=$(hostname -I | awk '{print $1}')
GPU_MEM=${GPU_MEM:-0.85}
NIC=bond0

MODEL=/mnt/cephfs/weight/GLM-5.3-w4a8-spec
MODEL_NAME=/mnt/cephfs/weight/GLM-5.3
HTTP_SERVER="$NODE0_IP:50088"
OUT=/mnt/cephfs/hw/runs/
SPECULATOR_HOME=/mnt/cephfs/hw/speculators

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC"
export TP_SOCKET_IFNAME="$NIC"
export HCCL_SOCKET_IFNAME="$NIC"
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=1024
export HCCL_TRANSFER_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=3600
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_FASTOKENS=1
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONHASHSEED=0
export TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=$OUT/cache/torchinductor
export VLLM_CACHE_ROOT=$OUT/cache/vllm
export HF_HOME=$OUT/cache/hf
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$HF_HOME"

exec python "$SPECULATOR_HOME"/scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids 1 20 38 56 75 \
    --hidden-states-backend http \
    --http-endpoint "$HTTP_SERVER" \
    --http-connection-pool-size 8 \
    --http-max-chunk-mib 512 \
    --http-writer-threads 4 \
    --trust-remote-code \
    --provenance-dir "${OUT}/vllm_launch/${LOCAL_IP}" \
    --no-hash-checkpoints \
    -- \
       --served-model-name "$MODEL_NAME" \
       --host 0.0.0.0 \
       --port 8008 \
       --safetensors-load-strategy prefetch \
       --data-parallel-size 1 \
       --tensor-parallel-size 16 \
       --enable-expert-parallel \
       --seed 1024 \
       --tool-call-parser glm47 \
       --reasoning-parser glm45 \
       --enable-auto-tool-choice \
       --enable-chunked-prefill \
       --enable-prefix-caching \
       --max-num-batched-tokens 32768 \
       --max-num-seqs 2 \
       --max-model-len 32769 \
       --gpu-memory-utilization "$GPU_MEM" \
       --quantization ascend \
       --enforce-eager \
       --trust-remote-code \
       --additional-config '{"enable_flashcomm1": true, "multistream_overlap_shared_expert": true}'
