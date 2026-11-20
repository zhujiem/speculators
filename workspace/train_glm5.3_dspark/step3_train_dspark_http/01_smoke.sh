#!/bin/bash
# 01_smoke.sh -- 训练前冒烟: 验证 LB + HTTP 隐状态传输 (在 02_start_training 前运行)。
# usage: bash 01_smoke.sh 
set -u
cd "$(dirname "$0")"
source env.sh
source lib.sh

fail=0
line() { echo "============================================================"; }

echo "=== Phase A: LB health ==="
if remote "${LB_IP}" "curl -sf --max-time 5 http://${LB_IP}:${LB_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[LB] http://${LB_IP}:${LB_PORT} OK"
else
    echo "[LB] http://${LB_IP}:${LB_PORT} NOT reachable"; fail=1
fi

echo "=== Phase B: LB generation x2 ==="
for i in 1 2; do
    resp=$(remote "${LB_IP}" "curl -s --max-time 60 http://${LB_IP}:${LB_PORT}/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"/mnt/cephfs/weight/GLM-5.3\", \"prompt\": \"hello\", \"max_tokens\": 2}'" 2>/dev/null)
    if echo "$resp" | grep -q '"choices"'; then
        echo "[LB] request $i OK (code=200, choices present)"
    else
        echo "[LB] request $i FAILED: $resp"; fail=1
    fi
done

echo "=== Phase C: HTTP server ==="
if remote "${HTTP_SERVER_IP}" "timeout 2 bash -c '</dev/tcp/${HTTP_SERVER_IP}/${HTTP_SERVER_PORT}'" 2>/dev/null; then
    echo "[http] server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} reachable"
else
    echo "[http] server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} NOT reachable"; fail=1
fi

echo "=== Phase D: HTTP hs fetch (端到端) ==="
# vllm 生成 -> HTTP 传输 -> 训练端读取
in_container python3 - << 'PYEOF'
import os, asyncio, sys


import torch
torch.zeros(1, device="npu")

import openai
from hs_connectors.http_store import HttpStore, HttpStoreConfig

# 1. setup http store (训练端)
store = HttpStore(HttpStoreConfig(endpoint="${HTTP_SERVER_IP}:${HTTP_SERVER_PORT}"))
store.setup()
print("[1/3] http store setup OK")

# 2. send generation request via LB (触发 vllm 提取隐状态并写入 HTTP)
async def fetch():
    client = openai.AsyncOpenAI(base_url="http://${LB_IP}:${LB_PORT}/v1", api_key="none")
    return await client.completions.create(
        model="/mnt/cephfs/weight/GLM-5.3",
        prompt="The quick brown fox jumps over the lazy dog.",
        max_tokens=1,
        extra_body={"return_token_ids": True},
        timeout=120,
    )

resp = asyncio.run(fetch())
kv = getattr(resp, "kv_transfer_params", None)
if kv is None:
    print("ERROR: no kv_transfer_params"); sys.exit(1)
print(f"[2/3] vllm generation OK, handle={kv.get('handle', 'N/A')[:25]}")

# 3. fetch hidden states from HTTP
result = store.get_sample(kv, timeout=30)
hs = result.get("hidden_states")
if hs is None or hs.numel() == 0:
    print("ERROR: hidden_states empty"); sys.exit(1)
print(f"[3/3] hs fetch OK: shape={list(hs.shape)}, dtype={hs.dtype}")
print("http hs fetch end-to-end PASSED")
PYEOF
if [ $? -eq 0 ]; then
    echo "[http] hs fetch OK"
else
    echo "[http] hs fetch FAILED"; fail=1
fi

line
if [ $fail -eq 0 ]; then
    echo "=== SMOKE PASSED (note: 完整 HTTP 写入链路由训练首步验证) ==="
else
    echo "=== SMOKE FAILED ==="
fi
exit $fail
