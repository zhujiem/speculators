#!/usr/bin/env bash
# 04_smoke.sh -- 冒烟测试 (在 02/03 之后运行):
#   Phase A: 各后端 health + 直连生成
#   Phase B: LB 生成
#   Phase C: mooncake master 端口探活
#   Phase D: 训练端隐状态端到端拉取 (vllm 生成 -> mooncake 传输 -> 训练端读取)
# Usage: bash 04_smoke.sh
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source env.sh; source lib.sh

gen_req() { # url  -> prints "code latency_s" on stdout
    local url="$1"
    curl -s -o /tmp/smoke_out.json -w "%{http_code} %{time_total}" \
        --max-time 120 "$url/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"你好，简单介绍一下你自己。\", \"max_tokens\": 16, \"temperature\": 0}"
}

check_resp() { # returns 0 if response has non-empty choices text
    python3 -c '
import json,sys
try:
    d = json.load(open("/tmp/smoke_out.json"))
    t = d["choices"][0]["text"]
    sys.exit(0 if t and t.strip() else 1)
except Exception:
    sys.exit(1)'
}

fail=0

echo "=== Phase A: per-backend health + direct generation ==="
for ip in "${INFER_IPS[@]}"; do
    url="http://${ip}:${SVC_PORT}"
    if curl -sf --max-time 5 "$url/v1/models" >/dev/null 2>&1; then
        echo "[$ip] models OK"
    else
        echo "[$ip] MODELS FAIL"; fail=1; continue
    fi
    out=$(gen_req "$url")
    code=${out%% *}; lat=${out##* }
    if [ "$code" = "200" ] && check_resp; then
        echo "[$ip] generation OK (${lat}s)"
    else
        echo "[$ip] generation FAIL (code=$code, ${lat}s)"; fail=1
    fi
done

echo "=== Phase B: LB round-robin generation x3 ==="
if curl -sf --max-time 5 "http://${LB_IP}:${LB_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[LB] models OK"
else
    echo "[LB] MODELS FAIL"; fail=1
fi
for i in 1 2 3; do
    out=$(gen_req "http://${LB_IP}:${LB_PORT}")
    code=${out%% *}; lat=${out##* }
    if [ "$code" = "200" ] && check_resp; then
        echo "[LB] request $i OK (${lat}s)"
    else
        echo "[LB] request $i FAIL (code=$code, ${lat}s)"; fail=1
    fi
done

echo "=== Phase C: mooncake master ==="
if timeout 2 bash -c "</dev/tcp/${MOONCAKE_MASTER_IP}/${MOONCAKE_MASTER_PORT}" 2>/dev/null; then
    echo "[mooncake] master ${MOONCAKE_MASTER_IP}:${MOONCAKE_MASTER_PORT} reachable"
else
    echo "[mooncake] master NOT reachable"; fail=1
fi

echo "=== Phase D: mooncake hs fetch (端到端) ==="
# vllm 生成 -> mooncake 传输 -> 训练端读取
in_container python3 - << 'PYEOF'
import os, asyncio, socket, sys

import torch
torch.zeros(1, device="npu")

import openai
from hs_connectors.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeStoreConfig,
)

# 1. setup mooncake store (训练端)
store = MooncakeHiddenStatesStore(MooncakeStoreConfig(
    local_hostname=os.environ.get(
        "MOONCAKE_LOCAL_HOSTNAME", socket.gethostbyname(socket.gethostname())
    ),
    metadata_server="P2PHANDSHAKE",
    master_server_address="${MOONCAKE_MASTER}",
    protocol="ascend",
))
store.setup()
print("[1/3] mooncake store setup OK")

# 2. send generation request via LB (触发 vllm 提取隐状态并写入 mooncake)
async def fetch():
    client = openai.AsyncOpenAI(base_url="http://${LB_IP}:${LB_PORT}/v1", api_key="none")
    return await client.completions.create(
        model="${MODEL_NAME}",
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

# 3. fetch hidden states from mooncake
result = store.get_sample(kv, timeout=30)
hs = result.get("hidden_states")
if hs is None or hs.numel() == 0:
    print("ERROR: hidden_states empty"); sys.exit(1)
print(f"[3/3] hs fetch OK: shape={list(hs.shape)}, dtype={hs.dtype}")
print("mooncake hs fetch end-to-end PASSED")
PYEOF
if [ $? -eq 0 ]; then
    echo "[mooncake] hs fetch OK"
else
    echo "[mooncake] hs fetch FAILED"; fail=1
fi

rm -f /tmp/smoke_out.json
if [ "$fail" -ne 0 ]; then
    echo "=== SMOKE FAILED ===" >&2; exit 1
fi
echo "=== SMOKE PASSED (推理链路 + 隐状态端到端均验证) ==="
