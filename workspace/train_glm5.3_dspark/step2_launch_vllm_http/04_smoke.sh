#!/usr/bin/env bash
# 04_smoke.sh -- 冒烟测试 (在 02/03 之后运行):
#   Phase A: 各后端 health + 直连生成
#   Phase B: LB 生成
#   Phase C: HTTP 服务器端口探活
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

echo "=== Phase C: HTTP server ==="
if timeout 2 bash -c "</dev/tcp/${HTTP_SERVER_IP}/${HTTP_SERVER_PORT}" 2>/dev/null; then
    echo "[http] server ${HTTP_SERVER_IP}:${HTTP_SERVER_PORT} reachable"
else
    echo "[http] server NOT reachable"; fail=1
fi

rm -f /tmp/smoke_out.json
if [ "$fail" -ne 0 ]; then
    echo "=== SMOKE FAILED ===" >&2; exit 1
fi
echo "=== SMOKE PASSED (note: 完整 HTTP 写入链路由训练首步验证) ==="
