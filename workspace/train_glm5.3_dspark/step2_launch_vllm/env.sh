# dspark http 2-node deployment config
# Pipeline: 00_free_npus -> 01_start_vllm -> 03_start_lb -> 04_smoke

BASE_DIR="/mnt/cephfs/hw/zhujieming/spectrain/step2_launch_vllm_2nodes_http"

# --- nodes ---
CONTAINER_NAME="vllm_ascend_v0.30.0"

# --- models ---
GLM53_MODEL="/mnt/cephfs/weight/GLM-5.3-w4a8-spec"

# --- service layout ---
VLLM_PORT=8008
SVC_READY_TIMEOUT=3600
LB_PORT=8100

# --- misc ---
LOG_DIR="$BASE_DIR/logs"

# --- HTTP server (hs_http_server.py) ---
HTTP_SERVER_MAX_BUFFER_GIB=128          # 128 GiB payload budget
STAGGER_S=60                # 错峰启动间隔 (秒)
MODEL_NAME="/mnt/cephfs/weight/GLM-5.3"   # vllm served-model-name
SVC_PORT=8008                    # vllm service port (per node)
