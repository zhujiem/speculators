# dspark http 2-node deployment config
# Pipeline: 00_free_npus -> 01_start_http_server -> 02_start_vllm_instances -> 03_start_lb
#           -> 04_smoke -> 05_start_training -> 06_status / 07_stop_training / 08_stop_all

# --- training (由 05-07 使用; 集群侧按节点填写) ---
# TRAIN_IP=
# TRAIN_LOG=
# OUTPUT_DIR=

# 脚本所在目录 (即 step2_train_dspark_http), 与部署路径解耦
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

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
