# dspark mooncake 2-node deployment config
# Pipeline: 00_free_npus -> 01_start_mooncake -> 02_start_vllm_instances -> 03_start_lb
#           -> 04_smoke -> 05_start_training -> 06_status / 07_stop_training / 08_stop_all

# 脚本所在目录 (即 step2_train_dspark_mooncake), 与部署路径解耦
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# --- misc ---
LOG_DIR="$BASE_DIR/logs"

# --- training (由 05-07 使用; 集群侧按节点填写) ---
# TRAIN_IP=
# TRAIN_LOG=
# OUTPUT_DIR=
