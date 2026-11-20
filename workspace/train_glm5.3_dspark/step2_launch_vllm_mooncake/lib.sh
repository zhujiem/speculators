#!/usr/bin/env bash
# Helpers shared by the 2-node control scripts. Source env.sh first.

remote() {
    local ip="$1"; shift
    SSHPASS="${SSH_PASS}" sshpass -e ssh \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=15 \
        -o UserKnownHostsFile=/dev/null \
        "${SSH_USER}@${ip}" "$@"
}

in_container() {
    local ip="$1"; shift
    remote "${ip}" "docker exec ${CONTAINER_NAME} $*"
}

# backend list "ip:port ..." for the LB, in stable RR order
lb_backend_list() {
    local out=()
    for ip in "${INFER_IPS[@]}"; do
        out+=("http://${ip}:${SVC_PORT}")
    done
    echo "${out[*]}"
}

# --- training-node helpers -----------------------------------------------------

my_ip() { hostname -I | awk '{print $1}'; }

on_train_node() { [ "$(my_ip)" = "${TRAIN_IP}" ]; }

# run a command on the training node (locally if we are on it, else via ssh)
train_run() {
    if on_train_node; then
        bash -c "$*"
    else
        remote "${TRAIN_IP}" "$*"
    fi
}

# latest global_step from the training log (0 if none)
train_step() {
    train_run "grep -oE 'global_step=[0-9]+' '${TRAIN_LOG}' 2>/dev/null | tail -1 | cut -d= -f2" 2>/dev/null | tr -d '\r'
}

# number of live training processes on the train node
train_procs() {
    train_run "ps aux | grep 'speculators.trai[n]' | wc -l" 2>/dev/null | tr -d '\r'
}
