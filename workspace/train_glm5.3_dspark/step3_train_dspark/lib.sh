#!/usr/bin/env bash
# Helpers for the training control scripts. Source env.sh first.

remote() {
    local ip="$1"; shift
    SSHPASS="${SSH_PASS}" sshpass -e ssh \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=15 \
        -o UserKnownHostsFile=/dev/null \
        "${SSH_USER}@${ip}" "$@"
}

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
