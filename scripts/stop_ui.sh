#!/usr/bin/env bash
set -euo pipefail

runtime_home="${XDG_RUNTIME_DIR:-/tmp/visionmodelquest-$(id -u)}"
runtime_root="$runtime_home/visionmodelquest"

stop_recorded_process() {
    local pid_file="$1"
    local expected="$2"
    local label="$3"
    [[ -f "$pid_file" ]] || return 0
    local pid
    pid="$(sed -n 's/.*"pid":[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p' "$pid_file" | head -n 1)"
    if [[ -z "$pid" || "$pid" -le 1 ]]; then
        printf 'Invalid %s PID record; no process was stopped.\n' "$label" >&2
        return 1
    fi
    if [[ ! -r "/proc/$pid/cmdline" ]]; then
        rm -f -- "$pid_file"
        return 0
    fi
    local command_line
    command_line="$(tr '\\0' ' ' < "/proc/$pid/cmdline")"
    if [[ "$command_line" != *"$expected"* ]]; then
        printf 'Recorded PID %s is not the expected %s; no process was stopped.\n' "$pid" "$label" >&2
        return 1
    fi
    kill -- "-$pid" 2>/dev/null || kill "$pid"
    rm -f -- "$pid_file"
    printf 'Stopped %s PID %s.\n' "$label" "$pid"
}

stop_recorded_process "$runtime_root/worker.pid" "visionmodelquest.experiment_worker" "worker"
stop_recorded_process "$runtime_root/application.pid" "visionmodelquest.ui.application" "application"
