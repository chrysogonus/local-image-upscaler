#!/bin/sh
# Start and stop the host ComfyUI used by `make up-comfyui`.
set -eu

usage() {
    echo "usage: $0 start|stop COMFYUI_ROOT [PORT]" >&2
    exit 2
}

[ "$#" -ge 2 ] && [ "$#" -le 3 ] || usage
action=$1
configured_root=$2
port=${3:-8188}

case "$action" in
    start|stop) ;;
    *) usage ;;
esac
case "$port" in
    ''|*[!0-9]*) echo "ComfyUI port must be a number: $port" >&2; exit 2 ;;
esac

if [ ! -d "$configured_root" ]; then
    echo "ComfyUI directory does not exist: $configured_root" >&2
    exit 2
fi
comfyui_root=$(CDPATH='' cd "$configured_root" && pwd -P)
python="$comfyui_root/.venv/bin/python"
main="$comfyui_root/main.py"

if [ ! -x "$python" ]; then
    echo "ComfyUI virtualenv Python is missing: $python" >&2
    exit 2
fi
if [ ! -f "$main" ]; then
    echo "ComfyUI launcher is missing: $main" >&2
    exit 2
fi
if [ ! -d "$comfyui_root/input" ]; then
    echo "ComfyUI input directory is missing: $comfyui_root/input" >&2
    exit 2
fi

script_dir=$(CDPATH='' cd "$(dirname "$0")" && pwd -P)
state_dir=$(dirname "$script_dir")/.upscaler
pid_file=$state_dir/comfyui.pid
log_file=$state_dir/comfyui.log
health_url=http://127.0.0.1:$port/system_stats
unit=upscaler-comfyui-$port.service

is_healthy() {
    "$python" -c \
        'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read(1)' \
        "$health_url" >/dev/null 2>&1
}

read_managed_pid() {
    managed_pid=
    if [ -f "$pid_file" ]; then
        IFS= read -r candidate < "$pid_file" || true
        case "$candidate" in
            ''|*[!0-9]*) rm -f "$pid_file" ;;
            *)
                if kill -0 "$candidate" 2>/dev/null; then
                    managed_pid=$candidate
                else
                    rm -f "$pid_file"
                fi
                ;;
        esac
    fi
}

has_user_systemd() {
    command -v systemctl >/dev/null 2>&1 \
        && command -v systemd-run >/dev/null 2>&1 \
        && systemctl --user show-environment >/dev/null 2>&1
}

if [ "$action" = stop ]; then
    if has_user_systemd && systemctl --user is-active --quiet "$unit"; then
        systemctl --user stop "$unit"
        rm -f "$pid_file"
        echo "ComfyUI stopped."
        exit 0
    fi

    read_managed_pid
    if [ -z "$managed_pid" ]; then
        echo "ComfyUI was not started by this project; leaving it alone."
        exit 0
    fi

    process_root=$(readlink -f "/proc/$managed_pid/cwd" 2>/dev/null || true)
    if [ "$process_root" != "$comfyui_root" ]; then
        echo "Refusing to stop PID $managed_pid: its working directory is not $comfyui_root" >&2
        rm -f "$pid_file"
        exit 1
    fi

    kill "$managed_pid"
    attempts=0
    while kill -0 "$managed_pid" 2>/dev/null && [ "$attempts" -lt 30 ]; do
        sleep 1
        attempts=$((attempts + 1))
    done
    if kill -0 "$managed_pid" 2>/dev/null; then
        echo "ComfyUI did not stop after 30 seconds; PID $managed_pid is still running." >&2
        exit 1
    fi
    rm -f "$pid_file"
    echo "ComfyUI stopped."
    exit 0
fi

if is_healthy; then
    echo "ComfyUI is already available at http://127.0.0.1:$port"
    exit 0
fi

mkdir -p "$state_dir"
managed_backend=
if has_user_systemd && systemctl --user is-active --quiet "$unit"; then
    managed_backend=systemd
else
    read_managed_pid
    if [ -n "$managed_pid" ]; then
        managed_backend=pid
    fi
fi

if [ -z "$managed_backend" ]; then
    echo "Starting ComfyUI from $comfyui_root"
    if has_user_systemd; then
        rm -f "$pid_file"
        systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
        systemd-run --user \
            --unit "$unit" \
            --collect \
            --service-type exec \
            --property "WorkingDirectory=$comfyui_root" \
            --property "StandardOutput=append:$log_file" \
            --property "StandardError=append:$log_file" \
            "$python" "$main" \
            --listen 127.0.0.1 \
            --port "$port" \
            --disable-auto-launch
        managed_backend=systemd
    else
        (
            cd "$comfyui_root"
            nohup "$python" "$main" \
                --listen 127.0.0.1 \
                --port "$port" \
                --disable-auto-launch >> "$log_file" 2>&1 &
            printf '%s\n' "$!" > "$pid_file"
        )
        read_managed_pid
        managed_backend=pid
    fi
fi

attempts=0
while ! is_healthy; do
    if [ "$managed_backend" = systemd ]; then
        still_running=$(systemctl --user is-active "$unit" 2>/dev/null || true)
        process_running=$([ "$still_running" = active ] && echo 1 || echo 0)
    elif [ -n "$managed_pid" ] && kill -0 "$managed_pid" 2>/dev/null; then
        process_running=1
    else
        process_running=0
    fi
    if [ "$process_running" != 1 ]; then
        rm -f "$pid_file"
        echo "ComfyUI exited during startup. See $log_file" >&2
        exit 1
    fi
    if [ "$attempts" -ge 60 ]; then
        echo "ComfyUI is still starting after 60 seconds. See $log_file" >&2
        exit 1
    fi
    sleep 1
    attempts=$((attempts + 1))
done

echo "ComfyUI is ready at http://127.0.0.1:$port"
