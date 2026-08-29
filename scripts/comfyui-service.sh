#!/bin/sh
# Start, stop, and preflight the ComfyUI that `make up` runs alongside the app.
#
# ComfyUI is a separate application, so this project starts a copy rather than
# containing one. Where it lives was decided once by `make setup-comfyui` and
# recorded in .upscaler/comfyui.conf; nothing here asks the user to name it
# again.
#
# usage: comfyui-service.sh check|start|stop [ROOT] [PORT]
set -eu

usage() {
    echo "usage: $0 check|start|stop [COMFYUI_ROOT] [PORT]" >&2
    exit 2
}

[ "$#" -ge 1 ] && [ "$#" -le 3 ] || usage
action=$1
case "$action" in
    check|start|stop) ;;
    *) usage ;;
esac

script_dir=$(CDPATH='' cd "$(dirname "$0")" && pwd -P)
state_dir=$(dirname "$script_dir")/.upscaler
conf_file=$state_dir/comfyui.conf
pid_file=$state_dir/comfyui.pid
log_file=$state_dir/comfyui.log
env_file=$state_dir/comfyui.env

# Read one key out of the installer's record. Parsed rather than sourced: this
# file names a path, and nothing that names a path should be executed.
conf_value() {
    [ -f "$conf_file" ] || return 0
    sed -n "s/^$1=//p" "$conf_file" | head -1
}

configured_root=${2:-$(conf_value COMFYUI_ROOT)}
port=${3:-$(conf_value COMFYUI_PORT)}
port=${port:-8188}
model=$(conf_value COMFYUI_MODEL)

if [ -z "$configured_root" ]; then
    echo "ComfyUI has not been set up. Run \`make setup-comfyui\` to install or adopt it." >&2
    exit 3
fi
case "$port" in
    ''|*[!0-9]*) echo "ComfyUI port must be a number: $port" >&2; exit 2 ;;
esac

if [ ! -d "$configured_root" ]; then
    echo "ComfyUI is recorded at $configured_root, which no longer exists." >&2
    echo "Run \`make setup-comfyui\` to install or adopt it again." >&2
    exit 3
fi
comfyui_root=$(CDPATH='' cd "$configured_root" && pwd -P)
python="$comfyui_root/.venv/bin/python"
main="$comfyui_root/main.py"

# Everything the Illustration mode needs before it can resolve, each with the
# command that fixes it. Exit 3 throughout: `make up` treats that as "carry on
# without Illustration" rather than as a failure to start the application.
run_check() {
    missing=0
    if [ ! -x "$python" ]; then
        echo "  missing: $python (the ComfyUI virtualenv)" >&2
        missing=1
    fi
    if [ ! -f "$main" ]; then
        echo "  missing: $main (the ComfyUI launcher)" >&2
        missing=1
    fi
    if [ ! -d "$comfyui_root/input" ]; then
        echo "  missing: $comfyui_root/input" >&2
        missing=1
    fi
    if [ -n "$model" ] && [ ! -f "$comfyui_root/models/upscale_models/$model" ]; then
        echo "  missing: models/upscale_models/$model (the illustration weight)" >&2
        missing=1
    fi
    if [ "$missing" = 1 ]; then
        echo "Run \`make setup-comfyui\` to repair the installation at $comfyui_root." >&2
        return 3
    fi
    return 0
}

if [ "$action" = check ]; then
    run_check
    echo "ComfyUI at $comfyui_root is ready to start on port $port."
    exit 0
fi

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
    rm -f "$env_file"
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

run_check

# Where the application container can reach this process.
#
# ComfyUI's default binding is the host's own loopback, which nothing inside a
# bridge-networked container can open: the app would get "Connection refused"
# and report Illustration unavailable with no hint why. host.docker.internal
# resolves to the Docker bridge gateway, so ComfyUI listens there as well.
#
# A bare `--listen` would solve it by binding 0.0.0.0 and publishing an
# unauthenticated ComfyUI to every network this machine is on. That is a far
# larger concession than the app needs, so it is never the fallback: without a
# gateway this binds loopback only and says what will not work.
listen=127.0.0.1
gateway=$(docker network inspect bridge \
    --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null || true)
case "$gateway" in
    *[!0-9.]*|'') 
        echo "Could not determine the Docker bridge gateway." >&2
        echo "Binding loopback only; the container will not reach ComfyUI." >&2
        gateway=
        ;;
    *) listen="127.0.0.1,$gateway" ;;
esac

write_env() {
    umask 077
    {
        if [ -n "$gateway" ]; then
            echo "UPSCALER_COMFYUI_URL=http://host.docker.internal:$port"
        fi
        echo "UPSCALER_COMFYUI_INPUT_DIR=$comfyui_root/input"
        echo "UPSCALER_UID=$(id -u)"
        echo "UPSCALER_GID=$(id -g)"
    } > "$env_file"
}

if is_healthy; then
    echo "ComfyUI is already available at http://127.0.0.1:$port"
    write_env
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
    echo "Starting ComfyUI from $comfyui_root (listening on $listen:$port)"
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
            --listen "$listen" \
            --port "$port" \
            --disable-auto-launch
        managed_backend=systemd
    else
        (
            cd "$comfyui_root"
            nohup "$python" "$main" \
                --listen "$listen" \
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

write_env
echo "ComfyUI is ready at http://127.0.0.1:$port"
