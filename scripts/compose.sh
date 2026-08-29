#!/bin/sh
# Run `docker compose` with the one configuration this project deploys.
#
# There is a single image and a single compose file. The only thing that varies
# between hosts is the GPU reservation, which Compose cannot make conditional:
# the device block fails outright where the request cannot be satisfied. So the
# decision is made here and docker-compose.gpu.yml is added when a GPU is really
# available, which keeps `make up` the same command on every machine.
#
# UPSCALER_GPU=1 forces the reservation on, UPSCALER_GPU=0 forces it off. The
# default, "auto", tries it.
set -eu

project_root=$(CDPATH='' cd "$(dirname "$0")/.." && pwd -P)
cd "$project_root"

# Must match the image name in docker-compose.yml: the probe below runs a real
# container, and this is the only one guaranteed to be on the machine.
image=local-image-upscaler

# Ask the daemon to actually hand a container a GPU.
#
# Nothing Docker reports statically is sufficient. A registered runtime named
# "nvidia" is one way to serve a device, but Docker also does it through CDI or
# its own device driver, and a host doing that has no nvidia entry in
# `docker info --format '{{json .Runtimes}}'`, no CDI spec directory contents,
# and a working `--gpus all` all at once. Reading any of those would tell this
# script the machine has no GPU while the machine plainly does, and the app
# would quietly run on the CPU. So this asks for the thing it wants, the same
# way probe_cuda runs a real convolution rather than trusting is_available().
gpu_is_available() {
    docker run --rm --gpus all --entrypoint true "$image" >/dev/null 2>&1
}

# What comfyui-service.sh recorded when it started ComfyUI: the URL, the input
# directory, and the user that owns it. Parsed rather than sourced, and exported
# here so nobody has to put any of it on the `make up` line.
comfyui_env=.upscaler/comfyui.env
if [ -f "$comfyui_env" ]; then
    while IFS='=' read -r key value; do
        [ -n "$key" ] || continue
        export "$key=$value"
    done < "$comfyui_env"
fi

subcommand=${1:-}
gpu=${UPSCALER_GPU:-auto}
case "$gpu" in
    auto)
        # Only `up` needs the answer: no other subcommand reserves a device, and
        # `config` is validated against both values explicitly by `make
        # compose-config`. This keeps the probe to once per start.
        if [ "$subcommand" != up ]; then
            gpu=0
        else
            # The probe needs something to run. On a first start the image does
            # not exist yet, and `up` would have built it a moment later anyway.
            if ! docker image inspect "$image" >/dev/null 2>&1; then
                docker compose -f docker-compose.yml build
            fi
            if gpu_is_available; then
                gpu=1
            else
                gpu=0
            fi
        fi
        ;;
    0|1) ;;
    *)
        echo "compose: UPSCALER_GPU must be 0, 1, or auto (got '$gpu')" >&2
        exit 2
        ;;
esac

set -- -f docker-compose.yml "$@"
if [ "$gpu" = 1 ]; then
    set -- -f docker-compose.gpu.yml "$@"
fi

# Only when starting: the resolved mode is the one thing a user cannot see from
# the outside, and silence here is how someone ends up wondering why a job ran
# on the CPU. Other subcommands stay quiet so their output is still pipeable.
if [ "$subcommand" = up ]; then
    if [ "$gpu" = 1 ]; then
        echo "compose: a GPU is available, reserving it" >&2
    else
        echo "compose: no GPU available, starting without a device reservation" >&2
    fi
fi

exec docker compose "$@"
