#!/bin/sh
# Fetch the model weights into the mounted volume on first start.
#
# The installers verify each file against the checksum pinned in
# models/manifest.json and are a no-op once the files are present, so this runs
# on every boot and only does work when the volume is empty.
#
# No download failure may stop the service: each engine reports itself
# unavailable with a reason and the app falls back to the deterministic
# resampler, which is better than refusing to start.
set -eu

# A pre-release image ran as root, so an existing development volume may still
# have root-only ownership. Fail with the exact one-time migration instead of
# starting privileged or failing later while an upload is in progress.
require_writable_dir() {
    data_dir="$1"
    if [ ! -d "${data_dir}" ] || [ ! -w "${data_dir}" ]; then
        echo "entrypoint: ${data_dir} must be writable by container uid $(id -u)" >&2
        if [ "$(id -u)" = "10001" ]; then
            echo "entrypoint: migrate old Compose volumes with:" >&2
            echo "  docker compose run --rm --user root --entrypoint chown upscaler -R 10001:10001 /weights /work" >&2
        else
            echo "entrypoint: check the owner of the configured bind-mounted work directory" >&2
        fi
        exit 70
    fi
}

require_writable_dir "${UPSCALER_WORK_ROOT:-/work}"
if [ "${UPSCALER_SKIP_WEIGHTS:-0}" != "1" ]; then
    require_writable_dir "${UPSCALER_REALESRGAN_WEIGHTS_DIR:-/weights}"
fi

if [ "${UPSCALER_SKIP_WEIGHTS:-0}" = "1" ]; then
    echo "entrypoint: UPSCALER_SKIP_WEIGHTS=1, not fetching weights"
else
    if [ "${UPSCALER_FETCH_REALESRGAN:-0}" = "1" ]; then
        if ! python3 /app/scripts/install-weights.py --group realesrgan \
                --dir "${UPSCALER_REALESRGAN_WEIGHTS_DIR:-/weights}"; then
            echo "entrypoint: Real-ESRGAN weights failed; the CUDA engine will be unavailable" >&2
        fi
    fi

    # Fetch flags make image behavior explicit. Import checks still protect a
    # custom build whose selected extras and requested weights disagree.
    if [ "${UPSCALER_FETCH_SWINIR:-0}" = "1" ] \
            && python3 -c "import spandrel" 2>/dev/null; then
        if ! python3 /app/scripts/install-weights.py --group swinir \
                --dir "${UPSCALER_SR_WEIGHTS_DIR:-/weights/spandrel-sr}"; then
            echo "entrypoint: SwinIR weights failed; Upscale falls back to Real-ESRGAN" >&2
        fi
    fi
fi
exec "$@"
