# Deployment

## Quick start with Docker

The default image is deliberately CPU-safe: it contains the complete web app,
but no CUDA runtime, PyTorch, neural weights, or GPU reservation. Nothing needs
to be installed on the host beyond Docker itself.

```bash
make up
```

Then open `http://127.0.0.1:8000`, or see
[Running on a remote machine](#running-on-a-remote-machine) to reach it over SSH.

This starts the deterministic Upscale and Sharpen paths on any ordinary Docker
host. It downloads no model weights.

For NVIDIA acceleration, use the explicit CUDA overlay. It requires the NVIDIA
Container Toolkit and selects the digest-pinned CUDA 13.0 runtime, the locked
PyTorch stack, and SwinIR:

```bash
make up-cuda

# Confirm the toolkit before the first build.
docker run --rm --gpus all \
  nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:c3fde347d52d578c84fd644bc177bc7ec333feaf11550d990da4084d7612e4c7 \
  nvidia-smi
```

The CUDA build and its first-start model downloads are several gigabytes. Image
layers and checksum-verified weights are cached separately, so a rebuild does
not empty the named weights volume. Every extra this image can select is
permissively licensed. The CPU image remains the supported lean image.

If ComfyUI is installed at `~/comfy/ComfyUI`, start it and the connected app in
one command:

```bash
make up-comfyui
```

This uses ComfyUI's own `.venv`, keeps both HTTP services on `127.0.0.1`, builds
the app with its ComfyUI connector, and mounts only the ComfyUI input directory
read-write so uploaded sources can be removed after a job. The connector runs as
your host uid/gid rather than root, with job scratch data in the ignored
`.upscaler/comfyui-container-work/` bind mount. The log is written to
`.upscaler/comfyui.log`; ComfyUI runs as a transient user service when systemd is
available, with a PID-managed background process as the fallback. Use
`make down-comfyui` to stop both processes. A different installation or port can
be selected with, for example:

```bash
make up-comfyui COMFYUI=/opt/ComfyUI COMFYUI_PORT=8189
```

The `make setup-*` targets in [Host installation](install.md) are for **host installs
only**. They install into a local `.venv` and `.upscaler/`, which a container ignores
entirely.

The published Compose path is one reviewed pair: the CUDA 13.0 base and the
CUDA 13 package set pinned in `uv.lock`. It is intentionally not retargetable by
an environment variable. A host that needs CUDA 12.8 can use the host-install
command in [CUDA GPUs](install.md#cuda-gpus-including-arm64-hosts-such-as-dgx-spark),
which selects the matching upstream wheel index explicitly.

Useful commands:

```bash
make up         # CPU-safe image; build and start detached
make up-cuda    # explicit NVIDIA image and GPU reservation
make logs       # follow logs
make shell      # shell inside the container
make down       # stop and remove
```

The image is configured as uid/gid `10001:10001`; image decoding, inference,
and downloads never run as root. If an older development volume was created by
the former root-running image, migrate its ownership once:

```bash
docker compose run --rm --user root --entrypoint chown upscaler \
  -R 10001:10001 /weights /work
```

Confirm which engine was selected:

```bash
curl -s http://127.0.0.1:8000/api/v1/capabilities | python3 -m json.tool
```

## Hardware-aware deployment policy

The default `UPSCALER_HARDWARE_POLICY=safe` exposes only installed engines and
ComfyUI workflows that fit the detected machine. The hardware panel shows effective RAM,
GPU/VRAM or unified memory, current availability, policy status, and the reasons omitted
features were excluded. CPU Upscale and Sharpen remain available when no GPU can be used.

Detection reads physical RAM and Linux cgroup limits, then probes CUDA with PyTorch and
`nvidia-smi`, and Vulkan with `vulkaninfo` and DRM sysfs. A configured ComfyUI is evaluated
from its own `/system_stats` response, which matters when it is outside the app container.
Container memory limits therefore reduce effective RAM even when the host has more.

The checked-in safe-policy v1 floors are conservative:

| Engine or workflow | VRAM | RAM | Unified memory |
| --- | ---: | ---: | ---: |
| CPU resample / sharpen | 0 GiB | 2 GiB | 2 GiB |
| NCNN Real-ESRGAN | 2 GiB | 4 GiB | 6 GiB |
| SwinIR-L / CUDA Real-ESRGAN / Illustration | 4 GiB | 8 GiB | 10 GiB |

The ComfyUI graph exposes no app tile control, because its tiling is part of the checked-in
workflow. Automatic tile sizing resolves to a concrete capacity-checked size, which is
recorded in the result.

Visibility and admission intentionally use different measurements. Choices remain stable
based on total capacity. After the upload is saved and its oriented decoded dimensions are
known, submission recomputes the image working set and refreshes currently free memory. A
dedicated GPU must retain the 4 GiB RAM and 1.5 GiB VRAM reserves; unified-memory systems are
checked once against one shared pool. If an idle ComfyUI's cached models make that check fail,
the app asks it to unload them, refreshes its memory report, and retries admission once. A job
that still does not fit receives HTTP 409 and its temporary upload is removed.

Configuration:

| Variable | Default | Meaning |
| --- | --- | --- |
| `UPSCALER_HARDWARE_POLICY` | `safe` | `safe` filters and admits by capacity; `off` restores runtime/model-only availability. |
| `UPSCALER_RAM_RESERVE_MIB` | `4096` | RAM left free during live admission. |
| `UPSCALER_VRAM_RESERVE_MIB` | `1536` | Dedicated VRAM left free during live admission. |
| `UPSCALER_RAM_MIB` | detected | Correct effective RAM capacity when container/OS reporting is wrong. |
| `UPSCALER_VRAM_MIB` | detected | Correct GPU capacity; also enables safe GPU choices when capacity was unknown. |
| `UPSCALER_GPU_NAME` | detected | Correct the display name. |
| `UPSCALER_MEMORY_KIND` | detected | Force `dedicated` or `unified`. Use this when a shared-memory accelerator is misclassified. |

These variables are passed through by `docker-compose.yml`. For example, a 128 GiB GB10
whose driver does not classify its shared pool correctly can be started with:

```bash
UPSCALER_RAM_MIB=131072 UPSCALER_VRAM_MIB=131072 \
UPSCALER_GPU_NAME="NVIDIA GB10" UPSCALER_MEMORY_KIND=unified make up-cuda
```

Safe mode is a versioned conservative policy, not a guarantee that a third-party driver or
custom workflow cannot run out of memory. Use `UPSCALER_HARDWARE_POLICY=off` only when you
want to manage those risks yourself.

Installing on the host directly, which is only necessary for development, is covered in
[Host installation](install.md).

## Running on a remote machine

The service binds to `127.0.0.1` and rejects requests whose `Host` header is not
a loopback name, so the supported way to use it from another computer is an SSH
tunnel — the app stays unexposed and SSH provides transport security:

```bash
# on the server
make up                          # `make up-cuda` for NVIDIA, or `make run` on the host

# on your computer
ssh -N -L 8000:127.0.0.1:8000 you@your-server
```

The compose file publishes the port on the host's loopback only
(`127.0.0.1:8000:8000`). The container binds `0.0.0.0` internally because it must
accept traffic from the Docker bridge; that is not exposure, and the Host check
still rejects anything that does not arrive as a loopback name.

Then open `http://127.0.0.1:8000`. Use `127.0.0.1` rather than `localhost`, which
may resolve to `::1` and miss the forward.
