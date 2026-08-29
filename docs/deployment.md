# Deployment

## Quick start

Docker is the only requirement, and there is one command:

```bash
make up
```

Then open `http://127.0.0.1:8000`, or see
[Running on a remote machine](#running-on-a-remote-machine) to reach it over SSH.

One image serves every host. It carries the complete web app together with the neural
stack: the digest-pinned CUDA 13.0 runtime, the locked PyTorch stack, SwinIR, and the
websocket client the Illustration mode uses. On a machine with the NVIDIA Container
Toolkit, `make up` reserves the GPU and **Upscale** resolves to SwinIR-L. On a machine
without one it starts the very same image; the engines probe for a device, find none, and
Upscale falls back to the deterministic Lanczos resampler and says which engine it
resolved to. **Sharpen** is identical either way.

The build and its first-start weight downloads are several gigabytes. Image layers and
checksum-verified weights are cached separately, so a rebuild does not empty the named
weights volume. Every extra the image installs is permissively licensed.

### How the GPU is decided

The device reservation is the one part of the deployment that cannot live in
`docker-compose.yml`. Compose has no conditional for a device request, and a static one
fails with `could not select device driver` on a host that cannot satisfy it. So
`scripts/compose.sh` decides, adds `docker-compose.gpu.yml` when a GPU is really there,
and prints which way it went on every `make up`.

It decides by *asking for one* — starting a throwaway container with `--gpus all` — rather
than by reading what Docker reports about itself. Nothing static is trustworthy here: a
host can serve GPUs through CDI or Docker's own device driver while listing no `nvidia`
runtime and no CDI specs at all, and a script that believed those reports would run such a
machine on the CPU without saying why.

Override it when the detection is wrong, or to measure the difference:

```bash
UPSCALER_GPU=1 make up    # require the reservation
UPSCALER_GPU=0 make up    # start without one
```

Confirm the toolkit before the first start on an NVIDIA host. This checks only that a GPU
container runs at all, so it takes the plain tag; the digest the image actually builds on
is pinned once, in the Dockerfile's `cuda-runtime` stage:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.1-runtime-ubuntu24.04 nvidia-smi
```

Useful commands:

```bash
make up         # build and start detached
make logs       # follow logs
make shell      # shell inside the container
make down       # stop and remove; the weights volume survives
```

The image runs as uid/gid `10001:10001` by default; image decoding, inference, and
downloads never run as root. Driving a ComfyUI on the host changes that user to
yours, so that the app can delete its upload from ComfyUI's input directory, and
`/weights` and `/work` are mode `1777` in the image to make both users work.

That mode, rather than chowning the volumes, because chowning them cannot be made
to stick: Docker re-initialises an *empty* named volume from the image every time
a container is created, so `/work` silently reverts to `10001` on the next start
while `/weights`, which has files in it, keeps the change. The sticky bit still
prevents one uid from removing another's files. A volume created by an older
image keeps its old mode, so fix one once with:

```bash
docker compose run --rm --user root --entrypoint sh upscaler \
  -c 'chmod 1777 /weights /work'
```

Note the shell. `--entrypoint chmod` with the arguments after the service name
looks equivalent and is not: Compose drops them, so that form exits zero having
changed nothing.

Confirm which engine was selected:

```bash
curl -s http://127.0.0.1:8000/api/v1/capabilities | python3 -m json.tool
```

## Illustration mode

For anime, line art, and digital illustration, the separate **Illustration** mode uses
Real-ESRGAN's compact x4 anime model instead of the photo model. Its graph stays entirely
in pixel space: model upscaling, one Lanczos resize to the exact target, restoration of
the source alpha plane, and websocket output. It has no prompt, latent diffusion, crop,
pad, or face generator, so it cannot reframe the picture.

It runs on ComfyUI, which is a separate application with its own models and its own queue.
This project does not contain one, so it installs one:

```bash
make setup-comfyui
```

That command adopts a ComfyUI already on the machine — the one it has recorded, then
`~/ComfyUI`, then `~/comfy/ComfyUI` — and clones the pinned one into `~/ComfyUI` only when
there is none. `COMFYUI_DIR=/somewhere/else` picks a different location. Either way it
makes sure the virtualenv can run ComfyUI, installs its requirements and the
checksum-pinned 17 MB illustration model, and records the result in
`.upscaler/comfyui.conf`.

**An adopted installation is never moved to the pinned revision.** The pin in
`models/manifest.json` governs a fresh clone; checking out somebody's existing working
tree, possibly behind the custom nodes they have installed, is not this command's decision.
The record says which of the two happened, so two machines can legitimately differ and you
can see that they do.

After that, the ordinary commands cover everything:

```bash
make up      # starts ComfyUI, then the app connected to it
make down    # stops the app, then ComfyUI
```

Nothing goes on the command line: `make up` reads the recorded installation, starts
ComfyUI, and passes the app the URL, the input directory, and the user to run as.
`UPSCALER_COMFYUI=0 make up` starts the app alone.

ComfyUI is started as a transient `systemd --user` unit where systemd is available and as a
background process with a PID file otherwise, logging to `.upscaler/comfyui.log`. `make
down` stops what it started and nothing else: a ComfyUI that was already running when the
app came up, or one you started yourself, is left alone and says so.

### What make up sets up for you

**ComfyUI must listen where the container can reach it.** Its default binding is
`127.0.0.1`, the host's own loopback, which nothing inside a bridge-networked container can
open — a ComfyUI started the usual way answers you in a browser and refuses the app with
`Connection refused`. `host.docker.internal` resolves to the Docker bridge gateway
(`172.17.0.1` by default), so the service script discovers that address and starts ComfyUI
on both:

```
--listen 127.0.0.1,172.17.0.1
```

This is why it is not a bare `--listen`, which would bind `0.0.0.0` and publish an
unauthenticated ComfyUI to every network the machine is on. If the gateway cannot be
determined the script binds loopback only and says that Illustration will be unreachable,
rather than reaching for the wider binding.

The adapter treats `host.docker.internal` as loopback because that is what it is — the
container's name for its own host — so this needs no off-device opt-in.

**The upload is cleaned up.** ComfyUI's API only accepts a source as a file in its input
directory and offers no way to delete one afterwards, so the app removes it. That needs
write permission on that directory, which means running the container as the host user;
`make up` passes `UPSCALER_UID`/`UPSCALER_GID` for exactly that. Anything left behind by an
earlier run is `make clean-data`'s job.

**A missing or broken ComfyUI never blocks the app.** `make up` checks the recorded
installation first and prints what is missing and how to repair it, then starts the app
regardless. Upscale and Sharpen work; Illustration reports itself unavailable, with a
reason, exactly as it does on a machine that never had ComfyUI.

Setting these by hand is still supported, for a ComfyUI this project did not install:

| Environment variable | Effect |
| --- | --- |
| `UPSCALER_COMFYUI_URL` | Where ComfyUI is listening. Unset means the engine is unavailable. |
| `UPSCALER_COMFYUI_ALLOW_REMOTE` | Permits a host that is genuinely not this machine. Refused without it. |
| `UPSCALER_COMFYUI_INPUT_DIR` | ComfyUI's `input/` directory on the host, bind-mounted so the uploaded source can be deleted after the job. |
| `UPSCALER_UID` / `UPSCALER_GID` | Run the container as this user, so deleting from that directory is permitted. |

The graph is checked in as an API-format template under
[`backend/upscaler/workflows/`](../backend/upscaler/workflows/), declared by
`scripts/build-illustration-workflow.py` and exported by `scripts/comfy-export-workflow.py`.
The builder also emits an ordinary ComfyUI workflow under `workflows/source/`, so the exact
model, resize, alpha path, and websocket output can be opened in ComfyUI, tuned, and
re-exported. Editing the graph in ComfyUI does not change what this application runs until
it is re-exported, which is deliberate: the app runs a version you can read and diff, not
whatever the editor last saved. The catalogue entry records the exact command that produced
it.

The mode remains unavailable with an actionable message until both ComfyUI and the exact
weight are detected. The model is BSD-3-Clause licensed. It is optimized for anime images,
but remains reconstructive super-resolution: tiny lettering and details that are truly gone
can still be inferred incorrectly and should be inspected at 1:1.

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
| `UPSCALER_MAX_JOBS` | `1` | How many jobs infer at once. Raising it multiplies peak memory. |
| `UPSCALER_MAX_QUEUED_JOBS` | `8` | How many jobs may be held at once, running and waiting together. |

`UPSCALER_MAX_QUEUED_JOBS` bounds the queue rather than the inference: an upload is
streamed to disk as it is accepted, so each waiting job already occupies up to
`UPSCALER_MAX_UPLOAD_BYTES` of the work root. A submission past the limit receives HTTP
429 and writes nothing.

These variables are passed through by `docker-compose.yml`. For example, a 128 GiB GB10
whose driver does not classify its shared pool correctly can be started with:

```bash
UPSCALER_RAM_MIB=131072 UPSCALER_VRAM_MIB=131072 \
UPSCALER_GPU_NAME="NVIDIA GB10" UPSCALER_MEMORY_KIND=unified make up
```

Safe mode is a versioned conservative policy, not a guarantee that a third-party driver or
custom workflow cannot run out of memory. Use `UPSCALER_HARDWARE_POLICY=off` only when you
want to manage those risks yourself.

Running the app from a host checkout is a development workflow rather than a deployment;
it is covered in [Development](development.md).

## Running on a remote machine

The service binds to `127.0.0.1` and rejects requests whose `Host` header is not
a loopback name, so the supported way to use it from another computer is an SSH
tunnel — the app stays unexposed and SSH provides transport security:

```bash
# on the server
make up

# on your computer
ssh -N -L 8000:127.0.0.1:8000 you@your-server
```

The compose file publishes the port on the host's loopback only
(`127.0.0.1:8000:8000`). The container binds `0.0.0.0` internally because it must
accept traffic from the Docker bridge; that is not exposure, and the Host check
still rejects anything that does not arrive as a loopback name.

Then open `http://127.0.0.1:8000`. Use `127.0.0.1` rather than `localhost`, which
may resolve to `::1` and miss the forward.
