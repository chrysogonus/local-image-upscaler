# Host installation

## Requirements

Only needed for a host install; the Docker path in [Deployment](deployment.md)
requires none of them.

- Python 3.10 or newer.
- Node.js 22.13 or newer.
- [`uv`](https://docs.astral.sh/uv/) and pnpm 10.34.5. The pnpm release is pinned in
  `frontend/package.json` so lockfile behavior is the same locally, in CI, and in Docker.
- For neural processing: a Vulkan-capable Intel, AMD, or NVIDIA GPU and the optional Real-ESRGAN runtime. A software Vulkan device can work but is much slower.

## Setup

Install the application and development dependencies:

```bash
make setup
```

Install the checksum-pinned official Real-ESRGAN Linux x86-64 release:

```bash
make setup-model
```

### Transformer upscaling (recommended on a CUDA GPU)

Real-ESRGAN's generator is a 2018 convolutional network. Every winning entry in
the NTIRE 2026 ×4 challenge is built on a transformer instead, and **Upscale**
prefers one when it is installed:

```bash
make setup-swinir         # torch and spandrel
make setup-model-swinir   # 142 MB, checksum-pinned
```

The pinned weight is SwinIR-L trained with the BSRGAN degradation pipeline, and
that qualifier is the whole reason for the choice. The headline checkpoints —
HAT-L, DRCT-L, classical SwinIR — are trained on bicubic downsampling only. Fed
a real photograph, whose softness comes from compression and resampling rather
than clean bicubic, they sharpen the artifacts along with the detail and land
*behind* Real-ESRGAN on the images this application actually receives. Picking
an architecture because it tops a benchmark, while ignoring what it was trained
to invert, is the trap `AGENTS.md` warns about.

`spandrel` also recognises HAT, DAT, DRCT, ATD, SPAN, RGT and PLKSR, so any
other checkpoint is a file rather than a code change:

```bash
export UPSCALER_SR_MODEL=/absolute/path/to/4xNomos8kSCHAT-L.pth
```

The model's own architecture and scale are read from the checkpoint and recorded
in the job result. If the engine is absent, Upscale falls back to Real-ESRGAN
and then to the resampler, saying so each time.

The model installer places the runtime and bundled weights under `.upscaler/`, which is ignored by Git. Other platforms can install the official `realesrgan-ncnn-vulkan` executable separately and set:

```bash
export UPSCALER_REALESRGAN_BIN=/absolute/path/to/realesrgan-ncnn-vulkan
```

If the app was already open during installation, return to the browser or use
**Detect again** in the warning. The backend rescans for the runtime without a
restart.

### CUDA GPUs (including ARM64 hosts such as DGX Spark)

The NCNN runtime above is an x86-64 binary and its installer refuses to run
elsewhere. On a CUDA machine use the PyTorch engine instead, which is
architecture independent:

```bash
make setup-cuda          # installs torch from the CUDA 13.0 index
make setup-model-cuda    # downloads the Real-ESRGAN weights
```

Match the index to your driver's CUDA version (`nvidia-smi`, top right). For a
CUDA 12.8 runtime:

```bash
make setup-cuda UPSCALER_TORCH_INDEX=https://download.pytorch.org/whl/cu128
```

The weight file is checksum-pinned in `models/manifest.json` and the install
is discarded on mismatch. Upstream publishes no digest of its own, so it was
computed from the official release download — it protects against corruption
and later tampering, not against a compromised original release. Adding a new
weight entry starts with `"sha256": null`; the installer then refuses it and
prints the digest, which you pin with
`uv run python scripts/install-weights.py --group <group> --pin`.

The engine reports itself unavailable — with a specific reason — if torch is
missing, if no CUDA device initialises, or if the installed wheel lacks kernels
for the GPU. Availability is confirmed by running a real convolution, because on
a recent architecture `torch.cuda.is_available()` can return true while the first
kernel launch fails. The resolved engine and device are shown in the interface, and
`GET /api/v1/capabilities` reports them per mode.

Precision defaults to bfloat16 where supported and float32 otherwise. Override
with `UPSCALER_CUDA_PRECISION=fp32|bf16|fp16`.

### Illustration upscaling through ComfyUI

For anime, line art, and digital illustration, the separate **Illustration** mode uses
Real-ESRGAN's compact x4 anime model instead of the photo model. Its graph stays entirely
in pixel space: model upscaling, one Lanczos resize to the exact target, restoration of the
source alpha plane, and websocket output. It has no prompt, latent diffusion, crop, pad, or
face generator, so it cannot reframe the picture.

ComfyUI is a separate application with its own models and its own queue; this one treats it
as a local worker. Install the websocket client, then the checksum-pinned 17 MB model into
that ComfyUI installation:

```bash
make setup-comfyui
make setup-model-comfyui-illustration \
  UPSCALER_COMFYUI_UPSCALE_MODELS_DIR=/path/to/ComfyUI/models/upscale_models
UPSCALER_COMFYUI_URL=http://127.0.0.1:8188 make dev-backend
```

Setting the URL is the whole opt-in: with it unset the engine does not exist and the mode
reports itself unavailable.

| Environment variable | Effect |
| --- | --- |
| `UPSCALER_COMFYUI_URL` | Where ComfyUI is listening. Unset means the engine is unavailable. |
| `UPSCALER_COMFYUI_ALLOW_REMOTE` | Permits a host that is not this machine. Refused without it. |
| `UPSCALER_COMFYUI_INPUT_DIR` | ComfyUI's `input/` directory, so the uploaded source can be deleted after the job. |

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
