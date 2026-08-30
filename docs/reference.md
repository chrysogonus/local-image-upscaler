# Reference

## Modes

A mode is the only decision that changes what the app does. Each one resolves its own
engine, weights, and sampling; there is no engine picker, model list, or quality profile,
because every one of those was a way to make the result worse rather than better.

- **Upscale** *(default)* — enlarges the long edge to 4K or 8K by reconstruction. It uses
  the best engine available — SwinIR-L through spandrel, then Real-ESRGAN on CUDA, then
  NCNN/Vulkan — and falls back to deterministic Lanczos resampling, saying so clearly when
  it does. Finishes with luminance sharpening at the final dimensions.
  A single Real-ESRGAN pass enlarges at most 4×, so a larger factor is covered by chaining
  passes — a 480×270 source bound for 4K needs 8×, which runs as two passes rather than
  leaving half the enlargement to a plain resample. The planned chain is shown before the
  job starts and recorded in the result; **Maximum neural passes** under *Advanced
  processing* bounds it. Each engine is only ever driven at a factor it genuinely
  produces (see [Engine scales](#engine-scales)), and any remainder is absorbed by the
  single exact resize that already ends the pipeline.
- **Illustration** — an explicit faithful path for anime, line art, and digital illustration.
  A local ComfyUI runs the dedicated Real-ESRGAN x4 anime model, then performs one exact
  resize and restores transparency. It does not use diffusion, prompts, face generation,
  padding, or cropping, so the source composition and aspect ratio stay fixed. See
  [Illustration mode](deployment.md#illustration-mode).
- **Sharpen** — keeps the oriented source width and height exactly, bypasses model inference
  and resizing, and improves existing edge contrast on the CPU. It does not claim to add
  detail or resolution.

### Finishing

Sharpening is chosen in the settings palette as **Off**, **Natural**, **Crisp**, or **Strong**,
and every mode opens on Natural. The exact percentage behind those anchors stays on a
slider under *Advanced processing*.

The filter works on luminance alone, so no strength can produce a coloured fringe, and it
works at three scales at once: a fine radius for edge acutance, a middle one for structure,
and a broad one for local contrast. That broad layer is a tonal change rather than only an
edge one, and it is why a sharpened result reads as improved at fit-to-screen and not only
at 1:1.

Two things keep it honest. Its radii are multiples of how far the *last* stage enlarged the
image rather than fixed pixel counts, because that is the width the softness actually has —
a neural engine that lands on the target is therefore treated far more gently than a plain
4× resample at the same setting. An engine that resizes to the target internally, as the
ComfyUI graphs do, reports the size its model stage actually reached, so a stretch that
happened inside it is measured too rather than read as no enlargement at all. And the result is clamped to the local envelope plus a
fraction of local contrast, which bounds halos instead of letting them grow with the
strength. None of this recovers detail the pixels do not already imply.

Output is always PNG. A tool whose purpose is recovering detail should not end by
discarding some to a lossy encoder, and PNG is also the only candidate that carries alpha
without compositing it against an arbitrary matte colour.

Everything else lives under *Advanced processing*, collapsed and rarely needed: the exact
sharpening strength, tile size, test-time augmentation, restore-before-reducing, and the
pass budget. Each of those appears only where the resolved engine can act on it, so the
panel is shorter for a single-pass engine than for a chaining one: augmentation is offered
only by engines whose inference averages the eight orientations, the pass budget only by
one that may run more than once, restore-before-reducing only against a source that already
meets the target, and the tile size only where the app does its own tiling. A control that
cannot change the result is dropped rather than disabled — greying it would imply it works
once something else changes. High sharpening strengths are warned about because they can
still create brittle-looking texture even with halos clamped. Every job snapshot records
`processing_mode` alongside the target, tile settings, finishing strength, and resolved
result engine.

### Engine scales

Each engine declares the enlargement factors it actually produces, and the pass planner
only ever asks for those.

| Engine | Native scales | Chained passes |
| --- | --- | --- |
| SwinIR-L / spandrel | 2×, 3×, 4× | up to 3 |
| Real-ESRGAN (NCNN/Vulkan) | 4× only | up to 3 |
| Real-ESRGAN (CUDA) | 2×, 3×, 4× | up to 3 |
| ComfyUI illustration / Real-ESRGAN anime | x4, then exact target | 1 |

This is a correctness constraint rather than a performance one. The NCNN runtime accepts
`-s 2` and `-s 3` against the 4× `x4plus` weights and returns an image with the right
dimensions but the wrong crop — a silent failure, since nothing about the result looks
broken until it is compared with the source. Those flags are meant for the
`animevideov3` weights, which ship real 2× and 3× variants. The NCNN engine is therefore
driven only at 4× and the exact resize brings the result down, which also fixes 2× jobs
such as 1080p to 4K. The CUDA adapter resamples its own 4× output internally, so smaller
scales are genuine there. The illustration graph reaches the target itself, so it reports
the width its model stage actually produced and the finishing sharpen sizes itself to that
rather than to the file it was handed.

If a source already exceeds the target, neural enlargement is skipped unless **Restore
before reducing** is enabled.

### Why SwinIR-L, and how to use a different checkpoint

Real-ESRGAN's generator is a 2018 convolutional network, and every winning entry in the
NTIRE 2026 ×4 challenge is built on a transformer instead, which is why **Upscale** prefers
one when a CUDA device is available.

The pinned weight is SwinIR-L trained with the **BSRGAN degradation pipeline**, and that
qualifier is the whole reason for the choice. The headline checkpoints — HAT-L, DRCT-L,
classical SwinIR — are trained on bicubic downsampling only. Fed a real photograph, whose
softness comes from compression and resampling rather than clean bicubic, they sharpen the
artifacts along with the detail and land *behind* Real-ESRGAN on the images this
application actually receives. Picking an architecture because it tops a benchmark, while
ignoring what it was trained to invert, is the trap [`AGENTS.md`](../AGENTS.md) warns about.

`spandrel` also recognises HAT, DAT, DRCT, ATD, SPAN, RGT and PLKSR, so any other
checkpoint is a file rather than a code change. Mount it into the container and name it:

```bash
UPSCALER_SR_MODEL=/weights/spandrel-sr/4xNomos8kSCHAT-L.pth
```

The model's own architecture and scale are read from the checkpoint and recorded in the job
result. If the engine is absent, Upscale falls back to Real-ESRGAN and then to the
resampler, saying so each time.

Availability is confirmed by running a real convolution, because on a recent architecture
`torch.cuda.is_available()` can return true while the first kernel launch fails. The engine
reports itself unavailable — with a specific reason — if torch is missing, if no CUDA device
initialises, or if the installed wheel lacks kernels for the GPU. Precision defaults to
bfloat16 where supported and float32 otherwise; override with
`UPSCALER_CUDA_PRECISION=fp32|bf16|fp16`.

The published image pairs the CUDA 13.0 base with the CUDA 13 package set pinned in
`uv.lock`, and is intentionally not retargetable by an environment variable. A host whose
driver needs a different CUDA runtime has to change that pinned pair in the Dockerfile and
lockfile together.

## Local API

The versioned API provides:

- `GET /api/v1/health`
- `GET /api/v1/capabilities`
- `POST /api/v1/jobs`
- `GET /api/v1/jobs/{id}`
- `GET /api/v1/jobs/{id}/events`
- `GET /api/v1/jobs/{id}/result`
- `DELETE /api/v1/jobs/{id}`

The complete OpenAPI schema is served locally at `GET /api/v1/openapi.json`. There is no
bundled Swagger UI or ReDoc page: both load their assets from a third-party CDN, which
would make the browser call out to one and would leave the page blank on an offline
machine. Point any local schema viewer at that URL instead.

`GET /api/v1/capabilities` includes backend and ComfyUI hardware reports, per-choice resource
requirements, safe target/tile choices, the full diagnostic mode list, actionable workflows,
and exclusion reasons. Completed results include the concrete `resolved_tile_size`.

## Current limitations

- Animated formats process the first frame only.
- The working neural path converts high-bit-depth input to 8-bit sRGB and reports that loss.
- The neural engines are adversarially trained, so Real-ESRGAN and SwinIR can emit plausible
  texture that was not present in the source. No mode is built to invent detail, but the prior
  can, and [`README.md`](../README.md#reconstruction-only) draws that line explicitly.
- The NCNN model installer targets Linux x86-64; other platforms need the CUDA engine or a manually installed compatible executable.
- The CUDA engine restores color channels only. Transparency is resampled with Lanczos and the result says so.
- Resource profiles are conservative estimates; custom ComfyUI graphs or driver behavior can still exceed them.
- Chained passes compound the previous pass's artifacts; inspect a heavily chained result at 1:1.
- Output is PNG only, so an 8K result is a large file.
- The ComfyUI engine runs the graphs checked into this repository, not whatever the ComfyUI
  editor last saved; changing a workflow means re-exporting it.
- ComfyUI has no API for deleting an uploaded input, so without `UPSCALER_COMFYUI_INPUT_DIR`
  the sources accumulate in its `input/upscaler/` directory.
- Job directories are swept only while the process that created them is alive; one killed
  mid-run leaves its workspace behind until `make clean-data`.
- The illustration model runs through a separately installed local ComfyUI; it is not a
  fallback for the default photo mode and has no non-ComfyUI adapter yet.
- HAT remains future model-selection work; the developer benchmark currently fixes its
  transformer candidate to the pinned SwinIR-L checkpoint so results stay comparable.
