# AGENTS.md

## Mission

A polished, local-first web app that enlarges images to 4K or 8K while preserving natural texture, clean edges, color, and transparency. All processing happens on the user's machine; images are never uploaded to a third party unless a future, explicitly opt-in feature says otherwise. Every mode here reconstructs: recovering detail the pixels still imply. Synthesising detail that was never in the source is out of scope for this repository.

## Working rules

1. **Think before coding.** State assumptions, surface tradeoffs, push back when warranted.
2. **Simplicity first.** The minimum code that solves the problem. Nothing speculative.
3. **Surgical changes.** Touch only what you must. Match existing conventions. Clean up only your own mess.
4. **Goal-driven execution.** Define success criteria, then loop until verified.

Plus, for this repo: inspect the current diff first and preserve unrelated user changes; run the narrowest relevant check while iterating and `make check` before handing off; if tooling is missing, say so instead of claiming a pass.

## Product principles

1. **Quality before apparent sharpness.** Detail recovery with controlled artifacts beats halos, oversharpening, and invented texture.
2. **Reconstruction and generation are different claims.** Recovering detail the pixels still imply is not the same operation as synthesising detail that was never there, and only the first belongs here. No engine or mode in this repository may claim `generative`; the test suite asserts it. The label itself stays in the adapter protocol, the capability report, and the UI so that a stage which ever did invent detail would be announced rather than indistinguishable. An enlargement is still an inference and never evidence about a person, and the product must say so. See `ACCEPTABLE_USE.md`.
3. **Local and private by default.** Loopback binding, no telemetry, sources and results stay on-device.
4. **Make tradeoffs visible.** Show model, scale, estimated memory, progress, and exact output dimensions; allow 1:1 before/after comparison.
5. **Work on ordinary hardware.** Tiled inference, overlap blending, cancellation, CPU fallback. Detect accelerators rather than assuming CUDA.
6. **Never silently damage an image.** Preserve aspect ratio and alpha, respect orientation and color profiles where practical, warn before lossy conversion or metadata loss.

## Architecture

Browser UI for selection, settings, comparison, and progress; a local Python service for heavyweight inference. Keep UI, job orchestration, image pipeline, model adapters, and device detection behind small, separate interfaces.

- **Frontend:** TypeScript/React, accessible components and styles.
- **Local API:** FastAPI on `127.0.0.1` — health, capabilities, job, progress, cancellation, download.
- **Inference:** pluggable model adapters. No model is universally best; evaluate general, restoration, and photo-real profiles independently.
- **Pipeline:** decode/normalize → optional denoise/deblock → tiled inference → seam-safe overlap blend → exact target resize → restrained finishing → color/alpha-aware encode.
- **Storage:** per-job temporary workspace, cleaned on download, cancellation, failure, or expiry. Never use an uploaded filename as a filesystem path.

Model weights, large outputs, and local job data stay out of Git. Downloads need checksums, a clear cache location, license info, and actionable offline/error states.

## Image processing

- 4K/8K are configurable long edges (initially 3840 and 7680) with preserved aspect ratio; report exact output dimensions before starting.
- Don't enlarge a source already past the target unless the user explicitly picks restoration-only.
- Tile with even spacing and raised-cosine blending, ramped over each tile's real overlap; keep tests that make seams and border errors detectable.
- Estimate memory from decoded dimensions, tile size, model scale, precision, and intermediate buffers — never from compressed file size.
- Apply EXIF orientation before inference. Keep ICC profiles and alpha where the decoder/model/encoder path allows, and surface unavoidable loss.
- For RGB-only models, composite transparency carefully and restore a correctly scaled alpha plane without bright or dark fringes.
- Chain model-native passes when the factor exceeds one pass, then one high-quality resize to the exact target. Avoid repeated arbitrary resampling.
- Sharpening is user-controlled and labelled: it can make an inferred texture read as a recovered one.
- Record enough to reproduce a result: model and weight version, target, tile/overlap, device, precision, preprocessing, finishing, encoder settings.

## Security and reliability

- Loopback only unless the user knowingly enables network access.
- Validate file signatures and decoded formats; enforce configurable pixel/dimension/job limits; handle decompression bombs and malformed images safely.
- Never pass filenames or user-controlled values to a shell.
- Cancellation is cooperative; release GPU/CPU memory and temp files on every terminal path.
- Limit concurrent jobs by available memory — queue rather than crash.
- Logs stay useful but free of image contents and unnecessarily identifying paths.

## UX and accessibility

- Keyboard navigation, visible focus, semantic controls, live status announcements, reduced-motion support.
- Distinct phases (loading model, analyzing, enhancing tiles, finishing, encoding). Never fabricate progress the backend cannot measure.
- Before/after pan and zoom at matched coordinates plus a split view; previews may be downsampled but must be distinguishable from full-resolution output.
- Explain model/profile differences in plain language, with a conservative default.
- Prevent duplicate submissions; make errors recoverable without reselecting the source when it is safe to retain it.

## Testing and dependencies

- Focused tests with small generated fixtures. Prefer deterministic pure functions for dimension math, tile coordinates, blending weights, and option validation.
- Cover portrait, landscape, odd dimensions, tiny inputs, already-large inputs, grayscale, alpha, EXIF-rotated, and malformed files.
- Never commit copyrighted photos, model weights, 4K/8K outputs, or private images.
- Don't add a dependency or pin a model because it is popular. Document license, hardware support, memory behavior, quality evidence, and maintenance status.

## Repository and commands

| Path | What it is |
| --- | --- |
| `frontend/` | React + TypeScript + Vite production UI. Keep its restrained visual direction. |
| `backend/upscaler/` | Loopback FastAPI service, jobs, imaging pipeline, device detection, model adapters. |
| `backend/upscaler/workflows/` | The API-format ComfyUI graph the Illustration mode runs, plus the catalogue describing it. Generated by `scripts/comfy-export-workflow.py` from the builder under `scripts/build-illustration-workflow.py`; never hand-edited. `source/` holds the openable ComfyUI workflow. |
| `models/manifest.json` | Metadata and checksums for optional runtimes and weights (binaries ignored). |
| `Dockerfile`, `docker-compose*.yml`, `docker/` | Digest-pinned CPU image plus explicit CUDA/ComfyUI overlays; weights in a named volume, port on host loopback only. |
| `docs/` | The long-form manual: deployment, host installation, engines, reference, development. `README.md` is the landing page and links here rather than absorbing it. |

- `make setup` — Python and frontend dependencies.
- `make setup-model` — checksum-pinned Linux Real-ESRGAN runtime under `.upscaler/`.
- `make setup-cuda` / `make setup-model-cuda` — optional torch extra and CUDA-engine weights.
- `make setup-swinir` / `make setup-model-swinir` — transformer Upscale engine (spandrel) and its pinned weights.
- `make setup-comfyui` — websocket client for driving the Illustration graph on a local ComfyUI. No weights: that installation owns them.
- `make setup-model-comfyui-illustration UPSCALER_COMFYUI_UPSCALE_MODELS_DIR=/path/to/ComfyUI/models/upscale_models` — checksum-pinned illustration model for the ComfyUI-backed Illustration mode.
- `make dev-backend` / `make dev-frontend` — development servers.
- `make up` / `make up-cuda`, then `make down`, `make logs`, `make shell` — CPU-safe or explicit NVIDIA containerised app.
- `make check` — backend lint/tests, frontend type/tests, production build. Fail-fast.
- `make ci-local` — every gate plus both lockfiles, continuing past failures; reports a gate whose toolchain is missing as BLOCKED rather than passed. `GATES=backend|frontend` narrows it.
- `make run` — build and serve the combined app on `127.0.0.1:8000`.
- `make clean-data COMFYUI=<path>` / `make clean-data-force` — erase every job workspace (host and Docker
  volume) plus ComfyUI's input/output/temp, saved workflows, history and queue. Reports and
  confirms first; refuses while the app is serving; never touches model weights. `COMFYUI`
  locates the install because a make shell does not inherit the app's variables; without it
  the ComfyUI half is skipped loudly. Browser local storage is out of reach and it says so.

Keep docs and example commands in sync with the code, in the same change that adds or removes a command.

## Definition of done

User-visible behavior works end to end, failure and cancellation paths are handled, privacy and resource implications are accounted for, relevant tests pass, and docs reflect any new setup or model requirements. Visual-only mock behavior must be labelled as such and must never masquerade as completed neural upscaling.

---

The four working rules are the widely circulated "Karpathy rules" — derived from Andrej Karpathy's public observations about LLM coding agents and packaged by Forrest Chang; Karpathy did not author or endorse the file itself.
