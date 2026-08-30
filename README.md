# Local 4K / 8K Image Upscaler

A local-first web application for high-quality image enlargement and refinement. It combines an accessible browser interface with an on-device processing service, exact 4K/8K output, safe job handling, color-aware decoding, and optional neural restoration through Real-ESRGAN.

Images are sent only to a service bound to `127.0.0.1`; no cloud API or telemetry is used.

## What it produces

![Illustration mode at 1:1 pixels. The left half is the 4K result: individually resolved fur strokes across the creature's ruff and crisp filigree on the stone arch behind it. The right half is the source at the same zoom, covering more of the scene at visibly softer detail.](docs/images/illustration-creature-comparison.webp)

![Illustration mode at 1:1 pixels. The left half is the 4K result: separated leaves, resolved spray at the lip of the falls, and defined carving on the arches above them. The right half is the source at the same zoom, wider and softer.](docs/images/illustration-waterfall-comparison.webp)

Both strips show **Illustration** mode at 1:1 output pixels — left the 4K result, right the
source. The result covers less of the scene at the same zoom because it has twice the
pixels across. They are WebP-compressed for the web; the application's own comparison view
is where sharpness is worth judging. The default **Upscale** mode targets photographs
instead, and [Modes](#modes) sets out what each one claims.

Both sources are the maintainer's own; [`docs/images/`](docs/images/README.md) records their
provenance and what each panel is.

## Reconstruction only

Every mode here **reconstructs**: it recovers detail the pixels still imply. No stage in
this application is built to invent detail. There is no diffusion restoration, no face
prior, and no prompt — not disabled, not behind a flag, simply not part of the project.
The test suite asserts it, and the interface reports a `generative` flag per mode that is
always false.

That flag is a claim about what a stage is *for*, not a promise about every pixel it
emits. The neural engines are adversarially trained, and an adversarial model asked for a
plausible edge will sometimes draw texture the source does not contain — fur, fabric
weave, film grain. That is a property of the prior rather than a stage doing its job, it
is recorded under [current limitations](docs/reference.md#current-limitations), and it is
why the line is drawn where it is: a model that invents a detail while trying to recover
one is a different thing from a mode whose purpose is to make detail up, and only the
first is in scope here.

That boundary is a scope decision rather than a limitation to work around. Reconstruction
is the weaker claim about a result and the one that stays honest: from a face forty pixels
wide there is nothing left to reconstruct, and this application says so instead of
inventing one.

It is still not evidence. A super-resolution model infers what a soft edge most likely
was, and on a small subject that inference is a statistical guess that happens to look
photographic. See [Modes](docs/reference.md#modes) for what each mode claims, and
[`ACCEPTABLE_USE.md`](ACCEPTABLE_USE.md) for what follows from that.

## Quick start

Running the app needs only Docker, and there is one command:

```bash
make up
```

Then open `http://127.0.0.1:8000`.

One image serves every host. Where the NVIDIA Container Toolkit is installed it reserves
the GPU and **Upscale** runs on SwinIR-L; where it is not, the same image starts without a
device, and Upscale falls back to the deterministic resampler and says so. **Sharpen** is
identical either way, and **Illustration** needs a ComfyUI you run yourself.

**Illustration** mode runs on ComfyUI, which is a separate application. One command
installs or adopts one and wires it in; after that it starts and stops with the app:

```bash
make setup-comfyui
make up      # starts ComfyUI, then the app connected to it
make down    # stops both
```

`make setup-comfyui` installs onto the host rather than into the container, so it needs
[uv](https://docs.astral.sh/uv/) alongside Docker; `make clean-data` below is the same. The
one-line install is in [Deployment](docs/deployment.md#illustration-mode), and the commands
say so themselves if it is missing.

Reaching the app over SSH, what that setup command actually does, and sizing the app to
your hardware are covered in [Deployment](docs/deployment.md).

## What is implemented

- React and TypeScript interface.
- Original/result/split comparison with matched coordinates and a scrollable 1:1 output-pixel view.
- Three modes — Upscale, Illustration, Sharpen — each of which resolves its own engine,
  weights, and settings, so there is nothing to configure to get the intended result.
- FastAPI service with bounded uploads, immutable job settings, one-at-a-time processing, SSE progress, cancellation, expiry cleanup, and safe temporary paths.
- Exact 3840/7680-pixel long-edge sizing with one final Lanczos resample.
- Chained neural passes so the model, not Lanczos, performs the whole enlargement even
  when the factor exceeds a single pass's 4× ceiling, with each engine driven only at a
  scale it genuinely produces.
- EXIF orientation, ICC-to-sRGB normalization, and lossless alpha-preserving PNG output.
- Multi-scale luminance-only finishing, sized to the enlargement and clamped against halos.
- Always-available deterministic resampler plus Real-ESRGAN adapters: PyTorch/CUDA (any
  CUDA GPU, including ARM64 hosts such as DGX Spark) and NCNN/Vulkan for a separately
  provided binary.
- An optional ComfyUI engine that runs the checked-in illustration graph on a ComfyUI you
  run yourself, over its HTTP API, with measured progress, cancellation, and no writes to
  ComfyUI's output directory.
- Per-mode capability reporting; a mode that cannot run says why instead of quietly
  producing something weaker.
- Geometry, tiling, image pipeline, API lifecycle, frontend DOM/accessibility, and browser
  end-to-end tests.

## Modes

A mode is the only decision that changes what the app does. Each resolves its own engine,
weights, and sampling; there is no engine picker, model list, or quality profile.

| Mode | What it does | Claim |
| --- | --- | --- |
| **Upscale** *(default)* | Enlarges the long edge to 4K or 8K with the best installed engine — SwinIR-L, then Real-ESRGAN, then Lanczos — chaining passes when the factor exceeds one. | Reconstruction |
| **Illustration** | Real-ESRGAN's x4 anime model through a local ComfyUI, then one exact resize. No prompt, diffusion, or cropping. | Reconstruction |
| **Sharpen** | Keeps the source dimensions exactly and improves existing edge contrast on the CPU. | Neither |

[Reference](docs/reference.md) covers each mode in full, along with finishing, the
enlargement factors every engine actually produces, and the local API.

## Documentation

| Page | What is in it |
| --- | --- |
| [Deployment](docs/deployment.md) | The one image and how the GPU is decided, Illustration through your own ComfyUI, the hardware-aware policy and its memory floors, and reaching the app over SSH. |
| [Reference](docs/reference.md) | Modes, finishing, engine scales and why SwinIR-L, the local API, and current limitations. |
| [Development](docs/development.md) | Dev servers and the host toolchain, erasing what a session leaves on disk, the verification gates, and the perceptual-quality benchmark. |

## Privacy and safety

- The application binds to loopback only.
- Host and origin checks reduce DNS-rebinding and cross-site request risks.
- Filenames never become storage paths or shell input.
- Upload bytes and decoded pixels have independent limits.
- Only one inference job runs by default, preventing accidental memory overcommit.
- Cancellation and shutdown clean job directories and stop model processes.
- No image content, paths, or telemetry are sent off-device. The optional ComfyUI engine
  posts the source to a ComfyUI you configure, and refuses any host that is not this machine
  unless `UPSCALER_COMFYUI_ALLOW_REMOTE` is set.
- `make clean-data` erases every picture a session left on disk, in this application and in
  the ComfyUI it drove. See [Erasing what a session leaves
  behind](docs/development.md#erasing-what-a-session-leaves-behind).
- Every mode reconstructs rather than generates, but an enlargement is still an inference
  and never an identification of a person.
  [`ACCEPTABLE_USE.md`](ACCEPTABLE_USE.md) states what this software must not be used for.

## Licensing

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
Third-party source vendored into the tree is listed in [`NOTICE`](NOTICE) — currently
the Real-ESRGAN RRDBNet generator in
[`backend/upscaler/models/rrdbnet.py`](backend/upscaler/models/rrdbnet.py), which is
BSD-3-Clause.

The wheel embeds both `LICENSE` and `NOTICE`, and the container exposes the same files at
`/usr/share/licenses/local-image-upscaler/`. The CUDA base image and installed Python and
operating-system packages retain their own upstream terms; the project licence does not
replace them.

**No model weights are distributed here.** Every checkpoint is downloaded from its
original publisher against a pinned checksum and stays under its own licence.
[`models/manifest.json`](models/manifest.json) is the authoritative record: it carries the
publisher, homepage, pinned URL or revision, SHA-256 checksum, and licence for each one.

| Component | Installed by | Licence | Commercial use |
| --- | --- | --- | --- |
| RealESRGAN_x4plus | The container, on first start | BSD-3-Clause | Yes |
| SwinIR-L real-world x4 GAN | The container, on first start | Apache-2.0 | Yes |
| RealESRGAN x4plus anime 6B | `make setup-comfyui`, into that ComfyUI | BSD-3-Clause | Yes |
| ComfyUI itself | `make setup-comfyui`, cloned onto your machine | GPL-3.0-only | Yes |
| Real-ESRGAN NCNN/Vulkan runtime | `scripts/install-realesrgan-linux.sh`; not in the image | MIT | Yes |

Every weight this project installs is permissively licensed and usable commercially. That
is deliberate: a model whose licence forbids commercial use would have to be labelled
wherever its output appeared, and none is needed for reconstruction.

ComfyUI is the one entry that is copyleft rather than permissive. That changes nothing for
this repository: `make setup-comfyui` clones it onto your machine, where its terms cover
that checkout, and nothing from it is distributed here or linked into this code.

The ComfyUI engine runs its graph against a model installed in *your* ComfyUI. The pinned
illustration weight above is the only one it needs; anything else in that installation is
outside this repository's control and carries whatever licence you accepted for it.

## Contributing and reporting

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — setup, how to verify a change with `make ci-local`,
  and what review asks about.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — the standard for issues and review, and
  how to report a concern privately.
- [`AGENTS.md`](AGENTS.md) — the engineering and image-quality charter these follow from.
- [`ACCEPTABLE_USE.md`](ACCEPTABLE_USE.md) — what this software is not for, and why
  generation is out of scope.
- [`SECURITY.md`](SECURITY.md) — how to report a vulnerability privately, and what is in
  scope for a local-first application.
