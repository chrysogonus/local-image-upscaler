# Development

## Running the development servers

Run the backend:

```bash
make dev-backend
```

In another terminal, run the frontend:

```bash
make dev-frontend
```

Open `http://127.0.0.1:5173`. Vite proxies local API requests to the backend on port 8000.

To build the frontend and run the combined local application:

```bash
make run
```

Then open `http://127.0.0.1:8000`.

## Erasing what a session leaves behind

```bash
make clean-data COMFYUI=/path/to/ComfyUI          # report, then ask before deleting
make clean-data-force COMFYUI=/path/to/ComfyUI    # same without the prompt, for scripts
```

`COMFYUI` is how the cleanup finds your ComfyUI. The variables the app reads
(`UPSCALER_COMFYUI_ROOT`, `UPSCALER_COMFYUI_INPUT_DIR`) are set on the command that
launches the app, so a shell running `make` does not normally carry them; without either,
the ComfyUI half is **skipped and says so** rather than reporting a clean it did not do.

Both refuse to run while the app is serving, so nothing is deleted from under a live job,
and `--dry-run` reports without touching anything. What goes:

- every per-job workspace under the work root (`UPSCALER_WORK_ROOT`, otherwise
  `/tmp/local-image-upscaler`) — the retention sweep only knows about jobs the running
  process created, so directories left by an earlier process are otherwise permanent;
- the ComfyUI connector's `.upscaler/comfyui-container-work/` bind-mounted workspaces;
- the `upscaler_work` Docker volume, which `make down` deliberately keeps;
- ComfyUI's `input/`, `output/` and `temp/` directories, its saved workflows under
  `user/default/workflows/`, and its run history and queue.

Model weights are never touched — not `.upscaler/`, not the `upscaler_weights` volume.
Neither are ComfyUI's own shipped placeholders or its settings. A wipe directory that is
itself a symlink is reported and skipped rather than followed, because emptying whatever
it points at is not a decision this command should make for you.

One thing it cannot reach: ComfyUI keeps the live canvas, including prompts typed but never
saved, in your **browser's** local storage. Clear that from the browser itself — "clear site
data", or devtools → Application → Local Storage. The command says so when it finishes.

## Verification

```bash
make ci-local
```

Runs every local quality gate and prints one verdict: dependency sync and lockfile checks;
Python vulnerability auditing across every optional engine plus a frontend audit; Ruff,
MyPy, ShellCheck, ESLint, Prettier, and
TypeScript checks; branch-aware backend and frontend coverage; manifest and generated
workflow integrity tests; an isolated wheel install/import smoke; the production frontend
build; a Chromium end-to-end and accessibility smoke; and Docker Compose configuration
validation. The first run downloads the pinned Playwright browser; later runs reuse its
local cache.

The checked-in coverage floors are 72% for the backend and 85% statements/lines, 70%
branches, and 55% functions for the frontend. They are deliberately below the current
measurements so ordinary refactors have room, but a material regression fails the gate.

It differs from `make check` in three ways that matter when something is wrong:

- It does not stop at the first failure, so one run reports every problem.
- It checks `uv.lock` and `pnpm-lock.yaml`, catching a resolution that only works
  because of what happens to be installed locally.
- A gate whose toolchain is absent is reported **BLOCKED**, never as passing, and
  fails the run. Missing Node must not read as a green frontend.

Exit status is `0` when everything passed, `1` when a gate failed, and `2` when a
gate could not run at all. Narrow it while iterating:

```bash
make ci-local GATES=backend
make ci-local GATES=frontend
```

Narrower still, when you already know which gate you are iterating on. Each is one
component of `make check`, and none of them check lockfiles or audit dependencies:

| Target | What it runs |
| --- | --- |
| `make lint` | Ruff check and format, MyPy over `backend/upscaler`, ShellCheck over `scripts/` |
| `make test` | Backend pytest with branch coverage and a term-missing report |
| `make test-frontend` | Frontend unit tests with coverage |
| `make test-e2e` | The Chromium end-to-end and accessibility pass |

`make check` remains as the fail-fast version for a quick loop. It runs every code,
coverage, browser, build, and wheel gate, but leaves the Docker Compose prerequisite to
the comprehensive local-CI command. The artifact checks can also be run directly:

```bash
make package         # build, inspect, install, and import the wheel in isolation
make compose-config  # resolve and validate the Compose configuration
```

GitHub Actions runs the same local CI contract for every pull request and every
push to `main`. Backend checks run against both the minimum supported Python
(3.10) and Python 3.14; frontend checks use Node.js 22 and pnpm 10.34.5. Every change also
validates the CPU, CUDA, and ComfyUI Compose configurations, builds the complete CPU image,
checks its licences and non-root user, starts it, serves the frontend, and completes a real
upload/job/download lifecycle. A weekly scheduled job (or a manual run with `extended`
selected) additionally builds the multi-gigabyte CUDA image, imports its neural stack,
downloads the checksum-pinned NCNN/Vulkan runtime, and performs real inference through it
using software Vulkan. Private/gated model downloads remain outside unattended CI.
Dependabot proposes weekly Python, frontend, Actions, and digest-pinned base-image updates;
the same gates decide whether those updates are safe to merge.

## Perceptual-quality benchmark

The developer benchmark compares real outputs from the complete production image pipeline,
not mock images or calls to raw model code. Its standard candidate set is the Lanczos
baseline, the pinned SwinIR-L checkpoint, and Real-ESRGAN x4plus. Real-ESRGAN's CUDA and
NCNN implementations count as one model: CUDA is preferred when it fits, with NCNN as its
fallback. The illustration workflow and the finishing sharpen are deliberately excluded,
because they make a different claim about an output than a general photographic upscale
does.

Install the transformer engine and at least one Real-ESRGAN runtime first. SwinIR-L requires
a usable CUDA device; a standard run fails with an actionable preflight error instead of
silently dropping a candidate.

```bash
make setup-swinir
make setup-model-swinir
make setup-model-cuda                     # preferred Real-ESRGAN weights; torch is above
# or: make setup-model                    # NCNN/Vulkan fallback
```

Then prepare and run the benchmark:

```bash
uv run upscaler-benchmark prepare
uv run upscaler-benchmark run
```

`prepare` downloads about 65 MiB of checksum-pinned public-domain originals from NASA's
image library. It validates their signatures and decoded dimensions, crops them, and creates
the fixed inputs under `.upscaler/benchmarks/data/`. Eight cases have high-resolution
references and deterministic clean, blur, noise/JPEG, or combined 4x degradations. Four
cases retain naturally occurring archival, optical, compression, and low-light degradation
and therefore have no reference. Source pages, creators, public-domain status, exact crops,
and SHA-256 digests live in the versioned dataset manifest.

`run` prints the generated run manifest path. It processes all 12 cases sequentially with
one neural pass, production tiling, no TTA, no sharpening, and an exact 4x target. Each of
the 36 outputs records the concrete engine, model/runtime hashes, device, precision, tile,
dimensions, timing, warnings, Git revision, and output hash. Outputs and workspaces remain
local under `.upscaler/benchmarks/`, which is ignored by Git. To resume an interrupted run,
repeat the command with its directory:

```bash
uv run upscaler-benchmark run --run-dir .upscaler/benchmarks/runs/<run-id>
```

Generate a fresh blinded review for each reviewer, then open the printed HTML path in a
browser:

```bash
uv run upscaler-benchmark review .upscaler/benchmarks/runs/<run-id>
```

The page contains no model names or model-named image paths. It randomizes all 36 A/B pairs,
keeps pan and zoom matched, supports keyboard review, persists only in that browser's local
storage, and exports a JSON session after every pair has an A, B, tie, or cannot-judge
decision. Reviewers may optionally mark halos, oversmoothing, invented texture, color shift,
or tile seams on either output. Nothing is submitted over a network.

Merge one or more exported sessions into an unblinded local gallery and machine-readable
summary:

```bash
uv run upscaler-benchmark report .upscaler/benchmarks/runs/<run-id> \
  ~/Downloads/review-<run-id>-*.json
```

The preference score awards 1 point for a win and 0.5 for a tie; cannot-judge pairs are
excluded. Overall, paired-reference, authentic-degradation, and content-tag results are
reported separately. RGB MAE and PSNR appear for paired cases only as fidelity diagnostics,
never as the perceptual ranking. Reports from different run IDs or changed output hashes are
rejected rather than merged, and a small number of local reviewers should not be presented
as statistically significant evidence.
