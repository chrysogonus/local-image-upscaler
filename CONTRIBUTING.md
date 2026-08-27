# Contributing

Thanks for looking. This is a small, opinionated project: a local-first image
upscaler that tries very hard not to lie about what it produced. Contributions
are welcome, and the fastest route to a merged change is understanding what the
project is fussy about before you write it.

Read [`AGENTS.md`](AGENTS.md) first. It is the engineering charter — mission,
product principles, architecture, and image-processing rules — and it is what
review decisions actually get argued from.
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) covers how people are expected to
treat each other while arguing from it.

## Before a large change, open an issue

Small fixes, tests, and documentation can go straight to a pull request. For a
new mode, model adapter, dependency, or anything that changes the pipeline,
open an issue first. The most common reason a large PR is declined here is not
quality — it is scope: something speculative, or a feature that is genuinely
useful but belongs in a different application. A short issue saves you that.

## Setup

```bash
make setup          # Python and frontend dependencies
make setup-model    # the checksum-pinned Real-ESRGAN runtime (optional)
make dev-backend    # then `make dev-frontend` in another terminal
```

The optional engines each have their own target (`make setup-swinir`,
`make setup-cuda`, and so on) and none of them are needed to work on the app,
the API, or the deterministic pipeline. See [`docs/install.md`](docs/install.md)
for what each installs.

## Verifying

Run the narrowest relevant check while iterating:

```bash
uv run pytest backend/tests/test_tiles.py -q     # one backend test file
pnpm --dir frontend test                          # frontend unit tests
```

Then one side at a time — `make lint`, `make test`, `make test-frontend`, or
`make test-e2e`. [`docs/development.md`](docs/development.md#verification) says
what each covers.

Then, before opening a pull request:

```bash
make ci-local
```

This runs every local gate — dependency sync, lockfiles, `pip-audit` and
`pnpm audit`, Ruff, MyPy, ShellCheck, ESLint, Prettier, TypeScript, backend and
frontend coverage floors, manifest and workflow integrity, an isolated wheel
smoke test, the production build, a Chromium end-to-end and accessibility pass,
and validation of every Compose variant. It continues past failures and prints
one verdict, and it reports a gate whose toolchain is missing as BLOCKED rather
than passing it. `GATES=backend` or `GATES=frontend` narrows it while you iterate.

Pull-request CI additionally builds the complete CPU release image and exercises an
upload/job/download lifecycle as its configured non-root user. Scheduled CI builds the
multi-gigabyte CUDA variant. Changes to either path should run the corresponding Docker
build locally when practical and say explicitly when that was not possible.

If a gate cannot run on your machine, say so in the pull request rather than
describing it as passing.

## What review will ask about

Most of this follows from `AGENTS.md`, but these are the recurring ones:

- **This application only reconstructs.** A change that adds a stage which
  synthesises detail the source never contained is out of scope here, and so is
  one that lets a resample read as recovered detail. See
  [`ACCEPTABLE_USE.md`](ACCEPTABLE_USE.md).
- **Nothing leaves the machine.** No telemetry, no outbound calls during
  processing, no third-party services. Loopback binding is not negotiable.
- **New dependencies need an argument.** Licence, hardware support, memory
  behaviour, quality evidence, and maintenance status — popularity is not a
  reason. The same applies to pinning a model.
- **Deterministic logic gets tests.** Dimension maths, tile coordinates,
  blending weights, and option validation should be pure functions with focused
  tests over small generated fixtures. Cover the awkward cases: portrait,
  landscape, odd dimensions, tiny inputs, already-large inputs, grayscale,
  alpha, EXIF-rotated, malformed.
- **Surgical diffs.** Touch what you must, match the surrounding conventions,
  and keep unrelated cleanup out.
- **Docs change in the same commit.** A new or removed `make` target, env var,
  or model requirement updates the README or the relevant page under `docs/`
  (and `AGENTS.md` if it is a rule) in the change that introduces it.

## Things that must never be committed

Model weights, copyrighted photographs, 4K/8K outputs, private images, or local
job data. `.gitignore` covers the usual paths; please check `git status` before
committing rather than relying on it. Test fixtures should be generated in code,
not checked in.

## Workflows

The files in `backend/upscaler/workflows/` are generated, not hand-edited.
Change the builder or the source graph and re-export it; the exact command that
produced each template is recorded in its `generated_by` field, and
`test_repository_integrity.py` enforces that they match.

## Commits and pull requests

Short conventional-commit subjects (`fix:`, `feat:`, `docs:`, `chore:`, `ci:`),
one logical change each. In the pull request, say what changed and why, and
which gates you ran. If your change affects image output, a small before/after
at 1:1 is worth more than a paragraph.

## Licence

Contributions are accepted under the [Apache License 2.0](LICENSE), the same
licence as the project. There is no CLA. If your change vendors third-party
code, add it to [`NOTICE`](NOTICE) with its copyright and licence in the same
commit.
