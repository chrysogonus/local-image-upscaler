# Security policy

## Reporting a vulnerability

Report privately through this repository's **GitHub security advisories**
("Security" → "Report a vulnerability"). Please do not open a public issue for
something exploitable.

I maintain this in my own time, so: expect an acknowledgement within about a
week, and a fix or an explanation of why it is not one before any disclosure.
If you want a coordinated disclosure date, say so and I will work to it. There
is no bounty programme.

Include what you would want to receive — the version or commit, the platform,
which optional engines were installed, and the smallest input or request that
reproduces it. If a malformed image triggers it, attach the file rather than a
photograph you care about; a synthetic reproducer is always preferred, and
please do not send images of real people.

## What this software is

A local-first application. It binds `127.0.0.1`, has no accounts, no telemetry,
no outbound calls during processing, and no server that anyone else can reach.
There is no hosted instance to attack. That shapes what counts as a
vulnerability here.

## In scope

- **Escaping loopback.** Anything that lets a remote or cross-origin page reach
  the API, drive a job, or read a result. The `Host` and `Origin` checks in
  `backend/upscaler/api/middleware.py` are a real boundary; bypasses are the
  most serious class of bug in this project.
- **Malicious images.** Crashes, hangs, unbounded memory, or code execution from
  a crafted or malformed file reaching the decoder, the tiling path, or a model
  adapter. Decompression bombs are handled explicitly and regressions there
  count.
- **Path handling.** Any way an uploaded filename or user-controlled value
  escapes a job workspace, becomes a filesystem path, or reaches a shell.
- **Job isolation and cleanup.** One job reading another's workspace or output,
  or a terminal path (cancel, failure, expiry, shutdown) leaving pixels on disk
  it claimed to erase.
- **Supply chain.** A wrong or unverified checksum in `models/manifest.json`, a
  download path that accepts an unpinned artifact, or anything that would let a
  `make setup-model-*` target install something other than what it names.
- **Leaks in logs.** Image contents or unnecessary identifying paths in log
  output.

## Out of scope

- **Reaching the API from the same machine.** Any local process running as you
  can already reach a loopback service, and read your files directly. That is
  the trust model, not a flaw.
- **Exposing it yourself.** Binding a public interface, publishing the port, or
  setting `UPSCALER_COMFYUI_ALLOW_REMOTE` are documented, deliberate choices.
  Use the SSH tunnel described in [`docs/deployment.md`](docs/deployment.md)
  instead.
- **What the models produce.** Artifacts, and texture a super-resolution model
  inferred rather than recovered, are properties of these models and are
  documented throughout. Misuse concerns belong in
  [`ACCEPTABLE_USE.md`](ACCEPTABLE_USE.md). A resample presented as recovered
  detail, or an engine that invents detail without saying so, is a real bug —
  report it as an issue.
- **Vulnerabilities in ComfyUI, PyTorch, or another upstream project.** Report
  those upstream. If this project's *use* of one is what makes it exploitable,
  that is in scope here.
- **Third-party model weights.** They are downloaded from their publishers and
  are not built or hosted here. Checksum or pinning problems on our side are in
  scope; the contents of the weights are not.
- Missing hardening headers, or a dependency advisory with no exploitable path,
  where you cannot describe the impact.

## Supported versions

This is pre-1.0 and there are no maintained release branches: fixes land on
`main`. Report against the current `main`, and expect the fix there. Python 3.10
through 3.14 are tested in CI.

`uv.lock` and `frontend/pnpm-lock.yaml` are committed and the dependency audits
(`pip-audit`, `pnpm audit`) are gates in `make ci-local`, so a known-vulnerable
dependency should fail CI before it reaches you. Dependabot opens weekly updates
for uv, npm, GitHub Actions, and Docker.

## Hardening already in place

Loopback binding with `Host` and `Origin` checks; independent limits on upload
bytes (`UPSCALER_MAX_UPLOAD_BYTES`) and decoded pixels
(`UPSCALER_MAX_INPUT_PIXELS`); explicit decompression-bomb handling; uploaded
filenames never used as paths or shell input; one job at a time by default;
per-job workspaces cleaned on every terminal path; checksum- and
revision-pinned model downloads; SHA-pinned GitHub Actions with read-only
`GITHUB_TOKEN` permissions and `persist-credentials: false`.
