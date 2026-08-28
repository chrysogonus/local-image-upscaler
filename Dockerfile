# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Release images are digest-pinned so the same commit cannot silently resolve a
# different operating system or build tool. Dependabot proposes digest updates.
ARG NODE_IMAGE=node:22-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1
ARG CERTIFICATES_IMAGE=python:3.12-slim@sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

FROM ${UV_IMAGE} AS uv
FROM ${CERTIFICATES_IMAGE} AS certificates

# The CUDA base, declared here as a real stage rather than restated in the
# Compose overlay and in CI. Dependabot's Docker updater maintains digests it
# finds on a FROM line, so this is the only place the CUDA pin can live and
# still be kept current; the overlay selects it with RUNTIME_IMAGE=cuda-runtime,
# which a FROM resolves as a stage name. BuildKit skips a stage nothing depends
# on, so the default CPU image neither pulls nor resolves this.
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:c3fde347d52d578c84fd644bc177bc7ec333feaf11550d990da4084d7612e4c7 AS cuda-runtime

# ---------------------------------------------------------------- frontend --
FROM ${NODE_IMAGE} AS frontend

WORKDIR /build

# Keep the container on the same exact package-manager release as package.json
# and CI so lockfile behavior cannot vary by environment.
ARG PNPM_VERSION=10.34.5
RUN npm install -g pnpm@${PNPM_VERSION}

# Dependencies are copied first so an edit to application source does not
# invalidate the install layer.
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./

# pnpm's default 16-way parallel fetch times out against the npm registry on
# constrained links. Fewer connections and longer patience cost a little on a
# fast network and are the difference between finishing and failing on a slow
# one. The store is cached like the pip layer above, so a retry resumes rather
# than re-downloading everything.
ARG PNPM_NETWORK_CONCURRENCY=4
RUN --mount=type=cache,target=/pnpm-store \
    pnpm config set store-dir /pnpm-store \
    && pnpm config set network-concurrency ${PNPM_NETWORK_CONCURRENCY} \
    && pnpm config set fetch-retries 5 \
    && pnpm config set fetch-retry-maxtimeout 120000 \
    && pnpm config set fetch-timeout 300000 \
    && pnpm install --frozen-lockfile

COPY frontend/ ./
RUN pnpm build

# ----------------------------------------------------------------- runtime --
# CPU-safe by default. docker-compose.cuda.yml replaces RUNTIME_IMAGE with the
# digest-pinned CUDA base and selects the CUDA extras.
FROM ${RUNTIME_IMAGE} AS runtime

# Optional feature sets. The CPU image selects none; the CUDA Compose overlay
# selects "swinir". Every extra here is permissively licensed.
ARG UPSCALER_EXTRAS=

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=certificates /etc/ssl/certs/ /etc/ssl/certs/
ARG PYTHON_VERSION=3.12.14
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install --no-bin "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

# The project metadata selects the locked dependency graph. The image does not
# build the project wheel (and therefore does not resolve an isolated, unlocked
# build backend); CI builds and inspects that artifact separately. LICENSE and
# NOTICE remain discoverable both here and under /usr/share/licenses below.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY backend/ ./backend/
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    extra_args=""; \
    old_ifs="${IFS}"; \
    IFS=','; \
    for extra in ${UPSCALER_EXTRAS}; do \
        if [ -n "${extra}" ]; then extra_args="${extra_args} --extra ${extra}"; fi; \
    done; \
    IFS="${old_ifs}"; \
    uv sync --locked --no-dev --no-install-project ${extra_args}

COPY --from=frontend /build/dist ./frontend/dist
COPY models/manifest.json ./models/
COPY scripts/install-weights.py ./scripts/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && install -d /usr/share/licenses/local-image-upscaler \
    && install -m 0644 LICENSE NOTICE /usr/share/licenses/local-image-upscaler/ \
    && groupadd --gid 10001 upscaler \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent upscaler \
    && install -d -o upscaler -g upscaler /weights /work

# Runtime dependencies come from the lock; the application itself runs from the
# copied source tree so no build-only dependency participates in the image.
ENV PYTHONPATH=/app/backend \
    UPSCALER_FRONTEND_DIST=/app/frontend/dist \
    UPSCALER_REALESRGAN_WEIGHTS_DIR=/weights \
    UPSCALER_SR_WEIGHTS_DIR=/weights/spandrel-sr \
    UPSCALER_WORK_ROOT=/work \
    UPSCALER_HOST=0.0.0.0 \
    UPSCALER_PORT=8000

# Weights are not baked in: they are large, separately licensed, and belong in
# a volume so a rebuild does not re-download them. New named volumes inherit the
# ownership above; both the web service and first-start download run unprivileged.
USER 10001:10001

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "-m", "upscaler"]
