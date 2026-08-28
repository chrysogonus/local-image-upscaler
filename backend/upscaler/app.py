from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from upscaler import __version__
from upscaler.api import router
from upscaler.api.middleware import LocalOnlyMiddleware
from upscaler.config import AppConfig, load_config
from upscaler.jobs import JobManager
from upscaler.models import ModelRegistry


def frontend_directory() -> Path:
    """Locate the built frontend.

    The path relative to this file only holds when running from a source
    checkout; an installed package (as in the container image) sits in
    site-packages instead, so the location is overridable.
    """
    configured = os.getenv("UPSCALER_FRONTEND_DIST")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(config: AppConfig | None = None) -> FastAPI:
    resolved_config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = ModelRegistry(resolved_config)
        manager = JobManager(resolved_config, registry)
        app.state.job_manager = manager
        try:
            yield
        finally:
            await manager.close()

    app = FastAPI(
        title="Local Image Upscaler",
        version=__version__,
        # The schema is served locally; the interactive pages are not served at
        # all. FastAPI's bundled Swagger UI and ReDoc load their JavaScript and
        # CSS from cdn.jsdelivr.net, so opening one would make the user's
        # browser call a third party — the one thing this application promises
        # it does not do — and would render blank on an offline machine. The
        # schema itself is complete, local, and what the page was reading.
        openapi_url="/api/v1/openapi.json",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    # No CORS middleware: nothing is meant to reach this API cross-origin. The
    # frontend calls a relative "/api/v1" path in both deployments — served from
    # this app in a build, proxied by Vite from port 5173 in development — so the
    # browser only ever issues same-origin requests. Granting an origin here
    # would only widen what a page in the user's browser can drive.
    app.add_middleware(LocalOnlyMiddleware)
    app.include_router(router)

    frontend_dist = frontend_directory()
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        def frontend_missing() -> JSONResponse:
            return JSONResponse(
                {
                    "service": "Local Image Upscaler",
                    "status": "backend ready",
                    "frontend": "Run the Vite development server or build frontend/dist.",
                }
            )

    return app


app = create_app()
