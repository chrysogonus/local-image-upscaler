from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class LocalOnlyMiddleware:
    """Reject DNS rebinding hosts and cross-site state-changing requests."""

    # Real loopback names only. This deliberately excludes ASGI test-client
    # conventions such as "testserver": SECURITY.md calls this check a real
    # boundary, and a name that only exists in the test suite still resolves
    # for anyone who can put it in /etc/hosts or answer it from a resolver,
    # which is the DNS-rebinding case this exists to stop. The tests address
    # the app as 127.0.0.1 instead.
    allowed_hosts = {"127.0.0.1", "localhost", "[::1]"}
    # Vite rewrites Host to the proxy target, so a job submitted from the dev
    # server arrives as Host: 127.0.0.1:8000 with Origin: http://127.0.0.1:5173
    # and never matches own_origins below. These two are what keep
    # `make dev-frontend` working; a build is served same-origin and needs none.
    allowed_dev_origins = {"http://127.0.0.1:5173", "http://localhost:5173"}

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        raw_host = headers.get("host", "").lower()
        host = "[::1]" if raw_host.startswith("[::1]") else raw_host.split(":", 1)[0]
        if host not in self.allowed_hosts:
            response = JSONResponse(
                {"detail": "This service accepts local hosts only."}, status_code=421
            )
            await response(scope, receive, send)
            return

        origin = headers.get("origin")
        method = scope.get("method", "GET").upper()
        # A state-changing request with no Origin at all is allowed through, on
        # purpose. Every current browser attaches Origin to a cross-origin POST,
        # form submissions included, so the header's absence means a local
        # non-browser client - curl, a script, the test suite - and SECURITY.md
        # puts same-machine access outside the trust boundary: such a process
        # can already read the user's files directly. Requiring the header would
        # break every local API client without closing a browser-reachable path.
        if origin and method not in {"GET", "HEAD", "OPTIONS"}:
            own_origins = {
                f"http://{headers.get('host')}",
                f"https://{headers.get('host')}",
            }
            if origin not in own_origins and origin not in self.allowed_dev_origins:
                await JSONResponse(
                    {"detail": "Cross-origin state changes are not allowed."}, status_code=403
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)
