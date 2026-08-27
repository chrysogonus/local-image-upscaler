from __future__ import annotations

import logging
import os

import uvicorn

LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}

logger = logging.getLogger("upscaler")


def server_options() -> tuple[str, int]:
    """Resolve the bind address, defaulting to loopback.

    Containers must bind 0.0.0.0 to receive traffic from the bridge network.
    That is not the same as exposing the service: the supplied compose file
    publishes the port on the host's loopback only, and LocalOnlyMiddleware
    still rejects any request that does not arrive with a loopback Host header.
    """
    host = os.getenv("UPSCALER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    raw_port = os.getenv("UPSCALER_PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"UPSCALER_PORT must be an integer, got {raw_port!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"UPSCALER_PORT must be between 1 and 65535, got {port}")
    return host, port


def main() -> None:
    host, port = server_options()
    if host not in LOOPBACK:
        logger.warning(
            "Binding %s rather than loopback. Ensure the port is published only where you "
            "intend it to be reachable.",
            host,
        )
    uvicorn.run(
        "upscaler.app:app",
        host=host,
        port=port,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
