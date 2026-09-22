"""Command line entry point: stdio by default, streamable HTTP on request.

HTTP mode requires a bearer token unless bound to loopback, because neural
data is sensitive and this server has no user accounts of its own.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import logging
import os
import sys

TOKEN_ENV = "EPHYS_MCP_TOKEN"
MIN_TOKEN_LEN = 16


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class BearerAuth:
    """ASGI middleware: every request must carry `Authorization: Bearer <token>`."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            header = dict(scope["headers"]).get(b"authorization", b"")
            scheme, _, presented = header.partition(b" ")
            if scheme.lower() != b"bearer" or not hmac.compare_digest(presented.strip(), self.token):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"text/plain"), (b"www-authenticate", b"Bearer")],
                    }
                )
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.app(scope, receive, send)


def build_http_app(host: str, token: str | None):
    """The ASGI app for HTTP mode, wrapped in bearer auth when a token is set."""
    from .server import mcp

    app = mcp.streamable_http_app(host=host)
    return BearerAuth(app, token) if token else app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ephys-mcp", description="MCP server for intracortical BCI recordings.")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    from .server import mcp

    if not args.http:
        mcp.run()
        return

    token = os.environ.get(TOKEN_ENV) or None
    if token and len(token) < MIN_TOKEN_LEN:
        sys.exit(f"{TOKEN_ENV} must be at least {MIN_TOKEN_LEN} characters")
    if not token and not _is_loopback(args.host):
        sys.exit(
            f"refusing to serve on {args.host} without {TOKEN_ENV} set: neural data must not be exposed unauthenticated"
        )
    if not token:
        logging.getLogger("ephys_mcp").warning("no %s set; HTTP is unauthenticated (loopback only)", TOKEN_ENV)

    import uvicorn

    uvicorn.run(build_http_app(args.host, token), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
