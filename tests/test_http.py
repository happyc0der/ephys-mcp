import json
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from ephys_mcp import cli
from ephys_mcp.server import mcp

TOKEN = "test-token-0123456789abcdef"
BASE = "http://127.0.0.1:8000"  # the SDK allows only loopback hosts with an explicit port


@asynccontextmanager
async def running_client(headers: dict):
    """An httpx client talking in-process to the HTTP app, with its session manager started
    (uvicorn does that through the lifespan hook; ASGITransport does not)."""
    app = cli.build_http_app("127.0.0.1", TOKEN)
    async with (
        mcp.session_manager.run(),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE, headers=headers) as c,
    ):
        yield c


async def test_rejects_missing_or_wrong_token():
    async with running_client({}) as c:
        for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic " + TOKEN}):
            r = await c.post("/mcp", json={}, headers=headers)
            assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


async def test_full_session_over_http():
    async with (
        running_client({"Authorization": f"Bearer {TOKEN}"}) as c,
        streamable_http_client(f"{BASE}/mcp", http_client=c) as (r, w),
        ClientSession(r, w) as s,
    ):
        await s.initialize()
        res = await s.call_tool("open_session", {"source": "synthetic", "params": {"duration_s": 5}})
        assert not res.is_error and "session_id" in json.loads(res.content[0].text)


def test_cli_refuses_public_bind_without_token(monkeypatch):
    monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit, match="refusing to serve"):
        cli.main(["--http", "--host", "0.0.0.0"])
    monkeypatch.setenv(cli.TOKEN_ENV, "short")
    with pytest.raises(SystemExit, match="at least"):
        cli.main(["--http"])
