"""Tests for CORS middleware (dev-gated)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from src.api.app import create_app


@pytest.mark.asyncio
async def test_cors_header_present_when_dev_mode(monkeypatch):
    monkeypatch.setenv("ENABLE_DEV_ROUTES", "true")
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.options(
            "/widget/v1/chat/ui",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8080"


@pytest.mark.asyncio
async def test_cors_header_absent_when_dev_mode_off(monkeypatch):
    monkeypatch.delenv("ENABLE_DEV_ROUTES", raising=False)
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.options(
            "/widget/v1/chat/ui",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.headers.get("access-control-allow-origin") != "http://localhost:8080"
