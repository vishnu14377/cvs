"""Tests for /dev/unqork-mock harness route."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from src.api.app import create_app


@pytest.mark.asyncio
async def test_unqork_mock_returns_html(monkeypatch):
    monkeypatch.setenv("ENABLE_DEV_ROUTES", "true")
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dev/unqork-mock")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_unqork_mock_contains_all_modes(monkeypatch):
    monkeypatch.setenv("ENABLE_DEV_ROUTES", "true")
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dev/unqork-mock")
    body = resp.text
    for mode in ("iframe-overlay", "iframe-replace", "b64-fragment", "raw"):
        assert mode in body


@pytest.mark.asyncio
async def test_unqork_mock_not_registered_without_dev_mode(monkeypatch):
    monkeypatch.delenv("ENABLE_DEV_ROUTES", raising=False)
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dev/unqork-mock")
    assert resp.status_code == 404
