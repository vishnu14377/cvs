"""Tests for /dev/demo-session bootstrap."""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from src.api.app import create_app


@pytest.mark.asyncio
async def test_demo_session_creates_session_in_registry(monkeypatch):
    monkeypatch.setenv("ENABLE_DEV_ROUTES", "true")
    monkeypatch.setenv("API_AUTH_TOKEN", "dev-token-12345")
    app = create_app()

    from src.api.dependencies import get_session_registry
    registry = get_session_registry()
    # Clear any leaked state from other tests
    registry.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/dev/demo-session", json={"session_id": "harness-xyz"})

    assert resp.status_code == 200
    assert resp.json()["session_id"] == "harness-xyz"
    assert "harness-xyz" in registry


@pytest.mark.asyncio
async def test_demo_session_rejects_invalid_id(monkeypatch):
    monkeypatch.setenv("ENABLE_DEV_ROUTES", "true")
    monkeypatch.setenv("API_AUTH_TOKEN", "dev-token-12345")
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/dev/demo-session", json={"session_id": "../evil"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_demo_session_not_registered_without_dev_mode(monkeypatch):
    monkeypatch.delenv("ENABLE_DEV_ROUTES", raising=False)
    monkeypatch.setenv("API_AUTH_TOKEN", "dev-token-12345")
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/dev/demo-session", json={"session_id": "abc"})
    assert resp.status_code == 404
