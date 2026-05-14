"""Tests for /dev/legacy-proxy route."""

from __future__ import annotations

import pytest
import respx
from httpx import ASGITransport, AsyncClient
from src.api.app import create_app


@pytest.fixture
def dev_app(monkeypatch):
    monkeypatch.setenv("ENABLE_DEV_ROUTES", "true")
    return create_app()


@pytest.mark.asyncio
async def test_legacy_proxy_forwards_html(dev_app):
    transport = ASGITransport(app=dev_app)
    async with (
        respx.mock(base_url="http://localhost:8080") as mock,
        AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        mock.post("/memberADR/renderHtml").respond(
            200, json={"renderedHtml": "<html><body>ok</body></html>"}
        )
        resp = await client.get(
            "/dev/legacy-proxy?url=http%3A%2F%2Flocalhost%3A8080%2FmemberADR%2FrenderHtml"
        )
    assert resp.status_code == 200
    assert "ok" in resp.text


@pytest.mark.asyncio
async def test_legacy_proxy_rejects_non_localhost(dev_app):
    transport = ASGITransport(app=dev_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dev/legacy-proxy?url=http%3A%2F%2Fevil.example.com%2F")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_legacy_proxy_not_registered_without_dev_mode(monkeypatch):
    monkeypatch.delenv("ENABLE_DEV_ROUTES", raising=False)
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dev/legacy-proxy?url=http%3A%2F%2Flocalhost%3A8080%2Ffoo")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_legacy_proxy_allows_docker_service_name(dev_app):
    transport = ASGITransport(app=dev_app)
    async with (
        respx.mock(base_url="http://legacy:8080") as mock,
        AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        mock.post("/memberADR/renderHtml").respond(
            200, json={"renderedHtml": "<html><body>docker</body></html>"}
        )
        resp = await client.get(
            "/dev/legacy-proxy?url=http%3A%2F%2Flegacy%3A8080%2FmemberADR%2FrenderHtml"
        )
    assert resp.status_code == 200
    assert "docker" in resp.text


@pytest.mark.asyncio
async def test_legacy_proxy_extra_host_from_env(monkeypatch):
    monkeypatch.setenv("ENABLE_DEV_ROUTES", "true")
    monkeypatch.setenv("DEV_LEGACY_PROXY_HOSTS", "legacy-alt, another-host")
    app = create_app()
    transport = ASGITransport(app=app)
    async with (
        respx.mock(base_url="http://legacy-alt:9090") as mock,
        AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        mock.post("/memberADR/renderHtml").respond(
            200, json={"renderedHtml": "<html><body>alt</body></html>"}
        )
        resp = await client.get(
            "/dev/legacy-proxy?url=http%3A%2F%2Flegacy-alt%3A9090%2FmemberADR%2FrenderHtml"
        )
    assert resp.status_code == 200
    assert "alt" in resp.text
