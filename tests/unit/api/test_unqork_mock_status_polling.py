"""Tests that the harness page wires status polling to /memberADR/v1/status."""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from src.api.app import create_app


@pytest.mark.asyncio
async def test_harness_includes_status_poll_hook(monkeypatch):
    monkeypatch.setenv("ENABLE_DEV_ROUTES", "true")
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dev/unqork-mock")
    body = resp.text
    # Should include a function that polls /memberADR/v1/status/{id} when buttons are clicked
    assert "memberADR/v1/status" in body
    assert "pollStatus" in body or "statusPoll" in body
