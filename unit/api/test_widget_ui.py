"""Tests for widget chat UI route."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from src.api.app import create_app


@pytest.mark.asyncio
async def test_widget_chat_ui_returns_html_200():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=abc&mode=iframe")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_widget_chat_ui_has_no_external_resources():
    """Unqork strips external CSS/JS loads — the page must be self-contained."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=abc&mode=iframe")
    body = resp.text
    assert '<link rel="stylesheet"' not in body
    assert "<script src=" not in body
    # Session ID must appear so the page can talk to the right agent session
    assert "abc" in body


@pytest.mark.asyncio
async def test_widget_chat_ui_emits_postmessage_events():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=abc&mode=iframe")
    body = resp.text
    assert "careconnect:ready" in body
    assert "careconnect:height" in body


@pytest.mark.asyncio
async def test_widget_chat_ui_rejects_invalid_session_id():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=../evil&mode=iframe")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_widget_chat_ui_rejects_unknown_mode():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=abc&mode=bogus")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_widget_chat_ui_all_outbound_events_in_body():
    """Ensure all three postMessage event names appear in the page source."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=abc&mode=iframe")
    for event in ("careconnect:ready", "careconnect:height", "careconnect:close"):
        assert event in resp.text


@pytest.mark.asyncio
async def test_widget_chat_fragment_returns_script_free_html():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/fragment?sessionId=abc")
    assert resp.status_code == 200
    body = resp.text
    assert "<script" not in body.lower()
    assert 'data-cc-session="abc"' in body
    # Must include elements Unqork's glue JS can target
    assert 'data-cc-role="input"' in body
    assert 'data-cc-role="send"' in body
    assert 'data-cc-role="messages"' in body


@pytest.mark.asyncio
async def test_widget_chat_fragment_rejects_invalid_session_id():
    """Same input validation as /ui route — consistent hygiene."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/fragment?sessionId=../evil")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_widget_chat_ui_embedded_layout_hides_header():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=abc&mode=iframe&layout=embedded")
    assert resp.status_code == 200
    assert "cc-layout-embedded" in resp.text


@pytest.mark.asyncio
async def test_widget_chat_ui_embedded_layout_has_class_on_body():
    """The body tag should have the cc-layout-embedded class when layout=embedded."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=abc&mode=iframe&layout=embedded")
    assert 'class="cc-layout-embedded"' in resp.text


@pytest.mark.asyncio
async def test_widget_chat_ui_default_layout_no_class_on_body():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/widget/v1/chat/ui?sessionId=abc&mode=iframe")
    assert resp.status_code == 200
    assert 'class="cc-layout-embedded"' not in resp.text
