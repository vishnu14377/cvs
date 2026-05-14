"""Tests for iFrame chat page endpoint."""

import time
from unittest.mock import MagicMock

import pytest


class TestChatPage:
    """Tests for GET /chat/{session_id}."""

    @pytest.mark.asyncio
    async def test_chat_page_returns_html(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        registry["sess_chat"] = (MagicMock(), time.time())

        try:
            response = await client.get("/chat/sess_chat")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            text = response.text
            assert "sess_chat" in text
            assert "chat-messages" in text
            assert "ADR AI Assistant" in text
        finally:
            registry.pop("sess_chat", None)

    @pytest.mark.asyncio
    async def test_chat_page_session_not_found(self, client):
        response = await client.get("/chat/nonexistent")
        assert response.status_code == 404
