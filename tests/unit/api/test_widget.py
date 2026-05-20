"""Tests for Unqork widget endpoint."""

import base64
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestWidgetQueryEndpoint:
    """Tests for POST /widget/v1/chat/query."""

    @pytest.mark.asyncio
    async def test_widget_query_returns_base64_html(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        mock_manager = MagicMock()
        mock_manager.session_id = "sess_widget"
        mock_manager.agent = MagicMock()
        registry["sess_widget"] = (mock_manager, time.time())

        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "The answer is **42**."
        mock_ai_msg.type = "ai"

        try:
            with patch("src.api.routes.widget.invoke_graph", new_callable=AsyncMock) as mock_invoke:
                mock_invoke.return_value = {"messages": [mock_ai_msg]}
                response = await client.post(
                    "/widget/v1/chat/query",
                    json={"session_id": "sess_widget", "message": "What is the answer?"},
                    headers=auth_headers,
                )

            assert response.status_code == 200
            data = response.json()

            # Verify base64 decodes to valid HTML
            decoded = base64.b64decode(data["content_base64"]).decode("utf-8")
            assert "<strong>42</strong>" in decoded
            assert "<script>" not in decoded

            # Verify HTML version matches
            assert "<strong>42</strong>" in data["content_html"]

            # Verify plain text
            assert "42" in data["content_text"]
        finally:
            registry.pop("sess_widget", None)

    @pytest.mark.asyncio
    async def test_widget_query_session_not_found(self, client, auth_headers):
        response = await client.post(
            "/widget/v1/chat/query",
            json={"session_id": "nonexistent", "message": "Hello"},
            headers=auth_headers,
        )
        assert response.status_code == 404
