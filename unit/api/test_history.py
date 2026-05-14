"""Tests for history endpoint."""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestHistoryEndpoint:
    """Tests for GET /api/v1/sessions/{sessionId}/history."""

    @pytest.mark.asyncio
    async def test_history_returns_messages(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        mock_manager = MagicMock()
        mock_manager.session_id = "sess_hist"
        mock_manager.agent = MagicMock()
        registry["sess_hist"] = (mock_manager, time.time())

        mock_human = MagicMock()
        mock_human.content = "What is X?"
        mock_human.type = "human"

        mock_ai = MagicMock()
        mock_ai.content = "X is a thing."
        mock_ai.type = "ai"

        try:
            with patch("src.api.routes.history.get_session_history") as mock_hist:
                mock_hist.return_value = [mock_human, mock_ai]
                response = await client.get(
                    "/api/v1/sessions/sess_hist/history",
                    headers=auth_headers,
                )

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "sess_hist"
            assert len(data["messages"]) == 2
            assert data["messages"][0]["role"] == "human"
            assert data["messages"][1]["role"] == "ai"
        finally:
            registry.pop("sess_hist", None)

    @pytest.mark.asyncio
    async def test_history_session_not_found(self, client, auth_headers):
        response = await client.get(
            "/api/v1/sessions/nonexistent/history",
            headers=auth_headers,
        )
        assert response.status_code == 404
