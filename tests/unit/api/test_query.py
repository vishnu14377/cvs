"""Tests for query endpoint."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQueryEndpoint:
    """Tests for POST /api/v1/sessions/{sessionId}/query."""

    @pytest.mark.asyncio
    async def test_query_success(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        mock_manager = MagicMock()
        mock_manager.session_id = "sess_query"
        mock_manager.agent = MagicMock()
        registry["sess_query"] = (mock_manager, time.time())

        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "The diagnosis is X."
        mock_ai_msg.type = "ai"

        mock_result = {"messages": [mock_ai_msg]}

        try:
            with (
                patch("src.api.routes.query.invoke_graph", new_callable=AsyncMock) as mock_invoke,
                patch("src.api.routes.query.judge_grounding", new_callable=AsyncMock) as mock_judge,
            ):
                mock_invoke.return_value = mock_result
                mock_judge.return_value = ("GROUNDED", "The diagnosis is X.")
                response = await client.post(
                    "/api/v1/sessions/sess_query/query",
                    json={"message": "What is the diagnosis?"},
                    headers=auth_headers,
                )

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "sess_query"
            assert data["message"]["content"] == "The diagnosis is X."
            assert data["message"]["content_html"] != ""
            assert data["message"]["content_base64"] != ""
            assert data["metadata"]["grounding"] == "GROUNDED"
        finally:
            registry.pop("sess_query", None)

    @pytest.mark.asyncio
    async def test_query_extracts_sources_and_token_usage(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        mock_manager = MagicMock()
        mock_manager.agent = MagicMock()
        registry["sess_src"] = (mock_manager, time.time())

        mock_tool_msg = MagicMock()
        mock_tool_msg.type = "tool"
        mock_tool_msg.name = "adr_search"
        mock_tool_msg.content = (
            "Found 1 relevant document(s):\n\n"
            "--- Document 1 ---\n"
            "Source: ADR-12345.pdf\n"
            "Page: 7\n"
            "Content:\nPatient shows signs of adverse reaction.\n"
        )

        mock_ai_msg = MagicMock()
        mock_ai_msg.type = "ai"
        mock_ai_msg.content = "The patient had an adverse reaction."
        mock_ai_msg.tool_calls = []
        mock_usage = MagicMock()
        mock_usage.input_tokens = 1200
        mock_usage.output_tokens = 350
        mock_ai_msg.usage_metadata = mock_usage

        try:
            with patch("src.api.routes.query.invoke_graph", new_callable=AsyncMock) as mock_invoke:
                mock_invoke.return_value = {"messages": [mock_tool_msg, mock_ai_msg]}
                response = await client.post(
                    "/api/v1/sessions/sess_src/query",
                    json={"message": "What happened?"},
                    headers=auth_headers,
                )

            assert response.status_code == 200
            data = response.json()
            assert len(data["sources"]) == 1
            assert data["sources"][0]["document"] == "ADR-12345.pdf"
            assert data["sources"][0]["page"] == 7
            assert "adverse reaction" in data["sources"][0]["chunk_text"]
            assert data["metadata"]["tokenUsage"]["prompt"] == 1200
            assert data["metadata"]["tokenUsage"]["completion"] == 350
        finally:
            registry.pop("sess_src", None)

    @pytest.mark.asyncio
    async def test_query_session_not_found(self, client, auth_headers):
        response = await client.post(
            "/api/v1/sessions/nonexistent/query",
            json={"message": "Hello"},
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_query_requires_auth(self, client):
        response = await client.post(
            "/api/v1/sessions/some_id/query",
            json={"message": "Hello"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_query_validates_empty_message(self, client, auth_headers):
        response = await client.post(
            "/api/v1/sessions/some_id/query",
            json={"message": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422
