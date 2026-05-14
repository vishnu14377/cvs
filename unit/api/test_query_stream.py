"""Tests for streaming chat endpoint."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestStreamGraph:
    """Tests for stream_graph async generator."""

    @pytest.mark.asyncio
    async def test_stream_yields_done_event(self):
        from src.agents.graph import stream_graph

        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "The answer is 42."
        mock_ai_msg.type = "ai"
        mock_ai_msg.tool_calls = []

        mock_graph = AsyncMock()

        async def fake_astream(*args, **kwargs):
            yield {"generate": {"messages": [mock_ai_msg]}}

        mock_graph.astream = fake_astream

        # Stub grounding judge to return GROUNDED so we can assert raw content.
        async def fake_judge(content, tool_messages, session_id, **kwargs):
            return ("GROUNDED", content)

        with patch("src.api.validation.grounding_judge.judge_grounding", side_effect=fake_judge):
            events = []
            async for event in stream_graph(mock_graph, "What is the answer?", "sess_stream"):
                events.append(event)

        assert len(events) >= 1
        last = events[-1]
        assert "event: done" in last
        payload = json.loads(last.split("data: ")[1])
        assert payload["content"] == "The answer is 42."
        assert payload["grounding"] == "GROUNDED"
        assert "message_id" in payload


class TestStreamEndpoint:
    """Tests for POST /api/v1/sessions/{id}/query/stream."""

    @pytest.mark.asyncio
    async def test_stream_returns_event_stream(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        mock_manager = MagicMock()
        mock_manager.agent = MagicMock()
        registry["sess_sse"] = (mock_manager, time.time())

        try:

            async def fake_stream(graph, message, session_id):
                yield 'event: done\ndata: {"message_id": "msg_1", "content": "hello", "content_html": "<p>hello</p>", "content_base64": "PHA+"}\n\n'

            with patch("src.api.routes.query_stream.stream_graph", side_effect=fake_stream):
                response = await client.post(
                    "/api/v1/sessions/sess_sse/query/stream",
                    json={"message": "Hello"},
                    headers=auth_headers,
                )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            assert "event: done" in response.text
        finally:
            registry.pop("sess_sse", None)

    @pytest.mark.asyncio
    async def test_stream_session_not_found(self, client, auth_headers):
        response = await client.post(
            "/api/v1/sessions/nonexistent/query/stream",
            json={"message": "Hello"},
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_requires_auth(self, client):
        response = await client.post(
            "/api/v1/sessions/some_id/query/stream",
            json={"message": "Hello"},
        )
        assert response.status_code == 401
