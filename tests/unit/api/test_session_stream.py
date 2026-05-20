"""Tests for streaming session init endpoint."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestSessionStreamEndpoint:
    """Tests for POST /api/v1/sessions/initialize/stream."""

    @pytest.mark.asyncio
    async def test_stream_init_returns_event_stream(self, client, auth_headers):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_pages_processed = 5
        mock_manager = MagicMock()

        with patch("src.api.routes.session_stream.initialize_session") as mock_init:
            mock_init.return_value = ("sess_stream_1", mock_result, mock_manager)
            response = await client.post(
                "/api/v1/sessions/initialize/stream",
                json={"gcs_uris": ["gs://bucket/doc.pdf"]},
                headers=auth_headers,
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        text = response.text
        assert "event: progress" in text
        assert "event: complete" in text

        # Parse complete event
        for block in text.strip().split("\n\n"):
            if "event: complete" in block:
                data_line = [line for line in block.split("\n") if line.startswith("data: ")][0]
                payload = json.loads(data_line[6:])
                assert payload["session_id"] == "sess_stream_1"
                assert payload["status"] == "ready"

    @pytest.mark.asyncio
    async def test_stream_init_error(self, client, auth_headers):
        with patch("src.api.routes.session_stream.initialize_session") as mock_init:
            mock_init.side_effect = Exception("OCR failed")
            response = await client.post(
                "/api/v1/sessions/initialize/stream",
                json={"gcs_uris": ["gs://bucket/doc.pdf"]},
                headers=auth_headers,
            )

        assert response.status_code == 200
        assert "event: error" in response.text

    @pytest.mark.asyncio
    async def test_stream_init_requires_auth(self, client):
        response = await client.post(
            "/api/v1/sessions/initialize/stream",
            json={"gcs_uris": ["gs://bucket/doc.pdf"]},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_stream_init_validates_empty_uris(self, client, auth_headers):
        response = await client.post(
            "/api/v1/sessions/initialize/stream",
            json={"gcs_uris": []},
            headers=auth_headers,
        )
        assert response.status_code == 422
