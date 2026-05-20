"""Tests for feedback endpoint."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestFeedbackEndpoint:
    """Tests for POST /api/v1/sessions/{sessionId}/feedback."""

    @pytest.mark.asyncio
    async def test_submit_feedback_success(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        mock_manager = MagicMock()
        mock_manager.session_id = "sess_fb"
        mock_manager.agent = MagicMock()
        registry["sess_fb"] = (mock_manager, time.time())

        try:
            with patch("src.api.routes.feedback.get_feedback_collection") as mock_coll:
                mock_collection = AsyncMock()
                mock_collection.insert_one = AsyncMock(
                    return_value=MagicMock(inserted_id="fb_test")
                )
                mock_coll.return_value = mock_collection

                response = await client.post(
                    "/api/v1/sessions/sess_fb/feedback",
                    json={
                        "message_id": "msg_001",
                        "rating": "positive",
                        "comment": "Great answer",
                    },
                    headers=auth_headers,
                )

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "sess_fb"
            assert data["message_id"] == "msg_001"
            assert data["status"] == "stored"
        finally:
            registry.pop("sess_fb", None)

    @pytest.mark.asyncio
    async def test_submit_feedback_session_not_found(self, client, auth_headers):
        response = await client.post(
            "/api/v1/sessions/nonexistent/feedback",
            json={"message_id": "msg_001", "rating": "positive"},
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_rating(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        registry["sess_fb2"] = (MagicMock(), time.time())
        try:
            response = await client.post(
                "/api/v1/sessions/sess_fb2/feedback",
                json={"message_id": "msg_001", "rating": "maybe"},
                headers=auth_headers,
            )
            assert response.status_code == 422
        finally:
            registry.pop("sess_fb2", None)


class TestGetFeedback:
    """Tests for GET /api/v1/sessions/{sessionId}/feedback."""

    @pytest.mark.asyncio
    async def test_get_feedback_success(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        registry["sess_get_fb"] = (MagicMock(), time.time())

        try:
            with patch("src.api.routes.feedback.get_feedback_collection") as mock_coll:
                mock_collection = AsyncMock()
                mock_cursor = AsyncMock()
                mock_cursor.to_list = AsyncMock(
                    return_value=[
                        {
                            "_id": "fb_1",
                            "session_id": "sess_get_fb",
                            "message_id": "msg_1",
                            "rating": "positive",
                            "comment": "Good",
                            "created_at": "2026-04-17T00:00:00",
                        },
                    ]
                )
                mock_find = MagicMock()
                mock_find.return_value.sort.return_value = mock_cursor
                mock_collection.find = mock_find
                mock_coll.return_value = mock_collection

                response = await client.get(
                    "/api/v1/sessions/sess_get_fb/feedback",
                    headers=auth_headers,
                )

            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "sess_get_fb"
            assert len(data["feedback"]) == 1
            assert data["feedback"][0]["rating"] == "positive"
        finally:
            registry.pop("sess_get_fb", None)

    @pytest.mark.asyncio
    async def test_get_feedback_session_not_found(self, client, auth_headers):
        response = await client.get(
            "/api/v1/sessions/nonexistent/feedback",
            headers=auth_headers,
        )
        assert response.status_code == 404
