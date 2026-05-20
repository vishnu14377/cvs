"""Tests for feedback repository."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.feedback_manager.models import FeedbackRecord
from src.feedback_manager.repository import FeedbackRepository


class TestFeedbackRepository:
    """Tests for FeedbackRepository."""

    @pytest.fixture
    def mock_collection(self):
        collection = AsyncMock()
        collection.insert_one = AsyncMock()
        return collection

    @pytest.fixture
    def repo(self, mock_collection):
        return FeedbackRepository(collection=mock_collection)

    @pytest.mark.asyncio
    async def test_store_feedback_returns_id(self, repo, mock_collection):
        mock_collection.insert_one.return_value = MagicMock(inserted_id="fb_123")

        record = FeedbackRecord(
            session_id="sess_1",
            message_id="msg_1",
            user_message="What is X?",
            ai_response="X is a thing.",
            rating="positive",
        )

        feedback_id = await repo.store(record)
        assert feedback_id is not None
        mock_collection.insert_one.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_feedback_includes_all_fields(self, repo, mock_collection):
        mock_collection.insert_one.return_value = MagicMock(inserted_id="fb_456")

        record = FeedbackRecord(
            session_id="sess_2",
            message_id="msg_2",
            user_message="Question",
            ai_response="Answer",
            rating="negative",
            comment="Wrong answer",
            document_names=["doc.pdf"],
        )

        await repo.store(record)
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["session_id"] == "sess_2"
        assert call_args["rating"] == "negative"
        assert call_args["comment"] == "Wrong answer"
        assert call_args["document_names"] == ["doc.pdf"]

    @pytest.mark.asyncio
    async def test_get_by_session_returns_records(self, repo, mock_collection):
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(
            return_value=[
                {
                    "_id": "fb_1",
                    "session_id": "sess_1",
                    "message_id": "msg_1",
                    "rating": "positive",
                    "created_at": "2026-04-17T00:00:00",
                },
                {
                    "_id": "fb_2",
                    "session_id": "sess_1",
                    "message_id": "msg_2",
                    "rating": "negative",
                    "created_at": "2026-04-17T01:00:00",
                },
            ]
        )
        mock_find = MagicMock()
        mock_find.return_value.sort.return_value = mock_cursor
        mock_collection.find = mock_find

        results = await repo.get_by_session("sess_1")
        assert len(results) == 2
        assert results[0]["_id"] == "fb_1"
        mock_collection.find.assert_called_once_with({"session_id": "sess_1"})

    @pytest.mark.asyncio
    async def test_get_by_session_empty(self, repo, mock_collection):
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_find = MagicMock()
        mock_find.return_value.sort.return_value = mock_cursor
        mock_collection.find = mock_find

        results = await repo.get_by_session("sess_empty")
        assert results == []
