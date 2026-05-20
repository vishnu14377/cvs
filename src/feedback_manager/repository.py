"""Feedback CRUD operations."""

from __future__ import annotations

import uuid
from dataclasses import asdict

from src.core.logger import get_logger
from src.feedback_manager.models import FeedbackRecord

logger = get_logger(__name__)


class FeedbackRepository:
    """Repository for storing and retrieving feedback records."""

    def __init__(self, collection):
        """Initialize with a MongoDB collection (or AsyncMock for testing)."""
        self._collection = collection

    async def store(self, record: FeedbackRecord) -> str:
        """Store a feedback record. Returns the feedback ID."""
        doc = asdict(record)
        doc["_id"] = f"fb_{uuid.uuid4().hex[:12]}"
        doc["created_at"] = record.created_at.isoformat()

        await self._collection.insert_one(doc)
        logger.info(
            "Stored feedback %s for session %s, message %s",
            doc["_id"],
            record.session_id,
            record.message_id,
        )
        return doc["_id"]

    async def get_by_session(self, session_id: str) -> list:
        """Get all feedback records for a session, ordered by creation time."""
        cursor = self._collection.find({"session_id": session_id}).sort("created_at", 1)
        results = await cursor.to_list(length=100)
        logger.info("Retrieved %d feedback records for session %s", len(results), session_id)
        return results
