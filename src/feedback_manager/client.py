"""MongoDB async client for feedback storage."""

from __future__ import annotations

import os

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from src.core.logger import get_logger

logger = get_logger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_feedback_db() -> AsyncIOMotorDatabase | None:
    """Get the feedback MongoDB database (lazy init).

    Returns None if MONGODB_URI is not configured.
    """
    global _client, _db

    if _db is not None:
        return _db

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        logger.debug("MONGODB_URI not set — feedback storage disabled")
        return None

    db_name = os.environ.get("MONGODB_DATABASE", "adr_ai_agent")
    _client = AsyncIOMotorClient(uri)
    _db = _client[db_name]
    logger.info("Connected to MongoDB database: %s", db_name)
    return _db


def get_feedback_collection():
    """Get the feedback collection. Returns None if not configured."""
    db = get_feedback_db()
    if db is None:
        return None
    return db["feedback"]


async def close_feedback_client():
    """Close the MongoDB client connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
