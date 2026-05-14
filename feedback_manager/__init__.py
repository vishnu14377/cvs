"""Feedback manager — capture and store user feedback in MongoDB."""

from src.feedback_manager.client import close_feedback_client, get_feedback_collection
from src.feedback_manager.models import FeedbackRecord
from src.feedback_manager.repository import FeedbackRepository

__all__ = [
    "FeedbackRecord",
    "FeedbackRepository",
    "get_feedback_collection",
    "close_feedback_client",
]
