"""Feedback endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from src.agents.graph import get_session_history
from src.api.dependencies import get_session_manager, verify_token
from src.api.models.feedback import (
    FeedbackEntry,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from src.api.validation.phi_redaction import redact_phi
from src.core.logger import get_logger
from src.feedback_manager.client import get_feedback_collection
from src.feedback_manager.models import FeedbackRecord
from src.feedback_manager.repository import FeedbackRepository

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["feedback"],
    dependencies=[Depends(verify_token)],
)


@router.post("/{session_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(session_id: str, body: FeedbackRequest):
    """Submit user feedback on an agent response."""
    manager = get_session_manager(session_id)

    collection = get_feedback_collection()
    if collection is None:
        logger.info("Feedback submitted but MongoDB not configured — silently discarding (session=%s)", session_id)
        return FeedbackResponse(
            feedback_id="not-stored",
            session_id=session_id,
            message_id=body.message_id,
            status="discarded",
        )

    # Extract user message, AI response, tools used, and document names from history
    user_message = ""
    ai_response = ""
    tools_used = []
    document_names = []
    try:
        history = get_session_history(manager.agent, session_id)
        for _i, msg in enumerate(history):
            msg_type = getattr(msg, "type", "")
            raw = msg.content or ""
            content = (
                str(raw)
                if not isinstance(raw, list)
                else " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in raw
                ).strip()
            )
            if msg_type == "human":
                user_message = content
            elif msg_type == "ai" and content:
                ai_response = content
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = (
                            tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                        )
                        if name and name not in tools_used:
                            tools_used.append(name)
            elif msg_type == "tool":
                tool_name = getattr(msg, "name", "")
                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)
                if "Source:" in content:
                    for line in content.split("\n"):
                        if line.startswith("Source: "):
                            doc = line.split(": ", 1)[1]
                            if doc not in document_names:
                                document_names.append(doc)
    except Exception as e:
        logger.warning("Could not extract history for feedback: %s", e)

    record = FeedbackRecord(
        session_id=session_id,
        message_id=body.message_id,
        user_message=redact_phi(user_message),
        ai_response=redact_phi(ai_response),
        rating=body.rating,
        comment=redact_phi(body.comment) if body.comment else None,
        document_names=document_names,
        tools_used=tools_used,
    )

    repo = FeedbackRepository(collection=collection)
    try:
        feedback_id = await repo.store(record)
    except Exception as e:
        logger.error("Feedback storage failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to store feedback. Please try again.",
        ) from None

    return FeedbackResponse(
        feedback_id=feedback_id,
        session_id=session_id,
        message_id=body.message_id,
        status="stored",
    )


@router.get("/{session_id}/feedback", response_model=FeedbackListResponse)
async def get_session_feedback(session_id: str):
    """Get all feedback for a session."""
    get_session_manager(session_id)

    collection = get_feedback_collection()
    if collection is None:
        return FeedbackListResponse(session_id=session_id, feedback=[])

    repo = FeedbackRepository(collection=collection)
    records = await repo.get_by_session(session_id)

    feedback = [
        FeedbackEntry(
            feedback_id=r.get("_id", ""),
            message_id=r.get("message_id", ""),
            rating=r.get("rating", ""),
            comment=r.get("comment"),
            created_at=r.get("created_at"),
        )
        for r in records
    ]

    return FeedbackListResponse(session_id=session_id, feedback=feedback)
