"""Conversation history endpoint."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from src.agents.graph import get_session_history
from src.api.dependencies import get_session_manager, verify_token
from src.api.models.query import HistoryMessage, HistoryResponse
from src.api.rendering.html_renderer import render_to_safe_html
from src.api.validation.phi_redaction import redact_phi
from src.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["history"],
    dependencies=[Depends(verify_token)],
)


@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get conversation history for a session."""
    manager = get_session_manager(session_id)
    graph = manager.agent

    raw_messages = get_session_history(graph, session_id)

    messages = []
    for msg in raw_messages:
        role = getattr(msg, "type", "unknown")
        raw_content = msg.content if msg.content else ""
        # LangGraph messages may carry list-of-parts content (tool call / multimodal).
        # Flatten to a plain string so the Pydantic model stays valid.
        if isinstance(raw_content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            ).strip()
        else:
            content = str(raw_content)
        content = redact_phi(content)
        content_html = render_to_safe_html(content) if role == "ai" else None
        messages.append(
            HistoryMessage(
                message_id=f"msg_{uuid.uuid4().hex[:12]}",
                role=role,
                content=content,
                content_html=content_html,
            )
        )

    return HistoryResponse(session_id=session_id, messages=messages)
