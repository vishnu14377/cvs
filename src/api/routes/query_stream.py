"""Streaming query endpoint — SSE for real-time agent responses."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from src.agents.graph import stream_graph
from src.api.dependencies import get_session_manager, verify_token
from src.api.models.query import QueryRequest
from src.api.validation.input_safety import check_injection_regex, classify_input_safety
from src.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["query-stream"],
    dependencies=[Depends(verify_token)],
)


@router.post("/{session_id}/query/stream")
async def query_session_stream(session_id: str, body: QueryRequest):
    """Stream agent response as Server-Sent Events."""
    manager = get_session_manager(session_id)
    graph = manager.agent

    if check_injection_regex(body.message):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your query could not be processed. Please rephrase your question.",
        )

    if await classify_input_safety(body.message) == "UNSAFE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your query could not be processed. Please rephrase your question.",
        )

    return StreamingResponse(
        stream_graph(graph, body.message, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
