"""Query endpoint — send a question, get agent response."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from src.agents.graph import invoke_graph
from src.api.dependencies import get_session_manager, verify_token
from src.api.models.query import MessageResponse, QueryRequest, QueryResponse, SourceReference
from src.api.rendering.html_renderer import render_to_base64, render_to_safe_html
from src.api.validation.grounding_judge import judge_grounding
from src.api.validation.input_safety import check_injection_regex, classify_input_safety
from src.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["query"],
    dependencies=[Depends(verify_token)],
)


@router.post("/{session_id}/query", response_model=QueryResponse)
async def query_session(session_id: str, body: QueryRequest):
    """Send a user question to the agent and get a response."""
    manager = get_session_manager(session_id)
    graph = manager.agent

    if check_injection_regex(body.message):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your query could not be processed. Please rephrase your question.",
        )

    classifier_task = asyncio.create_task(classify_input_safety(body.message))

    start = time.monotonic()
    try:
        result = await invoke_graph(graph, body.message, session_id)
    except Exception as e:
        logger.error("Query failed: session=%s, error=%s", session_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process your question. Please try again.",
        ) from None
    elapsed_ms = int((time.monotonic() - start) * 1000)

    try:
        safety = await classifier_task
    except Exception:
        safety = "SAFE"
    if safety == "UNSAFE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your query could not be processed. Please rephrase your question.",
        )

    messages = result.get("messages", [])

    # Extract the last AI message
    ai_content = ""
    token_usage = {}
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and msg.content:
            raw = msg.content
            if isinstance(raw, list):
                ai_content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in raw
                ).strip()
            else:
                ai_content = str(raw)
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                token_usage = {
                    "prompt": getattr(usage, "input_tokens", 0),
                    "completion": getattr(usage, "output_tokens", 0),
                }
            break

    if token_usage:
        logger.info(
            "token_usage",
            extra={
                "input_tokens": token_usage.get("prompt", 0),
                "output_tokens": token_usage.get("completion", 0),
                "session_id": session_id,
                "endpoint": "query",
            },
        )

    # Extract source references from tool messages
    tool_messages = [m for m in messages if getattr(m, "type", None) == "tool"]
    sources = []
    for msg in tool_messages:
        if getattr(msg, "name", "") in ("adr_search", "policy_search"):
            content = msg.content or ""
            current_source = None
            current_page = None
            current_text = ""
            for line in content.split("\n"):
                if line.startswith("Source: ") or line.startswith("Policy: "):
                    if current_source:
                        sources.append(
                            SourceReference(
                                document=current_source,
                                page=current_page,
                                chunk_text=current_text.strip()[:500]
                                if current_text.strip()
                                else None,
                            )
                        )
                    current_source = line.split(": ", 1)[1] if ": " in line else line
                    current_page = None
                    current_text = ""
                elif line.startswith("Page: "):
                    with contextlib.suppress(ValueError, IndexError):
                        current_page = int(line.split(": ")[1])
                elif line.startswith("Content:"):
                    current_text = ""
                elif current_source and not line.startswith("---"):
                    current_text += line + "\n"
            if current_source:
                sources.append(
                    SourceReference(
                        document=current_source,
                        page=current_page,
                        chunk_text=current_text.strip()[:500] if current_text.strip() else None,
                    )
                )

    # Grounding judge
    verdict, ai_content = await judge_grounding(ai_content, tool_messages, session_id)

    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    return QueryResponse(
        session_id=session_id,
        message_id=message_id,
        message=MessageResponse(
            role="assistant",
            content=ai_content,
            content_html=render_to_safe_html(ai_content),
            content_base64=render_to_base64(ai_content),
        ),
        sources=sources,
        metadata={
            "processing_time_ms": elapsed_ms,
            "tokenUsage": token_usage if token_usage else None,
            "grounding": verdict,
        },
    )
