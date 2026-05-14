"""Widget API for Unqork integration.

Returns base64-encoded safe HTML for the Unqork Plug-In + HTML Element pattern.
"""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.agents.graph import invoke_graph
from src.api.dependencies import get_session_manager, verify_token
from src.api.models.widget import WidgetQueryRequest, WidgetQueryResponse
from src.api.rendering.html_renderer import render_to_base64, render_to_safe_html
from src.api.validation.grounding_judge import judge_grounding
from src.api.validation.input_safety import check_injection_regex, classify_input_safety
from src.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/widget/v1/chat",
    tags=["widget"],
    dependencies=[Depends(verify_token)],
)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# No-auth router for browser-facing UI fragments (iframed into Unqork).
# The UI page itself is not sensitive; auth is enforced on /query.
ui_router = APIRouter(prefix="/widget/v1/chat", tags=["widget"])


@ui_router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def widget_chat_ui(
    request: Request,
    sessionId: str = Query(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    mode: Literal["iframe", "fragment", "separate-popup", "floating-bubble"] = "iframe",
    layout: Literal["default", "embedded"] = "default",
):
    """Serve the self-contained widget chat UI (for iframe embedding)."""
    return _templates.TemplateResponse(
        request,
        "widget_chat.html",
        {
            "session_id": sessionId,
            "mode": mode,
            "layout": layout,
            "api_base": os.getenv("CARECONNECT_API_BASE", ""),
            "auth_token": os.getenv("CARECONNECT_WIDGET_TOKEN", ""),
            "parent_origin": os.getenv("CARECONNECT_PARENT_ORIGIN", ""),
        },
    )


@ui_router.get("/fragment", response_class=HTMLResponse, include_in_schema=False)
async def widget_chat_fragment(
    request: Request,
    sessionId: str = Query(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
):
    """Return an initial chat shell as pre-sanitized HTML (no script tags).

    Used when Unqork injects HTML into a div and cannot allow iframes.
    The hosting Unqork widget supplies a single trusted JS that attaches
    delegated handlers to [data-cc-role] elements.
    """
    return _templates.TemplateResponse(
        request,
        "widget_fragment.html",
        {"session_id": sessionId},
    )


__all__ = ["router", "ui_router"]


@router.post("/query", response_model=WidgetQueryResponse)
async def widget_query(body: WidgetQueryRequest):
    """Query the agent and return Unqork-formatted response.

    Session ID is in the body (not the URL path) because Unqork
    Plug-In components may have limitations on dynamic URL construction.
    """
    from datetime import datetime, timezone

    manager = get_session_manager(body.session_id)
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

    try:
        result = await invoke_graph(graph, body.message, body.session_id)
    except Exception as e:
        logger.error("Widget query failed: session=%s, error=%s", body.session_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process your question. Please try again.",
        ) from None

    # Extract last AI message for plain text and grounding
    messages = result.get("messages", [])
    ai_content = ""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and msg.content:
            raw = msg.content
            if isinstance(raw, list):
                ai_content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part) for part in raw
                ).strip()
            else:
                ai_content = str(raw)
            break

    tool_messages = [m for m in messages if getattr(m, "type", None) == "tool"]
    verdict, ai_content = await judge_grounding(ai_content, tool_messages, body.session_id)

    generated_at = datetime.now(timezone.utc).isoformat()

    # Build conversation message list for full-conversation rendering
    conversation: list[dict] = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        raw_content = msg.content if msg.content else ""
        if isinstance(raw_content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            ).strip()
        else:
            content = str(raw_content)

        if msg_type == "human":
            conversation.append({"role": "user", "content": content})
        elif msg_type == "ai" and content:
            entry: dict = {"role": "assistant", "content": content}

            # Check if this is the last AI message (use grounded content)
            if msg is messages[-1] or all(
                getattr(m, "type", "") != "ai"
                for m in messages[messages.index(msg) + 1 :]
            ):
                entry["content"] = ai_content
                entry["generated_at"] = generated_at
                # Extract sources from tool messages
                sources: list[dict] = []
                for tm in tool_messages:
                    if getattr(tm, "name", "") in ("adr_search", "policy_search"):
                        tm_content = tm.content or ""
                        current_source = None
                        current_page = None
                        for line in tm_content.split("\n"):
                            if line.startswith("Source: ") or line.startswith("Policy: "):
                                if current_source:
                                    sources.append({"document": current_source, "gcs_uri": current_source, "page": current_page})
                                current_source = line.split(": ", 1)[1] if ": " in line else line
                                current_page = None
                            elif line.startswith("Page: "):
                                try:
                                    current_page = int(line.split(": ")[1])
                                except (ValueError, IndexError):
                                    pass
                        if current_source:
                            sources.append({"document": current_source, "gcs_uri": current_source, "page": current_page})
                if sources:
                    entry["sources"] = sources

            conversation.append(entry)

    from src.api.rendering.html_renderer import render_conversation_html

    conversation_html = render_conversation_html(conversation)
    conversation_base64 = base64.b64encode(conversation_html.encode("utf-8")).decode("ascii")

    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    return WidgetQueryResponse(
        content_base64=conversation_base64,
        content_html=conversation_html,
        content_text=ai_content,
        message_id=message_id,
        generated_at=generated_at,
    )
