"""iFrame chat page — serves a standalone HTML chat UI."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.api.dependencies import get_session_registry
from src.core.logger import get_logger

logger = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(tags=["chat"])


@router.get("/chat/{session_id}", response_class=HTMLResponse)
async def chat_page(request: Request, session_id: str):
    """Serve the standalone chat page for iFrame embedding."""
    registry = get_session_registry()
    if session_id not in registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session '{session_id}' not found"
        )

    api_base = str(request.base_url).rstrip("/")
    auth_token = os.environ.get("API_AUTH_TOKEN", "")

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "session_id": session_id,
            "api_base": api_base,
            "auth_token": auth_token,
        },
    )
