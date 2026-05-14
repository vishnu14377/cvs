"""Development/testing routes — only registered in non-production environments."""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(tags=["dev"])


@router.get("/", response_class=RedirectResponse, status_code=302)
async def root():
    """Redirect root to the test harness."""
    return "/dev/test"


@router.get("/dev/test", response_class=HTMLResponse)
async def test_harness(request: Request):
    """Serve the API test harness page (no auth — page handles tokens for API calls)."""
    return templates.TemplateResponse(
        request,
        "test_harness.html",
    )


_DEFAULT_ALLOWED_PROXY_HOSTS = {"localhost", "127.0.0.1", "legacy", "legacy-java"}


def _allowed_proxy_hosts() -> set[str]:
    """Resolve allowed proxy hosts — default set plus comma-separated env extras."""
    extras = os.getenv("DEV_LEGACY_PROXY_HOSTS", "")
    hosts = set(_DEFAULT_ALLOWED_PROXY_HOSTS)
    for host in extras.split(","):
        host = host.strip()
        if host:
            hosts.add(host)
    return hosts


@router.get("/dev/legacy-proxy")
async def legacy_proxy(
    url: str = Query(...),
    method: str = Query("POST"),
    body: str = Query(None),
):
    """Proxy a request to a legacy service (dev-only, allowlisted hosts only).

    Exists solely so the /dev/unqork-mock harness page can fetch legacy
    FTL-rendered HTML (POST /memberADR/renderHtml → HTML) or status JSON
    (GET /memberADR/v1/status/{id} → JSON) without CORS pain.

    Allowlist defaults to localhost, 127.0.0.1, and the `legacy` docker-compose
    service name. Extend via DEV_LEGACY_PROXY_HOSTS (comma-separated).
    """
    parsed = urlparse(url)
    if parsed.hostname not in _allowed_proxy_hosts():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="legacy-proxy host not in allowlist",
        )

    import json as _json
    body_json = _json.loads(body) if body else {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        if method.upper() == "POST":
            r = await client.post(url, json=body_json)
        else:
            r = await client.get(url)

    content_type = r.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = r.json()
            if isinstance(data, dict) and "renderedHtml" in data:
                return HTMLResponse(content=data["renderedHtml"])
            if isinstance(data, dict) and "htmlData" in data:
                import base64
                html = base64.b64decode(data["htmlData"]).decode("utf-8")
                return HTMLResponse(content=html)
            from fastapi.responses import JSONResponse
            return JSONResponse(content=data)
        except ValueError:
            pass
    return HTMLResponse(content=r.text)


@router.get("/dev/unqork-mock", response_class=HTMLResponse)
async def unqork_mock(request: Request):
    """Dev harness that simulates Unqork embedding modes.

    `default_legacy_url` lets docker-compose users point the harness at the
    `legacy` service hostname without editing the form each reload.
    """
    return templates.TemplateResponse(
        request,
        "unqork_mock.html",
        {
            "auth_token": os.getenv("CARECONNECT_WIDGET_TOKEN", os.getenv("API_AUTH_TOKEN", "")),
            "default_legacy_url": os.getenv(
                "DEV_DEFAULT_LEGACY_URL",
                "http://localhost:8080/memberADR/renderHtml",
            ),
            "default_legacy_java_url": os.getenv(
                "DEV_DEFAULT_LEGACY_JAVA_URL",
                "http://localhost:8081/caremanagement/cs/v1/edb/additionalclinicaldocumentations/at/adrs/details/retrieve",
            ),
        },
    )


class DemoSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class DemoSessionResponse(BaseModel):
    session_id: str
    status: str = "ready"


@router.post("/dev/demo-session", response_model=DemoSessionResponse)
async def create_demo_session(body: DemoSessionRequest):
    """Dev-only: register an in-memory session with no ADR documents.

    Lets the harness exercise the widget chat without running OCR or
    vector-DB setup. The agent answers from prompt context only.
    """
    from src.api.dependencies import get_session_registry
    from src.api.dev.stub_session import StubSessionManager

    registry = get_session_registry()
    # Idempotent: don't overwrite an existing session
    if body.session_id in registry:
        return DemoSessionResponse(session_id=body.session_id, status="exists")

    registry[body.session_id] = (StubSessionManager(body.session_id), time.time())
    return DemoSessionResponse(session_id=body.session_id, status="ready")
