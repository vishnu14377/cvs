"""Observability middleware — request ID, timing, structured logging."""

from __future__ import annotations

import re
import time
import uuid

from src.core.logger import get_logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = get_logger(__name__)

_SESSION_PATH_RE = re.compile(r"/api/v1/sessions/([^/]+)")


def _extract_session_and_endpoint(path: str) -> tuple[str | None, str | None]:
    match = _SESSION_PATH_RE.search(path)
    if not match:
        return None, None
    session_id = match.group(1)
    remainder = path[match.end() :]
    if remainder.startswith("/"):
        endpoint = remainder.strip("/").split("/")[0] or "sessions"
    else:
        endpoint = "sessions"
    return session_id, endpoint


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Adds request ID, response timing, and structured request logging."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:16])
        start = time.monotonic()

        response = await call_next(request)

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)

        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = str(elapsed_ms)

        session_id, endpoint = _extract_session_and_endpoint(request.url.path)

        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": elapsed_ms,
                "session_id": session_id,
                "endpoint": endpoint,
            },
        )

        return response
