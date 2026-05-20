"""Streaming session initialization — stub SSE with correct contract."""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from src.adr_document_processor import OcrModelType
from src.api.dependencies import get_session_registry, verify_token
from src.api.models.sessions import CreateSessionRequest
from src.core.logger import get_logger
from src.session_manager.initialization import initialize_session

logger = get_logger(__name__)

_OCR_ENGINE_MAP: dict[str, OcrModelType] = {
    "mistral": "mistral",
    "mistral-ocr": "mistral",
    "gemini-vision": "llm",
    "document-ai": "llm",
}

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["session-stream"],
    dependencies=[Depends(verify_token)],
)


async def _stream_init(body: CreateSessionRequest):
    """Async generator that yields SSE events during session init."""
    yield f"event: progress\ndata: {json.dumps({'status': 'initializing', 'message': 'Starting session...'})}\n\n"

    model_type = _OCR_ENGINE_MAP.get(body.ocr_engine)
    if model_type is None:
        yield f"event: error\ndata: {json.dumps({'detail': f'Unknown OCR engine: {body.ocr_engine}'})}\n\n"
        return

    gcs_uri = body.gcs_uris[0]
    start = time.monotonic()

    try:
        yield f"event: progress\ndata: {json.dumps({'status': 'processing', 'message': 'Processing documents...'})}\n\n"

        session_id, result, manager = await asyncio.to_thread(
            initialize_session,
            gcs_uri=gcs_uri,
            model_type=model_type,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        registry = get_session_registry()
        registry[session_id] = (manager, time.time())

        complete_payload = {
            "session_id": session_id,
            "status": "ready" if result.success else "error",
            "documents_processed": getattr(result, "total_pages_processed", 0),
            "processing_time_ms": elapsed_ms,
        }
        yield f"event: complete\ndata: {json.dumps(complete_payload)}\n\n"

    except Exception as e:
        logger.error("Streaming session init failed: %s", str(e), exc_info=True)
        yield f"event: error\ndata: {json.dumps({'detail': 'Session initialization failed. Please try again.'})}\n\n"


@router.post("/initialize/stream")
async def create_session_stream(body: CreateSessionRequest):
    """Initialize a session with SSE progress events."""
    return StreamingResponse(
        _stream_init(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
