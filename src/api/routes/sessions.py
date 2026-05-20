# src/api/routes/sessions.py
"""Session management endpoints."""

from __future__ import annotations

import os
import time

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from src.adr_document_processor import OcrModelType
from src.api.dependencies import get_session_manager, get_session_registry, verify_token
from src.api.models.sessions import (
    CreateSessionRequest,
    SessionDeleteResponse,
    SessionListResponse,
    SessionResponse,
)
from src.core.gcs_client import upload_to_gcs
from src.core.logger import get_logger
from src.session_manager.deletion import delete_session
from src.session_manager.initialization import initialize_session
from src.session_manager.warmup import warmup_session

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["sessions"],
    dependencies=[Depends(verify_token)],
)

# Map API engine names to the OcrModelType literal the engine expects
_OCR_ENGINE_MAP: dict[str, OcrModelType] = {
    "mistral": "mistral",
    "mistral-ocr": "mistral",
    "gemini-vision": "llm",
    "document-ai": "llm",  # TODO: add Document AI support when available
}


@router.get("", response_model=SessionListResponse)
async def list_sessions():
    """List all active sessions."""
    registry = get_session_registry()
    sessions = []
    for sid, (manager, _created) in registry.items():
        pages = 0
        if manager.result:
            pages = getattr(manager.result, "total_pages_processed", 0)
        sessions.append(
            SessionResponse(
                session_id=sid,
                status="ready",
                documents_processed=pages,
            )
        )
    return SessionListResponse(sessions=sessions)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SessionResponse)
async def create_session(body: CreateSessionRequest, background_tasks: BackgroundTasks):
    """Initialize a new session with ADR documents from GCS URIs."""
    registry = get_session_registry()

    model_type = _OCR_ENGINE_MAP.get(body.ocr_engine)
    if model_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown OCR engine: {body.ocr_engine}. Use: {list(_OCR_ENGINE_MAP.keys())}",
        )

    start = time.monotonic()

    # Process each GCS URI (first URI used as primary for now)
    gcs_uri = body.gcs_uris[0]
    try:
        session_id, result, manager = initialize_session(
            gcs_uri=gcs_uri,
            model_type=model_type,
        )
    except Exception as e:
        logger.error("Session creation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session initialization failed. Please try again.",
        ) from None

    elapsed_ms = int((time.monotonic() - start) * 1000)
    registry[session_id] = (manager, time.time())

    background_tasks.add_task(warmup_session, session_id)  # background_tasks always present here

    return SessionResponse(
        session_id=session_id,
        status="ready" if result.success else "error",
        documents_processed=getattr(result, "total_pages_processed", 0),
        processing_time_ms=elapsed_ms,
        metadata=body.metadata,
    )


@router.post("/upload", status_code=status.HTTP_201_CREATED, response_model=SessionResponse)
async def create_session_upload(
    files: list[UploadFile] = File(...),  # noqa: B008
    ocr_engine: str = Form(default="mistral"),
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]  # noqa: B008
):
    """Initialize a new session by uploading PDF files directly.

    Files are uploaded to GCS, then processed through the standard pipeline.
    """
    import tempfile
    import uuid as _uuid

    registry = get_session_registry()

    model_type = _OCR_ENGINE_MAP.get(ocr_engine)
    if model_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown OCR engine: {ocr_engine}. Use: {list(_OCR_ENGINE_MAP.keys())}",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one PDF file is required",
        )

    start = time.monotonic()

    # Save first file to temp, upload to GCS, then process
    upload_file = files[0]
    if not upload_file.filename or not upload_file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    upload_id = _uuid.uuid4().hex[:8]
    temp_dir = tempfile.mkdtemp()
    safe_filename = upload_file.filename.replace(" ", "_")
    local_path = os.path.join(temp_dir, safe_filename)

    try:
        content = await upload_file.read()
        with open(local_path, "wb") as f:
            f.write(content)

        gcs_path = f"uploads/{upload_id}/{safe_filename}"
        gcs_uri = upload_to_gcs(local_path, gcs_path)
    finally:
        if os.path.exists(local_path):
            os.unlink(local_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    try:
        session_id, result, manager = initialize_session(
            gcs_uri=gcs_uri,
            model_type=model_type,
        )
    except Exception as e:
        logger.error("Session upload init failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session initialization failed. Please try again.",
        ) from None

    elapsed_ms = int((time.monotonic() - start) * 1000)
    registry[session_id] = (manager, time.time())

    if background_tasks is not None:
        background_tasks.add_task(warmup_session, session_id)

    return SessionResponse(
        session_id=session_id,
        status="ready" if result.success else "error",
        documents_processed=getattr(result, "total_pages_processed", 0),
        processing_time_ms=elapsed_ms,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session status and metadata."""
    manager = get_session_manager(session_id)
    pages = 0
    if manager.result:
        pages = getattr(manager.result, "total_pages_processed", 0)

    return SessionResponse(
        session_id=session_id,
        status="ready",
        documents_processed=pages,
    )


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def delete_session_endpoint(session_id: str):
    """Delete a session and all its artifacts."""
    get_session_manager(session_id)

    try:
        result = delete_session(session_id)
    except Exception as e:
        logger.error("Session deletion failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session deletion failed. Please try again.",
        ) from None
    registry = get_session_registry()
    registry.pop(session_id, None)

    return SessionDeleteResponse(
        session_id=session_id,
        status="deleted",
        vectors_deleted=result.vectors_deleted,
        errors=result.errors,
    )
