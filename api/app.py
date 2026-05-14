"""FastAPI application factory for the ADR AI Agent API."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.api.dependencies import cleanup_expired_sessions
from src.api.middleware.observability import ObservabilityMiddleware
from src.api.routes.chat import router as chat_router
from src.api.routes.dev import router as dev_router
from src.api.routes.feedback import router as feedback_router
from src.api.routes.health import router as health_router
from src.api.routes.history import router as history_router
from src.api.routes.policies import router as policies_router
from src.api.routes.query import router as query_router
from src.api.routes.query_stream import router as query_stream_router
from src.api.routes.session_stream import router as session_stream_router
from src.api.routes.sessions import router as sessions_router
from src.api.routes.widget import router as widget_router
from src.api.routes.widget import ui_router as widget_ui_router
from src.core.logger import get_logger
from src.core.secrets import load_secrets_from_gcp

logger = get_logger(__name__)

load_secrets_from_gcp()


def _ensure_gcs_bucket(bucket_name: str) -> None:
    """Create a GCS bucket if it doesn't exist (for fake-gcs-server in dev)."""
    try:
        from core.gcs_client import get_gcs_client
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            client.create_bucket(bucket_name)
            logger.info("Created GCS bucket: %s", bucket_name)
    except Exception as e:
        logger.warning("Could not ensure GCS bucket %s: %s", bucket_name, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan — runs session cleanup task."""

    async def _cleanup_loop():
        # Wrap the body so a transient error (e.g. dict-size-change during
        # concurrent session creation) doesn't kill the task — if the
        # coroutine dies, cleanup is disabled for the rest of the process.
        while True:
            try:
                await asyncio.sleep(3600)
                count = cleanup_expired_sessions()
                if count:
                    logger.info("Cleaned up %d expired sessions", count)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Session cleanup iteration failed: %s", e, exc_info=True)

    seed_bucket = os.getenv("DEV_GCS_SEED_BUCKET")
    if seed_bucket:
        _ensure_gcs_bucket(seed_bucket)

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ADR AI Agent API",
        description="REST API for the CareConnect ADR AI Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — dev-only. Production CORS handled at the gateway layer.
    if os.getenv("ENABLE_DEV_ROUTES", "").lower() in ("1", "true", "yes"):
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:8080", "http://localhost:8000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(ObservabilityMiddleware)

    # Health routes (no auth required)
    app.include_router(health_router)

    # Sessions routes (auth required)
    app.include_router(sessions_router)
    app.include_router(session_stream_router)

    app.include_router(query_router)
    app.include_router(query_stream_router)
    app.include_router(history_router)
    app.include_router(feedback_router)
    app.include_router(widget_router)
    app.include_router(widget_ui_router)
    app.include_router(policies_router)

    # Static files for chat UI
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(chat_router)

    # Dev router (/dev/test test harness + debug endpoints) is no-auth and must
    # never be exposed in production. Require an explicit opt-in env var.
    if os.getenv("ENABLE_DEV_ROUTES", "").lower() in ("1", "true", "yes"):
        app.include_router(dev_router)
        golden_dir = Path(__file__).resolve().parent.parent.parent / "data" / "golden"
        if golden_dir.exists():
            app.mount("/dev/golden", StaticFiles(directory=str(golden_dir)), name="golden")

    return app
