"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness check — is the API process running?"""
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready():
    """Readiness check — are dependencies accessible?"""
    deps = {}

    # Postgres check
    try:
        from src.core.cloudsql_pg_client import get_cloudsql_client

        client = get_cloudsql_client()
        engine = client.engine
        if engine is None:
            raise RuntimeError("Engine not initialized")
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        deps["postgres"] = "connected"
    except Exception:
        deps["postgres"] = "unreachable"

    # GCS check
    try:
        from src.core.gcs_client import get_gcs_client

        gcs = get_gcs_client()
        deps["gcs"] = "accessible" if gcs else "not_configured"
    except Exception as e:
        if "credentials" in str(e).lower():
            deps["gcs"] = "no_credentials"
        else:
            deps["gcs"] = "unreachable"

    # MongoDB check
    try:
        from src.feedback_manager.client import get_feedback_db

        db = get_feedback_db()
        deps["mongodb"] = "connected" if db is not None else "not_configured"
    except Exception:
        deps["mongodb"] = "unreachable"

    # Vertex AI check
    try:
        from src.core.config import llm_config

        if not llm_config.GCP_PROJECT:
            deps["vertex_ai"] = "not_configured"
        else:
            import google.auth
            import google.auth.transport.requests

            credentials, project = google.auth.default()
            credentials.refresh(google.auth.transport.requests.Request())
            deps["vertex_ai"] = "authenticated"
    except Exception as e:
        if "credentials" in str(e).lower():
            deps["vertex_ai"] = "no_credentials"
        else:
            deps["vertex_ai"] = "unreachable"

    all_ok = all(
        v in ("connected", "accessible", "authenticated", "not_configured", "no_credentials")
        for v in deps.values()
    )
    return {
        "status": "ready" if all_ok else "degraded",
        "dependencies": deps,
    }
