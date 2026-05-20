"""Bootstrap environment variables from GCP Secret Manager.

Called once at startup. For each secret name in the list, if the
corresponding env var is not already set, fetches the latest version
from Secret Manager and injects it into os.environ.

No-ops when GCP_PROJECT_ID is unset (local dev with .env) or when
Secret Manager is unreachable (logs a warning, does not crash).
"""

from __future__ import annotations

import os

from src.core.logger import get_logger

logger = get_logger(__name__)

SECRETS = [
    "CLOUDSQL_HOST",
    "CLOUDSQL_PORT",
    "CLOUDSQL_DATABASE",
    "CLOUDSQL_DATABASE_SCHEMA",
    "CLOUDSQL_USER",
    "CLOUDSQL_PASSWORD",
    "MONGODB_URI",
    "MONGODB_DATABASE",
    "API_AUTH_TOKEN",
    "CARECONNECT_WIDGET_TOKEN",
]


def load_secrets_from_gcp() -> int:
    """Fetch missing env vars from GCP Secret Manager. Returns count loaded."""
    project = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        logger.debug("GCP_PROJECT_ID not set — skipping Secret Manager bootstrap")
        return 0

    if os.environ.get("STORAGE_EMULATOR_HOST"):
        logger.debug("GCS emulator detected — skipping Secret Manager bootstrap")
        return 0

    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
    except Exception as e:
        logger.warning("Could not create Secret Manager client: %s", e)
        return 0

    loaded = 0
    for name in SECRETS:
        if os.environ.get(name):
            continue
        try:
            resource = f"projects/{project}/secrets/{name}/versions/latest"
            response = client.access_secret_version(request={"name": resource})
            value = response.payload.data.decode("UTF-8")
            os.environ[name] = value
            loaded += 1
            logger.info("Loaded %s from Secret Manager", name)
        except Exception as e:
            logger.debug("Secret %s not available in Secret Manager: %s", name, e)

    if loaded:
        logger.info("Loaded %d secrets from GCP Secret Manager", loaded)
    return loaded
