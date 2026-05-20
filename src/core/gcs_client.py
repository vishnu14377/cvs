"""
Shared Google Cloud Storage client (lazy singleton).

Authentication (Application Default Credentials):
  - Local: ``gcloud auth application-default login``
  - CI / servers: set ``GOOGLE_APPLICATION_CREDENTIALS`` to a service account JSON path.

The client is created once per process and reused for all bucket/blob operations.
All upload/download functions automatically use the configured project bucket.
"""

import json
import os
import tempfile
from datetime import timedelta
from threading import Lock

from google.cloud import storage  # type: ignore[attr-defined]

from src.core.logger import get_logger

logger = get_logger(__name__)

_lock = Lock()
_client: storage.Client | None = None
_bucket_name: str | None = None
_base_folder: str | None = None


def _get_bucket_name() -> str:
    """Get the configured bucket name."""
    global _bucket_name
    if _bucket_name is None:
        from src.core.config import GlobalConfig

        _bucket_name = GlobalConfig.GCS_BUCKET_NAME
        if not _bucket_name:
            raise ValueError("GCS_BUCKET_NAME must be configured")
    return _bucket_name


def _get_base_folder() -> str:
    """Get the configured GCS working folder (base folder for all operations)."""
    global _base_folder
    if _base_folder is None:
        from src.core.config import GlobalConfig

        base = (GlobalConfig.GCS_WORKING_FOLDER or "").strip().strip("/")
        if not base:
            raise ValueError("GCS_WORKING_FOLDER must be configured")
        _base_folder = base
    return _base_folder


def _build_full_gcs_path(relative_path: str) -> str:
    """Build full GCS object path by prepending the base folder.

    Args:
        relative_path: Path relative to the base folder (e.g. ``session-id/tmp/file.pdf``)

    Returns:
        Full path including base folder (e.g. ``base-folder/session-id/tmp/file.pdf``)
    """
    base = _get_base_folder()
    # Clean up relative path
    relative_path = relative_path.strip().strip("/")
    return f"{base}/{relative_path}"


def get_gcs_client() -> storage.Client:
    """Return the process-wide storage.Client, creating it on first use."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            from src.core.config import GlobalConfig

            if GlobalConfig.GCP_PROJECT:
                _client = storage.Client(project=GlobalConfig.GCP_PROJECT)
            else:
                _client = storage.Client()
        return _client


def reset_gcs_client_for_testing() -> None:
    """Clear cached client (for tests only)."""
    global _client, _bucket_name, _base_folder
    with _lock:
        _client = None
        _bucket_name = None
        _base_folder = None


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a GCS URI into bucket name and object path.

    Args:
        gcs_uri: GCS URI (e.g. ``gs://bucket-name/path/to/file.pdf``)

    Returns:
        Tuple of (bucket_name, object_path)

    Raises:
        ValueError: If URI is not a valid GCS URI
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}. Must start with 'gs://'")

    # Remove gs:// prefix
    path = gcs_uri[5:]

    # Split into bucket and object path
    parts = path.split("/", 1)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}. Must be gs://bucket/path")

    return parts[0], parts[1]


def is_gcs_uri(path: str) -> bool:
    """Check if a path is a GCS URI.

    Args:
        path: Path to check

    Returns:
        True if path starts with 'gs://'
    """
    return path.startswith("gs://")


def download_from_gcs(
    gcs_uri: str,
    local_dir: str | None = None,
) -> str:
    """Download a file from Google Cloud Storage.

    Args:
        gcs_uri: GCS URI (e.g. ``gs://bucket-name/path/to/file.pdf``)
        local_dir: Local directory to download to. If None, uses temp directory.

    Returns:
        Path to the downloaded local file

    Raises:
        ValueError: If GCS URI is invalid
        Exception: If download fails
    """
    bucket_name, object_path = parse_gcs_uri(gcs_uri)

    # Get filename from object path
    filename = os.path.basename(object_path)

    # Determine local path
    if local_dir is None:
        local_dir = tempfile.mkdtemp()

    local_path = os.path.join(local_dir, filename)

    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)

        logger.info(f"Downloading {gcs_uri} to {local_path}")
        blob.download_to_filename(local_path)

        logger.info(f"Successfully downloaded to {local_path}")
        return local_path

    except Exception as e:
        logger.error(f"Failed to download {gcs_uri}: {str(e)}")
        raise


def upload_to_gcs(
    local_path: str,
    gcs_path: str,
) -> str:
    """Upload a file to Google Cloud Storage using the configured bucket and base folder.

    The base folder (GCS_WORKING_FOLDER) is automatically prepended to the path.

    Args:
        local_path: Path to the local file to upload
        gcs_path: Destination path relative to base folder (e.g. ``session-id/tmp/file.pdf``)

    Returns:
        Full GCS URI (e.g. ``gs://bucket-name/base-folder/session-id/tmp/file.pdf``)

    Raises:
        FileNotFoundError: If local file doesn't exist
        ValueError: If bucket or base folder is not configured
        Exception: If upload fails
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    bucket_name = _get_bucket_name()
    full_gcs_path = _build_full_gcs_path(gcs_path)

    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(full_gcs_path)

        # Upload file
        logger.info(
            f"Uploading {os.path.basename(local_path)} to gs://{bucket_name}/{full_gcs_path}"
        )
        blob.upload_from_filename(local_path)

        # Return full GCS URI
        gcs_uri = f"gs://{bucket_name}/{full_gcs_path}"
        logger.info(f"Successfully uploaded to {gcs_uri}")
        return gcs_uri

    except Exception as e:
        logger.error(f"Failed to upload {local_path} to GCS: {str(e)}")
        raise


def upload_json_to_gcs(
    data: dict,
    gcs_path: str,
) -> str:
    """Upload a JSON object directly to Google Cloud Storage using the configured bucket and base folder.

    The base folder (GCS_WORKING_FOLDER) is automatically prepended to the path.

    Args:
        data: Dictionary to upload as JSON
        gcs_path: Destination path relative to base folder (e.g. ``session-id/extracted_text/file.json``)

    Returns:
        Full GCS URI (e.g. ``gs://bucket-name/base-folder/session-id/extracted_text/file.json``)

    Raises:
        ValueError: If bucket or base folder is not configured
        Exception: If upload fails
    """
    bucket_name = _get_bucket_name()
    full_gcs_path = _build_full_gcs_path(gcs_path)

    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(full_gcs_path)

        # Convert dict to JSON string
        json_content = json.dumps(data, indent=2, ensure_ascii=False)

        # Upload JSON content
        logger.info(f"Uploading JSON to gs://{bucket_name}/{full_gcs_path}")
        blob.upload_from_string(json_content, content_type="application/json")

        # Return full GCS URI
        gcs_uri = f"gs://{bucket_name}/{full_gcs_path}"
        logger.info(f"Successfully uploaded JSON to {gcs_uri}")
        return gcs_uri

    except Exception as e:
        logger.error(f"Failed to upload JSON to GCS: {str(e)}")
        raise


def list_files_in_gcs_folder(
    folder_path: str,
    file_extension: str | None = None,
) -> list[str]:
    """List all files in a GCS folder using the configured bucket and base folder.

    The base folder (GCS_WORKING_FOLDER) is automatically prepended to the path.

    Args:
        folder_path: Folder path relative to base folder (e.g. ``session-id/tmp``)
        file_extension: Optional file extension filter (e.g. ``.pdf``).
                        Include the dot in the extension.

    Returns:
        List of full GCS URIs for files in the folder (e.g. ``gs://bucket-name/base-folder/session-id/tmp/file.pdf``)

    Raises:
        ValueError: If bucket or base folder is not configured
        Exception: If listing fails
    """
    bucket_name = _get_bucket_name()
    full_folder_path = _build_full_gcs_path(folder_path)

    # Ensure the folder path ends with / for proper prefix matching
    if not full_folder_path.endswith("/"):
        full_folder_path = f"{full_folder_path}/"

    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)

        logger.info(f"Listing files in gs://{bucket_name}/{full_folder_path}")

        blobs = bucket.list_blobs(prefix=full_folder_path)

        files: list[str] = []
        for blob in blobs:
            # Skip "directory" markers (blobs that end with /)
            if blob.name.endswith("/"):
                continue

            # Apply file extension filter if specified
            if file_extension and not blob.name.lower().endswith(file_extension.lower()):
                continue

            gcs_uri = f"gs://{bucket_name}/{blob.name}"
            files.append(gcs_uri)

        logger.info(f"Found {len(files)} files in folder")
        return files

    except Exception as e:
        logger.error(f"Failed to list files in GCS folder: {str(e)}")
        raise


def download_folder_files(
    folder_uri: str,
    local_dir: str,
    file_extension: str | None = None,
    max_workers: int = 5,
) -> list[str]:
    """Download all files from a GCS folder in parallel.

    Lists all files in the folder and downloads them concurrently using
    the existing download_from_gcs function.

    Args:
        folder_uri: GCS URI of the folder (e.g. ``gs://bucket/path/to/folder/``)
        local_dir: Local directory to download files to
        file_extension: Optional file extension filter (e.g. ``.pdf``)
        max_workers: Maximum number of parallel download workers

    Returns:
        List of local file paths for downloaded files

    Raises:
        ValueError: If folder_uri is not a valid GCS URI
        Exception: If listing or download fails
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Parse the folder URI to get bucket and path
    bucket_name, folder_path = parse_gcs_uri(folder_uri.rstrip("/") + "/dummy")
    folder_path = os.path.dirname(folder_path)

    # Ensure folder path ends with /
    if folder_path and not folder_path.endswith("/"):
        folder_path = f"{folder_path}/"

    # List all files in the folder
    logger.info(f"Listing files in gs://{bucket_name}/{folder_path}")

    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=folder_path)

        gcs_uris: list[str] = []
        for blob in blobs:
            # Skip directory markers
            if blob.name.endswith("/"):
                continue

            # Apply file extension filter
            if file_extension and not blob.name.lower().endswith(file_extension.lower()):
                continue

            gcs_uris.append(f"gs://{bucket_name}/{blob.name}")

        logger.info(f"Found {len(gcs_uris)} files to download")

    except Exception as e:
        logger.error(f"Failed to list files in GCS folder: {str(e)}")
        raise

    if not gcs_uris:
        logger.info("No files found in folder")
        return []

    # Download all files in parallel
    logger.info(
        "Downloading %d files in parallel with %d workers to %s",
        len(gcs_uris),
        max_workers,
        local_dir,
    )

    local_paths: list[str] = []
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks using existing download_from_gcs function
        future_to_uri = {
            executor.submit(download_from_gcs, uri, local_dir): uri for uri in gcs_uris
        }

        # Collect results as they complete
        for future in as_completed(future_to_uri):
            uri = future_to_uri[future]
            try:
                local_path = future.result()
                local_paths.append(local_path)
                logger.debug("Downloaded: %s -> %s", uri, local_path)
            except Exception as e:
                logger.error("Failed to download %s: %s", uri, e)
                errors.append((uri, str(e)))

    logger.info(
        "Parallel download complete: %d successful, %d failed",
        len(local_paths),
        len(errors),
    )

    if errors:
        error_msg = "; ".join([f"{uri}: {err}" for uri, err in errors])
        raise Exception(f"Failed to download {len(errors)} file(s): {error_msg}")

    return local_paths


def json_uri_to_pdf_uri(gcs_uri: str) -> str:
    """Derive the split-PDF GCS URI from the extracted-text JSON URI.

    The OCR pipeline stores split PDFs under ``<session>/tmp/`` and extracted
    JSON under ``<session>/extracted_text/`` with matching basenames. This
    function reverses the mapping so source citations can link to the PDF.

    Returns the original URI unchanged if it doesn't match the expected pattern.
    """
    if not is_gcs_uri(gcs_uri):
        return gcs_uri
    if "/extracted_text/" not in gcs_uri or not gcs_uri.endswith(".json"):
        return gcs_uri
    return gcs_uri.replace("/extracted_text/", "/tmp/").removesuffix(".json") + ".pdf"


def _gcs_emulator_host() -> str | None:
    """Return the external-facing GCS emulator base URL if one is configured.

    GCS_EXTERNAL_URL overrides the docker-internal STORAGE_EMULATOR_HOST so
    that URLs emitted in HTML responses resolve from the user's browser
    (e.g. http://localhost:4443 instead of http://gcs:4443).
    """
    return (
        os.environ.get("GCS_EXTERNAL_URL")
        or os.environ.get("STORAGE_EMULATOR_HOST")
        or os.environ.get("GCS_EMULATOR_HOST")
    )


def generate_signed_url(
    gcs_uri: str,
    expiration: timedelta | None = None,
) -> str | None:
    """Generate an HTTPS URL for a GCS object.

    When a GCS emulator is configured (STORAGE_EMULATOR_HOST), returns a
    direct emulator download URL instead of attempting to sign (the emulator
    uses anonymous credentials that cannot sign).

    For real GCS, attempts ``blob.generate_signed_url`` which works with
    service-account credentials or ADC-backed IAM signing.

    Returns None (instead of raising) when URL generation fails — callers
    should fall back to displaying the source as plain text.
    """
    bucket_name, object_path = parse_gcs_uri(gcs_uri)

    emulator = _gcs_emulator_host()
    if emulator:
        base = emulator.rstrip("/")
        return f"{base}/storage/v1/b/{bucket_name}/o/{object_path}?alt=media"

    if expiration is None:
        expiration = timedelta(hours=1)
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_path)
        return blob.generate_signed_url(expiration=expiration, method="GET")
    except Exception as e:
        logger.warning("Failed to sign %s: %s", gcs_uri, e)
        return None


def delete_session_folder(session_id: str) -> int:
    """Delete all GCS blobs under ``<GCS_WORKING_FOLDER>/<session_id>/``.

    This removes every object stored under the session prefix — typically
    the ``tmp/`` (split PDFs) and ``extracted_text/`` (OCR JSON output)
    sub-folders, plus any other blobs that may have been written.

    Uses the configured bucket (``GCS_BUCKET_NAME``) and base folder
    (``GCS_WORKING_FOLDER``).

    Args:
        session_id: The session whose GCS folder should be deleted.

    Returns:
        Number of blobs deleted.  Returns ``0`` if no blobs were found.

    Raises:
        ValueError: If ``session_id`` is empty, or bucket / base folder
            are not configured.
        Exception: If the GCS list or delete operation fails.

    Example:
        >>> from src.core.gcs_client import delete_session_folder
        >>> deleted = delete_session_folder("adr-20260414-a1b2c3d4")
        >>> print(f"Removed {deleted} blob(s)")
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    session_id = session_id.strip()

    # Build the full prefix: <base_folder>/<session_id>/
    prefix = _build_full_gcs_path(session_id)
    if not prefix.endswith("/"):
        prefix = f"{prefix}/"

    bucket_name = _get_bucket_name()
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        logger.debug(
            "No GCS blobs found under gs://%s/%s",
            bucket_name,
            prefix,
        )
        return 0

    # delete_blobs handles batching internally
    bucket.delete_blobs(blobs)

    logger.info(
        "Deleted %d GCS blob(s) under gs://%s/%s",
        len(blobs),
        bucket_name,
        prefix,
    )
    return len(blobs)
