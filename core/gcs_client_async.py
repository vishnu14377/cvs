"""Async wrappers for GCS client operations.

Thin async layer over the sync gcs_client functions using asyncio.to_thread().
download_folder_files uses asyncio.gather() for concurrent downloads.
"""

from __future__ import annotations

import asyncio

from core.config import GlobalConfig
from core.gcs_client import (
    download_from_gcs as _sync_download_from_gcs,
)
from core.gcs_client import (
    is_gcs_uri,
    parse_gcs_uri,
)
from core.gcs_client import (
    list_files_in_gcs_folder as _sync_list_files_in_gcs_folder,
)
from core.gcs_client import (
    upload_json_to_gcs as _sync_upload_json_to_gcs,
)
from core.gcs_client import (
    upload_to_gcs as _sync_upload_to_gcs,
)
from core.logger import get_logger

logger = get_logger(__name__)


async def download_from_gcs(gcs_uri: str, local_dir: str | None = None) -> str:
    return await asyncio.to_thread(_sync_download_from_gcs, gcs_uri, local_dir)


async def upload_to_gcs(local_path: str, gcs_path: str) -> str:
    return await asyncio.to_thread(_sync_upload_to_gcs, local_path, gcs_path)


async def upload_json_to_gcs(data: dict, gcs_path: str) -> str:
    return await asyncio.to_thread(_sync_upload_json_to_gcs, data, gcs_path)


def _to_relative_folder_path(folder: str) -> str:
    """Normalize a folder argument to the relative path the sync API expects.

    Accepts either a `gs://bucket/base-folder/session-id/tmp` URI or a relative
    `session-id/tmp` path. The sync `list_files_in_gcs_folder` prepends the
    configured base folder via `_build_full_gcs_path`, so passing a full URI
    produces a broken prefix like `base-folder/gs:/bucket/...` that matches
    nothing. Strip the `gs://bucket/` prefix and the leading base-folder if
    present.
    """
    if not is_gcs_uri(folder):
        return folder.strip("/")
    _, object_path = parse_gcs_uri(folder)
    base = (GlobalConfig.GCS_WORKING_FOLDER or "").strip().strip("/")
    object_path = object_path.strip("/")
    if base and (object_path == base or object_path.startswith(base + "/")):
        object_path = object_path[len(base) :].lstrip("/")
    return object_path


async def list_files_in_gcs_folder(folder_uri: str, extension: str | None = None) -> list[str]:
    relative = _to_relative_folder_path(folder_uri)
    return await asyncio.to_thread(_sync_list_files_in_gcs_folder, relative, extension)


async def download_folder_files(
    folder_uri: str,
    local_dir: str,
    extension: str | None = None,
    max_concurrent: int = 10,
) -> list[str]:
    """Download all files from a GCS folder concurrently using asyncio.gather."""
    file_uris = await list_files_in_gcs_folder(folder_uri, extension)
    if not file_uris:
        return []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _download_one(uri: str) -> str:
        async with semaphore:
            return await download_from_gcs(uri, local_dir)

    results = await asyncio.gather(
        *[_download_one(uri) for uri in file_uris],
        return_exceptions=True,
    )

    downloaded: list[str] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error("Failed to download %s: %s", file_uris[i], result)
        else:
            downloaded.append(result)
    return downloaded
