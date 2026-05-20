"""Tests for async GCS client wrappers.

Covers the live `asyncio.to_thread`-based API on src/core/gcs_client_async.py:
- download_from_gcs
- upload_to_gcs
- upload_json_to_gcs
- list_files_in_gcs_folder (including gs:// URI normalization)
- download_folder_files (concurrent gather + partial failure tolerance)

The module additionally exposes legacy ``*_async`` variants that reference
symbols not imported into the module (a merge artifact). Those are not
covered here because nothing in src/ calls them, and this suite's scope is
the live API only.
"""

from unittest.mock import patch

import pytest

from src.core.gcs_client_async import (
    download_folder_files,
    download_from_gcs,
    list_files_in_gcs_folder,
    upload_json_to_gcs,
    upload_to_gcs,
)

_MOD = "core.gcs_client_async"


class TestDownloadFromGcs:
    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_download_from_gcs")
    async def test_delegates_to_sync(self, mock_sync):
        mock_sync.return_value = "/tmp/file.pdf"
        result = await download_from_gcs("gs://bucket/file.pdf", "/tmp")
        mock_sync.assert_called_once_with("gs://bucket/file.pdf", "/tmp")
        assert result == "/tmp/file.pdf"

    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_download_from_gcs")
    async def test_default_local_dir_is_none(self, mock_sync):
        mock_sync.return_value = "/some/path.pdf"
        await download_from_gcs("gs://bucket/file.pdf")
        mock_sync.assert_called_once_with("gs://bucket/file.pdf", None)


class TestUploadToGcs:
    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_upload_to_gcs")
    async def test_delegates_to_sync(self, mock_sync):
        mock_sync.return_value = "gs://bucket/base/path.pdf"
        result = await upload_to_gcs("/tmp/file.pdf", "path.pdf")
        mock_sync.assert_called_once_with("/tmp/file.pdf", "path.pdf")
        assert result == "gs://bucket/base/path.pdf"


class TestUploadJsonToGcs:
    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_upload_json_to_gcs")
    async def test_delegates_to_sync(self, mock_sync):
        mock_sync.return_value = "gs://bucket/base/data.json"
        result = await upload_json_to_gcs({"key": "val"}, "data.json")
        mock_sync.assert_called_once_with({"key": "val"}, "data.json")
        assert result == "gs://bucket/base/data.json"


class TestListFilesInGcsFolder:
    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_list_files_in_gcs_folder")
    async def test_relative_path_passes_through(self, mock_sync):
        """Relative folder paths are passed to the sync API unchanged."""
        mock_sync.return_value = ["gs://b/f1.pdf", "gs://b/f2.pdf"]
        result = await list_files_in_gcs_folder("session-id/tmp", ".pdf")
        mock_sync.assert_called_once_with("session-id/tmp", ".pdf")
        assert result == ["gs://b/f1.pdf", "gs://b/f2.pdf"]

    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_list_files_in_gcs_folder")
    async def test_gs_uri_is_stripped_before_delegate(self, mock_sync):
        """The sync list_files_in_gcs_folder prepends GCS_WORKING_FOLDER; passing
        a gs:// URI unchanged would produce a broken prefix."""
        mock_sync.return_value = []
        with patch(f"{_MOD}.GlobalConfig") as mock_config:
            mock_config.GCS_WORKING_FOLDER = "base-folder"
            await list_files_in_gcs_folder("gs://my-bucket/base-folder/session-id/tmp", ".pdf")
        mock_sync.assert_called_once_with("session-id/tmp", ".pdf")

    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_list_files_in_gcs_folder")
    async def test_leading_and_trailing_slashes_stripped(self, mock_sync):
        mock_sync.return_value = []
        await list_files_in_gcs_folder("/session-id/tmp/", None)
        mock_sync.assert_called_once_with("session-id/tmp", None)


class TestDownloadFolderFiles:
    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_download_from_gcs")
    @patch(f"{_MOD}._sync_list_files_in_gcs_folder")
    async def test_downloads_all_files(self, mock_list, mock_download):
        mock_list.return_value = [
            "gs://b/f1.pdf",
            "gs://b/f2.pdf",
            "gs://b/f3.pdf",
        ]
        mock_download.side_effect = ["/tmp/f1.pdf", "/tmp/f2.pdf", "/tmp/f3.pdf"]
        result = await download_folder_files("gs://b/folder/", "/tmp", ".pdf")
        assert len(result) == 3
        assert mock_download.call_count == 3

    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_list_files_in_gcs_folder")
    async def test_empty_folder(self, mock_list):
        mock_list.return_value = []
        result = await download_folder_files("gs://b/empty/", "/tmp")
        assert result == []

    @pytest.mark.asyncio
    @patch(f"{_MOD}._sync_download_from_gcs")
    @patch(f"{_MOD}._sync_list_files_in_gcs_folder")
    async def test_partial_failure_is_tolerated(self, mock_list, mock_download):
        """One download fails — the successful ones are still returned."""
        mock_list.return_value = ["gs://b/f1.pdf", "gs://b/f2.pdf"]
        mock_download.side_effect = ["/tmp/f1.pdf", RuntimeError("download failed")]
        result = await download_folder_files("gs://b/folder/", "/tmp")
        assert result == ["/tmp/f1.pdf"]
