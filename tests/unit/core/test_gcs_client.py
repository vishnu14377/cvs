"""Tests for shared GCS client singleton and utility functions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.gcs_client import (
    _build_full_gcs_path,
    delete_session_folder,
    download_from_gcs,
    generate_signed_url,
    get_gcs_client,
    is_gcs_uri,
    json_uri_to_pdf_uri,
    parse_gcs_uri,
    reset_gcs_client_for_testing,
    upload_json_to_gcs,
    upload_to_gcs,
)


@pytest.fixture(autouse=True)
def clear_gcs_client():
    reset_gcs_client_for_testing()
    yield
    reset_gcs_client_for_testing()


class TestGetGcsClient:
    """Tests for GCS client singleton."""

    def test_get_gcs_client_returns_same_instance(self):
        mock_client = MagicMock()
        with patch("core.gcs_client.storage.Client", return_value=mock_client) as ctor:
            first = get_gcs_client()
            second = get_gcs_client()
        assert first is second is mock_client
        ctor.assert_called_once()


class TestIsGcsUri:
    """Tests for is_gcs_uri function."""

    def test_valid_gcs_uri(self):
        assert is_gcs_uri("gs://bucket/path/to/file.pdf") is True

    def test_gcs_uri_root(self):
        assert is_gcs_uri("gs://bucket/file.pdf") is True

    def test_local_path_not_gcs_uri(self):
        assert is_gcs_uri("/local/path/file.pdf") is False

    def test_http_url_not_gcs_uri(self):
        assert is_gcs_uri("https://example.com/file.pdf") is False

    def test_empty_string(self):
        assert is_gcs_uri("") is False


class TestParseGcsUri:
    """Tests for parse_gcs_uri function."""

    def test_valid_uri(self):
        bucket, path = parse_gcs_uri("gs://my-bucket/path/to/file.pdf")
        assert bucket == "my-bucket"
        assert path == "path/to/file.pdf"

    def test_uri_with_single_level_path(self):
        bucket, path = parse_gcs_uri("gs://bucket/file.pdf")
        assert bucket == "bucket"
        assert path == "file.pdf"

    def test_uri_with_deep_path(self):
        bucket, path = parse_gcs_uri("gs://bucket/a/b/c/d/file.pdf")
        assert bucket == "bucket"
        assert path == "a/b/c/d/file.pdf"

    def test_invalid_uri_no_gs_prefix(self):
        with pytest.raises(ValueError, match="Must start with 'gs://'"):
            parse_gcs_uri("https://bucket/path")

    def test_invalid_uri_no_path(self):
        with pytest.raises(ValueError, match="Must be gs://bucket/path"):
            parse_gcs_uri("gs://bucket")

    def test_invalid_uri_empty_bucket(self):
        with pytest.raises(ValueError, match="Must be gs://bucket/path"):
            parse_gcs_uri("gs:///path")


class TestBuildFullGcsPath:
    """Tests for _build_full_gcs_path function."""

    @patch("core.gcs_client._get_base_folder", return_value="base-folder")
    def test_builds_path_with_base_folder(self, mock_base):
        result = _build_full_gcs_path("session/tmp/file.pdf")
        assert result == "base-folder/session/tmp/file.pdf"

    @patch("core.gcs_client._get_base_folder", return_value="base-folder")
    def test_strips_leading_slashes(self, mock_base):
        result = _build_full_gcs_path("/session/tmp/file.pdf")
        assert result == "base-folder/session/tmp/file.pdf"

    @patch("core.gcs_client._get_base_folder", return_value="base-folder")
    def test_strips_trailing_slashes(self, mock_base):
        result = _build_full_gcs_path("session/tmp/file.pdf/")
        assert result == "base-folder/session/tmp/file.pdf"


class TestUploadToGcs:
    """Tests for upload_to_gcs function."""

    def test_upload_success(self):
        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            mock_blob = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.blob.return_value = mock_blob
            mock_client = MagicMock()
            mock_client.bucket.return_value = mock_bucket

            with (
                patch("core.gcs_client.get_gcs_client", return_value=mock_client),
                patch("core.gcs_client._get_bucket_name", return_value="test-bucket"),
                patch(
                    "core.gcs_client._build_full_gcs_path", return_value="base/session/tmp/file.pdf"
                ),
            ):
                result = upload_to_gcs(temp_path, "session/tmp/file.pdf")

            assert result == "gs://test-bucket/base/session/tmp/file.pdf"
            mock_blob.upload_from_filename.assert_called_once_with(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_upload_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Local file not found"):
            upload_to_gcs("/nonexistent/file.pdf", "session/tmp/file.pdf")


class TestUploadJsonToGcs:
    """Tests for upload_json_to_gcs function."""

    def test_upload_json_success(self):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        test_data = {"key": "value", "number": 42}

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            patch("core.gcs_client._get_bucket_name", return_value="test-bucket"),
            patch("core.gcs_client._build_full_gcs_path", return_value="base/session/data.json"),
        ):
            result = upload_json_to_gcs(test_data, "session/data.json")

        assert result == "gs://test-bucket/base/session/data.json"
        mock_blob.upload_from_string.assert_called_once()
        # Verify JSON content
        call_args = mock_blob.upload_from_string.call_args
        uploaded_content = call_args[0][0]
        assert json.loads(uploaded_content) == test_data


class TestDownloadFromGcs:
    """Tests for download_from_gcs function."""

    def test_download_success(self):
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            result = download_from_gcs("gs://bucket/path/to/file.pdf", temp_dir)

        assert result.endswith("file.pdf")
        mock_client.bucket.assert_called_once_with("bucket")
        mock_bucket.blob.assert_called_once_with("path/to/file.pdf")
        mock_blob.download_to_filename.assert_called_once()

    def test_download_invalid_uri(self):
        with pytest.raises(ValueError, match="Must start with 'gs://'"):
            download_from_gcs("https://bucket/file.pdf")


class TestDeleteSessionFolder:
    """Tests for delete_session_folder function."""

    def test_deletes_all_blobs_under_prefix(self):
        """Should list and delete all blobs under the session prefix."""
        mock_blob1 = MagicMock()
        mock_blob2 = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            patch("core.gcs_client._get_bucket_name", return_value="test-bucket"),
            patch("core.gcs_client._build_full_gcs_path", return_value="base/adr-123"),
        ):
            result = delete_session_folder("adr-123")

        assert result == 2
        mock_bucket.list_blobs.assert_called_once_with(prefix="base/adr-123/")
        mock_bucket.delete_blobs.assert_called_once_with([mock_blob1, mock_blob2])

    def test_returns_zero_when_no_blobs_found(self):
        """Should return 0 and not call delete_blobs when folder is empty."""
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            patch("core.gcs_client._get_bucket_name", return_value="test-bucket"),
            patch("core.gcs_client._build_full_gcs_path", return_value="base/adr-123"),
        ):
            result = delete_session_folder("adr-123")

        assert result == 0
        mock_bucket.delete_blobs.assert_not_called()

    def test_raises_on_empty_session_id(self):
        """Should raise ValueError for an empty session_id."""
        with pytest.raises(ValueError, match="session_id must not be empty"):
            delete_session_folder("")

    def test_raises_on_whitespace_only_session_id(self):
        """Should raise ValueError for a whitespace-only session_id."""
        with pytest.raises(ValueError, match="session_id must not be empty"):
            delete_session_folder("   ")

    def test_strips_whitespace_from_session_id(self):
        """Should strip whitespace before building the prefix."""
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            patch("core.gcs_client._get_bucket_name", return_value="test-bucket"),
            patch(
                "core.gcs_client._build_full_gcs_path", return_value="base/adr-123"
            ) as mock_build,
        ):
            delete_session_folder("  adr-123  ")

        mock_build.assert_called_once_with("adr-123")

    def test_appends_trailing_slash_to_prefix(self):
        """Prefix should always end with '/' for correct listing."""
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            patch("core.gcs_client._get_bucket_name", return_value="test-bucket"),
            patch("core.gcs_client._build_full_gcs_path", return_value="base/session"),
        ):
            delete_session_folder("session")

        mock_bucket.list_blobs.assert_called_once_with(prefix="base/session/")

    def test_propagates_gcs_exception(self):
        """GCS errors should bubble up to the caller."""
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.side_effect = Exception("Permission denied")
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            patch("core.gcs_client._get_bucket_name", return_value="test-bucket"),
            patch("core.gcs_client._build_full_gcs_path", return_value="base/adr-123"),
            pytest.raises(Exception, match="Permission denied"),
        ):
            delete_session_folder("adr-123")

    def test_propagates_delete_blobs_exception(self):
        """Errors from delete_blobs should bubble up to the caller."""
        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_bucket.delete_blobs.side_effect = Exception("Delete failed")
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            patch("core.gcs_client._get_bucket_name", return_value="test-bucket"),
            patch("core.gcs_client._build_full_gcs_path", return_value="base/adr-123"),
            pytest.raises(Exception, match="Delete failed"),
        ):
            delete_session_folder("adr-123")


class TestJsonUriToPdfUri:
    """Tests for json_uri_to_pdf_uri — derives original PDF path from extracted JSON path."""

    def test_standard_conversion(self):
        uri = "gs://bucket/base/session/extracted_text/sample_adr_1_p1-2.json"
        result = json_uri_to_pdf_uri(uri)
        assert result == "gs://bucket/base/session/tmp/sample_adr_1_p1-2.pdf"

    def test_deep_nested_path(self):
        uri = "gs://bucket/a/b/c/extracted_text/doc.json"
        result = json_uri_to_pdf_uri(uri)
        assert result == "gs://bucket/a/b/c/tmp/doc.pdf"

    def test_no_extracted_text_returns_original(self):
        uri = "gs://bucket/base/session/other/file.json"
        result = json_uri_to_pdf_uri(uri)
        assert result == uri

    def test_non_json_extension_returns_original(self):
        uri = "gs://bucket/base/session/extracted_text/file.txt"
        result = json_uri_to_pdf_uri(uri)
        assert result == uri

    def test_not_gcs_uri_returns_original(self):
        result = json_uri_to_pdf_uri("/local/extracted_text/file.json")
        assert result == "/local/extracted_text/file.json"


class TestGenerateSignedUrl:
    """Tests for generate_signed_url — creates time-limited HTTPS URLs for GCS objects."""

    def test_returns_emulator_url_when_emulator_configured(self):
        with patch.dict("os.environ", {"STORAGE_EMULATOR_HOST": "http://gcs:4443"}):
            result = generate_signed_url("gs://bucket/path/to/file.pdf")

        assert result == "http://gcs:4443/storage/v1/b/bucket/o/path/to/file.pdf?alt=media"

    def test_returns_signed_url_string(self):
        mock_blob = MagicMock()
        mock_blob.generate_signed_url.return_value = "https://storage.googleapis.com/bucket/path?X-Goog-Signature=abc"
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with (
            patch("core.gcs_client.get_gcs_client", return_value=mock_client),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os
            os.environ.pop("STORAGE_EMULATOR_HOST", None)
            os.environ.pop("GCS_EMULATOR_HOST", None)
            result = generate_signed_url("gs://bucket/path/to/file.pdf")

        assert result.startswith("https://")
        mock_bucket.blob.assert_called_once_with("path/to/file.pdf")
        mock_blob.generate_signed_url.assert_called_once()

    def test_custom_expiration(self):
        from datetime import timedelta

        mock_blob = MagicMock()
        mock_blob.generate_signed_url.return_value = "https://signed.url"
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("core.gcs_client.get_gcs_client", return_value=mock_client):
            generate_signed_url("gs://bucket/file.pdf", expiration=timedelta(hours=2))

        call_kwargs = mock_blob.generate_signed_url.call_args[1]
        assert call_kwargs["expiration"] == timedelta(hours=2)

    def test_invalid_uri_raises(self):
        with pytest.raises(ValueError, match="Must start with 'gs://'"):
            generate_signed_url("https://not-gcs/file.pdf")

    def test_returns_none_on_signing_failure(self):
        mock_blob = MagicMock()
        mock_blob.generate_signed_url.side_effect = Exception("No credentials")
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("core.gcs_client.get_gcs_client", return_value=mock_client):
            result = generate_signed_url("gs://bucket/file.pdf")

        assert result is None
