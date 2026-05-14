"""Tests for async sub file handler."""

import sys
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ocr.sub_file_handler_async import SubFileHandlerAsync


def _mistral_patches(mock_client, *, local_path="/tmp/local_doc.pdf"):
    """Build the set of patches the Mistral path needs for tests.

    Mistral requires a local file, so process_sub_file calls
    async_download_from_gcs before invoking client.process_pdf.
    """
    mock_module = MagicMock()
    mock_module.MistralOcrClientAsync.return_value = mock_client
    stack = ExitStack()
    stack.enter_context(patch.dict(sys.modules, {"ocr.mistral_ocr_client_async": mock_module}))
    mock_download = AsyncMock(return_value=local_path)
    stack.enter_context(patch("ocr.sub_file_handler_async.async_download_from_gcs", mock_download))
    return stack, mock_download


class TestSubFileHandlerAsync:
    @pytest.mark.asyncio
    async def test_process_sub_file_success_mistral_downloads_gcs(self):
        mock_client = MagicMock()
        mock_client.process_pdf = AsyncMock(
            return_value={
                "success": True,
                "pages": [{"index": 1, "extracted_text": "Page 1 text"}],
            }
        )

        stack, mock_download = _mistral_patches(mock_client)
        with stack:
            handler = SubFileHandlerAsync(session_id="s1", model_type="mistral")
            result = await handler.process_sub_file("gs://bucket/s1/tmp/doc_p1-10.pdf")

        # Mistral needs a local path — verify we downloaded and passed that, not the gs:// URI.
        mock_download.assert_awaited_once()
        mock_client.process_pdf.assert_awaited_once()
        called_path = mock_client.process_pdf.await_args.args[0]
        assert not called_path.startswith("gs://")
        # save_response must be False — the orchestrator handles persistence,
        # and the client's default True would write a JSON file to local disk
        # per sub-file (which fails in read-only containers).
        assert mock_client.process_pdf.await_args.kwargs.get("save_response") is False

        assert result.success is True
        assert len(result.pages) == 1
        assert result.metadata.base_page_number == 1
        assert result.metadata.end_page_number == 10

    @pytest.mark.asyncio
    async def test_process_sub_file_success_llm(self):
        mock_client = MagicMock()
        mock_client.process_document = AsyncMock(
            return_value={
                "success": True,
                "pages": [{"index": 1, "extracted_text": "LLM text"}],
            }
        )
        mock_module = MagicMock()
        mock_module.LlmOcrClientAsync.return_value = mock_client

        with patch.dict(sys.modules, {"ocr.llm_ocr_client_async": mock_module}):
            handler = SubFileHandlerAsync(session_id="s1", model_type="llm")
            result = await handler.process_sub_file("gs://bucket/s1/tmp/doc_p1-5.pdf")

        assert result.success is True
        assert result.model_used == "llm"
        # LLM path uses gs:// directly — no download needed.
        mock_client.process_document.assert_awaited_once_with("gs://bucket/s1/tmp/doc_p1-5.pdf")

    @pytest.mark.asyncio
    async def test_process_sub_file_ocr_failure(self):
        mock_client = MagicMock()
        mock_client.process_pdf = AsyncMock(
            return_value={
                "success": False,
                "error": "OCR timeout",
                "pages": [],
            }
        )

        stack, _ = _mistral_patches(mock_client)
        with stack:
            handler = SubFileHandlerAsync(session_id="s1", model_type="mistral")
            result = await handler.process_sub_file("gs://bucket/s1/tmp/doc_p1-5.pdf")

        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_process_sub_file_exception(self):
        mock_client = MagicMock()
        mock_client.process_pdf = AsyncMock(side_effect=RuntimeError("crash"))

        stack, _ = _mistral_patches(mock_client)
        with stack:
            handler = SubFileHandlerAsync(session_id="s1", model_type="mistral")
            result = await handler.process_sub_file("gs://bucket/s1/tmp/doc_p1-5.pdf")

        assert result.success is False
        assert "crash" in result.error

    @pytest.mark.asyncio
    async def test_run_saves_to_gcs(self):
        mock_client = MagicMock()
        mock_client.process_pdf = AsyncMock(
            return_value={
                "success": True,
                "pages": [{"index": 1, "extracted_text": "text"}],
            }
        )

        stack, _ = _mistral_patches(mock_client)
        with (
            stack,
            patch(
                "ocr.sub_file_handler_async.async_upload_json", new_callable=AsyncMock
            ) as mock_upload,
        ):
            mock_upload.return_value = "gs://bucket/output.json"
            handler = SubFileHandlerAsync(session_id="s1")
            result = await handler.run("gs://bucket/s1/tmp/doc_p1-5.pdf", "output.json")

        assert result.success is True
        mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_is_cached_across_calls(self):
        """Handler must reuse one client instance across calls to avoid leaking executors."""
        mock_client = MagicMock()
        mock_client.process_pdf = AsyncMock(
            return_value={
                "success": True,
                "pages": [{"index": 1, "extracted_text": "t"}],
            }
        )
        mock_module = MagicMock()
        mock_module.MistralOcrClientAsync.return_value = mock_client
        mock_download = AsyncMock(return_value="/tmp/local.pdf")

        with (
            patch.dict(sys.modules, {"ocr.mistral_ocr_client_async": mock_module}),
            patch("ocr.sub_file_handler_async.async_download_from_gcs", mock_download),
        ):
            handler = SubFileHandlerAsync(session_id="s1", model_type="mistral")
            await handler.process_sub_file("gs://b/s1/a_p1-2.pdf")
            await handler.process_sub_file("gs://b/s1/a_p3-4.pdf")

        assert mock_module.MistralOcrClientAsync.call_count == 1

    @pytest.mark.asyncio
    async def test_close_releases_llm_client(self):
        """close() (and async-with) must call the underlying client's close()."""
        mock_client = MagicMock()
        mock_module = MagicMock()
        mock_module.LlmOcrClientAsync.return_value = mock_client

        with patch.dict(sys.modules, {"ocr.llm_ocr_client_async": mock_module}):
            async with SubFileHandlerAsync(session_id="s1", model_type="llm") as handler:
                handler._get_client()  # force instantiation

        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_unparseable_filename_uses_1_based_pages(self):
        """When filename parsing fails, page numbers must remain 1-based."""
        mock_client = MagicMock()
        mock_client.process_pdf = AsyncMock(
            return_value={
                "success": True,
                "pages": [
                    {"index": 0, "extracted_text": "p1"},
                    {"index": 1, "extracted_text": "p2"},
                ],
            }
        )

        stack, _ = _mistral_patches(mock_client)
        with stack:
            handler = SubFileHandlerAsync(session_id="s1", model_type="mistral")
            # Filename without the _p<start>-<end> convention triggers the fallback path.
            result = await handler.process_sub_file("gs://bucket/s1/tmp/weird_name.pdf")

        assert result.success is True
        # 1-based page numbering contract — 0-based would mis-index downstream consumers.
        assert result.pages[0].original_page_number == 1
        assert result.pages[1].original_page_number == 2
