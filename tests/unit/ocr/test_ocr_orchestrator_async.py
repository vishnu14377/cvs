"""Tests for async OCR orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ocr.data_models.sub_file_models import SubFileMetadata, SubFileResult
from src.ocr.ocr_orchestrator_async import OcrOrchestratorAsync


def _make_result(doc_name="doc", start=1, end=5, success=True, error=None):
    return SubFileResult(
        metadata=SubFileMetadata(
            document_name=doc_name, base_page_number=start, end_page_number=end
        ),
        success=success,
        pages=[MagicMock()] if success else [],
        error=error,
        model_used="mistral" if success else None,
    )


def _make_handler_mock(process_sub_file):
    """Build a MagicMock handler that also supports `async with`."""
    mock_handler = MagicMock()
    mock_handler.process_sub_file = process_sub_file
    mock_handler.__aenter__ = AsyncMock(return_value=mock_handler)
    mock_handler.__aexit__ = AsyncMock(return_value=None)
    return mock_handler


class TestOcrOrchestratorAsync:
    @pytest.mark.asyncio
    @patch("ocr.ocr_orchestrator_async.SubFileHandlerAsync")
    @patch("ocr.ocr_orchestrator_async.list_files_in_gcs_folder", new_callable=AsyncMock)
    async def test_run_full_pipeline(self, mock_list, mock_handler_cls):
        mock_list.return_value = ["gs://b/f1.pdf", "gs://b/f2.pdf"]

        mock_handler = _make_handler_mock(
            AsyncMock(side_effect=[_make_result("f1"), _make_result("f2")])
        )
        mock_handler_cls.return_value = mock_handler

        orch = OcrOrchestratorAsync(session_id="s1", max_workers=2)
        result = await orch.run("gs://b/folder/")

        assert result.success is True
        assert result.total_sub_files == 2
        assert result.successful_sub_files == 2
        assert result.failed_sub_files == 0
        mock_handler.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("ocr.ocr_orchestrator_async.SubFileHandlerAsync")
    @patch("ocr.ocr_orchestrator_async.list_files_in_gcs_folder", new_callable=AsyncMock)
    async def test_run_partial_failure(self, mock_list, mock_handler_cls):
        mock_list.return_value = ["gs://b/f1.pdf", "gs://b/f2.pdf", "gs://b/f3.pdf"]

        mock_handler = _make_handler_mock(
            AsyncMock(
                side_effect=[
                    _make_result("f1"),
                    _make_result("f2", success=False, error="timeout"),
                    _make_result("f3"),
                ]
            )
        )
        mock_handler_cls.return_value = mock_handler

        orch = OcrOrchestratorAsync(session_id="s1")
        result = await orch.run("gs://b/folder/")

        assert result.success is False
        assert result.successful_sub_files == 2
        assert result.failed_sub_files == 1

    @pytest.mark.asyncio
    @patch("ocr.ocr_orchestrator_async.list_files_in_gcs_folder", new_callable=AsyncMock)
    async def test_run_empty_folder(self, mock_list):
        mock_list.return_value = []

        orch = OcrOrchestratorAsync(session_id="s1")
        result = await orch.run("gs://b/empty/")

        assert result.success is True
        assert result.total_sub_files == 0

    @pytest.mark.asyncio
    @patch("ocr.ocr_orchestrator_async.SubFileHandlerAsync")
    @patch("ocr.ocr_orchestrator_async.list_files_in_gcs_folder", new_callable=AsyncMock)
    async def test_semaphore_limits_concurrency(self, mock_list, mock_handler_cls):
        mock_list.return_value = [f"gs://b/f{i}.pdf" for i in range(6)]

        concurrent = {"current": 0, "max": 0}

        async def mock_process(uri):
            concurrent["current"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["current"])
            import asyncio

            await asyncio.sleep(0.01)
            concurrent["current"] -= 1
            return _make_result(uri.rsplit("/", 1)[-1])

        mock_handler = _make_handler_mock(mock_process)
        mock_handler_cls.return_value = mock_handler

        orch = OcrOrchestratorAsync(session_id="s1", max_workers=2)
        result = await orch.run("gs://b/folder/")

        assert result.total_sub_files == 6
        assert result.successful_sub_files == 6
        assert concurrent["max"] <= 2

    @pytest.mark.asyncio
    @patch("ocr.ocr_orchestrator_async.SubFileHandlerAsync")
    @patch("ocr.ocr_orchestrator_async.list_files_in_gcs_folder", new_callable=AsyncMock)
    async def test_exception_in_sub_file_handled(self, mock_list, mock_handler_cls):
        mock_list.return_value = ["gs://b/f1.pdf"]

        mock_handler = _make_handler_mock(AsyncMock(side_effect=RuntimeError("exploded")))
        mock_handler_cls.return_value = mock_handler

        orch = OcrOrchestratorAsync(session_id="s1")
        result = await orch.run("gs://b/folder/")

        assert result.total_sub_files == 1
        assert result.failed_sub_files == 1
        assert "exploded" in result.sub_file_results[0].error
