"""Unit tests for session warmup."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_MOD = "src.session_manager.warmup"


@pytest.mark.asyncio
class TestWarmupSession:
    @patch(f"{_MOD}.invoke_graph", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_agent")
    async def test_calls_invoke_graph_with_warmup_thread_id(
        self, mock_get_agent, mock_invoke_graph
    ):
        mock_graph = MagicMock(name="compiled_graph")
        mock_get_agent.return_value = mock_graph
        mock_invoke_graph.return_value = {"messages": []}

        from src.session_manager.warmup import warmup_session

        await warmup_session("sess-abc-123")

        mock_get_agent.assert_called_once()
        mock_invoke_graph.assert_called_once_with(
            mock_graph,
            "Summarize the key clinical findings in this document.",
            "sess-abc-123-warmup",
        )

    @patch(f"{_MOD}.invoke_graph", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_agent")
    async def test_does_not_propagate_errors(self, mock_get_agent, mock_invoke_graph):
        mock_get_agent.return_value = MagicMock()
        mock_invoke_graph.side_effect = RuntimeError("LLM connection failed")

        from src.session_manager.warmup import warmup_session

        await warmup_session("sess-abc-123")

    @patch(f"{_MOD}.get_agent", side_effect=Exception("agent build failed"))
    async def test_does_not_propagate_agent_factory_errors(self, mock_get_agent):
        from src.session_manager.warmup import warmup_session

        await warmup_session("sess-abc-123")

    @patch(f"{_MOD}.invoke_graph", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_agent")
    async def test_logs_timing_at_info(self, mock_get_agent, mock_invoke_graph, caplog):
        import logging

        mock_get_agent.return_value = MagicMock()
        mock_invoke_graph.return_value = {"messages": []}

        from src.session_manager.warmup import warmup_session

        with caplog.at_level(logging.INFO, logger=_MOD):
            await warmup_session("sess-xyz")

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("warmup" in msg.lower() and "sess-xyz" in msg for msg in info_messages)

    @patch(f"{_MOD}.invoke_graph", new_callable=AsyncMock)
    @patch(f"{_MOD}.get_agent")
    async def test_logs_error_at_warning(self, mock_get_agent, mock_invoke_graph, caplog):
        import logging

        mock_get_agent.return_value = MagicMock()
        mock_invoke_graph.side_effect = RuntimeError("timeout")

        from src.session_manager.warmup import warmup_session

        with caplog.at_level(logging.WARNING, logger=_MOD):
            await warmup_session("sess-fail")

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("sess-fail" in msg for msg in warning_messages)
