"""
Unit tests for the agent_factory module (singleton agent).

Tests cover:
- get_agent() — lazy creation, singleton behaviour, thread safety
- configure_agent() — config updates, cache invalidation
- reset_agent() — discards cached agent
- _build_agent() — delegates to graph.create_agent_graph

All external dependencies (graph.py, adr_search.py) are mocked so these
tests run in isolation.

Run with: pytest tests/unit/session_manager/test_agent_factory.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from src.session_manager.core import agent_factory
from src.session_manager.core.agent_factory import (
    configure_agent,
    get_agent,
    reset_agent,
)

# =============================================================================
# Patch targets (inside _build_agent's local imports)
# =============================================================================

_AF = "src.session_manager.core.agent_factory"


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton state before and after each test."""
    agent_factory._agent = None
    agent_factory._system_prompt = None
    agent_factory._max_turns = 10
    agent_factory._extra_tools = []
    agent_factory._checkpointer = None
    yield
    agent_factory._agent = None
    agent_factory._system_prompt = None
    agent_factory._max_turns = 10
    agent_factory._extra_tools = []
    agent_factory._checkpointer = None


# =============================================================================
# Test: get_agent — lazy creation
# =============================================================================


class TestGetAgent:
    """Tests for get_agent()."""

    @patch(f"{_AF}._build_agent")
    def test_builds_on_first_call(self, mock_build):
        """First call should trigger _build_agent."""
        mock_graph = MagicMock(name="compiled_graph")
        mock_build.return_value = mock_graph

        result = get_agent()

        mock_build.assert_called_once()
        assert result is mock_graph

    @patch(f"{_AF}._build_agent")
    def test_returns_cached_on_subsequent_calls(self, mock_build):
        """Second call should return the cached graph, not rebuild."""
        mock_graph = MagicMock(name="compiled_graph")
        mock_build.return_value = mock_graph

        first = get_agent()
        second = get_agent()

        assert first is second
        assert mock_build.call_count == 1

    @patch(f"{_AF}._build_agent")
    def test_singleton_identity(self, mock_build):
        """All calls should return the exact same object."""
        mock_graph = MagicMock()
        mock_build.return_value = mock_graph

        agents = [get_agent() for _ in range(5)]

        assert all(a is agents[0] for a in agents)
        assert mock_build.call_count == 1


# =============================================================================
# Test: reset_agent
# =============================================================================


class TestResetAgent:
    """Tests for reset_agent()."""

    @patch(f"{_AF}._build_agent")
    def test_reset_causes_rebuild(self, mock_build):
        """After reset_agent(), get_agent() should rebuild."""
        graph_a = MagicMock(name="graph_a")
        graph_b = MagicMock(name="graph_b")
        mock_build.side_effect = [graph_a, graph_b]

        first = get_agent()
        reset_agent()
        second = get_agent()

        assert first is graph_a
        assert second is graph_b
        assert mock_build.call_count == 2

    @patch(f"{_AF}._build_agent")
    def test_reset_without_prior_build_is_safe(self, mock_build):
        """reset_agent() should not error if no agent was built yet."""
        reset_agent()  # should not raise
        assert agent_factory._agent is None


# =============================================================================
# Test: configure_agent
# =============================================================================


class TestConfigureAgent:
    """Tests for configure_agent()."""

    def test_sets_system_prompt(self):
        """Should store the system prompt."""
        configure_agent(system_prompt="You are helpful.", rebuild=False)
        assert agent_factory._system_prompt == "You are helpful."

    def test_sets_max_turns(self):
        """Should store max_turns."""
        configure_agent(max_turns=5, rebuild=False)
        assert agent_factory._max_turns == 5

    def test_sets_extra_tools(self):
        """Should store extra tools as a list."""
        tool = MagicMock()
        configure_agent(extra_tools=[tool], rebuild=False)
        assert agent_factory._extra_tools == [tool]

    def test_extra_tools_defaults_to_empty_list(self):
        """None extra_tools should result in empty list."""
        configure_agent(extra_tools=None, rebuild=False)
        assert agent_factory._extra_tools == []

    def test_sets_checkpointer(self):
        """Should store the checkpointer."""
        cp = MagicMock()
        configure_agent(checkpointer=cp, rebuild=False)
        assert agent_factory._checkpointer is cp

    @patch(f"{_AF}._build_agent")
    def test_rebuild_true_invalidates_cache(self, mock_build):
        """rebuild=True should discard the cached agent."""
        mock_build.return_value = MagicMock()
        get_agent()  # populate cache
        assert agent_factory._agent is not None

        configure_agent(system_prompt="new", rebuild=True)
        assert agent_factory._agent is None

    @patch(f"{_AF}._build_agent")
    def test_rebuild_false_preserves_cache(self, mock_build):
        """rebuild=False should keep the cached agent."""
        mock_build.return_value = MagicMock()
        get_agent()  # populate cache

        configure_agent(system_prompt="new", rebuild=False)
        assert agent_factory._agent is not None


# =============================================================================
# Test: _build_agent — delegation
# =============================================================================


class TestBuildAgent:
    """Tests that _build_agent delegates correctly."""

    @patch(f"{_AF}.logger")
    def test_calls_create_agent_graph(self, mock_logger):
        """Should call create_agent_graph with the right arguments."""
        mock_graph = MagicMock(name="compiled_graph")
        mock_checkpointer = MagicMock(name="checkpointer")
        mock_tool = MagicMock(name="adr_search_tool")

        with (
            patch("src.agents.graph.create_agent_graph", return_value=mock_graph) as mock_create,
            patch(
                "src.agents.graph.create_checkpointer",
                return_value=mock_checkpointer,
            ),
            patch("src.tools.adr_search.adr_search_tool", mock_tool),
        ):
            from src.session_manager.core.agent_factory import _build_agent

            result = _build_agent()

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        # Tools list should contain the adr_search_tool
        assert mock_tool in call_kwargs.kwargs.get("tools", call_kwargs[1].get("tools", []))
        assert result is mock_graph

    @patch(f"{_AF}.logger")
    def test_uses_custom_checkpointer_if_set(self, mock_logger):
        """Should use the configured checkpointer instead of creating one."""
        custom_cp = MagicMock(name="custom_checkpointer")
        agent_factory._checkpointer = custom_cp

        mock_graph = MagicMock()
        mock_tool = MagicMock()

        with (
            patch("src.agents.graph.create_agent_graph", return_value=mock_graph) as mock_create,
            patch("src.agents.graph.create_checkpointer") as mock_create_cp,
            patch("src.tools.adr_search.adr_search_tool", mock_tool),
        ):
            from src.session_manager.core.agent_factory import _build_agent

            _build_agent()

        # Should NOT have called create_checkpointer since we provided one
        mock_create_cp.assert_not_called()
        call_kwargs = mock_create.call_args
        assert (
            call_kwargs.kwargs.get("checkpointer", call_kwargs[1].get("checkpointer")) is custom_cp
        )
