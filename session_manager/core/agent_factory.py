"""
Agent Singleton.

Thin wrapper that caches a **single** compiled LangGraph agent and returns
it on every ``get_agent()`` call.

All graph construction, tool wiring, and routing is handled by
``src.agents.graph.create_agent_graph``.  All tool configuration is
handled by ``src.tools.adr_search.get_adr_search_tool``.  This module's
only job is:

1. Call those functions **once**.
2. Cache the compiled graph.
3. Return the same instance on subsequent calls.

Usage:
    from src.session_manager.core.agent_factory import get_agent, configure_agent

    # Optionally customise once at app startup
    configure_agent(system_prompt="You are a helpful medical AI.")

    # Every call returns the same compiled graph
    graph = get_agent()
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.core.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Module-level singleton state
# ------------------------------------------------------------------ #

_agent = None
_lock = threading.Lock()

# Optional overrides — set via ``configure_agent()`` before the first
# ``get_agent()`` call.  When ``None`` the defaults from graph.py /
# adr_search.py are used.
_system_prompt: str | None = None
_max_turns: int = 10
_extra_tools: list[BaseTool] = []
_checkpointer: BaseCheckpointSaver | None = None


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #


def configure_agent(
    *,
    system_prompt: str | None = None,
    max_turns: int = 10,
    extra_tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    rebuild: bool = True,
) -> None:
    """
    (Re)configure the shared agent.

    Call once at application startup **before** the first ``get_agent()``
    call.  If the agent has already been built and *rebuild* is ``True``
    (the default), the cached instance is discarded so it will be
    recompiled with the new settings on the next access.

    Args:
        system_prompt: Custom system prompt for the LLM.
        max_turns: Max conversation turns kept in context.
        extra_tools: Additional LangChain tools beyond the default
            ``adr_search`` tool.
        checkpointer: LangGraph checkpointer (defaults to in-memory
            ``MemorySaver`` from ``graph.create_checkpointer``).
        rebuild: Invalidate the cached agent so it is rebuilt on
            next ``get_agent()`` call.
    """
    global _system_prompt, _max_turns, _extra_tools, _checkpointer, _agent

    _system_prompt = system_prompt
    _max_turns = max_turns
    _extra_tools = list(extra_tools) if extra_tools else []
    _checkpointer = checkpointer

    if rebuild:
        with _lock:
            _agent = None
        logger.info("Agent config updated – cached agent invalidated")
    else:
        logger.info("Agent config updated (rebuild deferred)")


def get_agent():
    """
    Return the singleton compiled LangGraph agent.

    The graph is compiled lazily on the first call (using
    ``graph.create_agent_graph``) and reused for every subsequent
    call.  Thread-safe.

    Returns:
        A compiled LangGraph agent.
    """
    global _agent

    if _agent is None:
        with _lock:
            if _agent is None:
                _agent = _build_agent()
    return _agent


def reset_agent() -> None:
    """
    Discard the cached agent (useful for testing).

    The next ``get_agent()`` call will recompile the graph.
    """
    global _agent

    with _lock:
        _agent = None
    logger.info("Singleton agent reset")


# ------------------------------------------------------------------ #
#  Internal
# ------------------------------------------------------------------ #


def _build_agent():
    """Compile the graph once using existing factories in graph.py and adr_search.py."""
    # Import here to keep module-level import footprint small and avoid
    # circular imports if graph.py ever imports from session_manager.
    from src.agents.graph import create_agent_graph, create_checkpointer
    from src.tools.adr_search import adr_search_tool
    from src.tools.adr_summary import AdrSummaryTool
    from src.tools.policy_list import PolicyListTool
    from src.tools.policy_search import PolicySearchTool
    from src.tools.policy_summary import PolicySummaryTool

    logger.info("Building singleton LangGraph agent")

    tools: list[BaseTool] = [
        adr_search_tool,
        AdrSummaryTool(),
        PolicySearchTool(),
        PolicyListTool(),
        PolicySummaryTool(),
    ]
    tools.extend(_extra_tools)

    checkpointer = _checkpointer or create_checkpointer()

    graph = create_agent_graph(
        tools=tools,
        system_prompt=_system_prompt,
        max_turns=_max_turns,
        checkpointer=checkpointer,
    )

    logger.info(
        "Singleton agent built: %d tool(s), checkpointer=%s",
        len(tools),
        type(checkpointer).__name__,
    )
    return graph
