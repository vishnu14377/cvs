"""
Tool node for executing tool calls in the agent workflow.

Injects ``session_id`` from the agent state into tool-call arguments that
need it (e.g. ``adr_search``) via a lightweight pre-processing node, then
delegates to LangGraph's prebuilt ``ToolNode`` for actual execution.

The graph wires the two nodes together so that the ``ToolNode`` receives the
full ``RunnableConfig`` (including the ``runtime`` key) directly from the
graph runner — avoiding the "Missing required config key" error that occurs
when ``ToolNode`` is called from inside a wrapper.

Graph wiring (handled in ``graph.py``):

    generate ─→ tools_condition ─→ inject_session_id ─→ tools ─→ generate
                                |
                                └─→ END

Usage:
    from src.agents.tool_node import (
        create_tool_node,
        create_inject_session_node,
        tools_condition,
    )

    inject_fn = create_inject_session_node()
    tool_fn   = create_tool_node(tools=[adr_search_tool])
    workflow.add_node("inject_session_id", inject_fn)
    workflow.add_node("tools", tool_fn)
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import tools_condition
from src.agents.state import AgentState
from src.core.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "ToolNode",
    "tools_condition",
    "create_tool_node",
    "create_inject_session_node",
    "NO_TOOLS_MESSAGE",
]

# =============================================================================
# Constants
# =============================================================================

NO_TOOLS_MESSAGE = "No tools are currently available. Please try again with a different request."

# Tool names whose calls should receive session_id injection.
_SESSION_INJECTED_TOOLS = frozenset(
    {
        "adr_search",
        "adr_summary",
        "policy_search",
        "policy_summary",
    }
)


# =============================================================================
# Helpers
# =============================================================================


def _get_last_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    """Return the last ``AIMessage`` in *messages*, or ``None``."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _create_no_tools_response(
    tool_calls: list[dict[str, Any]],
) -> dict[str, list[ToolMessage]]:
    """Build a ``ToolMessage`` per tool call saying no tools are available."""
    return {
        "messages": [
            ToolMessage(
                content=NO_TOOLS_MESSAGE,
                tool_call_id=call.get("id", f"no-id-{i}"),
            )
            for i, call in enumerate(tool_calls)
        ]
    }


def _inject_session_id(state: AgentState) -> AgentState:
    """Return a shallow copy of *state* with ``session_id`` injected into
    every tool-call whose name is in ``_SESSION_INJECTED_TOOLS``.

    The original state and its messages are **never** mutated.  If no
    injection is needed the original *state* reference is returned.
    """
    messages = state.get("messages", [])
    session_id = state.get("session_id")

    if not session_id or not messages:
        return state

    last_ai = _get_last_ai_message(messages)
    if not last_ai:
        return state

    tool_calls = getattr(last_ai, "tool_calls", [])
    if not tool_calls:
        return state

    if not any(tc.get("name") in _SESSION_INJECTED_TOOLS for tc in tool_calls):
        return state

    # Patch only the calls that need it.
    new_tool_calls = []
    for tc in tool_calls:
        if tc.get("name") in _SESSION_INJECTED_TOOLS:
            tc = copy.copy(tc)
            tc["args"] = {**tc.get("args", {}), "session_id": session_id}
        new_tool_calls.append(tc)

    new_ai = AIMessage(
        content=last_ai.content,
        tool_calls=new_tool_calls,
        additional_kwargs=last_ai.additional_kwargs,
        response_metadata=getattr(last_ai, "response_metadata", {}),
        id=last_ai.id,
    )

    new_messages = list(messages)
    for i in range(len(new_messages) - 1, -1, -1):
        if isinstance(new_messages[i], AIMessage):
            new_messages[i] = new_ai
            break

    return {**state, "messages": new_messages}


# =============================================================================
# Session-ID Injection Node  (runs *before* ToolNode in the graph)
# =============================================================================


async def _inject_session_id_node(state: AgentState) -> AgentState:
    """Graph node that injects ``session_id`` into tool-call args.

    This is a pure state-transform — it never calls the tool itself.
    It returns the (possibly modified) state so the next node (``ToolNode``)
    sees the ``session_id`` argument already present.
    """
    return _inject_session_id(state)


def create_inject_session_node():
    """Return the async function to use as the ``inject_session_id`` node."""
    return _inject_session_id_node


# =============================================================================
# Fallback (no tools configured)
# =============================================================================


async def _no_tools_node(state: AgentState) -> dict[str, list[ToolMessage]]:
    """Fallback node used when the agent has no tools.

    Returns a ``NO_TOOLS_MESSAGE`` for every tool call the LLM produced.
    """
    last_ai = _get_last_ai_message(state.get("messages", []))
    if not last_ai:
        return {"messages": []}

    tool_calls = getattr(last_ai, "tool_calls", [])
    if not tool_calls:
        return {"messages": []}

    logger.warning(
        "No tools available but LLM requested %d tool call(s)",
        len(tool_calls),
    )
    return _create_no_tools_response(tool_calls)


# =============================================================================
# Factory
# =============================================================================


def create_tool_node(
    tools: Sequence[BaseTool] | None = None,
) -> ToolNode | Any:
    """Create the tool node for the agent graph.

    * **With tools** → LangGraph ``ToolNode`` registered directly so it
      receives the full ``RunnableConfig`` (including ``runtime``) from the
      graph runner.  Session-ID injection is handled by a *separate*
      ``inject_session_id`` node wired immediately before this one.
    * **Without tools** → async fallback that replies with
      ``NO_TOOLS_MESSAGE``.

    Usage::

        workflow.add_node("tools", create_tool_node(tools=[adr_search_tool]))
    """
    if not tools:
        logger.debug("Creating tool node with no tools (fallback mode)")
        return _no_tools_node

    logger.debug("Creating ToolNode with %d tool(s)", len(tools))
    return ToolNode(tools=list(tools))
