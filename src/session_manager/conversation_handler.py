"""
Conversation Handler.

Retrieves and formats conversation history from the agent's checkpointer
state for a given session.  Only **human** and **AI** messages are
included in the output — internal tool-call and tool-result messages
are filtered out so the caller gets a clean conversation transcript.

The returned structure is a JSON-serialisable dictionary suitable for
API responses or logging.

Usage:
    from src.session_manager.conversation_handler import get_conversation_history

    history = get_conversation_history(graph, "adr-20260414-a1b2c3d4")
    print(history)
    # {
    #     "session_id": "adr-20260414-a1b2c3d4",
    #     "message_count": 4,
    #     "messages": [
    #         {"role": "human", "content": "What is the diagnosis?"},
    #         {"role": "ai",    "content": "The diagnosis is …"},
    #         ...
    #     ]
    # }
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from src.agents.graph import get_session_config
from src.core.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #


def get_conversation_history(
    graph,
    session_id: str,
) -> dict[str, Any]:
    """
    Retrieve the conversation history for a session.

    Reads the persisted ``AgentState`` from the graph's checkpointer
    using ``session_id`` as the ``thread_id``, then extracts every
    ``HumanMessage`` and ``AIMessage`` (in order) into a
    JSON-serialisable structure.

    Messages of other types (``ToolMessage``, ``SystemMessage``, etc.)
    are excluded — only the human ↔ AI turns are returned.

    Args:
        graph: The compiled LangGraph agent (must have a checkpointer).
        session_id: The session whose history to retrieve.

    Returns:
        A dictionary with the following shape::

            {
                "session_id": str,
                "message_count": int,
                "messages": [
                    {"role": "human" | "ai", "content": str},
                    ...
                ]
            }

        If the session has no history (or the checkpointer has no
        state for it), ``messages`` will be an empty list and
        ``message_count`` will be ``0``.

    Raises:
        ValueError: If ``session_id`` is empty or whitespace-only.

    Example:
        >>> from src.session_manager.conversation_handler import get_conversation_history
        >>> history = get_conversation_history(graph, "adr-20260414-a1b2c3d4")
        >>> for msg in history["messages"]:
        ...     print(f"[{msg['role']}] {msg['content'][:80]}")
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    session_id = session_id.strip()

    raw_messages = _get_raw_messages(graph, session_id)
    formatted = _format_messages(raw_messages)

    logger.debug(
        "Retrieved %d conversation message(s) for session '%s'",
        len(formatted),
        session_id,
    )

    return {
        "session_id": session_id,
        "message_count": len(formatted),
        "messages": formatted,
    }


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #


def _get_raw_messages(graph, session_id: str) -> list:
    """
    Read the raw message list from the graph's checkpointer state.

    Returns an empty list if the session has no persisted state or
    if an error occurs while reading.
    """
    try:
        config = get_session_config(session_id)
        state = graph.get_state(config)

        if state and state.values:
            messages = state.values.get("messages")
            return messages if isinstance(messages, list) else []
        return []
    except Exception as exc:
        logger.warning(
            "Failed to read agent state for session '%s': %s",
            session_id,
            exc,
        )
        return []


def _format_messages(raw_messages: list) -> list[dict[str, str]]:
    """
    Filter and format raw LangChain messages into JSON-friendly dicts.

    Only ``HumanMessage`` and ``AIMessage`` instances are kept.
    Content is normalised to a plain string (Gemini may return a list
    of content blocks rather than a plain string).
    """
    formatted: list[dict[str, str]] = []

    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            formatted.append(
                {
                    "role": "human",
                    "content": _extract_content(msg.content),
                }
            )
        elif isinstance(msg, AIMessage):
            formatted.append(
                {
                    "role": "ai",
                    "content": _extract_content(msg.content),
                }
            )
        # ToolMessage, SystemMessage, etc. are intentionally skipped

    return formatted


def _extract_content(content: Any) -> str:
    """
    Normalise message content to a plain string.

    Gemini (and some other providers) may return ``content`` as a
    ``list`` of content blocks rather than a plain ``str``.  This
    helper handles both forms.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)
