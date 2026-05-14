"""
Session Manager Package.

Provides session lifecycle management for ADR document processing:

- **initialization.py**            – ``initialize_session()`` convenience entry-point.
- **deletion.py**                  – ``delete_session()`` cleanup entry-point.
- **conversation_handler.py**      – ``get_conversation_history()`` for retrieving session chat history.
- **core/session_manager.py**      – ``SessionManager`` class (per-session processing & retriever).
- **core/session_id_generator.py** – ``generate_session_id()`` utility.
- **core/agent_factory.py**        – Singleton agent: ``get_agent()`` / ``configure_agent()``.

The agent is a **singleton** shared across all sessions.  Session isolation
is achieved via ``AgentState.session_id`` (routes tool calls to the correct
session's documents) and the checkpointer's ``thread_id`` (isolates
conversation history per session).

Usage:
    from src.session_manager import initialize_session
    from src.agents.graph import invoke_graph

    # Initialize session (OCR + ingestion)
    session_id, result, manager = initialize_session(
        gcs_uri="gs://bucket/path/to/document.pdf"
    )

    # Get the singleton agent and invoke it directly
    graph = manager.agent
    response = await invoke_graph(graph, "What is the diagnosis?", session_id)

    # Optionally configure the shared agent once at app startup
    from src.session_manager import configure_agent
    configure_agent(system_prompt="You are a helpful medical AI.")
"""

from src.session_manager.conversation_handler import get_conversation_history  # noqa: F401
from src.session_manager.core import (  # noqa: F401
    SessionManager,
    configure_agent,
    generate_session_id,
    get_agent,
    reset_agent,
)
from src.session_manager.deletion import SessionDeletionResult, delete_session  # noqa: F401
from src.session_manager.initialization import initialize_session  # noqa: F401

__all__ = [
    # Initialization
    "initialize_session",
    # Deletion
    "delete_session",
    "SessionDeletionResult",
    # Conversation history
    "get_conversation_history",
    # Per-session lifecycle
    "SessionManager",
    "generate_session_id",
    # Singleton agent
    "get_agent",
    "configure_agent",
    "reset_agent",
]
