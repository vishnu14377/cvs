"""
Core subpackage for the session_manager module.

Contains the internal building blocks:

- ``session_id_generator`` – unique session ID generation.
- ``session_manager``      – ``SessionManager`` class (per-session lifecycle).
- ``agent_factory``        – singleton LangGraph agent management.
"""

from src.session_manager.core.agent_factory import (  # noqa: F401
    configure_agent,
    get_agent,
    reset_agent,
)
from src.session_manager.core.session_id_generator import generate_session_id  # noqa: F401
from src.session_manager.core.session_manager import SessionManager  # noqa: F401

__all__ = [
    "generate_session_id",
    "SessionManager",
    "get_agent",
    "configure_agent",
    "reset_agent",
]
