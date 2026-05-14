"""
Session Deletion.

Deletes all artefacts associated with a session:

1. **Vector store documents** — chunks stored in PGVector with matching
   ``session_id`` metadata (via ``delete_session_documents``).
2. **Cached retrievers** — BM25 / ensemble retriever caches held by
   ``HybridRetrieverManager`` (via ``clear_session``).
3. **Conversation history** — checkpointed messages held by the agent's
   checkpointer (via ``clear_session_history``).
4. **GCS artefacts** — OCR split files (``<session_id>/tmp/``) and
   extracted text (``<session_id>/extracted_text/``) stored under the
   configured ``GCS_WORKING_FOLDER``.
5. **Local files** — Temporary files under ``data/<session_id>/``
   (via ``cleanup_local_data``).

Usage:
    from src.session_manager.deletion import delete_session

    result = delete_session("adr-20260414-a1b2c3d4", graph=agent_graph)
    print(result)
    # SessionDeletionResult(session_id='adr-20260414-a1b2c3d4',
    #     vectors_deleted=42, retriever_cache_cleared=True,
    #     history_cleared=True, gcs_blobs_deleted=15,
    #     local_files_cleaned=True, errors=[])
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.adr_vector_database.retriever import get_hybrid_retriever_manager
from src.adr_vector_database.vector_store import delete_session_documents
from src.agents.graph import clear_session_history
from src.core.gcs_client import delete_session_folder
from src.core.local_directory_handler import cleanup_local_data
from src.core.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Result
# ------------------------------------------------------------------ #


@dataclass
class SessionDeletionResult:
    """
    Result of a session deletion operation.

    Attributes:
        session_id: The session that was deleted.
        vectors_deleted: Number of vector-store rows removed.
        retriever_cache_cleared: Whether cached retrievers were found and cleared.
        history_cleared: Whether the checkpointer's conversation history was cleared.
        gcs_blobs_deleted: Number of GCS blobs removed.
        local_files_cleaned: Whether local data directory was cleaned up.
        errors: Non-fatal error messages (the operation is best-effort).
    """

    session_id: str
    vectors_deleted: int = 0
    retriever_cache_cleared: bool = False
    history_cleared: bool = False
    gcs_blobs_deleted: int = 0
    local_files_cleaned: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """``True`` if no errors occurred."""
        return len(self.errors) == 0


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #


def delete_session(
    session_id: str,
    *,
    graph=None,
    collection_name: str | None = None,
    delete_vectors: bool = True,
    clear_retriever_cache: bool = True,
    clear_history: bool = True,
    delete_gcs: bool = False,
    delete_local: bool = True,
) -> SessionDeletionResult:
    """
    Delete all artefacts for a session.

    This is a **best-effort** operation: each cleanup step runs
    independently, and failures in one step do not prevent the others
    from executing.  Check ``result.errors`` for any issues.

    Individual cleanup steps can be toggled via the boolean flags.
    All flags default to ``True`` (delete everything).

    Steps performed (when the corresponding flag is ``True``):
        1. Delete vector-store documents whose ``session_id`` metadata
           matches (via ``delete_session_documents``).
        2. Clear cached BM25 / ensemble retrievers for the session
           (via ``HybridRetrieverManager.clear_session``).
        3. Clear the checkpointer's conversation history for the
           session (via ``clear_session_history``).  Requires *graph*
           to be provided; skipped silently if *graph* is ``None``.
        4. Delete GCS blobs under ``<GCS_WORKING_FOLDER>/<session_id>/``
           (tmp split files and extracted text JSONs).
        5. Remove the local ``data/<session_id>/`` directory tree
           (via ``cleanup_local_data``).

    Args:
        session_id: The session to delete.
        graph: The compiled LangGraph agent (needed for clearing
            conversation history from the checkpointer).  If ``None``,
            the history-clearing step is skipped.
        collection_name: PGVector collection name override.  Defaults
            to the value from ``vectorstore_config``.
        delete_vectors: If ``True``, remove vector-store documents for
            the session.
        clear_retriever_cache: If ``True``, clear cached BM25 / ensemble
            retrievers for the session.
        clear_history: If ``True`` **and** *graph* is provided, delete
            the checkpointer's conversation history for the session.
        delete_gcs: If ``True``, delete GCS blobs under the session
            folder.
        delete_local: If ``True``, remove the local
            ``data/<session_id>/`` directory tree.

    Returns:
        A ``SessionDeletionResult`` summarising what was cleaned up.

    Note:
        Since the GCP user accounts do not have storage.delete permission, the GCS cleanup step will fail with a permissions error if *delete_gcs* is ``True``.
        This is expected and does not affect the other cleanup steps.

    Example:
        >>> # Delete everything (including conversation history)
        >>> result = delete_session("adr-20260414-a1b2c3d4", graph=agent_graph)
        >>> print(f"Deleted {result.vectors_deleted} vectors, "
        ...       f"history cleared={result.history_cleared}, "
        ...       f"{result.gcs_blobs_deleted} GCS blobs, success={result.success}")

        >>> # Delete only GCS and local files, keep vectors and history
        >>> result = delete_session(
        ...     "adr-20260414-a1b2c3d4",
        ...     delete_vectors=False,
        ...     clear_retriever_cache=False,
        ...     clear_history=False,
        ... )
    """
    if not session_id or not session_id.strip():
        raise ValueError("session_id must not be empty")

    session_id = session_id.strip()
    result = SessionDeletionResult(session_id=session_id)

    logger.info("Deleting session artefacts: %s", session_id)

    # ── 1. Vector store documents ───────────────────────────────────────
    if delete_vectors:
        try:
            result.vectors_deleted = delete_session_documents(
                session_id=session_id,
                collection_name=collection_name,
            )
            logger.info(
                "Deleted %d vector-store documents for session '%s'",
                result.vectors_deleted,
                session_id,
            )
        except Exception as exc:
            msg = f"Failed to delete vector-store documents: {exc}"
            logger.error(msg)
            result.errors.append(msg)
    else:
        logger.debug("Skipping vector-store deletion for session '%s'", session_id)

    # ── 2. Cached retrievers ────────────────────────────────────────────
    if clear_retriever_cache:
        try:
            hybrid_manager = get_hybrid_retriever_manager()
            had_cache = hybrid_manager.clear_session(session_id)
            # Mark as cleared whether entries existed or not — "nothing to
            # clear" is still a successful clear operation.
            result.retriever_cache_cleared = True
            if had_cache:
                logger.info("Cleared retriever cache for session '%s'", session_id)
            else:
                logger.debug(
                    "No cached retrievers found for session '%s' (nothing to clear)", session_id
                )
        except Exception as exc:
            msg = f"Failed to clear retriever cache: {exc}"
            logger.error(msg)
            result.errors.append(msg)
    else:
        logger.debug("Skipping retriever cache clear for session '%s'", session_id)

    # ── 3. Conversation history (checkpointer) ───────────────────────────
    if clear_history and graph is not None:
        try:
            result.history_cleared = clear_session_history(graph, session_id)
        except Exception as exc:
            msg = f"Failed to clear conversation history: {exc}"
            logger.error(msg)
            result.errors.append(msg)
    elif clear_history and graph is None:
        logger.debug(
            "Skipping conversation-history clear for session '%s' (no graph provided)",
            session_id,
        )
    else:
        logger.debug("Skipping conversation-history clear for session '%s'", session_id)

    # ── 4. GCS artefacts ────────────────────────────────────────────────
    if delete_gcs:
        try:
            result.gcs_blobs_deleted = delete_session_folder(session_id)
            logger.info(
                "Deleted %d GCS blob(s) for session '%s'",
                result.gcs_blobs_deleted,
                session_id,
            )
        except Exception as exc:
            msg = f"Failed to delete GCS artefacts: {exc}"
            logger.error(msg)
            result.errors.append(msg)
    else:
        logger.debug("Skipping GCS deletion for session '%s'", session_id)

    # ── 5. Local files ──────────────────────────────────────────────────
    if delete_local:
        try:
            result.local_files_cleaned = cleanup_local_data(session_id)
            if result.local_files_cleaned:
                logger.info("Cleaned up local data for session '%s'", session_id)
        except Exception as exc:
            msg = f"Failed to clean up local files: {exc}"
            logger.error(msg)
            result.errors.append(msg)
    else:
        logger.debug("Skipping local file deletion for session '%s'", session_id)

    # ── Summary ─────────────────────────────────────────────────────────
    if result.success:
        logger.info(
            "Session '%s' deleted successfully: %d vectors, "
            "%d GCS blobs, retriever cache cleared=%s, "
            "history cleared=%s, local cleaned=%s",
            session_id,
            result.vectors_deleted,
            result.gcs_blobs_deleted,
            result.retriever_cache_cleared,
            result.history_cleared,
            result.local_files_cleaned,
        )
    else:
        logger.warning(
            "Session '%s' deletion completed with %d error(s): %s",
            session_id,
            len(result.errors),
            "; ".join(result.errors),
        )

    return result
