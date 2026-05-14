"""
Session Retriever for ADR Vector Database.

Provides LangChain-compatible retrievers that return only documents
belonging to a specific session. Supports semantic and hybrid (BM25 + semantic) search.

Usage:
    from adr_vector_database.retriever import get_session_retriever, get_hybrid_retriever

    # Semantic retriever
    retriever = get_session_retriever(session_id="session-123", k=5)
    docs = retriever.invoke("What is the diagnosis?")

    # Hybrid retriever (BM25 + Semantic)
    hybrid_retriever = get_hybrid_retriever(session_id="session-123", k=5)
    docs = hybrid_retriever.invoke("MRN 12345")
"""

from __future__ import annotations

import threading

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from src.core.config import vectorstore_config
from src.core.embedding_client import get_embedding_client
from src.core.logger import get_logger
from src.core.pgvector_store import create_vector_store

logger = get_logger(__name__)


class VectorStoreSingleton:
    """Singleton manager for the vector store instance."""

    _instance: VectorStoreSingleton | None = None
    _lock: threading.Lock = threading.Lock()
    _store_lock: threading.Lock
    _vector_stores: dict[str, VectorStore]

    def __new__(cls) -> VectorStoreSingleton:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Cache one VectorStore per collection — ADR and policy
                    # collections share the singleton but must not share a store.
                    cls._instance._vector_stores = {}
                    cls._instance._store_lock = threading.Lock()
                    logger.info("VectorStoreSingleton initialized")
        return cls._instance

    def get_vector_store(self, collection_name: str | None = None) -> VectorStore:
        """Get or create the vector store instance for the given collection."""
        coll_name = collection_name or vectorstore_config.COLLECTION_NAME
        with self._store_lock:
            if coll_name not in self._vector_stores:
                embeddings = get_embedding_client().embeddings
                self._vector_stores[coll_name] = create_vector_store(
                    embeddings=embeddings,
                    collection_name=coll_name,
                )
                logger.info("Vector store created for collection '%s'", coll_name)
            return self._vector_stores[coll_name]

    def reset(self) -> None:
        """Reset all cached vector stores (useful for testing)."""
        with self._store_lock:
            self._vector_stores = {}
            logger.info("Vector store singleton reset")


def get_vector_store_singleton() -> VectorStoreSingleton:
    """Get the VectorStoreSingleton instance."""
    return VectorStoreSingleton()


class HybridRetrieverManager:
    """
    Singleton manager for hybrid retrievers.

    Caches BM25 and ensemble retrievers per session to avoid rebuilding
    the BM25 index on every query.
    """

    _instance: HybridRetrieverManager | None = None
    _lock: threading.Lock = threading.Lock()
    _bm25_retrievers: dict[str, BM25Retriever]
    _ensemble_retrievers: dict[str, EnsembleRetriever]
    _retriever_lock: threading.Lock

    def __new__(cls) -> HybridRetrieverManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._bm25_retrievers = {}
                    cls._instance._ensemble_retrievers = {}
                    cls._instance._retriever_lock = threading.Lock()
                    logger.info("HybridRetrieverManager singleton initialized")
        return cls._instance

    def get_retriever(
        self,
        session_id: str,
        k: int | None = None,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        semantic_search_type: str | None = None,
        fetch_k: int | None = None,
        lambda_mult: float | None = None,
        collection_name: str | None = None,
        max_documents_for_bm25: int = 1000,
        force_refresh: bool = False,
    ) -> BaseRetriever:
        """Get or create a hybrid retriever for the given session."""
        with self._retriever_lock:
            if session_id in self._ensemble_retrievers and not force_refresh:
                logger.debug("Returning cached hybrid retriever for session '%s'", session_id)
                return self._ensemble_retrievers[session_id]

            logger.info("Building hybrid retriever for session '%s'", session_id)

            documents = get_session_documents(
                session_id=session_id,
                collection_name=collection_name,
                limit=max_documents_for_bm25,
            )

            if not documents:
                raise ValueError(f"No documents found for session '{session_id}'.")

            k = k or vectorstore_config.DEFAULT_K
            semantic_search_type = semantic_search_type or "similarity"

            bm25_retriever = BM25Retriever.from_documents(documents=documents, k=k)
            self._bm25_retrievers[session_id] = bm25_retriever

            semantic_retriever = get_session_retriever(
                session_id=session_id,
                search_type=semantic_search_type,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                collection_name=collection_name,
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever],
                weights=[bm25_weight, semantic_weight],
            )
            self._ensemble_retrievers[session_id] = ensemble_retriever

            logger.info(
                "Hybrid retriever cached: session='%s', docs=%d, bm25_weight=%.2f",
                session_id,
                len(documents),
                bm25_weight,
            )

            return ensemble_retriever

    def refresh_session(self, session_id: str, **kwargs) -> BaseRetriever:
        """Force refresh the retriever for a session (use when new documents added)."""
        return self.get_retriever(session_id, force_refresh=True, **kwargs)

    def clear_session(self, session_id: str) -> bool:
        """Remove cached retrievers for a specific session."""
        with self._retriever_lock:
            found = False
            if session_id in self._bm25_retrievers:
                del self._bm25_retrievers[session_id]
                found = True
            if session_id in self._ensemble_retrievers:
                del self._ensemble_retrievers[session_id]
                found = True
            if found:
                logger.info("Cleared hybrid retriever cache for session '%s'", session_id)
            return found

    def clear_all(self) -> int:
        """Remove all cached retrievers. Returns number of sessions cleared."""
        with self._retriever_lock:
            count = len(self._ensemble_retrievers)
            self._bm25_retrievers.clear()
            self._ensemble_retrievers.clear()
            logger.info("Cleared all %d hybrid retriever caches", count)
            return count

    @property
    def active_sessions(self) -> list[str]:
        """Get list of sessions with cached retrievers."""
        with self._retriever_lock:
            return list(self._ensemble_retrievers.keys())

    def is_cached(self, session_id: str) -> bool:
        """Check if a session has a cached retriever."""
        with self._retriever_lock:
            return session_id in self._ensemble_retrievers


def get_hybrid_retriever_manager() -> HybridRetrieverManager:
    """Get the HybridRetrieverManager singleton instance."""
    return HybridRetrieverManager()


def get_session_retriever(
    session_id: str,
    search_type: str | None = None,
    k: int | None = None,
    score_threshold: float | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
    collection_name: str | None = None,
) -> BaseRetriever:
    """
    Create a LangChain retriever for a specific session.

    Args:
        session_id: The session ID to filter documents by.
        search_type: "similarity", "mmr", or "similarity_score_threshold".
        k: Number of documents to return.
        score_threshold: Minimum score for similarity_score_threshold search.
        fetch_k: Number of candidates for MMR reranking.
        lambda_mult: MMR diversity factor (0=diverse, 1=relevant).
        collection_name: Optional PGVector collection name.

    Returns:
        Configured LangChain BaseRetriever instance.
    """
    vector_store = get_vector_store_singleton().get_vector_store(collection_name=collection_name)

    search_type = search_type or vectorstore_config.DEFAULT_SEARCH_TYPE
    k = k or vectorstore_config.DEFAULT_K
    fetch_k = fetch_k or vectorstore_config.DEFAULT_FETCH_K
    lambda_mult = lambda_mult if lambda_mult is not None else vectorstore_config.DEFAULT_LAMBDA_MULT

    search_kwargs = {
        "k": k,
        "filter": {"session_id": session_id},
    }

    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult
    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = (
            score_threshold or vectorstore_config.DEFAULT_SCORE_THRESHOLD
        )

    logger.debug("Creating retriever for session '%s': type=%s, k=%d", session_id, search_type, k)

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    logger.info("Session retriever created: session='%s', type='%s'", session_id, search_type)

    return retriever


def get_hybrid_retriever(
    session_id: str,
    documents: list[Document] | None = None,
    k: int | None = None,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    semantic_search_type: str | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
    collection_name: str | None = None,
    max_documents_for_bm25: int = 1000,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> BaseRetriever:
    """
    Get or create a hybrid retriever combining BM25 (keyword) and semantic search.

    Args:
        session_id: The session ID to filter documents by.
        documents: Documents for BM25 index. If None, fetches from vector store.
        k: Number of documents to return.
        bm25_weight: Weight for BM25 results (0.0 to 1.0).
        semantic_weight: Weight for semantic results (0.0 to 1.0).
        semantic_search_type: Search type for semantic retriever.
        fetch_k: Number of candidates for MMR reranking.
        lambda_mult: MMR diversity factor.
        collection_name: Optional PGVector collection name.
        max_documents_for_bm25: Max documents to fetch for BM25 index.
        use_cache: If True (default), use cached retriever if available.
        force_refresh: If True, rebuild the retriever even if cached.

    Returns:
        EnsembleRetriever combining BM25 and semantic search.
    """
    if use_cache:
        return get_hybrid_retriever_manager().get_retriever(
            session_id=session_id,
            k=k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            semantic_search_type=semantic_search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            collection_name=collection_name,
            max_documents_for_bm25=max_documents_for_bm25,
            force_refresh=force_refresh,
        )

    # Non-cached version
    if not documents:
        documents = get_session_documents(
            session_id=session_id,
            collection_name=collection_name,
            limit=max_documents_for_bm25,
        )
        if not documents:
            raise ValueError(f"No documents found for session '{session_id}'.")

    k = k or vectorstore_config.DEFAULT_K
    semantic_search_type = semantic_search_type or "similarity"

    logger.debug(
        "Creating hybrid retriever: session='%s', k=%d, bm25=%.2f, semantic=%.2f",
        session_id,
        k,
        bm25_weight,
        semantic_weight,
    )

    bm25_retriever = BM25Retriever.from_documents(documents=documents, k=k)

    semantic_retriever = get_session_retriever(
        session_id=session_id,
        search_type=semantic_search_type,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        collection_name=collection_name,
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, semantic_weight],
    )

    logger.info(
        "Hybrid retriever created: session='%s', bm25=%.2f, docs=%d",
        session_id,
        bm25_weight,
        len(documents),
    )

    return ensemble_retriever


def get_session_documents(
    session_id: str,
    collection_name: str | None = None,
    limit: int = 1000,
) -> list[Document]:
    """
    Fetch all documents for a session from the vector store via direct SQL.

    Args:
        session_id: The session ID to fetch documents for.
        collection_name: Optional collection name override.
        limit: Maximum number of documents to fetch.

    Returns:
        List of Document objects for the session.
    """
    from sqlalchemy import text
    from src.core.cloudsql_pg_client import get_cloudsql_client

    engine = get_cloudsql_client().engine
    assert engine is not None
    coll_name = collection_name or vectorstore_config.COLLECTION_NAME

    query = text("""
        SELECT document, cmetadata
        FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
        )
        AND cmetadata->>'session_id' = :session_id
        LIMIT :limit
    """)

    documents = []
    with engine.connect() as conn:
        result = conn.execute(
            query,
            {
                "collection_name": coll_name,
                "session_id": session_id,
                "limit": limit,
            },
        )
        for row in result:
            documents.append(
                Document(
                    page_content=row[0],
                    metadata=row[1] if row[1] else {},
                )
            )

    logger.info("Fetched %d documents for session '%s'", len(documents), session_id)
    return documents


# Type alias for backwards compatibility
SessionRetriever = BaseRetriever


__all__ = [
    "get_session_retriever",
    "get_hybrid_retriever",
    "get_hybrid_retriever_manager",
    "get_session_documents",
    "get_vector_store_singleton",
    "VectorStoreSingleton",
    "HybridRetrieverManager",
    "SessionRetriever",
]
