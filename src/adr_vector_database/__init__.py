"""
ADR Vector Database Module.

Provides vector storage and retrieval for ADR documents.

Main Components:
    get_session_retriever: Factory for creating session-scoped retrievers
    get_hybrid_retriever: Factory for BM25 + semantic hybrid retrievers
    VectorStoreManager: Manages document storage operations
    ingest_session: Pipeline for ingesting extracted documents

Usage:
    # Retrieval
    from src.adr_vector_database import get_session_retriever, get_hybrid_retriever
    retriever = get_session_retriever(session_id="sess-123")
    docs = retriever.invoke("patient diagnosis")

    # Ingestion
    from src.adr_vector_database import ingest_session
    result = ingest_session(session_id="sess-123")
"""

from src.adr_vector_database.chunker import DocumentChunker
from src.adr_vector_database.data_models import (
    BatchIngestionResult,
    DocumentChunk,
    ExtractedDocument,
    ExtractedPage,
    IngestionResult,
)
from src.adr_vector_database.file_processor import FileProcessor

# Ingestion pipeline
from src.adr_vector_database.ingestion_pipeline import (
    ingest_document,
    ingest_document_from_gcs,  # Alias
    ingest_session,
    ingest_session_documents,  # Alias
)
from src.adr_vector_database.retriever import (
    HybridRetrieverManager,
    SessionRetriever,
    VectorStoreSingleton,
    get_hybrid_retriever,
    get_hybrid_retriever_manager,
    get_session_documents,
    get_session_retriever,
    get_vector_store_singleton,
)

# Vector store (insert/delete/cleanup operations only)
from src.adr_vector_database.vector_store import (
    VectorStoreManager,
    batch_insert_documents,
    cleanup_collection,
    delete_session_documents,
    get_vector_store_manager,
    insert_documents,
)

__all__ = [
    # Retriever
    "SessionRetriever",
    "get_session_retriever",
    "get_hybrid_retriever",
    "get_hybrid_retriever_manager",
    "get_session_documents",
    "get_vector_store_singleton",
    "VectorStoreSingleton",
    "HybridRetrieverManager",
    # Vector store
    "VectorStoreManager",
    "batch_insert_documents",
    "insert_documents",
    "cleanup_collection",
    "delete_session_documents",
    "get_vector_store_manager",
    # Ingestion
    "ingest_session",
    "ingest_document",
    "ingest_session_documents",
    "ingest_document_from_gcs",
    # Data models
    "ExtractedDocument",
    "ExtractedPage",
    "IngestionResult",
    "BatchIngestionResult",
    "DocumentChunk",
    # Processing
    "FileProcessor",
    "DocumentChunker",
]
