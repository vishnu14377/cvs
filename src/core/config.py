import os
from abc import ABC

from dotenv import load_dotenv

load_dotenv()


class GlobalConfig(ABC):  # noqa: B024
    """
    Abstract base for app-wide settings from the environment.

    Subclass for feature areas that need extra keys; shared GCS/GCP fields
    stay defined here once.
    """

    GCP_PROJECT = os.environ.get("GCP_PROJECT_ID")
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
    AI_AGENT_PREFIX_GCS = os.environ.get("AI_AGENT_PREFIX_GCS")
    GCS_WORKING_FOLDER = (
        f"{GCS_BUCKET_NAME}/{AI_AGENT_PREFIX_GCS}"
        if GCS_BUCKET_NAME and AI_AGENT_PREFIX_GCS
        else None
    )
    GCP_REGION = os.environ.get("GCP_REGION", "us-central1")

    # GCS temp folder for intermediate files
    GCS_TEMP_FOLDER = os.environ.get("OCR_GCS_TEMP_FOLDER", "tmp")
    # GCS folder for extracted text JSON files
    GCS_EXTRACTED_TEXT_FOLDER = os.environ.get("OCR_GCS_EXTRACTED_TEXT_FOLDER", "extracted_text")

    # Vertex AI mode: "real" (default — hit live Vertex) or "stub" (deterministic offline responses for CI).
    VERTEX_AI_MODE = os.environ.get("VERTEX_AI_MODE", "real")
    if VERTEX_AI_MODE not in ("real", "stub"):
        raise ValueError(f"VERTEX_AI_MODE must be 'real' or 'stub', got {VERTEX_AI_MODE!r}")


class VertexAIConfig(GlobalConfig):
    """Vertex AI configuration: region, endpoints, timeouts."""

    # Default timeout for Vertex AI requests (seconds)
    TIMEOUT_SECONDS = int(os.environ.get("VERTEX_TIMEOUT_SECONDS", "300"))
    RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
    INITIAL_DELAY = float(os.environ.get("RETRY_INITIAL_DELAY", "1.0"))
    MAX_DELAY = float(os.environ.get("RETRY_MAX_DELAY", "60.0"))
    RETRY_MULTIPLIER = float(os.environ.get("RETRY_MULTIPLIER", "2.0"))


class MistralOCRConfig:
    """Mistral OCR configuration: model selection, GCS temp folder, and PDF limits."""

    # GCS temp folder for intermediate files
    GCS_TEMP_FOLDER = os.environ.get("MISTRAL_OCR_GCS_TEMP_FOLDER", "mistral_ocr_temp")
    # Mistral OCR model configuration
    MODEL_TIMEOUT_SECONDS = int(os.environ.get("MISTRAL_OCR_TIMEOUT_SECONDS", "300"))
    MISTRAL_MODEL_ID = os.environ.get("MISTRAL_MODEL_ID", "mistral-ocr-2505")
    MISTRAL_PUBLISHER = os.environ.get("MISTRAL_PUBLISHER", "mistralai")


class OCRConfig(VertexAIConfig):
    """OCR module: inherits global GCS/GCP and Vertex AI config plus OCR-only options."""

    # File size limit for "small" files (MB)
    SMALL_SIZE_FILE_LIMIT_MB = int(os.environ.get("OCR_SMALL_SIZE_FILE_LIMIT_MB", "4"))

    # OCR-specific timeout (falls back to Vertex timeout)
    OCR_TIMEOUT_SECONDS = int(
        os.environ.get("OCR_TIMEOUT_SECONDS", str(VertexAIConfig.TIMEOUT_SECONDS))
    )

    # PDF processing limits
    MAX_PAGES_PER_DOCUMENT = int(os.environ.get("OCR_MAX_PAGES_PER_DOCUMENT", "30"))
    PAGES_PER_CHUNK = int(os.environ.get("OCR_PAGES_PER_CHUNK", "20"))

    # Local output folder for storing model responses
    OCR_RESPONSE_DIR = os.environ.get("OCR_RESPONSE_DIR", "ocr_responses")
    LOCAL_TMP_DIR = os.environ.get("LOCAL_TMP_DIR", "tmp")


class LLMConfig(GlobalConfig):
    """LLM configuration: model selection, generation parameters, and retry settings."""

    # Gemini model configuration
    GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-flash")
    # LLM-specific timeout (falls back to Vertex timeout)
    LLM_TIMEOUT_SECONDS = int(
        os.environ.get("LLM_TIMEOUT_SECONDS", str(VertexAIConfig.TIMEOUT_SECONDS))
    )

    # Generation parameters
    DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "8192"))
    DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
    DEFAULT_TOP_P = float(os.environ.get("LLM_TOP_P", "0.95"))
    DEFAULT_TOP_K = int(os.environ.get("LLM_TOP_K", "40"))

    # Retry configuration
    LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_INITIAL_DELAY = float(os.environ.get("LLM_RETRY_INITIAL_DELAY", "1.0"))
    LLM_RETRY_MAX_DELAY = float(os.environ.get("LLM_RETRY_MAX_DELAY", "60.0"))
    LLM_RETRY_MULTIPLIER = float(os.environ.get("LLM_RETRY_MULTIPLIER", "2.0"))


class CloudSQLConfig(GlobalConfig):
    """Cloud SQL PostgreSQL configuration for RAG vector database with pgvector.

    Connection settings are read lazily from os.environ so that secrets
    injected after import time (e.g. from GCP Secret Manager) are picked up.
    """

    @property
    def HOST(self) -> str:  # noqa: N802
        return os.environ.get("CLOUDSQL_HOST", "localhost")

    @property
    def PORT(self) -> int:  # noqa: N802
        return int(os.environ.get("CLOUDSQL_PORT", "5432"))

    @property
    def DATABASE(self) -> str:  # noqa: N802
        return os.environ.get("CLOUDSQL_DATABASE", "cargpgsd1_db")

    @property
    def USER(self) -> str:  # noqa: N802
        return os.environ.get("CLOUDSQL_USER", "postgres")

    @property
    def PASSWORD(self) -> str:  # noqa: N802
        return os.environ.get("CLOUDSQL_PASSWORD", "")

    @property
    def SCHEMA(self) -> str:  # noqa: N802
        return os.environ.get("CLOUDSQL_DATABASE_SCHEMA", "public")

    # Cloud SQL instance connection name (for Cloud SQL Connector/Proxy)
    # Format: project:region:instance
    INSTANCE_CONNECTION_NAME = os.environ.get("CARECONNECT_DEV_DATABASE")

    # Connection pool settings
    POOL_SIZE = int(os.environ.get("CLOUDSQL_POOL_SIZE", "10"))
    MAX_OVERFLOW = int(os.environ.get("CLOUDSQL_MAX_OVERFLOW", "20"))
    POOL_TIMEOUT = int(os.environ.get("CLOUDSQL_POOL_TIMEOUT", "30"))
    POOL_RECYCLE = int(os.environ.get("CLOUDSQL_POOL_RECYCLE", "1800"))


class vectorStoreConfig(GlobalConfig):
    """Configuration for vector store (pgvector) and retriever settings."""

    # Collection name for pgvector (can be overridden in create_vector_store)
    COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "ADR_session_documents")

    # Embedding configuration
    EMBEDDING_MODEL_ID = os.environ.get("VECTORSTORE_EMBEDDING_MODEL_ID", "text-embedding-004")
    EMBEDDING_DIMENSION = int(os.environ.get("VECTORSTORE_EMBEDDING_DIMENSION", "768"))

    # HNSW index configuration
    HNSW_M = int(os.environ.get("VECTORSTORE_HNSW_M", "16"))
    HNSW_EF_CONSTRUCTION = int(os.environ.get("VECTORSTORE_HNSW_EF_CONSTRUCTION", "64"))
    HNSW_EF_SEARCH = int(os.environ.get("VECTORSTORE_HNSW_EF_SEARCH", "40"))

    # Session settings for RAG
    SESSION_TTL_HOURS = int(os.environ.get("RAG_SESSION_TTL_HOURS", "24"))

    # ----- Vector Store Manager Settings -----
    # Default batch size for document insertions
    DEFAULT_BATCH_SIZE = int(os.environ.get("VECTORSTORE_BATCH_SIZE", "100"))

    # ----- Session Retriever Settings -----
    # Default number of documents to return
    DEFAULT_K = int(os.environ.get("RETRIEVER_DEFAULT_K", "4"))
    # Number of candidates to fetch for MMR reranking
    DEFAULT_FETCH_K = int(os.environ.get("RETRIEVER_FETCH_K", "20"))
    # MMR diversity factor (0=diverse, 1=relevant)
    DEFAULT_LAMBDA_MULT = float(os.environ.get("RETRIEVER_LAMBDA_MULT", "0.5"))
    # Default score threshold for threshold search (0.0 to 1.0)
    DEFAULT_SCORE_THRESHOLD = float(os.environ.get("RETRIEVER_SCORE_THRESHOLD", "0.0"))
    # Default search type: similarity, mmr, threshold
    DEFAULT_SEARCH_TYPE = os.environ.get("RETRIEVER_SEARCH_TYPE", "similarity")

    # ----- Ingestion Pipeline Settings -----
    # Default chunk size for text chunking
    DEFAULT_CHUNK_SIZE = int(os.environ.get("INGESTION_CHUNK_SIZE", "1000"))
    # Overlap between chunks
    DEFAULT_CHUNK_OVERLAP = int(os.environ.get("INGESTION_CHUNK_OVERLAP", "200"))
    # Number of parallel workers for file processing
    DEFAULT_MAX_WORKERS = int(os.environ.get("INGESTION_MAX_WORKERS", "4"))


# Singleton config instances
vertex_config = VertexAIConfig()
ocr_config = OCRConfig()
llm_config = LLMConfig()
mistral_ocr_config = MistralOCRConfig()
cloudsql_config = CloudSQLConfig()
vectorstore_config = vectorStoreConfig()
