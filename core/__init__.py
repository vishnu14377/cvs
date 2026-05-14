"""Core module: configuration, logging, and AI clients."""

from .cloudsql_pg_client import (
    CloudSQLClient,
    get_cloudsql_client,
    reset_cloudsql_client,
)
from .config import (
    CloudSQLConfig,
    cloudsql_config,
    llm_config,
    mistral_ocr_config,
    ocr_config,
    vertex_config,
)
from .embedding_client import (
    EmbeddingClient,
    get_embedding_client,
    get_embeddings,
)
from .langchain_client import (
    LangChainClient,
    LLMLoggingCallback,
    get_langchain_client,
)
from .local_directory_handler import (
    PROJECT_ROOT,
    cleanup_local_data,
    get_local_data_dir,
    get_local_data_path,
    get_local_temp_path,
    list_local_files,
)
from .logger import get_logger
from .pgvector_store import create_vector_store
from .vertex_ai_client import (
    DEFAULT_RETRY,
    VertexAIClient,
    create_retry_callback,
    create_retry_with_logging,
    get_vertex_ai_client,
)

__all__ = [
    # Config
    "vertex_config",
    "ocr_config",
    "llm_config",
    "mistral_ocr_config",
    "cloudsql_config",
    # Logger
    "get_logger",
    # Vertex AI Client
    "get_vertex_ai_client",
    "VertexAIClient",
    "DEFAULT_RETRY",
    "create_retry_callback",
    "create_retry_with_logging",
    # LangChain Client
    "LangChainClient",
    "get_langchain_client",
    "LLMLoggingCallback",
    # Embedding Client
    "EmbeddingClient",
    "get_embedding_client",
    "get_embeddings",
    # Local Directory Handler
    "get_local_data_path",
    "get_local_data_dir",
    "get_local_temp_path",
    "cleanup_local_data",
    "list_local_files",
    "PROJECT_ROOT",
    # Cloud SQL PostgreSQL Client
    "CloudSQLClient",
    "CloudSQLConfig",
    "get_cloudsql_client",
    "reset_cloudsql_client",
    # PGVector Store
    "create_vector_store",
]


# can use direct import if we want to avoid loading all modules in core when only one is needed, e.g.:
# from src.core.logger import get_logger
# if we see performance issues, we can use direct imports for specific modules instead of importing everything in __init__.py.
# right now we are using direct imports; but we also have all the imports here
# so we can later remove all the imports here and just use direct imports in the specific modules that need them.
