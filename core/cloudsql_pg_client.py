"""
Minimal Cloud SQL PostgreSQL engine client.

This module provides ONLY:
- Singleton SQLAlchemy engine with connection pooling
- Automatic pgvector extension initialization

All vector operations should use LangChain's PGVector class directly,
or through the PGVectorStore wrapper.

Usage:
    from core.cloudsql_pg_client import get_cloudsql_client
    from langchain_postgres import PGVector

    # Get engine
    engine = get_cloudsql_client().engine

    # Use with LangChain PGVector directly
    vector_store = PGVector(
        embeddings=my_embeddings,
        collection_name="my_collection",
        connection=engine,
        use_jsonb=True,
    )

    # Or use with PGVectorStore wrapper (adds session isolation)
    from core.pgvector_store import PGVectorStore
    store = PGVectorStore(engine=engine, embeddings=my_embeddings)
"""

from __future__ import annotations

from threading import Lock

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.pool import QueuePool
from src.core.config import CloudSQLConfig, cloudsql_config
from src.core.logger import get_logger

logger = get_logger(__name__)

_lock = Lock()
_client: CloudSQLClient | None = None


def _get_connection_string(config: CloudSQLConfig) -> str:
    """
    Build SQLAlchemy connection string for psycopg3 (sync).

    If a schema other than 'public' is configured, it will be set
    in the search_path via connection options.

    Args:
        config: CloudSQLConfig instance.

    Returns:
        PostgreSQL connection string for SQLAlchemy.
    """
    import urllib.parse

    base_url = (
        f"postgresql+psycopg://{config.USER}:{config.PASSWORD}"
        f"@{config.HOST}:{config.PORT}/{config.DATABASE}"
    )

    # Add schema to search_path if not public
    if config.SCHEMA and config.SCHEMA != "public":
        options = urllib.parse.quote(f"-csearch_path={config.SCHEMA},public")
        return f"{base_url}?options={options}"

    return base_url


def _get_async_connection_string(config: CloudSQLConfig) -> str:
    """
    Build SQLAlchemy async connection string for asyncpg.

    Args:
        config: CloudSQLConfig instance.

    Returns:
        PostgreSQL async connection string for SQLAlchemy.
    """
    return (
        f"postgresql+asyncpg://{config.USER}:{config.PASSWORD}"
        f"@{config.HOST}:{config.PORT}/{config.DATABASE}"
    )


class CloudSQLClient:
    """
    Minimal Cloud SQL PostgreSQL client.

    Provides only a singleton SQLAlchemy engine with connection pooling.
    All vector operations should use LangChain's PGVector class.
    """

    def __init__(self, config: CloudSQLConfig | None = None):
        """
        Initialize Cloud SQL client.

        Args:
            config: CloudSQLConfig instance. Uses default if not provided.
        """
        self._config = config or cloudsql_config
        self._engine: Engine | None = None
        self._initialized = False
        self._init_lock = Lock()

    def _initialize(self) -> None:
        """Initialize the database engine (lazy initialization)."""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            logger.info(
                f"Initializing Cloud SQL engine: "
                f"{self._config.HOST}:{self._config.PORT}/{self._config.DATABASE}"
                f" (schema: {self._config.SCHEMA})"
            )

            # Create engine with connection pooling
            self._engine = create_engine(
                _get_connection_string(self._config),
                poolclass=QueuePool,
                pool_size=self._config.POOL_SIZE,
                max_overflow=self._config.MAX_OVERFLOW,
                pool_timeout=self._config.POOL_TIMEOUT,
                pool_recycle=self._config.POOL_RECYCLE,
                pool_pre_ping=True,
            )

            # Verify pgvector extension is available
            try:
                with self._engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                    )
                    if not result.fetchone():
                        raise RuntimeError(
                            "pgvector extension is not installed. "
                            "A database admin must run: CREATE EXTENSION vector;"
                        )
                    conn.commit()
                logger.info("pgvector extension verified")
            except RuntimeError:
                raise
            except Exception as e:
                logger.error(f"Failed to verify database: {e}")
                raise

            self._initialized = True
            logger.info("Cloud SQL engine ready")

    @property
    def engine(self) -> Engine | None:
        """
        Get the SQLAlchemy engine.

        Returns:
            SQLAlchemy Engine instance with connection pooling.
        """
        self._initialize()
        return self._engine

    @property
    def config(self) -> CloudSQLConfig:
        """Get the configuration."""
        return self._config

    def close(self) -> None:
        """Dispose of the engine and close all connections."""
        if self._engine:
            self._engine.dispose()
            self._initialized = False
            logger.info("Cloud SQL engine disposed")


# =============================================================================
# Singleton Access
# =============================================================================


def get_cloudsql_client(config: CloudSQLConfig | None = None) -> CloudSQLClient:
    """
    Get the process-wide CloudSQLClient singleton.

    Args:
        config: Optional CloudSQLConfig. Uses default if not provided.
                Only used on first call to create the client.

    Returns:
        CloudSQLClient singleton instance.
    """
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            _client = CloudSQLClient(config)
    return _client


def reset_cloudsql_client() -> None:
    """
    Reset the singleton client (useful for testing).

    Closes existing connections and clears the singleton.
    """
    global _client
    with _lock:
        if _client is not None:
            _client.close()
            _client = None
            logger.info("Cloud SQL client reset")


if __name__ == "__main__":
    # Example usage
    client = get_cloudsql_client()
    engine = client.engine
    print("Cloud SQL engine initialized:", engine)
