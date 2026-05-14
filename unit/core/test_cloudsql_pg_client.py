"""Tests for Cloud SQL PostgreSQL client."""

from unittest.mock import MagicMock, patch

import pytest

from core.cloudsql_pg_client import (
    CloudSQLClient,
    _get_async_connection_string,
    _get_connection_string,
    get_cloudsql_client,
    reset_cloudsql_client,
)
from core.config import CloudSQLConfig


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton client before and after each test."""
    reset_cloudsql_client()
    yield
    reset_cloudsql_client()


@pytest.fixture
def mock_config():
    """Create a mock CloudSQLConfig for testing."""
    config = MagicMock(spec=CloudSQLConfig)
    config.HOST = "localhost"
    config.PORT = 5432
    config.DATABASE = "test_db"
    config.USER = "test_user"
    config.PASSWORD = "test_password"
    config.SCHEMA = "public"
    config.POOL_SIZE = 5
    config.MAX_OVERFLOW = 10
    config.POOL_TIMEOUT = 30
    config.POOL_RECYCLE = 1800
    return config


# =============================================================================
# Connection String Tests
# =============================================================================


class TestConnectionStrings:
    """Tests for connection string generation."""

    def test_get_connection_string_public_schema(self, mock_config):
        """Test connection string with public schema."""
        mock_config.SCHEMA = "public"
        conn_str = _get_connection_string(mock_config)

        assert "postgresql+psycopg://" in conn_str
        assert "test_user:test_password" in conn_str
        assert "localhost:5432" in conn_str
        assert "test_db" in conn_str
        assert "options" not in conn_str  # No schema options for public

    def test_get_connection_string_custom_schema(self, mock_config):
        """Test connection string with custom schema."""
        mock_config.SCHEMA = "custom_schema"
        conn_str = _get_connection_string(mock_config)

        assert "postgresql+psycopg://" in conn_str
        assert "options=" in conn_str
        assert "search_path" in conn_str

    def test_get_connection_string_empty_schema(self, mock_config):
        """Test connection string with empty schema defaults to no options."""
        mock_config.SCHEMA = ""
        conn_str = _get_connection_string(mock_config)

        assert "options" not in conn_str

    def test_get_async_connection_string(self, mock_config):
        """Test async connection string generation."""
        conn_str = _get_async_connection_string(mock_config)

        assert "postgresql+asyncpg://" in conn_str
        assert "test_user:test_password" in conn_str
        assert "localhost:5432" in conn_str
        assert "test_db" in conn_str


# =============================================================================
# CloudSQLClient Tests
# =============================================================================


class TestCloudSQLClient:
    """Tests for CloudSQLClient class."""

    def test_init_with_default_config(self):
        """Test client initialization with default config."""
        client = CloudSQLClient()

        assert client._config is not None
        assert client._engine is None
        assert client._initialized is False

    def test_init_with_custom_config(self, mock_config):
        """Test client initialization with custom config."""
        client = CloudSQLClient(config=mock_config)

        assert client._config == mock_config
        assert client._engine is None
        assert client._initialized is False

    def test_lazy_initialization(self, mock_config):
        """Test that engine is not created until accessed."""
        with patch("core.cloudsql_pg_client.create_engine") as mock_create_engine:
            client = CloudSQLClient(config=mock_config)

            # Engine should not be created yet
            mock_create_engine.assert_not_called()
            assert client._initialized is False

    @patch("core.cloudsql_pg_client.create_engine")
    def test_engine_property_initializes(self, mock_create_engine, mock_config):
        """Test that accessing engine property triggers initialization."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock the connect context manager
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        client = CloudSQLClient(config=mock_config)
        engine = client.engine

        assert engine == mock_engine
        assert client._initialized is True
        mock_create_engine.assert_called_once()

    @patch("core.cloudsql_pg_client.create_engine")
    def test_engine_only_initializes_once(self, mock_create_engine, mock_config):
        """Test that engine is only created once."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        client = CloudSQLClient(config=mock_config)

        # Access engine multiple times
        _ = client.engine
        _ = client.engine
        _ = client.engine

        # Should only create engine once
        mock_create_engine.assert_called_once()

    @patch("core.cloudsql_pg_client.create_engine")
    def test_pgvector_extension_created(self, mock_create_engine, mock_config):
        """Test that pgvector extension is created on initialization."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        client = CloudSQLClient(config=mock_config)
        _ = client.engine

        # Verify CREATE EXTENSION was called
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

    @patch("core.cloudsql_pg_client.create_engine")
    def test_close_disposes_engine(self, mock_create_engine, mock_config):
        """Test that close() disposes the engine."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        client = CloudSQLClient(config=mock_config)
        _ = client.engine  # Initialize

        client.close()

        mock_engine.dispose.assert_called_once()
        assert client._initialized is False

    def test_config_property(self, mock_config):
        """Test config property returns the configuration."""
        client = CloudSQLClient(config=mock_config)

        assert client.config == mock_config

    @patch("core.cloudsql_pg_client.create_engine")
    def test_initialization_failure_raises(self, mock_create_engine, mock_config):
        """Test that initialization failure raises exception."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Make connect raise an exception
        mock_engine.connect.side_effect = Exception("Connection failed")

        client = CloudSQLClient(config=mock_config)

        with pytest.raises(Exception, match="Connection failed"):
            _ = client.engine


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern functions."""

    @patch("core.cloudsql_pg_client.create_engine")
    def test_get_cloudsql_client_returns_singleton(self, mock_create_engine):
        """Test that get_cloudsql_client returns the same instance."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        client1 = get_cloudsql_client()
        client2 = get_cloudsql_client()

        assert client1 is client2

    @patch("core.cloudsql_pg_client.create_engine")
    def test_reset_cloudsql_client_clears_singleton(self, mock_create_engine):
        """Test that reset_cloudsql_client clears the singleton."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        client1 = get_cloudsql_client()
        _ = client1.engine  # Initialize

        reset_cloudsql_client()

        client2 = get_cloudsql_client()

        assert client1 is not client2

    @patch("core.cloudsql_pg_client.create_engine")
    def test_reset_closes_existing_client(self, mock_create_engine):
        """Test that reset closes the existing client."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        client = get_cloudsql_client()
        _ = client.engine  # Initialize

        reset_cloudsql_client()

        mock_engine.dispose.assert_called_once()

    def test_reset_when_no_client_exists(self):
        """Test that reset works even when no client exists."""
        # Should not raise any exception
        reset_cloudsql_client()
        reset_cloudsql_client()


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    @patch("core.cloudsql_pg_client.create_engine")
    def test_concurrent_initialization(self, mock_create_engine):
        """Test that concurrent access doesn't create multiple engines."""
        import threading

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        clients = []

        def get_client():
            client = get_cloudsql_client()
            clients.append(client)

        threads = [threading.Thread(target=get_client) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All clients should be the same instance
        assert all(c is clients[0] for c in clients)
