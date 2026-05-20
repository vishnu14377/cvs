"""
Unit tests for local_directory_handler module.

Tests cover:
- Getting the local data directory
- Creating local data paths
- Creating temporary paths
- Cleanup functionality
- Listing files in data directories
- Input validation (empty keys, invalid characters)

Run with: pytest tests/unit/test_local_directory_handler.py -v
"""

import os
from unittest.mock import patch

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory and patch LOCAL_DATA_DIR."""
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()

    with patch.dict(os.environ, {"LOCAL_DATA_DIR": str(test_data_dir)}):
        yield test_data_dir


@pytest.fixture
def sample_files(temp_data_dir):
    """Create sample files in the temp data directory."""
    # Create a key directory with some files
    key_dir = temp_data_dir / "test_key"
    key_dir.mkdir()

    # Create some files
    (key_dir / "file1.pdf").write_text("PDF content 1")
    (key_dir / "file2.pdf").write_text("PDF content 2")
    (key_dir / "data.json").write_text("{}")

    # Create a subdirectory with files
    tmp_dir = key_dir / "tmp"
    tmp_dir.mkdir()
    (tmp_dir / "temp_file.txt").write_text("Temp content")

    return key_dir


# =============================================================================
# Test: get_local_data_dir
# =============================================================================


class TestGetLocalDataDir:
    """Tests for get_local_data_dir function."""

    def test_returns_default_data_dir(self):
        """Should return default data directory when env var not set."""
        from src.core.local_directory_handler import DEFAULT_DATA_DIR, get_local_data_dir

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            result = get_local_data_dir()
            assert result == DEFAULT_DATA_DIR

    def test_returns_env_var_data_dir(self, temp_data_dir):
        """Should return env var data directory when set."""
        from src.core.local_directory_handler import get_local_data_dir

        result = get_local_data_dir()
        assert result == temp_data_dir


# =============================================================================
# Test: get_local_data_path
# =============================================================================


class TestGetLocalDataPath:
    """Tests for get_local_data_path function."""

    def test_creates_directory_for_key(self, temp_data_dir):
        """Should create directory for unique key if it doesn't exist."""
        from src.core.local_directory_handler import get_local_data_path

        result = get_local_data_path("new_key")

        assert result == temp_data_dir / "new_key"
        assert result.exists()
        assert result.is_dir()

    def test_returns_file_path_with_filename(self, temp_data_dir):
        """Should return full file path when filename is provided."""
        from src.core.local_directory_handler import get_local_data_path

        result = get_local_data_path("my_key", "document.pdf")

        assert result == temp_data_dir / "my_key" / "document.pdf"
        # Directory should be created, but file doesn't exist
        assert result.parent.exists()
        assert not result.exists()

    def test_returns_directory_path_without_filename(self, temp_data_dir):
        """Should return directory path when filename is None."""
        from src.core.local_directory_handler import get_local_data_path

        result = get_local_data_path("job_123")

        assert result == temp_data_dir / "job_123"
        assert result.is_dir()

    def test_empty_key_raises_error(self, temp_data_dir):
        """Should raise ValueError for empty unique_key."""
        from src.core.local_directory_handler import get_local_data_path

        with pytest.raises(ValueError) as exc_info:
            get_local_data_path("")

        assert "must not be empty" in str(exc_info.value)

    def test_whitespace_key_raises_error(self, temp_data_dir):
        """Should raise ValueError for whitespace-only unique_key."""
        from src.core.local_directory_handler import get_local_data_path

        with pytest.raises(ValueError) as exc_info:
            get_local_data_path("   ")

        assert "must not be empty" in str(exc_info.value)

    def test_key_with_dotdot_raises_error(self, temp_data_dir):
        """Should raise ValueError for key containing '..'."""
        from src.core.local_directory_handler import get_local_data_path

        with pytest.raises(ValueError) as exc_info:
            get_local_data_path("../escape")

        assert "must not contain '..'" in str(exc_info.value)

    def test_strips_key_whitespace(self, temp_data_dir):
        """Should strip whitespace from key."""
        from src.core.local_directory_handler import get_local_data_path

        result = get_local_data_path("  valid_key  ")

        assert result == temp_data_dir / "valid_key"

    def test_numeric_key_converted_to_string(self, temp_data_dir):
        """Should accept numeric keys (converted to string)."""
        from src.core.local_directory_handler import get_local_data_path

        result = get_local_data_path(12345)  # type: ignore

        assert result == temp_data_dir / "12345"


# =============================================================================
# Test: get_local_temp_path
# =============================================================================


class TestGetLocalTempPath:
    """Tests for get_local_temp_path function."""

    def test_creates_tmp_subfolder_by_default(self, temp_data_dir):
        """Should create tmp subfolder by default."""
        from src.core.local_directory_handler import get_local_temp_path

        result = get_local_temp_path("job_key")

        assert result == temp_data_dir / "job_key" / "tmp"
        assert result.exists()
        assert result.is_dir()

    def test_creates_custom_subfolder(self, temp_data_dir):
        """Should create custom subfolder when specified."""
        from src.core.local_directory_handler import get_local_temp_path

        result = get_local_temp_path("job_key", subfolder="intermediate")

        assert result == temp_data_dir / "job_key" / "intermediate"
        assert result.exists()


# =============================================================================
# Test: cleanup_local_data
# =============================================================================


class TestCleanupLocalData:
    """Tests for cleanup_local_data function."""

    def test_removes_existing_directory(self, sample_files):
        """Should remove existing data directory."""
        from src.core.local_directory_handler import cleanup_local_data

        # Verify directory exists
        assert sample_files.exists()

        result = cleanup_local_data("test_key")

        assert result is True
        assert not sample_files.exists()

    def test_returns_true_for_nonexistent_directory(self, temp_data_dir):
        """Should return True for non-existent directory."""
        from src.core.local_directory_handler import cleanup_local_data

        result = cleanup_local_data("nonexistent_key")

        assert result is True

    def test_handles_cleanup_error(self, sample_files):
        """Should return False when cleanup fails."""
        from src.core.local_directory_handler import cleanup_local_data

        # Mock shutil.rmtree to raise an error
        with patch("src.core.local_directory_handler.shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = PermissionError("Permission denied")

            result = cleanup_local_data("test_key")

            assert result is False


# =============================================================================
# Test: list_local_files
# =============================================================================


class TestListLocalFiles:
    """Tests for list_local_files function."""

    def test_lists_all_files(self, sample_files):
        """Should list all files in the key directory."""
        from src.core.local_directory_handler import list_local_files

        result = list_local_files("test_key")

        # Should include both PDF and JSON files (but not subdirectories or their contents)
        filenames = [f.name for f in result]
        assert "file1.pdf" in filenames
        assert "file2.pdf" in filenames
        assert "data.json" in filenames

    def test_filters_by_pattern(self, sample_files):
        """Should filter files by glob pattern."""
        from src.core.local_directory_handler import list_local_files

        result = list_local_files("test_key", pattern="*.pdf")

        filenames = [f.name for f in result]
        assert "file1.pdf" in filenames
        assert "file2.pdf" in filenames
        assert "data.json" not in filenames

    def test_returns_empty_for_nonexistent_key(self, temp_data_dir):
        """Should return empty list for non-existent key directory."""
        from src.core.local_directory_handler import list_local_files

        result = list_local_files("nonexistent_key")

        assert result == []

    def test_recursive_pattern(self, sample_files):
        """Should support recursive patterns."""
        from src.core.local_directory_handler import list_local_files

        result = list_local_files("test_key", pattern="**/*.txt")

        filenames = [f.name for f in result]
        assert "temp_file.txt" in filenames


# =============================================================================
# Test: PROJECT_ROOT constant
# =============================================================================


class TestProjectRoot:
    """Tests for PROJECT_ROOT constant."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should point to an existing directory."""
        from src.core.local_directory_handler import PROJECT_ROOT

        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_pyproject(self):
        """PROJECT_ROOT should contain pyproject.toml."""
        from src.core.local_directory_handler import PROJECT_ROOT

        pyproject = PROJECT_ROOT / "pyproject.toml"
        assert pyproject.exists()
