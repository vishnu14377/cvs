"""
Local directory handler for managing local file storage.

Provides utilities for creating and managing local directories that mirror
the GCS folder structure. This enables consistent file organization between
local development and cloud storage.

Directory Structure:
    {PROJECT_ROOT}/data/{unique_key}/[filename]

Example:
    >>> from src.core.local_directory_handler import get_local_data_path
    >>>
    >>> # Get path for a job
    >>> job_dir = get_local_data_path("job123")
    >>> # Returns: Path("data/job123")
    >>>
    >>> # Get path for a specific file
    >>> file_path = get_local_data_path("job123", "document.pdf")
    >>> # Returns: Path("data/job123/document.pdf")
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.core.logger import get_logger

logger = get_logger(__name__)

# Project root directory (where pyproject.toml is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default local data directory
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def get_local_data_dir() -> Path:
    """
    Get the base local data directory.

    Can be overridden via LOCAL_DATA_DIR environment variable.

    Returns:
        Path to the local data directory.
    """
    return Path(os.environ.get("LOCAL_DATA_DIR", str(DEFAULT_DATA_DIR)))


def get_local_data_path(unique_key: str, filename: str | None = None) -> Path:
    """
    Get local data path that mirrors GCS folder structure.

    Creates folder structure: data/{unique_key}/[filename]
    The directory is created automatically if it doesn't exist.

    Args:
        unique_key: Unique identifier for the processing job (e.g., session ID, document ID)
        filename: Optional filename to append to the path

    Returns:
        Path object for the local data directory or file

    Raises:
        ValueError: If unique_key is empty or contains invalid characters

    Example:
        >>> get_local_data_path("job123")
        Path("data/job123")
        >>> get_local_data_path("job123", "document.pdf")
        Path("data/job123/document.pdf")
    """
    # Validate unique_key
    unique_key = str(unique_key).strip()
    if not unique_key:
        raise ValueError("unique_key must not be empty")
    if ".." in unique_key:
        raise ValueError("unique_key must not contain '..'")

    base_path = get_local_data_dir() / unique_key

    # Create directory if it doesn't exist
    base_path.mkdir(parents=True, exist_ok=True)

    if filename:
        return base_path / filename
    return base_path


def get_local_temp_path(unique_key: str, subfolder: str = "tmp") -> Path:
    """
    Get local temporary path for intermediate files.

    Creates folder structure: data/{unique_key}/{subfolder}/

    Args:
        unique_key: Unique identifier for the processing job
        subfolder: Name of the temp subfolder (default: "tmp")

    Returns:
        Path to the temporary directory
    """
    temp_path = get_local_data_dir() / unique_key / subfolder
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


def cleanup_local_data(unique_key: str) -> bool:
    """
    Remove all local data for a specific key.

    Use with caution - this permanently deletes all files under the key's directory.

    Args:
        unique_key: Unique identifier for the processing job

    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        data_path = get_local_data_dir() / unique_key
        if data_path.exists():
            shutil.rmtree(data_path)
            logger.info("Cleaned up local data directory: %s", data_path)
            return True
        return True  # Directory doesn't exist, nothing to clean
    except Exception as e:
        logger.error("Failed to cleanup local data for key '%s': %s", unique_key, e)
        return False


def list_local_files(unique_key: str, pattern: str = "*") -> list[Path]:
    """
    List files in a local data directory.

    Args:
        unique_key: Unique identifier for the processing job
        pattern: Glob pattern for filtering files (default: "*" for all files)

    Returns:
        List of Path objects for matching files
    """
    data_path = get_local_data_dir() / unique_key
    if not data_path.exists():
        return []
    return list(data_path.glob(pattern))


__all__ = [
    "PROJECT_ROOT",
    "get_local_data_dir",
    "get_local_data_path",
    "get_local_temp_path",
    "cleanup_local_data",
    "list_local_files",
]
