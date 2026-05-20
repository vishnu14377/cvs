"""
Centralized logging configuration for the CareConnect ADR AI Agent.

This module provides:
- Consistent logging format across all modules
- Configurable log levels via environment variables
- Helper functions for creating module-specific loggers

Usage:
    from src.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("An error occurred", exc_info=True)

Environment Variables:
    LOG_LEVEL: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              Default: INFO
    LOG_FORMAT: Set the log format (simple, detailed, json)
              Default: detailed
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Configuration from environment
# =============================================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT_TYPE = os.environ.get("LOG_FORMAT", "detailed").lower()


# =============================================================================
# Log Formatters
# =============================================================================

# Simple format: timestamp - level - message
SIMPLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Detailed format: timestamp - module - level - message
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


class PHIRedactingFormatter(logging.Formatter):
    """Wraps another formatter and redacts PHI from log output."""

    def __init__(self, base_formatter: logging.Formatter):
        super().__init__()
        self._base = base_formatter

    def format(self, record: logging.LogRecord) -> str:
        output = self._base.format(record)
        try:
            from src.api.validation.phi_redaction import redact_phi

            return redact_phi(output)
        except ImportError:
            return output


def _get_formatter() -> logging.Formatter:
    """Get the appropriate formatter based on LOG_FORMAT_TYPE."""
    if LOG_FORMAT_TYPE == "json":
        return JsonFormatter()
    elif LOG_FORMAT_TYPE == "simple":
        return logging.Formatter(SIMPLE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    else:  # detailed (default)
        return logging.Formatter(DETAILED_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


# =============================================================================
# Logger Setup
# =============================================================================


def setup_logging() -> None:
    """
    Configure the root logger with the appropriate handler and formatter.

    Call this once at application startup to configure logging globally.
    """
    root_logger = logging.getLogger()

    # Avoid adding handlers multiple times
    if root_logger.handlers:
        return

    # Set the log level
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(PHIRedactingFormatter(_get_formatter()))

    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    Ensures logging is configured before returning the logger.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance

    Example:
        from src.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hello, world!")
    """
    setup_logging()
    return logging.getLogger(name)


# =============================================================================
# Initialize logging on module import
# =============================================================================

setup_logging()
