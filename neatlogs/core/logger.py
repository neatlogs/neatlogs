"""
Centralized logging configuration for Neatlogs SDK.

Provides structured logging with configurable levels and handlers.
"""

import logging
import os
import sys
from typing import Optional


# Create logger instance
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """
    Get or create the Neatlogs SDK logger.

    Returns:
        logging.Logger: Configured logger instance
    """
    global _logger

    if _logger is not None:
        return _logger

    # Create logger
    _logger = logging.getLogger("neatlogs")

    # Get log level from environment or default to INFO
    log_level_name = os.getenv("NEATLOGS_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    _logger.setLevel(log_level)

    # Only add handler if none exists (prevent duplicate handlers)
    if not _logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        _logger.addHandler(handler)

    # Prevent propagation to root logger
    _logger.propagate = False

    return _logger


def set_log_level(level: int) -> None:
    """
    Set the log level for the Neatlogs SDK logger.

    Args:
        level: logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger = get_logger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def enable_debug_logging() -> None:
    """Enable debug-level logging for troubleshooting."""
    set_log_level(logging.DEBUG)


def disable_logging() -> None:
    """Disable all logging output."""
    logger = get_logger()
    logger.disabled = True
