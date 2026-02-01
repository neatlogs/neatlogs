"""
Centralized logging configuration for Neatlogs SDK.

Provides structured logging with configurable levels and handlers.
"""

import logging
import os
import sys
from typing import Optional


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

    _logger = logging.getLogger("neatlogs")

    log_level_name = os.getenv("NEATLOGS_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    _logger.setLevel(log_level)

    if not _logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

        log_file = os.getenv("NEATLOGS_LOG_FILE")
        if log_file:
            try:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                _logger.addHandler(file_handler)
            except Exception as e:
                _logger.warning(f"Failed to setup file logging to {log_file}: {e}")

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
