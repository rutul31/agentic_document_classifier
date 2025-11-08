"""Utility functions for structured logging across the application."""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(
    name: str, level: int = logging.INFO, fmt: Optional[str] = None
) -> logging.Logger:
    """Return a configured logger instance.

    Args:
        name: Name of the logger to create or fetch.
        level: Logging level to apply.
        fmt: Optional logging format string. A sensible default is applied when
            omitted.

    Returns:
        A configured :class:`logging.Logger` instance ready for use.
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


__all__ = ["get_logger"]
