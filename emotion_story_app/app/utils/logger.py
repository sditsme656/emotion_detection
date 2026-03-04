"""Logging configuration."""

from __future__ import annotations

import logging


def configure_logger(name: str = "emotion_story", level: int = logging.INFO) -> logging.Logger:
    """Create/reuse logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
