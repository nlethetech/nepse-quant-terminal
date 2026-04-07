"""
Centralized logging configuration for NEPSE Quant Pro.

Provides:
- JSON-formatted rotating file logs (grep-able, 10 MB x 5 backups)
- Human-readable console output
"""

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


_CONSOLE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    name: str = "nepse_quant",
    log_dir: str = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure root-level logging for the application.

    Args:
        name: Logger (and log file) name.
        log_dir: Directory for log files (created if missing).
        level: Minimum log level.
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated backups to keep.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # --- Console handler (human-readable) ---
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT))
    logger.addHandler(console)

    # --- Rotating file handler (JSON) ---
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / f"{name}.jsonl",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_JSONFormatter())
    logger.addHandler(file_handler)

    return logger
