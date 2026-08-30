"""Logging setup for the package.

Progress messages go through `logging`, never `print`. The CLI is the only
caller of `configure_logging`; library modules just call `get_logger(__name__)`
and inherit whatever handler the application installed (or none at all, when
imported from a notebook or a test).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ..paths import PACKAGE_NAME

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Logger for a module. `get_logger(__name__)` is the intended call."""
    return logging.getLogger(name or PACKAGE_NAME)


def configure_logging(
    level: int | str = logging.INFO, log_file: Path | None = None
) -> logging.Logger:
    """Installs a stderr handler on the package logger, optionally tee'd to a file.

    stderr rather than stdout so that a caller redirecting the report to a file
    still sees progress. Existing handlers are cleared first, which keeps the
    function idempotent across repeated CLI invocations in one process (the
    smoke test does exactly that).
    """
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
