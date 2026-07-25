from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def local_logger(name: str, directory: Path) -> logging.Logger:
    logger = logging.getLogger(f"visionmodelquest.explorer.{name}")
    if logger.handlers:
        return logger
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    handler = RotatingFileHandler(
        directory / f"{name}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
