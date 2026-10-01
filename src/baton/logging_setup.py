"""Logging with mandatory redaction.

The redaction lives in a `logging.Filter` attached to the *handlers*, so it
applies to every record that reaches them, including ones emitted by
third-party libraries (httpx, uvicorn) that know nothing about Baton.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import ensure_private_dir
from .redact import get_redactor

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        redactor = get_redactor()
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not kill logging
            message = str(record.msg)
        record.msg = redactor.scrub(message)
        record.args = None
        if record.exc_info:
            # Render the traceback now so it can be scrubbed, then drop the live
            # exception object so no formatter can re-render the raw version.
            record.exc_text = redactor.scrub(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redactor.scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = redactor.scrub(record.stack_info)
        return True


def setup_logging(log_dir: Path | None, *, level: str = "INFO", console: bool = False) -> None:
    """Configure root logging. Safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(_FORMAT)
    redacting = RedactingFilter()

    if log_dir is not None:
        ensure_private_dir(log_dir)
        log_path = log_dir / "baton.log"
        # Size-capped rotation: a runaway retry loop must not fill the disk.
        file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redacting)
        root.addHandler(file_handler)
        if os.name == "posix":
            try:
                log_path.chmod(0o600)
            except OSError:
                pass

    if console or log_dir is None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(redacting)
        stream_handler.setLevel(logging.INFO if console else logging.WARNING)
        root.addHandler(stream_handler)

    # httpx logs every request line at INFO. URLs never carry keys (adapters use
    # headers), but the noise is useless and DEBUG could expose more.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
