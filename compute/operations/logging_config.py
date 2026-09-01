from __future__ import annotations

import datetime as dt
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from threading import Lock
from typing import Any, TextIO

from operations.console_format import HumanConsoleFormatter, supports_color, supports_unicode
from security.log_scrubber import SecretScrubber, scrub_text


MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 5
_configured = False


class _NullStream:
    """A structurally-real, always-non-interactive stand-in for a missing
    `sys.stdout`/`sys.stderr`. Deliberately NOT `open(os.devnull, "w")`:
    opening Windows' NUL device can report `isatty() == True` (a real,
    verified platform quirk -- opening it is a character device, and
    Windows' isatty() check is device-type-based, not "is this an actual
    console" based), which would wrongly select the human-interactive
    console formatter for a stream that goes nowhere."""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return False

    def write(self, _value: str) -> int:
        return 0

    def flush(self) -> None:
        return None


def safe_stream_or_devnull(stream: TextIO | None) -> TextIO:
    """A frozen `--windowed` executable launched with no stdio redirection
    (e.g. Task Scheduler's InteractiveToken action, which provides no pipes)
    has `sys.stdout`/`sys.stderr` set to `None` by the interpreter -- there
    is no console to give it a real handle. Every caller below assumes a
    real stream (`.isatty()`, `.write()`, `.encoding`, wrapping in
    `logging.StreamHandler`); substituting a safe stand-in once, here, keeps
    that assumption true everywhere else instead of scattering `is not None`
    guards through logging/console-format code that also runs in the
    interactive/dev case where the stream is real."""
    if stream is not None:
        return stream
    return _NullStream()


_original_stdout = safe_stream_or_devnull(sys.stdout)
_original_stderr = safe_stream_or_devnull(sys.stderr)
_console_interactive = False
_console_unicode = True


def is_console_interactive() -> bool:
    """True only when the console stream handler is rendering the human
    formatter (real TTY, not forced to JSON). Callers use this to decide
    whether to additionally log presentation-only output like the startup
    banner -- checking `sys.stderr.isatty()` directly would not work once
    `configure_structured_logging` has redirected stdout/stderr through
    `StructuredPrintStream`, which always reports `isatty() == False`."""
    return _console_interactive


def console_supports_unicode() -> bool:
    return _console_unicode


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": scrub_text(record.getMessage()),
        }
        if record.exc_info:
            payload["exception"] = scrub_text(self.formatException(record.exc_info))
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


class StructuredPrintStream:
    def __init__(self, logger: logging.Logger, level: int, fallback: TextIO) -> None:
        self.logger = logger
        self.level = level
        self.fallback = fallback
        self._buffer = ""
        self._lock = Lock()

    def write(self, value: str) -> int:
        with self._lock:
            self._buffer += str(value)
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self.logger.log(self.level, "%s", line.rstrip("\r"))
        return len(value)

    def flush(self) -> None:
        with self._lock:
            if self._buffer.strip():
                self.logger.log(self.level, "%s", self._buffer.rstrip("\r"))
            self._buffer = ""
        self.fallback.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self.fallback, "encoding", None) or "utf-8"


def configure_structured_logging(log_path: Path, *, redirect_prints: bool = True) -> None:
    global _configured
    if _configured:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()
    scrubber = SecretScrubber()
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=MAX_LOG_BYTES,
        backupCount=LOG_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(scrubber)
    stream_handler = logging.StreamHandler(_original_stderr)
    # The file handler's JsonFormatter is durable/machine logging and must
    # never change. The console stream gets a human-readable formatter only
    # when a real interactive terminal is attached; a scheduled/headless run
    # (no TTY, or explicitly forced) keeps the same JSON on console too --
    # deterministic, parseable, and never emits ANSI into redirected output.
    console_as_json = os.environ.get("BHARATTERMINAL_LOG_CONSOLE", "").strip().lower() == "json"
    global _console_interactive, _console_unicode
    _console_interactive = not console_as_json and _original_stderr.isatty()
    _console_unicode = supports_unicode(_original_stderr)
    if _console_interactive:
        stream_handler.setFormatter(
            HumanConsoleFormatter(
                color=supports_color(_original_stderr),
                unicode=_console_unicode,
            )
        )
    else:
        stream_handler.setFormatter(formatter)
    stream_handler.addFilter(scrubber)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.setLevel(logging.INFO)
    if redirect_prints:
        sys.stdout = StructuredPrintStream(logging.getLogger("stdout"), logging.INFO, _original_stdout)
        sys.stderr = StructuredPrintStream(logging.getLogger("stderr"), logging.ERROR, _original_stderr)
    _configured = True
