"""Human-readable console rendering for the collector's interactive terminal.

Kept entirely separate from `JsonFormatter` in `logging_config.py`: this
formatter never touches the file handler, so durable structured logs are
unaffected by anything here. A `logging.LogRecord` optionally carries a
`console_meta` dict (set via `extra={"console_meta": {...}}` at a handful of
lifecycle call sites in `smart_collector.py`); records without it still get a
concise one-line rendering instead of a raw JSON blob.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any

DEFAULT_WIDTH = 100

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_COLORS = {
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "red": "\x1b[31m",
    "cyan": "\x1b[36m",
}

# tag -> color name; anything absent renders with no color (neutral).
_TAG_COLOR = {
    "OK": "green",
    "SUCCESS": "green",
    "WAIT": "yellow",
    "WARN": "yellow",
    "SKIP": "yellow",
    "RETRY": "yellow",
    "ERROR": "red",
    "FAIL": "red",
    "ABANDONED": "red",
    "RUN": "cyan",
    "START": "cyan",
    "WRITER": "cyan",
    "SCAN": "cyan",
    "IDLE": "cyan",
}

# Levelname -> default tag when a call site supplies no console_meta.
_LEVEL_TAG = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARN",
    "ERROR": "ERROR",
    "CRITICAL": "ERROR",
}


def supports_unicode(stream: Any) -> bool:
    """False when the stream's encoding can't represent the box-drawing/
    arrow glyphs this module uses (e.g. a legacy Windows console still on
    the cp1252 code page). Reproduced directly: printing "↳"/"…"/"╭" through
    such a stream raises UnicodeEncodeError and crashes the whole collector
    process -- strictly worse than the plain JSON output this formatter
    replaces, so glyph choice must degrade to ASCII rather than risk that."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "↳…╭╮╰╯│─".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def supports_color(stream: Any) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("BHARATTERMINAL_FORCE_COLOR"):
        return True
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    return True


def terminal_width(default: int = DEFAULT_WIDTH) -> int:
    try:
        return max(60, shutil.get_terminal_size(fallback=(default, 24)).columns)
    except OSError:
        return default


def truncate(text: str, width: int, *, unicode: bool = True) -> str:
    ellipsis = "…" if unicode else "..."
    text = text.replace("\n", " ")
    if len(text) <= width:
        return text
    if width <= len(ellipsis):
        return text[:width]
    return text[: width - len(ellipsis)] + ellipsis


def short_path(path: str, keep: int = 2, *, unicode: bool = True) -> str:
    """Last `keep` path components, e.g. `data\\terminal.db` instead of the
    full absolute path. Durable logs still record the full path; this is
    presentation-only."""
    parts = [p for p in path.replace("/", "\\").split("\\") if p]
    if len(parts) <= keep:
        return path
    prefix = "…\\" if unicode else "...\\"
    return prefix + "\\".join(parts[-keep:])


class HumanConsoleFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        color: bool | None = None,
        width: int | None = None,
        unicode: bool | None = None,
    ) -> None:
        super().__init__()
        self._color = color
        self._width = width
        self._unicode = unicode if unicode is not None else True

    def format(self, record: logging.LogRecord) -> str:
        meta: dict[str, Any] = getattr(record, "console_meta", None) or {}
        color = self._color if self._color is not None else False
        width = self._width or terminal_width()
        arrow = "↳" if self._unicode else "->"

        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        tag = str(meta.get("tag") or _LEVEL_TAG.get(record.levelname, record.levelname))
        category = str(meta.get("category") or "")
        message = truncate(record.getMessage(), max(20, width - 28), unicode=self._unicode)

        line = f"{ts}  {tag:<7}{category:<11}{message}" if category else f"{ts}  {tag:<7}{message}"

        if color:
            color_name = _TAG_COLOR.get(tag)
            if color_name:
                line = f"{_COLORS[color_name]}{line}{_RESET}"

        detail = meta.get("detail")
        lines = [line]
        if detail:
            text = truncate(str(detail), max(20, width - 12), unicode=self._unicode)
            lines.append(f"           {arrow} {text}")
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            first_line = exc_text.strip().splitlines()[-1] if exc_text.strip() else ""
            if first_line:
                text = truncate(first_line, max(20, width - 12), unicode=self._unicode)
                lines.append(f"           {arrow} {text}")
        return "\n".join(lines)


def render_banner(fields: dict[str, str], *, width: int | None = None, unicode: bool = True) -> str:
    """One compact boxed status header, shown once at interactive startup.
    Every line (top border, body, bottom border) is exactly `box_width`
    characters, computed from the border chars rather than a hand-counted
    literal, so it can't silently drift out of alignment at odd widths."""
    if not unicode:
        return _render_banner_ascii(fields, width=width)
    box_width = min(max(width or terminal_width(), 24), 78)
    inner = box_width - 3  # "│ " (2 chars) + inner + "│" (1 char) == box_width

    title = " BharatTerminal Smart Collector "
    dash_budget = box_width - 2 - len(title)  # minus the two corner chars
    if dash_budget < 1:
        title = truncate(title.strip(), box_width - 4)
        dash_budget = box_width - 2 - len(title)
    top = "╭" + title + "─" * max(0, dash_budget) + "╮"
    top = top[: box_width - 1] + "╮"  # guarantee exact width even if truncated oddly

    label_width = max((len(k) for k in fields), default=0)
    lines = [top]
    for key, value in fields.items():
        text = truncate(f"{key:<{label_width}}  {value}", inner)
        lines.append(f"│ {text:<{inner}}│")
    lines.append("╰" + "─" * (box_width - 2) + "╯")
    return "\n".join(lines)


def _render_banner_ascii(fields: dict[str, str], *, width: int | None = None) -> str:
    box_width = min(max(width or terminal_width(), 24), 78)
    inner = box_width - 3
    title = " BharatTerminal Smart Collector "
    dash_budget = box_width - 2 - len(title)
    if dash_budget < 1:
        title = truncate(title.strip(), box_width - 4, unicode=False)
        dash_budget = box_width - 2 - len(title)
    top = "+" + title + "-" * max(0, dash_budget) + "+"
    top = top[: box_width - 1] + "+"

    label_width = max((len(k) for k in fields), default=0)
    lines = [top]
    for key, value in fields.items():
        text = truncate(f"{key:<{label_width}}  {value}", inner, unicode=False)
        lines.append(f"| {text:<{inner}}|")
    lines.append("+" + "-" * (box_width - 2) + "+")
    return "\n".join(lines)
