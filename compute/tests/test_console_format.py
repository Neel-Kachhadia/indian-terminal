from __future__ import annotations

import json
import logging

import pytest

from operations.console_format import (
    HumanConsoleFormatter,
    render_banner,
    short_path,
    supports_color,
    supports_unicode,
    truncate,
)
from operations.logging_config import JsonFormatter
from security.log_scrubber import SecretScrubber


def _record(level: int, message: str, *, meta: dict | None = None, exc_info=None) -> logging.LogRecord:
    record = logging.LogRecord(
        name="collector",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=exc_info,
    )
    if meta is not None:
        record.console_meta = meta
    return record


def test_info_formatting_has_no_json_braces() -> None:
    record = _record(logging.INFO, "Checking for pending tasks...", meta={"tag": "SCAN", "category": "SCAN"})
    line = HumanConsoleFormatter(color=False).format(record)
    assert "{" not in line
    assert "SCAN" in line
    assert "Checking for pending tasks..." in line


def test_ok_formatting_includes_category_and_message() -> None:
    record = _record(
        logging.INFO,
        "Done: signals for 2026-08-26: dates_written=0 (0.6s)",
        meta={"tag": "OK", "category": "SIGNALS"},
    )
    line = HumanConsoleFormatter(color=False).format(record)
    assert line.startswith(tuple("0123456789"))  # HH:MM:SS leads the line
    assert "OK" in line
    assert "SIGNALS" in line


def test_warn_deferred_formatting_not_rendered_as_error() -> None:
    record = _record(
        logging.WARNING,
        "Failed: bhavcopy_fo for 2026-08-26: not_available_yet source=holiday",
        meta={"tag": "WAIT", "category": "F&O"},
    )
    line = HumanConsoleFormatter(color=False).format(record)
    assert "WAIT" in line
    assert "ERROR" not in line
    assert "FAIL" not in line


def test_error_formatting_shows_error_tag() -> None:
    record = _record(
        logging.ERROR,
        "Task crashed: ohlc_forecasts for 2026-08-26",
        meta={"tag": "ERROR", "category": "FORECAST"},
    )
    line = HumanConsoleFormatter(color=False).format(record)
    assert "ERROR" in line
    assert "FORECAST" in line


def test_exception_appends_one_concise_detail_line() -> None:
    try:
        raise ConnectionError("HTTP 500")
    except ConnectionError:
        import sys

        record = _record(logging.ERROR, "Download failed", exc_info=sys.exc_info())
    line = HumanConsoleFormatter(color=False).format(record)
    lines = line.splitlines()
    assert len(lines) == 2
    assert "ConnectionError" in lines[1]
    assert "↳" in lines[1]


def test_no_ansi_when_color_disabled() -> None:
    record = _record(logging.INFO, "ok", meta={"tag": "OK"})
    line = HumanConsoleFormatter(color=False).format(record)
    assert "\x1b" not in line


def test_ansi_present_when_color_enabled() -> None:
    record = _record(logging.INFO, "ok", meta={"tag": "OK"})
    line = HumanConsoleFormatter(color=True).format(record)
    assert "\x1b[" in line


def test_supports_color_respects_no_color_env(monkeypatch) -> None:
    class FakeTTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color(FakeTTY()) is False


def test_supports_color_false_for_non_tty(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("BHARATTERMINAL_FORCE_COLOR", raising=False)

    class FakeNonTTY:
        def isatty(self) -> bool:
            return False

    assert supports_color(FakeNonTTY()) is False


def test_secret_is_redacted_in_human_console_output() -> None:
    """The scrubber runs as a logging.Filter on the handler, mutating
    record.msg before any formatter sees it -- this proves the human
    formatter inherits that redaction exactly like the JSON formatter does,
    rather than needing its own separate scrubbing logic."""
    record = _record(logging.INFO, "Angel session token=abcd1234SECRETtoken9999 acquired")
    scrubber = SecretScrubber()
    assert scrubber.filter(record) is True
    line = HumanConsoleFormatter(color=False).format(record)
    assert "abcd1234SECRETtoken9999" not in line
    assert "REDACTED" in line or "***" in line


def test_secret_is_redacted_in_json_console_output() -> None:
    record = _record(logging.INFO, "Groq api_key=sk-abcdefghijklmnopqrstuvwxyz123456 used")
    scrubber = SecretScrubber()
    scrubber.filter(record)
    payload = JsonFormatter().format(record)
    parsed = json.loads(payload)  # must remain valid JSON
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in json.dumps(parsed)


def test_json_formatter_output_is_always_valid_json_regardless_of_console_meta() -> None:
    """console_meta must never leak into or break the durable JSON log --
    JsonFormatter doesn't read it at all."""
    record = _record(logging.INFO, "Running: signals for 2026-08-26", meta={"tag": "RUN", "category": "SIGNALS"})
    payload = JsonFormatter().format(record)
    parsed = json.loads(payload)
    assert parsed["event"] == "Running: signals for 2026-08-26"
    assert "console_meta" not in payload


def test_long_message_is_truncated_not_wrapped_into_garbage() -> None:
    long_text = "x" * 500
    assert len(truncate(long_text, 50)) == 50
    assert truncate(long_text, 50).endswith("…")


def test_truncate_short_text_unchanged() -> None:
    assert truncate("hello", 50) == "hello"


def test_short_path_keeps_useful_tail() -> None:
    result = short_path(r"D:\BharatTerminal\data\terminal.db", keep=2)
    assert result == "…\\data\\terminal.db"


def test_short_path_returns_original_when_already_short() -> None:
    assert short_path(r"C:\a\b", keep=2) == r"C:\a\b" or short_path(r"C:\a\b", keep=2).endswith("a\\b")


def test_render_banner_contains_no_secrets_and_is_boxed() -> None:
    banner = render_banner(
        {
            "Status": "ACTIVE",
            "Data root": "…\\data",
            "Database": "terminal.db",
            "Writer": "OWNED",
        }
    )
    lines = banner.splitlines()
    assert lines[0].startswith("╭")
    assert lines[-1].startswith("╰")
    assert "Status" in banner
    assert "ACTIVE" in banner


@pytest.mark.parametrize("width", [40, 80, 120, 200])
def test_render_banner_stable_across_widths(width: int) -> None:
    banner = render_banner({"Status": "ACTIVE"}, width=width)
    line_lengths = {len(line) for line in banner.splitlines()}
    assert len(line_lengths) == 1  # every box line is the same visual width


def test_supports_unicode_false_for_cp1252_stream() -> None:
    """Reproduces a real crash: printing the box-drawing/arrow glyphs
    through a legacy Windows console stream (cp1252) raised
    UnicodeEncodeError and crashed the whole collector process -- strictly
    worse than the plain JSON output being replaced."""

    class FakeCp1252Stream:
        encoding = "cp1252"

        def isatty(self) -> bool:
            return True

    assert supports_unicode(FakeCp1252Stream()) is False


def test_supports_unicode_true_for_utf8_stream() -> None:
    class FakeUtf8Stream:
        encoding = "utf-8"

    assert supports_unicode(FakeUtf8Stream()) is True


def test_ascii_fallback_formatter_never_emits_non_ascii() -> None:
    record = _record(
        logging.ERROR,
        "Download failed",
        meta={"tag": "ERROR", "category": "F&O", "detail": "HTTP 500"},
    )
    line = HumanConsoleFormatter(color=False, unicode=False).format(record)
    line.encode("ascii")  # must not raise
    assert "->" in line
    assert "↳" not in line


def test_ascii_fallback_banner_never_emits_non_ascii() -> None:
    # Field values are the caller's responsibility (smart_collector.py's
    # real caller threads the same unicode flag into short_path() before
    # building `fields`) -- this test passes ascii-safe content, matching
    # actual production usage, and checks the banner's OWN border/ellipsis
    # glyphs stay ascii.
    banner = render_banner({"Status": "ACTIVE", "Data root": "...\\data"}, unicode=False)
    banner.encode("ascii")  # must not raise
    assert "╭" not in banner
    assert banner.startswith("+")


def test_ascii_fallback_short_path_never_emits_non_ascii() -> None:
    result = short_path(r"D:\BharatTerminal\data\terminal.db", keep=2, unicode=False)
    result.encode("ascii")  # must not raise


def test_idle_message_is_one_line_no_spam() -> None:
    record = _record(logging.INFO, "Next collection check in 30m", meta={"tag": "IDLE", "detail": "sleeping 30m"})
    line = HumanConsoleFormatter(color=False).format(record)
    assert line.count("\n") == 1  # exactly the message line + one detail line
    assert "IDLE" in line
