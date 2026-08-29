from __future__ import annotations

import logging

import pytest

from security.log_scrubber import install_log_scrubber


def _make_access_record(logger_name: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=logger_name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:52573", "GET", "/health", "1.1", 200),
        exc_info=None,
    )


def test_scrubber_never_breaks_uvicorn_access_formatter() -> None:
    from uvicorn.logging import AccessFormatter

    logger_name = "uvicorn.access"
    logger = logging.getLogger(logger_name)
    logger.filters.clear()

    install_log_scrubber()

    record = _make_access_record(logger_name)
    assert logger.filter(record)  # runs every attached filter, mutates in place if broken

    # Pre-fix: SecretScrubber flattened record.args to () for every existing
    # logger including uvicorn.access, so AccessFormatter's 5-tuple unpack
    # raised ValueError here. Must not raise post-fix.
    formatted = AccessFormatter().format(record)
    assert "GET" in formatted and "/health" in formatted and "200" in formatted


def test_scrubber_still_applies_to_non_uvicorn_loggers() -> None:
    logger_name = "bharatterminal.some_module"
    logger = logging.getLogger(logger_name)
    logger.filters.clear()

    install_log_scrubber()

    record = logging.LogRecord(
        name=logger_name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=%s",
        args=("super-secret-value",),
        exc_info=None,
    )
    assert logger.filter(record)
    assert record.args == ()
    assert "[REDACTED]" in record.msg
    assert "super-secret-value" not in record.msg


@pytest.mark.parametrize("logger_name", ["uvicorn", "uvicorn.error", "uvicorn.access"])
def test_scrubber_skips_all_uvicorn_loggers(logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    logger.filters.clear()

    install_log_scrubber()

    assert logger.filters == []
