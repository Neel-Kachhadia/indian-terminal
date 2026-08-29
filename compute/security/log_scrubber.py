from __future__ import annotations

import logging
import re
from typing import Any

SECRET_PATTERNS = [
    re.compile(r"\b(gsk_[A-Za-z0-9_-]{12,})\b"),
    re.compile(r"\b(eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"),
    re.compile(r"(?i)\b(api[_-]?key|password|passwd|totp|token|jwt|feedtoken)\s*[:=]\s*['\"]?[^'\"\s,}]+"),
]


class SecretScrubber(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = scrub_text(str(record.getMessage()))
        record.args = ()
        return True


def scrub_text(message: str) -> str:
    cleaned = message
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub(_redact_match, cleaned)
    return cleaned


def scrub_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if _is_secret_key(key) else scrub_payload(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_payload(item) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def install_log_scrubber() -> None:
    # uvicorn's own AccessFormatter (uvicorn/logging.py) unpacks
    # record.args as a positional 5-tuple (client_addr, method, full_path,
    # http_version, status_code). SecretScrubber.filter() flattens msg+args
    # for safe re-serialization -- attaching it to uvicorn's own loggers
    # empties those args before uvicorn's formatter ever reads them, crashing
    # every access-log line with "ValueError: not enough values to unpack".
    # Those loggers carry structured request metadata, not free-text that
    # needs secret scrubbing, so they're excluded rather than reworked.
    scrubber = SecretScrubber()
    root = logging.getLogger()
    root.addFilter(scrubber)
    for logger_name in logging.root.manager.loggerDict:
        if logger_name == "uvicorn" or logger_name.startswith("uvicorn."):
            continue
        logging.getLogger(logger_name).addFilter(scrubber)


def _is_secret_key(key: Any) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in ("password", "api_key", "apikey", "totp", "token", "secret", "jwt"))


def _redact_match(match: re.Match[str]) -> str:
    text = match.group(0)
    if ":" in text:
        return f"{text.split(':', 1)[0]}: [REDACTED]"
    if "=" in text:
        return f"{text.split('=', 1)[0]}=[REDACTED]"
    return "[REDACTED]"
