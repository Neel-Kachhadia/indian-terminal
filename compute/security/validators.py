from __future__ import annotations

import re
from datetime import date

from fastapi import HTTPException

VALID_SYMBOL = re.compile(r"^[A-Z0-9&\-. ]{1,30}$")
VALID_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_EXPIRY = re.compile(r"^\d{2}[A-Z]{3}\d{4}$")


def validate_symbol(value: str) -> str:
    symbol = str(value or "").strip().upper()
    if not VALID_SYMBOL.fullmatch(symbol):
        raise ValueError("Invalid symbol")
    return symbol


def validate_date(value: str) -> str:
    text = str(value or "").strip()
    if not VALID_DATE.fullmatch(text):
        raise ValueError("Invalid date")
    date.fromisoformat(text)
    return text


def validate_expiry(value: str) -> str:
    expiry = str(value or "").strip().upper()
    if not VALID_EXPIRY.fullmatch(expiry):
        raise ValueError("Invalid expiry")
    return expiry


def http_symbol(value: str) -> str:
    try:
        return validate_symbol(value)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

def http_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return validate_date(value)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
