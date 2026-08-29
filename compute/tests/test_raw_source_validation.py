from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from data.nse_bhavcopy import (
    NSEBhavcopy,
    RawSourceValidationError,
    _validate_csv_payload,
)


VALID_EQUITY_CSV = (
    b"SctySrs,FinInstrmTp,ISIN,TckrSymb,TradDt,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n"
    b"EQ,STK,INE000A00000,RELIANCE,2024-01-02,100,110,90,105,1000\n"
)

HTML_200_ERROR_BODY = b"<!DOCTYPE html>\r\n<html><body>Access Denied</body></html>\r\n"
HTML_403_CHALLENGE = b"<html><head><title>403 Forbidden</title></head><body>blocked</body></html>"
HTML_500_PAGE = b"<HTML><BODY>Internal Server Error</BODY></HTML>"
EMPTY_BODY = b""
MALFORMED_CSV = b"this is not delimited text at all just one line"


class DummyResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        (HTML_200_ERROR_BODY, "200_html_error_body"),
        (HTML_403_CHALLENGE, "403_html_challenge_page"),
        (HTML_500_PAGE, "500_html_page"),
        (EMPTY_BODY, "zero_byte_body"),
        (MALFORMED_CSV, "malformed_non_csv"),
    ],
)
def test_bad_payload_rejected_before_persisting(payload: bytes, label: str) -> None:
    with pytest.raises(RawSourceValidationError):
        _validate_csv_payload(payload, context=label)


def test_valid_csv_payload_accepted() -> None:
    _validate_csv_payload(VALID_EQUITY_CSV, context="valid.csv")  # must not raise


def test_extract_first_csv_never_writes_html_error_page(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bhav.csv", HTML_200_ERROR_BODY)

    bhav = NSEBhavcopy.__new__(NSEBhavcopy)  # skip __init__ (no network/session needed)
    target = tmp_path / "20240102.csv"
    with pytest.raises(RawSourceValidationError):
        bhav._extract_first_csv(buffer.getvalue(), target)
    assert not target.exists()


def test_extract_first_csv_persists_valid_csv(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bhav.csv", VALID_EQUITY_CSV)

    bhav = NSEBhavcopy.__new__(NSEBhavcopy)
    target = tmp_path / "20240102.csv"
    bhav._extract_first_csv(buffer.getvalue(), target)
    assert target.read_bytes() == VALID_EQUITY_CSV


def test_move_to_standard_name_rejects_zero_byte_file(tmp_path: Path) -> None:
    source = tmp_path / "jugaad_output.csv"
    source.write_bytes(EMPTY_BODY)
    target = tmp_path / "20240102.csv"

    bhav = NSEBhavcopy.__new__(NSEBhavcopy)
    with pytest.raises(RawSourceValidationError):
        bhav._move_to_standard_name(str(source), target)
    assert not target.exists()
    assert source.exists()  # source untouched, never silently replaced
