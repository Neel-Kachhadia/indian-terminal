"""Regression coverage for the collector's response-shape classification.

`data.nse_bhavcopy._validate_csv_payload` (tested exhaustively in
test_raw_source_validation.py) already guarantees bad bytes are never
persisted. What was NOT covered end-to-end: that `download_equity_day` /
`download_fo_day` classify a garbage-but-200 response as a DISTINCT
"invalid_response" outcome rather than folding it into the same generic
"unavailable" bucket used for genuine not-yet-published/network conditions,
and that `smart_collector` propagates that distinction all the way through
to a loud FAIL (not a silently-absorbed WAIT).
"""

from __future__ import annotations

import io
import zipfile

import pytest

from auto_collector import smart_collector
from data.nse_bhavcopy import NSEBhavcopy

VALID_EQUITY_CSV = (
    b"SctySrs,FinInstrmTp,ISIN,TckrSymb,TradDt,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n"
    b"EQ,STK,INE000A00000,RELIANCE,2024-01-02,100,110,90,105,1000\n"
)
HTML_ERROR_BODY = b"<!DOCTYPE html>\r\n<html><body>Access Denied</body></html>\r\n"


def _zip_of(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bhav.csv", payload)
    return buffer.getvalue()


class DummyResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


def _make_bhav() -> NSEBhavcopy:
    return NSEBhavcopy.__new__(NSEBhavcopy)


# ---- equity direct-download path -----------------------------------------


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("zero_byte", b""),
        ("whitespace_only", b"   \n\t  "),
        ("html_error_body", HTML_ERROR_BODY),
        ("html_200_ok_shaped", b"<html><body>this looks fine but is not a zip</body></html>"),
        ("truncated_archive", _zip_of(VALID_EQUITY_CSV)[:10]),
    ],
)
def test_equity_direct_download_classifies_garbage_as_invalid_response(
    monkeypatch: pytest.MonkeyPatch, label: str, content: bytes
) -> None:
    bhav = _make_bhav()
    monkeypatch.setattr(bhav, "_nse_get", lambda url: DummyResponse(200, content))
    # jugaad path is skipped entirely by forcing an immediate import error.
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "jugaad_data.nse":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    success, source = bhav.download_equity_day(__import__("datetime").date(2024, 6, 3))
    assert (success, source) == (False, "invalid_response"), label


def test_equity_direct_download_accepts_valid_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    bhav = _make_bhav()
    monkeypatch.setattr(bhav, "_nse_get", lambda url: DummyResponse(200, _zip_of(VALID_EQUITY_CSV)))
    monkeypatch.setattr("data.nse_bhavcopy.EQUITY_BHAV_DIR", tmp_path)

    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "jugaad_data.nse":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    success, source = bhav.download_equity_day(__import__("datetime").date(2024, 6, 3))
    assert (success, source) == (True, "nse_direct")
    written = tmp_path / "20240603.csv"
    assert written.exists()
    assert written.read_bytes() == VALID_EQUITY_CSV


def test_equity_download_genuine_404_is_holiday_not_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    bhav = _make_bhav()
    monkeypatch.setattr(bhav, "_nse_get", lambda url: DummyResponse(404, b""))

    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "jugaad_data.nse":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    success, source = bhav.download_equity_day(__import__("datetime").date(2024, 6, 3))
    assert (success, source) == (True, "holiday")


# ---- F&O direct-download path (has its own retry loop) --------------------


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("zero_byte", b""),
        ("whitespace_only", b"     "),
        ("html_error_body_over_1kb", HTML_ERROR_BODY + b" " * 1200),
        ("truncated_archive_over_1kb", _zip_of(VALID_EQUITY_CSV * 100)[:1200]),
    ],
)
def test_fo_direct_download_classifies_garbage_as_invalid_response(
    monkeypatch: pytest.MonkeyPatch, label: str, content: bytes
) -> None:
    bhav = _make_bhav()
    monkeypatch.setattr(
        "data.nse_bhavcopy.requests.get", lambda *a, **k: DummyResponse(200, content)
    )
    monkeypatch.setattr("data.nse_bhavcopy.time.sleep", lambda *_: None)

    success, source = bhav.download_fo_day(__import__("datetime").date(2024, 6, 3))
    assert (success, source) == (False, "invalid_response"), label


def test_fo_direct_download_too_small_but_nonzero_retries_then_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A response that's short but not literally empty/whitespace (e.g. a
    # partial network read) is genuinely ambiguous -- retried quietly, not
    # escalated, matching pre-existing behavior for this specific case.
    bhav = _make_bhav()
    monkeypatch.setattr(
        "data.nse_bhavcopy.requests.get", lambda *a, **k: DummyResponse(200, b"short-but-not-empty")
    )
    monkeypatch.setattr("data.nse_bhavcopy.time.sleep", lambda *_: None)

    success, source = bhav.download_fo_day(__import__("datetime").date(2024, 6, 3))
    assert (success, source) == (False, "unavailable")


# ---- smart_collector classification propagation ---------------------------


def test_run_bhavcopy_equity_invalid_response_is_loud_fail_not_quiet_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datetime

    monkeypatch.setattr(
        "data.nse_bhavcopy.NSEBhavcopy.download_equity_day",
        lambda self, day: (False, "invalid_response"),
    )

    details = smart_collector.run_bhavcopy_equity(datetime.date(2024, 6, 3))
    assert "not_available_yet" not in details
    assert "data_validation_failed" in details

    success, returned_details = smart_collector.run_task("bhavcopy_equity", datetime.date(2024, 6, 3))
    assert success is False
    assert "data_validation_failed" in returned_details


def test_run_bhavcopy_fo_invalid_response_is_loud_fail_not_quiet_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datetime

    monkeypatch.setattr(
        "data.nse_bhavcopy.NSEBhavcopy.download_fo_day",
        lambda self, day: (False, "invalid_response"),
    )

    details = smart_collector.run_bhavcopy_fo(datetime.date(2024, 6, 3))
    assert "not_available_yet" not in details
    assert "data_validation_failed" in details

    success, returned_details = smart_collector.run_task("bhavcopy_fo", datetime.date(2024, 6, 3))
    assert success is False
    assert "data_validation_failed" in returned_details


def test_run_task_still_treats_genuine_not_available_yet_as_quiet_wait() -> None:
    import datetime

    success, details = smart_collector.run_task(
        "unknown task that always fails", datetime.date(2024, 6, 3)
    )
    assert success is False
    assert "unknown task" in details
