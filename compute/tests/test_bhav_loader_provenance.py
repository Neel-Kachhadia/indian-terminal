from __future__ import annotations

import datetime

from db.bhav_loader import _parse_bhav_row, _parse_index_row, _parse_trade_date


EQUITY_CM_ROW = {
    "TradDt": "02-01-2024",
    "BizDt": "02-01-2024",
    "Sgmt": "CM",
    "SctySrs": "EQ",
    "TckrSymb": "RELIANCE",
    "OpnPric": "2500.00",
    "HghPric": "2550.00",
    "LwPric": "2480.00",
    "ClsPric": "2530.00",
    "TtlTradgVol": "1000000",
}

NIFTY50_INDEX_ROW = {
    "Index Name": "NIFTY 50",
    "Index Date": "02-01-2024",
    "Open Index Value": "21750.00",
    "High Index Value": "21850.00",
    "Low Index Value": "21700.00",
    "Closing Index Value": "21800.00",
    "Volume": "250000000",
}

NIFTY_BANK_INDEX_ROW = {
    "Index Name": "NIFTY BANK",
    "Index Date": "02-01-2024",
    "Open Index Value": "47000.00",
    "High Index Value": "47300.00",
    "Low Index Value": "46900.00",
    "Closing Index Value": "47150.00",
    "Volume": "50000000",
}


def test_equity_cm_row_is_labeled_equity_not_index() -> None:
    parsed = _parse_bhav_row(EQUITY_CM_ROW)
    assert parsed is not None
    assert parsed["symbol"] == "RELIANCE"
    assert parsed["source"] == "NSE_EQUITY_BHAVCOPY"


def test_nifty50_index_row_is_labeled_canonical_index_source() -> None:
    parsed = _parse_index_row(NIFTY50_INDEX_ROW)
    assert parsed is not None
    assert parsed["symbol"] == "NIFTY 50"
    assert parsed["source"] == "NSE_INDEX_BHAVCOPY"


def test_nifty_bank_index_row_is_captured_and_labeled_canonical() -> None:
    parsed = _parse_index_row(NIFTY_BANK_INDEX_ROW)
    assert parsed is not None
    assert parsed["symbol"] == "NIFTY BANK"
    assert parsed["source"] == "NSE_INDEX_BHAVCOPY"


def test_non_index_row_is_rejected_not_silently_mislabeled() -> None:
    row = dict(NIFTY50_INDEX_ROW)
    row["Index Name"] = "NIFTY IT"
    assert _parse_index_row(row) is None


def test_real_nse_finnifty_display_name_maps_to_canonical_symbol() -> None:
    # Real ind_close_all_*.csv "Index Name" spelling, confirmed 2026-08-20/21
    # downloads -- NOT the short "FINNIFTY" alias.
    row = dict(NIFTY50_INDEX_ROW)
    row["Index Name"] = "Nifty Financial Services"
    parsed = _parse_index_row(row)
    assert parsed is not None
    assert parsed["symbol"] == "FINNIFTY"
    assert parsed["source"] == "NSE_INDEX_BHAVCOPY"


def test_real_nse_midcpnifty_display_name_maps_to_canonical_symbol() -> None:
    # Real ind_close_all_*.csv spelling -- NOT the short "MIDCPNIFTY" alias.
    row = dict(NIFTY50_INDEX_ROW)
    row["Index Name"] = "Nifty Midcap Select"
    parsed = _parse_index_row(row)
    assert parsed is not None
    assert parsed["symbol"] == "MIDCPNIFTY"
    assert parsed["source"] == "NSE_INDEX_BHAVCOPY"


def test_legacy_cnx_nifty_name_maps_to_canonical_nifty_50() -> None:
    # Verified via real price continuity (2015-11-06 CNX Nifty close ->
    # 2015-11-09 Nifty 50 open), not name similarity.
    row = dict(NIFTY50_INDEX_ROW)
    row["Index Name"] = "CNX Nifty"
    parsed = _parse_index_row(row)
    assert parsed is not None
    assert parsed["symbol"] == "NIFTY 50"
    assert parsed["source"] == "NSE_INDEX_BHAVCOPY"


def test_legacy_sp_cnx_nifty_name_maps_to_canonical_nifty_50() -> None:
    row = dict(NIFTY50_INDEX_ROW)
    row["Index Name"] = "S&P CNX Nifty"
    parsed = _parse_index_row(row)
    assert parsed is not None
    assert parsed["symbol"] == "NIFTY 50"


def test_legacy_cnx_bank_name_maps_to_canonical_nifty_bank() -> None:
    row = dict(NIFTY_BANK_INDEX_ROW)
    row["Index Name"] = "CNX Bank"
    parsed = _parse_index_row(row)
    assert parsed is not None
    assert parsed["symbol"] == "NIFTY BANK"


def test_slash_separated_trade_date_is_parsed() -> None:
    # Real ind_close_all_09062014.csv uses "09/06/2014" (DD/MM/YYYY) --
    # crashed the full historical backfill before this fix.
    assert _parse_trade_date("09/06/2014") == datetime.date(2014, 6, 9)


def test_unrelated_broader_index_still_rejected() -> None:
    # "Nifty Midcap 100" is a real, different, untracked NSE index -- must not
    # be swept up by the Midcap Select alias.
    row = dict(NIFTY50_INDEX_ROW)
    row["Index Name"] = "NIFTY Midcap 100"
    assert _parse_index_row(row) is None
