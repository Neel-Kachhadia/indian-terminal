import csv
import datetime
import io
import time
import zipfile
from collections.abc import Callable
from typing import Any

import requests

from db.duckdb_client import DB_WRITE_LOCK, get_connection

BHAV_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)
INDEX_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{date}.csv"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com",
}


def download_bhav(date: datetime.date) -> list[dict[str, Any]] | None:
    response = None
    try:
        response = requests.get(
            BHAV_URL.format(date=date.strftime("%Y%m%d")),
            headers=NSE_HEADERS,
            timeout=30,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return _parse_bhav_zip(response.content)
    finally:
        if response is not None:
            time.sleep(2)


def download_index_bhav(date: datetime.date) -> list[dict[str, Any]] | None:
    response = None
    try:
        response = requests.get(
            INDEX_URL.format(date=date.strftime("%d%m%Y")),
            headers=NSE_HEADERS,
            timeout=30,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return _parse_index_csv(response.text)
    finally:
        if response is not None:
            time.sleep(2)


def get_missing_dates(
    from_date: datetime.date,
    to_date: datetime.date,
) -> list[datetime.date]:
    if from_date > to_date:
        return []

    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT DISTINCT date
            FROM ohlcv
            WHERE symbol IN ('NIFTY', 'NIFTY 50')
              AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            """,
            [from_date.isoformat(), to_date.isoformat()],
        ).fetchall()
    finally:
        con.close()

    present_dates = {_coerce_date(row[0]) for row in rows}
    dates: list[datetime.date] = []
    current = from_date
    one_day = datetime.timedelta(days=1)

    while current <= to_date:
        if current.weekday() < 5 and current not in present_dates:
            dates.append(current)
        current += one_day

    return dates


def load_date_range(
    from_date: datetime.date,
    to_date: datetime.date,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, int]:
    result = {"downloaded": 0, "skipped": 0, "errors": 0}

    for date in get_missing_dates(from_date, to_date):
        try:
            rows = download_index_bhav(date)
            if not rows:
                result["skipped"] += 1
                _notify_progress(result, progress_callback)
                continue

            _insert_ohlcv_rows(rows)
            result["downloaded"] += 1
        except Exception as error:
            result["errors"] += 1
            print(f"bhav: {date.isoformat()} failed: {error}")

        _notify_progress(result, progress_callback)

    return result


def _parse_bhav_zip(content: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_name = next(
            name for name in archive.namelist() if name.lower().endswith(".csv")
        )
        with archive.open(csv_name) as csv_file:
            text_file = io.TextIOWrapper(csv_file, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_file)
            for raw_row in reader:
                parsed = _parse_bhav_row(raw_row)
                if parsed is not None:
                    rows.append(parsed)

    return rows


def _parse_index_csv(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(content))

    for raw_row in reader:
        parsed = _parse_index_row(raw_row)
        if parsed is not None:
            rows.append(parsed)

    return rows


def _parse_bhav_row(raw_row: dict[str, str | None]) -> dict[str, Any] | None:
    row = {str(key).strip(): value for key, value in raw_row.items()}

    if str(row.get("SctySrs", "")).strip().upper() != "EQ":
        return None

    try:
        open_price = _positive_float(row.get("OpnPric"))
        high_price = _positive_float(row.get("HghPric"))
        low_price = _positive_float(row.get("LwPric"))
        close_price = _positive_float(row.get("ClsPric"))
    except ValueError:
        return None

    return {
        "symbol": str(row["TckrSymb"]).strip().upper(),
        "date": _parse_trade_date(str(row["TradDt"])).isoformat(),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": _volume(row.get("TtlTradgVol")),
        "source": "bhav",
    }


def _parse_index_row(raw_row: dict[str, str | None]) -> dict[str, Any] | None:
    row = {str(key).strip(): value for key, value in raw_row.items()}
    symbol = str(row.get("Index Name", "")).strip().upper()
    if symbol != "NIFTY 50":
        return None

    try:
        open_price = _positive_float(row.get("Open Index Value"))
        high_price = _positive_float(row.get("High Index Value"))
        low_price = _positive_float(row.get("Low Index Value"))
        close_price = _positive_float(row.get("Closing Index Value"))
    except ValueError:
        return None

    return {
        "symbol": symbol,
        "date": _parse_trade_date(str(row["Index Date"])).isoformat(),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": _volume(row.get("Volume")),
        "source": "bhav",
    }


def _insert_ohlcv_rows(rows: list[dict[str, Any]]) -> None:
    values = [
        (
            row["symbol"],
            row["date"],
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
            row["source"],
        )
        for row in rows
    ]

    with DB_WRITE_LOCK:
        con = get_connection()
        try:
            con.executemany(
                """
                INSERT OR IGNORE INTO ohlcv
                    (symbol, date, open, high, low, close, volume, source)
                VALUES (?, CAST(? AS DATE), ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        finally:
            con.close()


def _positive_float(value: str | None) -> float:
    if value is None:
        raise ValueError("missing price")

    parsed = float(value.strip().replace(",", ""))
    if parsed <= 0:
        raise ValueError("zero price")

    return parsed


def _volume(value: str | None) -> int:
    if value is None or value.strip() == "":
        return 0

    return int(float(value.strip().replace(",", "")))


def _parse_trade_date(value: str) -> datetime.date:
    cleaned = value.strip()
    for date_format in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue

    raise ValueError(f"unsupported trade date: {value}")


def _coerce_date(value: Any) -> datetime.date:
    if isinstance(value, datetime.date):
        return value

    return datetime.date.fromisoformat(str(value))


def _notify_progress(
    result: dict[str, int],
    progress_callback: Callable[[dict[str, int]], None] | None,
) -> None:
    if progress_callback is not None:
        progress_callback(dict(result))
