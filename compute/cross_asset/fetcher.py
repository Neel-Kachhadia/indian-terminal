from __future__ import annotations

import copy
import datetime as _dt
import math
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import Lock
from typing import Any

import requests as _requests


TICKERS = {
    "usdinr": "USDINR=X",
    "brent": "BZ=F",
    "dow_fut": "YM=F",
    "sp500": "ES=F",
    "us_10y": "^TNX",
    "vix_us": "^VIX",
    "gold": "GC=F",
    "nikkei": "^N225",
    "hangseng": "^HSI",
    "sgx_nifty": "^NSEI",
}

_YFINANCE_TIMEOUT_SECONDS = 15
_STALE_AFTER_SECONDS = 30 * 60
_LAST_GOOD_SNAPSHOT: dict[str, Any] | None = None
_LAST_GOOD_LOCK = Lock()


def fetch_cross_asset_snapshot(nifty_spot: float = 0.0) -> dict[str, Any]:
    """
    Fetch cross-asset market data in batched yfinance calls.

    yfinance is delayed for most instruments. GIFT NIFTY has no reliable free
    source here, so it remains null by design.
    """
    try:
        import yfinance as yf
    except ImportError as error:
        return _failed_snapshot(nifty_spot, f"yfinance unavailable: {error}")

    session = _nse_session()
    usdinr_data = (
        fetch_usdinr_nse(session)
        or fetch_usdinr_frankfurter()
        or fetch_usdinr_rbi()
        or _empty_usdinr()
    )

    try:
        daily, intraday = _download_with_timeout(yf)
    except Exception as error:
        return _failed_snapshot(nifty_spot, f"yfinance download failed: {error}")

    quotes = {
        name: _quote_from_frames(ticker, daily, intraday)
        for name, ticker in TICKERS.items()
        if name != "usdinr"
    }
    if not any(_valid_price(quote.get("price")) for quote in quotes.values()):
        return _failed_snapshot(nifty_spot, "yfinance returned no usable prices")

    india_vix = _fetch_india_vix()
    sgx_nifty = _with_nifty_premium(quotes["sgx_nifty"], nifty_spot)

    snapshot: dict[str, Any] = {
        "usdinr": usdinr_data,
        "brent": quotes["brent"],
        "dow_futures": quotes["dow_fut"],
        "sp500_futures": quotes["sp500"],
        "us_10y_yield": _yield_quote(quotes["us_10y"]),
        "gold": quotes["gold"],
        "nikkei": quotes["nikkei"],
        "hangseng": quotes["hangseng"],
        "vix_us": quotes["vix_us"],
        "sgx_nifty": sgx_nifty,
        "india_vix": india_vix,
        "gift_nifty": {
            "price": None,
            "premium": None,
            "premium_pct": None,
        },
        "fetched_at": _utc_now().isoformat(),
    }
    snapshot["staleness_warning"] = any(
        _asset_is_stale(snapshot.get(key))
        for key in (
            "usdinr",
            "brent",
            "dow_futures",
            "sp500_futures",
            "us_10y_yield",
            "gold",
            "nikkei",
            "sgx_nifty",
        )
    )

    _store_last_good(snapshot)
    return snapshot


def compute_premarket_bias(
    snapshot: dict[str, Any],
    nifty_spot: float,
) -> dict[str, Any]:
    """
    Compute additive pre-market bias score from global signals.
    """
    score = 0.0
    signals: list[str] = []

    dow_pct = _number_at(snapshot, "dow_futures", "change_pct")
    if dow_pct is not None:
        if dow_pct > 0.3:
            score += 20
            signals.append(f"Dow futures {dow_pct:+.1f}%")
        elif dow_pct < -0.3:
            score -= 20
            signals.append(f"Dow futures {dow_pct:+.1f}%")

    sp_pct = _number_at(snapshot, "sp500_futures", "change_pct")
    if sp_pct is not None:
        if sp_pct > 0.3:
            score += 15
            signals.append(f"S&P futures {sp_pct:+.1f}%")
        elif sp_pct < -0.3:
            score -= 15
            signals.append(f"S&P futures {sp_pct:+.1f}%")

    usdinr = _number_at(snapshot, "usdinr", "price")
    if usdinr is not None:
        if usdinr < 83.5:
            score += 10
            signals.append(f"rupee firm at {usdinr:.2f}")
        elif usdinr > 84.5:
            score -= 10
            signals.append(f"rupee weak at {usdinr:.2f}")

    brent_pct = _number_at(snapshot, "brent", "change_pct")
    if brent_pct is not None:
        if brent_pct > 2.0:
            score -= 15
            signals.append(f"crude {brent_pct:+.1f}%")
        elif brent_pct < -2.0:
            score += 5
            signals.append(f"crude {brent_pct:+.1f}%")

    nikkei_pct = _number_at(snapshot, "nikkei", "change_pct")
    if nikkei_pct is not None:
        if nikkei_pct > 0.5:
            score += 10
            signals.append(f"Nikkei {nikkei_pct:+.1f}%")
        elif nikkei_pct < -0.5:
            score -= 10
            signals.append(f"Nikkei {nikkei_pct:+.1f}%")

    us_10y = _number_at(snapshot, "us_10y_yield", "price")
    if us_10y is not None:
        if us_10y > 4.5:
            score -= 15
            signals.append(f"US 10Y high at {us_10y:.2f}%")
        elif us_10y < 4.0:
            score += 10
            signals.append(f"US 10Y low at {us_10y:.2f}%")

    score = max(-100.0, min(100.0, score))
    if score > 15:
        bias = "bullish"
        prefix = "Bullish open"
    elif score < -15:
        bias = "bearish"
        prefix = "Bearish open"
    else:
        bias = "neutral"
        prefix = "Neutral open"

    summary_parts = signals[:3]
    if not summary_parts:
        summary_parts = ["mixed global signals"]
    if bias == "neutral" and nifty_spot > 0:
        summary_parts.append(f"watch {nifty_spot:,.0f} for direction")

    return {
        "bias": bias,
        "score": score,
        "signals": signals,
        "summary": f"{prefix}: {', '.join(summary_parts)}",
    }


def _download_with_timeout(yf: Any) -> tuple[Any, Any]:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_download_yfinance, yf)
    try:
        return future.result(timeout=_YFINANCE_TIMEOUT_SECONDS)
    except FutureTimeout as error:
        future.cancel()
        raise TimeoutError("yfinance timed out after 15 seconds") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _download_yfinance(yf: Any) -> tuple[Any, Any]:
    tickers = " ".join(
        ticker for name, ticker in TICKERS.items() if name != "usdinr"
    )
    daily = yf.download(
        tickers=tickers,
        period="2d",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=True,
        auto_adjust=False,
        timeout=12,
    )
    intraday = yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        group_by="ticker",
        progress=False,
        threads=True,
        auto_adjust=False,
        timeout=12,
    )
    return daily, intraday


def fetch_usdinr_nse(session: _requests.Session) -> dict[str, Any] | None:
    urls = (
        "https://www.nseindia.com/api/allIndices",
        "https://www.nseindia.com/api/refrates?index=currencyspotrates",
        "https://www.nseindia.com/api/quote-currency?key=USDINR",
        "https://www.nseindia.com/api/currencies-snapshot",
        "https://www.nseindia.com/api/forex-snapshot",
    )
    for url in urls:
        try:
            response = session.get(url, headers={"Accept": "application/json"}, timeout=8)
            if response.status_code != 200 or response.text.lstrip().startswith("<"):
                continue
            quote = _extract_usdinr(response.json())
            if quote is not None:
                quote["source"] = "NSE"
                return quote
        except (_requests.RequestException, ValueError) as error:
            print(f"cross_asset: NSE USD/INR fetch failed: {error}")
    return None


def fetch_usdinr_frankfurter() -> dict[str, Any] | None:
    try:
        today = _requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=INR", timeout=8
        ).json()
        date = _dt.date.fromisoformat(str(today["date"]))
        prev = _requests.get(
            f"https://api.frankfurter.app/{date - _dt.timedelta(days=1)}?from=USD&to=INR",
            timeout=8,
        ).json()
        price = _to_float(today.get("rates", {}).get("INR"))
        previous = _to_float(prev.get("rates", {}).get("INR"))
        if not 70 <= price <= 100:
            return None
        change = price - previous if previous > 0 else 0.0
        return _usdinr_quote(price, change, "Frankfurter")
    except (_requests.RequestException, KeyError, ValueError) as error:
        print(f"cross_asset: Frankfurter USD/INR fetch failed: {error}")
        return None


def fetch_usdinr_rbi() -> dict[str, Any] | None:
    return None


def _nse_session() -> _requests.Session:
    session = _requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.nseindia.com",
    })
    try:
        session.get("https://www.nseindia.com", timeout=8)
    except _requests.RequestException:
        pass
    return session


def _extract_usdinr(payload: Any) -> dict[str, Any] | None:
    rows = payload if isinstance(payload, list) else []
    if isinstance(payload, dict):
        rows = [payload]
        for key in ("data", "rows", "records"):
            if isinstance(payload.get(key), list):
                rows.extend(payload[key])
    for row in rows:
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("metadata"), dict):
            row = row["metadata"]
        label = " ".join(str(row.get(k) or "") for k in (
            "index", "symbol", "name", "currency", "unit", "identifier", "underlying",
        )).upper().replace("/", "")
        is_usdinr = (
            "USDINR" in label
            or ("USD" in label and ("INR" in label or "RUPEE" in label))
            or (row.get("unit") == "USD" and row.get("rate") is not None)
        )
        if not is_usdinr:
            continue
        price = _field_number(row, ("rate", "last", "lastPrice", "ltp", "value", "close"))
        if price is None or not 70 <= price <= 100:
            continue
        change = _field_number(row, ("change", "netChange", "changeValue")) or 0.0
        return _usdinr_quote(price, change, "NSE")
    return None


def _usdinr_quote(price: float, change: float, source: str) -> dict[str, Any]:
    base = price - change
    return {
        "price": _round(price, 4),
        "change": _round(change, 4),
        "change_pct": _round((change / base) * 100.0, 4) if base > 0 else 0.0,
        "direction": _direction(change),
        "stale": False,
        "stale_minutes": 0.0,
        "last_updated": _utc_now().isoformat(),
        "source": source,
    }


def _field_number(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    lookup = {key.lower(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key, lookup.get(key.lower()))
        if value not in (None, "", "-"):
            return _to_float(value)
    return None


def _empty_usdinr() -> dict[str, Any]:
    asset = _empty_asset()
    asset.update({"change": 0.0, "change_pct": 0.0, "source": None})
    return asset


def _quote_from_frames(ticker: str, daily: Any, intraday: Any) -> dict[str, Any]:
    intraday_close = _series_for(intraday, ticker, "Close")
    daily_close = _series_for(daily, ticker, "Close")

    price, last_updated = _last_valid(intraday_close)
    if price is None:
        price, last_updated = _last_valid(daily_close)

    previous_close = _previous_close(daily_close)
    change = None
    change_pct = None
    if price is not None and previous_close is not None and previous_close > 0:
        change = price - previous_close
        change_pct = (change / previous_close) * 100.0

    stale_minutes = _stale_minutes(last_updated)
    return {
        "price": _round(price, 4),
        "change": _round(change, 4),
        "change_pct": _round(change_pct, 4),
        "direction": _direction(change_pct),
        "stale": stale_minutes is None or stale_minutes > 30,
        "stale_minutes": _round(stale_minutes, 1),
        "last_updated": _iso_datetime(last_updated),
    }


def _yield_quote(quote: dict[str, Any]) -> dict[str, Any]:
    result = dict(quote)
    raw_change = result.get("change")
    result["change"] = (
        None
        if raw_change is None
        else _round(_to_float(raw_change) * 100.0, 2)
    )
    result["direction"] = _direction(result.get("change"))
    result.pop("change_pct", None)
    return result


def _with_nifty_premium(
    quote: dict[str, Any],
    nifty_spot: float,
) -> dict[str, Any]:
    result = dict(quote)
    price = _to_float(result.get("price"))
    if price > 0 and nifty_spot > 0:
        premium = price - nifty_spot
        result["premium"] = _round(premium, 2)
        result["premium_pct"] = _round((premium / nifty_spot) * 100.0, 4)
    else:
        result["premium"] = None
        result["premium_pct"] = None
    return result


def _series_for(frame: Any, ticker: str, field: str) -> Any:
    if frame is None or getattr(frame, "empty", True):
        return None

    columns = getattr(frame, "columns", None)
    if columns is None:
        return None

    if getattr(columns, "nlevels", 1) > 1:
        for key in ((ticker, field), (field, ticker)):
            if key in columns:
                return frame[key]
        return None

    if field in columns:
        return frame[field]
    return None


def _last_valid(series: Any) -> tuple[float | None, _dt.datetime | None]:
    values = _valid_series(series)
    if values is None or len(values) == 0:
        return None, None

    value = _to_float(values.iloc[-1])
    timestamp = _to_datetime(values.index[-1])
    if value <= 0:
        return None, timestamp
    return value, timestamp


def _previous_close(series: Any) -> float | None:
    values = _valid_series(series)
    if values is None or len(values) < 2:
        return None

    previous = _to_float(values.iloc[-2])
    return previous if previous > 0 else None


def _valid_series(series: Any) -> Any:
    if series is None:
        return None
    try:
        values = series.dropna()
    except AttributeError:
        return None
    return values


def _fetch_india_vix() -> float | None:
    try:
        from signals.market_signals import fetch_india_vix

        return fetch_india_vix()
    except Exception as error:
        print(f"cross_asset: India VIX fetch failed: {error}")
        return None


def _failed_snapshot(nifty_spot: float, error: str) -> dict[str, Any]:
    cached = _last_good_snapshot()
    if cached is not None:
        cached["staleness_warning"] = True
        cached["error"] = error
        _mark_assets_stale(cached)
        return cached

    snapshot = {
        "usdinr": _empty_asset(),
        "brent": _empty_asset(),
        "dow_futures": _empty_asset(),
        "sp500_futures": _empty_asset(),
        "us_10y_yield": _empty_yield(),
        "gold": _empty_asset(),
        "nikkei": _empty_asset(),
        "hangseng": _empty_asset(),
        "vix_us": _empty_asset(),
        "sgx_nifty": _with_nifty_premium(_empty_asset(), nifty_spot),
        "india_vix": None,
        "gift_nifty": {
            "price": None,
            "premium": None,
            "premium_pct": None,
        },
        "fetched_at": _utc_now().isoformat(),
        "staleness_warning": True,
        "error": error,
    }
    return snapshot


def _empty_asset() -> dict[str, Any]:
    return {
        "price": None,
        "change": None,
        "change_pct": None,
        "direction": "flat",
        "stale": True,
        "stale_minutes": None,
        "last_updated": None,
    }


def _empty_yield() -> dict[str, Any]:
    asset = _empty_asset()
    asset.pop("change_pct", None)
    return asset


def _mark_assets_stale(snapshot: dict[str, Any]) -> None:
    for value in snapshot.values():
        if isinstance(value, dict) and "stale" in value:
            value["stale"] = True


def _store_last_good(snapshot: dict[str, Any]) -> None:
    global _LAST_GOOD_SNAPSHOT
    with _LAST_GOOD_LOCK:
        _LAST_GOOD_SNAPSHOT = copy.deepcopy(snapshot)


def _last_good_snapshot() -> dict[str, Any] | None:
    with _LAST_GOOD_LOCK:
        if _LAST_GOOD_SNAPSHOT is None:
            return None
        return copy.deepcopy(_LAST_GOOD_SNAPSHOT)


def _asset_is_stale(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    return bool(value.get("stale")) or not _valid_price(value.get("price"))


def _number_at(snapshot: dict[str, Any], section: str, field: str) -> float | None:
    value = snapshot.get(section)
    if not isinstance(value, dict):
        return None
    number = _to_float(value.get(field))
    return number if number != 0 or value.get(field) in (0, 0.0) else None


def _direction(value: Any) -> str:
    number = _to_float(value)
    if number > 0.01:
        return "up"
    if number < -0.01:
        return "down"
    return "flat"


def _stale_minutes(timestamp: _dt.datetime | None) -> float | None:
    if timestamp is None:
        return None
    elapsed = (_utc_now() - timestamp).total_seconds()
    if elapsed < 0:
        return 0.0
    return elapsed / 60.0


def _iso_datetime(value: _dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _to_datetime(value: Any) -> _dt.datetime | None:
    if value is None:
        return None
    try:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
    except (OverflowError, ValueError, TypeError):
        return None

    if isinstance(value, _dt.datetime):
        timestamp = value
    elif isinstance(value, _dt.date):
        timestamp = _dt.datetime.combine(value, _dt.time.min)
    else:
        return None

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=_dt.timezone.utc)
    return timestamp.astimezone(_dt.timezone.utc)


def _to_float(value: Any) -> float:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _valid_price(value: Any) -> bool:
    return _to_float(value) > 0


def _round(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    number = _to_float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)
