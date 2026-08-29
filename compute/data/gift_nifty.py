"""
GIFT NIFTY source with fallback chain.

Priority:
1. NSE IX official endpoint
2. Angel One NFO front-month future proxy
3. NSE allIndices
4. Dow futures estimate
5. Unavailable with clear metadata
"""

import time
from datetime import datetime, timedelta
from typing import Any

import requests

from data.error_logger import _logger as elog
from data.fallback import DataResult
from db.write_service import ReadOperation, resolve_read

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
}


class GiftNifty:
    def get_live_price(
        self,
        nifty_spot: float = 0,
        angel_fetcher=None,
        dow_change_pct: float = 0,
    ) -> DataResult:
        try:
            data = self._try_nseix()
            if data:
                premium = round(data["price"] - nifty_spot, 2) if nifty_spot else None
                return DataResult(
                    data={
                        "price": data["price"],
                        "change": data.get("change"),
                        "change_pct": data.get("change_pct"),
                        "premium_to_nifty": premium,
                        "source": "nseix",
                    },
                    source="nseix",
                    confidence="high",
                    note="NSE IX official, delayed if source is delayed",
                )
        except Exception as error:
            elog.log(
                "gift_nifty",
                "nseix_failed",
                str(error),
                fallback_used="angel_proxy",
            )

        if angel_fetcher:
            try:
                result = angel_fetcher.find_nifty_front_future()
                if result:
                    exchange, token = result
                    frame = angel_fetcher.fetch(
                        exchange,
                        token,
                        "FIVE_MINUTE",
                        datetime.now() - timedelta(minutes=30),
                        datetime.now(),
                    )
                    if not frame.empty:
                        price = float(frame["Close"].iloc[-1])
                        premium = round(price - nifty_spot, 2) if nifty_spot else None
                        return DataResult(
                            data={
                                "price": price,
                                "premium_to_nifty": premium,
                                "source": "angel_nfo_proxy",
                            },
                            source="angel_nfo_proxy",
                            confidence="high",
                            note="NSE front-month future proxy",
                        )
            except Exception as error:
                elog.log(
                    "gift_nifty",
                    "angel_proxy_failed",
                    str(error),
                    fallback_used="nse_all_indices",
                )

        try:
            data = self._try_nse_all_indices()
            if data:
                return DataResult(
                    data=data,
                    source="nse_all_indices",
                    confidence="medium",
                    note="From NSE allIndices endpoint",
                )
        except Exception as error:
            elog.log(
                "gift_nifty",
                "all_indices_failed",
                str(error),
                fallback_used="dow_estimate",
            )

        if dow_change_pct and nifty_spot:
            try:
                last = self._get_last_known()
                if last:
                    estimated = round(last * (1 + dow_change_pct / 100), 2)
                    return DataResult(
                        data={
                            "price": estimated,
                            "premium_to_nifty": round(estimated - nifty_spot, 2),
                            "source": "dow_estimate",
                        },
                        source="dow_estimate",
                        is_estimated=True,
                        confidence="low",
                        note=f"Estimated from Dow {dow_change_pct:+.2f}%",
                    )
            except Exception as error:
                elog.log("gift_nifty", "estimate_failed", str(error))

        return DataResult(
            data=None,
            source="unavailable",
            confidence="none",
            note="GIFT NIFTY temporarily unavailable",
        )

    def premarket_gap_signal(
        self,
        gift_price: float,
        prev_nifty_close: float,
    ) -> dict[str, Any]:
        if not gift_price or not prev_nifty_close:
            return {"gap_pct": None, "direction": "unknown"}

        gap_pct = (gift_price - prev_nifty_close) / prev_nifty_close * 100
        gap_points = round(gift_price - prev_nifty_close, 2)
        strength = "strong" if abs(gap_pct) > 0.5 else "moderate" if abs(gap_pct) > 0.2 else "weak"
        direction = "gap_up" if gap_pct > 0.1 else "gap_down" if gap_pct < -0.1 else "flat"
        interpretations = {
            "gap_up": f"Expect gap up open +{gap_points:.0f} pts",
            "gap_down": f"Expect gap down open {gap_points:.0f} pts",
            "flat": "Flat open expected",
        }
        return {
            "gap_pct": round(gap_pct, 3),
            "gap_points": gap_points,
            "direction": direction,
            "signal_strength": strength,
            "interpretation": interpretations.get(direction, ""),
        }

    def _try_nseix(self) -> dict[str, Any] | None:
        endpoints = [
            "https://www.nseix.com/api/giftnifty/livedata",
            "https://www.nseix.com/api/market/giftnifty",
        ]
        for url in endpoints:
            try:
                response = requests.get(url, headers=NSE_HEADERS, timeout=8)
                if response.status_code != 200:
                    continue
                data = response.json()
                price = (
                    data.get("lastTradedPrice")
                    or data.get("ltp")
                    or data.get("last")
                    or data.get("price")
                )
                if price:
                    return {"price": float(price)}
            except Exception:
                continue
        return None

    def _try_nse_all_indices(self) -> dict[str, Any] | None:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com", timeout=8)
        time.sleep(0.5)
        response = session.get("https://www.nseindia.com/api/allIndices", timeout=8)
        if response.status_code != 200:
            return None
        for item in response.json().get("data", []):
            index_name = str(item.get("index", "")).upper()
            if "GIFT" in index_name or "IFSC" in index_name:
                return {
                    "price": float(item.get("last", 0)),
                    "source": "nse_all_indices",
                }
        return None

    def _get_last_known(self) -> float | None:
        result = resolve_read(ReadOperation.LOAD_GIFT_NIFTY_LAST_CLOSE, {})
        return result["close"]
