"""
Angel One historical data fetcher.

Fetches candle ranges in API-safe chunks, re-authenticates on session expiry,
and logs persistent failures without crashing the caller.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import pyotp
from SmartApi import SmartConnect

from data.error_logger import _logger as elog
from db.write_service import ReadOperation, resolve_read
from security.credential_manager import get_angel_credentials

CHUNK_DAYS = {
    "ONE_MINUTE": 25,
    "THREE_MINUTE": 55,
    "FIVE_MINUTE": 90,
    "FIFTEEN_MINUTE": 180,
    "THIRTY_MINUTE": 180,
    "ONE_HOUR": 380,
    "ONE_DAY": 1900,
}

class AngelFetcher:
    def __init__(self) -> None:
        self.smart = self._authenticate()

    @staticmethod
    def _authenticate() -> SmartConnect:
        credentials = get_angel_credentials()
        try:
            smart = SmartConnect(credentials["ANGEL_API_KEY"])
            totp = pyotp.TOTP(credentials["ANGEL_TOTP_SECRET"]).now()
            session = smart.generateSession(
                credentials["ANGEL_CLIENT_ID"],
                credentials["ANGEL_PASSWORD"],
                totp,
            )
            if not session.get("status"):
                raise RuntimeError("Angel One authentication failed")
            print("angel_fetcher: authenticated")
            return smart
        finally:
            credentials.clear()

    def fetch(
        self,
        exchange: str,
        token: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        chunk_days = CHUNK_DAYS.get(interval, 25)
        frames: list[pd.DataFrame] = []
        current = start_dt

        while current < end_dt:
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)
            for attempt in range(3):
                try:
                    response = self.smart.getCandleData(
                        {
                            "exchange": exchange,
                            "symboltoken": str(token),
                            "interval": interval,
                            "fromdate": current.strftime("%Y-%m-%d %H:%M"),
                            "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
                        }
                    )
                    if response and response.get("data"):
                        frame = pd.DataFrame(
                            response["data"],
                            columns=["DateTime", "Open", "High", "Low", "Close", "Volume"],
                        )
                        frame["DateTime"] = pd.to_datetime(frame["DateTime"])
                        frame.set_index("DateTime", inplace=True)
                        frames.append(frame)
                    break
                except Exception as error:
                    lowered = str(error).lower()
                    if "session" in lowered or "token" in lowered:
                        print("angel: session expired, re-authenticating")
                        try:
                            self.smart = self._authenticate()
                        except Exception as auth_error:
                            elog.log(
                                "angel_fetcher",
                                "reauthentication_failed",
                                type(auth_error).__name__,
                            )
                    elif "429" in lowered or "rate" in lowered:
                        print("angel: rate limit, waiting 60s")
                        time.sleep(60)
                    elif attempt == 2:
                        elog.log(
                            "angel_fetcher",
                            "chunk_failed",
                            str(error),
                            date_str=str(current.date()),
                        )
                    time.sleep(2)

            current = chunk_end + timedelta(minutes=1)
            time.sleep(0.4)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).drop_duplicates().sort_index()

    def fetch_by_symbol(
        self,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime,
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        from data.instrument_master import master

        token = master.get_token(symbol, exchange)
        if not token:
            elog.log(
                "angel_fetcher",
                "token_not_found",
                f"No token for {symbol}",
                symbol=symbol,
            )
            return pd.DataFrame()
        return self.fetch(exchange, token, interval, start_dt, end_dt)

    def find_nifty_front_future(self) -> tuple[str, str] | None:
        try:
            result = resolve_read(ReadOperation.FIND_NIFTY_FRONT_FUTURE_TOKEN, {})
            token = result["token"]
            return ("NFO", token) if token else None
        except Exception as error:
            elog.log("angel_fetcher", "front_future_lookup_failed", str(error))
            return None
