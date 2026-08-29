"""Single entry point for data source fallback chains."""

from datetime import date, datetime

from data.error_logger import _logger as elog
from data.fallback import DataResult
from db.write_service import ReadOperation, resolve_read


def get_data(
    symbol: str,
    data_type: str,
    start: datetime | date,
    end: datetime | date,
    interval: str | None = None,
    exchange: str = "NSE",
    angel_fetcher=None,
) -> DataResult:
    if data_type == "eod":
        return _get_eod(symbol, start, end, exchange, angel_fetcher)
    if data_type == "fo_eod":
        return _get_fo_eod(symbol, start, end)
    if data_type == "intraday":
        return _get_intraday(symbol, interval, start, end, exchange, angel_fetcher)
    if data_type == "gift_live":
        return _get_gift_live(angel_fetcher)
    raise ValueError(f"Unknown data_type: {data_type}")


def _get_eod(symbol, start, end, exchange, angel_fetcher) -> DataResult:
    try:
        from jugaad_data.nse import stock_df

        from_day = start if isinstance(start, date) else start.date()
        to_day = end if isinstance(end, date) else end.date()
        frame = stock_df(symbol=symbol, from_date=from_day, to_date=to_day, series="EQ")
        if not frame.empty:
            return DataResult(data=frame, source="jugaad", confidence="high")
    except Exception as error:
        elog.log(
            "data_router",
            "jugaad_eod_failed",
            str(error),
            symbol=symbol,
            fallback_used="angel_one",
        )

    if angel_fetcher:
        try:
            from_day = start if isinstance(start, date) else start.date()
            to_day = end if isinstance(end, date) else end.date()
            frame = angel_fetcher.fetch_by_symbol(
                symbol,
                "ONE_DAY",
                datetime.combine(from_day, datetime.min.time()),
                datetime.combine(to_day, datetime.min.time()),
                exchange=exchange,
            )
            if not frame.empty:
                return DataResult(data=frame, source="angel_one", confidence="high")
        except Exception as error:
            elog.log("data_router", "angel_eod_failed", str(error), symbol=symbol)

    return DataResult(
        data=None,
        source="unavailable",
        confidence="none",
        error="All EOD sources failed",
    )


def _get_fo_eod(symbol, start, end) -> DataResult:
    try:
        from_day = start if isinstance(start, date) else start.date()
        to_day = end if isinstance(end, date) else end.date()
        result = resolve_read(
            ReadOperation.LOAD_FO_EOD_RANGE,
            {
                "symbol": symbol.upper(),
                "from_date": from_day.isoformat(),
                "to_date": to_day.isoformat(),
            },
        )
        rows = result["rows"]
        if rows:
            import pandas as pd

            return DataResult(
                data=pd.DataFrame(rows),
                source="duckdb_fo_eod",
                confidence="high",
            )
    except Exception as error:
        elog.log("data_router", "fo_eod_duckdb_failed", str(error), symbol=symbol)

    return DataResult(
        data=None,
        source="unavailable",
        confidence="none",
        note="fo_eod table empty - run backfill first",
    )


def _get_intraday(symbol, interval, start, end, exchange, angel_fetcher) -> DataResult:
    if not interval:
        return DataResult(
            data=None,
            source="unavailable",
            confidence="none",
            error="interval is required for intraday data",
        )
    if not angel_fetcher:
        return DataResult(
            data=None,
            source="unavailable",
            confidence="none",
            error="Angel fetcher not initialised",
        )
    try:
        frame = angel_fetcher.fetch_by_symbol(symbol, interval, start, end, exchange=exchange)
        if not frame.empty:
            return DataResult(data=frame, source="angel_one", confidence="high")
    except Exception as error:
        elog.log("data_router", "intraday_failed", str(error), symbol=symbol)

    return DataResult(
        data=None,
        source="unavailable",
        confidence="none",
        error="Angel One intraday failed",
    )


def _get_gift_live(angel_fetcher) -> DataResult:
    from data.gift_nifty import GiftNifty

    return GiftNifty().get_live_price(angel_fetcher=angel_fetcher)
