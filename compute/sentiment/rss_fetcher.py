from __future__ import annotations

import calendar
import datetime as dt
import email.utils
import html
import re
import time
from typing import Any

import feedparser
import requests


RSS_FEEDS = {
    "moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "livemint": "https://www.livemint.com/rss/markets",
    "hindu_business": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "cnbc_tv18": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
    "news18_markets": "https://www.news18.com/commonfeeds/v1/eng/rss/markets.xml",
}

NSE_HOME = "https://www.nseindia.com"
NSE_ANNOUNCEMENTS_PAGE = (
    "https://www.nseindia.com/companies-listing/"
    "corporate-filings-announcements"
)
NSE_ANNOUNCEMENTS_URL = (
    "https://www.nseindia.com/api/corporate-announcements?index=equities"
)

HIGH_IMPACT_ANNOUNCEMENT_KEYWORDS = (
    "board meeting",
    "results",
    "dividend",
    "bonus",
    "split",
    "merger",
    "acquisition",
    "buyback",
    "delisting",
    "sebi order",
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/rss+xml;q=0.8,application/json;q=0.8,*/*;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_all_feeds(max_age_hours: int = 4) -> list[dict]:
    """
    Fetch and parse all RSS feeds.

    Returns articles newer than max_age_hours, deduped by case-insensitive
    title, newest first. One broken feed never blocks the rest.
    """
    now = _utc_now()
    cutoff = now - dt.timedelta(hours=max_age_hours)
    seen_titles: set[str] = set()
    articles: list[dict[str, Any]] = []

    for source, url in RSS_FEEDS.items():
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=5)
            if response.status_code >= 400:
                print(f"sentiment: {source} RSS returned {response.status_code}")
                continue
            parsed = feedparser.parse(response.content)
        except (requests.RequestException, TypeError, ValueError) as error:
            print(f"sentiment: {source} RSS fetch failed: {error}")
            continue

        for entry in parsed.entries:
            title = _clean_text(_entry_value(entry, "title"))
            if not title:
                continue

            dedupe_key = re.sub(r"\s+", " ", title).casefold()
            if dedupe_key in seen_titles:
                continue

            published_at = _entry_datetime(entry, now)
            if published_at < cutoff:
                continue

            seen_titles.add(dedupe_key)
            articles.append({
                "source": source,
                "title": title,
                "summary": _clean_text(_entry_value(entry, "summary"))[:200],
                "url": _entry_value(entry, "link"),
                "published_at": published_at.isoformat(),
                "age_minutes": _age_minutes(now, published_at),
            })

    articles.sort(key=lambda item: item["published_at"], reverse=True)
    return articles


def fetch_nse_announcements() -> list[dict]:
    """
    Fetch NSE corporate announcements with the browser-session pattern.

    Returns empty list on any NSE/session failure.
    """
    try:
        session = _nse_session()
        _warm_nse_session(session)
        response = session.get(
            NSE_ANNOUNCEMENTS_URL,
            headers={
                "Accept": "application/json",
                "Referer": NSE_ANNOUNCEMENTS_PAGE,
            },
            timeout=10,
        )
        if not _json_response(response):
            return []

        rows = _extract_announcement_rows(response.json())
    except (requests.RequestException, ValueError) as error:
        print(f"sentiment: NSE announcements fetch failed: {error}")
        return []

    announcements: list[dict[str, Any]] = []
    for row in rows:
        symbol = _first_text(
            row,
            ("symbol", "sm_symbol", "companySymbol", "issuer", "name"),
        )
        subject = _first_text(
            row,
            (
                "subject",
                "desc",
                "description",
                "announcementDesc",
                "attchmntText",
                "purpose",
            ),
        )
        published_text = _first_text(
            row,
            ("an_dt", "date", "disseminationTime", "sort_date", "createdDate"),
        )
        if not subject:
            continue

        announcements.append({
            "symbol": symbol,
            "subject": subject,
            "published_at": _parse_datetime_text(published_text).isoformat(),
            "impact": _announcement_impact(subject),
        })

    announcements.sort(key=lambda item: item["published_at"], reverse=True)
    return announcements[:50]


def _nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    session.headers.update({"Connection": "keep-alive"})
    return session


def _warm_nse_session(session: requests.Session) -> None:
    session.get(NSE_HOME, timeout=10)
    time.sleep(0.5)
    session.get(NSE_ANNOUNCEMENTS_PAGE, timeout=10)
    time.sleep(0.5)


def _json_response(response: requests.Response) -> bool:
    if response.status_code != 200:
        return False
    text = response.text.lstrip()
    if not text or text.startswith("<"):
        return False
    content_type = response.headers.get("content-type", "")
    return "json" in content_type.lower() or text[0] in "[{"


def _extract_announcement_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("data", "rows", "result", "announcements"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return [payload]


def _entry_datetime(entry: Any, fallback: dt.datetime) -> dt.datetime:
    parsed_tuple = (
        entry.get("published_parsed")
        or entry.get("updated_parsed")
        or entry.get("created_parsed")
    )
    if parsed_tuple:
        return dt.datetime.fromtimestamp(
            calendar.timegm(parsed_tuple),
            tz=dt.timezone.utc,
        )

    for key in ("published", "updated", "created"):
        value = _entry_value(entry, key)
        if not value:
            continue
        parsed = _parse_datetime_text(value, fallback=None)
        if parsed is not None:
            return parsed

    return fallback


def _parse_datetime_text(
    value: str,
    fallback: dt.datetime | None = None,
) -> dt.datetime:
    clean = str(value or "").strip()
    if clean:
        try:
            parsed = email.utils.parsedate_to_datetime(clean)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
        except (TypeError, ValueError):
            pass

        for fmt in (
            "%d-%b-%Y %H:%M:%S",
            "%d-%b-%Y %H:%M",
            "%d-%b-%Y",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                parsed = dt.datetime.strptime(clean, fmt)
                return parsed.replace(tzinfo=dt.timezone.utc)
            except ValueError:
                pass

    return fallback if fallback is not None else _utc_now()


def _entry_value(entry: Any, key: str) -> str:
    value = entry.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    lower_lookup = {key.lower(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key)
        if value is None:
            value = lower_lookup.get(key.lower())
        if value is not None and str(value).strip():
            return _clean_text(str(value))
    return ""


def _announcement_impact(subject: str) -> str:
    lower_subject = subject.casefold()
    if any(keyword in lower_subject for keyword in HIGH_IMPACT_ANNOUNCEMENT_KEYWORDS):
        return "high"
    if any(word in lower_subject for word in ("outcome", "meeting", "order")):
        return "medium"
    return "low"


def _age_minutes(now: dt.datetime, published_at: dt.datetime) -> int:
    return max(0, int((now - published_at).total_seconds() // 60))


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)
