from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)


class BSESession:
    """Bounded HTTP session for BSE's public API (api.bseindia.com /
    www.bseindia.com). Verified live (2026-09-03) that these need no
    cookie/anti-bot dance -- unlike NSE, a plain browser-like User-Agent and
    Referer are sufficient. Still bounded: fixed timeout, at most one retry
    on a 5xx, no unbounded backoff loop."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, text/html, */*",
                "Referer": "https://www.bseindia.com/",
            }
        )

    def get(self, url: str, **kwargs):
        try:
            response = self.session.get(url, timeout=10, **kwargs)
            if response.status_code >= 500:
                time.sleep(1.0)
                response = self.session.get(url, timeout=10, **kwargs)
            return response
        except requests.RequestException as error:
            logger.error("BSE request failed: %s", type(error).__name__)
            return None


bse = BSESession()
