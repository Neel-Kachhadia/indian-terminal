from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)


class NSESession:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com",
            "Connection": "keep-alive",
        })
        self._last_refresh = 0.0
        self._cookie_valid = False

    def get(self, url: str, **kwargs):
        if time.time() - self._last_refresh > 1800 or not self._cookie_valid:
            self._refresh_cookie()
        try:
            response = self.session.get(url, timeout=10, **kwargs)
            if response.status_code == 403:
                self._refresh_cookie()
                response = self.session.get(url, timeout=10, **kwargs)
            return response
        except requests.RequestException as error:
            logger.error("NSE request failed: %s", type(error).__name__)
            return None

    def _refresh_cookie(self) -> None:
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
            self._last_refresh = time.time()
            self._cookie_valid = True
            logger.info("NSE session cookie refreshed")
        except requests.RequestException as error:
            logger.error("NSE cookie refresh failed: %s", type(error).__name__)
            self._cookie_valid = False


nse = NSESession()
