from __future__ import annotations

import datetime as dt
import random
import threading
import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime

import httpx

from openenergy.providers.base import ApiRole

BASE_URLS: dict[ApiRole, str] = {
    ApiRole.FORECAST: "https://api.open-meteo.com/v1/forecast",
    ApiRole.ARCHIVE: "https://archive-api.open-meteo.com/v1/archive",
    ApiRole.PREVIOUS_RUNS: "https://previous-runs-api.open-meteo.com/v1/forecast",
    ApiRole.ENSEMBLE: "https://ensemble-api.open-meteo.com/v1/ensemble",
    # Single Runs (run=) is served by the previous-runs host; kept as a distinct
    # role so provenance/routing stays explicit.
    ApiRole.SINGLE_RUNS: "https://previous-runs-api.open-meteo.com/v1/forecast",
}

CUSTOMER_URLS: dict[ApiRole, str] = {
    ApiRole.FORECAST: "https://customer-api.open-meteo.com/v1/forecast",
    ApiRole.ARCHIVE: "https://customer-archive-api.open-meteo.com/v1/archive",
    ApiRole.PREVIOUS_RUNS: "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
    ApiRole.ENSEMBLE: "https://customer-ensemble-api.open-meteo.com/v1/ensemble",
    ApiRole.SINGLE_RUNS: "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
}

_RETRY_STATUS = {429, 500, 502, 503, 504}


class FixedWindowRateLimiter:
    """Thread-safe, weight-aware limiter for Open-Meteo's rolling quotas.

    A batched request is still charged for every location, so callers pass that
    cost as ``weight`` even though only one HTTP request is made.
    """

    def __init__(
        self,
        limit: int = 600,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window_seconds must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._window_start = clock()
        self._used = 0
        self._lock = threading.Lock()

    def acquire(self, weight: int = 1) -> None:
        if weight <= 0 or weight > self.limit:
            raise ValueError(f"weight must be between 1 and {self.limit}")
        with self._lock:
            now = self._clock()
            if now >= self._window_start + self.window_seconds:
                windows = int((now - self._window_start) // self.window_seconds)
                self._window_start += max(1, windows) * self.window_seconds
                self._used = 0
            if self._used + weight > self.limit:
                wait = max(0.0, self._window_start + self.window_seconds - now)
                self._sleep(wait)
                # Advance from the previous boundary as well as the injected
                # clock, so a deterministic no-op sleep cannot loop forever.
                self._window_start = max(
                    self._clock(), self._window_start + self.window_seconds
                )
                self._used = 0
            self._used += weight


def _retry_after_seconds(value: str | None, now: Callable[[], dt.datetime]) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=dt.timezone.utc)
        current = now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=dt.timezone.utc)
        return max(0.0, (target - current).total_seconds())


class OpenMeteoClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        rate_limiter: FixedWindowRateLimiter | None = None,
        rng: Callable[[], float] | random.Random | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.rate_limiter = rate_limiter or FixedWindowRateLimiter()
        if rng is None:
            self._rng = random.random
        elif callable(rng):
            self._rng = rng
        else:
            self._rng = rng.random
        self._sleep = sleep
        self._now = now or (lambda: dt.datetime.now(dt.timezone.utc))
        self._urls = CUSTOMER_URLS if api_key else BASE_URLS

    def get(self, role: ApiRole, params: dict, *, weight: int = 1) -> dict | list[dict]:
        url = self._urls[role]
        query = dict(params)
        if self.api_key:
            query["apikey"] = self.api_key
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                self.rate_limiter.acquire(weight)
                resp = client.get(url, params=query)
                if resp.status_code in _RETRY_STATUS and attempt < self.max_retries:
                    cap = self.backoff_base * (2 ** attempt)
                    jitter = self._rng() * cap
                    retry_after = _retry_after_seconds(resp.headers.get("Retry-After"), self._now)
                    self._sleep(max(jitter, retry_after or 0.0))
                    continue
                resp.raise_for_status()
                return resp.json()
        raise AssertionError("unreachable: retry loop always raises or returns")
