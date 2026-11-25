"""One polite fetch: timeout, bounded retries with exponential backoff + jitter, Retry-After."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "findex-crawler/0.7 (+https://github.com/ua-python-student/findex; student project)"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int  # 0 = no HTTP response (timeout, connection error, redirect loop)
    final_url: str = ""
    content_type: str = ""
    text: str = ""
    nbytes: int = 0
    elapsed: float = 0.0
    attempts: int = 1
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:
    """min(cap, base * 2**attempt) + jitter, so retrying clients do not synchronise."""
    return min(cap, base * 2**attempt) + random.uniform(0, base)


def retry_after_seconds(value: str | None, cap: float = 60.0) -> float | None:
    """Retry-After is either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return min(cap, float(value))
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(cap, when.timestamp() - time.time()))


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FetchResult:
    t0 = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url)
        except TimeoutError:
            error, status, wait = "timeout", 0, None
        except httpx.TooManyRedirects:
            return FetchResult(
                url, 0, elapsed=time.perf_counter() - t0, attempts=attempt, error="redirect loop"
            )
        except httpx.TransportError as e:  # connection refused/reset, DNS...
            error, status, wait = type(e).__name__, 0, None
        else:
            status = resp.status_code
            if status not in RETRY_STATUS:
                return FetchResult(
                    url,
                    status,
                    str(resp.url),
                    resp.headers.get("content-type", ""),
                    resp.text if status < 400 else "",
                    len(resp.content),
                    time.perf_counter() - t0,
                    attempt,
                    "" if status < 400 else f"HTTP {status}",  # 404 etc.: never retried
                )
            error = f"HTTP {status}"
            wait = retry_after_seconds(resp.headers.get("retry-after")) if status == 429 else None
        if attempt > max_retries:
            return FetchResult(url, status, elapsed=time.perf_counter() - t0, attempts=attempt, error=error)
        delay = wait if wait is not None else backoff_delay(attempt - 1, base_delay)
        log.info("retry %d/%d for %s after %s, sleeping %.2f s", attempt, max_retries, url, error, delay)
        await sleep(delay)
