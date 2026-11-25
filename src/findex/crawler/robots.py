"""robots.txt, fetched once per host and cached."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger(__name__)
DISALLOW_ALL = ["User-agent: *", "Disallow: /"]


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str, timeout: float = 10.0) -> None:
        self._client = client
        self._ua = user_agent
        self._timeout = timeout
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _parser(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if (rp := self._parsers.get(origin)) is not None:
            return rp
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:  # the first worker fetches; the others wait for its result
            if (rp := self._parsers.get(origin)) is not None:
                return rp
            rp = RobotFileParser(origin + "/robots.txt")
            try:
                async with asyncio.timeout(self._timeout):
                    resp = await self._client.get(origin + "/robots.txt")
                if resp.status_code >= 500:
                    rp.parse(DISALLOW_ALL)  # server trouble: assume we may not crawl
                elif resp.status_code >= 400:
                    rp.parse([])  # no robots.txt: everything allowed
                else:
                    rp.parse(resp.text.splitlines())
            except (TimeoutError, httpx.TransportError):
                rp.parse(DISALLOW_ALL)
            log.info("robots.txt for %s cached", origin)
            self._parsers[origin] = rp
            return rp

    async def allowed(self, url: str) -> bool:
        return (await self._parser(url)).can_fetch(self._ua, url)

    async def crawl_delay(self, url: str) -> float | None:
        d = (await self._parser(url)).crawl_delay(self._ua)
        return float(d) if d is not None else None
