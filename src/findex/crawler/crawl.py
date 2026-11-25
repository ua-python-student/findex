"""The crawler: a Queue frontier, worker tasks in a TaskGroup, streamed out as an async generator."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, domain_allowed, host_of

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass
class CrawlStats:
    started: float = field(default_factory=time.perf_counter)
    pages: int = 0
    fetched: int = 0
    errors: int = 0
    retries: int = 0
    robots_blocked: int = 0
    in_flight: int = 0
    queued: int = 0
    nbytes: int = 0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def pages_per_s(self) -> float:
        return self.pages / self.elapsed if self.elapsed > 0 else 0.0


def parse_html(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """(title, visible text, absolute canonical links). Pure CPU work, but fast (C parser)."""
    tree = HTMLParser(html)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""
    links: list[str] = []
    for a in tree.css("a[href]"):
        href = a.attributes.get("href")
        if href and (u := canonicalize(href, base_url)) is not None:
            links.append(u)
    for node in tree.css("script, style, noscript"):
        node.decompose()
    body = tree.body
    text = body.text(separator=" ", strip=True) if body else ""
    return title, text, links


class _HostGate:
    """Per-host politeness: at most `per_host` requests in flight and `delay` s between starts."""

    def __init__(self, per_host: int, delay: float) -> None:
        self.per_host, self.delay = per_host, delay
        self.sems: defaultdict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(self.per_host))
        self.next_start: dict[str, float] = {}

    @contextlib.asynccontextmanager
    async def slot(self, host: str) -> AsyncGenerator[None]:
        async with self.sems[host]:
            now = time.monotonic()
            start = max(now, self.next_start.get(host, now))
            self.next_start[host] = start + self.delay  # reserve the slot before awaiting
            if start > now:
                await asyncio.sleep(start - now)  # the sleep that does not block the loop
            yield


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    per_host_delay: float = 0.1,
    allowed_domains: Iterable[str] = (),
    timeout: float = 10.0,
    max_retries: int = 3,
    user_agent: str = USER_AGENT,
    log_file: TextIO | None = None,
    stats: CrawlStats | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Page]:
    """Yield pages as they are fetched, never more than ``max_pages``."""
    stats = stats if stats is not None else CrawlStats()
    seed_urls = [u for s in seeds if (u := canonicalize(s)) is not None]
    allowed = frozenset(d.lower() for d in allowed_domains) or frozenset(
        (host_of(u).split(":")[0]) for u in seed_urls
    )
    out: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=2 * concurrency)  # backpressure
    frontier: asyncio.Queue[str] = asyncio.Queue()
    seen: set[str] = set()
    stop = asyncio.Event()
    in_flight = asyncio.Semaphore(concurrency)  # global bound on requests
    gate = _HostGate(per_host, per_host_delay)

    def enqueue(url: str) -> None:
        if url not in seen and domain_allowed(url, allowed):
            seen.add(url)
            frontier.put_nowait(url)
            stats.queued = frontier.qsize()

    def record(r: FetchResult) -> None:
        if log_file is not None:
            ts = datetime.now(UTC).isoformat(timespec="milliseconds")
            log_file.write(
                f"{ts}\t{r.status}\t{r.nbytes}\t{r.elapsed * 1000:.0f}ms\t{r.attempts}\t{r.url}\t{r.error}\n"
            )

    async def worker(client: httpx.AsyncClient, robots: RobotsCache) -> None:
        while True:
            url = await frontier.get()
            try:
                if stop.is_set():
                    continue
                if not await robots.allowed(url):
                    stats.robots_blocked += 1
                    record(FetchResult(url, 0, error="disallowed by robots.txt", attempts=0))
                    continue
                async with in_flight, gate.slot(host_of(url)):
                    stats.in_flight += 1
                    try:
                        r = await fetch(client, url, timeout=timeout, max_retries=max_retries)
                    finally:
                        stats.in_flight -= 1
                stats.fetched += 1
                stats.retries += r.attempts - 1
                stats.nbytes += r.nbytes
                record(r)
                if not r.ok:
                    stats.errors += 1
                    continue
                if "html" not in r.content_type:
                    continue
                final = canonicalize(r.final_url or url) or url
                seen.add(final)  # a redirect target counts as seen too
                title, text, links = parse_html(r.text, final)
                for link in links:
                    enqueue(link)
                if stop.is_set():
                    continue
                stats.pages += 1
                if stats.pages >= max_pages:
                    stop.set()
                await out.put(Page(final, title, text, datetime.now(UTC).isoformat(timespec="seconds")))
            finally:
                stats.queued = frontier.qsize()
                frontier.task_done()

    async def run() -> None:
        try:
            limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
            async with httpx.AsyncClient(  # one client = one connection pool for the whole crawl
                headers={"User-Agent": user_agent},
                follow_redirects=True,
                max_redirects=5,
                limits=limits,
                timeout=timeout,
                transport=transport,
            ) as client:
                robots = RobotsCache(client, user_agent, timeout)
                for u in seed_urls:
                    enqueue(u)
                async with asyncio.TaskGroup() as tg:
                    workers = [tg.create_task(worker(client, robots)) for _ in range(concurrency)]
                    drained = tg.create_task(frontier.join())
                    stopped = tg.create_task(stop.wait())
                    await asyncio.wait([drained, stopped], return_when=asyncio.FIRST_COMPLETED)
                    for t in [*workers, drained, stopped]:
                        t.cancel()  # clean stop: CancelledError at each worker's current await
        finally:
            await out.put(None)

    runner = asyncio.create_task(run())  # keep the reference: unreferenced tasks can be collected
    try:
        while (page := await out.get()) is not None:
            yield page
        await runner  # re-raise anything that went wrong inside
    finally:
        if not runner.done():
            runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner


def write_jsonl(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")
