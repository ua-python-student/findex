"""Crawler tests: no network — httpx.MockTransport plays the web server."""

import asyncio
import io
from collections import Counter
from collections.abc import Callable

import httpx
import pytest

from findex.crawler import CrawlStats, FetchResult, crawl, fetch
from findex.crawler.fetch import backoff_delay, retry_after_seconds
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, domain_allowed


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Example.COM:80/a/b/?z=1&a=2#frag", "http://example.com/a/b?a=2&z=1"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com:8443//x//y/", "https://example.com:8443/x/y"),
        ("mailto:me@example.com", None),
        ("javascript:void(0)", None),
    ],
)
def test_canonicalize(raw: str, expected: str | None) -> None:
    assert canonicalize(raw) == expected


def test_canonicalize_relative_and_domains() -> None:
    assert canonicalize("../c.html#x", "http://h.org/a/b/page.html") == "http://h.org/a/c.html"
    allowed = frozenset({"docs.python.org"})
    assert domain_allowed("https://docs.python.org/3/", allowed)
    assert not domain_allowed("https://python.org/", allowed)


def test_backoff_and_retry_after() -> None:
    assert 0.5 <= backoff_delay(0) <= 1.0
    assert 4.0 <= backoff_delay(3) <= 4.5
    assert backoff_delay(20, cap=10) <= 10.5
    assert retry_after_seconds("7") == 7.0
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past
    assert retry_after_seconds(None) is None


Handler = Callable[[httpx.Request], httpx.Response]


def run_fetch(handler: Handler, max_retries: int = 3) -> tuple[FetchResult, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def go() -> FetchResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch(c, "http://t/x", sleep=fake_sleep, max_retries=max_retries)

    return asyncio.run(go()), sleeps


def test_retries_5xx_then_succeeds() -> None:
    calls = Counter[str]()

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, text="ok")

    r, sleeps = run_fetch(handler)
    assert r.status == 200 and r.attempts == 3 and len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 1.5  # base*2**n + jitter in [0, base]


def test_429_honours_retry_after() -> None:
    calls = Counter[str]()

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "4"})
        return httpx.Response(200, text="ok")

    r, sleeps = run_fetch(handler)
    assert r.ok and sleeps == [4.0]


def test_404_is_not_retried() -> None:
    r, sleeps = run_fetch(lambda _req: httpx.Response(404))
    assert r.status == 404 and r.attempts == 1 and sleeps == [] and r.error == "HTTP 404"


def test_gives_up_after_max_retries() -> None:
    r, sleeps = run_fetch(lambda _req: httpx.Response(500), max_retries=3)
    assert r.status == 500 and r.attempts == 4 and len(sleeps) == 3 and not r.ok


def test_connection_error_is_retried() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    r, sleeps = run_fetch(handler, max_retries=2)
    assert r.status == 0 and r.error == "ConnectError" and len(sleeps) == 2


def test_timeout() -> None:
    async def slow(req: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200)

    async def go() -> FetchResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as c:

            async def no_sleep(s: float) -> None:
                pass

            return await fetch(c, "http://t/", timeout=0.05, max_retries=1, sleep=no_sleep)

    r = asyncio.run(go())
    assert r.error == "timeout" and r.attempts == 2


def test_robots_cache_fetches_once() -> None:
    hits = Counter[str]()

    def handler(req: httpx.Request) -> httpx.Response:
        hits[req.url.path] += 1
        return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")

    async def go() -> list[bool]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            rc = RobotsCache(c, "findex-test")
            urls = ["http://h/a", "http://h/private/x", "http://h/b"] * 5
            return await asyncio.gather(*(rc.allowed(u) for u in urls))

    res = asyncio.run(go())
    assert res[:3] == [True, False, True] and hits["/robots.txt"] == 1


# --- a whole tiny website -------------------------------------------------------
SITE = {
    "/": '<title>Home</title><a href="/a">a</a><a href="/b?y=2&x=1">b</a><a href="/b?x=1&y=2#f">b again</a>'
    '<a href="https://other.org/">ext</a><a href="/private/p">p</a><a href="/missing">m</a>',
    "/a": '<title>A</title><p>alpha text</p><a href="/">home</a><a href="/c/">c</a><script>bad()</script>',
    "/b": '<title>B</title><a href="/a#x">a</a>',
    "/c": "<title>C</title>gamma",
}


def site_handler(fetched: Counter[str]) -> Handler:
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path.rstrip("/") or "/"
        fetched[str(req.url)] += 1
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        if req.url.host != "site.test":
            raise AssertionError(f"left the allowlist: {req.url}")
        if path in SITE:
            return httpx.Response(200, text=SITE[path], headers={"content-type": "text/html"})
        return httpx.Response(404)

    return handler


def run_crawl(max_pages: int, concurrency: int = 3) -> tuple[list[str], Counter[str], CrawlStats, str]:
    fetched = Counter[str]()
    stats = CrawlStats()
    log = io.StringIO()

    async def go() -> list[str]:
        pages = crawl(
            ["http://site.test/"],
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=2,
            per_host_delay=0,
            transport=httpx.MockTransport(site_handler(fetched)),
            stats=stats,
            log_file=log,
        )
        return [p.url async for p in pages]

    return asyncio.run(go(), debug=True), fetched, stats, log.getvalue()


def test_crawl_whole_site_no_duplicates() -> None:
    urls, fetched, stats, log = run_crawl(100)
    assert sorted(urls) == [
        "http://site.test/",
        "http://site.test/a",
        "http://site.test/b?x=1&y=2",
        "http://site.test/c",
    ]
    assert all(n == 1 for n in fetched.values())  # nothing fetched twice
    assert stats.robots_blocked == 1 and "disallowed by robots.txt" in log
    assert stats.errors == 1 and "HTTP 404" in log


def test_crawl_stops_at_max_pages() -> None:
    urls, _, stats, _ = run_crawl(2, concurrency=1)
    assert len(urls) == 2 and stats.pages == 2


def test_page_text_has_no_script() -> None:
    async def go() -> str:
        async for p in crawl(
            ["http://site.test/a"],
            max_pages=1,
            per_host_delay=0,
            transport=httpx.MockTransport(site_handler(Counter())),
        ):
            return p.text
        return ""

    text = asyncio.run(go())
    assert "alpha text" in text and "bad()" not in text
