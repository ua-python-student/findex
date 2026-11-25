"""A local mirror of the corpus as a 3-host website, for crawling without bothering anyone.

    uv run python scripts/mirror_server.py [--port 8765] [--latency 0.05:0.2]

Hosts (all loopback): 127.0.0.1 = Python docs, 127.0.0.2 = Django, 127.0.0.3 = NumPy.
Each document becomes /docs/<path>.html; every section has an index page; each page
links to neighbours in its directory, to the other hosts' home pages, to an external
site (must be filtered by the allowlist) and to URLs that differ only by fragment,
trailing slash or query order (must be deduplicated by canonicalisation).

Misbehaviour, on purpose: random latency; every 50th request -> 429 with Retry-After: 1;
every 70th -> 503; /loop/a <-> /loop/b redirect loop; /private/ is disallowed in robots.txt;
/docs/missing.html -> 404.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import itertools
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

import uvicorn

from findex.models import title_of

HOSTS = {"127.0.0.1": "cpython", "127.0.0.2": "django", "127.0.0.3": "numpy"}
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]


def build_site(data: Path, port: int) -> dict[str, dict[str, bytes]]:
    site: dict[str, dict[str, bytes]] = {}
    for host, project in HOSTS.items():
        pages: dict[str, bytes] = {}
        root = data / project
        files = sorted(root.rglob("*.txt"))
        by_dir: defaultdict[Path, list[Path]] = defaultdict(list)
        for f in files:
            by_dir[f.parent.relative_to(root)].append(f)
        others = [f"http://{h}:{port}/" for h in HOSTS if h != host]

        def sec(d: Path) -> str:
            return "root" if str(d) == "." else str(d)

        def href(f: Path) -> str:
            return "/docs/" + str(f.relative_to(root).with_suffix(".html"))  # noqa: B023

        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            title = title_of(text, f.stem)
            sib = by_dir[f.parent.relative_to(root)]
            i = sib.index(f)
            neigh = [sib[j] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(sib) and j != i]
            links = [f'<a href="{href(n)}">{html.escape(n.stem)}</a>' for n in neigh]
            links += [
                f'<a href="{href(f)}#top">self, with fragment</a>',
                '<a href="/?b=2&a=1">home</a>',
                '<a href="/?a=1&b=2">home, other query order</a>',
                '<a href="https://example.com/">external</a>',
                f'<a href="/sections/{sec(f.parent.relative_to(root))}/">section</a>',
            ]
            links += [f'<a href="{o}">{o}</a>' for o in others]
            body = "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n"))
            pages[href(f)] = (
                f"<!doctype html><html><head><title>{html.escape(title)}</title>"
                f"<style>p{{max-width:70ch}}</style><script>var x=1;</script></head>"
                f"<body><nav>{' | '.join(links)}</nav><main>{body}</main></body></html>"
            ).encode()
        for d, fs in by_dir.items():
            items = "".join(f'<li><a href="{href(f)}">{html.escape(f.stem)}</a></li>' for f in fs)
            pages[f"/sections/{sec(d)}"] = (
                f"<html><head><title>{project}: {sec(d)}</title></head><body><ul>{items}</ul></body></html>".encode()
            )
        sections = "".join(
            f'<li><a href="/sections/{sec(d)}/">{sec(d)}</a></li>' for d in sorted(by_dir, key=str)
        )
        pages["/"] = (
            f"<html><head><title>{project} mirror</title></head><body><h1>{project}</h1><ul>{sections}</ul>"
            f'<a href="/private/secret.html">private</a> <a href="/loop/a">loop</a> '
            f'<a href="/docs/missing.html">missing</a> {" ".join(f"<a href={o}>{o}</a>" for o in others)}</body></html>'
        ).encode()
        pages["/robots.txt"] = b"User-agent: *\nDisallow: /private/\n"
        site[host] = pages
    return site


def make_app(
    site: dict[str, dict[str, bytes]], latency: tuple[float, float]
) -> Callable[..., Awaitable[None]]:
    counter = itertools.count(1)
    rng = random.Random(42)

    async def app(
        scope: Scope, receive: Callable[[], Awaitable[Message]], send: Callable[[Message], Awaitable[None]]
    ) -> None:
        if scope["type"] != "http":
            return
        host = dict(scope["headers"]).get(b"host", b"").decode().split(":")[0]
        path = scope["path"].rstrip("/") or "/"
        n = next(counter)
        await asyncio.sleep(rng.uniform(*latency))
        headers = [(b"content-type", b"text/html; charset=utf-8")]
        if path == "/robots.txt":
            status, body, headers = 200, site[host][path], [(b"content-type", b"text/plain")]
        elif n % 50 == 0:
            status, body, headers = 429, b"slow down", headers + [(b"retry-after", b"1")]
        elif n % 70 == 0:
            status, body = 503, b"unavailable"
        elif path == "/loop/a":
            status, body, headers = 302, b"", [(b"location", b"/loop/b")]
        elif path == "/loop/b":
            status, body, headers = 302, b"", [(b"location", b"/loop/a")]
        elif host in site and path in site[host]:
            status, body = 200, site[host][path]
        else:
            status, body = 404, b"not found"
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--latency", default="0.05:0.2", help="min:max seconds per response")
    a = p.parse_args()
    lo, hi = (float(x) for x in a.latency.split(":"))
    site = build_site(a.data, a.port)
    print(f"mirror: {sum(len(v) for v in site.values())} pages on {list(HOSTS)} port {a.port}", flush=True)
    uvicorn.run(make_app(site, (lo, hi)), host="0.0.0.0", port=a.port, log_level="warning", backlog=1024)


if __name__ == "__main__":
    main()
