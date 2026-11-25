"""A small oha-like load generator: N concurrent keep-alive connections for D seconds.

usage: python scripts/loadtest.py URL_TEMPLATE [-c 50] [-z 30] [--queries N]
URL_TEMPLATE contains {q}; queries rotate through N generated two-term queries
(more than the server's 256-entry lru_cache, so most requests are cache misses).
"""

import argparse
import asyncio
import random
import statistics
import time
import urllib.parse

import httpx


def make_queries(n: int, seed: int = 7) -> list[str]:
    words = (
        "loop event task queue thread dict list string file socket error type class module import "
        "array dtype shape model query field view form cache template signal migration request response "
        "test logging json pickle path regex date time random math decimal enum iterator generator "
        "context decorator property slots hash buffer memory unicode bytes encode async await"
    ).split()
    rng = random.Random(seed)
    return [f"python {rng.choice(words)} OR {rng.choice(words)}" for _ in range(n)]


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("-c", type=int, default=50)
    p.add_argument("-z", type=float, default=30.0)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--no-keepalive", action="store_true", help="new connection per request")
    a = p.parse_args()
    queries = make_queries(a.queries)
    lat: list[float] = []
    errors = 0
    counter = iter(range(10**9))
    deadline = time.perf_counter() + a.z
    limits = httpx.Limits(max_connections=a.c, max_keepalive_connections=0 if a.no_keepalive else a.c)
    async with httpx.AsyncClient(limits=limits, timeout=60) as client:

        async def worker() -> None:
            nonlocal errors
            while time.perf_counter() < deadline:
                q = queries[next(counter) % len(queries)]
                t = time.perf_counter()
                try:
                    r = await client.get(a.url.format(q=urllib.parse.quote(q)))
                    ok = r.status_code == 200
                except httpx.HTTPError:
                    ok = False
                if ok:
                    lat.append(time.perf_counter() - t)
                else:
                    errors += 1

        t0 = time.perf_counter()
        async with asyncio.TaskGroup() as tg:
            for _ in range(a.c):
                tg.create_task(worker())
        wall = time.perf_counter() - t0
    qs = statistics.quantiles(lat, n=100)
    print(
        f"requests={len(lat)} errors={errors} rps={len(lat) / wall:.1f} "
        f"p50={qs[49] * 1000:.1f} p95={qs[94] * 1000:.1f} p99={qs[98] * 1000:.1f} max={max(lat) * 1000:.1f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
