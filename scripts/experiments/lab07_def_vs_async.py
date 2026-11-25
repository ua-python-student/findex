"""M2: 20 concurrent /search requests + a /health probe sent while they run, for each route style.

Start the server first:  INDEX_PATH=index.bin findex serve --port 8801
"""

import asyncio
import statistics
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8801"
ROUTES = {
    "async def (blocking)": "/_bench/search-async",
    "def (thread pool)": "/search",
    "async def + to_thread": "/_bench/search-to-thread",
}
QUERY = "python OR django OR numpy OR array OR model"  # broad: ~1,400 matches, real CPU work


async def one(client: httpx.AsyncClient, route: str) -> tuple[float, float, list[float]]:
    async def timed_get(url: str, **params: str | int) -> float:
        t = time.perf_counter()
        r = await client.get(url, params=params)
        r.raise_for_status()
        return time.perf_counter() - t

    for i in range(3):  # warm-up, distinct k defeats the lru_cache
        await timed_get(route, q=QUERY, k=90 + i)
    t0 = time.perf_counter()
    searches = [asyncio.create_task(timed_get(route, q=QUERY, k=10 + i)) for i in range(20)]
    await asyncio.sleep(0.05)  # the searches are in flight; now probe /health
    health = await timed_get("/health")
    lats = await asyncio.gather(*searches)
    return time.perf_counter() - t0, health, lats


async def main() -> None:
    print("| Route | 20 requests, wall (s) | mean latency (ms) | max (ms) | /health during load (ms) |")
    print("|---|---|---|---|---|")
    async with httpx.AsyncClient(base_url=BASE, timeout=120) as client:
        for name, route in ROUTES.items():
            walls, healths, means, maxes = [], [], [], []
            for _ in range(3):
                w, h, lats = await one(client, route)
                walls.append(w)
                healths.append(h)
                means.append(statistics.mean(lats))
                maxes.append(max(lats))
            print(
                f"| {name} | {statistics.median(walls):.2f} | {statistics.median(means) * 1000:.0f} | "
                f"{statistics.median(maxes) * 1000:.0f} | {statistics.median(healths) * 1000:.0f} |"
            )


asyncio.run(main())
