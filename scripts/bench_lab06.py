"""Crawl the local mirror at several concurrency levels (each run in a fresh process).

Start the mirror first:  uv run python scripts/mirror_server.py --port 8765
"""

import asyncio
import json
import resource
import subprocess
import sys
import time

from findex.crawler import CrawlStats, crawl

SEED = "http://127.0.0.1:8765/"
HOSTS = ["127.0.0.1", "127.0.0.2", "127.0.0.3"]


async def one(concurrency: int, per_host: int, delay: float, max_pages: int) -> dict[str, float]:
    stats = CrawlStats()
    t = time.perf_counter()
    n = 0
    async for _ in crawl(
        [SEED],
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=per_host,
        per_host_delay=delay,
        allowed_domains=HOSTS,
        stats=stats,
    ):
        n += 1
    wall = time.perf_counter() - t
    return {
        "concurrency": concurrency,
        "per_host": per_host,
        "delay": delay,
        "pages": n,
        "wall": wall,
        "pps": n / wall,
        "errors": stats.errors,
        "retries": stats.retries,
        "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        c, ph, d, m = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
        print(json.dumps(asyncio.run(one(c, ph, d, m))))
        sys.exit()
    configs = [(1, 2, 0.1), (5, 2, 0.1), (20, 2, 0.1), (20, 20, 0.0)]
    print(
        "| Concurrency | Per-host limit | Per-host delay (s) | Pages | Wall (s) | Pages/s | Errors | Retries | Peak RSS (MiB) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for c, ph, d in configs:
        r = json.loads(
            subprocess.run(
                [sys.executable, __file__, str(c), str(ph), str(d), "300"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        print(
            f"| {c} | {ph} | {d} | {r['pages']} | {r['wall']:.1f} | {r['pps']:.1f} | {r['errors']} | {r['retries']} | {r['rss']:.0f} |",
            flush=True,
        )
