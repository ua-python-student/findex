"""A shared Counter updated from threads: does `+=` lose updates? Run on 3.13 and 3.13t."""

import sys
import threading
from collections import Counter

N_THREADS, N = 4, 200_000
shared: Counter[str] = Counter()
lock = threading.Lock()


def unsafe() -> None:
    for i in range(N):
        shared[str(i % 100)] += 1  # read, add, write: three steps, a switch can land between them


def safe() -> None:
    for i in range(N):
        with lock:
            shared[str(i % 100)] += 1


gil = getattr(sys, "_is_gil_enabled", lambda: True)()
for fn in (unsafe, safe):
    shared.clear()
    ts = [threading.Thread(target=fn) for _ in range(N_THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    total = shared.total()
    lost = N_THREADS * N - total
    print(
        f"python {sys.version.split()[0]} GIL={gil}: {fn.__name__:6} total={total:,} expected={N_THREADS * N:,} lost={lost:,}"
    )
