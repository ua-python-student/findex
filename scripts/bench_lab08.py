"""Lab 3 scorer vs NumPy scorer (ranking only: parse + match + score + top-k), and load times."""

import statistics
import time
import timeit
from pathlib import Path

from findex import store
from findex.fast import NumpyIndex
from findex.ranking import rank
from findex.scoring import BM25

PATH = Path("index.bin")


def best_ms(fn, repeat: int = 7) -> float:
    n, _ = timeit.Timer(fn).autorange()
    return min(timeit.Timer(fn).repeat(repeat, n)) / n * 1000


ix = store.load(PATH)
nx = NumpyIndex.open(PATH)
bm = BM25()
print("| Query | Docs matched | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |")
print("|---|---|---|---|---|")
for label, q in (
    ("common term `the`", "the"),
    ("rare term `pickletools`", "pickletools"),
    ("3-term `python OR thread OR process`", "python OR thread OR process"),
    ("3-term AND `python thread process`", "python thread process"),
):
    total, a = rank(ix, q, bm, 10)
    _, b = nx.rank(q, "bm25", 10)
    assert [d for d, _ in a] == [d for d, _ in b]
    t_py = best_ms(lambda: rank(ix, q, bm, 10))
    t_np = best_ms(lambda: nx.rank(q, "bm25", 10))
    print(f"| {label} | {total:,} | {t_py:.3f} | {t_np:.3f} | {t_py / t_np:.1f}× |")

print("\n| Load | Time (s, median of 5) |\n|---|---|")
for label, fn in (
    ("`store.load` (Lab 2 binary → `Posting` objects)", lambda: store.load(PATH)),
    ("`NumpyIndex.open` (`np.memmap` views)", lambda: NumpyIndex.open(PATH)),
):
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    print(f"| {label} | {statistics.median(ts):.3f} |")
