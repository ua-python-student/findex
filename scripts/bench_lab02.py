"""Lab 2 measurements: store formats, merge vs set, memory study of posting representations."""

from __future__ import annotations

import gc
import pickle
import statistics
import sys
import tempfile
import time
import timeit
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex import store
from findex.corpus import iter_documents
from findex.index import build_index
from findex.search import OPS, doc_ids
from findex.tokenize import tokenize

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
TMP = Path(tempfile.mkdtemp())


def median_time(fn, runs: int = 3) -> float:
    ts = []
    for _ in range(runs):
        gc.collect()
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


# ---------------------------------------------------------------- store formats
print("## Save / load (no tracemalloc, median of 3)\n")
print("| Format | Positions | File size (MiB) | Save (s) | Load (s) |")
print("|---|---|---|---|---|")
indexes = {
    False: build_index(iter_documents(ROOT)),
    True: build_index(iter_documents(ROOT), positions=True),
}
for pos in (False, True):
    for fmt in ("pickle", "binary"):
        path = TMP / f"ix.{fmt}.{pos}"
        t_save = median_time(lambda: store.save(indexes[pos], path, fmt=fmt))
        t_load = median_time(lambda: store.load(path))
        assert store.load(path) == indexes[pos]
        print(
            f"| {fmt} | {'yes' if pos else 'no'} | {path.stat().st_size / 2**20:.1f} | {t_save:.2f} | {t_load:.2f} |"
        )

# ---------------------------------------------------------------- merge vs set
index = indexes[False]
by_df = sorted(index.postings, key=lambda t: len(index.postings[t]))
common = by_df[-2:]
rare = [t for t in by_df if len(index.postings[t]) >= 2][:2]  # rarest with df>=2
mid = [t for t in by_df if 100 <= len(index.postings[t]) <= 120][:1]
print("\n## Boolean AND/OR: merge vs set (timeit, µs per query)\n")
print("| Terms (df) | Op | merge (µs) | set (µs) | result |")
print("|---|---|---|---|---|")
for pair in (common, rare, [common[-1], mid[0]]):
    a, b = (doc_ids(index, t) for t in pair)
    for op in ("AND", "OR"):
        res = {}
        for eng in ("merge", "set"):
            fn = OPS[eng][op]
            n, total = timeit.Timer(lambda: fn(a, b)).autorange()
            res[eng] = total / n * 1e6
        out = OPS["merge"][op](a, b)
        assert out == OPS["set"][op](a, b)
        label = " & ".join(f"`{t}` ({len(index.postings[t])})" for t in pair)
        print(f"| {label} | {op} | {res['merge']:.1f} | {res['set']:.1f} | {len(out)} docs |")
del indexes, index
gc.collect()


# ---------------------------------------------------------------- memory study
@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlotPosting:
    doc_id: int
    tf: int


def build(kind: str):
    if kind == "arrays":
        post: defaultdict = defaultdict(lambda: (array("I"), array("I")))
        for doc in iter_documents(ROOT):
            for term, tf in Counter(tokenize(doc.text)).items():
                ids, tfs = post[term]
                ids.append(doc.doc_id)
                tfs.append(tf)
        return dict(post)
    cls = PlainPosting if kind == "plain" else SlotPosting
    lists: defaultdict = defaultdict(list)
    for doc in iter_documents(ROOT):
        for term, tf in Counter(tokenize(doc.text)).items():
            lists[term].append(cls(doc.doc_id, tf))
    return dict(lists)


print("\n## Memory study (postings only; tracemalloc)\n")
print(
    "| Postings representation | Peak memory, build (MiB) | Retained after build (MiB) | Pickle file (MiB) | Load (s) |"
)
print("|---|---|---|---|---|")
labels = {
    "plain": "`list[Posting]`, plain `@dataclass`",
    "slots": "`list[Posting]`, `slots=True`",
    "arrays": "`array('I')` pairs, no objects",
}
for kind in ("plain", "slots", "arrays"):
    gc.collect()
    tracemalloc.start()
    obj = build(kind)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    path = TMP / f"mem.{kind}"
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    del obj
    gc.collect()

    def _load(path: Path = path) -> None:
        with open(path, "rb") as f:
            pickle.load(f)

    t = median_time(_load)
    print(
        f"| {labels[kind]} | {peak / 2**20:.1f} | {cur / 2**20:.1f} | {path.stat().st_size / 2**20:.1f} | {t:.2f} |"
    )
