"""Boolean search over sorted postings: two-pointer merges, and a set-based alternative.

    python -m findex.search index.bin "query" [--engine merge|set]

Query syntax (left to right, no precedence): ``a b`` = a AND b,
``a OR b``, ``a NOT b``.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from findex.index import IndexData
from findex.tokenize import tokenize

Engine = Literal["merge", "set"]


def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        if x == y:
            out.append(x)
            i += 1
            j += 1
        elif x < y:
            out.append(x)
            i += 1
        else:
            out.append(y)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_not(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """a AND NOT b."""
    i = j = 0
    out: list[int] = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


def set_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_not(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


OPS = {
    "merge": {"AND": merge_and, "OR": merge_or, "NOT": merge_not},
    "set": {"AND": set_and, "OR": set_or, "NOT": set_not},
}


def doc_ids(index: IndexData, term: str) -> list[int]:
    return [p.doc_id for p in index.postings.get(term, ())]


def boolean_search(index: IndexData, query: str, engine: Engine = "merge") -> list[int]:
    ops = OPS[engine]
    result: list[int] | None = None
    op = "AND"
    for word in query.split():
        if word in ("OR", "NOT", "AND"):
            op = word
            continue
        for term in tokenize(word):
            ids = doc_ids(index, term)
            result = ids if result is None else ops[op](result, ids)
            op = "AND"
    return result or []


def main(argv: list[str] | None = None) -> None:
    from findex import store

    p = argparse.ArgumentParser(prog="python -m findex.search")
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--engine", choices=["merge", "set"], default="merge")
    p.add_argument("--show", type=int, default=10)
    args = p.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = store.load(args.index)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    hits = boolean_search(index, args.query, args.engine)
    t_search = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    print(f"{len(hits)} documents match {args.query!r}")
    for d in hits[: args.show]:
        m = index.doc_meta[d]
        print(f"  [{d}] {m.title}  ({m.path})")
    print(
        f"load: {t_load:.2f} s, search: {t_search * 1000:.2f} ms ({args.engine}), "
        f"peak memory: {peak / 2**20:.1f} MiB"
    )


if __name__ == "__main__":
    main()
