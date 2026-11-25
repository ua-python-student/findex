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

from findex.index import Index
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


def doc_ids(index: Index, term: str) -> list[int]:
    return [p.doc_id for p in index.get(term, ())]


def boolean_search(index: Index, query: str, engine: Engine = "merge") -> list[int]:
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
    import logging

    from findex.ranking import search
    from findex.scoring import SCORERS
    from findex.store import open_index

    p = argparse.ArgumentParser(prog="python -m findex.search")
    p.add_argument("index", type=Path)
    p.add_argument("query", nargs="+", help="one or more queries; repeated ones hit the cache")
    p.add_argument("--rank", choices=["bm25", "tfidf", "none"], default="bm25")
    p.add_argument("--engine", choices=["merge", "set"], default="merge", help="for --rank none")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG log: @timed, cache hits")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(name)s: %(message)s"
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    with open_index(args.index) as index:
        print(f"{index!r} loaded in {time.perf_counter() - t0:.2f} s")
        for q in args.query:
            t1 = time.perf_counter()
            if args.rank == "none":
                hits = boolean_search(index, q, args.engine)
                ms = (time.perf_counter() - t1) * 1000
                print(f"{len(hits)} documents match {q!r} ({ms:.2f} ms, {args.engine})")
                for d in hits[: args.k]:
                    print(f"  [{d}] {index.title(d)}  ({index.doc_meta[d].path})")
                continue
            results = search(index, q, SCORERS[args.rank], args.k)
            ms = (time.perf_counter() - t1) * 1000
            print(f"\n{q!r}: top {len(results)} by {args.rank} ({ms:.2f} ms)")
            for i, r in enumerate(results, 1):
                print(f"{i:>2}. {r.score:7.3f}  {r.title}  ({r.path})")
                print(f"           {r.snippet}")
    _, peak = tracemalloc.get_traced_memory()
    print(f"peak memory: {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
