"""Corpus statistics in one streaming pass.

python -m findex.stats data/ [--limit N] [--eager] [--top 50]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass
class Stats:
    documents: int
    tokens: int
    vocabulary: int
    top: list[tuple[str, int]]


def lazy_stats(docs: Iterable[Document], top: int = 50) -> Stats:
    """Generators all the way down: only the Counter grows."""
    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))
    return Stats(n_docs, counts.total(), len(counts), counts.most_common(top))


def eager_stats(docs: Iterable[Document], top: int = 50) -> Stats:
    """The deliberately eager version: every stage is a list."""
    all_docs = list(docs)  # whole corpus text in RAM
    all_tokens = [list(tokenize(d.text)) for d in all_docs]  # every token as a str object
    flat = [t for toks in all_tokens for t in toks]  # and once more, flattened
    counts = Counter(flat)
    return Stats(len(all_docs), len(flat), len(counts), counts.most_common(top))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m findex.stats")
    p.add_argument("root", type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eager", action="store_true", help="run the list-based version")
    p.add_argument("--top", type=int, default=50)
    args = p.parse_args(argv)

    docs = iter_documents(args.root)
    if args.limit is not None:
        docs = islice(docs, args.limit)

    tracemalloc.start()
    t0 = time.perf_counter()
    stats = (eager_stats if args.eager else lazy_stats)(docs, args.top)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"documents:  {stats.documents:,}")
    print(f"tokens:     {stats.tokens:,}")
    print(f"vocabulary: {stats.vocabulary:,}")
    print(f"top {args.top}:")
    for i, (term, n) in enumerate(stats.top, 1):
        print(f"  {i:>2}. {term:<16} {n:,}")
    print(f"mode: {'eager' if args.eager else 'lazy'}")
    print(f"elapsed: {elapsed:.2f} s")
    print(f"peak memory: {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
