"""Inverted index built in one streaming pass.

python -m findex.index data/ --out index.bin [--positions] [--format pickle|binary]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.models import DocMeta, Posting, title_of
from findex.tokenize import tokenize, tokenize_with_positions


@dataclass
class IndexData:
    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)


def build_index(docs: Iterable[Document], positions: bool = False) -> IndexData:
    """One pass: per document, count terms in a Counter, then flush it into the postings.

    Documents arrive in increasing doc_id order, so appending keeps every
    postings list sorted by doc_id without a separate sort.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for doc in docs:
        if positions:
            pos: defaultdict[str, list[int]] = defaultdict(list)
            n = 0
            for i, tok in tokenize_with_positions(doc.text):
                pos[tok].append(i)
                n = i + 1
            for term, ps in pos.items():
                postings[term].append(Posting(doc.doc_id, len(ps), tuple(ps)))
        else:
            counts = Counter(tokenize(doc.text))
            n = counts.total()
            for term, tf in counts.items():
                postings[term].append(Posting(doc.doc_id, tf))
        doc_lengths[doc.doc_id] = n
        doc_meta[doc.doc_id] = DocMeta(doc.path, title_of(doc.text, doc.path))
    return IndexData(dict(postings), doc_lengths, doc_meta)


def main(argv: list[str] | None = None) -> None:
    from findex import store

    p = argparse.ArgumentParser(prog="python -m findex.index")
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--positions", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--format", choices=["pickle", "binary"], default="binary")
    args = p.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    docs = iter_documents(args.root)
    if args.limit is not None:
        docs = islice(docs, args.limit)
    index = build_index(docs, positions=args.positions)
    t_build = time.perf_counter() - t0
    store.save(index, args.out, fmt=args.format)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    print(
        f"indexed {len(index.doc_lengths):,} docs, {len(index.postings):,} terms "
        f"-> {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB, {args.format})"
    )
    print(f"build: {t_build:.2f} s, total: {elapsed:.2f} s, peak memory: {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
