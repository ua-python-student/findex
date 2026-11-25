"""The inverted index as a Python object, and its one-pass builder.

python -m findex.index data/ --out index.bin [--positions] [--format pickle|binary]
"""

from __future__ import annotations

import argparse
import functools
import logging
import time
import tracemalloc
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.models import DocMeta, Posting, title_of
from findex.timing import timed
from findex.tokenize import tokenize, tokenize_with_positions

log = logging.getLogger(__name__)


class Index(Mapping[str, list[Posting]]):
    """term -> postings (sorted by doc_id). Behaves like a read-only dict.

    ``collections.abc.Mapping`` gives ``keys``, ``items``, ``values`` and
    ``get`` for free from ``__getitem__``/``__len__``/``__iter__``.
    """

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        texts: dict[int, bytes] | None = None,
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self._texts = texts or {}  # zlib-compressed document texts (snippets, document view)

    # --- the Mapping protocol ---------------------------------------------------
    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]  # KeyError if absent, as a Mapping must

    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __contains__(self, term: object) -> bool:  # one dict lookup instead of Mapping's try/except
        return term in self._postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_}, positions={self.has_positions})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return (
            self._postings == other._postings
            and self._doc_lengths == other._doc_lengths
            and self._doc_meta == other._doc_meta
        )

    def __getstate__(self) -> dict[str, object]:  # pickle: skip caches (an lru_cache can't be pickled)
        return {k: v for k, v in self.__dict__.items() if k.startswith("_") and k != "_ranker"}

    __hash__ = None  # type: ignore[assignment]  # value equality on a big container: unhashable

    # --- derived attributes -----------------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        return sum(self._doc_lengths.values()) / max(self.num_docs, 1)

    @functools.cached_property
    def has_positions(self) -> bool:
        return any(plist[0].positions for plist in self._postings.values() if plist)

    @property
    def doc_lengths(self) -> dict[int, int]:
        return self._doc_lengths

    @property
    def doc_meta(self) -> dict[int, DocMeta]:
        return self._doc_meta

    @property
    def texts(self) -> dict[int, bytes]:
        return self._texts

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def title(self, doc_id: int) -> str:
        return self._doc_meta[doc_id].title

    def doc_text(self, doc_id: int) -> str:
        blob = self._texts.get(doc_id)
        return zlib.decompress(blob).decode() if blob is not None else ""

    def close(self) -> None:
        """Release the big structures (called by the open_index context manager)."""
        self._postings = {}
        self._texts = {}
        self.__dict__.pop("avg_doc_length", None)
        self.__dict__.pop("has_positions", None)
        self.__dict__.pop("_ranker", None)  # the per-index search cache (findex.ranking)


@timed
def build_index(docs: Iterable[Document], positions: bool = False, keep_text: bool = True) -> Index:
    """One pass: per document, count terms in a Counter, then flush it into the postings.

    Documents arrive in increasing doc_id order, so appending keeps every
    postings list sorted by doc_id without a separate sort.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    texts: dict[int, bytes] = {}
    for doc in docs:
        n = 0
        if positions:
            pos: defaultdict[str, list[int]] = defaultdict(list)
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
        if keep_text:
            texts[doc.doc_id] = zlib.compress(doc.text.encode(), 6)
    log.info("indexed %d docs, %d terms", len(doc_lengths), len(postings))
    return Index(dict(postings), doc_lengths, doc_meta, texts)


def main(argv: list[str] | None = None) -> None:
    from findex import store

    p = argparse.ArgumentParser(prog="python -m findex.index")
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--positions", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--format", choices=["pickle", "binary"], default="binary")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

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
    print(f"{index!r} -> {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB, {args.format})")
    print(f"build: {t_build:.2f} s, total: {elapsed:.2f} s, peak memory: {peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
