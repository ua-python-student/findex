"""Parallel index construction: map (build_partial per chunk of paths) + reduce (merge).

Workers receive *paths* and a doc_id range, read their own files, and send back
compact postings (``array('I')`` buffers, which pickle as raw bytes) instead of
millions of ``Posting`` objects.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
import zlib
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from findex.corpus import read_text, txt_paths
from findex.index import Index
from findex.models import DocMeta, Posting, title_of
from findex.timing import timed
from findex.tokenize import tokenize, tokenize_with_positions

log = logging.getLogger(__name__)

ExecutorName = Literal["serial", "threads", "processes"]


@dataclass
class PartialIndex:
    """Postings for one contiguous doc_id range: term -> (doc_ids, tfs, positions)."""

    start_id: int
    postings: dict[str, tuple[array[int], array[int], array[int]]] = field(
        default_factory=dict[str, tuple[array[int], array[int], array[int]]]
    )
    doc_lengths: dict[int, int] = field(default_factory=dict[int, int])
    doc_meta: dict[int, DocMeta] = field(default_factory=dict[int, DocMeta])
    texts: dict[int, bytes] = field(default_factory=dict[int, bytes])


def build_partial(root: Path, paths: Sequence[Path], start_id: int, positions: bool = True) -> PartialIndex:
    """Module-level and argument-picklable, so a ProcessPoolExecutor can run it."""
    part = PartialIndex(start_id)
    post: defaultdict[str, tuple[array[int], array[int], array[int]]] = defaultdict(
        lambda: (array("I"), array("I"), array("I"))
    )
    for doc_id, path in enumerate(paths, start_id):
        text = read_text(path)
        n = 0
        if positions:
            pos: defaultdict[str, list[int]] = defaultdict(list)
            for i, tok in tokenize_with_positions(text):
                pos[tok].append(i)
                n = i + 1
            for term, ps in pos.items():
                ids, tfs, flat = post[term]
                ids.append(doc_id)
                tfs.append(len(ps))
                flat.extend(ps)
        else:
            counts = Counter(tokenize(text))
            n = counts.total()
            for term, tf in counts.items():
                ids, tfs, _ = post[term]
                ids.append(doc_id)
                tfs.append(tf)
        part.doc_lengths[doc_id] = n
        rel = str(path.relative_to(root))
        part.doc_meta[doc_id] = DocMeta(rel, title_of(text, rel))
        part.texts[doc_id] = zlib.compress(text.encode(), 6)
    part.postings = dict(post)
    return part


@timed
def merge(partials: Iterable[PartialIndex]) -> Index:
    """Reduce: concatenate per-term postings in doc_id order and materialise Posting objects.

    Chunks cover disjoint, increasing doc_id ranges, so concatenating them in
    ``start_id`` order *is* the sorted merge.
    """
    ordered = sorted(partials, key=lambda p: p.start_id)
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    texts: dict[int, bytes] = {}
    for part in ordered:
        for term, (ids, tfs, flat) in part.postings.items():
            plist = postings[term]
            if flat:
                k = 0
                for d, tf in zip(ids, tfs, strict=True):
                    plist.append(Posting(d, tf, tuple(flat[k : k + tf])))
                    k += tf
            else:
                plist.extend(Posting(d, tf) for d, tf in zip(ids, tfs, strict=True))
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)
        texts.update(part.texts)
    return Index(dict(postings), doc_lengths, doc_meta, texts)


def chunked[T](items: Sequence[T], n_chunks: int) -> list[tuple[int, Sequence[T]]]:
    """Split into ~n_chunks contiguous slices; returns (start_offset, slice)."""
    n_chunks = max(1, min(n_chunks, len(items)))
    size, rem = divmod(len(items), n_chunks)
    out: list[tuple[int, Sequence[T]]] = []
    start = 0
    for i in range(n_chunks):
        end = start + size + (1 if i < rem else 0)
        out.append((start, items[start:end]))
        start = end
    return out


@dataclass
class BuildTimings:
    map_s: float = 0.0
    merge_s: float = 0.0


def make_executor(kind: ExecutorName, workers: int) -> Executor | None:
    if kind == "threads":
        return ThreadPoolExecutor(max_workers=workers)
    if kind == "processes":
        # spawn: a fresh interpreter per worker — safe with threads, identical on Linux/macOS/Windows.
        # It re-imports __main__, hence the `if __name__ == "__main__"` rule for scripts.
        return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    return None


def build_parallel(
    root: Path,
    workers: int = os.cpu_count() or 1,
    executor: ExecutorName = "processes",
    positions: bool = True,
    chunks_per_worker: int = 4,
    timings: BuildTimings | None = None,
) -> Index:
    paths = list(txt_paths(root))
    t0 = time.perf_counter()
    if executor == "serial":
        partials = [build_partial(root, paths, 0, positions)]
    else:
        chunks = chunked(paths, workers * chunks_per_worker)
        pool = make_executor(executor, workers)
        assert pool is not None
        partials: list[PartialIndex] = []
        with pool:
            futures: list[Future[PartialIndex]] = [
                pool.submit(build_partial, root, chunk, start, positions) for start, chunk in chunks
            ]
            try:
                for fut in as_completed(futures):
                    partials.append(fut.result())  # re-raises the worker's exception (with its traceback)
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    t1 = time.perf_counter()
    index = merge(partials)
    t2 = time.perf_counter()
    if timings is not None:
        timings.map_s, timings.merge_s = t1 - t0, t2 - t1
    log.info("map %.2f s, merge %.2f s (%s x%d)", t1 - t0, t2 - t1, executor, workers)
    return index
