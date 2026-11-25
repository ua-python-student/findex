"""Ranked search: parse -> match -> score -> top-k (heapq) -> results with snippets."""

from __future__ import annotations

import functools
import heapq
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from findex.index import Index
from findex.query import parse
from findex.scoring import BM25, Scorer
from findex.snippets import snippet
from findex.timing import timed

log = logging.getLogger(__name__)

type Ranked = tuple[int, tuple[tuple[int, float], ...]]  # (total matches, ((doc_id, score), ...))


@dataclass(frozen=True, order=True)
class SearchResult:
    """Ordered by score, then doc_id, so ``sorted(results, reverse=True)`` ranks them."""

    score: float
    doc_id: int
    title: str = field(compare=False)
    path: str = field(compare=False, default="")
    snippet: str = field(compare=False, default="")


@timed  # logged only on a cache miss: a hit never reaches this function
def rank(index: Index, query: str, scorer: Scorer, k: int) -> Ranked:
    node = parse(query)
    matched = node.evaluate(index)
    if not matched:
        return 0, ()
    allowed = set(matched)
    scores: defaultdict[int, float] = defaultdict(float)
    for d in matched:
        scores[d] = 0.0
    for term in set(node.terms()):
        for p in index.get(term, ()):
            if p.doc_id in allowed:
                scores[p.doc_id] += scorer.score(term, p, index)
    # O(n log k) instead of sorting all n matches; ties broken by doc_id for determinism
    top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
    return len(matched), tuple(top)


def _ranker(index: Index) -> Callable[[str, Scorer, int], Ranked]:
    """A per-index lru_cache: the cache key is (query, scorer, k) — all hashable."""
    fn = index.__dict__.get("_ranker")
    if fn is None:

        @functools.lru_cache(maxsize=256)
        def cached(query: str, scorer: Scorer, k: int) -> Ranked:
            return rank(index, query, scorer, k)

        index.__dict__["_ranker"] = fn = cached
    return fn


@timed
def search_page(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    offset: int = 0,
    with_snippets: bool = True,
) -> tuple[int, list[SearchResult]]:
    scorer = scorer or BM25()
    ranker = _ranker(index)
    hits_before = ranker.cache_info().hits  # type: ignore[attr-defined]
    total, top = ranker(query, scorer, k + offset)
    if ranker.cache_info().hits > hits_before:  # type: ignore[attr-defined]
        log.debug("cache hit for %r (%s, k=%d)", query, scorer.name, k + offset)
    terms = list(parse(query).terms()) if with_snippets else []
    results = []
    for doc_id, score in top[offset:]:
        meta = index.doc_meta[doc_id]
        snip = snippet(index.doc_text(doc_id), terms) if with_snippets else ""
        results.append(SearchResult(round(score, 6), doc_id, meta.title, meta.path, snip))
    return total, results


def search(index: Index, query: str, scorer: Scorer | None = None, k: int = 10) -> list[SearchResult]:
    return search_page(index, query, scorer, k)[1]
