"""Ranked search: parse -> match -> score -> top-k (heapq) -> results with snippets."""

from __future__ import annotations

import functools
import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from findex.index import Index
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.scoring import BM25, Scorer
from findex.snippets import snippet
from findex.timing import timed

if TYPE_CHECKING:
    from findex.fast import NumpyIndex
    from findex.semantic import SemanticIndex

log = logging.getLogger(__name__)
Mode = Literal["keyword", "semantic", "hybrid"]

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


class CachedRanker(Protocol):
    """What functools.lru_cache returns, as far as we use it."""

    def __call__(self, query: str, scorer: Scorer, k: int, /) -> Ranked: ...

    def cache_info(self) -> tuple[int, int, int | None, int]: ...  # (hits, misses, maxsize, currsize)


def _ranker(index: Index) -> CachedRanker:
    """A per-index lru_cache: the cache key is (query, scorer, k) — all hashable."""
    fn: CachedRanker | None = index.__dict__.get("_ranker")
    if fn is None:

        @functools.lru_cache(maxsize=256)
        def cached(query: str, scorer: Scorer, k: int) -> Ranked:
            return rank(index, query, scorer, k)

        index.__dict__["_ranker"] = fn = cached
    return fn


def _has_operators(node: Node) -> bool:
    if isinstance(node, Term):
        return False
    if isinstance(node, And):
        return _has_operators(node.left) or _has_operators(node.right)
    return isinstance(node, (Or, Not, Phrase))


def keyword_query(query: str) -> str:
    """For semantic/hybrid modes: a plain query ("save objects to a file") becomes an OR of its
    words, so BM25 ranks instead of AND-filtering; a query with operators is kept as typed."""
    node = parse(query)
    if _has_operators(node):
        return query
    return " OR ".join(dict.fromkeys(node.terms()))


def _keyword_top(
    index: Index, query: str, scorer: Scorer, k: int, nx: NumpyIndex | None
) -> tuple[int, tuple[tuple[int, float], ...]]:
    if nx is not None and scorer.name in ("bm25", "tfidf"):
        total, top = nx.rank(query, "bm25" if scorer.name == "bm25" else "tfidf", k)
        return total, tuple(top)
    ranker = _ranker(index)
    hits_before = ranker.cache_info()[0]
    res = ranker(query, scorer, k)
    if ranker.cache_info()[0] > hits_before:
        log.debug("cache hit for %r (%s, k=%d)", query, scorer.name, k)
    return res


@timed
def search_page(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    offset: int = 0,
    with_snippets: bool = True,
    mode: Mode = "keyword",
    nx: NumpyIndex | None = None,
    sem: SemanticIndex | None = None,
) -> tuple[int, list[SearchResult]]:
    scorer = scorer or BM25()
    if mode != "keyword" and sem is None:
        raise ValueError(f"mode={mode!r} needs embeddings: run `findex embed` first")
    if mode == "keyword":
        total, top = _keyword_top(index, query, scorer, k + offset, nx)
    elif mode == "semantic":
        assert sem is not None
        top = tuple(sem.rank(query, k + offset))
        total = index.num_docs
    else:  # hybrid: Reciprocal Rank Fusion of the two top-100 lists
        from findex.semantic import rrf

        assert sem is not None
        depth = max(100, k + offset)
        _, kw = _keyword_top(index, keyword_query(query), scorer, depth, nx)
        se = sem.rank(query, depth)
        top = tuple(rrf([[d for d, _ in kw], [d for d, _ in se]], limit=k + offset))
        total = len({d for d, _ in kw} | {d for d, _ in se})
    terms = list(parse(query).terms()) if with_snippets else []
    results: list[SearchResult] = []
    for doc_id, score in top[offset:]:
        meta = index.doc_meta[doc_id]
        snip = snippet(index.doc_text(doc_id), terms) if with_snippets else ""
        results.append(SearchResult(round(score, 6), doc_id, meta.title, meta.path, snip))
    return total, results


def search(index: Index, query: str, scorer: Scorer | None = None, k: int = 10) -> list[SearchResult]:
    return search_page(index, query, scorer, k)[1]
