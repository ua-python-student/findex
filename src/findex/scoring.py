"""Scorers: anything with ``score(term, posting, index) -> float`` is a Scorer (duck typing)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from findex.models import Posting

if TYPE_CHECKING:
    from findex.index import Index


class Scorer(Protocol):
    """Structural interface: no inheritance needed to satisfy it."""

    @property
    def name(self) -> str: ...  # read-only: frozen dataclasses satisfy it

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """(1 + log tf) * log(N / df) — sublinear tf, classic idf."""

    name: str = "tfidf"

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if not df or not posting.tf:
            return 0.0
        return (1.0 + math.log(posting.tf)) * math.log(index.num_docs / df)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25 with tf saturation (k1) and document-length normalisation (b)."""

    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"

    @staticmethod
    def idf(n_docs: int, df: int) -> float:
        # Lucene's variant: always positive, even for terms in > half the docs.
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if not df:
            return 0.0
        tf = posting.tf
        norm = self.k1 * (1.0 - self.b + self.b * index.doc_length(posting.doc_id) / index.avg_doc_length)
        return self.idf(index.num_docs, df) * tf * (self.k1 + 1.0) / (tf + norm)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


SCORERS: dict[str, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}
