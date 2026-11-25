"""NumPy postings and vectorised scoring.

Postings live in two flat ``int32`` arrays (CSR layout): ``doc_ids`` and ``tfs``,
with ``offsets[t] : offsets[t+1]`` the slice of term ``t``. ``NumpyIndex.open``
maps a Lab 2 binary index file with ``np.memmap`` and only builds views — no
``Posting`` objects, and load time does not grow with the number of postings.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.scoring import BM25

if TYPE_CHECKING:
    from findex.index import Index

IntArray = npt.NDArray[np.int64]
I32 = npt.NDArray[np.int32]
U32 = npt.NDArray[np.uint32]
F64 = npt.NDArray[np.float64]
ScorerName = Literal["bm25", "tfidf"]
EMPTY = np.empty(0, dtype=np.int64)


class NumpyIndex:
    def __init__(
        self,
        postings: dict[str, tuple[U32, U32, U32 | None]],
        doc_len: I32,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._post = postings  # term -> (doc_ids, tfs, positions or None) as array views
        self.doc_len = doc_len
        self.num_docs = int(doc_len.shape[0])
        self.avg_doc_length = float(sum(int(x) for x in doc_len)) / max(self.num_docs, 1)
        self.k1, self.b = k1, b

    # --- construction ---------------------------------------------------------------
    @classmethod
    def from_index(cls, ix: Index) -> NumpyIndex:
        post: dict[str, tuple[U32, U32, U32 | None]] = {}
        for term, plist in ix.items():
            ids = np.fromiter((p.doc_id for p in plist), dtype=np.uint32, count=len(plist))
            tfs = np.fromiter((p.tf for p in plist), dtype=np.uint32, count=len(plist))
            pos = (
                np.fromiter((x for p in plist for x in p.positions), dtype=np.uint32)
                if plist[0].positions
                else None
            )
            post[term] = (ids, tfs, pos)
        doc_len = np.array([ix.doc_length(d) for d in range(ix.num_docs)], dtype=np.int32)
        return cls(post, doc_len)

    @classmethod
    def open(cls, path: Path) -> NumpyIndex:
        """Memory-map a binary index: views into the file, paged in on demand."""
        from findex.store import read_header

        header, start = read_header(path)
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        post: dict[str, tuple[U32, U32, U32 | None]] = {}
        for term, (off, n, has_pos) in header["terms"].items():
            base = start + off
            ids = np.frombuffer(mm, dtype=np.uint32, count=n, offset=base)
            tfs = np.frombuffer(mm, dtype=np.uint32, count=n, offset=base + 4 * n)
            pos = None
            if has_pos:  # positions length = sum(tfs): computed lazily in _positions()
                pos = np.frombuffer(mm, dtype=np.uint32, offset=base + 8 * n, count=0)
            post[term] = (ids, tfs, pos)
        lengths = header["doc_lengths"]
        doc_len = np.array([lengths[str(d)] for d in range(len(lengths))], dtype=np.int32)
        inst = cls(post, doc_len)
        inst._mm = mm  # keep the mapping alive  # pyright: ignore[reportAttributeAccessIssue]
        inst._starts = {t: start + o for t, (o, _n, _h) in header["terms"].items()}  # pyright: ignore[reportAttributeAccessIssue]
        return inst

    # --- postings ---------------------------------------------------------------------
    def __contains__(self, term: str) -> bool:
        return term in self._post

    def df(self, term: str) -> int:
        p = self._post.get(term)
        return 0 if p is None else int(p[0].shape[0])

    def doc_ids(self, term: str) -> IntArray:
        p = self._post.get(term)
        return EMPTY if p is None else p[0].astype(np.int64)

    def _positions(self, term: str) -> tuple[IntArray, U32]:
        """(offsets into the flat positions array per posting, flat positions)."""
        ids, tfs, pos = self._post[term]
        if pos is None:
            raise ValueError("phrase queries need an index built with --positions")
        offsets = np.zeros(ids.shape[0] + 1, dtype=np.int64)
        np.cumsum(tfs, out=offsets[1:])
        if pos.shape[0] != offsets[-1]:  # memmapped: materialise the view now that we know its length
            mm = getattr(self, "_mm")  # noqa: B009
            start = getattr(self, "_starts")[term] + 8 * ids.shape[0]  # noqa: B009
            pos = np.frombuffer(mm, dtype=np.uint32, count=int(offsets[-1]), offset=start)
        return offsets, pos

    # --- Boolean evaluation with NumPy set operations ------------------------------------
    def evaluate(self, node: Node) -> IntArray:
        if isinstance(node, Term):
            return self.doc_ids(node.term)
        if isinstance(node, And):
            left = self.evaluate(node.left)
            if isinstance(node.right, Not):
                return np.setdiff1d(left, self.evaluate(node.right.child), assume_unique=True)
            return np.intersect1d(left, self.evaluate(node.right), assume_unique=True)
        if isinstance(node, Or):
            return np.union1d(self.evaluate(node.left), self.evaluate(node.right))
        if isinstance(node, Not):
            return np.setdiff1d(
                np.arange(self.num_docs, dtype=np.int64), self.evaluate(node.child), assume_unique=True
            )
        if isinstance(node, Phrase):
            return self._phrase(node.words)
        raise TypeError(f"unknown node {node!r}")

    def _phrase(self, words: tuple[str, ...]) -> IntArray:
        if not all(w in self for w in words):
            return EMPTY
        docs = self.doc_ids(words[0])
        for w in words[1:]:
            docs = np.intersect1d(docs, self.doc_ids(w), assume_unique=True)
        if docs.shape[0] == 0:
            return EMPTY
        per_word: list[tuple[IntArray, IntArray, U32]] = []
        for w in words:
            ids = self.doc_ids(w)
            offsets, pos = self._positions(w)
            rows = np.searchsorted(ids, docs).astype(np.int64)
            per_word.append((rows, offsets, pos))
        keep: list[int] = []
        for j, d in enumerate(docs.tolist()):
            rows0, off0, pos0 = per_word[0]
            r = int(rows0[j])
            cand = pos0[off0[r] : off0[r + 1]].astype(np.int64)
            for i, (rows, offs, pos) in enumerate(per_word[1:], 1):
                r = int(rows[j])
                cand = np.intersect1d(
                    cand, pos[offs[r] : offs[r + 1]].astype(np.int64) - i, assume_unique=True
                )
                if cand.shape[0] == 0:
                    break
            if cand.shape[0]:
                keep.append(d)
        return np.array(keep, dtype=np.int64)

    # --- vectorised scoring -------------------------------------------------------------
    def term_scores(self, term: str, scorer: ScorerName) -> tuple[IntArray, F64]:
        """Scores of one term for all its postings at once: no Python loop over documents."""
        ids, tfs_u, _ = self._post[term]
        ids64 = ids.astype(np.int64)
        tf = tfs_u.astype(np.float64)
        df = ids.shape[0]
        if scorer == "tfidf":
            return ids64, (1.0 + np.log(tf)) * math.log(self.num_docs / df)
        idf = BM25.idf(self.num_docs, df)  # one scalar per term: same math.log as Lab 3
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[ids64] / self.avg_doc_length)  # fancy indexing
        return ids64, idf * tf * (self.k1 + 1.0) / (tf + norm)

    def rank(
        self, query: str | Node, scorer: ScorerName = "bm25", k: int = 10
    ) -> tuple[int, list[tuple[int, float]]]:
        node = parse(query) if isinstance(query, str) else query
        matched = self.evaluate(node)
        if matched.shape[0] == 0:
            return 0, []
        scores = np.zeros(self.num_docs, dtype=np.float64)
        for term in set(node.terms()):  # same summation order as findex.ranking.rank
            if term in self:
                ids, s = self.term_scores(term, scorer)
                scores[ids] += s  # ids are unique within a term: plain fancy-index add is safe
        total = int(matched.shape[0])
        cand = scores[matched]
        if total > k:  # O(n) partition, then sort only the survivors (+ ties at the cut)
            kth = np.partition(cand, total - k)[total - k]
            keep = cand >= kth
            matched, cand = matched[keep], cand[keep]
        order = np.lexsort((matched, -cand))[:k]  # score desc, then doc_id asc: Lab 3's tie-break
        return total, [(int(matched[i]), float(cand[i])) for i in order]
