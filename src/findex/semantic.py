# scikit-learn and sentence-transformers ship no type information: everything that crosses
# into them is Any. These four checks are relaxed for THIS module only (the typed boundary is
# the Embedder protocol, which returns float32 ndarrays).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Semantic search: chunk embeddings, brute-force cosine (E @ q), and Reciprocal Rank Fusion.

Two embedding back-ends behind one interface:

* ``sentence-transformers`` (e.g. ``all-MiniLM-L6-v2``) — a neural model, used when the
  package is installed and the model can be downloaded from Hugging Face;
* ``lsa`` — Latent Semantic Analysis: TF-IDF over chunks, then a truncated SVD to 256
  dimensions (scikit-learn). Words that occur in similar contexts end up in similar
  directions, so a query can match a chunk with *none* of its words. It needs no
  download and is trained on our own corpus. (Used for the reported numbers: model
  downloads were not possible in the environment where this lab was run.)

Files written next to the index:  <index>.emb/{E.npy, chunk_doc.npy, meta.json[, lsa.pkl]}
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from findex.index import Index

log = logging.getLogger(__name__)
F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]
CHUNK_WORDS = 200
RRF_K = 60


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> F32: ...


def chunks(text: str, size: int = CHUNK_WORDS, overlap: int = 40) -> Iterator[str]:
    """~200-word passages with a 40-word overlap (a 40-page document as one vector is mush)."""
    words = re.sub(r"\s+", " ", text).split(" ")
    if len(words) <= size:
        yield " ".join(words)
        return
    for start in range(0, len(words) - overlap, size - overlap):
        yield " ".join(words[start : start + size])


def l2_normalize(m: F32) -> F32:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.maximum(norms, 1e-12)).astype(np.float32)


class LsaEmbedder:
    def __init__(self, vectorizer: Any = None, svd: Any = None, dims: int = 256) -> None:
        self.name = f"lsa-{dims}"
        self.vectorizer, self.svd, self.dims = vectorizer, svd, dims

    def fit(self, texts: Sequence[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        small = len(texts) < 100  # tiny corpora (tests): keep every term
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=1 if small else 2,
            max_df=1.0 if small else 0.5,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z_]\w+\b",
        )
        x = self.vectorizer.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=self.dims, random_state=0, algorithm="randomized", n_iter=7)
        self.svd.fit(x)
        log.info(
            "LSA: %d chunks, %d terms -> %d dims, explained variance %.1f%%",
            x.shape[0],
            x.shape[1],
            self.dims,
            100 * self.svd.explained_variance_ratio_.sum(),
        )

    def embed(self, texts: Sequence[str]) -> F32:
        return l2_normalize(
            np.asarray(self.svd.transform(self.vectorizer.transform(texts)), dtype=np.float32)
        )


class SentenceTransformerEmbedder:
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer  # pyright: ignore  # optional dependency

        self.name = model
        self._m = SentenceTransformer(model)

    def embed(self, texts: Sequence[str]) -> F32:
        v = self._m.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


def emb_dir(index_path: Path) -> Path:
    return index_path.with_name(index_path.name + ".emb")


def build_embeddings(
    index: Index,
    index_path: Path,
    model: str = "lsa",
    progress: Callable[[int, int], None] | None = None,
    dims: int = 256,
) -> Path:
    """Chunk every document, embed, L2-normalise, save E (float32[N_chunks, d]) + chunk -> doc map."""
    texts: list[str] = []
    owner: list[int] = []
    for doc_id in range(index.num_docs):
        title = index.title(doc_id)
        for c in chunks(index.doc_text(doc_id)):
            texts.append(f"{title}. {c}")
            owner.append(doc_id)
    out = emb_dir(index_path)
    out.mkdir(exist_ok=True)
    embedder: Embedder
    if model == "lsa":
        lsa = LsaEmbedder(dims=dims)
        lsa.fit(texts)
        with open(out / "lsa.pkl", "wb") as f:
            pickle.dump((lsa.vectorizer, lsa.svd), f)  # our own file, loaded only by us (see store.py)
        embedder = lsa
    else:
        embedder = SentenceTransformerEmbedder(model)
    parts: list[F32] = []
    step = 512
    for i in range(0, len(texts), step):
        parts.append(embedder.embed(texts[i : i + step]))
        if progress:
            progress(min(i + step, len(texts)), len(texts))
    e = np.vstack(parts).astype(np.float32)
    np.save(out / "E.npy", e)
    np.save(out / "chunk_doc.npy", np.array(owner, dtype=np.int32))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "model": embedder.name,
                "backend": model,
                "chunks": len(texts),
                "dims": int(e.shape[1]),
                "chunk_words": CHUNK_WORDS,
            }
        )
    )
    return out


@dataclass
class SemanticIndex:
    embedder: Embedder
    e: F32  # (N_chunks, d), unit rows
    chunk_doc: I32  # (N_chunks,)
    num_docs: int

    @classmethod
    def open(cls, index_path: Path, num_docs: int) -> SemanticIndex:
        d = emb_dir(index_path)
        meta = json.loads((d / "meta.json").read_text())
        embedder: Embedder
        if meta["backend"] == "lsa":
            with open(d / "lsa.pkl", "rb") as f:
                vec, svd = pickle.load(f)  # written by build_embeddings, never downloaded
            embedder = LsaEmbedder(vec, svd, meta["dims"])
        else:
            embedder = SentenceTransformerEmbedder(meta["model"])
        e = np.load(d / "E.npy", mmap_mode="r")
        return cls(embedder, e, np.load(d / "chunk_doc.npy"), num_docs)

    def doc_scores(self, query: str) -> npt.NDArray[np.float64]:
        """Cosine similarity of the query with every chunk (one mat-vec), best chunk per document."""
        q = self.embedder.embed([query])[0]
        sims = self.e @ q  # (N_chunks,) — brute-force vector search
        best = np.full(self.num_docs, -np.inf)
        np.maximum.at(best, self.chunk_doc, sims.astype(np.float64))  # a document = its best chunk
        return best

    def rank(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        best = self.doc_scores(query)
        k = min(k, self.num_docs)
        top = np.argpartition(-best, k - 1)[:k]
        top = top[np.lexsort((top, -best[top]))]
        return [(int(d), float(best[d])) for d in top if np.isfinite(best[d])]


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K, limit: int = 10) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d)), ranks from 1."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + r)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
