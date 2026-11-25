from pathlib import Path

import numpy as np
import numpy.typing as npt

from findex import store
from findex.index import Index
from findex.ranking import keyword_query, search_page
from findex.semantic import SemanticIndex, build_embeddings, chunks, rrf


def test_chunks_overlap() -> None:
    words = [f"w{i}" for i in range(500)]
    cs = list(chunks(" ".join(words), size=200, overlap=40))
    assert cs[0].split()[-40:] == cs[1].split()[:40]
    assert cs[-1].split()[-1] == "w499"
    assert list(chunks("short text")) == ["short text"]


def test_rrf() -> None:
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert [d for d, _ in fused][:2] == [1, 3]  # in both lists near the top
    assert abs(fused[0][1] - (1 / 61 + 1 / 62)) < 1e-12


def test_keyword_query() -> None:
    assert keyword_query("save objects to file") == "save OR objects OR to OR file"
    assert keyword_query('a OR "b c"') == 'a OR "b c"'


def test_embed_and_search(tiny_index: Index, tmp_path: Path) -> None:
    p = tmp_path / "ix.bin"
    store.save(tiny_index, p)
    d = build_embeddings(tiny_index, p, dims=3)  # LSA needs dims < number of terms
    sem = SemanticIndex.open(p, tiny_index.num_docs)
    e: npt.NDArray[np.float32] = np.load(d / "E.npy")
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    top = sem.rank("event loop coroutines", 6)
    assert len(top) == 6 and top[0][1] >= top[-1][1]
    total, res = search_page(tiny_index, "event loop", k=3, mode="hybrid", sem=sem)
    assert total >= 2 and len(res) == 3
