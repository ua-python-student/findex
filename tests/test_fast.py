"""The NumPy scorer must rank exactly like the Lab 3 scorer."""

import math
from pathlib import Path

import pytest

from findex import store
from findex.fast import NumpyIndex
from findex.index import Index
from findex.ranking import rank
from findex.scoring import SCORERS

QUERIES = [
    "event",
    "event loop",
    '"event loop"',
    "django OR python",
    "python NOT asyncio",
    "(django OR numpy) NOT cache",
    "NOT django",
    "the OR a OR is",
    "nothing",
    'lazy "hash table"',
]


@pytest.fixture(scope="module", params=["from_index", "memmap"])
def nx(
    request: pytest.FixtureRequest, tiny_index: Index, tmp_path_factory: pytest.TempPathFactory
) -> NumpyIndex:
    if request.param == "from_index":
        return NumpyIndex.from_index(tiny_index)
    p = tmp_path_factory.mktemp("np") / "ix.bin"
    store.save(tiny_index, p)
    return NumpyIndex.open(p)


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
def test_identical_ranking(nx: NumpyIndex, tiny_index: Index, query: str, scorer: str) -> None:
    total_py, top_py = rank(tiny_index, query, SCORERS[scorer], 5)
    total_np, top_np = nx.rank(query, "bm25" if scorer == "bm25" else "tfidf", 5)
    assert total_np == total_py
    assert [d for d, _ in top_np] == [d for d, _ in top_py]
    for (_, a), (_, b) in zip(top_np, top_py, strict=True):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def test_real_corpus_identical() -> None:
    path = Path(__file__).parents[1] / "index.bin"
    if not path.exists():
        pytest.skip("no full index")
    ix = store.load(path)
    nx = NumpyIndex.open(path)
    for q in ("python OR thread OR process", "asyncio TaskGroup", '"event loop" NOT trio', "the"):
        _, a = rank(ix, q, SCORERS["bm25"], 20)
        _, b = nx.rank(q, "bm25", 20)
        assert [d for d, _ in a] == [d for d, _ in b]


test_real_corpus_identical = pytest.mark.slow(test_real_corpus_identical)
