from pathlib import Path

import pytest

from findex import store
from findex.corpus import Document
from findex.index import build_index
from findex.models import Posting
from findex.ranking import SearchResult, search
from findex.scoring import BM25, Scorer, TfIdf

FILLER = " ".join(f"w{i}" for i in range(200))
DOCS = [
    Document(0, "rare", "zebra common"),
    Document(1, "common", "common common text"),
    Document(2, "short", "needle here"),
    Document(3, "long", "needle " + FILLER),
] + [Document(i, f"f{i}", "common filler") for i in range(4, 12)]


@pytest.fixture
def ix():  # type: ignore[no-untyped-def]
    return build_index(DOCS)


def test_dunders(ix) -> None:  # type: ignore[no-untyped-def]
    assert len(ix) > 0 and "zebra" in ix and "nope" not in ix
    assert ix["zebra"] == [Posting(0, 1)]
    with pytest.raises(KeyError):
        ix["nope"]
    assert "zebra" in list(ix)
    assert repr(ix).startswith("Index(terms=")
    assert ix.num_docs == 12 and ix.df("common") == 10


def test_rare_beats_common(ix) -> None:  # type: ignore[no-untyped-def]
    for scorer in (BM25(), TfIdf()):
        assert search(ix, "zebra OR common", scorer)[0].path == "rare"


def test_bm25_saturates(ix) -> None:  # type: ignore[no-untyped-def]
    s = BM25()
    gain_1_2 = s.score("needle", Posting(2, 2), ix) - s.score("needle", Posting(2, 1), ix)
    gain_19_20 = s.score("needle", Posting(2, 20), ix) - s.score("needle", Posting(2, 19), ix)
    assert gain_19_20 < gain_1_2 / 20


def test_short_beats_long(ix) -> None:  # type: ignore[no-untyped-def]
    res = search(ix, "needle", BM25())
    assert [r.path for r in res] == ["short", "long"]


def test_scorer_protocol() -> None:
    scorers: list[Scorer] = [BM25(), TfIdf()]
    assert {s.name for s in scorers} == {"bm25", "tfidf"}


def test_results_sort_by_score() -> None:
    a, b = SearchResult(1.0, 5, "a"), SearchResult(2.0, 1, "b")
    assert sorted([a, b], reverse=True) == [b, a]


def test_open_index_releases_on_error(tmp_path: Path, ix) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "ix.bin"
    store.save(ix, p)
    with pytest.raises(RuntimeError), store.open_index(p) as loaded:
        assert loaded == ix
        raise RuntimeError("boom")
    assert len(loaded) == 0  # close() ran


def test_snippet_highlights(ix) -> None:  # type: ignore[no-untyped-def]
    r = search(ix, "needle", BM25())[0]
    assert "**needle**" in r.snippet
