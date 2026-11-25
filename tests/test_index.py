from pathlib import Path

from findex import store
from findex.corpus import Document
from findex.index import build_index
from findex.models import DocMeta, Posting
from findex.search import OPS, boolean_search

DOCS = [
    Document(0, "a.txt", "Python generators are lazy. Python!"),
    Document(1, "b.txt", "Dictionaries in Python are hash tables"),
    Document(2, "c.txt", "Hash tables and sets"),
]


def test_postings_sorted_and_tf() -> None:
    ix = build_index(DOCS)
    assert [p.doc_id for p in ix["python"]] == [0, 1]
    assert ix["python"][0].tf == 2
    for plist in ix.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)
    assert ix.doc_lengths[2] == 4


def test_records_are_hashable() -> None:
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2
    assert len({DocMeta("a", "t"), DocMeta("a", "t")}) == 1


def test_merge_equals_set() -> None:
    a, b = [1, 3, 5, 7, 9], [2, 3, 4, 9, 10]
    for op in ("AND", "OR", "NOT"):
        assert OPS["merge"][op](a, b) == OPS["set"][op](a, b)


def test_boolean_query() -> None:
    ix = build_index(DOCS)
    assert boolean_search(ix, "hash tables") == [1, 2]
    assert boolean_search(ix, "hash NOT python") == [2]
    assert boolean_search(ix, "generators OR sets") == [0, 2]
    assert boolean_search(ix, "nothing") == []


def test_store_round_trip(tmp_path: Path) -> None:
    for positions in (False, True):
        ix = build_index(DOCS, positions=positions)
        formats: tuple[store.Format, ...] = ("pickle", "binary")
        for fmt in formats:
            p = tmp_path / f"{fmt}{positions}"
            store.save(ix, p, fmt=fmt)
            assert store.load(p) == ix
