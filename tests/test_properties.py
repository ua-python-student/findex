"""Hypothesis: properties, not examples."""

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from findex import store
from findex.corpus import Document
from findex.index import build_index
from findex.query import And, Node, Not, Or, Term, parse
from findex.search import merge_and, merge_not, merge_or
from findex.tokenize import tokenize

sorted_ids = st.lists(st.integers(0, 500), max_size=60, unique=True).map(sorted)


@given(sorted_ids, sorted_ids)
def test_merges_agree_with_sets(a: list[int], b: list[int]) -> None:
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


@given(st.text())
def test_tokenize_idempotent_and_nonempty(text: str) -> None:
    toks = list(tokenize(text))
    assert all(toks)
    assert list(tokenize(" ".join(toks))) == toks


words = st.text(alphabet="abcdefgh", min_size=1, max_size=3)
docs_strategy = st.lists(st.lists(words, max_size=12).map(" ".join), min_size=1, max_size=8)


@settings(max_examples=40, deadline=None)
@given(docs_strategy, st.booleans())
def test_store_round_trip(texts: list[str], positions: bool) -> None:
    ix = build_index([Document(i, f"d{i}", t) for i, t in enumerate(texts)], positions=positions)
    with TemporaryDirectory() as tmp:
        for fmt in ("pickle", "binary"):
            p = Path(tmp) / fmt
            store.save(ix, p, fmt="pickle" if fmt == "pickle" else "binary")
            assert store.load(p) == ix
    for plist in ix.values():  # invariant: postings sorted by doc_id
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)


trees: st.SearchStrategy[Node] = st.recursive(
    words.map(Term),
    lambda kids: st.one_of(st.builds(And, kids, kids), st.builds(Or, kids, kids), st.builds(Not, kids)),
    max_leaves=8,
)


@given(trees)
def test_parser_round_trip(tree: Node) -> None:
    assert parse(str(tree)) == tree
