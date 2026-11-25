import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse

DOCS = [
    Document(0, "a", "python async await event loop"),
    Document(1, "b", "java event loop and python"),
    Document(2, "c", "loop event python async"),
]


def test_precedence_tree() -> None:
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse("(a OR b) c") == And(Or(Term("a"), Term("b")), Term("c"))
    assert parse("a AND NOT b") == And(Term("a"), Not(Term("b")))
    assert parse('x "Event Loop"') == And(Term("x"), Phrase(("event", "loop")))


def test_operators_build_same_tree() -> None:
    assert Term("a") & (Term("b") | Term("c")) & ~Term("d") == parse("a (b OR c) NOT d")


def test_evaluate_and_phrase() -> None:
    ix = build_index(DOCS, positions=True)
    assert parse("python AND (async OR await) NOT java").evaluate(ix) == [0, 2]
    assert parse('"event loop"').evaluate(ix) == [0, 1]
    assert parse('"loop event"').evaluate(ix) == [2]
    assert parse("NOT java").evaluate(ix) == [0, 2]


def test_phrase_needs_positions() -> None:
    with pytest.raises(QuerySyntaxError):
        parse('"event loop"').evaluate(build_index(DOCS))


@pytest.mark.parametrize("bad", ["", "(a", "a)", "a OR", "NOT"])
def test_syntax_errors(bad: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse(bad)
