"""Query language: a recursive-descent parser producing a tree of operator-overloading nodes.

Grammar::

    or_expr  := and_expr ('OR' and_expr)*
    and_expr := not_expr (['AND'] not_expr)*        # implicit AND
    not_expr := 'NOT' not_expr | atom
    atom     := WORD | "quoted phrase" | '(' or_expr ')'

``parse('a OR b c') == Or(Term('a'), And(Term('b'), Term('c')))``: AND binds
tighter than OR, exactly like ``*`` and ``+``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from findex.search import merge_and, merge_not, merge_or
from findex.tokenize import tokenize

if TYPE_CHECKING:
    from findex.index import Index


class QuerySyntaxError(ValueError):
    pass


class Node:
    """Base of the query tree. Nodes compose with &, | and ~."""

    def evaluate(self, index: Index) -> list[int]:
        """Sorted doc_ids that match."""
        raise NotImplementedError

    def terms(self) -> Iterator[str]:
        """Positive terms (not under NOT) — these are the ones that get scored."""
        return iter(())

    def __and__(self, other: Node) -> Node:
        return And(self, other)

    def __or__(self, other: Node) -> Node:
        return Or(self, other)

    def __invert__(self) -> Node:
        return Not(self)


@dataclass(frozen=True)
class Term(Node):
    term: str

    def evaluate(self, index: Index) -> list[int]:
        return [p.doc_id for p in index.get(self.term, ())]

    def terms(self) -> Iterator[str]:
        yield self.term

    def __str__(self) -> str:
        return self.term


@dataclass(frozen=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> list[int]:
        if not all(w in index for w in self.words):
            return []
        if not index.has_positions:
            raise QuerySyntaxError("phrase queries need an index built with --positions")
        # 1) docs that contain every word: two-pointer merges over doc_ids
        lists = [index[w] for w in self.words]
        docs = [p.doc_id for p in lists[0]]
        for plist in lists[1:]:
            docs = merge_and(docs, [p.doc_id for p in plist])
        # 2) positions: word i must appear at p + i
        by_doc = [{p.doc_id: p.positions for p in plist} for plist in lists]
        out = []
        for d in docs:
            first, *rest = (by_doc[i][d] for i in range(len(self.words)))
            rest_sets = [set(r) for r in rest]
            if any(all(p + i + 1 in s for i, s in enumerate(rest_sets)) for p in first):
                out.append(d)
        return out

    def terms(self) -> Iterator[str]:
        yield from self.words

    def __str__(self) -> str:
        return '"' + " ".join(self.words) + '"'


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> list[int]:
        if isinstance(self.right, Not):  # a AND NOT b: one merge, no complement
            return merge_not(self.left.evaluate(index), self.right.child.evaluate(index))
        return merge_and(self.left.evaluate(index), self.right.evaluate(index))

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> list[int]:
        return merge_or(self.left.evaluate(index), self.right.evaluate(index))

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index) -> list[int]:  # bare NOT: complement over all docs
        return merge_not(list(range(index.num_docs)), self.child.evaluate(index))

    def __str__(self) -> str:
        return f"NOT {self.child}"


# --- lexer + parser ------------------------------------------------------------

_LEX = re.compile(r'\s*(?:(?P<phrase>"[^"]*")|(?P<paren>[()])|(?P<word>[^\s()"]+))')


def _lex(text: str) -> list[str]:
    pos, out = 0, []
    text = text.strip()
    while pos < len(text):
        m = _LEX.match(text, pos)
        if not m or m.end() == pos:
            raise QuerySyntaxError(f"unexpected character at {pos}: {text[pos:]!r}")
        out.append(m.group(m.lastgroup or "word"))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def or_expr(self) -> Node | None:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            right = self.and_expr()
            if node is None or right is None:
                raise QuerySyntaxError("OR needs an operand on both sides")
            node = Or(node, right)
        return node

    def and_expr(self) -> Node | None:
        node = self.not_expr()
        while (tok := self.peek()) is not None and tok not in ("OR", ")"):
            if tok == "AND":
                self.take()
            right = self.not_expr()
            if right is None:
                continue
            node = right if node is None else And(node, right)
        return node

    def not_expr(self) -> Node | None:
        if self.peek() == "NOT":
            self.take()
            child = self.not_expr()
            if child is None:
                raise QuerySyntaxError("NOT needs an operand")
            return Not(child)
        return self.atom()

    def atom(self) -> Node | None:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("unexpected end of query")
        self.take()
        if tok == "(":
            node = self.or_expr()
            if self.peek() != ")":
                raise QuerySyntaxError("missing closing parenthesis")
            self.take()
            return node
        if tok == ")":
            raise QuerySyntaxError("unbalanced ')'")
        words = tuple(tokenize(tok.strip('"')))
        if not words:
            return None  # pure punctuation: ignore
        if tok.startswith('"') or len(words) > 1:
            return Phrase(words) if len(words) > 1 else Term(words[0])
        return Term(words[0])


def parse(text: str) -> Node:
    p = _Parser(_lex(text))
    node = p.or_expr()
    if p.peek() is not None:
        raise QuerySyntaxError(f"unexpected {p.peek()!r}")
    if node is None:
        raise QuerySyntaxError("empty query")
    return node
