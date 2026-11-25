"""Record types of the index. Frozen (=> hashable) and slotted (=> no per-instance __dict__)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


def title_of(text: str, fallback: str) -> str:
    """First line that looks like prose (skips rst directives, labels and ==== underlines)."""
    for line in text.splitlines()[:40]:
        s = line.strip()
        if not s or s.startswith(("..", ":", "|")) or not any(c.isalpha() for c in s):
            continue
        return s.strip("*`#= ")[:120] or fallback
    return fallback
