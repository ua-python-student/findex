"""Record types of the index. Frozen (=> hashable) and slotted (=> no per-instance __dict__)."""

from __future__ import annotations

import re
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


_ROLE = re.compile(r":[\w:.-]+:`!?~?([^`<]+?)(?:\s*<[^>]*>)?`")


def _clean(s: str) -> str:
    s = _ROLE.sub(r"\1", s)  # :mod:`!asyncio` -> asyncio
    return s.replace("``", "").replace("`", "").strip("*#= ")


def title_of(text: str, fallback: str) -> str:
    """The first reStructuredText heading (a line underlined with =, -, ~, * or #).

    Falls back to the first line that looks like prose, then to ``fallback``.
    """
    lines = text.splitlines()[:80]
    for line, under in zip(lines, lines[1:], strict=False):
        s, u = line.strip(), under.strip()
        if (
            s
            and any(c.isalpha() for c in s)
            and len(u) >= max(3, len(s) - 2)
            and len(set(u)) == 1
            and u[0] in "=-~*#^"
        ):
            return _clean(s)[:120] or fallback
    for line in lines:
        s = line.strip()
        if not s or s.startswith(("..", ":", "|")) or not any(c.isalpha() for c in s):
            continue
        return _clean(s)[:120] or fallback
    return fallback
