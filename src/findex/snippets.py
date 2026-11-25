"""Snippets: a window of text around the best match, with the query terms highlighted."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from itertools import islice


def snippet(text: str, terms: Iterable[str], width: int = 80, mark: tuple[str, str] = ("**", "**")) -> str:
    """±``width`` characters around the position where the most distinct query terms meet."""
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    text = unicodedata.normalize("NFC", text)
    if not terms:
        return " ".join(text[: 2 * width].split())
    pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, terms)) + r")(?!\w)", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return " ".join(text[: 2 * width].split())
    # best centre = the match whose ±width window holds the most *distinct* terms
    best, best_score = matches[0], -1
    j = 0
    for m in matches:
        while matches[j].start() < m.start() - width:
            j += 1
        window = set()
        for x in islice(matches, j, None):
            if x.start() > m.start() + width:
                break
            window.add(x.group().casefold())
        if len(window) > best_score:
            best, best_score = m, len(window)
    lo = max(0, best.start() - width)
    hi = min(len(text), best.end() + width)
    piece = pattern.sub(lambda m: f"{mark[0]}{m.group()}{mark[1]}", text[lo:hi])
    piece = " ".join(piece.split())
    return ("…" if lo > 0 else "") + piece + ("…" if hi < len(text) else "")
