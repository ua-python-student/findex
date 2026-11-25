"""Streaming tokenizer.

Policy (documented and tested in tests/test_tokenize.py):

* Text is NFC-normalised, then ``casefold()``-ed (so ``Straße`` == ``STRASSE``).
* A token is a run of Unicode word characters (``\\w``: letters of any script,
  digits, underscore). Underscore is kept on purpose: the corpus is code
  documentation, and ``__init__`` or ``read_csv`` must stay one token.
* Apostrophes *inside* a word are kept: ``don't``, ``п'ять``. Typographic
  apostrophes (’ ʼ) are mapped to ASCII ``'`` first. Leading/trailing
  apostrophes are dropped.
* Hyphens split words: ``built-in`` -> ``built``, ``in``.
* Digits are tokens: ``3.13`` -> ``3``, ``13``; ``utf8`` stays ``utf8``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

WORD_RE = re.compile(r"\w+(?:'\w+)*")
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'"})


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_APOSTROPHES).casefold()


def tokenize(text: str) -> Iterator[str]:
    """Yield tokens lazily (re.finditer, never re.findall)."""
    for m in WORD_RE.finditer(normalize(text)):
        yield m.group()


def tokenize_with_positions(text: str) -> Iterator[tuple[int, str]]:
    """Yield (token_index, token) pairs; the index is the token's ordinal position."""
    for i, m in enumerate(WORD_RE.finditer(normalize(text))):
        yield i, m.group()
