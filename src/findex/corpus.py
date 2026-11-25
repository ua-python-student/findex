"""Lazy corpus reader: yields one document at a time, never the whole corpus."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: str
    text: str


def txt_paths(root: Path) -> Iterator[Path]:
    # sorted() over a lazy rglob materialises only the *paths* (a few KB),
    # never the texts; it makes doc_ids stable between runs.
    yield from sorted(root.rglob("*.txt"))


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        log.warning("undecodable bytes in %s, replacing", path)
        return raw.decode("utf-8", errors="replace")


def _iter_jsonl(path: Path, start_id: int) -> Iterator[Document]:
    with path.open(encoding="utf-8", errors="replace") as f:
        doc_id = start_id
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("bad JSON at %s:%d, skipped", path, lineno)
                continue
            text = rec.get("text", "")
            title = rec.get("title") or ""
            yield Document(doc_id, rec.get("url") or f"{path}:{lineno}", f"{title}\n{text}")
            doc_id += 1


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents one by one from a directory of .txt files or a .jsonl file."""
    root = Path(root)
    if root.is_file() and root.suffix == ".jsonl":
        yield from _iter_jsonl(root, 0)
        return
    for doc_id, path in enumerate(txt_paths(root)):
        yield Document(doc_id, str(path.relative_to(root)), read_text(path))
