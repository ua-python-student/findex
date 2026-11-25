import inspect
from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python PYTHON python", ["python", "python", "python"]),
        ("Straße", ["strasse"]),  # casefold, not lower
        ("Привіт, Світе! Пошукова система", ["привіт", "світе", "пошукова", "система"]),
        ("café", ["café"]),
        ("café", ["café"]),  # combining acute accent -> NFC
        ("built-in, (lazy); end.", ["built", "in", "lazy", "end"]),
        ("don't 'quoted' п’ять", ["don't", "quoted", "п'ять"]),
        ("Python 3.13 __init__ utf8", ["python", "3", "13", "__init__", "utf8"]),
        ("", []),
        ("  ... --- !!! ", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_generators() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


def test_iter_documents(corpus_dir: Path) -> None:
    docs = list(iter_documents(corpus_dir))
    assert [d.doc_id for d in docs] == list(range(6))
    assert docs[0].path == "django/cache.txt"  # sorted paths => stable ids


def test_iter_documents_jsonl_and_bad_bytes(tmp_path: Path) -> None:
    (tmp_path / "c.jsonl").write_text('{"title": "T", "text": "hello"}\nnot json\n{"text": "x"}\n')
    docs = list(iter_documents(tmp_path / "c.jsonl"))
    assert [d.text for d in docs] == ["T\nhello", "\nx"]
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "bad.txt").write_bytes(b"ok \xff\xfe bytes")
    assert "ok" in next(iter_documents(tmp_path / "d")).text
