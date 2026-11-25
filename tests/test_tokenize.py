import inspect

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(s: str) -> list[str]:
    return list(tokenize(s))


def test_is_generator() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)


def test_mixed_case_casefold() -> None:
    assert toks("Python PYTHON python") == ["python"] * 3
    assert toks("Straße") == toks("STRASSE") == ["strasse"]


def test_cyrillic() -> None:
    assert toks("Привіт, Світе! Пошукова система") == [
        "привіт",
        "світе",
        "пошукова",
        "система",
    ]


def test_combining_mark_nfc() -> None:
    assert toks("café") == toks("café") == ["café"]


def test_punctuation_and_hyphens() -> None:
    assert toks("built-in, (lazy); end.") == ["built", "in", "lazy", "end"]


def test_apostrophes() -> None:
    assert toks("don't 'quoted' п’ять") == ["don't", "quoted", "п'ять"]


def test_digits_and_underscores() -> None:
    assert toks("Python 3.13 __init__ utf8") == [
        "python",
        "3",
        "13",
        "__init__",
        "utf8",
    ]


def test_empty() -> None:
    assert toks("") == []
    assert toks("  ... --- !!! ") == []
