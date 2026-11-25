"""Shared fixtures: a tiny on-disk corpus and its index, built once per session."""

from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import Index, build_index

TINY = {
    "python/generators.txt": "Generators\n==========\n\nA generator function uses yield. Generators are lazy.",
    "python/asyncio.txt": "Asyncio\n=======\n\nThe event loop runs coroutines. An async def function returns a coroutine.",
    "python/dict.txt": "Dictionaries\n============\n\nA dict is a hash table. Keys must be hashable; lists are not.",
    "django/cache.txt": "Cache\n=====\n\nDjango cache backends: memcached, redis, database, local memory.",
    "django/i18n.txt": "Translation\n===========\n\nDjango translates strings with gettext. Привіт, світе!",
    "numpy/broadcast.txt": "Broadcasting\n============\n\nArrays of different shapes are broadcast. The event loop is unrelated.",
}


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for rel, text in TINY.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def tiny_index(corpus_dir: Path) -> Index:
    """Session-scoped: shared by many tests, so no test may mutate it."""
    return build_index(iter_documents(corpus_dir), positions=True)
