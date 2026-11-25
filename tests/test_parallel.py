from pathlib import Path

import pytest

from findex import parallel
from findex.corpus import iter_documents
from findex.index import build_index
from findex.parallel import ExecutorName, build_parallel, chunked


def test_chunked_covers_everything() -> None:
    items = list(range(10))
    chunks = chunked(items, 4)
    assert [s for s, _ in chunks] == [0, 3, 6, 8]
    assert [x for _, c in chunks for x in c] == items


@pytest.mark.parametrize("positions", [True, False])
def test_serial_equals_lab4_builder(corpus_dir: Path, positions: bool) -> None:
    expected = build_index(iter_documents(corpus_dir), positions=positions)
    assert build_parallel(corpus_dir, 1, "serial", positions) == expected


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_parallel_identical(corpus_dir: Path, executor: ExecutorName) -> None:
    expected = build_index(iter_documents(corpus_dir), positions=True)
    got = build_parallel(corpus_dir, 3, executor, True, chunks_per_worker=1)
    assert got == expected
    assert got.texts == expected.texts


def test_worker_exception_propagates(corpus_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(path: Path) -> str:
        raise RuntimeError(f"cannot read {path.name}")

    monkeypatch.setattr(parallel, "read_text", boom)
    with pytest.raises(RuntimeError, match="cannot read"):
        build_parallel(corpus_dir, 2, "threads")
