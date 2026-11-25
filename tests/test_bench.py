"""pytest-benchmark: run with `pytest --benchmark-only`; CI compares against the committed baseline."""

from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import iter_documents
from findex.fast import NumpyIndex
from findex.index import Index, build_index
from findex.ranking import rank
from findex.scoring import BM25

Q = "event OR loop OR django OR python"


@pytest.mark.benchmark(group="search")
def test_bench_rank_python(benchmark: BenchmarkFixture, tiny_index: Index) -> None:
    benchmark(rank, tiny_index, Q, BM25(), 10)


@pytest.mark.benchmark(group="search")
def test_bench_rank_numpy(benchmark: BenchmarkFixture, tiny_index: Index) -> None:
    nx = NumpyIndex.from_index(tiny_index)
    benchmark(nx.rank, Q, "bm25", 10)


@pytest.mark.benchmark(group="build")
def test_bench_build_index(benchmark: BenchmarkFixture, corpus_dir: Path) -> None:
    benchmark(lambda: build_index(iter_documents(corpus_dir), positions=True))
