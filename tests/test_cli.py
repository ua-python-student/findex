import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex import __version__
from findex.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def index_file(corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("ix") / "index.bin"
    res = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out)])
    assert res.exit_code == 0, res.output
    return out


def test_version() -> None:
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0 and __version__ in res.stdout


def test_search_table(index_file: Path) -> None:
    res = runner.invoke(app, ["search", str(index_file), '"event loop"'])
    assert res.exit_code == 0
    assert "asyncio.txt" in res.stdout and "broadcast.txt" in res.stdout


def test_search_json(index_file: Path) -> None:
    res = runner.invoke(app, ["search", str(index_file), "django NOT cache", "--json"])
    lines = [json.loads(x) for x in res.stdout.splitlines()]
    assert [r["path"] for r in lines] == ["django/i18n.txt"]
    assert lines[0]["rank"] == 1


def test_stats(index_file: Path) -> None:
    res = runner.invoke(app, ["stats", str(index_file), "--top", "3"])
    assert res.exit_code == 0 and "vocabulary" in res.stdout


def test_missing_index_is_one_line_error(tmp_path: Path) -> None:
    res = runner.invoke(app, ["search", str(tmp_path / "nope.bin"), "x"])
    assert res.exit_code == 1
    assert "index file not found" in res.stderr and "Traceback" not in res.stderr


def test_bad_query_exit_code(index_file: Path) -> None:
    res = runner.invoke(app, ["search", str(index_file), "(unclosed"])
    assert res.exit_code == 2 and "bad query" in res.stderr


def test_bad_option_value(index_file: Path) -> None:
    assert runner.invoke(app, ["search", str(index_file), "x", "--k", "0"]).exit_code == 2


@pytest.mark.slow
def test_real_corpus_slice(tmp_path: Path) -> None:
    data = Path(__file__).parents[1] / "data"
    if not data.exists():
        pytest.skip("corpus not downloaded")
    out = tmp_path / "ix.bin"
    assert runner.invoke(app, ["index", str(data), "--out", str(out), "--limit", "200"]).exit_code == 0
    res = runner.invoke(app, ["search", str(out), "python", "--json", "-k", "3"])
    assert len(res.stdout.splitlines()) == 3
