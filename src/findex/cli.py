"""The ``findex`` command: index, search, stats.

stdout carries data (tables, ``--json`` lines); every diagnostic goes to stderr via logging.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Iterator
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex import __version__, store
from findex.corpus import Document, iter_documents
from findex.index import build_index
from findex.query import QuerySyntaxError
from findex.ranking import search_page
from findex.scoring import SCORERS

app = typer.Typer(add_completion=False, no_args_is_help=True, help="findex — a small search engine.")
out = Console()  # stdout: data
err = Console(stderr=True)  # stderr: progress, diagnostics
log = logging.getLogger("findex")


class ScorerName(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


def _fail(msg: str, code: int = 1) -> typer.Exit:
    err.print(f"[red]error:[/red] {msg}")
    return typer.Exit(code)


def _version(value: bool) -> None:
    if value:
        out.print(f"findex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="-v INFO, -vv DEBUG")] = 0,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Configure logging once, here, for the whole program (library code never does)."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s",
        handlers=[RichHandler(console=err, show_time=False, show_path=False)],
        force=True,
    )


def _with_progress(docs: Iterator[Document], total: int | None) -> Iterator[Document]:
    columns = (TextColumn("indexing"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn())
    with Progress(*columns, console=err, transient=True) as progress:
        task = progress.add_task("index", total=total)
        for doc in docs:
            yield doc
            progress.advance(task)


@app.command()
def index(
    corpus: Annotated[Path, typer.Argument(help="Directory of .txt files or a .jsonl file.")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Index file to write.")] = Path("index.bin"),
    positions: Annotated[bool, typer.Option(help="Store token positions (phrase queries).")] = True,
    limit: Annotated[int | None, typer.Option(help="Index only the first N documents.")] = None,
    fmt: Annotated[store.Format, typer.Option("--format")] = "binary",
) -> None:
    """Build an index from a corpus and save it."""
    if not corpus.exists():
        raise _fail(f"corpus not found: {corpus}")
    total = sum(1 for _ in corpus.rglob("*.txt")) if corpus.is_dir() else None
    if limit is not None and total is not None:
        total = min(total, limit)
    docs: Iterator[Document] = iter_documents(corpus)
    if limit is not None:
        docs = islice(docs, limit)
    t0 = time.perf_counter()
    ix = build_index(_with_progress(docs, total), positions=positions)
    store.save(ix, out_path, fmt=fmt)
    size = out_path.stat().st_size / 2**20
    err.print(f"{ix!r} -> {out_path} ({size:.1f} MiB) in {time.perf_counter() - t0:.1f} s")


def _rich_snippet(s: str) -> str:
    s = s.replace("[", r"\[")
    parts = s.split("**")
    return "".join(f"[bold yellow]{p}[/]" if i % 2 else p for i, p in enumerate(parts))


@app.command("search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX")],
    query: Annotated[str, typer.Argument()],
    k: Annotated[int, typer.Option("--k", "-k", min=1, max=1000)] = 10,
    scorer: Annotated[ScorerName, typer.Option()] = ScorerName.bm25,
    as_json: Annotated[bool, typer.Option("--json", help="JSON lines on stdout.")] = False,
) -> None:
    """Ranked search: AND/OR/NOT, parentheses, "phrases"."""
    if not index_path.exists():
        raise _fail(f"index file not found: {index_path} (run `findex index` first)")
    try:
        with store.open_index(index_path) as ix:
            t0 = time.perf_counter()
            total, results = search_page(ix, query, SCORERS[scorer.value], k)
            took = (time.perf_counter() - t0) * 1000
    except QuerySyntaxError as e:
        raise _fail(f"bad query: {e}", code=2) from None
    except ValueError as e:
        raise _fail(f"cannot read index: {e}") from None
    if as_json:
        for rank, r in enumerate(results, 1):
            rec = {
                "rank": rank,
                "doc_id": r.doc_id,
                "score": r.score,
                "title": r.title,
                "path": r.path,
                "snippet": r.snippet,
            }
            sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    table = Table(title=f"{query!r}: {total} matches, top {len(results)} by {scorer.value} ({took:.1f} ms)")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("document", overflow="fold", max_width=40)
    table.add_column("snippet", overflow="fold")
    for rank, r in enumerate(results, 1):
        table.add_row(
            str(rank), f"{r.score:.2f}", f"[bold]{r.title}[/]\n[dim]{r.path}[/]", _rich_snippet(r.snippet)
        )
    out.print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX")],
    top: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    """Corpus statistics from a saved index."""
    if not index_path.exists():
        raise _fail(f"index file not found: {index_path}")
    with store.open_index(index_path) as ix:
        cf: Counter[str] = Counter({t: sum(p.tf for p in plist) for t, plist in ix.items()})
        table = Table(title=str(index_path), show_header=False)
        table.add_row("documents", f"{ix.num_docs:,}")
        table.add_row("tokens", f"{sum(ix.doc_lengths.values()):,}")
        table.add_row("vocabulary", f"{len(ix):,}")
        table.add_row("avg doc length", f"{ix.avg_doc_length:,.0f}")
        table.add_row("positions", str(ix.has_positions))
        out.print(table)
        terms = Table("#", "term", "count", "df", title=f"top {top} terms")
        for i, (t, n) in enumerate(cf.most_common(top), 1):
            terms.add_row(str(i), t, f"{n:,}", f"{ix.df(t):,}")
        out.print(terms)
