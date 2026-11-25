"""The ``findex`` command: index, search, stats.

stdout carries data (tables, ``--json`` lines); every diagnostic goes to stderr via logging.
"""

from __future__ import annotations

import json
import logging
import os
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
from findex.parallel import BuildTimings, build_parallel
from findex.query import QuerySyntaxError
from findex.ranking import search_page
from findex.scoring import SCORERS

app = typer.Typer(add_completion=False, no_args_is_help=True, help="findex — a small search engine.")
out = Console()  # stdout: data
err = Console(stderr=True)  # stderr: progress, diagnostics
log = logging.getLogger("findex")


class Exec(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


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
    workers: Annotated[int, typer.Option(min=1, help="Parallel workers.")] = os.cpu_count() or 1,
    executor: Annotated[Exec, typer.Option(help="How to parallelise the build.")] = Exec.processes,
) -> None:
    """Build an index from a corpus and save it."""
    if not corpus.exists():
        raise _fail(f"corpus not found: {corpus}")
    if corpus.is_dir() and limit is None:  # map-reduce builder (serial = one chunk)
        t0 = time.perf_counter()
        timings = BuildTimings()
        with err.status(f"indexing with {workers} {executor.value}..."):
            ix = build_parallel(corpus, workers, executor.value, positions, timings=timings)
        store.save(ix, out_path, fmt=fmt)
        size = out_path.stat().st_size / 2**20
        err.print(
            f"{ix!r} -> {out_path} ({size:.1f} MiB) in {time.perf_counter() - t0:.1f} s "
            f"(map {timings.map_s:.1f} s, merge {timings.merge_s:.1f} s)"
        )
        return
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


@app.command()
def crawl(
    seeds: Annotated[list[str], typer.Argument(help="Seed URL(s).")],
    max_pages: Annotated[int, typer.Option(min=1)] = 500,
    concurrency: Annotated[int, typer.Option(min=1, max=200)] = 10,
    per_host: Annotated[int, typer.Option(min=1, help="Max requests in flight per host.")] = 2,
    delay: Annotated[float, typer.Option(min=0, help="Min seconds between requests to one host.")] = 0.1,
    allow_domain: Annotated[
        list[str] | None, typer.Option(help="Allowed domains (default: seed hosts).")
    ] = None,
    out_path: Annotated[Path, typer.Option("--out", "-o")] = Path("data/crawl.jsonl"),
    log_path: Annotated[Path, typer.Option("--log")] = Path("crawl.log"),
    debug: Annotated[
        bool, typer.Option(help="asyncio debug mode: warns about callbacks blocking the loop.")
    ] = False,
) -> None:
    """Crawl politely; stream pages to JSON lines that `findex index` can read."""
    import asyncio

    from rich.live import Live

    from findex.crawler import CrawlStats
    from findex.crawler import crawl as crawl_pages

    stats = CrawlStats()

    def render() -> Table:
        t = Table(title="findex crawl", show_header=False)
        t.add_row("pages", f"{stats.pages:,} / {max_pages:,}")
        t.add_row("pages/s", f"{stats.pages_per_s:.1f}")
        t.add_row("in flight", str(stats.in_flight))
        t.add_row("queue", f"{stats.queued:,}")
        t.add_row("errors / retries", f"{stats.errors} / {stats.retries}")
        t.add_row("robots.txt blocked", str(stats.robots_blocked))
        return t

    async def run() -> None:
        with out_path.open("w", encoding="utf-8") as out_f, log_path.open("w", encoding="utf-8") as log_f:
            log_f.write("time\tstatus\tbytes\telapsed\tattempts\turl\terror\n")
            with Live(render(), console=err, refresh_per_second=4) as live:
                async for page in crawl_pages(
                    seeds,
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host=per_host,
                    per_host_delay=delay,
                    allowed_domains=allow_domain or (),
                    log_file=log_f,
                    stats=stats,
                ):
                    rec = {
                        "url": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # streamed, never buffered
                    live.update(render())
                live.update(render())

    asyncio.run(run(), debug=debug)
    err.print(
        f"{stats.pages} pages in {stats.elapsed:.1f} s ({stats.pages_per_s:.1f}/s), {stats.errors} errors, "
        f"{stats.retries} retries -> {out_path}; per-URL log: {log_path}"
    )


@app.command()
def serve(
    index_path: Annotated[Path | None, typer.Option("--index", help="Sets INDEX_PATH.")] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(envvar="PORT")] = 8000,
    workers: Annotated[int, typer.Option(min=1, help="uvicorn worker processes.")] = 1,
) -> None:
    """Run the web API + UI with uvicorn (config from the environment / .env)."""
    import uvicorn

    if index_path is not None:
        os.environ["INDEX_PATH"] = str(index_path)  # workers are separate processes: pass config via env
    level = logging.getLevelName(logging.getLogger().getEffectiveLevel()).lower()
    uvicorn.run(
        "findex.web.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level=level,
        proxy_headers=True,
        backlog=2048,
    )
