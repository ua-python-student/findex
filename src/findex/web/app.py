"""The FastAPI application.

INDEX_PATH=index.bin uvicorn findex.web.app:app      (or: findex serve)
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex import __version__, store
from findex.index import Index
from findex.query import QuerySyntaxError
from findex.ranking import search_page
from findex.scoring import SCORERS
from findex.web.schemas import DocumentOut, Health, SearchHit, SearchResponse, StatsOut
from findex.web.settings import Settings, get_settings

log = logging.getLogger("findex.web")
HERE = Path(__file__).parent
ScorerName = Literal["bm25", "tfidf"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup: load the index ONCE (in a thread, so the loop stays responsive). Shutdown: release it."""
    settings = get_settings()
    if not logging.getLogger().handlers:  # running under plain uvicorn: logs to stdout (12-factor XI)
        logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("findex").setLevel(settings.log_level)
    app.state.index = None
    app.state.started = time.monotonic()
    if settings.index_path.exists():
        app.state.index = await asyncio.to_thread(store.load, settings.index_path)
        log.info("loaded %r from %s", app.state.index, settings.index_path)
    else:
        log.error("index not found at %s: /health will report 503", settings.index_path)
    yield
    if app.state.index is not None:
        app.state.index.close()
        app.state.index = None
    log.info("shut down cleanly")


app = FastAPI(
    title="findex",
    version=__version__,
    description="A search engine built from scratch: BM25 / TF-IDF over an inverted index.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def highlight(snippet: str) -> Markup:
    """Escape first, then turn the ranking module's **term** markers into <mark>."""
    parts = str(escape(snippet)).split("**")
    return Markup("".join(f"<mark>{p}</mark>" if i % 2 else p for i, p in enumerate(parts)))


templates.env.filters["highlight"] = highlight


# --- dependencies ---------------------------------------------------------------
def get_index(request: Request) -> Index:
    index: Index | None = request.app.state.index
    if index is None:
        raise HTTPException(503, "index is not loaded")
    return index


IndexDep = Annotated[Index, Depends(get_index)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# --- request ids + clean 500s ---------------------------------------------------
@app.middleware("http")
async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error, request_id=%s %s %s", rid, request.method, request.url.path)
        response = JSONResponse({"detail": "internal server error", "request_id": rid}, status_code=500)
    response.headers["x-request-id"] = rid
    log.debug(
        "%s %s -> %d in %.1f ms [%s]",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - t0) * 1000,
        rid,
    )
    return response


# --- the API ----------------------------------------------------------------------
def run_search(index: Index, q: str, k: int, scorer: ScorerName, page: int) -> SearchResponse:
    """Blocking, CPU-bound: parse + score + snippets."""
    t0 = time.perf_counter()
    try:
        total, results = search_page(index, q, SCORERS[scorer], k=k, offset=(page - 1) * k)
    except QuerySyntaxError as e:
        raise HTTPException(400, f"bad query: {e}") from None
    return SearchResponse(
        query=q,
        scorer=scorer,
        total=total,
        page=page,
        k=k,
        pages=max(1, math.ceil(total / k)),
        took_ms=round((time.perf_counter() - t0) * 1000, 2),
        results=[SearchHit.from_result(r) for r in results],
    )


Q = Annotated[
    str, Query(min_length=1, max_length=200, description='e.g. asyncio (TaskGroup OR gather) "event loop"')
]
K = Annotated[int, Query(ge=1, le=100)]
PageNo = Annotated[int, Query(ge=1, le=1000)]


@app.get("/search", response_model=SearchResponse, tags=["api"])
def search(index: IndexDep, q: Q, k: K = 10, scorer: ScorerName = "bm25", page: PageNo = 1) -> SearchResponse:
    """Ranked search. Plain `def`: FastAPI runs it in its thread pool, so the CPU work
    never blocks the event loop (measured: see README, Lab 7)."""
    return run_search(index, q, k, scorer, page)


@app.get("/docs/{doc_id}", response_model=DocumentOut, tags=["api"])
def document(index: IndexDep, doc_id: int) -> DocumentOut:
    if doc_id not in index.doc_meta:
        raise HTTPException(404, f"no document {doc_id}")
    meta = index.doc_meta[doc_id]
    return DocumentOut(
        doc_id=doc_id,
        title=meta.title,
        path=meta.path,
        length=index.doc_length(doc_id),
        text=index.doc_text(doc_id),
    )


@app.get("/stats", response_model=StatsOut, tags=["api"])
def stats(request: Request, index: IndexDep, settings: SettingsDep) -> StatsOut:
    path = settings.index_path
    return StatsOut(
        documents=index.num_docs,
        vocabulary=len(index),
        tokens=sum(index.doc_lengths.values()),
        positions=index.has_positions,
        index_path=str(path),
        index_bytes=path.stat().st_size if path.exists() else 0,
        uptime_s=round(time.monotonic() - request.app.state.started, 1),
        version=__version__,
    )


@app.get("/health", response_model=Health, tags=["ops"], responses={503: {"model": Health}})
def health(request: Request) -> JSONResponse:
    if request.app.state.index is None:
        return JSONResponse({"status": "index not loaded"}, status_code=503)
    return JSONResponse({"status": "ok"})


# --- the M2 experiment: the same work in an `async def` ------------------------
@app.get("/_bench/search-async", response_model=SearchResponse, include_in_schema=False)
async def search_async_blocking(index: IndexDep, q: Q, k: K = 10) -> SearchResponse:
    """WRONG on purpose: CPU work directly inside a coroutine blocks the loop."""
    return run_search(index, q, k, "bm25", 1)


@app.get("/_bench/search-to-thread", response_model=SearchResponse, include_in_schema=False)
async def search_async_to_thread(index: IndexDep, q: Q, k: K = 10) -> SearchResponse:
    """async def done right: offload with asyncio.to_thread."""
    return await asyncio.to_thread(run_search, index, q, k, "bm25", 1)


# --- the UI -------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request, q: str = "", page: PageNo = 1, scorer: ScorerName = "bm25") -> HTMLResponse:
    index: Index | None = request.app.state.index
    ctx: dict[str, object] = {
        "q": q,
        "page": page,
        "scorer": scorer,
        "res": None,
        "error": None,
        "num_docs": index.num_docs if index else 0,
    }
    if q.strip() and index is not None:
        try:
            ctx["res"] = run_search(index, q[:200], 10, scorer, page)
        except HTTPException as e:
            ctx["error"] = e.detail
    template = "_results.html" if request.headers.get("hx-request") else "index.html"
    return templates.TemplateResponse(request, template, ctx)


@app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
def doc_page(request: Request, index: IndexDep, doc_id: int, q: str = "") -> HTMLResponse:
    d = document(index, doc_id)
    return templates.TemplateResponse(request, "doc.html", {"d": d, "q": q})
