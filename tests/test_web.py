"""API and UI tests: a tiny fixture index injected with dependency_overrides — no index file."""

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web.app import app, get_index
from findex.web.settings import get_settings


@pytest.fixture
def client(tiny_index: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[httpx.Client]:
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "missing.bin"))  # config comes from the environment
    get_settings.cache_clear()
    app.dependency_overrides[get_index] = lambda: tiny_index
    with TestClient(app) as tc:  # runs the lifespan; INDEX_PATH points nowhere -> state.index is None
        c: httpx.Client = tc  # TestClient *is* an httpx.Client; the annotation gives pyright its types
        yield c
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_search_ok(client: httpx.Client) -> None:
    r = client.get("/search", params={"q": '"event loop"', "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2 and body["pages"] == 2 and len(body["results"]) == 1
    assert "**event**" in body["results"][0]["snippet"]
    assert r.headers["x-request-id"]


def test_pagination(client: httpx.Client) -> None:
    p1 = client.get("/search", params={"q": '"event loop"', "k": 1, "page": 1}).json()["results"]
    p2 = client.get("/search", params={"q": '"event loop"', "k": 1, "page": 2}).json()["results"]
    assert p1[0]["doc_id"] != p2[0]["doc_id"]


@pytest.mark.parametrize(
    "params", [{"q": "x", "k": "abc"}, {"q": ""}, {"q": "x", "k": 0}, {"q": "x", "scorer": "pagerank"}, {}]
)
def test_validation_422(client: httpx.Client, params: dict[str, str | int]) -> None:
    r = client.get("/search", params=params)
    assert r.status_code == 422 and "detail" in r.json()


def test_bad_query_400(client: httpx.Client) -> None:
    r = client.get("/search", params={"q": "(unclosed"})
    assert r.status_code == 400 and r.json()["detail"].startswith("bad query")


def test_document_and_404(client: httpx.Client) -> None:
    assert client.get("/docs/0").json()["path"] == "django/cache.txt"
    r = client.get("/docs/999")
    assert r.status_code == 404 and r.json() == {"detail": "no document 999"}


def test_stats(client: httpx.Client) -> None:
    assert client.get("/stats").json()["documents"] == 6


def test_health_503_when_unloaded(client: httpx.Client) -> None:
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "index not loaded"


def test_health_ok_when_loaded(client: httpx.Client, tiny_index: Index) -> None:
    app.state.index = tiny_index
    try:
        assert client.get("/health").json() == {"status": "ok"}
    finally:
        app.state.index = None


def test_ui(client: httpx.Client, tiny_index: Index) -> None:
    app.state.index = tiny_index
    try:
        page = client.get("/", params={"q": "django"}).text
        assert "<mark>Django</mark>" in page and "result" in page
        frag = client.get("/", params={"q": "django"}, headers={"HX-Request": "true"}).text
        assert "<html" not in frag and "<mark>" in frag
        assert "Django cache backends" in client.get("/doc/0").text
    finally:
        app.state.index = None


def test_openapi_lists_routes(client: httpx.Client) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= set(paths)
    assert "/_bench/search-async" not in paths
