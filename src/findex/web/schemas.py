"""Pydantic models at the edge. Internal code keeps the Lab 2–3 dataclasses."""

from __future__ import annotations

from pydantic import BaseModel, Field

from findex.ranking import SearchResult


class SearchHit(BaseModel):
    doc_id: int
    title: str
    path: str
    score: float
    snippet: str

    @classmethod
    def from_result(cls, r: SearchResult) -> SearchHit:  # the one deliberate seam
        return cls(doc_id=r.doc_id, title=r.title, path=r.path, score=r.score, snippet=r.snippet)


class SearchResponse(BaseModel):
    query: str
    scorer: str
    total: int = Field(description="documents matching the query")
    page: int
    k: int
    pages: int
    took_ms: float
    results: list[SearchHit]


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int = Field(description="tokens")
    text: str


class StatsOut(BaseModel):
    documents: int
    vocabulary: int
    tokens: int
    positions: bool
    index_path: str
    index_bytes: int
    uptime_s: float
    version: str


class Health(BaseModel):
    status: str
