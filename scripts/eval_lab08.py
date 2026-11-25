"""precision@5 for keyword (BM25), semantic (embeddings) and hybrid (RRF) on all 15 labelled queries."""

import json
import sys
from pathlib import Path

from findex.fast import NumpyIndex
from findex.ranking import keyword_query, search_page
from findex.scoring import BM25
from findex.semantic import SemanticIndex
from findex.store import open_index

path = Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin")
qs = json.loads(Path("eval/queries.json").read_text())["queries"]
with open_index(path) as ix:
    nx, sem = NumpyIndex.open(path), SemanticIndex.open(path, ix.num_docs)
    print("| # | Query | type | P@5 keyword | P@5 semantic | P@5 hybrid |")
    print("|---|---|---|---|---|---|")
    sums = {"keyword": [0.0, 0.0], "semantic": [0.0, 0.0], "hybrid": [0.0, 0.0]}
    for i, q in enumerate(qs, 1):
        rel = set(q["relevant"])
        row = {}
        for mode in ("keyword", "semantic", "hybrid"):
            query = keyword_query(q["q"]) if mode == "keyword" else q["q"]
            _, res = search_page(ix, query, BM25(), 5, mode=mode, nx=nx, sem=sem, with_snippets=False)
            row[mode] = sum(r.path in rel for r in res) / 5
            sums[mode][1 if q.get("paraphrase") else 0] += row[mode]
        kind = "paraphrase" if q.get("paraphrase") else "keyword"
        print(
            f"| {i} | {q['q']} | {kind} | {row['keyword']:.1f} | {row['semantic']:.1f} | {row['hybrid']:.1f} |"
        )
    n_kw = sum(1 for q in qs if not q.get("paraphrase"))
    n_pp = len(qs) - n_kw
    for label, idx, n in (("mean, 10 original", 0, n_kw), ("mean, 5 paraphrases", 1, n_pp)):
        print(
            f"| | **{label}** | | **{sums['keyword'][idx] / n:.2f}** | **{sums['semantic'][idx] / n:.2f}** | **{sums['hybrid'][idx] / n:.2f}** |"
        )
    tot = {m: (v[0] + v[1]) / len(qs) for m, v in sums.items()}
    print(
        f"| | **mean, all 15** | | **{tot['keyword']:.2f}** | **{tot['semantic']:.2f}** | **{tot['hybrid']:.2f}** |"
    )
