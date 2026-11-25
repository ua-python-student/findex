"""Where does one /search request's time go after the scorer was vectorised?"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
from loadtest import make_queries  # noqa: E402

from findex import store  # noqa: E402
from findex.fast import NumpyIndex  # noqa: E402
from findex.query import parse  # noqa: E402
from findex.ranking import rank  # noqa: E402
from findex.scoring import BM25  # noqa: E402
from findex.snippets import snippet  # noqa: E402

ix = store.load(Path("index.bin"))
nx = NumpyIndex.open(Path("index.bin"))
qs = make_queries(200)
t = {"rank (Lab 3, Python)": 0.0, "rank (NumPy)": 0.0, "decompress 10 docs": 0.0, "snippets for 10 docs": 0.0}
for q in qs:
    a = time.perf_counter()
    rank(ix, q, BM25(), 10)
    b = time.perf_counter()
    _, top = nx.rank(q, "bm25", 10)
    c = time.perf_counter()
    texts = [ix.doc_text(d) for d, _ in top]
    d_ = time.perf_counter()
    terms = list(parse(q).terms())
    for tx in texts:
        snippet(tx, terms)
    e = time.perf_counter()
    t["rank (Lab 3, Python)"] += b - a
    t["rank (NumPy)"] += c - b
    t["decompress 10 docs"] += d_ - c
    t["snippets for 10 docs"] += e - d_
print("| Step (per request, mean of 200 load-test queries) | ms |\n|---|---|")
for k, v in t.items():
    print(f"| {k} | {v / len(qs) * 1000:.2f} |")
