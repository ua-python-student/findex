"""precision@5 of TF-IDF vs BM25 on the hand-labelled queries (non-paraphrase ones).

Each query is run as an OR of its words: ranking, not Boolean matching, decides.
"""

import json
import sys
from pathlib import Path

from findex.ranking import search
from findex.scoring import BM25, TfIdf
from findex.store import open_index
from findex.tokenize import tokenize

idx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin")
qs = [q for q in json.loads(Path("eval/queries.json").read_text())["queries"] if not q.get("paraphrase")]


def p_at_5(results, relevant) -> float:
    return sum(r.path in relevant for r in results[:5]) / 5


with open_index(idx_path) as ix:
    print("| # | Query | relevant | P@5 TF-IDF | P@5 BM25 |")
    print("|---|---|---|---|---|")
    tot = {"tfidf": 0.0, "bm25": 0.0}
    for i, q in enumerate(qs, 1):
        or_query = " OR ".join(tokenize(q["q"]))
        row = {}
        for name, scorer in (("tfidf", TfIdf()), ("bm25", BM25())):
            row[name] = p_at_5(search(ix, or_query, scorer, 5), set(q["relevant"]))
            tot[name] += row[name]
        print(f"| {i} | {q['q']} | {len(q['relevant'])} | {row['tfidf']:.1f} | {row['bm25']:.1f} |")
    n = len(qs)
    print(f"| | **mean** | | **{tot['tfidf'] / n:.2f}** | **{tot['bm25'] / n:.2f}** |")
