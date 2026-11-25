"""The three ranking sanity checks of Lab 3, on the real corpus."""

import sys
from pathlib import Path

from findex.ranking import search
from findex.scoring import BM25, TfIdf
from findex.store import open_index

with open_index(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin")) as ix:
    bm25, tfidf = BM25(), TfIdf()
    # 1) rare term beats common term
    q = "the OR pickletools"
    res = search(ix, q, bm25, 2000)
    rare_docs = {p.doc_id for p in ix["pickletools"]}
    best_rare = next(r for r in res if r.doc_id in rare_docs)
    best_common_only = next(r for r in res if r.doc_id not in rare_docs)
    print(f"1) query {q!r}: df(the)={ix.df('the')}, df(pickletools)={ix.df('pickletools')}")
    print(
        f"   best doc with 'pickletools': rank {res.index(best_rare) + 1}, score {best_rare.score:.2f} ({best_rare.path})"
    )
    print(
        f"   best doc with only 'the':    rank {res.index(best_common_only) + 1}, score {best_common_only.score:.2f} ({best_common_only.path})"
    )

    # 2) saturation: contribution of one term as tf grows, doc of average length
    term = "generator"
    from findex.models import Posting

    avg_doc = min(ix.doc_lengths, key=lambda d: abs(ix.doc_length(d) - ix.avg_doc_length))
    print(
        f"2) BM25 contribution of {term!r} (df={ix.df(term)}) in a doc of average length ({ix.doc_length(avg_doc)} tokens):"
    )
    prev = 0.0
    for tf in (1, 2, 5, 10, 19, 20):
        s = bm25.score(term, Posting(avg_doc, tf), ix)
        t = tfidf.score(term, Posting(avg_doc, tf), ix)
        print(f"   tf={tf:>2}: BM25={s:.3f}  TF-IDF={t:.3f}")
        prev = s
    s19, s20 = (bm25.score(term, Posting(avg_doc, tf), ix) for tf in (19, 20))
    print(
        f"   the 20th occurrence adds {s20 - s19:.4f} ({(s20 - s19) / s20:.2%} of the score); k1={bm25.k1}, upper bound = idf*(k1+1) = {bm25.idf(ix.num_docs, ix.df(term)) * (bm25.k1 + 1):.3f}"
    )

    # 3) short doc with one match beats long doc with one match
    term = "sentinel"
    once = [p for p in ix[term] if p.tf == 1]
    short = min(once, key=lambda p: ix.doc_length(p.doc_id))
    long_ = max(once, key=lambda p: ix.doc_length(p.doc_id))
    print(f"3) {term!r} occurs once in both docs:")
    for label, p in (("short", short), ("long", long_)):
        print(
            f"   {label:5} {ix.doc_length(p.doc_id):>6} tokens  BM25={bm25.score(term, p, ix):.3f}  TF-IDF={tfidf.score(term, p, ix):.3f}  ({ix.doc_meta[p.doc_id].path})"
        )
    res = search(ix, term, bm25, 1000)
    ranks = {r.doc_id: i for i, r in enumerate(res, 1)}
    print(
        f"   rank in `{term}` results: short #{ranks[short.doc_id]}, long #{ranks[long_.doc_id]} of {len(res)}"
    )
