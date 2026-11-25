"""Profiling targets for Lab 8.

  python scripts/profile_lab08.py search [--scorer python|numpy]   # 30 x a 3-term query, no cache
  python scripts/profile_lab08.py index                            # serial index build
Run under cProfile / py-spy / scalene.
"""

import sys
import time
from pathlib import Path

QUERY = "python OR thread OR process"


def search(engine: str) -> None:
    from findex import store
    from findex.ranking import rank
    from findex.scoring import BM25

    ix = store.load(Path("index.bin"))
    nx = None
    if engine == "numpy":
        from findex.fast import NumpyIndex

        nx = NumpyIndex.open(Path("index.bin"))
    prof = None
    if "--cprofile" in sys.argv:  # profile only the ranking loop, not the load
        import cProfile

        prof = cProfile.Profile()
        prof.enable()
    t = time.perf_counter()
    if nx is not None:
        for i in range(30):
            nx.rank(QUERY, "bm25", 10 + i)
    else:
        for i in range(30):
            rank(ix, QUERY, BM25(), 10 + i)
    if prof is not None:
        prof.disable()
        prof.dump_stats(f"docs/profiles/rank_{engine}.prof")
    print(
        f"30 x rank({QUERY!r}) [{engine}]: {(time.perf_counter() - t) * 1000 / 30:.1f} ms/query",
        file=sys.stderr,
    )


def index() -> None:
    from findex.parallel import build_parallel

    build_parallel(Path("data"), 1, "serial", positions=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "search":
        search("numpy" if "numpy" in sys.argv else "python")
    else:
        index()
