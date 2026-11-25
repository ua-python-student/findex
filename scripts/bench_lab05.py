"""Run every (executor, workers) configuration RUNS times in fresh processes; write JSONL.

usage: python scripts/bench_lab05.py <python-exe> <out.jsonl> [executors] [workers] [runs]
"""

import json
import subprocess
import sys

py, out = sys.argv[1], sys.argv[2]
executors = (sys.argv[3] if len(sys.argv) > 3 else "serial,threads,processes").split(",")
workers = [int(w) for w in (sys.argv[4] if len(sys.argv) > 4 else "1,2,4,8").split(",")]
runs = int(sys.argv[5]) if len(sys.argv) > 5 else 3

with open(out, "a") as f:
    for ex in executors:
        for w in [1] if ex == "serial" else workers:
            for r in range(runs + 1):  # first run = warm-up (page cache, .pyc), discarded
                res = subprocess.run(
                    [py, "scripts/bench_lab05_one.py", "data", ex, str(w)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                rec = json.loads(res.stdout)
                if r == 0:
                    continue
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(
                    ex,
                    w,
                    f"{rec['wall']:.2f}s cpu {rec['cpu']:.2f}s rss {rec['rss_mib']:.0f} MiB "
                    f"merge {rec['merge']:.2f}s gil={rec['gil']}",
                    flush=True,
                )
