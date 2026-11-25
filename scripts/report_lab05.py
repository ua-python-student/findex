"""Aggregate bench/lab05*.jsonl into the README table and docs/lab05_speedup.png."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = defaultdict(list)
for name, label in (("bench/lab05.jsonl", "3.13"), ("bench/lab05_ft.jsonl", "3.13t")):
    for line in Path(name).read_text().splitlines():
        r = json.loads(line)
        rows[(label, r["executor"], r["workers"])].append(r)

base = statistics.median(r["wall"] for r in rows[("3.13", "serial", 1)])
print(
    "| Python | Executor | Workers | Wall (s, median) | spread | CPU (s) | Peak RSS (MiB) | Merge (s) | Speedup vs 3.13 serial |"
)
print("|---|---|---|---|---|---|---|---|---|")
med = {}
for (py, ex, w), rs in rows.items():
    wall = statistics.median(r["wall"] for r in rs)
    med[(py, ex, w)] = wall
    print(
        f"| {py}{' (GIL off)' if not rs[0]['gil'] else ''} | {ex} | {w} | {wall:.2f} | ±{(max(r['wall'] for r in rs) - min(r['wall'] for r in rs)) / 2:.2f} | "
        f"{statistics.median(r['cpu'] for r in rs):.2f} | {statistics.median(r['rss_mib'] for r in rs):.0f} | "
        f"{statistics.median(r['merge'] for r in rs):.2f} | {base / wall:.2f}× |"
    )

ws = [1, 2, 4, 8]
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
ax.plot([1, 2, 8], [1, 2, 2], color="#9c9c9c", lw=1.5, ls="--", label="ideal on 2 cores")
ax.axvline(2, color="#cccccc", lw=1)
ax.text(2.05, 2.45, "2 cores = os.cpu_count()", color="#555555", fontsize=8)
series = [
    ("3.13", "threads", "#257de0", "o", "threads, GIL"),
    ("3.13", "processes", "#f47230", "s", "processes"),
    ("3.13t", "threads", "#00b894", "^", "threads, free-threaded 3.13t"),
]
for py, ex, color, marker, label in series:
    own_base = med[(py, "serial", 1)]
    ys = [own_base / med[(py, ex, w)] for w in ws]
    ax.plot(ws, ys, color=color, lw=2, marker=marker, ms=7, label=label)
    ax.annotate(
        f"{ys[1]:.2f}×", (2, ys[1]), textcoords="offset points", xytext=(8, -4), fontsize=8, color="#333333"
    )
ax.set_xscale("log", base=2)
ax.set_xticks(ws, [str(w) for w in ws])
ax.set_ylim(0, 2.6)
ax.set_xlabel("workers")
ax.set_ylabel("speedup vs serial (same interpreter)")
ax.set_title("findex index: parallel build speedup, 1,596 docs", fontsize=11, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#eeeeee")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("docs/lab05_speedup.png")
