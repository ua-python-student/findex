"""Top lines of a Scalene JSON profile, split into Python / native / system time."""

import json
import sys

d = json.load(open(sys.argv[1]))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rows = []
for fname, f in d["files"].items():
    if "findex" not in fname:
        continue
    for line in f["lines"]:
        py, nat, sysp = line["n_cpu_percent_python"], line["n_cpu_percent_c"], line["n_sys_percent"]
        if py + nat + sysp > 1:
            rows.append(
                (
                    py + nat + sysp,
                    py,
                    nat,
                    sysp,
                    fname.split("/")[-1],
                    line["lineno"],
                    line["line"].strip()[:60],
                )
            )
print("| CPU % | Python % | native % | system % | line | code |\n|---|---|---|---|---|---|")
for r in sorted(rows, reverse=True)[:n]:
    print(f"| {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} | `{r[4]}:{r[5]}` | `{r[6]}` |")
