"""Eager vs lazy benchmark for the README (median of N runs, fresh process each)."""

import re
import statistics
import subprocess
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

print("| Version | Documents | Tokens | Peak memory (MiB) | Elapsed (s) |")
print("|---|---|---|---|---|")
for mode in ("eager", "lazy"):
    peaks, times = [], []
    for _ in range(RUNS):
        cmd = [sys.executable, "-m", "findex.stats", ROOT, "--top", "1"]
        out = subprocess.run(
            cmd + (["--eager"] if mode == "eager" else []),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        docs = re.search(r"documents:\s+([\d,]+)", out).group(1)
        toks = re.search(r"tokens:\s+([\d,]+)", out).group(1)
        times.append(float(re.search(r"elapsed: ([\d.]+)", out).group(1)))
        peaks.append(float(re.search(r"peak memory: ([\d.]+)", out).group(1)))
    label = "eager (lists)" if mode == "eager" else "lazy (generators)"
    print(
        f"| {label} | {docs} | {toks} | {statistics.median(peaks):.1f} | "
        f"{statistics.median(times):.2f} (±{(max(times) - min(times)) / 2:.2f}) |"
    )
