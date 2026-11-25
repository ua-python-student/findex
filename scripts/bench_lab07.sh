#!/usr/bin/env bash
# Load-test table for Lab 7. Each row: fresh server, 3 s warm-up, then 30 s at concurrency 50.
set -euo pipefail
KA=${KA:-}   # KA=--no-keepalive: new TCP connection per request
PORT=8800
run() {  # label, workers, route
  INDEX_PATH=index.bin uv run findex serve --port $PORT --workers "$2" >/tmp/serve_bench.log 2>&1 &
  local pid=$!
  until curl -sf localhost:$PORT/health >/dev/null; do sleep 0.5; done
  uv run python scripts/loadtest.py "http://127.0.0.1:$PORT$3?q={q}&k=10" -c 50 -z 3 >/dev/null
  echo "$1 | $(uv run python scripts/loadtest.py "http://127.0.0.1:$PORT$3?q={q}&k=10" -c 50 -z 30 $KA)"
  kill $pid; wait $pid 2>/dev/null || true; sleep 1
}
run "1 worker, async def (blocking)" 1 /_bench/search-async
run "1 worker, def (thread pool)"     1 /search
run "1 worker, async def + to_thread" 1 /_bench/search-to-thread
run "2 workers, def"                  2 /search
run "4 workers, def"                  4 /search
