"""I/O-bound counter-benchmark: 50 HTTP requests to a local server that answers after 100 ms."""

import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Slow(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # simulates network latency
        time.sleep(0.1)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass


if __name__ == "__main__":
    ThreadingHTTPServer.request_queue_size = 128  # default listen backlog 5 drops SYNs at 25+ clients
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"
    print("| Threads | 50 requests, wall (s) | Speedup |")
    print("|---|---|---|")
    base = None
    for workers in (1, 5, 10, 25, 50):
        t = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda _: urllib.request.urlopen(url).read(), range(50)))
        wall = time.perf_counter() - t
        base = base or wall
        print(f"| {workers} | {wall:.2f} | {base / wall:.1f}× |")
    srv.shutdown()
