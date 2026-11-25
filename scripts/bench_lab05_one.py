"""One build in a fresh process; prints a JSON line. Used by bench_lab05.py."""

import json
import resource
import sys
import sysconfig
import time
from pathlib import Path

from findex.parallel import BuildTimings, build_parallel


def main() -> None:
    root, executor, workers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    t = BuildTimings()
    w0, c0 = time.perf_counter(), time.process_time()
    build_parallel(root, workers, executor, positions=True, timings=t)  # type: ignore[arg-type]
    wall = time.perf_counter() - w0
    self_ru = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = self_ru.ru_utime + self_ru.ru_stime + kids.ru_utime + kids.ru_stime
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        json.dumps(
            {
                "executor": executor,
                "workers": workers,
                "wall": wall,
                "cpu": cpu,
                "cpu_parent": time.process_time() - c0,
                "rss_mib": self_ru.ru_maxrss / 1024,
                "map": t.map_s,
                "merge": t.merge_s,
                "gil": gil,
                "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
                "python": sys.version.split()[0],
            }
        )
    )


if __name__ == "__main__":  # required: spawn re-imports this module in every worker
    main()
