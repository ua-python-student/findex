"""@timed: log how long a function took (DEBUG level)."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger("findex.timing")


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)  # keeps __name__, __doc__, __wrapped__ of the original
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug("%s took %.1f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)

    return wrapper
