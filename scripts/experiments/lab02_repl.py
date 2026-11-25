"""Lab 2 'prove it to yourself' experiments, as a script."""

import subprocess
import sys
import timeit
import tracemalloc
from array import array
from dataclasses import dataclass

d, sizes = {}, {}
for i in range(23):
    d[i] = i
    sizes[len(d)] = sys.getsizeof(d)
print("1) dict getsizeof by len:", {k: sizes[k] for k in (1, 5, 6, 11, 12, 22, 23)})


class Key:
    def __init__(self, v):
        self.v = v

    def __eq__(self, o):
        return isinstance(o, Key) and self.v == o.v


try:
    {Key(1)}
except TypeError as e:
    print("2) eq without hash:", e)


class Key42(Key):
    def __hash__(self):
        return 42


class KeyGood(Key):
    def __hash__(self):
        return hash(self.v)


for cls in (KeyGood, Key42):
    s = {cls(i) for i in range(2000)}
    probe = cls(1999)
    print(
        f"   {cls.__name__}: 10,000 lookups in a 2,000-element set: {timeit.timeit(lambda: probe in s, number=10_000) * 1000:.1f} ms"
    )

out = [
    subprocess.run(
        [sys.executable, "-c", "print(hash('hello'))"],
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    for env in (None, None, {"PYTHONHASHSEED": "0"}, {"PYTHONHASHSEED": "0"})
]
print("3) hash('hello') in 4 processes (last two PYTHONHASHSEED=0):", out)


@dataclass
class P3:
    a: int
    b: int
    c: int


@dataclass(slots=True)
class S3:
    a: int
    b: int
    c: int


for cls in (P3, S3):
    tracemalloc.start()
    xs = [cls(i, i, i) for i in range(1_000_000)]
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"4) 1M x {cls.__name__} ({'slots' if cls is S3 else 'plain'}): {cur / 2**20:.1f} MiB, getsizeof(one)={sys.getsizeof(xs[0])}"
    )
    del xs

a, l = array("I", range(10**6)), list(range(10**6))
print(
    f"5) getsizeof array('I')={sys.getsizeof(a) / 2**20:.2f} MiB, list={sys.getsizeof(l) / 2**20:.2f} MiB "
    f"(+ {sum(sys.getsizeof(x) for x in l) / 2**20:.1f} MiB of int objects the list points to)"
)
