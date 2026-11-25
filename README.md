# findex — a search engine built from scratch in Python

Semester project for the course *Python — Build a Search Engine*. One lab = one language feature + one project step.

## Corpus

Documentation of the Python ecosystem: **CPython `Doc/`, Django `docs/`, NumPy `doc/source/`** — 1,596 plain-text documents, ~26 MB, ~3.1 M tokens, 63.6 K distinct terms. English prose with a lot of code identifiers (`__init__`, `read_csv`), which shaped the tokenizer policy.

The corpus is **not** in git (`data/` is in `.gitignore`). To get it:

```bash
./scripts/get_corpus.sh        # sparse-clones the three repos, copies sources to data/<project>/...txt
```

## Lab 1 — Streams, not lists

```bash
uv sync
uv run python -m findex.stats data/ [--limit N] [--eager]
uv run pytest
```

* `findex.corpus.iter_documents(root)` — generator; opens one file at a time; supports a directory of `.txt` files and a `.jsonl` file (one document per line). Undecodable bytes are logged and replaced, the build does not crash.
* `findex.tokenize.tokenize(text)` — generator over `re.finditer`: NFC → `casefold()` → regex `\w+(?:'\w+)*`.
* `findex.stats` — one pass, a `Counter` is the only thing that grows; `--limit` uses `itertools.islice`.

**Tokenizer policy.** Letters of any script, digits and `_` form a token (`__init__`, `read_csv`, `utf8` stay whole — this is code documentation). Apostrophes inside a word are kept (`don't`, `п'ять`; `’` and `ʼ` are mapped to `'` first), leading/trailing ones are dropped. Hyphens split (`built-in` → `built`, `in`). Numbers split on the dot (`3.13` → `3`, `13`).

### Eager vs lazy (full corpus, median of 3 runs, `tracemalloc` peak)

| Version | Documents | Tokens | Peak memory (MiB) | Elapsed (s) |
|---|---|---|---|---|
| eager (lists) | 1,596 | 3,075,752 | 212.0 | 11.01 (±0.19) |
| lazy (generators) | 1,596 | 3,075,752 | 8.1 | 10.27 (±0.38) |

**Where the memory went.** The eager version keeps three things alive at the same time: the full text of all 1,596 documents (~26 MB of `str`), a list of lists with a reference to every one of the 3.1 M tokens (8 bytes per pointer plus a new `str` object for most tokens, ~50 bytes each), and the flattened copy of that list (another 3.1 M pointers, ~25 MB). The lazy version holds at most one document's text and the current match; what remains in its 8 MiB peak is the `Counter` itself — 63.6 K keys and int values in a hash table — plus the list of file *paths*. That is the answer, so it has to be in memory; everything upstream is a stream. Time is roughly the same: the work (regex + hashing) is identical, the eager version just additionally allocates and frees the lists.

## Lab 2 — The inverted index

```bash
uv run python -m findex.index data/ --out index.bin [--positions] [--format pickle|binary]
uv run python -m findex.search index.bin "dict OR set NOT django" [--engine merge|set]
uv run python scripts/bench_lab02.py data        # every table below
```

* `findex.models` — `Posting(doc_id, tf, positions=())` and `DocMeta(path, title)`, both `@dataclass(frozen=True, slots=True)`: hashable, no per-instance `__dict__`.
* `findex.index.build_index` — one pass over the lazy Lab 1 pipeline; per document a `Counter` is flushed into a `defaultdict(list)`. Documents come in `doc_id` order, so the postings are sorted by construction. Output: `postings`, `doc_lengths`, `doc_meta`. `--positions` stores token offsets.
* `findex.search` — two-pointer `merge_and/or/not` and `set_*` alternatives; left-to-right query `a b`, `a OR b`, `a NOT b`.
* `findex.store` — `pickle` (with the RCE warning as a comment in `load_pickle`) and a binary format: magic + JSON header (term → offset, count) + contiguous `array('I')` buffers.

Index: 1,596 documents, 63,582 terms, 650,769 postings.

### Save / load (median of 3, without tracemalloc)

| Format | Positions | File size (MiB) | Save (s) | Load (s) |
|---|---|---|---|---|
| pickle | no | 10.9 | 2.71 | 3.62 |
| binary | no | 6.7 | 0.17 | 1.69 |
| pickle | yes | 20.2 | 3.51 | 4.01 |
| binary | yes | 18.5 | 0.74 | 2.12 |

Positions cost ×2.8 in the binary file (6.7 → 18.5 MiB): one 4-byte number per *token* (3.1 M) instead of two per *posting* (0.65 M). Loading is dominated by re-creating 650 K `Posting` objects, not by I/O; the binary format saves fast because `array.tobytes()` is one memcpy per term.

> **pickle is a security hole.** `pickle.load` runs the opcodes in the file, and a `__reduce__` can call any function (`os.system`). It is fine only for an index this program wrote itself — never for a file from the network.

### Merge vs set (µs per operation, `timeit`)

| Terms (df) | Op | merge (µs) | set (µs) | result |
|---|---|---|---|---|
| `the` (1474) & `in` (1480) | AND | 153.9 | 62.6 | 1429 docs |
| `the` (1474) & `in` (1480) | OR | 170.1 | 71.0 | 1525 docs |
| `prebuilt` (2) & `sphinxbuild` (2) | AND | 0.2 | 0.3 | 1 docs |
| `prebuilt` (2) & `sphinxbuild` (2) | OR | 0.3 | 0.4 | 3 docs |
| `in` (1480) & `author` (100) | AND | 138.2 | 22.6 | 100 docs |
| `in` (1480) & `author` (100) | OR | 146.9 | 41.2 | 1480 docs |

Both are O(len(a) + len(b)), but the set version does its loop in C (building the hash tables, intersecting, sorting), while the merge does every comparison and index increment as Python bytecode — 2.5–6× slower on the frequent terms. On the rare terms (two elements) the merge wins slightly: no set objects to allocate. The merge is still the right *algorithm* for an engine: it streams, needs no extra memory and works on compressed or memory-mapped postings (Lab 8 moves it into NumPy).

### Memory study (postings only, same corpus, `tracemalloc`)

| Postings representation | Peak memory, build (MiB) | Retained (MiB) | Pickle file (MiB) | Load (s) |
|---|---|---|---|---|
| `list[Posting]`, plain `@dataclass` | 71.3 | 69.5 | 12.7 | 0.41 |
| `list[Posting]`, `slots=True` | 46.5 | 44.6 | 10.2 | 1.49 |
| `array('I')` pairs, no objects | 26.3 | 24.4 | 7.8 | 0.09 |

**Where the bytes went.** 650,769 postings. *Plain dataclass*: ~110 bytes per posting — the object header and its attribute storage (on 3.13 the values live inline, so the per-instance dict is smaller than the "100+ bytes" of older Pythons, but each object still carries the inline-values block and a dict pointer), plus an 8-byte pointer from the list. *Slots*: the object is a fixed 2-slot C struct (`getsizeof` 48 → no hidden dict), ~72 bytes per posting with the list pointer; ×1.5 smaller. Small `int`s (< 257) are cached singletons, and larger doc_ids are separate 28-byte `int` objects in both object rows. *Arrays*: 8 bytes per posting (two raw `uint32`), i.e. ~5 MiB for all postings; the remaining ~20 MiB is per-*term* overhead — 63.6 K keys (`str` objects ~55 B), dict entries and two `array` headers per term — and most terms have df = 1, so for this corpus term overhead dominates. Pickle load of the slotted class is *slower* (1.49 s vs 0.41 s) because `slots=True` dataclasses pickle through a Python-level `__getstate__/__setstate__`, while plain instances use the C fast path for `__dict__`.

REPL experiments (`scripts/experiments/lab02_repl.py`): `dict` grows at 6 → 11 → 22 entries (2/3 load factor); a `__hash__` returning 42 makes 10,000 lookups in a 2,000-element set take 1,202 ms instead of 0.9 ms; `hash('hello')` differs between processes unless `PYTHONHASHSEED=0`; 1 M three-field objects take 130 MiB plain vs 92 MiB slotted; `array('I', range(10**6))` is 3.9 MiB vs 7.6 MiB for the list *plus* 26.7 MiB of `int` objects it points to.
