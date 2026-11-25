"""Persistence: pickle and a purpose-built binary format.

Binary layout::

    b"FIDX1\\n" | uint64 header_len | header (UTF-8 JSON) | postings blob

The header maps term -> [byte_offset, n_postings, has_positions] and carries
doc_lengths, doc_meta and doc_id -> [offset, length] of the zlib-compressed
document texts, which follow the postings in the blob. For each term the blob holds three uint32 arrays
back to back: doc_ids[n], tfs[n] and, if positions were indexed,
positions[sum(tfs)] (a posting has exactly ``tf`` positions).
"""

from __future__ import annotations

import json
import pickle
import struct
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, TypedDict, cast

from findex.index import Index
from findex.models import DocMeta, Posting
from findex.timing import timed

MAGIC = b"FIDX1\n"
Format = Literal["pickle", "binary"]


class Header(TypedDict):
    terms: dict[str, list[int]]  # term -> [offset, n_postings, has_positions]
    doc_lengths: dict[str, int]  # JSON object keys are always strings
    doc_meta: dict[str, list[str]]  # doc_id -> [path, title]
    texts: dict[str, list[int]]  # doc_id -> [offset, length]


def save(index: Index, path: Path, fmt: Format = "binary") -> None:
    if fmt == "pickle":
        save_pickle(index, path)
    else:
        save_binary(index, path)


@timed
def load(path: Path) -> Index:
    with open(path, "rb") as f:
        head = f.read(len(MAGIC))
    return load_binary(path) if head == MAGIC else load_pickle(path)


@contextmanager
def open_index(path: Path) -> Generator[Index]:
    """``with open_index(p) as ix:`` — load, and release the index even if the body raises."""
    index = load(path)
    try:
        yield index
    finally:
        index.close()


# --- pickle --------------------------------------------------------------------


def save_pickle(index: Index, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Index:
    # SECURITY: pickle.load executes whatever the byte stream tells it to:
    # a pickle can carry a __reduce__ that calls os.system(...). This is safe
    # ONLY because we load an index file this program wrote itself. Never
    # call this on a file downloaded from the internet or received over a
    # network — that is remote code execution.
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Index):
        raise TypeError(f"{path} does not contain an Index")
    return obj


# --- binary ---------------------------------------------------------------------


def save_binary(index: Index, path: Path) -> None:
    blob = bytearray()
    terms: dict[str, list[int]] = {}
    for term, plist in index.items():
        has_pos = bool(plist and plist[0].positions)
        terms[term] = [len(blob), len(plist), int(has_pos)]
        blob += array("I", (p.doc_id for p in plist)).tobytes()
        blob += array("I", (p.tf for p in plist)).tobytes()
        if has_pos:
            blob += array("I", (x for p in plist for x in p.positions)).tobytes()
    texts: dict[int, list[int]] = {}
    for doc_id, blob_ in index.texts.items():
        texts[doc_id] = [len(blob), len(blob_)]
        blob += blob_
    header = json.dumps(
        {
            "terms": terms,
            "doc_lengths": index.doc_lengths,
            "doc_meta": {d: [m.path, m.title] for d, m in index.doc_meta.items()},
            "texts": texts,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(blob)


def read_header(path: Path) -> tuple[Header, int]:
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not a findex binary index")
        (n,) = struct.unpack("<Q", f.read(8))
        # cast: json.loads returns Any; the layout is ours (save_binary), not external input
        header = cast(Header, json.loads(f.read(n)))
    return header, len(MAGIC) + 8 + n


def load_binary(path: Path) -> Index:
    header, data_start = read_header(path)
    data = memoryview(Path(path).read_bytes())[data_start:]
    postings: dict[str, list[Posting]] = {}
    terms = header["terms"]
    for term, (off, n, has_pos) in terms.items():
        ids = array("I")
        ids.frombytes(data[off : off + 4 * n])
        tfs = array("I")
        tfs.frombytes(data[off + 4 * n : off + 8 * n])
        if has_pos:
            pos = array("I")
            pos.frombytes(data[off + 8 * n : off + 8 * n + 4 * sum(tfs)])
            plist: list[Posting] = []
            k = 0
            for d, tf in zip(ids, tfs, strict=True):
                plist.append(Posting(d, tf, tuple(pos[k : k + tf])))
                k += tf
            postings[term] = plist
        else:
            postings[term] = [Posting(d, tf) for d, tf in zip(ids, tfs, strict=True)]
    lengths = header["doc_lengths"]
    meta = header["doc_meta"]
    text_ranges = header.get("texts", {})
    return Index(
        postings,
        {int(d): n for d, n in lengths.items()},  # JSON turned int keys into strings
        {int(d): DocMeta(p, t) for d, (p, t) in meta.items()},
        {int(d): bytes(data[o : o + n]) for d, (o, n) in text_ranges.items()},
    )
