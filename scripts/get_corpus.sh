#!/usr/bin/env bash
# Downloads the corpus: Python, Django and NumPy documentation sources (~1600 docs, ~26 MB).
# Each .rst/.txt source file becomes one plain-text document under data/<project>/...
set -euo pipefail
OUT="$(pwd)/data"
TMP=$(mktemp -d)
clone() { git clone -q --depth 1 --filter=blob:none --sparse "https://github.com/$1.git" "$TMP/$2" && git -C "$TMP/$2" sparse-checkout set "$3"; }
copy() {  # copy <project> <subdir> <ext>: every *.<ext> under $TMP/<project>/<subdir> -> $OUT/<project>/...txt
  (cd "$TMP/$1/$2" && find . -name "*.$3" | while read -r f; do
    rel="${f#./}"
    mkdir -p "$OUT/$1/$(dirname "$rel")"
    cp "$f" "$OUT/$1/${rel%.*}.txt"
  done)
}
clone python/cpython cpython Doc
clone django/django django docs
clone numpy/numpy numpy doc/source
mkdir -p "$OUT"
copy cpython Doc rst; copy django docs txt; copy numpy doc/source rst
rm -rf "$TMP"
find "$OUT" -name '*.txt' | wc -l
