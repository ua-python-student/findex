#!/usr/bin/env bash
# Downloads the corpus: Python, Django and NumPy documentation sources (~1600 docs, ~26 MB).
# Each .rst/.txt source file becomes one plain-text document under data/<project>/...
set -euo pipefail
TMP=$(mktemp -d)
clone() { git clone -q --depth 1 --filter=blob:none --sparse "https://github.com/$1.git" "$TMP/$2" && git -C "$TMP/$2" sparse-checkout set "$3"; }
clone python/cpython cpython Doc
clone django/django django docs
clone numpy/numpy numpy doc/source
copy() { (cd "$TMP/$1/$2" && find . -name "*.$3" | while read -r f; do
  mkdir -p "data/$1/$(dirname "$f")"; cp "$f" "$OLDPWD/data/$1/${f%.*}.txt"; done); }
mkdir -p data
copy cpython Doc rst; copy django docs txt; copy numpy doc/source rst
find data -name '*.txt' | wc -l
