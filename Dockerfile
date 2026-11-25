# syntax=docker/dockerfile:1
# --- build stage: resolve and install dependencies from uv.lock ------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# dependencies first (their layer is cached until uv.lock changes), then the project itself
COPY pyproject.toml uv.lock README.md /app/
RUN uv sync --locked --no-install-project --no-dev
COPY . /app
RUN uv sync --locked --no-dev
# Build the index inside the image (the corpus is not in git): download the docs, index, embed, drop the raw corpus.
# --no-positions keeps the running service under 512 MB (free hosting tiers); phrase queries need positions.
ARG FINDEX_POSITIONS=--no-positions
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/* \
 && ./scripts/get_corpus.sh \
 && .venv/bin/findex index data --out index.bin --executor serial ${FINDEX_POSITIONS} \
 && .venv/bin/findex embed index.bin \
 && rm -rf data

# --- runtime stage: slim image, no uv, no build tools, non-root -------------------------
FROM python:3.13-slim-bookworm
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 INDEX_PATH=/app/index.bin PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
# the index is built into the image above; mount another one over /app/index.bin to replace it
CMD ["sh", "-c", "uvicorn findex.web.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
