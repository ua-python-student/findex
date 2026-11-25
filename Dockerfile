# syntax=docker/dockerfile:1
# --- build stage: resolve and install dependencies from uv.lock ------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

# --- runtime stage: slim image, no uv, no build tools, non-root -------------------------
FROM python:3.13-slim-bookworm
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 INDEX_PATH=/app/index.bin PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
# the index is baked into the image (built in CI: `findex index data/ --out index.bin`), or mount one over it
CMD ["sh", "-c", "uvicorn findex.web.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
