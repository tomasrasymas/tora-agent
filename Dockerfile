# syntax=docker/dockerfile:1

# ---- Builder: resolve and install dependencies with uv ----
# uv's official image ships uv + Python; multi-arch (works on Apple Silicon).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Bytecode-compile installed packages and copy (not symlink) from the cache,
# so the resulting /app/.venv is self-contained and faster to start.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (without the project) for better layer caching:
# this layer is reused as long as pyproject.toml / uv.lock don't change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Now add the project source and install the project itself.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---- Runtime: slim image with just Python + the built venv ----
FROM python:3.13-slim-bookworm AS runtime

# Put the venv's executables (uvicorn, tora, python) first on PATH.
# UV_NO_SYNC: `uv run` reuses the venv baked at build time instead of trying to
# re-sync (which would need network access) on every invocation at runtime.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    UV_NO_SYNC=1

WORKDIR /app

# Bring uv along so skills that invoke `uv run <script>` work inside the
# container (it lives only in the builder stage otherwise).
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

# Handy CLI tools for poking around inside the container.
# ripgrep (rg) is the fast file searcher the bash tool prefers over grep.
# smbclient lets skills read files from SMB shares (e.g. the supplements plan).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl nano ripgrep smbclient \
    && rm -rf /var/lib/apt/lists/*

# Copy the resolved virtualenv and application code from the builder.
COPY --from=builder /app /app

# Defaults; override via docker-compose.yml or `docker run -e`.
# TORA_HOME points at the bind-mounted skills/config dir (see compose).
ENV TORA_HOST=0.0.0.0 \
    TORA_PORT=8888 \
    TORA_HOME=/root/.tora

EXPOSE 8888

# Console script defined in pyproject.toml ([project.scripts] tora=...).
CMD ["tora"]
