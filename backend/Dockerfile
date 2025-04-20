# ──── STAGE 1: Builder ────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install build tools & curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed by uv & transformer for bytecode compilation)
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Pull in uv & uvx executables
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# Prep our virtualenv + UV settings
ENV PATH="/app/.venv/bin:${PATH}" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Create venv & install Python deps 
COPY pyproject.toml uv.lock ./
RUN python -m venv .venv

# Install all dependencies into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy /app dir and compile its bytecode
COPY app/ ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# ──── STAGE 2: Runtime ────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Bring in only what we need for runtime:
COPY --from=builder /app/.venv    /app/.venv
COPY --from=builder /bin/uv       /bin/uv
COPY --from=builder /bin/uvx      /bin/uvx
COPY --from=builder /app/app      /app/app


CMD ["fastapi", "run", "--workers", "4", "app/main.py"]


