# ── base: shared deps ─────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install prod dependencies only (no dev, no project code yet)
RUN uv sync --frozen --no-install-project --no-dev


# ── dev: mounts code via volume, runs with reload ─────────────
FROM base AS dev

# Install dev dependencies too
RUN uv sync --frozen --no-install-project

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# ── prod: copies code, no reload ──────────────────────────────
FROM base AS prod

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]