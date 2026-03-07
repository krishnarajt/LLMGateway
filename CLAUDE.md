# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Common Development Commands

| Task | Command | Notes |
|------|---------|-------|
| **Run the application locally** | `uvicorn main:app --host 0.0.0.0 --port 8000` | Uses the FastAPI entry point defined in `main.py`. |
| **Run with auto‑reload (development)** | `uvicorn main:app --reload` | Picks up code changes without restarting. |
| **Run the test suite** | `pytest` | Tests are defined in the `tests/` directory (if present). |
| **Run a single test file** | `pytest tests/path/to/test_file.py` | Replace the path with the desired test file. |
| **Run a single test case** | `pytest tests/path/to/test_file.py::TestClass::test_method` | Use the fully‑qualified test identifier. |
| **Build a Docker image** | `docker build -t scron:latest .` | Dockerfile is in the repository root. |
| **Start services with Docker Compose** | `docker compose up` | Brings up the FastAPI service and the PostgreSQL container defined in `docker-compose.yml`. |
| **Run lint / format (if installed)** | `ruff check .` or `black .` | Not part of the default dependencies, but you can add your preferred linter/formatter. |
| **Initialize the database** | `python -c "from app.db.database import init_db; init_db()"` | Creates tables using SQLAlchemy models. |

---

## High‑Level Architecture Overview

```
└─ scron/                     # Repository root
   ├─ app/                    # Application package
   │   ├─ __init__.py
   │   ├─ main.py             # FastAPI app bootstrapping, logging, CORS
   │   ├─ api/                # FastAPI routers (currently auth_routes)
   │   │   ├─ __init__.py
   │   │   └─ auth_routes.py
   │   ├─ services/           # Business‑logic layer
   │   │   └─ auth_service.py
   │   ├─ db/                 # Database layer (SQLAlchemy)
   │   │   ├─ __init__.py
   │   │   ├─ database.py     # Engine, session factory, `init_db`
   │   │   └─ models.py       # ORM models: User, RefreshToken (plus placeholders)
   │   └─ bot/                # Telegram bot integration (not detailed here)
   ├─ Dockerfile              # Container image definition
   ├─ docker-compose.yml      # Local multi‑service orchestration (app + Postgres)
   ├─ k8s/                    # Kubernetes manifests (base & overlays)
   ├─ .github/workflows/     # CI/CD – builds & pushes Docker image
   ├─ pyproject.toml          # Build system and dev dependencies (pytest)
   ├─ requirements.txt        # Runtime dependencies (FastAPI, SQLAlchemy, JWT, etc.)
   ├─ README.md               # Project description and problem statement
   └─ .env.example            # Example environment variables (SECRET_KEY, DB URL)
```

* **FastAPI** (`main.py`) creates the application, configures logging, CORS (origins from `CORS_ORIGINS` env var), and registers routers.
* **Routers** live under `app/api/`.  Currently only `auth_routes.py` is implemented, exposing endpoints for login, signup, token refresh, and logout.
* **Services** (`app/services/`) contain reusable business logic.  `auth_service.py` handles password hashing (PBKDF2‑SHA256), JWT generation/verification, and refresh‑token persistence.
* **Database** (`app/db/`) uses SQLAlchemy Core/ORM.  `database.py` builds the engine from `DATABASE_URL` (supports both `postgres://` and `postgresql://`).  `init_db()` pulls in all models and creates tables.
* **Models** (`models.py`) define the `User` and `RefreshToken` entities.  Future models such as `Schedule`, `Task`, and `TelegramLinkCode` are referenced in `init_db()` but not yet present.
* **Deployment** is container‑first: Dockerfile builds a slim Python image, installs system deps (`gcc`, `libpq-dev`) for the PostgreSQL driver.
* **Orchestration**: `docker-compose.yml` runs the app together with a PostgreSQL container, wiring environment variables and health‑checks.
* **Kubernetes** manifests (`k8s/`) provide a base deployment and service definitions; the GitHub Actions workflow automatically updates the image tag on push to `main`.

---

## Important Files & Settings

* **`.env.example`** – shows required environment variables:
  * `SECRET_KEY` – JWT signing key (replace in production).
  * `DATABASE_URL` – PostgreSQL connection string.
  * `TELEGRAM_BOT_TOKEN` – optional bot token.
* **`pyproject.toml`** – declares the project name/version and development dependencies (`pytest`).
* **`requirements.txt`** – exact runtime dependencies (FastAPI, SQLModel, etc.).
* **`Dockerfile`** – builds the production image.
* **`docker-compose.yml`** – local dev stack.
* **`build-and-push.yml`** – CI pipeline that builds the Docker image, pushes to GHCR, and updates the K8s deployment manifest.

---

## Extending the Repository

When new routers, services, or models are added, follow the existing pattern:
1. Create a file under the appropriate `app/api/` or `app/services/` directory.
2. Import and register the router in `main.py` (or via `app/api/__init__.py`).
3. Add any new ORM models to `app/db/models.py` and ensure they are imported in `init_db()`.
4. Update the Dockerfile only if additional system packages are required.
5. Adjust `docker-compose.yml` or K8s manifests if new containers or env vars are needed.

---

*Generated with Claude Code*