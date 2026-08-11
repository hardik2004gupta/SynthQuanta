# SynthQuanta Backend

FastAPI backend for SynthQuanta — Synthetic Data–Driven Fine-Tuning and Quantized Model Serving Runtime.

## Local Development

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp ../.env.example .env
# Edit .env — set DATABASE_URL to your local PostgreSQL instance
```

### 3. Start PostgreSQL (Docker)

```bash
docker run -d \
  --name sq-postgres \
  -e POSTGRES_USER=synthquanta \
  -e POSTGRES_PASSWORD=synthquanta \
  -e POSTGRES_DB=synthquanta \
  -p 5432:5432 \
  postgres:16-alpine
```

### 4. Apply migrations

```bash
cd backend
alembic upgrade head
```

### 5. Start the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs
Health check: http://localhost:8000/api/v1/health

### 6. Run tests

```bash
cd backend
pytest tests/ -v
```

## Architecture

```
app/
├── main.py          — FastAPI application, middleware, exception handlers
├── core/
│   ├── config.py    — Pydantic Settings (env-based configuration)
│   └── logging.py   — Structured logging configuration
├── api/v1/
│   ├── router.py    — Top-level v1 router
│   └── routes/      — Thin route handlers (Phase 1: health only)
├── schemas/         — Pydantic request/response models
├── services/        — Application services (artifact_store, future: dataset, training…)
├── db/
│   ├── base.py      — SQLAlchemy declarative base
│   ├── session.py   — Engine, session factory, get_db() dependency
│   └── models/      — Metadata ORM models (Dataset, Experiment, Evaluation, MLModel, Benchmark, Job)
├── data/            — Phase 2: synthetic data engine
├── ml/              — Phase 3: model training, LoRA/QLoRA
├── quantization/    — Phase 4: FP32 → INT8
├── runtime/         — Phase 5: SQRuntime inference engine
└── benchmarking/    — Phase 5: latency/throughput/memory benchmarks
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` / `production` / `test` |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `ARTIFACT_ROOT` | `../artifacts` | Filesystem path for generated artifacts |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `LOG_LEVEL` | `INFO` | Logging level |

## Migration Commands

```bash
# Create a new migration (after changing ORM models)
alembic revision --autogenerate -m "description"

# Apply all pending migrations
alembic upgrade head

# Roll back one migration
alembic downgrade -1
```
