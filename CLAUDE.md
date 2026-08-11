# CLAUDE.md — SynthQuanta Engineering Contract

> This file is the permanent operational engineering contract for every Claude Code session on this project.
> It was distilled from the two authoritative source documents. If this file conflicts with those documents, consult the source documents.

---

## §1 — Project Identity

**Full name:** SynthQuanta: Synthetic Data–Driven Fine-Tuning and Quantized Model Serving Runtime
**Short name:** SynthQuanta
**Category:** Model Engineering / ML Infrastructure / Inference Optimization
**Core application:** Synthetic Sensor-Log Quality Inspector
**Tagline:** *From synthetic data to measurable inference.*

**One-paragraph description:**
SynthQuanta is a local-first model engineering platform that generates deterministic synthetic sensor data with controlled fault injection, fine-tunes a task-specific time-series model using LoRA/QLoRA, evaluates robustness under distribution shift, quantizes the optimized model to INT8, and serves it through a custom inference runtime (SQRuntime) with reproducible latency, throughput, and memory benchmarks. The goal is to demonstrate the complete model-engineering lifecycle — not merely anomaly classification.

**Central engineering question:**
> How effectively can synthetic sensor data be used to fine-tune a task-specific model for sensor-quality inspection, and how do fine-tuning, distribution shift, quantization, and runtime optimization affect both predictive quality and inference efficiency?

**The lifecycle this project covers:**

```
Synthetic Data → Fault Injection → Dataset Validation → Fine-Tuning
→ Evaluation → Distribution Shift → Quantization → Custom Runtime
→ Serving → Benchmarking
```

---

## §2 — Core Engineering Objective

SynthQuanta must demonstrate all six engineering disciplines simultaneously:

```
CONTROL THE DATA   → data engineering
TRAIN THE MODEL    → model engineering
BREAK ASSUMPTIONS  → robustness engineering
COMPRESS THE MODEL → model optimization
BUILD THE RUNTIME  → inference systems
MEASURE INFERENCE  → performance engineering
```

This is not a project about detecting noisy sensor logs. It is a demonstration of what happens when you take a model from controlled synthetic data all the way to an optimized, benchmarked, production-style inference runtime.

---

## §3 — Permanent Engineering Principles

| Principle | Rule |
|---|---|
| **Reproducible** | Same config + seed → same dataset/experiment (within numerical representation tolerance). |
| **Measurable** | Every engineering claim must have an actual metric. No vague assertions. |
| **Modular** | Data, ML, evaluation, quantization, runtime, benchmarking must be independently testable. |
| **Observable** | System exposes progress, metrics, errors, and runtime state. Nothing should be a black box. |
| **Honest** | Never fabricate ML metrics, benchmark numbers, training progress, or runtime measurements. Expose actual failures — never fake success. |

---

## §4 — Architecture Contract

**Pattern:** Modular monolith. One frontend, one FastAPI backend, one PostgreSQL database, one SQRuntime module, one artifact directory.

```
Next.js Frontend (existing, approved)
           ↓ REST/JSON
       FastAPI API
           ↓
   Application Services
           ↓
┌──────────┬──────────┬──────────┐
│ Data     │ ML       │ Runtime  │
│ Engine   │ Pipeline │ SQRuntime│
└──────────┴──────────┴──────────┘
           ↓
  PostgreSQL (metadata) + filesystem (artifacts)
```

**Layering rule:**
```
API Routes → Service Layer → Domain Modules → Infrastructure
```
API routes must not contain ML, data generation, or inference logic.

**Frontend boundary:** visualization, navigation, forms, interaction, charts, local UI state, API requests, loading/error states.
**Backend boundary:** data generation, training, evaluation, quantization, inference, benchmarking, persistence, artifact management.

---

## §5 — FRONTEND IS FROZEN AND APPROVED ⚠️

**The existing `frontend/` directory is the approved product UI. Do not replace it.**

### What exists (confirmed by reconnaissance)

- **Framework:** Next.js 16.3.0, React 19, TypeScript
- **Styling:** Tailwind CSS v4, shadcn/ui, tw-animate-css
- **Icons:** lucide-react
- **Visual system:** dark graphite (`#080b10`), fine grid overlay, Geist/Geist Mono typography, cyan `#56d7f5`, violet `#9b8cff`, green `#61d99a`, amber `#f5b95f`
- **Routing:** client-side via `useState` in single `app/page.tsx`
- **Navigation stages:** OVERVIEW → DATA (Signal Studio) → TRAIN (Adapter Lab) → EVALUATE (Robustness Lab) → OPTIMIZE (Quantization Lab) → RUNTIME (Serve Console)
- **Mock data present:** all metrics, topology nodes, signal waveform, telemetry, timeline, benchmark bars — all hardcoded

### Current mock data to eventually replace with real API data

| Panel | Mock values |
|---|---|
| Metric grid | F1 93.4%, P95 5.8 ms, 184 req/s, 412 MB |
| Pipeline Topology | DS-0042, EXP-0042, EVAL-021, Q-INT8 |
| Signal Integrity | hardcoded SVG path + fault overlay |
| Runtime Telemetry | GPU 68%, Memory 4.2/8 GB, Queue 3 |
| Experiment Timeline | 3 hardcoded steps |
| Benchmark Delta | hardcoded bar widths |

### Rules

Claude Code **MUST:**
- Inspect frontend before any backend implementation
- Preserve the visual system (colors, typography, dark theme, grid)
- Preserve the layout and panel structure
- Design API contracts to match what the existing UI expects
- Replace mock data with real data during Phase 6

Claude Code **MUST NOT:**
- Replace, regenerate, or redesign the frontend
- Create a generic dashboard in place of the existing UI
- Add an unnecessary sidebar or nav restructure
- Replace the visual system
- Rewrite working components without a documented backend-integration reason

Small, targeted frontend changes are allowed only when required for real API integration (e.g., adding an API client module, swapping a hardcoded value for a fetched value).

---

## §6 — MVP Scope

### Data
- Deterministic signal generation (sinusoidal, multi-frequency, composite, trend)
- Configurable: duration, sampling rate, seed
- Fault types: `NOISE`, `DRIFT`, `DROPOUT`, `CLIPPING`, `TIMESTAMP_GAP`, `SAMPLING_JITTER`
- Composite (multi-fault) support
- Dataset validation (structural, temporal, statistical, ground-truth)
- Windowing + dataset splitting (TRAIN / VALIDATION / IID TEST / SHIFT TEST)

### ML
- Compact time-series transformer (local training scale — not GPU-scale)
- Classification: NORMAL + 6 fault classes
- LoRA fine-tuning
- QLoRA fine-tuning
- Checkpointing + experiment metadata

### Evaluation
- Precision, Recall, F1 (per-class + aggregate)
- Confusion matrix
- False alarm rate
- Fault localization + Interval IoU (mean, median, IoU@0.5)
- Distribution-shift evaluation (5 scenarios — see §10)

### Optimization
- FP32 baseline artifact
- INT8 quantized artifact (separate, FP32 immutable)
- Quality comparison (F1 delta), size/memory/latency comparison

### Runtime (SQRuntime)
- Model loading + warmup
- Single + batch inference
- Runtime status (offline / loading / ready / error)
- Runtime metrics (request count, latency, throughput, memory)

### Benchmarking
- Cold start + warm inference distinction
- Warmup excluded from stats
- Latency: P50, P90, P95, P99, min, max, mean
- Throughput (req/s)
- Memory (baseline, peak, model)
- Batch sizes: 1, 8, 16, 32

### Application
- FastAPI backend at `/api/v1`
- PostgreSQL metadata store
- Local artifact storage
- Docker Compose (frontend + backend + postgres)
- Automated tests (unit + integration + E2E)
- API documentation at `/docs`

---

## §7 — Explicitly Out of Scope for MVP

Do NOT implement these until the MVP is complete:

- Adaptive fault generation / active learning
- Model calibration / uncertainty estimation
- Dynamic runtime batching / multi-model caching / model hot-swapping
- Benchmark history / experiment leaderboard / Pareto model selection
- Runtime deployment profiles
- Automatic model recommendation
- Kubernetes, distributed training, multi-GPU serving
- Authentication, billing, teams, RBAC
- LLM chat, RAG, vector database
- FP16 as an intermediate quantization step (MVP is FP32 → INT8 only)

**Rule:** One excellent model + one excellent runtime > many incomplete features.

---

## §8 — Data Architecture Contract

```
Configuration (YAML + seed)
         ↓
Clean Signal (sinusoidal / composite / trend)
         ↓
Fault Injection (ordered, deterministic, each fault has common interface)
         ↓
Ground Truth (fault_id, type, start_index, end_index, severity, parameters)
         ↓
Validation (structural + temporal + statistical + label integrity)
         ↓
Windowing (fixed-length windows with labels + fault intervals)
         ↓
Dataset Artifact (NPZ tensors + JSON metadata)
```

**Fault interface contract:** `Fault.apply(timestamps, values, rng)` → `(modified_values, modified_timestamps, fault_intervals, metadata)`

**Determinism rule:** Same config + same seed + same software version → identical dataset. Store seed, generator version, and full configuration with every dataset artifact.

**Dropout rule:** Never silently convert dropout to zeros without preserving ground-truth metadata.

**Artifact format:** `artifacts/datasets/DS-XXXX/data.npz` + `metadata.json`

**Dataset splits:** 70% train / 10% validation / 10% IID test / 10% shift test. Never random row-split only.

---

## §9 — ML Architecture Contract

**Architecture:** Compact time-series model — normalization → 1D temporal embedding → compact transformer encoder → pooling → classification head → fault class + probabilities.

**Do not** make the architecture unnecessarily large. This demonstrates model engineering, not GPU-scale training.

**Training pipeline:** Dataset → DataLoader → Model → Adapter → Training loop → Validation → Checkpoint → Experiment result

**LoRA/QLoRA:** Store base model, rank, alpha, dropout, trainable parameter count with every experiment.

**QLoRA failure rule:** If hardware cannot support the requested QLoRA config, the system must detect → report → fail clearly. Never silently substitute full fine-tuning and claim QLoRA ran.

**Training start rule:** Always begin with a tiny config (small model, few epochs, small dataset) to verify the complete pipeline before scaling.

**Experiment tracking:** Every run gets a unique `EXP-XXXX` ID. Store dataset_id, architecture, method, hyperparameters, seed, duration, checkpoint path, metrics, hardware, and software versions.

**Checkpoint path:** `artifacts/models/MODEL-XXXX/base/` + `adapter/` + `metadata.json`

---

## §10 — Evaluation Contract

Evaluation must be a separate module from training. Input: model + dataset. Output: structured `EvaluationResult`.

**Required metrics:** Precision, Recall, F1 (per-class for all 7 classes + aggregate), Confusion matrix, False alarm rate, Interval IoU (mean, median, IoU@0.5)

**Distribution-shift scenarios (all 5 required for MVP):**

| Scenario | Train | Test |
|---|---|---|
| Noise shift | σ = 0.10 | σ = 0.30 |
| Amplitude shift | amplitude = 1.0 | amplitude = 1.5 |
| Frequency shift | 10 Hz | 15 Hz |
| Severity shift | mild/moderate | severe |
| Compound fault shift | isolated faults | multiple simultaneous |

**Shift output:** IID F1, Shifted F1, absolute degradation, relative degradation — per scenario.

**Hard rules:**
- Evaluation uses actual model outputs — no hardcoded metrics
- Training and test sets must use different generator configs (no data leakage)

---

## §11 — Quantization Contract

**MVP path:** FP32 → INT8 (only these two for MVP)

**Rules:**
- FP32 artifact is immutable after creation; INT8 is a separate artifact
- Both must produce valid inference results after creation
- Quality comparison uses actual inference on test dataset
- Size, memory, latency comparisons use actual measurements
- Quantization failure → job FAILED + explicit diagnostic; never silently fall back to FP32

**Validation:** Load quantized model → run test dataset → compute F1 → record (original_F1, quantized_F1, F1_delta, size_reduction, latency_ratio, memory_ratio)

---

## §12 — SQRuntime Contract

**Name meaning:** Synthetic Quality Runtime

**Architecture (strict layering):**
```
API Route
    ↓
InferenceService
    ↓
SQRuntime (ModelLoader + InferenceEngine + Preprocessor + Postprocessor + RuntimeMetrics)
    ↓
Model artifact
```

**Never** put inference logic directly inside the FastAPI route.

**Required components:**
- `ModelLoader`: load_model(path), validate artifact, warmup, unload
- `Preprocessor`: timestamps + values → normalized model tensor (must match training preprocessing exactly)
- `InferenceEngine`: predict(window), predict_batch(windows)
- `Postprocessor`: raw logits → structured result (quality, fault_type, confidence, fault_interval)
- `RuntimeMetrics`: request_count, error_count, latency, throughput, memory, queue_depth

**State machine:** `offline → loading → ready → error`

**MVP model support:** one active model at a time. Loaded at startup/activation. Warmup before serving.

**Testability rule:** SQRuntime must be testable without HTTP — instantiate and call directly in tests.

---

## §13 — Benchmarking Contract

**Primary UI latency metric:** P95

**Required measurements:** warmup (excluded from stats), cold start, P50, P90, P95, P99, min, max, mean, throughput (req/s), baseline memory, peak memory, model memory

**Batch sizes to test:** 1, 8, 16, 32

**Every benchmark record must contain:**
model, precision, device, batch_size, iterations, warmup_count, runtime_version, python_version, hardware (CPU/GPU/RAM), timestamp, seed

**Hard rule:** Benchmark values must originate from actual execution. Never hardcode benchmark results. Raw per-request latencies must be persisted; derived statistics are computed automatically.

---

## §14 — Persistence Contract

**PostgreSQL — metadata only:**

| Entity | Key fields |
|---|---|
| Dataset | id, name, seed, configuration, artifact_path, sample_count, created_at, status |
| Experiment | id, dataset_id, model_id, method, configuration, metrics, artifact_path, status, created_at |
| Evaluation | id, experiment_id, dataset_id, evaluation_type, metrics, results, created_at |
| Model | id, name, version, base_model, precision, quantization, artifact_path, metrics, status, created_at |
| Benchmark | id, model_id, device, batch_size, iterations, warmup, latency_metrics, throughput, memory, created_at |
| Job | id, type, status, artifact_id, error, created_at, updated_at |

**Filesystem — large/immutable artifacts:**
```
artifacts/
├── datasets/DS-XXXX/
├── models/MODEL-XXXX/
├── experiments/EXP-XXXX/
└── benchmarks/BENCH-XXXX/
```

**Hard rules:**
- Never store model weights, generated datasets, or benchmark dumps in PostgreSQL
- Never commit large artifacts to Git
- Dataset and model artifacts are immutable after registration; configuration changes create new IDs
- Lineage must be maintained: Dataset → Experiment → Checkpoint → Quantized Model → Benchmark → Runtime

---

## §15 — API Contract

**Base namespace:** `/api/v1`

**API routes (thin — delegate to services):**

```
GET  /api/v1/health
GET  /api/v1/runtime/status

POST /api/v1/datasets/generate
GET  /api/v1/datasets
GET  /api/v1/datasets/{dataset_id}

POST /api/v1/training/run
GET  /api/v1/training/{experiment_id}

POST /api/v1/evaluation/run
GET  /api/v1/evaluation/{evaluation_id}

POST /api/v1/quantization/run
GET  /api/v1/models/{model_id}/quantization

GET  /api/v1/models
GET  /api/v1/models/{model_id}

POST /api/v1/inference
POST /api/v1/inference/batch

POST /api/v1/benchmarks/run
GET  /api/v1/benchmarks
GET  /api/v1/benchmarks/{benchmark_id}
```

**Rules:**
- All payloads use Pydantic schemas — never return raw dicts from routes
- API contracts are designed around the existing frontend requirements
- Errors return structured JSON: `{"error": {"code": "...", "message": "..."}}`
- HTTP codes: 400 invalid input, 404 not found, 409 state conflict, 422 validation, 500 runtime failure, 503 model unavailable

---

## §16 — Job Execution Contract

Long-running operations (training, quantization, benchmarking) use a lightweight background job model:

```
QUEUED → RUNNING → COMPLETED
                 ↘ FAILED
                 ↘ CANCELLED
```

POST endpoint returns `job_id`. Frontend polls status endpoint until terminal state.

Do not introduce Celery, Redis, or distributed task infrastructure unless the MVP genuinely cannot function without them.

---

## §17 — Testing Contract

Three required layers:

**Unit tests** — test in isolation:
- Signal generation (all types), deterministic seed
- All 6 fault types independently
- Timestamp gap detection, jitter statistics
- Interval IoU calculation
- Precision/Recall/F1 calculation
- Percentile (P50/P95/P99) calculation
- Runtime preprocessing, output schema validation

**Integration tests:**
- Dataset generation → artifact creation → metadata persistence
- Training → checkpoint → experiment record
- Model → SQRuntime → inference result
- API route → service → domain module

**End-to-end test (critical):**
```
Generate dataset (tiny config)
    → Train small model (1-2 epochs)
    → Evaluate
    → Quantize
    → Load SQRuntime
    → Run inference
    → Run benchmark
```
Use minimal dataset/model so E2E runs in CI in reasonable time.

**Reproducibility test (required):**
```python
assert generate(seed=42) == generate(seed=42)  # within numerical tolerance
```

---

## §18 — Security Contract

Even for local-first MVP, enforce:
- Validate all uploaded/received data (shape, range, type)
- Restrict file upload sizes
- Validate tensor shapes before model input
- Validate numerical ranges (no NaN/Inf without explicit handling)
- Never execute uploaded code or model artifacts as arbitrary Python
- Restrict model artifact paths — no arbitrary filesystem access via API parameters
- Sanitize all log output (no secrets, no raw user data)
- Secrets in environment variables only — never in code

---

## §19 — Dependency Philosophy

Use the smallest reasonable set. Add a dependency only when it solves a real requirement that existing deps cannot address.

**Backend core:**
```
FastAPI, Pydantic, SQLAlchemy, asyncpg/psycopg2
PyTorch, NumPy, SciPy, scikit-learn
PEFT, Transformers
```

**Frontend:** Preserve the existing stack. Do not replace dependencies without documented reason.
Expected: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui, lucide-react

Add quantization/runtime dependencies (e.g., bitsandbytes, ONNX Runtime) only when implementing those phases.

---

## §20 — Repository Architecture

```
SynthQuanta/
├── frontend/                    ← existing, approved, DO NOT replace
├── backend/
│   ├── app/
│   │   ├── main.py, config.py, dependencies.py
│   │   ├── api/routes/          ← health, datasets, training, evaluation,
│   │   │                           models, inference, quantization, benchmarks
│   │   ├── schemas/             ← Pydantic models for all entities
│   │   ├── services/            ← dataset, training, evaluation, model,
│   │   │                           inference, quantization, benchmark services
│   │   ├── data/                ← generator, signals, faults, validator,
│   │   │                           windowing, dataset
│   │   ├── ml/                  ← model, train, evaluate, metrics,
│   │   │                           localization, distribution_shift, adapters
│   │   ├── quantization/        ← quantize, validate
│   │   ├── runtime/             ← engine, loader, preprocessing,
│   │   │                           postprocessing, metrics
│   │   ├── benchmarking/        ← runner, latency, throughput, memory
│   │   ├── db/                  ← database, models, repositories
│   │   └── utils/
│   ├── tests/unit/, tests/integration/, tests/e2e/
│   ├── pyproject.toml
│   └── Dockerfile
├── configs/data/, configs/training/, configs/evaluation/,
│           configs/quantization/, configs/benchmarks/
├── artifacts/datasets/, artifacts/models/,
│            artifacts/experiments/, artifacts/benchmarks/
├── scripts/
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## §21 — Implementation Order

Each phase must produce a **working increment** — not a collection of disconnected files. Do not begin a phase until the previous checkpoint passes.

```
PHASE 0  Engineering Contract (current — complete before any code)
    ↓
PHASE 1  Repo Reconnaissance + Backend Foundation
         Checkpoint: GET /api/v1/health returns {"status": "ok"}
    ↓
PHASE 2  Synthetic Data Engine + Dataset API
         Checkpoint: generate(seed=42) == generate(seed=42); frontend Generate button creates real dataset
    ↓
PHASE 3  ML Training + LoRA/QLoRA
         Checkpoint: real model trains end-to-end on generated data; method is configurable and recorded
    ↓
PHASE 4  Evaluation + Distribution Shift + Quantization
         Checkpoint: all 5 shift scenarios produce F1 delta; FP32 and INT8 artifacts both produce valid inference
    ↓
PHASE 5  SQRuntime + Inference API + Benchmark Engine
         Checkpoint: user submits sensor window from UI, sees real prediction + P95 latency in Benchmark Arena
    ↓
PHASE 6  Full Frontend Integration + Docker + Tests + E2E
         Checkpoint: docker compose up → complete lifecycle works from clean environment
```

---

## §22 — Definition of Done

The MVP is complete **only when all of the following are true:**

```
DATA
✓ deterministic generation (seed reproducibility test passes)
✓ all 6 fault types with ground truth metadata
✓ dataset validation
✓ persisted dataset artifacts

ML
✓ model trains on generated data
✓ LoRA training configurable and recorded
✓ QLoRA training configurable and recorded (or explicit failure with diagnostic)
✓ checkpoints persisted with full experiment metadata

EVALUATION
✓ precision, recall, F1 (per-class + aggregate)
✓ confusion matrix
✓ false alarm rate
✓ interval IoU
✓ all 5 distribution-shift scenarios produce structured results

OPTIMIZATION
✓ FP32 artifact (immutable)
✓ INT8 artifact (separate)
✓ quality comparison (F1 delta from actual inference)
✓ size/memory/latency comparison from actual measurements

RUNTIME
✓ model loads and warms up
✓ single inference returns structured result
✓ batch inference works
✓ runtime status endpoint reflects actual state

BENCHMARK
✓ P50, P90, P95, P99, throughput, memory — all from actual execution
✓ batch size comparison (1, 8, 16, 32)
✓ benchmark record includes full hardware/software context

PRODUCT
✓ existing frontend preserved (visual system intact)
✓ all major mock data replaced with real API responses
✓ API documented at /docs
✓ docker compose up starts complete MVP
✓ unit + integration + E2E tests pass
✓ complete lifecycle executable from clean environment
```

---

## §23 — NON-NEGOTIABLE RULES

> Violations of these rules are implementation defects, not judgment calls.

1. **Never replace the existing frontend.** The `frontend/` directory is approved. Do not regenerate it.
2. **Never redesign the product into a generic dashboard.** The existing visual system is intentional.
3. **Never fabricate ML metrics.** All metrics must come from actual model evaluation.
4. **Never fabricate benchmark results.** All numbers must come from actual execution.
5. **Never fabricate training progress.** Loss curves and epochs must reflect real training.
6. **Never hide failures.** Quantization failure, training failure, runtime error → explicit job FAILED + diagnostic.
7. **Never silently substitute a requested ML method.** If QLoRA cannot run → fail with diagnostic, do not silently use LoRA or full fine-tuning.
8. **Never store large artifacts in PostgreSQL.** Model weights, datasets, benchmark dumps → filesystem only.
9. **Never put inference logic inside API routes.** All inference goes through SQRuntime.
10. **Never introduce unnecessary infrastructure.** No Celery/Redis/Kubernetes unless the MVP cannot function without them.
11. **Never break reproducibility.** Every experiment must be reproducible from its stored configuration + seed.
12. **Never commit secrets.** All secrets go in environment variables. `.env` is gitignored.
13. **Never commit large generated artifacts.** `artifacts/` contents are gitignored.
14. **Never add advanced/stretch features before the MVP works end-to-end.**
15. **Never claim something works without actually running it.** Test before reporting completion.

---

## §24 — Claude Code Operating Rules

Every session must follow this workflow:

```
1. Read CLAUDE.md
2. Read relevant sections of source documents if needed
3. Inspect existing implementation state
4. Understand architecture of affected components
5. Identify all files that will change
6. Plan the minimal change that achieves the goal
7. Implement
8. Run relevant tests
9. Verify the application behaves correctly
10. Report what actually works and what does not
```

**Before modifying an existing architectural component:** Understand → Verify → Modify. Do not blindly rewrite.

**Before adding a dependency:** Ask "Can the existing dependencies solve this?" If yes, use them. If no, add the smallest reasonable dependency and document why in the commit.

**On incremental vs. large rewrites:** Prefer targeted edits. A large rewrite is justified only when the existing structure fundamentally prevents the required capability.

---

## §25 — Source of Truth Hierarchy

For project requirements:

```
1. SynthQuanta — Master Architecture & Implementation Specification.md  (primary)
2. SynthQuanta Master Documentation.md                                  (secondary)
3. CLAUDE.md                                                            (operational contract)
4. Existing implementation                                              (current state)
```

If implementation conflicts with documented architecture → documentation wins.
If CLAUDE.md appears inconsistent with source documents → consult source documents.
Do not invent requirements not present in any of these sources.

**Known conflict between source documents:**
- Doc 1 (§31) describes FP32 → FP16 → INT8 as the quantization path.
- Doc 2 (§45) specifies MVP as FP32 → INT8 only.
- **Resolution:** Doc 2 is the implementation specification. MVP quantization is FP32 → INT8 only. FP16 is a post-MVP extension.

---

## §26 — Final Product Narrative

The completed application demonstrates this user journey:

```
OPEN SYNTHQUANTA
       ↓ (existing approved UI loads)
GENERATE SENSOR DATA
       ↓ (real deterministic signal with injected faults)
INSPECT CORRUPTED WAVEFORM
       ↓ (signal integrity panel shows real data)
START QLORA EXPERIMENT
       ↓ (real training, real loss curves)
SHOW MODEL F1
       ↓ (actual evaluation result)
RUN DISTRIBUTION SHIFT
       ↓ (actual F1 degradation across 5 scenarios)
QUANTIZE TO INT8
       ↓ (real size reduction, real F1 delta)
DEPLOY TO SQRUNTIME
       ↓ (actual model load + warmup)
RUN LIVE INFERENCE
       ↓ (real prediction overlaid on waveform)
RUN BENCHMARK
       ↓ (actual P95, throughput, memory)
```

The viewer should leave understanding that SynthQuanta demonstrates:
**data engineering + model engineering + robustness engineering + model optimization + inference systems + performance engineering** — simultaneously, reproducibly, and honestly.
