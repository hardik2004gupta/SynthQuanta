# SynthQuanta — Master Architecture & Implementation Specification

## Synthetic Data–Driven Fine-Tuning and Quantized Model Serving Runtime

> **From synthetic data to measurable inference.**

---

# 0. Document Purpose

This document is the **single source of truth for implementing SynthQuanta**.

It defines:

- product scope
- MVP boundaries
- system architecture
- repository architecture
- frontend/backend boundaries
- data-generation pipeline
- synthetic fault engine
- ML architecture
- LoRA/QLoRA training
- evaluation
- distribution-shift testing
- quantization
- custom inference runtime
- benchmarking
- persistence
- API contracts
- frontend integration
- testing
- Docker architecture
- implementation phases
- acceptance criteria

The existing `frontend/` directory is considered the **approved frontend implementation**.

The backend must be built **around the existing frontend**, not by replacing or redesigning it.

---

# 1. Product Definition

## Product

**SynthQuanta**

## Full name

**SynthQuanta: Synthetic Data–Driven Fine-Tuning and Quantized Model Serving Runtime**

## Core application

**Synthetic Sensor-Log Quality Inspector**

## Category

**Model Engineering / ML Infrastructure / Inference Optimization**

## Tagline

> **From synthetic data to measurable inference.**

---

# 2. Product Positioning

SynthQuanta demonstrates an end-to-end model-engineering lifecycle:

```text
Synthetic Data
      ↓
Fault Injection
      ↓
Dataset Validation
      ↓
Fine-Tuning
      ↓
Evaluation
      ↓
Distribution Shift
      ↓
Quantization
      ↓
Custom Runtime
      ↓
Serving
      ↓
Benchmarking
```

The central engineering question is:

> **Can a task-specific model trained on controlled synthetic sensor failures remain useful under distribution shift while becoming significantly more efficient through quantization and optimized serving?**

SynthQuanta therefore evaluates two dimensions simultaneously:

```text
MODEL QUALITY
+
SYSTEM PERFORMANCE
```

---

# 3. MVP Objective

The MVP is successful when a user can complete this entire workflow:

```text
Generate synthetic sensor dataset
          ↓
Inspect generated faults
          ↓
Train LoRA / QLoRA model
          ↓
Evaluate model
          ↓
Measure F1 / Precision / Recall
          ↓
Measure fault localization
          ↓
Run distribution-shift evaluation
          ↓
Quantize model
          ↓
Compare FP32 vs INT8
          ↓
Load optimized model
          ↓
Run inference through SQRuntime
          ↓
Benchmark latency
          ↓
Benchmark throughput
          ↓
Measure memory
          ↓
Visualize everything in frontend
```

---

# 4. Explicit MVP Scope

## Included

### Data

- deterministic signal generation
- configurable sampling rate
- configurable duration
- random seed
- clean signals
- noise
- drift
- dropout
- clipping
- timestamp gaps
- sampling jitter
- combined faults
- dataset validation

### ML

- time-series classification
- normal/fault classification
- fault type classification
- LoRA
- QLoRA
- checkpointing
- experiment metadata

### Evaluation

- precision
- recall
- F1
- confusion matrix
- fault localization
- interval IoU
- false alarm rate
- distribution-shift testing

### Optimization

- FP32 baseline
- INT8 quantization
- model-size comparison
- memory comparison
- latency comparison

### Runtime

- model loading
- preprocessing
- inference
- postprocessing
- health status
- runtime metrics

### Benchmarking

- warmup
- P50
- P95
- P99
- throughput
- memory
- batch-size comparison

### Application

- FastAPI backend
- PostgreSQL metadata store
- local artifact storage
- Next.js frontend
- Docker Compose
- automated tests

---

# 5. Explicitly Out of Scope for MVP

Do NOT implement initially:

- Kubernetes
- distributed training
- multi-node inference
- multi-GPU serving
- cloud-specific infrastructure
- authentication
- billing
- teams
- subscriptions
- complex RBAC
- active learning
- automatic hyperparameter optimization
- model marketplace
- arbitrary model support
- LLM chat
- RAG
- vector database
- autonomous agents
- complex distributed task queues
- production-grade autoscaling

These may become future extensions.

---

# 6. High-Level Architecture

```text
                         ┌───────────────────────┐
                         │    SYNTHQUANTA UI     │
                         │       Next.js         │
                         │  Existing v0 frontend │
                         └───────────┬───────────┘
                                     │
                                  REST/JSON
                                     │
                         ┌───────────▼───────────┐
                         │       FastAPI         │
                         │      API Layer        │
                         └───────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
      ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
      │ Data Engine   │      │ ML Pipeline   │      │ SQRuntime     │
      │               │      │               │      │               │
      │ Generation    │      │ Training      │      │ Loading       │
      │ Faults        │      │ Evaluation    │      │ Inference     │
      │ Validation    │      │ Shift Testing │      │ Metrics       │
      └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │     Artifact Store    │
                         │                       │
                         │ datasets/             │
                         │ models/               │
                         │ benchmarks/           │
                         └────────────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │      PostgreSQL       │
                         │       Metadata        │
                         └───────────────────────┘
```

---

# 7. Core Architectural Principle

The system is a **modular monolith**.

Do not create microservices for the MVP.

There should be:

```text
1 frontend application
1 FastAPI backend
1 PostgreSQL database
1 runtime module
1 artifact directory
```

The internal backend should be modular enough that components can later be extracted if required.

---

# 8. Architectural Layers

The backend follows these layers:

```text
API
 ↓
Application Services
 ↓
Domain Modules
 ↓
Infrastructure
```

More specifically:

```text
API Routes
    ↓
Service Layer
    ↓
Data / ML / Runtime / Benchmark Modules
    ↓
Artifact + Database Infrastructure
```

API routes must not contain ML implementation logic.

---

# 9. Repository Architecture

Recommended repository:

```text
synthquanta/
│
├── frontend/
│
├── backend/
│   │
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   │
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── health.py
│   │   │   │   ├── datasets.py
│   │   │   │   ├── training.py
│   │   │   │   ├── evaluation.py
│   │   │   │   ├── models.py
│   │   │   │   ├── inference.py
│   │   │   │   ├── quantization.py
│   │   │   │   └── benchmarks.py
│   │   │   └── router.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── datasets.py
│   │   │   ├── training.py
│   │   │   ├── evaluation.py
│   │   │   ├── models.py
│   │   │   ├── inference.py
│   │   │   └── benchmarks.py
│   │   │
│   │   ├── services/
│   │   │   ├── dataset_service.py
│   │   │   ├── training_service.py
│   │   │   ├── evaluation_service.py
│   │   │   ├── model_service.py
│   │   │   ├── inference_service.py
│   │   │   ├── quantization_service.py
│   │   │   └── benchmark_service.py
│   │   │
│   │   ├── data/
│   │   │   ├── generator.py
│   │   │   ├── signals.py
│   │   │   ├── faults.py
│   │   │   ├── validator.py
│   │   │   ├── windowing.py
│   │   │   └── dataset.py
│   │   │
│   │   ├── ml/
│   │   │   ├── model.py
│   │   │   ├── config.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   ├── metrics.py
│   │   │   ├── localization.py
│   │   │   ├── distribution_shift.py
│   │   │   └── adapters.py
│   │   │
│   │   ├── quantization/
│   │   │   ├── quantize.py
│   │   │   └── validate.py
│   │   │
│   │   ├── runtime/
│   │   │   ├── engine.py
│   │   │   ├── loader.py
│   │   │   ├── preprocessing.py
│   │   │   ├── postprocessing.py
│   │   │   └── metrics.py
│   │   │
│   │   ├── benchmarking/
│   │   │   ├── runner.py
│   │   │   ├── latency.py
│   │   │   ├── throughput.py
│   │   │   └── memory.py
│   │   │
│   │   ├── db/
│   │   │   ├── database.py
│   │   │   ├── models.py
│   │   │   └── repositories/
│   │   │
│   │   └── utils/
│   │
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   │
│   ├── pyproject.toml
│   └── Dockerfile
│
├── configs/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── quantization/
│   └── benchmarks/
│
├── artifacts/
│   ├── datasets/
│   ├── models/
│   ├── experiments/
│   └── benchmarks/
│
├── scripts/
│
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

# 10. Frontend Rule

The existing:

```text
frontend/
```

directory is the approved UI.

The backend implementation must:

1. inspect the existing frontend
2. identify pages/components
3. identify mock data
4. identify expected state
5. preserve the visual design
6. create API contracts matching the frontend
7. replace mock data with backend data

Do NOT regenerate the frontend.

Do NOT replace the existing layout with a generic dashboard.

Do NOT introduce a sidebar.

Do NOT redesign the UI unless a backend integration genuinely requires a small change.

---

# 11. Frontend ↔ Backend Boundary

Frontend handles:

```text
visualization
navigation
forms
interaction
charts
local UI state
API requests
loading states
error states
```

Backend handles:

```text
data generation
training
evaluation
quantization
inference
benchmarking
persistence
artifact management
```

---

# 12. Synthetic Data Engine

The synthetic data engine is the first major domain component.

Its responsibility:

```text
configuration
 ↓
clean signal
 ↓
fault injection
 ↓
ground truth
 ↓
validation
 ↓
dataset artifact
```

---

# 13. Signal Generation

MVP supports:

- sinusoidal signal
- multi-frequency composite
- trend
- periodic signal

Minimum required implementation:

```python
generate_signal(
    duration,
    sampling_rate,
    signal_type,
    seed
)
```

Output:

```text
timestamps
values
metadata
```

---

# 14. Fault Model

Every fault should have a common interface.

Conceptually:

```python
Fault.apply(
    timestamps,
    values,
    rng
)
```

Returns:

```text
modified values
modified timestamps
fault intervals
metadata
```

---

# 15. Fault Types

Required:

```text
NORMAL
NOISE
DRIFT
DROPOUT
CLIPPING
TIMESTAMP_GAP
SAMPLING_JITTER
```

---

# 16. Fault Metadata

Every fault event must contain:

```json
{
  "fault_id": "fault_0012",
  "type": "clipping",
  "start_index": 412,
  "end_index": 459,
  "severity": 0.84,
  "parameters": {}
}
```

This metadata is essential for localization evaluation.

---

# 17. Noise

Noise modifies signal values without changing timestamps.

Config:

```yaml
noise:
  enabled: true
  std: 0.1
```

Support deterministic generation through the dataset seed.

---

# 18. Drift

Drift gradually shifts the baseline.

Parameters:

```yaml
drift:
  enabled: true
  magnitude: 0.2
  direction: positive
```

---

# 19. Dropout

Dropout creates missing signal intervals.

Representation should be explicit.

Do not silently convert dropout to ordinary zeros without preserving the ground-truth metadata.

---

# 20. Clipping

Apply configurable upper/lower bounds.

Example:

```yaml
clipping:
  enabled: true
  upper: 1.0
  lower: -1.0
```

---

# 21. Timestamp Gaps

Timestamp gaps modify temporal continuity.

The validator must detect:

```text
expected interval != actual interval
```

---

# 22. Sampling Jitter

Timestamp intervals receive controlled perturbation.

Example:

```text
expected:
10ms
10ms
10ms

actual:
10ms
11ms
9ms
12ms
```

---

# 23. Composite Faults

The generator must support multiple simultaneous fault transformations.

Example:

```yaml
faults:
  - noise
  - drift
  - clipping
```

The order of transformations must be deterministic and recorded.

---

# 24. Dataset Configuration

Every dataset must be generated from configuration.

Example:

```yaml
dataset:
  name: sensor-quality-demo
  seed: 42
  samples: 5000

signal:
  type: composite
  duration_seconds: 60
  sampling_rate_hz: 100

faults:
  noise:
    enabled: true
    std: 0.1

  drift:
    enabled: true
    magnitude: 0.2

  dropout:
    enabled: true
    probability: 0.01

  clipping:
    enabled: true
    lower: -1
    upper: 1

  timestamp_gap:
    enabled: true

  sampling_jitter:
    enabled: true
```

---

# 25. Reproducibility Requirement

The following must generate identical data:

```text
same configuration
+
same seed
+
same software version
```

At minimum, store:

```text
seed
generator version
configuration
```

with every dataset.

---

# 26. Dataset Format

Use a compact artifact format rather than giant CSV files.

Recommended:

```text
NPZ / Parquet
```

For MVP, NPZ is acceptable for numerical tensors.

Metadata should be JSON.

Example:

```text
artifacts/datasets/ds_0001/
    data.npz
    metadata.json
```

---

# 27. Dataset Validation

Validation checks:

### Structural

- timestamps present
- values present
- matching lengths
- valid numeric values

### Temporal

- timestamps monotonic where expected
- gap detection
- jitter statistics

### Statistical

- mean
- std
- min
- max

### Ground truth

- fault intervals valid
- start < end
- severity ∈ [0,1]

---

# 28. Windowing

Convert continuous logs into fixed-length windows.

Example:

```text
sampling rate = 100Hz
window = 128 samples
```

Each training example:

```text
128 sensor observations
```

with labels.

Store:

```text
window_id
values
timestamps
fault labels
fault intervals
```

---

# 29. Dataset Splitting

Never perform a naive random row split.

Create:

```text
TRAIN
VALIDATION
IID TEST
SHIFT TEST
```

Recommended:

```text
70% train
10% validation
10% IID test
10% shift test
```

---

# 30. ML Model

Use a compact time-series model suitable for local training and inference.

Recommended MVP architecture:

```text
Input
  ↓
Normalization
  ↓
1D Temporal Embedding
  ↓
Compact Transformer / Temporal Encoder
  ↓
Pooling
  ↓
Classification Head
  ↓
Fault Class
```

Do not make the architecture unnecessarily large.

The objective is to demonstrate **model engineering**, not GPU-scale training.

---

# 31. Model Outputs

Minimum:

```text
fault_class
class_probabilities
confidence
```

Recommended MVP extension:

```text
fault_interval
```

Optional:

```text
severity
```

---

# 32. Classification Classes

```text
NORMAL
NOISE
DRIFT
DROPOUT
CLIPPING
TIMESTAMP_GAP
SAMPLING_JITTER
```

---

# 33. Training

Training pipeline:

```text
Dataset
 ↓
DataLoader
 ↓
Model
 ↓
Adapter
 ↓
Training
 ↓
Validation
 ↓
Checkpoint
 ↓
Experiment Result
```

---

# 34. LoRA

The model must support parameter-efficient fine-tuning where the selected architecture permits it.

Store:

```text
base model
LoRA rank
LoRA alpha
LoRA dropout
trainable parameters
```

---

# 35. QLoRA

QLoRA should be supported as an alternative training configuration.

Configuration:

```yaml
training:
  method: qlora

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
```

If hardware does not support the desired QLoRA configuration, the system must fail gracefully with a useful diagnostic rather than silently switching methods.

---

# 36. Training Configuration

Example:

```yaml
experiment:
  name: qlora-baseline
  seed: 42

model:
  architecture: sensor-transformer-small

training:
  method: qlora
  epochs: 5
  batch_size: 16
  learning_rate: 0.0002
  weight_decay: 0.01

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
```

---

# 37. Experiment Tracking

Each training run receives:

```text
experiment_id
```

Example:

```text
EXP-0042
```

Store:

```text
dataset_id
model architecture
training method
hyperparameters
seed
duration
checkpoint path
metrics
hardware
software versions
```

---

# 38. Checkpoints

Store model checkpoints under:

```text
artifacts/models/
```

Example:

```text
artifacts/models/
    model_0042/
        base/
        adapter/
        metadata.json
```

Do not store huge generated model artifacts inside Git.

---

# 39. Evaluation Engine

Evaluation must be separate from training.

Input:

```text
model
+
dataset
```

Output:

```text
EvaluationResult
```

---

# 40. Required Metrics

## Precision

Measure false-positive behavior.

## Recall

Measure fault detection.

## F1

Primary aggregate classification metric.

## Confusion matrix

Required for class-level analysis.

## False alarm rate

Required for normal-window reliability.

---

# 41. Fault Localization

For models producing fault intervals:

```text
predicted interval
vs
ground-truth interval
```

Calculate:

```text
IoU
```

Store:

```text
mean IoU
median IoU
IoU@0.5
```

---

# 42. Distribution Shift

Distribution shift is a first-class evaluation module.

The test data must intentionally differ from training conditions.

---

# 43. Shift Scenarios

MVP:

### Noise shift

```text
train σ = 0.10
test σ = 0.30
```

### Amplitude shift

```text
train amplitude = 1.0
test amplitude = 1.5
```

### Frequency shift

```text
train = 10Hz
test = 15Hz
```

### Severity shift

Train on moderate faults, test on more severe faults.

### Compound fault shift

Train primarily on isolated faults, test on multiple simultaneous faults.

---

# 44. Shift Metrics

For every scenario:

```text
IID F1
Shifted F1
Absolute degradation
Relative degradation
```

Store these in a structured result.

---

# 45. Quantization

MVP quantization:

```text
FP32 → INT8
```

The implementation must preserve a separate artifact.

Do not overwrite the FP32 model.

---

# 46. Quantization Validation

After quantization:

```text
load quantized model
run test dataset
compare predictions
measure F1
```

Record:

```text
original F1
quantized F1
F1 delta
```

---

# 47. Quantization Performance

Measure:

```text
model size
memory
latency
throughput
```

against FP32.

---

# 48. SQRuntime

SQRuntime means:

**Synthetic Quality Runtime**

It is the custom inference layer.

Its responsibility is:

```text
model artifact
 ↓
load
 ↓
prepare
 ↓
infer
 ↓
postprocess
 ↓
metrics
```

---

# 49. Runtime Components

```text
ModelLoader
InferenceEngine
Preprocessor
Postprocessor
RuntimeMetrics
```

---

# 50. Model Loader

Responsibilities:

```text
load model
validate artifact
load metadata
initialize runtime
warm model
unload model
```

Expose:

```python
load_model(path)
```

---

# 51. Preprocessor

Input:

```text
timestamps
values
```

Output:

```text
normalized model tensor
```

The exact preprocessing used during training must be reproduced at inference.

---

# 52. Inference Engine

Expose:

```python
predict(window)
predict_batch(windows)
```

Return structured results.

Example:

```json
{
  "quality": "degraded",
  "fault_type": "clipping",
  "confidence": 0.972,
  "fault_interval": {
    "start": 412,
    "end": 459
  }
}
```

---

# 53. Runtime State

Runtime status:

```text
offline
loading
ready
error
```

Expose:

```text
model
version
precision
device
memory
request count
latency
throughput
```

---

# 54. FastAPI API

Base:

```text
/api/v1
```

---

# 55. Health API

```http
GET /api/v1/health
```

Response:

```json
{
  "status": "ok"
}
```

---

# 56. Runtime API

```http
GET /api/v1/runtime/status
```

---

# 57. Dataset API

```http
POST /api/v1/datasets/generate
GET  /api/v1/datasets
GET  /api/v1/datasets/{dataset_id}
```

Generate request:

```json
{
  "seed": 42,
  "sampling_rate": 100,
  "duration": 60,
  "faults": {
    "noise": true,
    "drift": true,
    "dropout": true,
    "clipping": true,
    "timestamp_gap": true,
    "sampling_jitter": true
  }
}
```

---

# 58. Training API

```http
POST /api/v1/training/run
GET  /api/v1/training/{experiment_id}
```

---

# 59. Evaluation API

```http
POST /api/v1/evaluation/run
GET  /api/v1/evaluation/{evaluation_id}
```

---

# 60. Quantization API

```http
POST /api/v1/quantization/run
GET  /api/v1/models/{model_id}/quantization
```

---

# 61. Model API

```http
GET /api/v1/models
GET /api/v1/models/{model_id}
```

---

# 62. Inference API

```http
POST /api/v1/inference
POST /api/v1/inference/batch
```

---

# 63. Benchmark API

```http
POST /api/v1/benchmarks/run
GET  /api/v1/benchmarks
GET  /api/v1/benchmarks/{benchmark_id}
```

---

# 64. API Design Rule

All API payloads must use Pydantic schemas.

Never return arbitrary Python dictionaries from routes when a stable response model can be defined.

---

# 65. Long-Running Jobs

Training, quantization and benchmarking may take longer than a normal HTTP request.

The MVP should support a simple job model.

Example:

```text
POST /training/run
       ↓
job_id
       ↓
status = queued
       ↓
running
       ↓
completed
```

For the MVP, a lightweight background execution mechanism is acceptable.

Do not introduce Celery/Redis unless genuinely required.

---

# 66. Job States

```text
QUEUED
RUNNING
COMPLETED
FAILED
CANCELLED
```

---

# 67. PostgreSQL

PostgreSQL stores metadata only.

Required entities:

```text
Dataset
Experiment
Evaluation
Model
Benchmark
Job
```

---

# 68. Dataset Table

Conceptual fields:

```text
id
name
seed
configuration
artifact_path
sample_count
created_at
status
```

---

# 69. Experiment Table

```text
id
dataset_id
model_id
method
configuration
metrics
artifact_path
status
created_at
```

---

# 70. Evaluation Table

```text
id
experiment_id
dataset_id
evaluation_type
metrics
results
created_at
```

---

# 71. Model Table

```text
id
name
version
base_model
precision
quantization
artifact_path
metrics
status
created_at
```

---

# 72. Benchmark Table

```text
id
model_id
device
batch_size
iterations
warmup
latency_metrics
throughput
memory
created_at
```

---

# 73. Artifact Storage

Use:

```text
artifacts/
```

for binary/large files.

Never store:

- model weights
- generated datasets
- benchmark dumps

inside PostgreSQL.

Never commit large artifacts to Git.

---

# 74. Artifact Naming

Use immutable IDs.

Example:

```text
artifacts/
├── datasets/
│   └── DS-0001/
├── models/
│   └── MODEL-0001/
├── experiments/
│   └── EXP-0001/
└── benchmarks/
    └── BENCH-0001/
```

---

# 75. Benchmark Engine

The benchmark engine must distinguish:

```text
cold start
warm inference
```

---

# 76. Warmup

Before collecting latency:

```text
warmup = N requests
```

Warmup measurements are excluded from final latency statistics.

---

# 77. Latency

Collect individual request durations.

Calculate:

```text
P50
P90
P95
P99
min
max
mean
```

P95 is the primary UI metric.

---

# 78. Throughput

Calculate:

```text
requests / second
```

using controlled benchmark duration or iteration count.

---

# 79. Memory

Measure process memory where possible.

Store:

```text
baseline memory
peak memory
model memory
```

---

# 80. Batch Benchmark

Benchmark:

```text
batch = 1
batch = 8
batch = 16
batch = 32
```

Only run supported sizes.

---

# 81. Benchmark Integrity

Every benchmark must record:

```text
model
precision
device
batch size
iterations
warmup
runtime version
Python version
hardware
timestamp
```

Benchmark results must be generated programmatically.

Never manually hardcode benchmark values.

---

# 82. Frontend Integration Architecture

The existing frontend should communicate through:

```text
frontend
   ↓
API client
   ↓
FastAPI
```

Recommended:

```text
frontend/
├── lib/
│   └── api/
│       ├── client.ts
│       ├── datasets.ts
│       ├── training.ts
│       ├── evaluation.ts
│       ├── models.ts
│       ├── inference.ts
│       ├── quantization.ts
│       ├── benchmarks.ts
│       └── runtime.ts
```

---

# 83. Frontend Data Model

Define TypeScript interfaces matching backend schemas.

Required:

```text
Dataset
FaultEvent
TrainingExperiment
EvaluationResult
DistributionShiftResult
Model
QuantizationResult
RuntimeStatus
InferenceResult
BenchmarkResult
```

---

# 84. Mock-to-Real Migration

The frontend currently contains v0-generated mock data.

Replace this gradually.

Correct migration:

```text
Existing UI
     ↓
Typed interface
     ↓
Mock API service
     ↓
Real API service
```

Do not rewrite UI components simply to connect APIs.

---

# 85. Frontend Pages

The frontend architecture should map to:

```text
Overview
Synthetic Data Studio
Training / Experiment Lab
Evaluation Lab
Distribution Shift
Quantization Lab
Runtime Console
Inference Playground
Benchmark Arena
Model Lineage
```

The existing v0 design determines exact routing/component structure.

---

# 86. Overview Data

Overview should consume:

```text
active model
F1
shift robustness
P95 latency
throughput
memory
model size
runtime status
```

These values must ultimately come from backend APIs.

---

# 87. Synthetic Data UI

Frontend sends configuration:

```text
POST /datasets/generate
```

Backend returns:

```text
dataset_id
```

Frontend then fetches dataset summary and visualization data.

---

# 88. Training UI

Frontend:

```text
POST /training/run
```

receives:

```text
job_id
experiment_id
```

Then polls:

```text
GET /training/{experiment_id}
```

until completed.

---

# 89. Evaluation UI

Frontend starts evaluation:

```text
POST /evaluation/run
```

Then displays:

```text
F1
Precision
Recall
IoU
False alarms
Confusion matrix
```

---

# 90. Quantization UI

Frontend starts:

```text
POST /quantization/run
```

Then displays:

```text
FP32
vs
INT8
```

---

# 91. Runtime UI

Runtime status should come from:

```text
GET /runtime/status
```

The UI should periodically refresh telemetry.

Avoid aggressive polling.

---

# 92. Inference Playground

User selects:

```text
model
dataset/window
```

Then:

```text
POST /inference
```

The response drives:

```text
prediction
confidence
fault type
fault interval
```

and waveform overlays.

---

# 93. Benchmark UI

User selects:

```text
model
precision
batch size
iterations
```

Then:

```text
POST /benchmarks/run
```

Results populate:

```text
P50
P95
P99
throughput
memory
```

---

# 94. Error Handling

Frontend must support:

```text
loading
success
empty
error
retry
```

Backend errors should be structured.

Example:

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "The requested model artifact could not be loaded."
  }
}
```

---

# 95. Runtime Error Philosophy

Never silently recover from model or data errors.

Bad:

```text
quantization failed
→ silently use FP32
```

Good:

```text
quantization failed
→ job FAILED
→ explicit error
→ diagnostic information
```

---

# 96. Configuration Management

Environment variables:

```text
DATABASE_URL
ARTIFACT_ROOT
API_HOST
API_PORT
MODEL_ROOT
LOG_LEVEL
```

Use:

```text
.env
```

locally.

Provide:

```text
.env.example
```

Never commit secrets.

---

# 97. Logging

Backend logging should be structured enough to debug:

```text
dataset generation
training
evaluation
quantization
runtime loading
inference
benchmarking
```

Every major operation should include its relevant ID.

Example:

```text
[EXP-0042] Training started
[EXP-0042] Epoch 1/5
[EXP-0042] Validation F1=0.912
[EXP-0042] Checkpoint saved
```

---

# 98. Testing Architecture

Three layers:

```text
UNIT
 ↓
INTEGRATION
 ↓
END-TO-END
```

---

# 99. Unit Tests

Required:

### Data

- signal generation
- deterministic seed
- noise
- drift
- dropout
- clipping
- timestamp gap
- jitter
- validation

### ML

- dataset loading
- model forward pass
- metric calculations
- interval IoU

### Runtime

- preprocessing
- model loading
- inference
- output schema

### Benchmarking

- percentile calculation
- throughput calculation

---

# 100. Integration Tests

Test:

```text
dataset generation
→ artifact creation
→ metadata persistence
```

Test:

```text
training
→ checkpoint
→ experiment record
```

Test:

```text
model
→ runtime
→ inference
```

---

# 101. End-to-End Test

The critical E2E test:

```text
Generate dataset
        ↓
Train small model
        ↓
Evaluate
        ↓
Quantize
        ↓
Load runtime
        ↓
Inference
        ↓
Benchmark
```

Use a very small dataset/model so CI remains practical.

---

# 102. Reproducibility Tests

At least one test should assert:

```text
generate(seed=42)
==
generate(seed=42)
```

within exact numerical expectations appropriate to the representation.

---

# 103. Docker Architecture

Use:

```text
docker-compose.yml
```

Services:

```text
frontend
backend
postgres
```

SQRuntime is initially part of backend.

Do not create a separate runtime container unless required.

---

# 104. Development Flow

Local:

```text
docker compose up
```

Frontend:

```text
localhost:3000
```

Backend:

```text
localhost:8000
```

API documentation:

```text
localhost:8000/docs
```

PostgreSQL:

```text
localhost:5432
```

---

# 105. Dependency Philosophy

Use the smallest reasonable dependency set.

Core backend:

```text
FastAPI
Pydantic
SQLAlchemy
PostgreSQL driver
PyTorch
NumPy
SciPy
scikit-learn
PEFT
Transformers
```

Add quantization/runtime dependencies only where needed.

---

# 106. Frontend Dependencies

Preserve the existing v0 stack where possible.

Expected:

```text
Next.js
React
TypeScript
Tailwind
shadcn/ui
Lucide
charting library
Framer Motion
```

Do not replace dependencies without reason.

---

# 107. API Versioning

All application APIs begin with:

```text
/api/v1
```

This allows future breaking changes.

---

# 108. Model Versioning

Use:

```text
MODEL-0001
MODEL-0002
```

and semantic model versions where appropriate.

A model artifact is immutable after registration.

---

# 109. Dataset Versioning

Datasets should also be immutable.

Example:

```text
DS-0001
DS-0002
```

A dataset configuration change creates a new dataset.

---

# 110. Experiment Lineage

The system must maintain:

```text
Dataset
   ↓
Experiment
   ↓
Checkpoint
   ↓
Quantized Model
   ↓
Benchmark
   ↓
Runtime
```

This lineage is important both technically and visually.

---

# 111. Model Promotion

A model should be eligible for "production/demo ready" status only if:

```text
F1 threshold passed
AND
localization threshold passed
AND
false-alarm threshold passed
AND
distribution-shift threshold passed
AND
runtime benchmark completed
```

Thresholds should be configurable.

---

# 112. Suggested Default Thresholds

```yaml
promotion:
  min_f1: 0.90
  min_interval_iou: 0.75
  max_false_alarm_rate: 0.05
  max_shift_degradation: 0.12
```

These are engineering defaults, not scientific universal standards.

---

# 113. Security Requirements

Even for local-first MVP:

- validate uploaded files
- restrict file sizes
- validate tensor shapes
- validate numerical ranges
- never execute uploaded code
- restrict model artifact paths
- avoid arbitrary filesystem paths from API requests
- sanitize logs
- keep secrets in environment variables

---

# 114. Performance Principles

The backend should avoid:

```text
loading model for every request
```

Instead:

```text
startup
 ↓
load selected model
 ↓
warmup
 ↓
serve requests
```

---

# 115. Model Caching

SQRuntime should keep the active model loaded.

MVP supports:

```text
one active model
```

Future:

```text
multiple resident models
```

---

# 116. Inference Batch Support

Implement:

```python
predict_batch(windows)
```

even if the frontend initially uses single-window inference.

This is required for benchmarking.

---

# 117. Runtime Separation

Never implement inference directly inside:

```text
/api/v1/inference
```

Instead:

```text
API route
 ↓
InferenceService
 ↓
SQRuntime
 ↓
Model
```

This separation is critical.

---

# 118. Service Separation

Similarly:

```text
DatasetRoute
 ↓
DatasetService
 ↓
DataGenerator
```

and:

```text
TrainingRoute
 ↓
TrainingService
 ↓
Trainer
```

This keeps the system testable.

---

# 119. Recommended Implementation Order

Implementation should proceed in this exact order.

## Phase 1 — Repository Reconnaissance

Inspect:

```text
frontend/
```

Identify:

- routes
- components
- mock data
- chart components
- expected interactions
- current dependencies

Do not modify design.

---

# 120. Phase 2 — Backend Foundation

Implement:

```text
FastAPI
configuration
logging
Pydantic schemas
health endpoint
database connection
artifact storage
```

Verify:

```text
GET /api/v1/health
```

works.

---

# 121. Phase 3 — Synthetic Data Engine

Implement:

```text
signals
faults
generator
validator
windowing
artifact persistence
```

Add tests.

Checkpoint:

> Generate the same dataset twice with the same seed and verify reproducibility.

---

# 122. Phase 4 — Dataset API

Implement:

```text
POST /datasets/generate
GET /datasets
GET /datasets/{id}
```

Connect to frontend Synthetic Data Studio.

Checkpoint:

> Clicking "Generate" in the frontend creates a real dataset.

---

# 123. Phase 5 — ML Model

Implement:

```text
model
dataset loader
training loop
checkpoint
metrics
```

Start with a tiny configuration.

Checkpoint:

> A real model trains end-to-end on generated data.

---

# 124. Phase 6 — LoRA / QLoRA

Add:

```text
LoRA
QLoRA
experiment configuration
adapter persistence
```

Checkpoint:

> Training method is configurable and recorded.

---

# 125. Phase 7 — Evaluation

Implement:

```text
precision
recall
F1
confusion matrix
false alarm rate
interval IoU
```

Checkpoint:

> Evaluation results are persisted and visible through API.

---

# 126. Phase 8 — Distribution Shift

Implement:

```text
noise shift
amplitude shift
frequency shift
severity shift
compound faults
```

Checkpoint:

> UI can compare IID and shifted performance.

---

# 127. Phase 9 — Quantization

Implement:

```text
FP32 artifact
INT8 artifact
validation
comparison
```

Checkpoint:

> Both models produce valid inference results.

---

# 128. Phase 10 — SQRuntime

Implement:

```text
ModelLoader
Preprocessor
InferenceEngine
Postprocessor
RuntimeMetrics
```

Checkpoint:

```text
model artifact
 ↓
runtime
 ↓
prediction
```

works independently of HTTP.

---

# 129. Phase 11 — Inference API

Expose:

```text
POST /api/v1/inference
POST /api/v1/inference/batch
GET /api/v1/runtime/status
```

Connect to frontend.

Checkpoint:

> User can submit a sensor window from the UI and see a real prediction.

---

# 130. Phase 12 — Benchmark Engine

Implement:

```text
warmup
latency
P50
P95
P99
throughput
memory
batch sizes
```

Checkpoint:

> Real benchmark numbers appear in Benchmark Arena.

---

# 131. Phase 13 — Complete Frontend Integration

Replace all major mock data with real API responses.

Priority:

```text
Overview
 ↓
Data
 ↓
Training
 ↓
Evaluation
 ↓
Quantization
 ↓
Runtime
 ↓
Benchmark
```

---

# 132. Phase 14 — Integration Hardening

Test:

```text
frontend
 ↕
API
 ↕
services
 ↕
database
 ↕
artifacts
```

Fix:

- loading states
- API failures
- stale state
- incorrect IDs
- race conditions
- invalid inputs
- missing artifacts

---

# 133. Phase 15 — Docker

Ensure:

```text
docker compose up
```

starts the complete MVP.

Verify:

```text
frontend → backend → database
```

works.

---

# 134. Phase 16 — E2E Validation

Execute:

```text
Generate
→ Train
→ Evaluate
→ Shift
→ Quantize
→ Serve
→ Benchmark
```

from a clean environment.

---

# 135. Definition of Done

The MVP is complete only when all of the following work.

### Data

```text
✓ deterministic generation
✓ six required fault types
✓ ground truth
✓ validation
✓ persisted dataset
```

### ML

```text
✓ model training
✓ LoRA
✓ QLoRA
✓ checkpoint
✓ experiment metadata
```

### Evaluation

```text
✓ F1
✓ precision
✓ recall
✓ confusion matrix
✓ false alarm rate
✓ interval IoU
✓ distribution shift
```

### Optimization

```text
✓ FP32
✓ INT8
✓ quality comparison
```

### Runtime

```text
✓ model loading
✓ preprocessing
✓ inference
✓ batch inference
✓ runtime status
```

### Benchmark

```text
✓ P50
✓ P95
✓ P99
✓ throughput
✓ memory
```

### Product

```text
✓ existing v0 frontend preserved
✓ backend connected
✓ API documented
✓ Docker works
✓ tests pass
```

---

# 136. Critical Engineering Constraints

## Constraint 1

**Do not replace the existing frontend.**

---

## Constraint 2

**Do not build a generic dashboard.**

The existing visual system is intentional.

---

## Constraint 3

**Do not hardcode benchmark results.**

All benchmark numbers must originate from actual measurements.

---

## Constraint 4

**Do not fabricate ML metrics.**

Frontend mock data is acceptable during UI development, but final displayed metrics must come from real experiments.

---

## Constraint 5

**Do not hide failures.**

Training, quantization and runtime failures must be explicit.

---

## Constraint 6

**Do not over-engineer the MVP.**

One excellent model and one excellent runtime are preferable to many incomplete features.

---

## Constraint 7

**Everything important must be reproducible.**

Dataset generation, training, evaluation and benchmarks must have configurations and seeds.

---

# 137. Final System Flow

The completed application should behave as follows:

```text
                         USER
                           │
                           ▼
                    SYNTHQUANTA UI
                           │
                           ▼
                     FASTAPI API
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           DATA           ML          RUNTIME
             │             │             │
             ▼             ▼             ▼
         Dataset       Training      Inference
             │             │             │
             └──────┬──────┘             │
                    ▼                    │
                Evaluation              │
                    │                    │
                    ▼                    │
             Distribution Shift         │
                    │                    │
                    ▼                    │
               Quantization             │
                    │                    │
                    └────────┬───────────┘
                             ▼
                        BENCHMARK
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
                  LATENCY  MEMORY  THROUGHPUT
                    │        │        │
                    └────────┼────────┘
                             ▼
                      FRONTEND RESULTS
```

---

# 138. Final Architecture in One Sentence

> **SynthQuanta is a modular monolithic model-engineering platform in which a Next.js instrumentation-style frontend communicates with a FastAPI backend that orchestrates deterministic synthetic sensor-data generation, parameter-efficient fine-tuning, robustness evaluation, quantization, custom SQRuntime inference, and reproducible performance benchmarking, with PostgreSQL storing metadata and the filesystem storing immutable ML artifacts.**

---

# 139. Recommended MVP Milestone Structure

```text
M0  Frontend frozen
 ↓
M1  Backend foundation
 ↓
M2  Synthetic data engine
 ↓
M3  Dataset API + frontend
 ↓
M4  Training pipeline
 ↓
M5  Evaluation + distribution shift
 ↓
M6  Quantization
 ↓
M7  SQRuntime
 ↓
M8  Benchmarking
 ↓
M9  Full frontend integration
 ↓
M10 Docker + tests + E2E
```

Each milestone should produce a **working increment**, not merely a collection of files.

---

# 140. Final Portfolio Demonstration

The final demo should take approximately this path:

```text
OPEN SYNTHQUANTA
       ↓
GENERATE SENSOR DATA
       ↓
SHOW CORRUPTED WAVEFORM
       ↓
START QLORA EXPERIMENT
       ↓
SHOW TRAINING CURVES
       ↓
SHOW MODEL F1
       ↓
RUN DISTRIBUTION SHIFT
       ↓
SHOW ROBUSTNESS DROP
       ↓
QUANTIZE TO INT8
       ↓
SHOW MODEL SIZE REDUCTION
       ↓
DEPLOY TO SQRuntime
       ↓
RUN LIVE INFERENCE
       ↓
SHOW FAULT ON WAVEFORM
       ↓
RUN BENCHMARK
       ↓
SHOW:
   P95 LATENCY
   THROUGHPUT
   MEMORY
```

The viewer should leave with a clear understanding that SynthQuanta is **not merely an anomaly-classification model**.

It demonstrates:

```text
DATA ENGINEERING
       +
MODEL ENGINEERING
       +
ROBUSTNESS ENGINEERING
       +
MODEL OPTIMIZATION
       +
INFERENCE SYSTEMS
       +
PERFORMANCE ENGINEERING
```

---

# 141. Final Engineering Philosophy

The project should follow five principles.

### 1. Reproducible

Same configuration → same experiment.

### 2. Measurable

Every major claim has a metric.

### 3. Modular

Data, ML, runtime and benchmarking remain independently testable.

### 4. Observable

The user can see what the system is doing.

### 5. Honest

No fabricated performance numbers, no fake training progress, and no claims that are not backed by actual execution.

---

# 142. Ultimate MVP Goal

The finished repository should allow a technically sophisticated reviewer to clone the project and understand:

```text
Where does the data come from?
        ↓
How are faults generated?
        ↓
How is the model trained?
        ↓
How is it fine-tuned?
        ↓
How is it evaluated?
        ↓
How does it behave under distribution shift?
        ↓
How is it quantized?
        ↓
How is it served?
        ↓
How fast is it?
        ↓
How much memory does it consume?
```

And, most importantly, the reviewer should be able to **see and reproduce the complete chain themselves**.

That is the definition of SynthQuanta's MVP.