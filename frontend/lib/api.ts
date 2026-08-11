const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SignalConfig {
  type: 'sinusoidal' | 'composite' | 'trend' | 'periodic'
  duration: number
  sampling_rate: number
  amplitude: number
  frequency: number
}

export interface FaultItem {
  enabled: boolean
  std?: number
  magnitude?: number
  direction?: string
  start_frac?: number
  end_frac?: number
  severity?: number
  lower?: number
  upper?: number
  position_frac?: number
  gap_seconds?: number
  jitter_std_seconds?: number
}

export interface FaultConfig {
  noise?: FaultItem
  drift?: FaultItem
  dropout?: FaultItem
  clipping?: FaultItem
  timestamp_gap?: FaultItem
  sampling_jitter?: FaultItem
}

export interface DatasetGenerateRequest {
  name: string
  seed: number
  signal: SignalConfig
  faults: FaultConfig
  window_size: number
}

export interface FaultAnnotation {
  fault_id: string
  fault_type: string
  start_index: number
  end_index: number
  severity: number
  parameters: Record<string, unknown>
}

export interface ValidationSummary {
  valid: boolean
  issue_count: number
  nan_count: number
  gap_count: number
  annotation_count: number
  fault_types_present: string[]
  statistics: {
    mean?: number
    std?: number
    min?: number
    max?: number
  }
}

export interface DatasetResponse {
  dataset_id: string
  human_id: string
  name: string
  status: string
  seed: number
  sample_count: number
  window_count: number
  fault_count: number
  signal_type: string
  duration: number
  sampling_rate: number
  fault_annotations: FaultAnnotation[]
  validation: ValidationSummary
  signal_preview: [number, number | null][]
  split_counts: Record<string, number>
  configuration: Record<string, unknown>
  artifact_path: string
  created_at: string
}

export interface DatasetSummary {
  dataset_id: string
  human_id: string
  name: string
  status: string
  seed: number
  sample_count: number
  window_count: number
  fault_count: number
  signal_type: string
  duration: number
  sampling_rate: number
  created_at: string
}

export interface ListDatasetsResponse {
  items: DatasetSummary[]
  total: number
  limit: number
  offset: number
}

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`API ${res.status}: ${body}`)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Training types
// ---------------------------------------------------------------------------

export interface LoRAConfigRequest {
  rank: number
  alpha: number
  dropout: number
  target_modules: string[]
}

export interface ModelConfigRequest {
  architecture: string
  window_size: number
  embedding_dim: number
  num_layers: number
  num_heads: number
  ffn_dim: number
  dropout: number
}

export interface TrainingRunRequest {
  dataset_id: string
  name: string
  method: 'full' | 'lora' | 'qlora'
  epochs: number
  batch_size: number
  learning_rate: number
  weight_decay: number
  seed: number
  lora: LoRAConfigRequest
  model: ModelConfigRequest
}

export interface TrainingStartResponse {
  experiment_id: string
  human_id: string
  status: string
  message: string
}

export interface EpochMetrics {
  epoch: number
  train_loss: number
  val_loss: number
  val_accuracy: number
  duration_seconds: number
  learning_rate: number
}

export interface ExperimentResponse {
  experiment_id: string
  human_id: string
  name: string
  dataset_id: string
  method: string
  status: string
  configuration: Record<string, unknown>
  metrics: Record<string, unknown> | null
  training_history: EpochMetrics[] | null
  artifact_path: string | null
  duration_seconds: number | null
  hardware_info: Record<string, unknown> | null
  error_message: string | null
  created_at: string
  model_id: string | null
  model_human_id: string | null
}

// ---------------------------------------------------------------------------
// Evaluation types
// ---------------------------------------------------------------------------

export interface EvaluationRunRequest {
  experiment_id: string
  batch_size?: number
  include_shift?: boolean
}

export interface EvaluationStartResponse {
  evaluation_id: string
  human_id: string
  experiment_id: string
  status: string
  message: string
}

export interface ShiftScenario {
  scenario: string
  iid_macro_f1: number
  shifted_macro_f1: number
  absolute_degradation: number
  relative_degradation: number
  robustness_ratio: number
  n_shifted_windows: number
  seed: number
  error?: string | null
}

export interface EvaluationResponse {
  evaluation_id: string
  human_id: string
  experiment_id: string
  model_id: string | null
  dataset_id: string
  status: string
  evaluation_type: string
  metrics: Record<string, unknown> | null
  results: {
    iid?: {
      metrics?: {
        macro_f1: number
        weighted_f1: number
        false_alarm_rate: number
        n_samples: number
        per_class?: Record<string, { precision: number; recall: number; f1: number; support: number }>
      }
      localization?: { mean_iou: number; iou_at_50: number; n_gt_intervals: number }
      n_windows: number
      duration_seconds: number
    }
    distribution_shift?: ShiftScenario[]
  } | null
  duration_seconds: number | null
  hardware_info: Record<string, unknown> | null
  artifact_path: string | null
  created_at: string
}

// ---------------------------------------------------------------------------
// Quantization types
// ---------------------------------------------------------------------------

export interface QuantizationRunRequest {
  source_model_id: string
  dataset_id?: string | null
  benchmark_iterations?: number
  benchmark_warmup?: number
}

export interface QuantizationStartResponse {
  quantization_id: string
  human_id: string
  source_model_id: string
  status: string
  message: string
}

export interface ComparisonMetrics {
  fp32_macro_f1: number
  int8_macro_f1: number
  f1_delta: number
  fp32_size_bytes: number
  int8_size_bytes: number
  size_reduction_ratio: number
  fp32_latency_ms: number
  int8_latency_ms: number
  latency_speedup: number
  n_test_windows: number
}

export interface QuantizationResponse {
  quantization_id: string
  human_id: string
  source_model_id: string
  quantized_model_id: string | null
  dataset_id: string | null
  status: string
  method: string
  backend: string | null
  comparison: ComparisonMetrics | null
  artifact_path: string | null
  duration_seconds: number | null
  error: string | null
  created_at: string
  updated_at: string
}

// ---------------------------------------------------------------------------
// Runtime types
// ---------------------------------------------------------------------------

export interface RuntimeLoadRequest {
  model_id?: string
  quantization_id?: string
}

export interface RuntimeHealthResponse {
  status: string
  model_id: string | null
  artifact_path: string | null
  precision: string | null
  runtime_variant: string | null
  device: string
  backend: string | null
  loaded_at: string | null
  request_count: number
  error: string | null
}

export interface PredictRequest {
  values: number[]
}

export interface PredictionResponse {
  predicted_class: string
  predicted_class_index: number
  confidence: number
  probabilities: Record<string, number>
  latency_ms: number
}

export interface BatchPredictRequest {
  windows: number[][]
}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[]
}

export interface TelemetryResponse {
  request_count: number
  success_count: number
  error_count: number
  last_latency_ms: number | null
  mean_latency_ms: number | null
  p50_latency_ms: number | null
  p95_latency_ms: number | null
  p99_latency_ms: number | null
}

// ---------------------------------------------------------------------------
// Benchmark types
// ---------------------------------------------------------------------------

export interface BenchmarkRunRequest {
  model_id?: string
  quantization_id?: string
  batch_sizes?: number[]
  iterations?: number
  warmup?: number
  seed?: number
}

export interface BenchmarkStartResponse {
  benchmark_id: string
  human_id: string
  model_id: string
  runtime_variant: string
  status: string
  message: string
}

export interface BatchResult {
  batch_size: number
  iterations: number
  warmup_count: number
  latency_stats: Record<string, number>
  throughput_rps: number
  throughput_sps: number
  memory_peak_mb: number | null
  duration_seconds: number
  status: string
  error: string | null
}

export interface BenchmarkResponse {
  benchmark_id: string
  human_id: string
  model_id: string | null
  runtime_variant: string | null
  device: string | null
  status: string
  iterations: number | null
  warmup_count: number | null
  batch_results: BatchResult[] | null
  latency_metrics: Record<string, number> | null
  throughput: number | null
  memory: Record<string, number> | null
  hardware_info: Record<string, unknown> | null
  artifact_path: string | null
  duration_seconds: number | null
  error: string | null
  created_at: string
}

export const api = {
  datasets: {
    generate: (req: DatasetGenerateRequest) =>
      request<DatasetResponse>('/datasets', {
        method: 'POST',
        body: JSON.stringify(req),
      }),

    list: (limit = 50, offset = 0) =>
      request<ListDatasetsResponse>(`/datasets?limit=${limit}&offset=${offset}`),

    get: (id: string) =>
      request<DatasetResponse>(`/datasets/${id}`),
  },

  training: {
    run: (req: TrainingRunRequest) =>
      request<TrainingStartResponse>('/training/run', {
        method: 'POST',
        body: JSON.stringify(req),
      }),

    get: (experimentId: string) =>
      request<ExperimentResponse>(`/training/${experimentId}`),
  },

  evaluation: {
    run: (req: EvaluationRunRequest) =>
      request<EvaluationStartResponse>('/evaluation/run', {
        method: 'POST',
        body: JSON.stringify(req),
      }),

    get: (evaluationId: string) =>
      request<EvaluationResponse>(`/evaluation/${evaluationId}`),
  },

  quantization: {
    run: (req: QuantizationRunRequest) =>
      request<QuantizationStartResponse>('/quantization/run', {
        method: 'POST',
        body: JSON.stringify(req),
      }),

    get: (quantizationId: string) =>
      request<QuantizationResponse>(`/quantization/${quantizationId}`),

    list: (limit = 50, offset = 0) =>
      request<QuantizationResponse[]>(`/quantization?limit=${limit}&offset=${offset}`),
  },

  runtime: {
    load: (req: RuntimeLoadRequest) =>
      request<RuntimeHealthResponse>('/runtime/load', {
        method: 'POST',
        body: JSON.stringify(req),
      }),

    health: () =>
      request<RuntimeHealthResponse>('/runtime/health'),

    predict: (req: PredictRequest) =>
      request<PredictionResponse>('/runtime/predict', {
        method: 'POST',
        body: JSON.stringify(req),
      }),

    predictBatch: (req: BatchPredictRequest) =>
      request<BatchPredictionResponse>('/runtime/predict/batch', {
        method: 'POST',
        body: JSON.stringify(req),
      }),

    telemetry: () =>
      request<TelemetryResponse>('/runtime/telemetry'),
  },

  benchmarks: {
    run: (req: BenchmarkRunRequest) =>
      request<BenchmarkStartResponse>('/benchmarks/run', {
        method: 'POST',
        body: JSON.stringify(req),
      }),

    get: (benchmarkId: string) =>
      request<BenchmarkResponse>(`/benchmarks/${benchmarkId}`),

    list: (limit = 50, offset = 0) =>
      request<BenchmarkResponse[]>(`/benchmarks?limit=${limit}&offset=${offset}`),
  },
}
