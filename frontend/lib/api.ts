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
}
