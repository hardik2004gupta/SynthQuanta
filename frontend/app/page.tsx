'use client'

<<<<<<< Updated upstream
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, Check, ChevronRight, CircleDot, Cpu, Database, Gauge, GitBranch, Hexagon, Keyboard, Layers3, Menu, Play, Radio, Search, Server, Settings2, Sparkles, Terminal, Zap } from 'lucide-react'
import { api, type BatchResult, type BenchmarkResponse, type ComparisonMetrics, type DatasetResponse, type EvaluationResponse, type ExperimentResponse, type PredictionResponse, type QuantizationResponse, type RuntimeHealthResponse, type ShiftScenario, type TelemetryResponse } from '../lib/api'

const stages = [
  { id: 'data', num: '01', label: 'DATA', sub: 'SIGNAL STUDIO', icon: Database },
  { id: 'train', num: '02', label: 'TRAIN', sub: 'ADAPTER LAB', icon: Layers3 },
  { id: 'evaluate', num: '03', label: 'EVALUATE', sub: 'ROBUSTNESS LAB', icon: BarChart3 },
  { id: 'optimize', num: '04', label: 'OPTIMIZE', sub: 'QUANTIZATION LAB', icon: Zap },
  { id: 'runtime', num: '05', label: 'RUNTIME', sub: 'SERVE CONSOLE', icon: Server },
  { id: 'benchmark', num: '06', label: 'BENCHMARK', sub: 'ARENA', icon: Gauge },
]
=======
import { useState, useCallback } from 'react'
import WorkflowNav from '@/components/WorkflowNav'
import Overview from '@/components/Overview'
import DataStudio from '@/components/DataStudio'
import TrainLab from '@/components/TrainLab'
import EvalLab from '@/components/EvalLab'
import QuantizeLab from '@/components/QuantizeLab'
import RuntimeConsole from '@/components/RuntimeConsole'
import BenchmarkLab from '@/components/BenchmarkLab'
import type { DatasetResponse, EvaluationResponse, QuantizationResponse, BenchmarkResponse } from '@/lib/api'

export type Stage = 'OVERVIEW' | 'DATA' | 'TRAIN' | 'EVALUATE' | 'OPTIMIZE' | 'RUNTIME' | 'BENCHMARK'
>>>>>>> Stashed changes

export interface WorkflowState {
  datasetId: string | null
<<<<<<< Updated upstream
  datasetHumanId: string | null
  onExperimentCreated?: (experimentId: string, humanId: string) => void
  onModelReady?: (modelId: string) => void
}

function LossChart({ history }: { history: { epoch: number; train_loss: number; val_loss: number }[] }) {
  if (history.length === 0) return null
  const W = 400, H = 80, pad = 4
  const losses = history.flatMap(h => [h.train_loss, h.val_loss])
  const minL = Math.min(...losses), maxL = Math.max(...losses)
  const rL = maxL - minL || 0.01
  const toX = (i: number) => pad + (i / Math.max(history.length - 1, 1)) * (W - 2 * pad)
  const toY = (v: number) => H - pad - ((v - minL) / rL) * (H - 2 * pad)
  const trainPts = history.map((h, i) => `${toX(i).toFixed(1)},${toY(h.train_loss).toFixed(1)}`).join(' ')
  const valPts = history.map((h, i) => `${toX(i).toFixed(1)},${toY(h.val_loss).toFixed(1)}`).join(' ')
  return (
    <div style={{ marginTop: '12px' }}>
      <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '4px' }}>LOSS CURVES</div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', overflow: 'visible' }} aria-label="Training loss curves">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--border)" strokeWidth="0.5" />
        <polyline points={trainPts} fill="none" stroke="var(--cyan)" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
        <polyline points={valPts} fill="none" stroke="var(--violet)" strokeWidth="1.2" vectorEffect="non-scaling-stroke" strokeDasharray="3 2" />
      </svg>
      <div style={{ display: 'flex', gap: '14px', marginTop: '4px' }}>
        <span style={{ fontSize: '8px', display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--cyan)' }}><i style={{ display: 'inline-block', width: '14px', height: '2px', background: 'var(--cyan)' }} />TRAIN</span>
        <span style={{ fontSize: '8px', display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--violet)' }}><i style={{ display: 'inline-block', width: '14px', height: '2px', background: 'var(--violet)', opacity: 0.7 }} />VALIDATION</span>
      </div>
    </div>
  )
}

function AdapterLab({ datasetId, datasetHumanId, onExperimentCreated, onModelReady }: AdapterLabProps) {
  const [method, setMethod] = useState<'full' | 'lora' | 'qlora'>('lora')
  const [epochs, setEpochs] = useState(3)
  const [batchSize, setBatchSize] = useState(16)
  const [lr, setLr] = useState(0.001)
  const [seed, setSeed] = useState(42)
  const [loraRank, setLoraRank] = useState(4)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [experiment, setExperiment] = useState<ExperimentResponse | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current !== null) { clearInterval(pollRef.current); pollRef.current = null }
  }

  useEffect(() => () => stopPolling(), [])

  const startPolling = useCallback((expId: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const exp = await api.training.get(expId)
        setExperiment(exp)
        if (exp.status === 'COMPLETED' || exp.status === 'FAILED') {
          stopPolling()
          if (exp.status === 'COMPLETED' && exp.model_id) onModelReady?.(exp.model_id)
        }
      } catch { /* keep polling */ }
    }, 2000)
  }, [])

  const startTraining = useCallback(async () => {
    if (!datasetId) { setError('Generate a dataset first (DATA stage).'); return }
    setLoading(true); setError(null)
    try {
      const resp = await api.training.run({
        dataset_id: datasetId,
        name: `${datasetHumanId ?? 'ds'}-${method}`,
        method, epochs, batch_size: batchSize,
        learning_rate: lr, weight_decay: 0.01, seed,
        lora: { rank: loraRank, alpha: loraRank * 2, dropout: 0.05, target_modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj'] },
        model: { architecture: 'sensor-transformer-small', window_size: 128, embedding_dim: 32, num_layers: 2, num_heads: 4, ffn_dim: 64, dropout: 0.1 },
      })
      const initial: ExperimentResponse = {
        experiment_id: resp.experiment_id,
        human_id: resp.human_id,
        name: `${datasetHumanId ?? 'ds'}-${method}`,
        dataset_id: datasetId,
        method, status: resp.status,
        configuration: {}, metrics: null, training_history: null,
        artifact_path: null, duration_seconds: null, hardware_info: null,
        error_message: null, created_at: new Date().toISOString(),
        model_id: null, model_human_id: null,
      }
      setExperiment(initial)
      startPolling(resp.experiment_id)
      onExperimentCreated?.(resp.experiment_id, resp.human_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed to start')
    } finally {
      setLoading(false)
    }
  }, [datasetId, datasetHumanId, method, epochs, batchSize, lr, seed, loraRank, startPolling])

  const history = experiment?.training_history ?? []
  const isRunning = experiment?.status === 'PENDING' || experiment?.status === 'RUNNING'
  const statusColor = experiment?.status === 'COMPLETED' ? 'var(--green)' : experiment?.status === 'FAILED' ? 'var(--red)' : experiment?.status === 'RUNNING' ? 'var(--cyan)' : '#6d7e90'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: '340px' }}>
      {/* Left: config form */}
      <div style={{ padding: '24px 28px', borderRight: '1px solid var(--border)' }}>
        {datasetId ? (
          <div style={{ fontSize: '9px', letterSpacing: '.08em', color: 'var(--cyan)', marginBottom: '14px', fontFamily: 'var(--font-mono)' }}>
            DATASET: {datasetHumanId ?? datasetId}
          </div>
        ) : (
          <div style={{ fontSize: '9px', letterSpacing: '.08em', color: 'var(--amber)', marginBottom: '14px' }}>
            No dataset — go to DATA stage first.
          </div>
        )}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 16px' }}>
          <div style={{ gridColumn: '1 / -1' }}>
            <label style={_lbl}>METHOD</label>
            <select style={_inp} value={method} onChange={e => setMethod(e.target.value as 'full' | 'lora' | 'qlora')}>
              <option value="lora">LORA (RECOMMENDED)</option>
              <option value="full">FULL FINE-TUNING</option>
              <option value="qlora">QLORA (REQUIRES CUDA)</option>
            </select>
          </div>
          <div>
            <label style={_lbl}>EPOCHS</label>
            <input style={_inp} type="number" value={epochs} onChange={e => setEpochs(Number(e.target.value))} min={1} max={1000} />
          </div>
          <div>
            <label style={_lbl}>BATCH SIZE</label>
            <input style={_inp} type="number" value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} min={1} max={512} />
          </div>
          <div>
            <label style={_lbl}>LEARNING RATE</label>
            <input style={_inp} type="number" value={lr} onChange={e => setLr(Number(e.target.value))} step={0.0001} min={0.00001} max={1} />
          </div>
          <div>
            <label style={_lbl}>SEED</label>
            <input style={_inp} type="number" value={seed} onChange={e => setSeed(Number(e.target.value))} min={0} />
          </div>
          {method !== 'full' && (
            <div>
              <label style={_lbl}>LORA RANK</label>
              <input style={_inp} type="number" value={loraRank} onChange={e => setLoraRank(Number(e.target.value))} min={1} max={64} />
            </div>
          )}
        </div>
        <div style={{ marginTop: '20px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            className="primary-button"
            onClick={startTraining}
            disabled={loading || isRunning || !datasetId}
            style={{ opacity: (loading || isRunning || !datasetId) ? 0.65 : 1 }}
          >
            {loading ? <><Activity size={14} />STARTING…</> : isRunning ? <><Activity size={14} />TRAINING…</> : <><Sparkles size={14} />START TRAINING</>}
          </button>
          {error && <span style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', letterSpacing: '.05em' }}>{error}</span>}
        </div>
      </div>
      {/* Right: experiment status + loss curves */}
      <div style={{ padding: '24px 28px' }}>
        {experiment ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', color: 'var(--violet)', letterSpacing: '.05em' }}>{experiment.human_id}</span>
              <span style={{ border: `1px solid ${statusColor}22`, color: statusColor, padding: '3px 7px', fontFamily: 'var(--font-mono)', fontSize: '9px', letterSpacing: '.08em' }}>
                {experiment.status}
              </span>
              {isRunning && <Activity size={12} style={{ color: 'var(--cyan)', animation: 'spin 1s linear infinite' }} />}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              {([
                ['METHOD', experiment.method.toUpperCase()],
                ['EPOCHS', history.length > 0 ? `${history.length} / ${experiment.configuration?.epochs ?? '?'}` : String(experiment.configuration?.epochs ?? '—')],
                experiment.metrics?.best_val_loss != null ? ['BEST VAL LOSS', (experiment.metrics.best_val_loss as number).toFixed(4)] : null,
                experiment.metrics?.final_val_accuracy != null ? ['VAL ACCURACY', `${((experiment.metrics.final_val_accuracy as number) * 100).toFixed(1)}%`] : null,
                experiment.duration_seconds != null ? ['DURATION', `${experiment.duration_seconds.toFixed(1)}s`] : null,
                experiment.model_human_id ? ['MODEL', experiment.model_human_id] : null,
              ] as ([string, string] | null)[]).filter(Boolean).map(item => {
                const [label, val] = item!
                return (
                  <div key={label} style={{ padding: '8px 10px', background: '#090e16', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90' }}>{label}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', marginTop: '4px' }}>{val}</div>
                  </div>
                )
              })}
            </div>
            {experiment.status === 'FAILED' && experiment.error_message && (
              <div style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', padding: '8px', background: '#160a0a', border: '1px solid #4a1515', letterSpacing: '.04em', lineHeight: 1.5 }}>
                ERROR: {experiment.error_message}
              </div>
            )}
            <LossChart history={history as { epoch: number; train_loss: number; val_loss: number }[]} />
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '14px', color: '#405060', textAlign: 'center' }}>
            <Layers3 size={28} strokeWidth={1} />
            <div>
              <div style={{ fontSize: '10px', letterSpacing: '.12em', marginBottom: '6px' }}>ADAPTER LAB READY</div>
              <div style={{ fontSize: '9px', letterSpacing: '.08em', color: '#354454', lineHeight: 1.6 }}>Configure LoRA hyperparameters and start a training experiment on your generated dataset.</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Robustness Lab — EVALUATE stage panel
// ---------------------------------------------------------------------------

const SHIFT_LABELS: Record<string, string> = {
  noise_shift: 'NOISE SHIFT',
  amplitude_shift: 'AMPLITUDE SHIFT',
  frequency_shift: 'FREQUENCY SHIFT',
  severity_shift: 'SEVERITY SHIFT',
  compound_shift: 'COMPOUND SHIFT',
}

interface RobustnessLabProps {
  experimentId: string | null
  experimentHumanId: string | null
}

function RobustnessLab({ experimentId, experimentHumanId }: RobustnessLabProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current !== null) { clearInterval(pollRef.current); pollRef.current = null }
  }

  useEffect(() => () => stopPolling(), [])

  const startPolling = useCallback((evalId: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const ev = await api.evaluation.get(evalId)
        setEvaluation(ev)
        if (ev.status === 'COMPLETED' || ev.status === 'FAILED') stopPolling()
      } catch { /* keep polling */ }
    }, 2000)
  }, [])

  const startEvaluation = useCallback(async () => {
    if (!experimentId) { setError('Train a model first (TRAIN stage).'); return }
    setLoading(true); setError(null)
    try {
      const resp = await api.evaluation.run({ experiment_id: experimentId, include_shift: true })
      const initial: EvaluationResponse = {
        evaluation_id: resp.evaluation_id,
        human_id: resp.human_id,
        experiment_id: resp.experiment_id,
        model_id: null, dataset_id: '',
        status: resp.status,
        evaluation_type: 'iid+shift',
        metrics: null, results: null,
        duration_seconds: null, hardware_info: null, artifact_path: null,
        created_at: new Date().toISOString(),
      }
      setEvaluation(initial)
      startPolling(resp.evaluation_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evaluation failed to start')
    } finally {
      setLoading(false)
    }
  }, [experimentId, startPolling])

  const isRunning = evaluation?.status === 'PENDING' || evaluation?.status === 'RUNNING'
  const statusColor = evaluation?.status === 'COMPLETED' ? 'var(--green)' : evaluation?.status === 'FAILED' ? 'var(--red)' : evaluation?.status === 'RUNNING' ? 'var(--cyan)' : '#6d7e90'
  const iidMetrics = evaluation?.results?.iid?.metrics
  const iidLocalization = evaluation?.results?.iid?.localization
  const shiftScenarios: ShiftScenario[] = (evaluation?.results?.distribution_shift ?? []) as ShiftScenario[]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: '340px' }}>
      {/* Left: IID metrics */}
      <div style={{ padding: '24px 28px', borderRight: '1px solid var(--border)' }}>
        {experimentId ? (
          <div style={{ fontSize: '9px', letterSpacing: '.08em', color: 'var(--violet)', marginBottom: '14px', fontFamily: 'var(--font-mono)' }}>
            EXPERIMENT: {experimentHumanId ?? experimentId}
          </div>
        ) : (
          <div style={{ fontSize: '9px', letterSpacing: '.08em', color: 'var(--amber)', marginBottom: '14px' }}>
            No trained model — go to TRAIN stage first.
          </div>
        )}
        <div style={{ marginBottom: '16px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            className="primary-button"
            onClick={startEvaluation}
            disabled={loading || isRunning || !experimentId}
            style={{ opacity: (loading || isRunning || !experimentId) ? 0.65 : 1 }}
          >
            {loading ? <><Activity size={14} />STARTING…</> : isRunning ? <><Activity size={14} />EVALUATING…</> : <><BarChart3 size={14} />RUN EVALUATION</>}
          </button>
          {error && <span style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', letterSpacing: '.05em' }}>{error}</span>}
        </div>
        {evaluation && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--amber)', letterSpacing: '.05em' }}>{evaluation.human_id}</span>
              <span style={{ border: `1px solid ${statusColor}22`, color: statusColor, padding: '3px 7px', fontFamily: 'var(--font-mono)', fontSize: '9px', letterSpacing: '.08em' }}>
                {evaluation.status}
              </span>
              {isRunning && <Activity size={12} style={{ color: 'var(--cyan)', animation: 'spin 1s linear infinite' }} />}
            </div>
            {iidMetrics && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                {([
                  ['MACRO F1', (iidMetrics.macro_f1 * 100).toFixed(1) + '%'],
                  ['WEIGHTED F1', (iidMetrics.weighted_f1 * 100).toFixed(1) + '%'],
                  ['FALSE ALARM RATE', (iidMetrics.false_alarm_rate * 100).toFixed(1) + '%'],
                  iidLocalization ? ['MEAN IoU', iidLocalization.mean_iou.toFixed(3)] : null,
                  iidLocalization ? ['IoU@0.5', (iidLocalization.iou_at_50 * 100).toFixed(0) + '%'] : null,
                  evaluation.duration_seconds != null ? ['EVAL TIME', `${evaluation.duration_seconds.toFixed(1)}s`] : null,
                ] as ([string, string] | null)[]).filter(Boolean).map(item => {
                  const [label, val] = item!
                  return (
                    <div key={label} style={{ padding: '8px 10px', background: '#090e16', border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90' }}>{label}</div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', marginTop: '4px', color: label === 'FALSE ALARM RATE' ? 'var(--amber)' : 'inherit' }}>{val}</div>
                    </div>
                  )
                })}
              </div>
            )}
            {evaluation.status === 'FAILED' && !!evaluation.metrics?.error && (
              <div style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', padding: '8px', background: '#160a0a', border: '1px solid #4a1515', letterSpacing: '.04em', lineHeight: 1.5 }}>
                ERROR: {String(evaluation.metrics?.error ?? '')}
              </div>
            )}
          </div>
        )}
        {!evaluation && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '200px', gap: '12px', color: '#405060', textAlign: 'center' }}>
            <BarChart3 size={28} strokeWidth={1} />
            <div>
              <div style={{ fontSize: '10px', letterSpacing: '.12em', marginBottom: '6px' }}>ROBUSTNESS LAB READY</div>
              <div style={{ fontSize: '9px', letterSpacing: '.08em', color: '#354454', lineHeight: 1.6 }}>IID evaluation + 5 distribution-shift scenarios on the trained model.</div>
            </div>
          </div>
        )}
      </div>
      {/* Right: shift scenario table */}
      <div style={{ padding: '24px 28px', overflowY: 'auto' }}>
        <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '12px' }}>DISTRIBUTION SHIFT SCENARIOS</div>
        {shiftScenarios.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {shiftScenarios.map((s) => (
              <div key={s.scenario} style={{ padding: '10px 12px', background: '#090e16', border: '1px solid var(--border)', display: 'grid', gridTemplateColumns: '1fr auto auto auto', alignItems: 'center', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '8px', letterSpacing: '.1em', color: '#6d7e90' }}>{SHIFT_LABELS[s.scenario] ?? s.scenario}</div>
                  {s.error && <div style={{ fontSize: '8px', color: 'var(--red)', marginTop: '2px' }}>FAILED</div>}
                </div>
                {!s.error && (
                  <>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>IID F1</div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--green)' }}>{(s.iid_macro_f1 * 100).toFixed(1)}%</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>SHIFTED F1</div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: s.shifted_macro_f1 < s.iid_macro_f1 * 0.85 ? 'var(--red)' : 'var(--amber)' }}>{(s.shifted_macro_f1 * 100).toFixed(1)}%</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>ROBUSTNESS</div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: s.robustness_ratio >= 0.9 ? 'var(--green)' : s.robustness_ratio >= 0.7 ? 'var(--amber)' : 'var(--red)' }}>{(s.robustness_ratio * 100).toFixed(0)}%</div>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '200px', gap: '10px', color: '#354454', textAlign: 'center' }}>
            <div style={{ fontSize: '9px', letterSpacing: '.08em', lineHeight: 1.7 }}>
              Five scenarios will run automatically with evaluation:<br />
              Noise · Amplitude · Frequency · Severity · Compound
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// QuantizationLab — OPTIMIZE stage
// ---------------------------------------------------------------------------

interface QuantizationLabProps {
  modelId: string | null
  datasetId: string | null
  onQuantizationReady?: (quantizationId: string) => void
}

function QuantizationLab({ modelId, datasetId, onQuantizationReady }: QuantizationLabProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [quant, setQuant] = useState<QuantizationResponse | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current !== null) { clearInterval(pollRef.current); pollRef.current = null }
  }

  useEffect(() => () => stopPolling(), [])

  const startPolling = useCallback((quantId: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const q = await api.quantization.get(quantId)
        setQuant(q)
        if (q.status === 'COMPLETED' || q.status === 'FAILED') {
          stopPolling()
          if (q.status === 'COMPLETED') onQuantizationReady?.(quantId)
        }
      } catch { /* keep polling */ }
    }, 2000)
  }, [])

  const startQuantization = useCallback(async () => {
    if (!modelId) { setError('Train a model first (TRAIN stage).'); return }
    setLoading(true); setError(null)
    try {
      const resp = await api.quantization.run({
        source_model_id: modelId,
        dataset_id: datasetId ?? undefined,
        benchmark_iterations: 50,
        benchmark_warmup: 10,
      })
      const initial: QuantizationResponse = {
        quantization_id: resp.quantization_id,
        human_id: resp.human_id,
        source_model_id: resp.source_model_id,
        quantized_model_id: null,
        dataset_id: datasetId ?? null,
        status: resp.status,
        method: 'dynamic_int8',
        backend: null,
        comparison: null,
        artifact_path: null,
        duration_seconds: null,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
      setQuant(initial)
      startPolling(resp.quantization_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Quantization failed to start')
    } finally {
      setLoading(false)
    }
  }, [modelId, datasetId, startPolling])

  const isRunning = quant?.status === 'PENDING' || quant?.status === 'RUNNING'
  const statusColor = quant?.status === 'COMPLETED' ? 'var(--green)' : quant?.status === 'FAILED' ? 'var(--red)' : quant?.status === 'RUNNING' ? 'var(--cyan)' : '#6d7e90'
  const cmp: ComparisonMetrics | null = quant?.comparison ?? null

  const sizeReduction = cmp ? ((1 - 1 / cmp.size_reduction_ratio) * 100).toFixed(0) : null
  const f1DeltaPct = cmp ? (cmp.f1_delta * 100).toFixed(1) : null
  const latencySpeedup = cmp ? cmp.latency_speedup.toFixed(2) : null

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: '340px' }}>
      {/* Left: config + status */}
      <div style={{ padding: '24px 28px', borderRight: '1px solid var(--border)' }}>
        {modelId ? (
          <div style={{ fontSize: '9px', letterSpacing: '.08em', color: 'var(--cyan)', marginBottom: '14px', fontFamily: 'var(--font-mono)' }}>
            FP32 MODEL: {modelId.slice(0, 8)}…
          </div>
        ) : (
          <div style={{ fontSize: '9px', letterSpacing: '.08em', color: 'var(--amber)', marginBottom: '14px' }}>
            No trained model — go to TRAIN stage first.
          </div>
        )}
        <div style={{ marginBottom: '16px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            className="primary-button"
            onClick={startQuantization}
            disabled={loading || isRunning || !modelId}
            style={{ opacity: (loading || isRunning || !modelId) ? 0.65 : 1 }}
          >
            {loading ? <><Activity size={14} />STARTING…</> : isRunning ? <><Activity size={14} />QUANTIZING…</> : <><Zap size={14} />QUANTIZE TO INT8</>}
          </button>
          {error && <span style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', letterSpacing: '.05em' }}>{error}</span>}
        </div>
        {quant && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--violet)', letterSpacing: '.05em' }}>{quant.human_id}</span>
              <span style={{ border: `1px solid ${statusColor}22`, color: statusColor, padding: '3px 7px', fontFamily: 'var(--font-mono)', fontSize: '9px', letterSpacing: '.08em' }}>
                {quant.status}
              </span>
              {isRunning && <Activity size={12} style={{ color: 'var(--cyan)', animation: 'spin 1s linear infinite' }} />}
            </div>
            {quant.backend && (
              <div style={{ fontSize: '8px', color: '#6d7e90', fontFamily: 'var(--font-mono)', letterSpacing: '.08em' }}>
                BACKEND: {quant.backend.toUpperCase()} · METHOD: DYNAMIC INT8
              </div>
            )}
            {quant.status === 'FAILED' && quant.error && (
              <div style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', padding: '8px', background: '#160a0a', border: '1px solid #4a1515', letterSpacing: '.04em', lineHeight: 1.5 }}>
                ERROR: {quant.error}
              </div>
            )}
            {quant.quantized_model_id && (
              <div style={{ padding: '8px 10px', background: '#090e16', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '8px', letterSpacing: '.1em', color: '#6d7e90', marginBottom: '4px' }}>INT8 ARTIFACT</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--green)' }}>{quant.quantized_model_id.slice(0, 8)}…</div>
              </div>
            )}
          </div>
        )}
        {!quant && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '200px', gap: '12px', color: '#405060', textAlign: 'center' }}>
            <Zap size={28} strokeWidth={1} />
            <div>
              <div style={{ fontSize: '10px', letterSpacing: '.12em', marginBottom: '6px' }}>QUANTIZATION LAB READY</div>
              <div style={{ fontSize: '9px', letterSpacing: '.08em', color: '#354454', lineHeight: 1.6 }}>FP32 → INT8 via PyTorch dynamic quantization.<br />Compares F1, model size, and latency.</div>
            </div>
          </div>
        )}
      </div>
      {/* Right: FP32 vs INT8 comparison */}
      <div style={{ padding: '24px 28px', overflowY: 'auto' }}>
        <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '12px' }}>FP32 vs INT8 COMPARISON</div>
        {cmp ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {/* Quality */}
            <div style={{ padding: '10px 12px', background: '#090e16', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: '7px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '8px' }}>CLASSIFICATION QUALITY</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>FP32 F1</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--cyan)' }}>{(cmp.fp32_macro_f1 * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>INT8 F1</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--green)' }}>{(cmp.int8_macro_f1 * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>F1 DELTA</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: cmp.f1_delta > 0.02 ? 'var(--red)' : cmp.f1_delta < -0.005 ? 'var(--green)' : 'var(--amber)' }}>
                    {cmp.f1_delta > 0 ? '-' : '+'}{Math.abs(cmp.f1_delta * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
            {/* Size */}
            <div style={{ padding: '10px 12px', background: '#090e16', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: '7px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '8px' }}>MODEL SIZE</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>FP32</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{(cmp.fp32_size_bytes / 1024).toFixed(0)} KB</div>
                </div>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>INT8</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{(cmp.int8_size_bytes / 1024).toFixed(0)} KB</div>
                </div>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>REDUCTION</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--violet)' }}>{sizeReduction}%</div>
                </div>
              </div>
              <div style={{ marginTop: '8px' }}>
                <div style={{ height: '4px', background: '#1a2535', borderRadius: '2px', overflow: 'hidden', display: 'flex', gap: '2px' }}>
                  <div style={{ width: '100%', background: 'var(--cyan)', opacity: 0.4 }} />
                </div>
                <div style={{ height: '4px', background: '#1a2535', borderRadius: '2px', overflow: 'hidden', marginTop: '3px', display: 'flex' }}>
                  <div style={{ width: `${100 / cmp.size_reduction_ratio}%`, background: 'var(--violet)' }} />
                </div>
                <div style={{ display: 'flex', gap: '12px', marginTop: '4px' }}>
                  <span style={{ fontSize: '7px', color: 'var(--cyan)', opacity: 0.7 }}>━ FP32</span>
                  <span style={{ fontSize: '7px', color: 'var(--violet)' }}>━ INT8</span>
                </div>
              </div>
            </div>
            {/* Latency */}
            <div style={{ padding: '10px 12px', background: '#090e16', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: '7px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '8px' }}>INFERENCE LATENCY (SINGLE WINDOW)</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>FP32</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{cmp.fp32_latency_ms.toFixed(2)} ms</div>
                </div>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>INT8</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{cmp.int8_latency_ms.toFixed(2)} ms</div>
                </div>
                <div>
                  <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>SPEEDUP</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--green)' }}>{latencySpeedup}×</div>
                </div>
              </div>
            </div>
            <div style={{ fontSize: '8px', color: '#405060', letterSpacing: '.06em', lineHeight: 1.5 }}>
              {cmp.n_test_windows} test windows · {quant?.duration_seconds?.toFixed(1) ?? '—'}s total
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '240px', gap: '10px', color: '#354454', textAlign: 'center' }}>
            <div style={{ fontSize: '9px', letterSpacing: '.08em', lineHeight: 1.7 }}>
              Comparison metrics appear after quantization completes:<br />
              F1 delta · Size reduction · Latency speedup
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ServeConsole — RUNTIME stage
// ---------------------------------------------------------------------------

interface ServeConsoleProps {
  modelId: string | null
  quantizationId: string | null
}

function ServeConsole({ modelId, quantizationId }: ServeConsoleProps) {
  const [health, setHealth] = useState<RuntimeHealthResponse | null>(null)
  const [loadLoading, setLoadLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [predictValues, setPredictValues] = useState('')
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictError, setPredictError] = useState<string | null>(null)
  const [telemetry, setTelemetry] = useState<TelemetryResponse | null>(null)

  useEffect(() => {
    const tick = async () => {
      try {
        const [h, t] = await Promise.all([api.runtime.health(), api.runtime.telemetry()])
        setHealth(h)
        setTelemetry(t)
      } catch { /* backend may not be running */ }
    }
    tick()
    const interval = setInterval(tick, 3000)
    return () => clearInterval(interval)
  }, [])

  const loadModel = useCallback(async (precision: 'fp32' | 'int8') => {
    setLoadLoading(true); setLoadError(null)
    try {
      const req = precision === 'fp32' ? { model_id: modelId! } : { quantization_id: quantizationId! }
      const h = await api.runtime.load(req)
      setHealth(h)
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Load failed')
    } finally {
      setLoadLoading(false)
    }
  }, [modelId, quantizationId])

  const runPredict = useCallback(async () => {
    setPredictLoading(true); setPredictError(null)
    try {
      const values = predictValues.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
      if (values.length === 0) { setPredictError('Enter comma-separated numbers.'); return }
      const result = await api.runtime.predict({ values })
      setPrediction(result)
    } catch (err) {
      setPredictError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setPredictLoading(false)
    }
  }, [predictValues])

  const statusColor = health?.status === 'ready' ? 'var(--green)' : health?.status === 'failed' ? 'var(--red)' : health?.status === 'loading' ? 'var(--cyan)' : '#6d7e90'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: '340px' }}>
      <div style={{ padding: '24px 28px', borderRight: '1px solid var(--border)' }}>
        <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '12px' }}>RUNTIME STATUS</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: statusColor, letterSpacing: '.05em' }}>
            {health?.status?.toUpperCase() ?? 'OFFLINE'}
          </span>
          {health?.model_id && (
            <span style={{ fontSize: '9px', color: '#6d7e90', fontFamily: 'var(--font-mono)' }}>
              {health.model_id.slice(0, 8)}… · {health.precision?.toUpperCase()}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
          <button className="primary-button" onClick={() => loadModel('fp32')} disabled={loadLoading || !modelId} style={{ opacity: (!modelId || loadLoading) ? 0.65 : 1 }}>
            {loadLoading ? <><Activity size={14} />LOADING…</> : <><Server size={14} />LOAD FP32</>}
          </button>
          <button className="outline-button" onClick={() => loadModel('int8')} disabled={loadLoading || !quantizationId} style={{ opacity: (!quantizationId || loadLoading) ? 0.65 : 1 }}>
            {loadLoading ? <><Activity size={14} />LOADING…</> : <><Zap size={14} />LOAD INT8</>}
          </button>
        </div>
        {loadError && <div style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', letterSpacing: '.04em', marginBottom: '12px' }}>{loadError}</div>}
        {telemetry && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {([
              ['REQUESTS', String(telemetry.request_count)],
              ['ERRORS', String(telemetry.error_count)],
              ['MEAN LATENCY', telemetry.mean_latency_ms != null ? `${telemetry.mean_latency_ms.toFixed(2)} ms` : '—'],
              ['P95 LATENCY', telemetry.p95_latency_ms != null ? `${telemetry.p95_latency_ms.toFixed(2)} ms` : '—'],
            ] as [string, string][]).map(([label, val]) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', fontFamily: 'var(--font-mono)', padding: '4px 0', borderBottom: '1px solid #13202e' }}>
                <span style={{ color: '#6d7e90', letterSpacing: '.08em' }}>{label}</span>
                <span style={{ color: 'var(--cyan)' }}>{val}</span>
              </div>
            ))}
          </div>
        )}
        {!modelId && !quantizationId && (
          <div style={{ fontSize: '9px', color: 'var(--amber)', letterSpacing: '.06em', marginTop: '8px', lineHeight: 1.6 }}>Train a model (TRAIN stage) and optionally quantize it (OPTIMIZE stage) to enable runtime loading.</div>
        )}
      </div>
      <div style={{ padding: '24px 28px' }}>
        <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '12px' }}>LIVE INFERENCE</div>
        <div style={{ marginBottom: '10px' }}>
          <label style={_lbl}>SENSOR VALUES (comma-separated floats)</label>
          <textarea
            style={{ ..._inp, height: '56px', resize: 'vertical', lineHeight: 1.5 } as React.CSSProperties}
            value={predictValues}
            onChange={e => setPredictValues(e.target.value)}
            placeholder="0.0, 0.1, -0.2, 0.3, …"
          />
        </div>
        <div style={{ marginBottom: '14px', display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button className="primary-button" onClick={runPredict} disabled={predictLoading || health?.status !== 'ready'} style={{ opacity: (predictLoading || health?.status !== 'ready') ? 0.65 : 1 }}>
            {predictLoading ? <><Activity size={14} />RUNNING…</> : <><Radio size={14} />PREDICT</>}
          </button>
          {predictError && <span style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)' }}>{predictError}</span>}
        </div>
        {prediction ? (
          <div style={{ padding: '12px', background: '#090e16', border: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', color: 'var(--cyan)', letterSpacing: '.05em' }}>{prediction.predicted_class}</span>
              <span style={{ fontSize: '9px', color: '#6d7e90' }}>PREDICTED CLASS</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <div>
                <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>CONFIDENCE</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--green)' }}>{(prediction.confidence * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ fontSize: '7px', color: '#6d7e90', letterSpacing: '.08em' }}>LATENCY</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--amber)' }}>{prediction.latency_ms.toFixed(2)} ms</div>
              </div>
            </div>
            <div>
              <div style={{ fontSize: '7px', letterSpacing: '.1em', color: '#6d7e90', marginBottom: '6px' }}>CLASS PROBABILITIES</div>
              {Object.entries(prediction.probabilities).map(([cls, prob]) => (
                <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '3px' }}>
                  <span style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: '#6d7e90', width: '120px', flexShrink: 0 }}>{cls}</span>
                  <div style={{ flex: 1, height: '4px', background: '#1a2535', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ width: `${prob * 100}%`, height: '100%', background: cls === prediction.predicted_class ? 'var(--cyan)' : 'var(--violet)', opacity: 0.7 }} />
                  </div>
                  <span style={{ fontSize: '8px', fontFamily: 'var(--font-mono)', color: '#6d7e90', width: '36px', textAlign: 'right' }}>{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '160px', gap: '10px', color: '#405060', textAlign: 'center' }}>
            <Radio size={26} strokeWidth={1} />
            <div style={{ fontSize: '9px', letterSpacing: '.08em', lineHeight: 1.6 }}>Load a model then enter sensor values<br />to run live inference through SQRuntime.</div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// BenchmarkArena — BENCHMARK stage
// ---------------------------------------------------------------------------

interface BenchmarkArenaProps {
  modelId: string | null
  quantizationId: string | null
}

function BenchmarkArena({ modelId, quantizationId }: BenchmarkArenaProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null)
  const [iterations, setIterations] = useState(50)
  const [warmup, setWarmup] = useState(10)
  const [precision, setPrecision] = useState<'fp32' | 'int8'>('fp32')
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = () => {
    if (pollRef.current !== null) { clearInterval(pollRef.current); pollRef.current = null }
  }
  useEffect(() => () => stopPolling(), [])

  const startPolling = useCallback((benchId: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const b = await api.benchmarks.get(benchId)
        setBenchmark(b)
        if (b.status === 'COMPLETED' || b.status === 'FAILED') stopPolling()
      } catch { /* keep polling */ }
    }, 2000)
  }, [])

  const runBenchmark = useCallback(async () => {
    if (precision === 'fp32' && !modelId) { setError('Train a model first (TRAIN stage).'); return }
    if (precision === 'int8' && !quantizationId) { setError('Quantize a model first (OPTIMIZE stage).'); return }
    setLoading(true); setError(null)
    try {
      const resp = await api.benchmarks.run({
        ...(precision === 'fp32' ? { model_id: modelId! } : { quantization_id: quantizationId! }),
        batch_sizes: [1, 4, 8, 16],
        iterations,
        warmup,
        seed: 42,
      })
      const initial: BenchmarkResponse = {
        benchmark_id: resp.benchmark_id,
        human_id: resp.human_id,
        model_id: resp.model_id,
        runtime_variant: resp.runtime_variant,
        device: 'cpu',
        status: resp.status,
        iterations,
        warmup_count: warmup,
        batch_results: null,
        latency_metrics: null,
        throughput: null,
        memory: null,
        hardware_info: null,
        artifact_path: null,
        duration_seconds: null,
        error: null,
        created_at: new Date().toISOString(),
      }
      setBenchmark(initial)
      startPolling(resp.benchmark_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed to start')
    } finally {
      setLoading(false)
    }
  }, [precision, modelId, quantizationId, iterations, warmup, startPolling])

  const isRunning = benchmark?.status === 'PENDING' || benchmark?.status === 'RUNNING'
  const statusColor = benchmark?.status === 'COMPLETED' ? 'var(--green)' : benchmark?.status === 'FAILED' ? 'var(--red)' : benchmark?.status === 'RUNNING' ? 'var(--cyan)' : '#6d7e90'
  const batchResults: BatchResult[] = benchmark?.batch_results ?? []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: '340px' }}>
      <div style={{ padding: '24px 28px', borderRight: '1px solid var(--border)' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 16px', marginBottom: '20px' }}>
          <div style={{ gridColumn: '1 / -1' }}>
            <label style={_lbl}>PRECISION TARGET</label>
            <select style={_inp} value={precision} onChange={e => setPrecision(e.target.value as 'fp32' | 'int8')}>
              <option value="fp32">FP32 MODEL</option>
              <option value="int8">INT8 QUANTIZED</option>
            </select>
          </div>
          <div>
            <label style={_lbl}>ITERATIONS</label>
            <input style={_inp} type="number" value={iterations} onChange={e => setIterations(Number(e.target.value))} min={10} max={1000} step={10} />
          </div>
          <div>
            <label style={_lbl}>WARMUP RUNS</label>
            <input style={_inp} type="number" value={warmup} onChange={e => setWarmup(Number(e.target.value))} min={1} max={100} />
          </div>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button className="primary-button" onClick={runBenchmark} disabled={loading || isRunning} style={{ opacity: (loading || isRunning) ? 0.65 : 1 }}>
            {loading ? <><Activity size={14} />STARTING…</> : isRunning ? <><Activity size={14} />RUNNING…</> : <><Gauge size={14} />RUN BENCHMARK</>}
          </button>
          {error && <span style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', letterSpacing: '.05em' }}>{error}</span>}
        </div>
        {benchmark && (
          <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--violet)', letterSpacing: '.05em' }}>{benchmark.human_id}</span>
            <span style={{ border: `1px solid ${statusColor}22`, color: statusColor, padding: '3px 7px', fontFamily: 'var(--font-mono)', fontSize: '9px', letterSpacing: '.08em' }}>{benchmark.status}</span>
            {isRunning && <Activity size={12} style={{ color: 'var(--cyan)', animation: 'spin 1s linear infinite' }} />}
          </div>
        )}
        {benchmark?.status === 'FAILED' && benchmark.error && (
          <div style={{ marginTop: '10px', fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', padding: '8px', background: '#160a0a', border: '1px solid #4a1515', letterSpacing: '.04em', lineHeight: 1.5 }}>
            ERROR: {benchmark.error}
          </div>
        )}
        {!modelId && !quantizationId && (
          <div style={{ marginTop: '16px', fontSize: '9px', color: 'var(--amber)', letterSpacing: '.06em', lineHeight: 1.6 }}>Train a model then optionally quantize it to benchmark.</div>
        )}
      </div>
      <div style={{ padding: '24px 28px', overflowY: 'auto' }}>
        <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '12px' }}>BATCH SIZE RESULTS · P50 / P95 / P99 / THROUGHPUT</div>
        {batchResults.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '50px 1fr 1fr 1fr 1fr', gap: '8px', fontSize: '7px', letterSpacing: '.1em', color: '#6d7e90', paddingBottom: '6px', borderBottom: '1px solid var(--border)' }}>
              <span>BATCH</span><span>P50</span><span>P95</span><span>P99</span><span>THROUGHPUT</span>
            </div>
            {batchResults.filter(r => r.status === 'COMPLETED').map(r => (
              <div key={r.batch_size} style={{ display: 'grid', gridTemplateColumns: '50px 1fr 1fr 1fr 1fr', gap: '8px', fontSize: '10px', fontFamily: 'var(--font-mono)', padding: '6px 0', borderBottom: '1px solid #13202e', alignItems: 'center' }}>
                <span style={{ color: 'var(--cyan)' }}>×{r.batch_size}</span>
                <span>{r.latency_stats?.p50?.toFixed(2) ?? '—'} ms</span>
                <span style={{ color: 'var(--amber)' }}>{r.latency_stats?.p95?.toFixed(2) ?? '—'} ms</span>
                <span>{r.latency_stats?.p99?.toFixed(2) ?? '—'} ms</span>
                <span style={{ color: 'var(--green)' }}>{r.throughput_rps?.toFixed(0) ?? '—'}/s</span>
              </div>
            ))}
            {benchmark?.duration_seconds != null && (
              <div style={{ fontSize: '8px', color: '#405060', letterSpacing: '.06em', marginTop: '6px', lineHeight: 1.6 }}>
                {benchmark.duration_seconds.toFixed(1)}s total · {benchmark.runtime_variant?.toUpperCase()} · {benchmark.device?.toUpperCase()}<br />
                {benchmark.iterations} iterations · {benchmark.warmup_count} warmup runs excluded
              </div>
            )}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '220px', gap: '12px', color: '#405060', textAlign: 'center' }}>
            <BarChart3 size={28} strokeWidth={1} />
            <div>
              <div style={{ fontSize: '10px', letterSpacing: '.12em', marginBottom: '6px' }}>BENCHMARK ARENA READY</div>
              <div style={{ fontSize: '9px', letterSpacing: '.08em', color: '#354454', lineHeight: 1.6 }}>Run benchmark to measure P50/P95/P99 latency<br />and throughput across batch sizes 1 · 4 · 8 · 16.</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [active, setActive] = useState('overview')
  const [palette, setPalette] = useState(false)
  const [running, setRunning] = useState(false)

  // Dataset generation state
  const [dsName, setDsName] = useState('sensor-dataset')
  const [dsSeed, setDsSeed] = useState(42)
  const [dsSigType, setDsSigType] = useState('composite')
  const [dsDuration, setDsDuration] = useState(10.0)
  const [dsRate, setDsRate] = useState(100.0)
  const [dsAmplitude, setDsAmplitude] = useState(1.0)
  const [dsFrequency, setDsFrequency] = useState(10.0)
  const [dsWindowSize, setDsWindowSize] = useState(128)
  const [dsNoise, setDsNoise] = useState(true)
  const [dsDrift, setDsDrift] = useState(false)
  const [dsDropout, setDsDropout] = useState(false)
  const [dsClipping, setDsClipping] = useState(false)
  const [dsLoading, setDsLoading] = useState(false)
  const [dsResult, setDsResult] = useState<DatasetResponse | null>(null)
  const [dsError, setDsError] = useState<string | null>(null)

  // Latest experiment created in TRAIN stage — passed to EVALUATE + OPTIMIZE stages
  const [trainedExpId, setTrainedExpId] = useState<string | null>(null)
  const [trainedExpHumanId, setTrainedExpHumanId] = useState<string | null>(null)
  // FP32 model_id set when experiment completes — passed to OPTIMIZE stage
  const [trainedModelId, setTrainedModelId] = useState<string | null>(null)
  // INT8 quantization_id set when quantization completes — passed to RUNTIME + BENCHMARK stages
  const [quantizationId, setQuantizationId] = useState<string | null>(null)

  const generateDataset = useCallback(async () => {
    setDsLoading(true)
    setDsError(null)
    try {
      const result = await api.datasets.generate({
        name: dsName,
        seed: dsSeed,
        signal: { type: dsSigType as 'sinusoidal' | 'composite' | 'trend' | 'periodic', duration: dsDuration, sampling_rate: dsRate, amplitude: dsAmplitude, frequency: dsFrequency },
        faults: {
          ...(dsNoise ? { noise: { enabled: true, std: 0.1 } } : {}),
          ...(dsDrift ? { drift: { enabled: true, magnitude: 0.3, direction: 'positive' } } : {}),
          ...(dsDropout ? { dropout: { enabled: true, start_frac: 0.4, end_frac: 0.5 } } : {}),
          ...(dsClipping ? { clipping: { enabled: true, lower: -1.5, upper: 1.5 } } : {}),
        },
        window_size: dsWindowSize,
      })
      setDsResult(result)
    } catch (err) {
      setDsError(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setDsLoading(false)
    }
  }, [dsName, dsSeed, dsSigType, dsDuration, dsRate, dsAmplitude, dsFrequency, dsWindowSize, dsNoise, dsDrift, dsDropout, dsClipping])

  const activeStage = stages.find((stage) => stage.id === active) ?? stages[0]
  const ActiveIcon = activeStage.icon
  const title = active === 'overview' ? 'MODEL ENGINEERING OVERVIEW' : `${activeStage.label} CONTROL SURFACE`

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); setPalette(true) }
      if (event.key === 'Escape') setPalette(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const commandItems = useMemo(() => ['Return to overview', ...stages.map((stage) => `Open ${stage.label.toLowerCase()}`)], [])

  // Signal Integrity panel — use real data when available, fall back to mock
  const sigPath = dsResult ? previewToSvgPath(dsResult.signal_preview) || signalA : signalA
  const sigEyebrow = dsResult ? `${dsResult.human_id} / SENSOR TELEMETRY` : 'DS-0042 / SENSOR TELEMETRY'
  const sigRateLabel = dsResult ? `${dsResult.sampling_rate} Hz · ${dsResult.duration} SEC WINDOW` : '100 Hz · 8.0 SEC WINDOW'
  const sigSamples = dsResult ? `${dsResult.sample_count.toLocaleString()} SAMPLES` : '48,000 / 48,000 SAMPLES'
  const sigIntegrity = dsResult ? (dsResult.validation?.valid ? '100% INTEGRITY' : 'ISSUES DETECTED') : '98.7% INTEGRITY'
  const firstFault = dsResult?.fault_annotations[0]
  const faultMarkerLeft = firstFault && dsResult ? `${(firstFault.start_index / dsResult.sample_count) * 100}%` : '57%'
  const faultMarkerLabel = firstFault ? firstFault.fault_type : 'FAULT INJECTION'
  const faultMarkerSub = firstFault ? firstFault.fault_id : 'F-12'
  const showFaultOverlay = !dsResult  // keep mock fault path only when no real data

  return <main className="app-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark"><Hexagon size={21} strokeWidth={1.5} /></div><div><div className="brand-name">SYNTH<span>QUANTA</span></div><div className="brand-tag">SYNTHETIC DATA / INFERENCE SYSTEMS</div></div></div><div className="topbar-center"><span className="live-dot" /> <span>LOCAL ENVIRONMENT</span><span className="divider" /><span className="mono muted">sq-dev-01</span></div><div className="top-actions"><button className="command-trigger" onClick={() => setPalette(true)}><Search size={14} /><span>Command palette</span><kbd>⌘ K</kbd></button><button className="icon-button"><Settings2 size={16} /></button><button className="mobile-menu" aria-label="Open menu"><Menu size={18} /></button></div></header>
    <nav className="pipeline" aria-label="Model engineering pipeline"><button className={`overview-link ${active === 'overview' ? 'selected' : ''}`} onClick={() => setActive('overview')}><CircleDot size={16} /><span>OVERVIEW</span></button><div className="rail-line" />{stages.map((stage, index) => { const Icon = stage.icon; return <button key={stage.id} className={`stage ${active === stage.id ? 'selected' : ''}`} onClick={() => setActive(stage.id)}><div className="stage-top"><span className="stage-num">{stage.num}</span><span className="stage-label">{stage.label}</span><Icon size={14} /></div><div className="stage-node"><span /></div><small>{stage.sub}</small>{index < stages.length - 1 && <ChevronRight className="stage-chevron" size={13} />}</button> })}</nav>
    <div className="content"><div className="page-intro"><div><div className="breadcrumb"><span>WORKSPACE</span><ChevronRight size={12} /><span className="active-crumb">{active === 'overview' ? 'OVERVIEW' : activeStage?.label}</span></div><h1>{title}</h1><p>From synthetic data to measurable inference.</p></div><div className="intro-actions"><span className="status-chip"><span className="status-pulse" /> PIPELINE HEALTHY</span><button className="outline-button" onClick={() => setRunning(!running)}>{running ? <><Activity size={14} /> RUNNING</> : <><Play size={14} /> RUN INFERENCE</>}</button></div></div>
      {active === 'data' ? (
        <Panel title="SIGNAL STUDIO" eyebrow="SYNTHETIC DATA ENGINE" className="focused-panel">
          <SignalStudio
            name={dsName} onName={setDsName}
            seed={dsSeed} onSeed={setDsSeed}
            sigType={dsSigType} onSigType={setDsSigType}
            duration={dsDuration} onDuration={setDsDuration}
            rate={dsRate} onRate={setDsRate}
            amplitude={dsAmplitude} onAmplitude={setDsAmplitude}
            frequency={dsFrequency} onFrequency={setDsFrequency}
            windowSize={dsWindowSize} onWindowSize={setDsWindowSize}
            noise={dsNoise} onNoise={setDsNoise}
            drift={dsDrift} onDrift={setDsDrift}
            dropout={dsDropout} onDropout={setDsDropout}
            clipping={dsClipping} onClipping={setDsClipping}
            loading={dsLoading} error={dsError} result={dsResult}
            onGenerate={generateDataset}
          />
        </Panel>
      ) : active === 'train' ? (
        <Panel title="ADAPTER LAB" eyebrow="LORA / QLORA FINE-TUNING" className="focused-panel">
          <AdapterLab
            datasetId={dsResult?.dataset_id ?? null}
            datasetHumanId={dsResult?.human_id ?? null}
            onExperimentCreated={(id, hid) => { setTrainedExpId(id); setTrainedExpHumanId(hid) }}
            onModelReady={(mid) => setTrainedModelId(mid)}
          />
        </Panel>
      ) : active === 'evaluate' ? (
        <Panel title="ROBUSTNESS LAB" eyebrow="IID + DISTRIBUTION SHIFT EVALUATION" className="focused-panel">
          <RobustnessLab
            experimentId={trainedExpId}
            experimentHumanId={trainedExpHumanId}
          />
        </Panel>
      ) : active === 'optimize' ? (
        <Panel title="QUANTIZATION LAB" eyebrow="FP32 → INT8 DYNAMIC QUANTIZATION" className="focused-panel">
          <QuantizationLab
            modelId={trainedModelId}
            datasetId={dsResult?.dataset_id ?? null}
            onQuantizationReady={(qid) => setQuantizationId(qid)}
          />
        </Panel>
      ) : active === 'runtime' ? (
        <Panel title="SERVE CONSOLE" eyebrow="SQRUNTIME / LIVE INFERENCE" className="focused-panel">
          <ServeConsole modelId={trainedModelId} quantizationId={quantizationId} />
        </Panel>
      ) : active === 'benchmark' ? (
        <Panel title="BENCHMARK ARENA" eyebrow="LATENCY · THROUGHPUT · MEMORY" className="focused-panel">
          <BenchmarkArena modelId={trainedModelId} quantizationId={quantizationId} />
        </Panel>
      ) : active !== 'overview' ? (
        <Panel title={`${activeStage?.label} MODULE`} eyebrow="ACTIVE STAGE" className="focused-panel"><div className="focused-stage"><div className="focused-icon"><ActiveIcon size={30} /></div><div><h3>{activeStage.sub}</h3><p>This module is connected to the <span className="mono">sensor-transformer-int8</span> lineage. Select a downstream node or return to overview to inspect the full system map.</p></div><button className="primary-button" onClick={() => setActive('overview')}>VIEW SYSTEM MAP <ArrowUpRight size={14} /></button></div></Panel>
      ) : <>
      <div className="metric-grid">{metrics.map(([label, value, delta, tone], i) => <div className="metric" key={label}><div className="metric-label">{label}<span className="metric-index">0{i + 1}</span></div><div className="metric-value">{value}</div><div className={`metric-delta ${tone}`}>{delta} <span>vs baseline</span></div><Sparkline color={tone === 'positive' ? 'green' : tone} /></div>)}</div>
      <div className="workspace-grid"><Panel title="PIPELINE TOPOLOGY" eyebrow="LIVE LINEAGE" className="topology-panel"><div className="panel-meta"><span><span className="legend-dot cyan" /> ACTIVE PATH</span><span className="mono">UPDATED 14:32:08 UTC</span></div><Topology /><div className="topology-footer"><div><span className="eyebrow">CURRENT ARTIFACT</span><strong className="mono">sensor-transformer-int8</strong></div><div><span className="eyebrow">VERSION</span><strong className="mono">v0.8.4</strong></div><div><span className="eyebrow">STATUS</span><strong className="positive-text"><Check size={13} /> VERIFIED</strong></div></div></Panel><Panel title="SIGNAL INTEGRITY" eyebrow={sigEyebrow} className="signal-panel"><div className="signal-head"><div><strong>Temperature sensor / channel 04</strong><span className="mono muted">{sigRateLabel}</span></div><span className="integrity-badge">{sigIntegrity}</span></div><div className="signal-chart"><div className="chart-axis"><span>1.0</span><span>0.5</span><span>0.0</span><span>-0.5</span></div><svg viewBox="0 0 800 120" preserveAspectRatio="none" aria-label="Signal integrity waveform"><path className="grid-path" d="M0 30H800M0 60H800M0 90H800" /><path className="signal-clean" d={sigPath} />{showFaultOverlay && <path className="signal-fault" d="M0 86 C18 72 35 78 48 52 S75 104 94 65 S112 22 125 49 M405 58 L435 42 L455 79 L480 54" />}</svg>{(showFaultOverlay || firstFault) && <span className="fault-marker" style={{ left: faultMarkerLeft }}>{faultMarkerLabel} <b>{faultMarkerSub}</b></span>}</div><div className="signal-legend"><span><i className="line cyan" /> CLEAN TRACE</span><span><i className="line amber" /> INJECTED FAULT</span><span className="mono muted">{sigSamples}</span></div></Panel></div>
      <div className="lower-grid"><Panel title="RUNTIME TELEMETRY" eyebrow="CUSTOM INFERENCE SERVER"><div className="runtime-status"><div className="server-icon"><Radio size={17} /></div><div><strong>sq-runtime-01</strong><span className="mono muted">CUDA / INT8 KERNELS</span></div><span className="runtime-online">ONLINE</span></div><div className="telemetry-bars"><div><span>GPU UTILIZATION</span><b>68%</b><i><em style={{ width: '68%' }} /></i></div><div><span>MEMORY ALLOCATION</span><b>4.2 / 8 GB</b><i><em className="violet-fill" style={{ width: '52%' }} /></i></div><div><span>QUEUE DEPTH</span><b>03</b><i><em className="green-fill" style={{ width: '18%' }} /></i></div></div></Panel><Panel title="EXPERIMENT TIMELINE" eyebrow="EXP-0042 / TRAINING RUN"><div className="timeline"><div className="timeline-item complete"><span /><div><strong>DATA VALIDATION</strong><small>48K samples · 0.4s</small></div><b>DONE</b></div><div className="timeline-item complete"><span /><div><strong>ADAPTER FINE-TUNE</strong><small>12 epochs · 18m 42s</small></div><b>DONE</b></div><div className="timeline-item active"><span /><div><strong>ROBUSTNESS EVALUATION</strong><small>Shift suite · running</small></div><b>RUNNING</b></div></div></Panel><Panel title="BENCHMARK DELTA" eyebrow="BASELINE VS OPTIMIZED"><div className="benchmark"><div className="bench-row"><span>LATENCY</span><div><i style={{ width: '74%' }} /><i className="accent" style={{ width: '42%' }} /></div><strong>-41%</strong></div><div className="bench-row"><span>THROUGHPUT</span><div><i style={{ width: '48%' }} /><i className="accent" style={{ width: '71%' }} /></div><strong>+16%</strong></div><div className="bench-row"><span>MEMORY</span><div><i style={{ width: '82%' }} /><i className="accent" style={{ width: '31%' }} /></div><strong>-68%</strong></div><div className="bench-legend"><span><i /> BASELINE</span><span><i className="accent" /> INT8 RUNTIME</span></div></div></Panel></div></>}
    </div><footer className="footer"><span><Terminal size={13} /> SYNTHQUANTA ENGINEERING CONSOLE <span className="muted">/</span> BUILD 0.8.4</span><span className="footer-right"><span className="live-dot" /> ALL SYSTEMS NOMINAL <span className="muted">·</span> <Keyboard size={13} /> SHORTCUTS</span></footer>
    {palette && <div className="palette-overlay" onClick={() => setPalette(false)}><div className="palette" role="dialog" aria-modal="true" aria-label="Command palette" onClick={(event) => event.stopPropagation()}><div className="palette-search"><Search size={17} /><input autoFocus placeholder="Search command surface..." /></div><div className="palette-list">{commandItems.map((item, index) => <button key={item} onClick={() => { setActive(index === 0 ? 'overview' : stages[index - 1].id); setPalette(false) }}><span>{index === 0 ? <CircleDot size={15} /> : <Sparkles size={15} />}{item}</span><kbd>{index === 0 ? '⌘ 1' : `⌘ ${index + 1}`}</kbd></button>)}</div></div></div>}
  </main>
}

export default App
=======
  dataset: DatasetResponse | null
  experimentId: string | null
  modelId: string | null
  evaluationId: string | null
  evaluation: EvaluationResponse | null
  quantizationId: string | null
  quantization: QuantizationResponse | null
  benchmarkId: string | null
  benchmark: BenchmarkResponse | null
}

export default function App() {
  const [stage, setStage] = useState<Stage>('OVERVIEW')
  const [state, setState] = useState<WorkflowState>({
    datasetId: null,
    dataset: null,
    experimentId: null,
    modelId: null,
    evaluationId: null,
    evaluation: null,
    quantizationId: null,
    quantization: null,
    benchmarkId: null,
    benchmark: null,
  })

  const onDatasetReady = useCallback((ds: DatasetResponse) => {
    setState(s => ({ ...s, datasetId: ds.dataset_id, dataset: ds }))
  }, [])

  const onTrainComplete = useCallback((experimentId: string, modelId: string) => {
    setState(s => ({ ...s, experimentId, modelId }))
  }, [])

  const onEvalComplete = useCallback((evaluationId: string, evaluation: EvaluationResponse) => {
    setState(s => ({ ...s, evaluationId, evaluation }))
  }, [])

  const onQuantizationReady = useCallback((quantizationId: string, quantization: QuantizationResponse) => {
    setState(s => ({ ...s, quantizationId, quantization }))
  }, [])

  const onBenchmarkComplete = useCallback((benchmarkId: string, benchmark: BenchmarkResponse) => {
    setState(s => ({ ...s, benchmarkId, benchmark }))
  }, [])

  function renderStage() {
    switch (stage) {
      case 'OVERVIEW':
        return <Overview state={state} onNavigate={setStage} />
      case 'DATA':
        return <DataStudio onDatasetReady={onDatasetReady} existingDataset={state.dataset} />
      case 'TRAIN':
        return (
          <TrainLab
            datasetId={state.datasetId!}
            dataset={state.dataset}
            onTrainComplete={onTrainComplete}
            existingExperimentId={state.experimentId}
          />
        )
      case 'EVALUATE':
        return (
          <EvalLab
            experimentId={state.experimentId!}
            modelId={state.modelId}
            onEvalComplete={onEvalComplete}
            existingEvaluationId={state.evaluationId}
            existingEvaluation={state.evaluation}
          />
        )
      case 'OPTIMIZE':
        return (
          <QuantizeLab
            modelId={state.modelId!}
            datasetId={state.datasetId}
            onQuantizationReady={onQuantizationReady}
            existingQuantizationId={state.quantizationId}
            existingQuantization={state.quantization}
          />
        )
      case 'RUNTIME':
        return (
          <RuntimeConsole
            modelId={state.modelId}
            quantizationId={state.quantizationId}
            dataset={state.dataset}
          />
        )
      case 'BENCHMARK':
        return (
          <BenchmarkLab
            modelId={state.modelId}
            quantizationId={state.quantizationId}
            onBenchmarkComplete={onBenchmarkComplete}
            existingBenchmarkId={state.benchmarkId}
            existingBenchmark={state.benchmark}
          />
        )
    }
  }

  return (
    <div className="sq-shell">
      <header className="sq-topbar">
        <div className="sq-brand">
          <span className="sq-wordmark">
            SYNTH<em>QUANTA</em>
          </span>
          <span className="sq-brand-pipe" />
          <span className="sq-brand-sub">MODEL ENGINEERING CONSOLE</span>
        </div>
        <div className="sq-topbar-right">
          {state.datasetId && (
            <span className="sq-model-chip">{state.dataset?.human_id ?? state.datasetId.slice(0, 8)}</span>
          )}
          {state.modelId && (
            <span className="sq-model-chip">EXP active</span>
          )}
          <div className="sq-sys-status">
            <span className="sq-pulse" />
            SYSTEM READY
          </div>
        </div>
      </header>

      <main className="sq-workspace">
        <div className="sq-workspace-scroll">
          <div key={stage} className="sq-stage-anim">
            {renderStage()}
          </div>
        </div>
      </main>

      <WorkflowNav stage={stage} state={state} onNavigate={setStage} />
    </div>
  )
}
>>>>>>> Stashed changes
