'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, Check, ChevronRight, CircleDot, Cpu, Database, Gauge, GitBranch, Hexagon, Keyboard, Layers3, Menu, Play, Radio, Search, Server, Settings2, Sparkles, Terminal, Zap } from 'lucide-react'
import { api, type DatasetResponse, type EvaluationResponse, type ExperimentResponse, type ShiftScenario } from '../lib/api'

const stages = [
  { id: 'data', num: '01', label: 'DATA', sub: 'SIGNAL STUDIO', icon: Database },
  { id: 'train', num: '02', label: 'TRAIN', sub: 'ADAPTER LAB', icon: Layers3 },
  { id: 'evaluate', num: '03', label: 'EVALUATE', sub: 'ROBUSTNESS LAB', icon: BarChart3 },
  { id: 'optimize', num: '04', label: 'OPTIMIZE', sub: 'QUANTIZATION LAB', icon: Zap },
  { id: 'runtime', num: '05', label: 'RUNTIME', sub: 'SERVE CONSOLE', icon: Server },
]

const metrics = [
  ['F1 SCORE', '93.4%', '+2.8%', 'positive'],
  ['P95 LATENCY', '5.8 ms', '-41%', 'positive'],
  ['THROUGHPUT', '184 req/s', '+16%', 'cyan'],
  ['ARTIFACT SIZE', '412 MB', '-68%', 'violet'],
]

const signalA = 'M0 86 C12 72 18 78 30 52 S48 40 61 66 S80 104 94 65 S111 22 125 49 S145 77 160 54 S175 28 188 56 S208 82 222 48 S240 35 255 63 S272 94 286 58 S305 24 320 50 S338 76 353 55 S369 31 385 57 S402 87 420 48 S437 30 452 52 S470 72 488 46 S505 34 520 58 S540 88 556 48 S574 27 590 51 S608 72 625 48 S642 36 660 55 S678 78 694 44 S712 24 730 50 S748 82 766 46 S784 28 800 53'

function previewToSvgPath(preview: [number, number | null][]): string {
  const pts = preview.filter(p => p[1] !== null) as [number, number][]
  if (pts.length < 2) return ''
  const minT = pts[0][0], maxT = pts[pts.length - 1][0]
  const vs = pts.map(p => p[1])
  const minV = Math.min(...vs), maxV = Math.max(...vs)
  const rT = maxT - minT || 1
  const cV = (minV + maxV) / 2
  const aV = (maxV - minV) / 2 || 1
  return pts.map(([t, v], i) => {
    const x = ((t - minT) / rT) * 800
    const y = 60 - ((v - cV) / aV) * 50
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
}

function Panel({ title, eyebrow, children, className = '' }: { title: string; eyebrow?: string; children: React.ReactNode; className?: string }) {
  return <section className={`panel ${className}`}><div className="panel-head"><div><span className="eyebrow">{eyebrow ?? 'SYSTEM MODULE'}</span><h2>{title}</h2></div><button className="icon-button" aria-label={`Open ${title}`}><ArrowUpRight size={14} /></button></div>{children}</section>
}

function Sparkline({ color = 'cyan', points = '0,32 12,26 24,30 36,18 48,22 60,11 72,17 84,8 96,12' }: { color?: string; points?: string }) {
  return <svg className="sparkline" viewBox="0 0 100 40" preserveAspectRatio="none" aria-hidden="true"><polyline points={points} fill="none" stroke={`var(--${color})`} strokeWidth="1.4" vectorEffect="non-scaling-stroke" /></svg>
}

function Topology() {
  const nodes = [
    { x: 12, y: 50, name: 'DS-0042', desc: '48K SAMPLES', icon: Database, tone: 'cyan' },
    { x: 36, y: 26, name: 'EXP-0042', desc: 'ADAPTER V3', icon: GitBranch, tone: 'violet' },
    { x: 60, y: 50, name: 'EVAL-021', desc: 'ROBUSTNESS', icon: Gauge, tone: 'amber' },
    { x: 84, y: 26, name: 'Q-INT8', desc: '412 MB ARTIFACT', icon: Cpu, tone: 'positive' },
  ]
  return <div className="topology"><svg viewBox="0 0 100 76" preserveAspectRatio="none" aria-hidden="true"><path d="M12 50 L36 26 L60 50 L84 26" /><path className="dash" d="M12 50 L36 26 L60 50 L84 26" /></svg>{nodes.map(({ x, y, name, desc, icon: Icon, tone }) => <div key={name} className={`topo-node ${tone}`} style={{ left: `${x}%`, top: `${y}%` }}><div className="node-core"><Icon size={15} /></div><div className="node-copy"><strong>{name}</strong><span>{desc}</span></div></div>)}</div>
}

// ---------------------------------------------------------------------------
// Signal Studio — DATA stage panel
// ---------------------------------------------------------------------------

const _inp: React.CSSProperties = { width: '100%', padding: '6px 9px', background: '#060a10', border: '1px solid var(--border)', color: 'var(--foreground)', fontSize: '11px', fontFamily: 'var(--font-mono)', marginTop: '4px', outline: 'none' }
const _lbl: React.CSSProperties = { fontSize: '9px', letterSpacing: '.12em', color: '#6d7e90', display: 'block' }

interface StudioProps {
  name: string; onName: (v: string) => void
  seed: number; onSeed: (v: number) => void
  sigType: string; onSigType: (v: string) => void
  duration: number; onDuration: (v: number) => void
  rate: number; onRate: (v: number) => void
  amplitude: number; onAmplitude: (v: number) => void
  frequency: number; onFrequency: (v: number) => void
  windowSize: number; onWindowSize: (v: number) => void
  noise: boolean; onNoise: (v: boolean) => void
  drift: boolean; onDrift: (v: boolean) => void
  dropout: boolean; onDropout: (v: boolean) => void
  clipping: boolean; onClipping: (v: boolean) => void
  loading: boolean; error: string | null; result: DatasetResponse | null
  onGenerate: () => void
}

function SignalStudio(p: StudioProps) {
  const faults: Array<[string, boolean, (v: boolean) => void]> = [
    ['NOISE', p.noise, p.onNoise],
    ['DRIFT', p.drift, p.onDrift],
    ['DROPOUT', p.dropout, p.onDropout],
    ['CLIPPING', p.clipping, p.onClipping],
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', minHeight: '340px' }}>
      <div style={{ padding: '24px 28px', borderRight: '1px solid var(--border)' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 16px' }}>
          <div style={{ gridColumn: '1 / -1' }}>
            <label style={_lbl}>DATASET NAME</label>
            <input style={_inp} value={p.name} onChange={e => p.onName(e.target.value)} />
          </div>
          <div>
            <label style={_lbl}>SIGNAL TYPE</label>
            <select style={_inp} value={p.sigType} onChange={e => p.onSigType(e.target.value)}>
              <option value="sinusoidal">SINUSOIDAL</option>
              <option value="composite">COMPOSITE</option>
              <option value="trend">TREND</option>
              <option value="periodic">PERIODIC</option>
            </select>
          </div>
          <div>
            <label style={_lbl}>SEED</label>
            <input style={_inp} type="number" value={p.seed} onChange={e => p.onSeed(Number(e.target.value))} min={0} max={2147483647} />
          </div>
          <div>
            <label style={_lbl}>DURATION (s)</label>
            <input style={_inp} type="number" value={p.duration} onChange={e => p.onDuration(Number(e.target.value))} min={0.1} max={3600} step={0.5} />
          </div>
          <div>
            <label style={_lbl}>SAMPLING RATE (Hz)</label>
            <input style={_inp} type="number" value={p.rate} onChange={e => p.onRate(Number(e.target.value))} min={1} max={10000} />
          </div>
          <div>
            <label style={_lbl}>AMPLITUDE</label>
            <input style={_inp} type="number" value={p.amplitude} onChange={e => p.onAmplitude(Number(e.target.value))} step={0.1} min={0.01} />
          </div>
          <div>
            <label style={_lbl}>FREQUENCY (Hz)</label>
            <input style={_inp} type="number" value={p.frequency} onChange={e => p.onFrequency(Number(e.target.value))} step={1} min={0.1} />
          </div>
          <div>
            <label style={_lbl}>WINDOW SIZE</label>
            <input style={_inp} type="number" value={p.windowSize} onChange={e => p.onWindowSize(Number(e.target.value))} min={8} max={4096} step={8} />
          </div>
          <div style={{ gridColumn: '1 / -1' }}>
            <span style={_lbl}>FAULT INJECTION</span>
            <div style={{ display: 'flex', gap: '16px', marginTop: '8px', flexWrap: 'wrap' }}>
              {faults.map(([label, checked, setter]) => (
                <label key={label} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '9px', letterSpacing: '.1em', color: checked ? 'var(--amber)' : 'var(--muted)' }}>
                  <input type="checkbox" checked={checked} onChange={e => setter(e.target.checked)} style={{ accentColor: 'var(--amber)', width: '12px', height: '12px' }} />
                  {label}
                </label>
              ))}
            </div>
          </div>
        </div>
        <div style={{ marginTop: '20px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button className="primary-button" onClick={p.onGenerate} disabled={p.loading} style={{ opacity: p.loading ? 0.65 : 1 }}>
            {p.loading ? <><Activity size={14} />GENERATING…</> : <><Sparkles size={14} />GENERATE DATASET</>}
          </button>
          {p.error && <span style={{ fontSize: '9px', color: 'var(--red)', fontFamily: 'var(--font-mono)', letterSpacing: '.05em' }}>{p.error}</span>}
        </div>
      </div>
      <div style={{ padding: '24px 28px' }}>
        {p.result ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '14px', color: 'var(--cyan)', letterSpacing: '.05em' }}>{p.result.human_id}</span>
              <span style={{ border: '1px solid #245441', color: 'var(--green)', padding: '3px 7px', fontFamily: 'var(--font-mono)', fontSize: '9px', letterSpacing: '.08em' }}>COMPLETED</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              {([
                ['SAMPLES', p.result.sample_count.toLocaleString()],
                ['WINDOWS', p.result.window_count.toLocaleString()],
                ['FAULTS', String(p.result.fault_count)],
                ['SIGNAL', p.result.signal_type.toUpperCase()],
                ['DURATION', `${p.result.duration}s`],
                ['RATE', `${p.result.sampling_rate} Hz`],
              ] as [string, string][]).map(([label, val]) => (
                <div key={label} style={{ padding: '8px 10px', background: '#090e16', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90' }}>{label}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', marginTop: '4px' }}>{val}</div>
                </div>
              ))}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
              {Object.entries(p.result.split_counts).map(([k, v]) => (
                <div key={k} style={{ textAlign: 'center', padding: '6px 4px', background: '#090e16', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '7px', letterSpacing: '.08em', color: '#6d7e90' }}>{k.replace('_', ' ').toUpperCase()}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--cyan)', marginTop: '3px' }}>{v}</div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: '9px', letterSpacing: '.08em', color: p.result.validation?.valid ? 'var(--green)' : 'var(--red)', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <Check size={11} />
              {p.result.validation?.valid ? 'VALIDATION PASSED' : `VALIDATION ISSUES (${p.result.validation?.issue_count ?? '?'})`}
            </div>
            {p.result.fault_annotations.length > 0 && (
              <div>
                <div style={{ fontSize: '8px', letterSpacing: '.12em', color: '#6d7e90', marginBottom: '6px' }}>FAULT ANNOTATIONS</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', maxHeight: '80px', overflowY: 'auto' }}>
                  {p.result.fault_annotations.map(a => (
                    <div key={a.fault_id} style={{ display: 'flex', gap: '8px', fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--amber)', letterSpacing: '.04em' }}>
                      <span style={{ color: '#6d7e90' }}>{a.fault_id}</span>
                      <span>{a.fault_type}</span>
                      <span style={{ color: '#6d7e90' }}>[{a.start_index}..{a.end_index}]</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '14px', color: '#405060', textAlign: 'center' }}>
            <Database size={28} strokeWidth={1} />
            <div>
              <div style={{ fontSize: '10px', letterSpacing: '.12em', marginBottom: '6px' }}>AWAITING GENERATION</div>
              <div style={{ fontSize: '9px', letterSpacing: '.08em', color: '#354454', lineHeight: 1.6 }}>Configure signal parameters and inject faults, then generate a deterministic synthetic dataset.</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Adapter Lab — TRAIN stage panel
// ---------------------------------------------------------------------------

interface AdapterLabProps {
  datasetId: string | null
  datasetHumanId: string | null
  onExperimentCreated?: (experimentId: string, humanId: string) => void
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

function AdapterLab({ datasetId, datasetHumanId, onExperimentCreated }: AdapterLabProps) {
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
        if (exp.status === 'COMPLETED' || exp.status === 'FAILED') stopPolling()
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

  // Latest experiment created in TRAIN stage — passed to EVALUATE stage
  const [trainedExpId, setTrainedExpId] = useState<string | null>(null)
  const [trainedExpHumanId, setTrainedExpHumanId] = useState<string | null>(null)

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
          />
        </Panel>
      ) : active === 'evaluate' ? (
        <Panel title="ROBUSTNESS LAB" eyebrow="IID + DISTRIBUTION SHIFT EVALUATION" className="focused-panel">
          <RobustnessLab
            experimentId={trainedExpId}
            experimentHumanId={trainedExpHumanId}
          />
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
