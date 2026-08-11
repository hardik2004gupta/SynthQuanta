'use client'

<<<<<<< Updated upstream
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, Check, ChevronRight, CircleDot, Cpu, Database, Gauge, GitBranch, Hexagon, Keyboard, Layers3, Menu, Play, Radio, Search, Server, Settings2, Sparkles, Terminal, Zap } from 'lucide-react'
import { api, type DatasetResponse, type ExperimentResponse } from '../lib/api'

const stages = [
  { id: 'data', num: '01', label: 'DATA', sub: 'SIGNAL STUDIO', icon: Database },
  { id: 'train', num: '02', label: 'TRAIN', sub: 'ADAPTER LAB', icon: Layers3 },
  { id: 'evaluate', num: '03', label: 'EVALUATE', sub: 'ROBUSTNESS LAB', icon: BarChart3 },
  { id: 'optimize', num: '04', label: 'OPTIMIZE', sub: 'QUANTIZATION LAB', icon: Zap },
  { id: 'runtime', num: '05', label: 'RUNTIME', sub: 'SERVE CONSOLE', icon: Server },
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

function AdapterLab({ datasetId, datasetHumanId }: AdapterLabProps) {
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
