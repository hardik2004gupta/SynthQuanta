'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { api, type RuntimeHealthResponse, type PredictionResponse, type DatasetResponse } from '@/lib/api'

interface Props {
  modelId: string | null
  quantizationId: string | null
  dataset: DatasetResponse | null
}

const FAULT_LABELS = ['NORMAL', 'NOISE', 'DRIFT', 'DROPOUT', 'CLIPPING', 'TIMESTAMP_GAP', 'SAMPLING_JITTER']

function ProbBar({ label, prob, isTop }: { label: string; prob: number; isTop: boolean }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
        <span className="sq-mono" style={{ color: isTop ? 'var(--accent)' : 'var(--fg-2)', fontWeight: isTop ? 700 : 400 }}>{label}</span>
        <span className="sq-data" style={{ fontSize: 11, color: isTop ? 'var(--accent)' : 'var(--muted)' }}>{(prob * 100).toFixed(2)}%</span>
      </div>
      <div style={{ height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${prob * 100}%`,
          background: isTop ? 'var(--accent)' : 'var(--surface-muted)',
          borderRadius: 3,
          border: `1px solid ${isTop ? 'var(--accent-muted)' : 'var(--border)'}`,
          transition: 'width 0.4s var(--easing)',
        }} />
      </div>
    </div>
  )
}

export default function RuntimeConsole({ modelId, quantizationId, dataset }: Props) {
  const [precision, setPrecision] = useState<'fp32' | 'int8'>(quantizationId ? 'int8' : 'fp32')
  const [health, setHealth] = useState<RuntimeHealthResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [windowSize, setWindowSize] = useState(64)
  const [signalInput, setSignalInput] = useState<'zeros' | 'sin' | 'noise'>('sin')
  const [inferencing, setInferencing] = useState(false)

  const healthPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchHealth = useCallback(async () => {
    try {
      const h = await api.runtime.health()
      setHealth(h)
    } catch {
      // runtime may be offline
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    healthPollRef.current = setInterval(fetchHealth, 5000)
    return () => { if (healthPollRef.current) clearInterval(healthPollRef.current) }
  }, [fetchHealth])

  async function loadRuntime() {
    setLoading(true)
    setError(null)
    try {
      const req = precision === 'int8' && quantizationId
        ? { quantization_id: quantizationId }
        : { model_id: modelId ?? undefined }
      const h = await api.runtime.load(req)
      setHealth(h)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Load failed')
    } finally {
      setLoading(false)
    }
  }

  function buildWindow(): number[] {
    const n = windowSize
    if (signalInput === 'zeros') return Array(n).fill(0)
    if (signalInput === 'sin') {
      return Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * 5 * (i / 100)))
    }
    return Array.from({ length: n }, () => (Math.random() - 0.5) * 2)
  }

  async function runInference() {
    setInferencing(true)
    setError(null)
    try {
      const values = buildWindow()
      const pred = await api.runtime.predict({ values })
      setPrediction(pred)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Inference failed')
    } finally {
      setInferencing(false)
    }
  }

  const isReady = health?.status === 'ready'
  const isLoaded = health?.model_id != null

  function statusBadge(status: string | undefined) {
    if (!status) return <span className="sq-badge sq-badge-neu">OFFLINE</span>
    const map: Record<string, string> = { ready: 'sq-badge-ok', loading: 'sq-badge-run', error: 'sq-badge-err', offline: 'sq-badge-neu' }
    return <span className={`sq-badge ${map[status] ?? 'sq-badge-neu'}`}>{status.toUpperCase()}</span>
  }

  return (
    <div style={{ padding: '0 0 48px' }}>
      <div className="sq-head">
        <div className="sq-stage-tag">05 — RUNTIME</div>
        <h2 className="sq-title">SQRuntime Console</h2>
        <p className="sq-desc">Load a model into SQRuntime and run live inference on signal windows.</p>
      </div>

      <div style={{ padding: '28px 40px 0', display: 'grid', gridTemplateColumns: '1fr 360px', gap: 24 }}>
        {/* Left: load + inference */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Runtime status */}
          <div className="sq-panel">
            <div className="sq-panel-head" style={{ justifyContent: 'space-between' }}>
              <span>RUNTIME STATUS</span>
              {statusBadge(health?.status)}
            </div>
            <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 12 }}>
              {[
                ['Status', health?.status ?? 'offline'],
                ['Loaded Model', health?.model_id ?? '—'],
                ['Precision', health?.precision ?? '—'],
                ['Device', health?.device ?? '—'],
                ['Requests', (health?.request_count ?? 0).toLocaleString()],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="sq-muted">{k}</span>
                  <span className="sq-data">{v}</span>
                </div>
              ))}
              {health?.error && <div className="sq-banner-err" style={{ marginTop: 6 }}>{health.error}</div>}
            </div>
          </div>

          {/* Load model */}
          <div className="sq-panel">
            <div className="sq-panel-head">LOAD MODEL</div>
            <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="sq-seg" role="group">
                <button
                  className={`sq-seg-opt${precision === 'fp32' ? ' sq-seg-on' : ''}`}
                  onClick={() => setPrecision('fp32')}
                  disabled={loading}
                >
                  FP32
                </button>
                <button
                  className={`sq-seg-opt${precision === 'int8' ? ' sq-seg-on' : ''}`}
                  onClick={() => quantizationId && setPrecision('int8')}
                  disabled={loading || !quantizationId}
                  title={!quantizationId ? 'Run quantization first' : undefined}
                >
                  INT8
                </button>
              </div>
              {!modelId && (
                <p style={{ fontSize: 11, color: 'var(--warning)', lineHeight: 1.5 }}>No trained model available. Complete training first.</p>
              )}
              <button
                className="sq-btn sq-btn-primary"
                onClick={loadRuntime}
                disabled={loading || !modelId}
                style={{ alignSelf: 'flex-start' }}
              >
                {loading ? 'Loading…' : 'Load into SQRuntime'}
              </button>
            </div>
          </div>

          {/* Inference */}
          <div className="sq-panel">
            <div className="sq-panel-head">LIVE INFERENCE</div>
            <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <div className="sq-field">
                  <label className="sq-label">Input Signal</label>
                  <select className="sq-input sq-select" value={signalInput} onChange={e => setSignalInput(e.target.value as typeof signalInput)} disabled={!isReady}>
                    <option value="sin">Sinusoidal (normal)</option>
                    <option value="noise">Pure Noise</option>
                    <option value="zeros">Zeros</option>
                  </select>
                </div>
                <div className="sq-field">
                  <label className="sq-label">Window Size — {windowSize}</label>
                  <input className="sq-slider" type="range" min={16} max={256} step={16} value={windowSize} onChange={e => setWindowSize(+e.target.value)} disabled={!isReady} />
                </div>
              </div>
              <button
                className="sq-btn sq-btn-primary"
                onClick={runInference}
                disabled={!isReady || inferencing}
                style={{ alignSelf: 'flex-start' }}
              >
                {inferencing ? 'Running…' : 'Run Inference'}
              </button>
              {!isLoaded && (
                <p style={{ fontSize: 11, color: 'var(--muted)' }}>Load a model first to enable inference.</p>
              )}
            </div>
          </div>

          {error && <div className="sq-banner-err">{error}</div>}

          {!health && (
            <div className="sq-empty">
              <span>Runtime not connected.</span>
              <span style={{ fontSize: 11, marginTop: 4 }}>Start the backend and load a model.</span>
            </div>
          )}
        </div>

        {/* Right: prediction result */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {prediction && (
            <>
              <div className="sq-panel">
                <div className="sq-panel-head">PREDICTION RESULT</div>
                <div className="sq-panel-body">
                  <div style={{ textAlign: 'center', padding: '16px 0 20px' }}>
                    <div className="sq-data-xl" style={{ color: 'var(--accent)' }}>{prediction.predicted_class}</div>
                    <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6 }}>
                      {(prediction.confidence * 100).toFixed(2)}% confidence · {prediction.latency_ms.toFixed(2)} ms
                    </div>
                  </div>
                  <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
                    {FAULT_LABELS.map(lbl => {
                      const prob = prediction.probabilities[lbl] ?? 0
                      return (
                        <ProbBar
                          key={lbl}
                          label={lbl}
                          prob={prob}
                          isTop={lbl === prediction.predicted_class}
                        />
                      )
                    })}
                  </div>
                </div>
              </div>

              <div className="sq-panel">
                <div className="sq-panel-head">LATENCY</div>
                <div className="sq-panel-body" style={{ padding: '14px 16px' }}>
                  <div className="sq-label">LAST INFERENCE</div>
                  <div className="sq-data-lg">{prediction.latency_ms.toFixed(3)} ms</div>
                </div>
              </div>
            </>
          )}

          {!prediction && (
            <div className="sq-empty">
              <span>No prediction yet.</span>
              <span style={{ fontSize: 11, marginTop: 4 }}>Load a model and run inference.</span>
            </div>
          )}

          {/* Telemetry */}
          {isLoaded && (
            <TelemetryPanel />
          )}
        </div>
      </div>
    </div>
  )
}

function TelemetryPanel() {
  const [telemetry, setTelemetry] = useState<Awaited<ReturnType<typeof api.runtime.telemetry>> | null>(null)

  useEffect(() => {
    const poll = async () => {
      try { setTelemetry(await api.runtime.telemetry()) } catch {}
    }
    poll()
    const id = setInterval(poll, 3000)
    return () => clearInterval(id)
  }, [])

  if (!telemetry) return null

  return (
    <div className="sq-panel">
      <div className="sq-panel-head">TELEMETRY</div>
      <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 12 }}>
        {[
          ['Requests', telemetry.request_count.toLocaleString()],
          ['Success', telemetry.success_count.toLocaleString()],
          ['Errors', telemetry.error_count.toLocaleString()],
          ['P50', telemetry.p50_latency_ms != null ? `${telemetry.p50_latency_ms.toFixed(2)} ms` : '—'],
          ['P95', telemetry.p95_latency_ms != null ? `${telemetry.p95_latency_ms.toFixed(2)} ms` : '—'],
          ['P99', telemetry.p99_latency_ms != null ? `${telemetry.p99_latency_ms.toFixed(2)} ms` : '—'],
          ['Mean', telemetry.mean_latency_ms != null ? `${telemetry.mean_latency_ms.toFixed(2)} ms` : '—'],
        ].map(([k, v]) => (
          <div key={k} style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span className="sq-muted">{k}</span>
            <span className="sq-data">{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
