'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { api, type QuantizationResponse } from '@/lib/api'

interface Props {
  modelId: string
  datasetId: string | null
  onQuantizationReady: (quantizationId: string, quantization: QuantizationResponse) => void
  existingQuantizationId: string | null
  existingQuantization: QuantizationResponse | null
}

function CompressionBar({ fp32Bytes, int8Bytes }: { fp32Bytes: number; int8Bytes: number }) {
  const ratio = fp32Bytes > 0 ? int8Bytes / fp32Bytes : 0
  const fp32MB = (fp32Bytes / 1024 / 1024).toFixed(2)
  const int8MB = (int8Bytes / 1024 / 1024).toFixed(2)

  return (
    <div style={{ padding: '0 0 8px' }}>
      <div style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
          <span className="sq-mono" style={{ color: 'var(--accent)', fontWeight: 600 }}>FP32</span>
          <span className="sq-data">{fp32MB} MB</span>
        </div>
        <div style={{ height: 12, background: 'var(--accent)', borderRadius: 3, width: '100%' }} />
      </div>
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
          <span className="sq-mono" style={{ color: 'var(--success)', fontWeight: 600 }}>INT8</span>
          <span className="sq-data">{int8MB} MB</span>
        </div>
        <div style={{ height: 12, background: 'var(--surface-muted)', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${ratio * 100}%`, background: 'var(--success)', borderRadius: 3 }} />
        </div>
      </div>
      <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 10, lineHeight: 1.5 }}>
        INT8 is <strong>{(ratio * 100).toFixed(1)}%</strong> the size of FP32 — a <strong>{((1 - ratio) * 100).toFixed(1)}%</strong> reduction.
      </p>
    </div>
  )
}

export default function QuantizeLab({ modelId, datasetId, onQuantizationReady, existingQuantizationId, existingQuantization }: Props) {
  const [benchIterations, setBenchIterations] = useState(50)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [quantization, setQuantization] = useState<QuantizationResponse | null>(existingQuantization)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const pollQuant = useCallback(async (id: string) => {
    try {
      const q = await api.quantization.get(id)
      setQuantization(q)
      if (q.status === 'COMPLETED') {
        stopPoll()
        onQuantizationReady(q.quantization_id, q)
      } else if (q.status === 'FAILED') {
        stopPoll()
        setError(q.error ?? 'Quantization failed')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Poll error')
      stopPoll()
    }
  }, [onQuantizationReady, stopPoll])

  useEffect(() => {
    if (existingQuantizationId && !existingQuantization) {
      pollQuant(existingQuantizationId)
    }
  }, [existingQuantizationId])

  useEffect(() => () => stopPoll(), [stopPoll])

  async function runQuantization() {
    setLoading(true)
    setError(null)
    try {
      const resp = await api.quantization.run({
        source_model_id: modelId,
        dataset_id: datasetId ?? undefined,
        benchmark_iterations: benchIterations,
        benchmark_warmup: 10,
      })
      setLoading(false)
      pollRef.current = setInterval(() => pollQuant(resp.quantization_id), 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Launch failed')
      setLoading(false)
    }
  }

  const isRunning = loading || quantization?.status === 'RUNNING'
  const isComplete = quantization?.status === 'COMPLETED'
  const isFailed = quantization?.status === 'FAILED'
  const cmp = quantization?.comparison

  return (
    <div style={{ padding: '0 0 48px' }}>
      <div className="sq-head">
        <div className="sq-stage-tag">04 — OPTIMIZE</div>
        <h2 className="sq-title">Quantize Lab</h2>
        <p className="sq-desc">
          Compress the FP32 model to INT8 using dynamic quantization. Measures F1 delta, size reduction, and latency speedup from actual inference.
        </p>
      </div>

      <div style={{ padding: '28px 40px 0', display: 'grid', gridTemplateColumns: '1fr 380px', gap: 24 }}>
        {/* Left: controls + results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="sq-panel">
            <div className="sq-panel-head">QUANTIZATION CONFIG</div>
            <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ padding: '12px 16px', background: 'var(--surface-muted)', borderRadius: 'var(--r)', border: '1px solid var(--border)', fontSize: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span className="sq-muted">Method</span>
                  <span className="sq-data sq-mono">PyTorch Dynamic INT8</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span className="sq-muted">Path</span>
                  <span className="sq-data">FP32 → INT8</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="sq-muted">FP32 Artifact</span>
                  <span style={{ fontSize: 11, color: 'var(--success)', fontWeight: 600 }}>IMMUTABLE</span>
                </div>
              </div>
              <div className="sq-field">
                <label className="sq-label">Benchmark Iterations — {benchIterations}</label>
                <input className="sq-slider" type="range" min={10} max={200} step={10} value={benchIterations} onChange={e => setBenchIterations(+e.target.value)} disabled={isRunning} />
              </div>
            </div>
          </div>

          {error && <div className="sq-banner-err">{error}</div>}

          <button
            className="sq-btn sq-btn-primary sq-btn-lg"
            onClick={runQuantization}
            disabled={isRunning || isComplete}
            style={{ alignSelf: 'flex-start' }}
          >
            {isRunning ? 'Quantizing…' : isComplete ? 'Quantization Complete' : 'Run Quantization (FP32 → INT8)'}
          </button>

          {/* Comparison metrics */}
          {cmp && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 14 }}>
              {[
                ['FP32 F1', `${(cmp.fp32_macro_f1 * 100).toFixed(2)}%`],
                ['INT8 F1', `${(cmp.int8_macro_f1 * 100).toFixed(2)}%`],
                ['F1 Delta', `${(cmp.f1_delta * 100 >= 0 ? '+' : '')}${(cmp.f1_delta * 100).toFixed(2)}%`],
                ['Size Reduction', `${((1 - cmp.size_reduction_ratio) * 100).toFixed(1)}%`],
                ['Latency Speedup', `${cmp.latency_speedup.toFixed(2)}×`],
                ['FP32 Latency', `${cmp.fp32_latency_ms.toFixed(2)} ms`],
                ['INT8 Latency', `${cmp.int8_latency_ms.toFixed(2)} ms`],
                ['Test Windows', cmp.n_test_windows.toLocaleString()],
              ].map(([label, val]) => (
                <div key={label} className="sq-panel">
                  <div className="sq-panel-body" style={{ padding: '12px 14px' }}>
                    <div className="sq-label" style={{ marginBottom: 4 }}>{label}</div>
                    <div className="sq-data-lg" style={{ fontSize: 18 }}>{val}</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {quantization && (
            <div className="sq-panel">
              <div className="sq-panel-head" style={{ justifyContent: 'space-between' }}>
                <span>QUANTIZATION JOB</span>
                <span className={`sq-badge ${isComplete ? 'sq-badge-ok' : isFailed ? 'sq-badge-err' : 'sq-badge-run'}`}>
                  {quantization.status}
                </span>
              </div>
              <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="sq-muted">ID</span>
                  <span className="sq-data">{quantization.human_id}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="sq-muted">Backend</span>
                  <span className="sq-data">{quantization.backend ?? '—'}</span>
                </div>
                {quantization.duration_seconds && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="sq-muted">Duration</span>
                    <span className="sq-data">{quantization.duration_seconds.toFixed(1)}s</span>
                  </div>
                )}
                {isFailed && quantization.error && (
                  <div className="sq-banner-err" style={{ marginTop: 6 }}>{quantization.error}</div>
                )}
              </div>
            </div>
          )}

          {!quantization && !loading && (
            <div className="sq-empty">
              <span>No quantization run yet.</span>
              <span style={{ fontSize: 11, marginTop: 4 }}>Run quantization to produce an INT8 model artifact.</span>
            </div>
          )}
        </div>

        {/* Right: compression visualization */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="sq-panel">
            <div className="sq-panel-head">COMPRESSION VISUALIZATION</div>
            <div className="sq-panel-body">
              {cmp ? (
                <CompressionBar fp32Bytes={cmp.fp32_size_bytes} int8Bytes={cmp.int8_size_bytes} />
              ) : (
                <div style={{ padding: '24px 0', textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.8 }}>
                    <div style={{ height: 12, background: 'var(--border)', borderRadius: 3, marginBottom: 12, width: '100%' }} />
                    <div style={{ height: 12, background: 'var(--border)', borderRadius: 3, width: '45%' }} />
                    <p style={{ marginTop: 14 }}>FP32 → INT8 compression will appear after quantization completes.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="sq-panel">
            <div className="sq-panel-head">QUANTIZATION PATH</div>
            <div className="sq-panel-body">
              <svg viewBox="0 0 260 120" style={{ width: '100%', height: 120 }}>
                {/* FP32 box */}
                <rect x={10} y={20} width={90} height={44} rx={4} fill="var(--accent-light)" stroke="var(--accent)" strokeWidth={1.5} />
                <text x={55} y={40} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--accent)">FP32</text>
                <text x={55} y={54} textAnchor="middle" fontSize={9} fill="var(--muted)">32 bits/weight</text>
                {/* Arrow */}
                <line x1={100} y1={42} x2={150} y2={42} stroke="var(--border-strong)" strokeWidth={1.5} markerEnd="url(#arr2)" />
                <text x={125} y={35} textAnchor="middle" fontSize={8} fill="var(--muted)">dynamic</text>
                <text x={125} y={55} textAnchor="middle" fontSize={8} fill="var(--muted)">quantize</text>
                {/* INT8 box */}
                <rect x={154} y={20} width={90} height={44} rx={4} fill={cmp ? 'var(--success-bg)' : 'var(--surface-muted)'} stroke={cmp ? 'var(--success-border)' : 'var(--border)'} strokeWidth={1.5} />
                <text x={199} y={40} textAnchor="middle" fontSize={11} fontWeight={700} fill={cmp ? 'var(--success)' : 'var(--muted)'}>INT8</text>
                <text x={199} y={54} textAnchor="middle" fontSize={9} fill="var(--muted)">8 bits/weight</text>
                {/* Immutability note */}
                <text x={55} y={84} textAnchor="middle" fontSize={8} fill="var(--accent)">IMMUTABLE</text>
                {cmp && (
                  <text x={199} y={84} textAnchor="middle" fontSize={8} fill="var(--success)">VALIDATED</text>
                )}
                <defs>
                  <marker id="arr2" viewBox="0 0 10 10" refX={5} refY={5} markerWidth={5} markerHeight={5} orient="auto-start-reverse">
                    <path d="M0,0 L10,5 L0,10 z" fill="var(--border-strong)" />
                  </marker>
                </defs>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
