'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { api, type BenchmarkResponse, type BatchResult } from '@/lib/api'

interface Props {
  modelId: string | null
  quantizationId: string | null
  onBenchmarkComplete: (benchmarkId: string, benchmark: BenchmarkResponse) => void
  existingBenchmarkId: string | null
  existingBenchmark: BenchmarkResponse | null
}

const DEFAULT_BATCH_SIZES = [1, 8, 16, 32]

function LatencyChart({ results }: { results: BatchResult[] }) {
  if (!results || results.length === 0) return null

  const W = 480, H = 180, PL = 52, PB = 40, PR = 16, PT = 16
  const cW = W - PL - PR
  const cH = H - PT - PB

  const allP95 = results.map(r => r.latency_stats.p95).filter(v => v > 0)
  const allP50 = results.map(r => r.latency_stats.p50).filter(v => v > 0)
  const maxY = Math.max(...allP95, ...allP50) * 1.15
  const minY = 0

  function toX(i: number) { return PL + (i / Math.max(1, results.length - 1)) * cW }
  function toY(v: number) { return PT + (1 - (v - minY) / (maxY - minY)) * cH }

  const p95Path = results.map((r, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(r.latency_stats.p95 ?? 0).toFixed(1)}`).join(' ')
  const p50Path = results.map((r, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(r.latency_stats.p50 ?? 0).toFixed(1)}`).join(' ')

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(f => minY + f * (maxY - minY))

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H }}>
      {yTicks.map(t => (
        <g key={t}>
          <line x1={PL} y1={toY(t)} x2={W - PR} y2={toY(t)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={PL - 4} y={toY(t) + 3} textAnchor="end" fontSize={8} fill="var(--muted)">{t.toFixed(2)}</text>
        </g>
      ))}
      <text x={PL - 30} y={PT + cH / 2} textAnchor="middle" fontSize={8} fill="var(--muted)" transform={`rotate(-90,${PL - 30},${PT + cH / 2})`}>ms</text>

      {results.map((r, i) => (
        <text key={i} x={toX(i)} y={H - 4} textAnchor="middle" fontSize={8} fill="var(--muted)">{r.batch_size}</text>
      ))}
      <text x={PL + cW / 2} y={H} textAnchor="middle" fontSize={8} fill="var(--muted)">batch size</text>

      <path d={p95Path} fill="none" stroke="var(--accent)" strokeWidth={2} />
      <path d={p50Path} fill="none" stroke="var(--success)" strokeWidth={1.5} strokeDasharray="5 3" />

      {results.map((r, i) => (
        <g key={i}>
          <circle cx={toX(i)} cy={toY(r.latency_stats.p95 ?? 0)} r={3.5} fill="var(--accent)" />
          <circle cx={toX(i)} cy={toY(r.latency_stats.p50 ?? 0)} r={3.5} fill="var(--success)" />
        </g>
      ))}

      <g transform={`translate(${PL + 4},${PT + 4})`}>
        <line x1={0} y1={5} x2={14} y2={5} stroke="var(--accent)" strokeWidth={2} />
        <text x={18} y={8} fontSize={8} fill="var(--fg-2)">P95</text>
        <line x1={44} y1={5} x2={58} y2={5} stroke="var(--success)" strokeWidth={1.5} strokeDasharray="5 3" />
        <text x={62} y={8} fontSize={8} fill="var(--fg-2)">P50</text>
      </g>
    </svg>
  )
}

function ThroughputBars({ results }: { results: BatchResult[] }) {
  if (!results || results.length === 0) return null
  const maxRps = Math.max(...results.map(r => r.throughput_rps))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {results.map(r => (
        <div key={r.batch_size}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
            <span className="sq-mono">batch={r.batch_size}</span>
            <span className="sq-data">{r.throughput_rps.toFixed(1)} req/s</span>
          </div>
          <div style={{ height: 8, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${(r.throughput_rps / maxRps) * 100}%`,
              background: 'var(--accent)',
              borderRadius: 3,
              transition: 'width 0.4s var(--easing)',
            }} />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function BenchmarkLab({ modelId, quantizationId, onBenchmarkComplete, existingBenchmarkId, existingBenchmark }: Props) {
  const [precision, setPrecision] = useState<'fp32' | 'int8'>(quantizationId ? 'int8' : 'fp32')
  const [batchSizes, setBatchSizes] = useState(DEFAULT_BATCH_SIZES)
  const [iterations, setIterations] = useState(100)
  const [warmup, setWarmup] = useState(20)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(existingBenchmark)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const pollBenchmark = useCallback(async (id: string) => {
    try {
      const b = await api.benchmarks.get(id)
      setBenchmark(b)
      if (b.status === 'COMPLETED') {
        stopPoll()
        onBenchmarkComplete(b.benchmark_id, b)
      } else if (b.status === 'FAILED') {
        stopPoll()
        setError(b.error ?? 'Benchmark failed')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Poll error')
      stopPoll()
    }
  }, [onBenchmarkComplete, stopPoll])

  useEffect(() => {
    if (existingBenchmarkId && !existingBenchmark) {
      pollBenchmark(existingBenchmarkId)
    }
  }, [existingBenchmarkId])

  useEffect(() => () => stopPoll(), [stopPoll])

  async function runBenchmark() {
    setLoading(true)
    setError(null)
    try {
      const req = precision === 'int8' && quantizationId
        ? { quantization_id: quantizationId, batch_sizes: batchSizes, iterations, warmup }
        : { model_id: modelId ?? undefined, batch_sizes: batchSizes, iterations, warmup }
      const resp = await api.benchmarks.run(req)
      setLoading(false)
      pollRef.current = setInterval(() => pollBenchmark(resp.benchmark_id), 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Launch failed')
      setLoading(false)
    }
  }

  const isRunning = loading || benchmark?.status === 'RUNNING'
  const isComplete = benchmark?.status === 'COMPLETED'
  const batchResults = benchmark?.batch_results ?? []
  const primaryResult = batchResults.find(r => r.batch_size === 1) ?? batchResults[0]

  function toggleBatchSize(n: number) {
    setBatchSizes(prev => prev.includes(n) ? prev.filter(x => x !== n) : [...prev, n].sort((a, b) => a - b))
  }

  return (
    <div style={{ padding: '0 0 48px' }}>
      <div className="sq-head">
        <div className="sq-stage-tag">06 — BENCHMARK</div>
        <h2 className="sq-title">Benchmark Lab</h2>
        <p className="sq-desc">Measure P50/P95/P99 latency, throughput, and memory across batch sizes. All numbers from actual execution.</p>
      </div>

      <div style={{ padding: '28px 40px 0', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Left: config */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="sq-panel">
            <div className="sq-panel-head">BENCHMARK CONFIG</div>
            <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Precision */}
              <div className="sq-field">
                <label className="sq-label">Model Precision</label>
                <div className="sq-seg">
                  <button className={`sq-seg-opt${precision === 'fp32' ? ' sq-seg-on' : ''}`} onClick={() => setPrecision('fp32')} disabled={isRunning}>FP32</button>
                  <button className={`sq-seg-opt${precision === 'int8' ? ' sq-seg-on' : ''}`} onClick={() => quantizationId && setPrecision('int8')} disabled={isRunning || !quantizationId}>INT8</button>
                </div>
              </div>

              {/* Batch sizes */}
              <div className="sq-field">
                <label className="sq-label">Batch Sizes</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  {DEFAULT_BATCH_SIZES.map(n => (
                    <button
                      key={n}
                      className={`sq-seg-opt${batchSizes.includes(n) ? ' sq-seg-on' : ''}`}
                      onClick={() => toggleBatchSize(n)}
                      disabled={isRunning}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              <div className="sq-field">
                <label className="sq-label">Iterations — {iterations}</label>
                <input className="sq-slider" type="range" min={20} max={500} step={20} value={iterations} onChange={e => setIterations(+e.target.value)} disabled={isRunning} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Warmup — {warmup}</label>
                <input className="sq-slider" type="range" min={5} max={50} step={5} value={warmup} onChange={e => setWarmup(+e.target.value)} disabled={isRunning} />
              </div>

              {!modelId && (
                <p style={{ fontSize: 11, color: 'var(--warning)', lineHeight: 1.5 }}>No model available. Complete training first.</p>
              )}

              <button
                className="sq-btn sq-btn-primary sq-btn-lg"
                onClick={runBenchmark}
                disabled={isRunning || !modelId || batchSizes.length === 0}
                style={{ alignSelf: 'flex-start' }}
              >
                {isRunning ? 'Benchmarking…' : 'Run Benchmark'}
              </button>

              {benchmark && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12 }}>
                  <span className="sq-muted">{benchmark.human_id}</span>
                  <span className={`sq-badge ${isComplete ? 'sq-badge-ok' : benchmark.status === 'FAILED' ? 'sq-badge-err' : 'sq-badge-run'}`}>
                    {benchmark.status}
                  </span>
                </div>
              )}
            </div>
          </div>

          {error && <div className="sq-banner-err">{error}</div>}

          {/* Primary metrics */}
          {primaryResult && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: 12 }}>
              {[
                ['P50', primaryResult.latency_stats.p50 != null ? `${primaryResult.latency_stats.p50.toFixed(3)} ms` : '—'],
                ['P95', primaryResult.latency_stats.p95 != null ? `${primaryResult.latency_stats.p95.toFixed(3)} ms` : '—'],
                ['P99', primaryResult.latency_stats.p99 != null ? `${primaryResult.latency_stats.p99.toFixed(3)} ms` : '—'],
                ['Throughput', `${primaryResult.throughput_rps.toFixed(1)} req/s`],
                ['Peak Mem', primaryResult.memory_peak_mb != null ? `${primaryResult.memory_peak_mb.toFixed(1)} MB` : '—'],
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

          {!benchmark && !loading && (
            <div className="sq-empty">
              <span>No benchmark run yet.</span>
              <span style={{ fontSize: 11, marginTop: 4 }}>Configure and run a benchmark to see actual latency measurements.</span>
            </div>
          )}
        </div>

        {/* Right: charts */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {batchResults.length > 0 && (
            <>
              <div className="sq-panel">
                <div className="sq-panel-head">LATENCY vs BATCH SIZE</div>
                <div className="sq-panel-body" style={{ padding: '12px 8px' }}>
                  <LatencyChart results={batchResults} />
                </div>
              </div>

              <div className="sq-panel">
                <div className="sq-panel-head">THROUGHPUT (req/s)</div>
                <div className="sq-panel-body">
                  <ThroughputBars results={batchResults} />
                </div>
              </div>

              <div className="sq-panel">
                <div className="sq-panel-head">BATCH RESULTS</div>
                <div className="sq-panel-body" style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        {['Batch', 'P50 ms', 'P95 ms', 'P99 ms', 'req/s', 'Status'].map(h => (
                          <th key={h} style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid var(--border)', color: 'var(--muted)', fontWeight: 600, fontSize: 10, letterSpacing: '0.05em' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.map(r => (
                        <tr key={r.batch_size} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td style={{ padding: '7px 8px', fontFamily: 'var(--font-mono)' }}>{r.batch_size}</td>
                          <td style={{ padding: '7px 8px', fontFamily: 'var(--font-mono)' }}>{r.latency_stats.p50?.toFixed(3) ?? '—'}</td>
                          <td style={{ padding: '7px 8px', fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent)' }}>{r.latency_stats.p95?.toFixed(3) ?? '—'}</td>
                          <td style={{ padding: '7px 8px', fontFamily: 'var(--font-mono)' }}>{r.latency_stats.p99?.toFixed(3) ?? '—'}</td>
                          <td style={{ padding: '7px 8px', fontFamily: 'var(--font-mono)' }}>{r.throughput_rps.toFixed(1)}</td>
                          <td style={{ padding: '7px 8px' }}>
                            <span className={`sq-badge ${r.status === 'COMPLETED' ? 'sq-badge-ok' : 'sq-badge-err'}`}>{r.status}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {batchResults.length === 0 && !loading && (
            <div className="sq-empty">
              <span>No benchmark data.</span>
              <span style={{ fontSize: 11, marginTop: 4 }}>Latency charts will appear after a benchmark run completes.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
