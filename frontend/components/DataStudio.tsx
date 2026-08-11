'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { api, type DatasetGenerateRequest, type DatasetResponse } from '@/lib/api'

interface Props {
  onDatasetReady: (ds: DatasetResponse) => void
  existingDataset: DatasetResponse | null
}

const FAULT_LABELS: Record<string, string> = {
  noise:            'Gaussian Noise',
  drift:            'Amplitude Drift',
  dropout:          'Sensor Dropout',
  clipping:         'Signal Clipping',
  timestamp_gap:    'Timestamp Gap',
  sampling_jitter:  'Sampling Jitter',
}

function buildPreviewPath(duration: number, frequency: number, amplitude: number, noiseFrac: number): string {
  const W = 600, H = 120, PAD = 10
  const drawH = H - PAD * 2
  const n = 300
  const pts: string[] = []
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * duration
    const v = amplitude * Math.sin(2 * Math.PI * frequency * t)
    const noise = noiseFrac * amplitude * (Math.random() * 2 - 1) * 0.3
    const x = (i / (n - 1)) * W
    const y = PAD + (1 - (v + noise + amplitude) / (2 * amplitude)) * drawH
    pts.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
  }
  return pts.join(' ')
}

export default function DataStudio({ onDatasetReady, existingDataset }: Props) {
  const [name, setName] = useState('sensor-run-001')
  const [seed, setSeed] = useState(42)
  const [signalType, setSignalType] = useState<'sinusoidal' | 'composite' | 'trend'>('sinusoidal')
  const [duration, setDuration] = useState(10)
  const [samplingRate, setSamplingRate] = useState(100)
  const [amplitude, setAmplitude] = useState(1.0)
  const [frequency, setFrequency] = useState(5.0)
  const [windowSize, setWindowSize] = useState(64)
  const [faults, setFaults] = useState<Record<string, boolean>>({
    noise: false,
    drift: false,
    dropout: false,
    clipping: false,
    timestamp_gap: false,
    sampling_jitter: false,
  })

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DatasetResponse | null>(existingDataset)

  const previewSeed = useRef(0)
  const [previewPath, setPreviewPath] = useState('')

  useEffect(() => {
    previewSeed.current++
    const path = buildPreviewPath(duration, frequency, amplitude, faults.noise ? 0.3 : 0.05)
    setPreviewPath(path)
  }, [duration, frequency, amplitude, faults.noise])

  const toggleFault = useCallback((key: string) => {
    setFaults(f => ({ ...f, [key]: !f[key] }))
  }, [])

  async function generate() {
    setLoading(true)
    setError(null)
    try {
      const req: DatasetGenerateRequest = {
        name,
        seed,
        signal: { type: signalType, duration, sampling_rate: samplingRate, amplitude, frequency },
        faults: Object.fromEntries(
          Object.entries(faults)
            .filter(([, enabled]) => enabled)
            .map(([key]) => [key, { enabled: true }])
        ),
        window_size: windowSize,
      }
      const ds = await api.datasets.generate(req)
      setResult(ds)
      onDatasetReady(ds)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: '0 0 48px' }}>
      <div className="sq-head">
        <div className="sq-stage-tag">01 — DATA</div>
        <h2 className="sq-title">Signal Studio</h2>
        <p className="sq-desc">Configure and generate a deterministic synthetic sensor dataset with controlled fault injection.</p>
      </div>

      <div style={{ padding: '28px 40px 0', display: 'grid', gridTemplateColumns: '1fr 380px', gap: 24 }}>
        {/* Left: controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Basic config */}
          <div className="sq-panel">
            <div className="sq-panel-head">SIGNAL CONFIGURATION</div>
            <div className="sq-panel-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <div className="sq-field" style={{ gridColumn: '1/-1' }}>
                <label className="sq-label">Dataset Name</label>
                <input className="sq-input sq-input-mono" value={name} onChange={e => setName(e.target.value)} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Seed</label>
                <input className="sq-input sq-input-mono" type="number" value={seed} onChange={e => setSeed(+e.target.value)} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Signal Type</label>
                <select className="sq-input sq-select" value={signalType} onChange={e => setSignalType(e.target.value as typeof signalType)}>
                  <option value="sinusoidal">Sinusoidal</option>
                  <option value="composite">Composite</option>
                  <option value="trend">Trend</option>
                </select>
              </div>
              <div className="sq-field">
                <label className="sq-label">Duration (s) — {duration}</label>
                <input className="sq-slider" type="range" min={2} max={60} value={duration} onChange={e => setDuration(+e.target.value)} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Sampling Rate — {samplingRate} Hz</label>
                <input className="sq-slider" type="range" min={10} max={500} step={10} value={samplingRate} onChange={e => setSamplingRate(+e.target.value)} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Amplitude — {amplitude.toFixed(1)}</label>
                <input className="sq-slider" type="range" min={0.1} max={5} step={0.1} value={amplitude} onChange={e => setAmplitude(+e.target.value)} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Frequency — {frequency.toFixed(1)} Hz</label>
                <input className="sq-slider" type="range" min={0.5} max={50} step={0.5} value={frequency} onChange={e => setFrequency(+e.target.value)} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Window Size — {windowSize}</label>
                <input className="sq-slider" type="range" min={16} max={256} step={16} value={windowSize} onChange={e => setWindowSize(+e.target.value)} />
              </div>
            </div>
          </div>

          {/* Fault injection */}
          <div className="sq-panel">
            <div className="sq-panel-head">FAULT INJECTION</div>
            <div className="sq-panel-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {Object.entries(FAULT_LABELS).map(([key, label]) => (
                <label key={key} className="sq-toggle">
                  <span className={`sq-toggle-track${faults[key] ? ' sq-on' : ''}`} onClick={() => toggleFault(key)}>
                    <span className="sq-toggle-thumb" />
                  </span>
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </div>

          {error && <div className="sq-banner-err">{error}</div>}

          <button
            className="sq-btn sq-btn-primary sq-btn-lg"
            onClick={generate}
            disabled={loading}
            style={{ alignSelf: 'flex-start' }}
          >
            {loading ? 'Generating…' : 'Generate Dataset'}
          </button>
        </div>

        {/* Right: preview + result */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Waveform preview */}
          <div className="sq-panel">
            <div className="sq-panel-head">WAVEFORM PREVIEW</div>
            <div className="sq-panel-body" style={{ padding: 0 }}>
              <svg
                viewBox="0 0 600 120"
                style={{ width: '100%', height: 120, display: 'block', background: 'var(--surface-muted)' }}
                preserveAspectRatio="none"
              >
                {/* Zero line */}
                <line x1={0} y1={60} x2={600} y2={60} stroke="var(--border)" strokeWidth={1} />
                {/* Signal */}
                {previewPath && (
                  <path d={previewPath} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
                )}
                {/* Fault overlay if noise enabled */}
                {faults.noise && (
                  <rect x={0} y={0} width={600} height={120} fill="var(--warning)" opacity={0.06} />
                )}
              </svg>
              <div style={{ padding: '8px 14px', display: 'flex', gap: 16, borderTop: '1px solid var(--border)' }}>
                <span className="sq-data" style={{ fontSize: 11 }}>~{Math.round(duration * samplingRate)} samples</span>
                <span className="sq-data" style={{ fontSize: 11 }}>
                  {Object.values(faults).filter(Boolean).length} fault type{Object.values(faults).filter(Boolean).length !== 1 ? 's' : ''} active
                </span>
              </div>
            </div>
          </div>

          {/* Result specimen */}
          {result && (
            <div className="sq-panel">
              <div className="sq-panel-head" style={{ justifyContent: 'space-between' }}>
                <span>DATASET ARTIFACT</span>
                <span className="sq-badge sq-badge-ok">READY</span>
              </div>
              <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {[
                  ['ID', result.human_id],
                  ['Samples', result.sample_count.toLocaleString()],
                  ['Windows', result.window_count.toLocaleString()],
                  ['Faults', result.fault_count.toLocaleString()],
                  ['Signal', result.signal_type],
                  ['Duration', `${result.duration}s`],
                  ['Valid', result.validation.valid ? 'Yes' : 'No'],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                    <span className="sq-muted">{k}</span>
                    <span className="sq-data">{v}</span>
                  </div>
                ))}
                {result.validation.fault_types_present.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                    {result.validation.fault_types_present.map(ft => (
                      <span key={ft} className="sq-badge sq-badge-warn">{ft}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {!result && !loading && (
            <div className="sq-empty">
              <span>No dataset generated yet.</span>
              <span style={{ fontSize: 11, marginTop: 4 }}>Configure parameters and click Generate Dataset.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
