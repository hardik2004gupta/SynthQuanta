'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { api, type TrainingRunRequest, type ExperimentResponse, type EpochMetrics, type DatasetResponse } from '@/lib/api'

interface Props {
  datasetId: string
  dataset: DatasetResponse | null
  onTrainComplete: (experimentId: string, modelId: string) => void
  existingExperimentId: string | null
}

type Method = 'full' | 'lora' | 'qlora'

const ARCH_LAYERS = ['Input (window_size, 1)', '1D Temporal Embedding', 'Transformer Encoder ×N', 'Pooling', 'Classification Head (7 classes)']

function LossChart({ history }: { history: EpochMetrics[] }) {
  if (history.length === 0) return null

  const W = 380, H = 140, PL = 40, PB = 24, PR = 12, PT = 12
  const cW = W - PL - PR
  const cH = H - PT - PB

  const allLoss = history.flatMap(h => [h.train_loss, h.val_loss]).filter(v => isFinite(v) && v > 0)
  if (allLoss.length === 0) return null

  const maxL = Math.max(...allLoss) * 1.1
  const minL = 0

  function toX(i: number) { return PL + (i / Math.max(1, history.length - 1)) * cW }
  function toY(v: number) { return PT + (1 - (v - minL) / (maxL - minL)) * cH }

  const trainPath = history.map((h, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(h.train_loss).toFixed(1)}`).join(' ')
  const valPath = history.map((h, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(h.val_loss).toFixed(1)}`).join(' ')

  const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => minL + f * (maxL - minL))

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H }}>
      {ticks.map(t => (
        <g key={t}>
          <line x1={PL} y1={toY(t)} x2={W - PR} y2={toY(t)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={PL - 4} y={toY(t) + 3} textAnchor="end" fontSize={8} fill="var(--muted)">{t.toFixed(2)}</text>
        </g>
      ))}
      {history.map((_, i) => (
        <text key={i} x={toX(i)} y={H - 4} textAnchor="middle" fontSize={8} fill="var(--muted)">{i + 1}</text>
      ))}
      <path d={trainPath} fill="none" stroke="var(--accent)" strokeWidth={2} />
      <path d={valPath} fill="none" stroke="var(--success)" strokeWidth={1.5} strokeDasharray="4 2" />
      <g transform={`translate(${PL + 4},${PT + 4})`}>
        <line x1={0} y1={5} x2={14} y2={5} stroke="var(--accent)" strokeWidth={2} />
        <text x={18} y={8} fontSize={8} fill="var(--fg-2)">Train</text>
        <line x1={44} y1={5} x2={58} y2={5} stroke="var(--success)" strokeWidth={1.5} strokeDasharray="4 2" />
        <text x={62} y={8} fontSize={8} fill="var(--fg-2)">Val</text>
      </g>
    </svg>
  )
}

function ArchDiagram() {
  const W = 280, boxH = 30, gap = 12
  const totalH = ARCH_LAYERS.length * (boxH + gap) - gap + 20
  return (
    <svg viewBox={`0 0 ${W} ${totalH}`} style={{ width: '100%', height: totalH }}>
      {ARCH_LAYERS.map((layer, i) => {
        const y = i * (boxH + gap)
        const isKey = i === 2
        return (
          <g key={i}>
            {i > 0 && (
              <line
                x1={W / 2} y1={y - gap + 2}
                x2={W / 2} y2={y - 2}
                stroke="var(--border-strong)" strokeWidth={1}
                markerEnd="url(#arr)"
              />
            )}
            <rect
              x={20} y={y} width={W - 40} height={boxH} rx={3}
              fill={isKey ? 'var(--accent-light)' : 'var(--surface-muted)'}
              stroke={isKey ? 'var(--accent-muted)' : 'var(--border)'}
              strokeWidth={1}
            />
            <text
              x={W / 2} y={y + boxH / 2 + 4}
              textAnchor="middle" fontSize={10}
              fill={isKey ? 'var(--accent)' : 'var(--fg-2)'}
              fontWeight={isKey ? 600 : 400}
            >
              {layer}
            </text>
          </g>
        )
      })}
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX={5} refY={5} markerWidth={5} markerHeight={5} orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--border-strong)" />
        </marker>
      </defs>
    </svg>
  )
}

export default function TrainLab({ datasetId, dataset, onTrainComplete, existingExperimentId }: Props) {
  const [method, setMethod] = useState<Method>('lora')
  const [epochs, setEpochs] = useState(10)
  const [batchSize, setBatchSize] = useState(32)
  const [lr, setLr] = useState(0.001)
  const [seed, setSeed] = useState(42)
  const [loraRank, setLoraRank] = useState(4)
  const [loraAlpha, setLoraAlpha] = useState(8)
  const [embDim, setEmbDim] = useState(64)
  const [numLayers, setNumLayers] = useState(2)
  const [numHeads, setNumHeads] = useState(4)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [experiment, setExperiment] = useState<ExperimentResponse | null>(null)
  const [history, setHistory] = useState<EpochMetrics[]>([])

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const pollExperiment = useCallback(async (id: string) => {
    try {
      const exp = await api.training.get(id)
      setExperiment(exp)
      setHistory(exp.training_history ?? [])
      if (exp.status === 'COMPLETED' && exp.model_id) {
        stopPoll()
        onTrainComplete(exp.experiment_id, exp.model_id)
      } else if (exp.status === 'FAILED') {
        stopPoll()
        setError(exp.error_message ?? 'Training failed')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Poll error')
      stopPoll()
    }
  }, [onTrainComplete, stopPoll])

  useEffect(() => {
    if (existingExperimentId && !experiment) {
      pollExperiment(existingExperimentId)
    }
  }, [existingExperimentId])

  useEffect(() => () => stopPoll(), [stopPoll])

  async function startTraining() {
    setLoading(true)
    setError(null)
    setHistory([])
    try {
      const req: TrainingRunRequest = {
        dataset_id: datasetId,
        name: `exp-${Date.now()}`,
        method,
        epochs,
        batch_size: batchSize,
        learning_rate: lr,
        weight_decay: 0.01,
        seed,
        lora: { rank: loraRank, alpha: loraAlpha, dropout: 0.1, target_modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj'] },
        model: { architecture: 'SensorTransformer', window_size: 64, embedding_dim: embDim, num_layers: numLayers, num_heads: numHeads, ffn_dim: embDim * 2, dropout: 0.1 },
      }
      const resp = await api.training.run(req)
      setLoading(false)
      pollRef.current = setInterval(() => pollExperiment(resp.experiment_id), 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Launch failed')
      setLoading(false)
    }
  }

  const isRunning = experiment?.status === 'RUNNING' || loading
  const isComplete = experiment?.status === 'COMPLETED'
  const isFailed = experiment?.status === 'FAILED'

  function statusBadge() {
    if (!experiment) return null
    if (isComplete) return <span className="sq-badge sq-badge-ok">COMPLETED</span>
    if (isFailed)   return <span className="sq-badge sq-badge-err">FAILED</span>
    if (isRunning)  return <span className="sq-badge sq-badge-run">RUNNING</span>
    return <span className="sq-badge sq-badge-neu">{experiment.status}</span>
  }

  return (
    <div style={{ padding: '0 0 48px' }}>
      <div className="sq-head">
        <div className="sq-stage-tag">02 — TRAIN</div>
        <h2 className="sq-title">Adapter Lab</h2>
        <p className="sq-desc">Fine-tune SensorTransformer using LoRA, QLoRA, or full fine-tuning on the generated dataset.</p>
        {dataset && <p className="sq-desc" style={{ marginTop: 6 }}>Dataset: <span className="sq-data">{dataset.human_id}</span> · {dataset.sample_count.toLocaleString()} samples</p>}
      </div>

      <div style={{ padding: '28px 40px 0', display: 'grid', gridTemplateColumns: '1fr 340px', gap: 24 }}>
        {/* Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Method */}
          <div className="sq-panel">
            <div className="sq-panel-head">FINE-TUNING METHOD</div>
            <div className="sq-panel-body">
              <div className="sq-seg" role="group">
                {(['full', 'lora', 'qlora'] as Method[]).map(m => (
                  <button
                    key={m}
                    className={`sq-seg-opt${method === m ? ' sq-seg-on' : ''}`}
                    onClick={() => setMethod(m)}
                    disabled={isRunning}
                  >
                    {m === 'full' ? 'FULL' : m === 'lora' ? 'LoRA' : 'QLoRA'}
                  </button>
                ))}
              </div>
              {method === 'qlora' && (
                <p style={{ fontSize: 11, color: 'var(--warning)', marginTop: 10, lineHeight: 1.5 }}>
                  QLoRA requires compatible hardware. If unavailable, the job will FAIL with a diagnostic — never silently substitute.
                </p>
              )}
            </div>
          </div>

          {/* Training config */}
          <div className="sq-panel">
            <div className="sq-panel-head">TRAINING CONFIG</div>
            <div className="sq-panel-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <div className="sq-field">
                <label className="sq-label">Epochs — {epochs}</label>
                <input className="sq-slider" type="range" min={1} max={50} value={epochs} onChange={e => setEpochs(+e.target.value)} disabled={isRunning} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Batch Size — {batchSize}</label>
                <input className="sq-slider" type="range" min={8} max={128} step={8} value={batchSize} onChange={e => setBatchSize(+e.target.value)} disabled={isRunning} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Learning Rate — {lr.toExponential(0)}</label>
                <input className="sq-slider" type="range" min={-5} max={-2} step={0.5} value={Math.log10(lr)} onChange={e => setLr(Math.pow(10, +e.target.value))} disabled={isRunning} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Seed</label>
                <input className="sq-input sq-input-mono" type="number" value={seed} onChange={e => setSeed(+e.target.value)} disabled={isRunning} />
              </div>
            </div>
          </div>

          {/* LoRA config */}
          {(method === 'lora' || method === 'qlora') && (
            <div className="sq-panel">
              <div className="sq-panel-head">LoRA PARAMETERS</div>
              <div className="sq-panel-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <div className="sq-field">
                  <label className="sq-label">Rank — {loraRank}</label>
                  <input className="sq-slider" type="range" min={1} max={32} value={loraRank} onChange={e => setLoraRank(+e.target.value)} disabled={isRunning} />
                </div>
                <div className="sq-field">
                  <label className="sq-label">Alpha — {loraAlpha}</label>
                  <input className="sq-slider" type="range" min={1} max={64} value={loraAlpha} onChange={e => setLoraAlpha(+e.target.value)} disabled={isRunning} />
                </div>
              </div>
            </div>
          )}

          {/* Model config */}
          <div className="sq-panel">
            <div className="sq-panel-head">ARCHITECTURE</div>
            <div className="sq-panel-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <div className="sq-field">
                <label className="sq-label">Embedding Dim — {embDim}</label>
                <input className="sq-slider" type="range" min={16} max={256} step={16} value={embDim} onChange={e => setEmbDim(+e.target.value)} disabled={isRunning} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Layers — {numLayers}</label>
                <input className="sq-slider" type="range" min={1} max={8} value={numLayers} onChange={e => setNumLayers(+e.target.value)} disabled={isRunning} />
              </div>
              <div className="sq-field">
                <label className="sq-label">Heads — {numHeads}</label>
                <input className="sq-slider" type="range" min={1} max={8} value={numHeads} onChange={e => setNumHeads(+e.target.value)} disabled={isRunning} />
              </div>
            </div>
          </div>

          {error && <div className="sq-banner-err">{error}</div>}

          <button
            className="sq-btn sq-btn-primary sq-btn-lg"
            onClick={startTraining}
            disabled={isRunning || isComplete}
            style={{ alignSelf: 'flex-start' }}
          >
            {isRunning ? 'Training…' : isComplete ? 'Training Complete' : 'Start Training'}
          </button>
        </div>

        {/* Right: architecture + loss */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="sq-panel">
            <div className="sq-panel-head">SENSOR TRANSFORMER</div>
            <div className="sq-panel-body" style={{ padding: '12px 16px' }}>
              <ArchDiagram />
            </div>
          </div>

          {experiment && (
            <div className="sq-panel">
              <div className="sq-panel-head" style={{ justifyContent: 'space-between' }}>
                <span>EXPERIMENT</span>
                {statusBadge()}
              </div>
              <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                  <span className="sq-muted">ID</span>
                  <span className="sq-data">{experiment.human_id}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                  <span className="sq-muted">Method</span>
                  <span className="sq-data">{experiment.method}</span>
                </div>
                {experiment.metrics && typeof experiment.metrics === 'object' && 'macro_f1' in experiment.metrics && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                    <span className="sq-muted">F1</span>
                    <span className="sq-data">{((experiment.metrics as Record<string, number>).macro_f1 * 100).toFixed(1)}%</span>
                  </div>
                )}
                {history.length > 0 && (
                  <>
                    <div className="sq-label" style={{ marginTop: 8 }}>LOSS TRAJECTORY</div>
                    <LossChart history={history} />
                  </>
                )}
                {isRunning && history.length === 0 && (
                  <div style={{ fontSize: 11, color: 'var(--muted)', padding: '12px 0', textAlign: 'center' }}>
                    Waiting for first epoch…
                  </div>
                )}
              </div>
            </div>
          )}

          {!experiment && (
            <div className="sq-empty">
              <span>No experiment running.</span>
              <span style={{ fontSize: 11, marginTop: 4 }}>Configure and click Start Training.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
