'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { api, type EvaluationResponse, type ShiftScenario } from '@/lib/api'

interface Props {
  experimentId: string
  modelId: string | null
  onEvalComplete: (evaluationId: string, evaluation: EvaluationResponse) => void
  existingEvaluationId: string | null
  existingEvaluation: EvaluationResponse | null
}

const FAULT_CLASSES = ['NORMAL', 'NOISE', 'DRIFT', 'DROPOUT', 'CLIPPING', 'TIMESTAMP_GAP', 'SAMPLING_JITTER']
const SCENARIO_LABELS: Record<string, string> = {
  noise_shift: 'Noise σ',
  amplitude_shift: 'Amplitude',
  frequency_shift: 'Frequency',
  severity_shift: 'Severity',
  compound_fault_shift: 'Compound Fault',
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  if (!matrix || matrix.length === 0) return null
  const n = matrix.length
  const labels = FAULT_CLASSES.slice(0, n)
  const max = Math.max(...matrix.flat())

  const CELL = 36, LABEL_W = 72, LABEL_H = 72
  const W = LABEL_W + n * CELL
  const H = LABEL_H + n * CELL

  const [hoveredCell, setHoveredCell] = useState<[number, number] | null>(null)

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: W, height: H, display: 'block', minWidth: W }}>
        {/* Row labels */}
        {labels.map((lbl, i) => (
          <text key={i} x={LABEL_W - 4} y={LABEL_H + i * CELL + CELL / 2 + 4} textAnchor="end" fontSize={7} fill="var(--muted)" fontFamily="var(--font-mono)">
            {lbl}
          </text>
        ))}
        {/* Col labels (rotated) */}
        {labels.map((lbl, j) => (
          <text key={j} x={LABEL_W + j * CELL + CELL / 2} y={LABEL_H - 4} textAnchor="middle" fontSize={7} fill="var(--muted)" fontFamily="var(--font-mono)" transform={`rotate(-45,${LABEL_W + j * CELL + CELL / 2},${LABEL_H - 4})`}>
            {lbl}
          </text>
        ))}
        {/* Cells */}
        {matrix.map((row, i) =>
          row.map((val, j) => {
            const intensity = max > 0 ? val / max : 0
            const isDiag = i === j
            const isHov = hoveredCell?.[0] === i && hoveredCell?.[1] === j
            const fill = isDiag
              ? `rgba(28,68,190,${0.1 + intensity * 0.85})`
              : `rgba(0,0,0,${intensity * 0.25})`
            const textColor = isDiag && intensity > 0.5 ? '#fff' : 'var(--fg)'
            return (
              <g key={`${i}-${j}`} onMouseEnter={() => setHoveredCell([i, j])} onMouseLeave={() => setHoveredCell(null)}>
                <rect
                  x={LABEL_W + j * CELL} y={LABEL_H + i * CELL}
                  width={CELL} height={CELL}
                  fill={fill}
                  stroke={isHov ? 'var(--accent)' : 'var(--border)'}
                  strokeWidth={isHov ? 1.5 : 0.5}
                />
                <text
                  x={LABEL_W + j * CELL + CELL / 2}
                  y={LABEL_H + i * CELL + CELL / 2 + 3}
                  textAnchor="middle"
                  fontSize={val > 999 ? 7 : 8}
                  fill={textColor}
                  fontFamily="var(--font-mono)"
                >
                  {val}
                </text>
              </g>
            )
          })
        )}
        {/* Hover label */}
        {hoveredCell && (
          <text x={LABEL_W} y={H - 2} fontSize={8} fill="var(--accent)" fontFamily="var(--font-mono)">
            {labels[hoveredCell[0]]} → {labels[hoveredCell[1]]}: {matrix[hoveredCell[0]][hoveredCell[1]]}
          </text>
        )}
      </svg>
    </div>
  )
}

function RobustnessChart({ scenarios }: { scenarios: ShiftScenario[] }) {
  if (!scenarios || scenarios.length === 0) return null

  const W = 400, H = 180, PL = 100, PB = 24, PR = 20, PT = 16
  const cW = W - PL - PR
  const cH = H - PT - PB

  const yTicks = [0, 0.25, 0.5, 0.75, 1.0]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H }}>
      {yTicks.map(t => (
        <g key={t}>
          <line x1={PL} y1={PT + (1 - t) * cH} x2={W - PR} y2={PT + (1 - t) * cH} stroke="var(--border)" strokeWidth={0.5} />
          <text x={PL - 4} y={PT + (1 - t) * cH + 3} textAnchor="end" fontSize={8} fill="var(--muted)">{(t * 100).toFixed(0)}%</text>
        </g>
      ))}
      {scenarios.map((sc, i) => {
        const x = PL + (i / Math.max(1, scenarios.length - 1)) * cW
        const yCiid = PT + (1 - sc.iid_macro_f1) * cH
        const yShift = PT + (1 - sc.shifted_macro_f1) * cH
        return (
          <g key={sc.scenario}>
            <circle cx={x} cy={yCiid} r={4} fill="var(--accent)" />
            <circle cx={x} cy={yShift} r={4} fill="var(--error)" />
            <line x1={x} y1={yCiid} x2={x} y2={yShift} stroke="var(--border-strong)" strokeWidth={1} strokeDasharray="3 2" />
            <text x={x} y={H - 2} textAnchor="middle" fontSize={7} fill="var(--muted)">
              {(SCENARIO_LABELS[sc.scenario] ?? sc.scenario).split(' ')[0]}
            </text>
          </g>
        )
      })}
      {/* Legend */}
      <circle cx={PL + 4} cy={PT + 8} r={3} fill="var(--accent)" />
      <text x={PL + 10} y={PT + 11} fontSize={8} fill="var(--fg-2)">IID F1</text>
      <circle cx={PL + 56} cy={PT + 8} r={3} fill="var(--error)" />
      <text x={PL + 62} y={PT + 11} fontSize={8} fill="var(--fg-2)">Shifted F1</text>
    </svg>
  )
}

export default function EvalLab({ experimentId, modelId, onEvalComplete, existingEvaluationId, existingEvaluation }: Props) {
  const [includeShift, setIncludeShift] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(existingEvaluation)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const pollEval = useCallback(async (id: string) => {
    try {
      const ev = await api.evaluation.get(id)
      setEvaluation(ev)
      if (ev.status === 'COMPLETED') {
        stopPoll()
        onEvalComplete(ev.evaluation_id, ev)
      } else if (ev.status === 'FAILED') {
        stopPoll()
        setError('Evaluation failed')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Poll error')
      stopPoll()
    }
  }, [onEvalComplete, stopPoll])

  useEffect(() => {
    if (existingEvaluationId && !existingEvaluation) {
      pollEval(existingEvaluationId)
    }
  }, [existingEvaluationId])

  useEffect(() => () => stopPoll(), [stopPoll])

  async function runEval() {
    setLoading(true)
    setError(null)
    try {
      const resp = await api.evaluation.run({ experiment_id: experimentId, include_shift: includeShift })
      setLoading(false)
      pollRef.current = setInterval(() => pollEval(resp.evaluation_id), 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Launch failed')
      setLoading(false)
    }
  }

  const isRunning = loading || evaluation?.status === 'RUNNING'
  const isComplete = evaluation?.status === 'COMPLETED'
  const iidMetrics = evaluation?.results?.iid?.metrics
  const shiftScenarios = evaluation?.results?.distribution_shift ?? []
  const confMatrix = iidMetrics && 'confusion_matrix' in iidMetrics ? (iidMetrics as unknown as { confusion_matrix: number[][] }).confusion_matrix : null
  const perClass = iidMetrics?.per_class

  return (
    <div style={{ padding: '0 0 48px' }}>
      <div className="sq-head">
        <div className="sq-stage-tag">03 — EVALUATE</div>
        <h2 className="sq-title">Robustness Lab</h2>
        <p className="sq-desc">Evaluate model quality on IID data and under 5 distribution-shift scenarios.</p>
      </div>

      <div style={{ padding: '28px 40px 0', display: 'flex', flexDirection: 'column', gap: 24 }}>
        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
          <label className="sq-toggle">
            <span className={`sq-toggle-track${includeShift ? ' sq-on' : ''}`} onClick={() => setIncludeShift(v => !v)}>
              <span className="sq-toggle-thumb" />
            </span>
            <span>Include Distribution-Shift Evaluation</span>
          </label>
          <button
            className="sq-btn sq-btn-primary"
            onClick={runEval}
            disabled={isRunning || isComplete}
          >
            {isRunning ? 'Evaluating…' : isComplete ? 'Evaluation Complete' : 'Run Evaluation'}
          </button>
          {evaluation && !isRunning && (
            <span className={`sq-badge ${isComplete ? 'sq-badge-ok' : 'sq-badge-err'}`}>
              {evaluation.status}
            </span>
          )}
        </div>

        {error && <div className="sq-banner-err">{error}</div>}
        {isRunning && !iidMetrics && (
          <div className="sq-banner-info">Evaluation running — this may take a few minutes for all shift scenarios.</div>
        )}

        {/* IID metrics */}
        {iidMetrics && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 14 }}>
            {[
              ['Macro F1', `${(iidMetrics.macro_f1 * 100).toFixed(2)}%`],
              ['Weighted F1', `${(iidMetrics.weighted_f1 * 100).toFixed(2)}%`],
              ['False Alarm Rate', `${(iidMetrics.false_alarm_rate * 100).toFixed(2)}%`],
              ['IID Samples', iidMetrics.n_samples.toLocaleString()],
            ].map(([label, val]) => (
              <div key={label} className="sq-panel">
                <div className="sq-panel-body" style={{ padding: '14px 16px' }}>
                  <div className="sq-label">{label}</div>
                  <div className="sq-data-lg">{val}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: confMatrix ? '1fr 1fr' : '1fr', gap: 24 }}>
          {/* Confusion matrix */}
          {confMatrix && (
            <div className="sq-panel">
              <div className="sq-panel-head">CONFUSION MATRIX</div>
              <div className="sq-panel-body" style={{ overflowX: 'auto' }}>
                <ConfusionMatrix matrix={confMatrix} />
              </div>
            </div>
          )}

          {/* Per-class breakdown */}
          {perClass && (
            <div className="sq-panel">
              <div className="sq-panel-head">PER-CLASS F1</div>
              <div className="sq-panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {Object.entries(perClass).map(([cls, m]) => (
                  <div key={cls}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, fontSize: 11 }}>
                      <span className="sq-mono">{cls}</span>
                      <span className="sq-data">{(m.f1 * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{
                        height: '100%',
                        width: `${m.f1 * 100}%`,
                        background: m.f1 > 0.7 ? 'var(--success)' : m.f1 > 0.4 ? 'var(--accent)' : 'var(--error)',
                        borderRadius: 3,
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Distribution shift */}
        {shiftScenarios.length > 0 && (
          <div className="sq-panel">
            <div className="sq-panel-head">DISTRIBUTION SHIFT — 5 SCENARIOS</div>
            <div className="sq-panel-body">
              <RobustnessChart scenarios={shiftScenarios} />
              <div style={{ overflowX: 'auto', marginTop: 16 }}>
                <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      {['Scenario', 'IID F1', 'Shifted F1', 'Δ F1', 'Rel. Degr.'].map(h => (
                        <th key={h} style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid var(--border)', color: 'var(--muted)', fontWeight: 600, letterSpacing: '0.06em', fontSize: 10 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {shiftScenarios.map(sc => (
                      <tr key={sc.scenario}>
                        <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', fontSize: 10 }}>{SCENARIO_LABELS[sc.scenario] ?? sc.scenario}</td>
                        <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)' }}>{(sc.iid_macro_f1 * 100).toFixed(2)}%</td>
                        <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)' }}>{(sc.shifted_macro_f1 * 100).toFixed(2)}%</td>
                        <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', color: sc.absolute_degradation > 0 ? 'var(--error)' : 'var(--success)' }}>
                          {sc.absolute_degradation > 0 ? '−' : '+'}{(Math.abs(sc.absolute_degradation) * 100).toFixed(2)}%
                        </td>
                        <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)', color: 'var(--muted)' }}>
                          {(sc.relative_degradation * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {!evaluation && !loading && (
          <div className="sq-empty">
            <span>No evaluation run yet.</span>
            <span style={{ fontSize: 11, marginTop: 4 }}>Click Run Evaluation to evaluate the trained model.</span>
          </div>
        )}
      </div>
    </div>
  )
}
