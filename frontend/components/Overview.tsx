'use client'

import type { Stage, WorkflowState } from '@/app/page'

interface Props {
  state: WorkflowState
  onNavigate: (s: Stage) => void
}

interface PipelineNode {
  id: Stage
  label: string
  sub: string
  cx: number
  cy: number
}

const NODES: PipelineNode[] = [
  { id: 'DATA',      label: 'Signal Studio',     sub: 'Synthetic Data',    cx: 120,  cy: 160 },
  { id: 'TRAIN',     label: 'Adapter Lab',        sub: 'LoRA / QLoRA',      cx: 310,  cy: 100 },
  { id: 'EVALUATE',  label: 'Robustness Lab',     sub: 'Shift Evaluation',  cx: 500,  cy: 160 },
  { id: 'OPTIMIZE',  label: 'Quantize',           sub: 'FP32 → INT8',       cx: 690,  cy: 100 },
  { id: 'RUNTIME',   label: 'SQRuntime',          sub: 'Inference Engine',  cx: 880,  cy: 160 },
  { id: 'BENCHMARK', label: 'Benchmark',          sub: 'P50/P95/P99',       cx: 1070, cy: 100 },
]

const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
]

function nodeStatus(id: Stage, state: WorkflowState): 'done' | 'active' | 'idle' {
  switch (id) {
    case 'DATA':      return state.datasetId ? 'done' : 'idle'
    case 'TRAIN':     return state.modelId ? 'done' : state.datasetId ? 'active' : 'idle'
    case 'EVALUATE':  return state.evaluationId ? 'done' : state.modelId ? 'active' : 'idle'
    case 'OPTIMIZE':  return state.quantizationId ? 'done' : state.modelId ? 'active' : 'idle'
    case 'RUNTIME':   return state.modelId || state.quantizationId ? 'active' : 'idle'
    case 'BENCHMARK': return state.benchmarkId ? 'done' : state.modelId ? 'active' : 'idle'
    default:          return 'idle'
  }
}

export default function Overview({ state, onNavigate }: Props) {
  const hasData = !!(state.datasetId)
  const hasTrain = !!(state.modelId)
  const hasEval = !!(state.evaluationId)
  const hasOpt = !!(state.quantizationId)
  const hasBench = !!(state.benchmarkId)

  const completedSteps = [hasData, hasTrain, hasEval, hasOpt, hasBench].filter(Boolean).length

  return (
    <div style={{ padding: '40px 40px 48px' }}>
      {/* Hero */}
      <div style={{ maxWidth: 680, marginBottom: 48 }}>
        <div className="sq-stage-tag">MODEL ENGINEERING LIFECYCLE</div>
        <h1 style={{
          fontSize: 44,
          fontWeight: 700,
          letterSpacing: '-0.025em',
          lineHeight: 1.1,
          color: 'var(--fg)',
          margin: '0 0 18px',
        }}>
          From synthetic data<br />
          <span style={{ color: 'var(--accent)' }}>to measurable inference.</span>
        </h1>
        <p className="sq-desc" style={{ fontSize: 15, maxWidth: 520 }}>
          SynthQuanta demonstrates the complete model engineering lifecycle —
          deterministic data generation, LoRA/QLoRA fine-tuning, distribution-shift
          evaluation, INT8 quantization, and custom inference runtime benchmarking.
        </p>
      </div>

      {/* Progress strip */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        marginBottom: 40,
        padding: '14px 20px',
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-lg)',
      }}>
        <span className="sq-label" style={{ margin: 0 }}>LIFECYCLE PROGRESS</span>
        <div style={{ flex: 1, height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${(completedSteps / 5) * 100}%`,
            background: 'var(--accent)',
            borderRadius: 2,
            transition: 'width 0.6s var(--easing)',
          }} />
        </div>
        <span className="sq-data" style={{ fontSize: 12 }}>{completedSteps} / 5</span>
      </div>

      {/* Pipeline SVG map */}
      <div style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-lg)',
        marginBottom: 36,
        overflow: 'hidden',
      }}>
        <div style={{ padding: '16px 20px 0', borderBottom: '1px solid var(--border)' }}>
          <span className="sq-label">PIPELINE TOPOLOGY</span>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <svg
            viewBox="0 0 1200 260"
            style={{ width: '100%', minWidth: 700, height: 260, display: 'block' }}
            aria-label="Pipeline topology"
          >
            {/* Edges */}
            {EDGES.map(([a, b]) => {
              const na = NODES[a], nb = NODES[b]
              const statusA = nodeStatus(na.id, state)
              const edgeDone = statusA === 'done'
              return (
                <line
                  key={`${a}-${b}`}
                  x1={na.cx + 60} y1={na.cy}
                  x2={nb.cx - 60} y2={nb.cy}
                  stroke={edgeDone ? 'var(--accent-muted)' : 'var(--border)'}
                  strokeWidth={edgeDone ? 1.5 : 1}
                  strokeDasharray={edgeDone ? undefined : '4 4'}
                />
              )
            })}

            {/* Nodes */}
            {NODES.map((node) => {
              const st = nodeStatus(node.id, state)
              const fillBg = st === 'done' ? 'var(--accent)' : st === 'active' ? 'var(--accent-light)' : 'var(--surface)'
              const strokeC = st === 'done' ? 'var(--accent)' : st === 'active' ? 'var(--accent)' : 'var(--border-strong)'
              const textC = st === 'done' ? '#fff' : 'var(--fg)'
              const subC = st === 'done' ? 'rgba(255,255,255,0.7)' : 'var(--muted)'

              return (
                <g
                  key={node.id}
                  transform={`translate(${node.cx - 60},${node.cy - 36})`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => onNavigate(node.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => e.key === 'Enter' && onNavigate(node.id)}
                >
                  <rect
                    x={0} y={0} width={120} height={72}
                    rx={4}
                    fill={fillBg}
                    stroke={strokeC}
                    strokeWidth={1.5}
                  />
                  {st === 'done' && (
                    <circle cx={104} cy={14} r={8} fill="rgba(255,255,255,0.25)" />
                  )}
                  {st === 'done' && (
                    <path d="M100 14 l3 3 5-5" stroke="#fff" strokeWidth={1.5} fill="none" strokeLinecap="round" strokeLinejoin="round" />
                  )}
                  <text x={60} y={28} textAnchor="middle" fontSize={11} fontWeight={700} fill={textC} letterSpacing="0.06em">
                    {node.label}
                  </text>
                  <text x={60} y={44} textAnchor="middle" fontSize={9} fill={subC} letterSpacing="0.04em">
                    {node.sub}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>
      </div>

      {/* Quick-action cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 14 }}>
        {[
          {
            stage: 'DATA' as Stage,
            title: 'Signal Studio',
            desc: 'Generate deterministic sensor data with fault injection',
            done: hasData,
            locked: false,
          },
          {
            stage: 'TRAIN' as Stage,
            title: 'Adapter Lab',
            desc: 'Fine-tune SensorTransformer with LoRA or QLoRA',
            done: hasTrain,
            locked: !hasData,
          },
          {
            stage: 'EVALUATE' as Stage,
            title: 'Robustness Lab',
            desc: 'Evaluate F1 under 5 distribution-shift scenarios',
            done: hasEval,
            locked: !hasTrain,
          },
          {
            stage: 'OPTIMIZE' as Stage,
            title: 'Quantize Lab',
            desc: 'Compress FP32 model to INT8 and compare metrics',
            done: hasOpt,
            locked: !hasTrain,
          },
          {
            stage: 'RUNTIME' as Stage,
            title: 'SQRuntime',
            desc: 'Load model and run live inference on signal windows',
            done: false,
            locked: !hasTrain,
          },
          {
            stage: 'BENCHMARK' as Stage,
            title: 'Benchmark Lab',
            desc: 'Measure P95, throughput, and memory across batch sizes',
            done: hasBench,
            locked: !hasTrain,
          },
        ].map((card) => (
          <button
            key={card.stage}
            className="sq-btn"
            disabled={card.locked}
            onClick={() => !card.locked && onNavigate(card.stage)}
            style={{
              flexDirection: 'column',
              alignItems: 'flex-start',
              padding: '16px',
              height: 'auto',
              gap: 6,
              minWidth: 0,
              borderColor: card.done ? 'var(--accent-muted)' : undefined,
              background: card.done ? 'var(--accent-light)' : undefined,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
              <span style={{ fontWeight: 700, fontSize: 12, color: card.done ? 'var(--accent)' : 'var(--fg)' }}>
                {card.title}
              </span>
              {card.done && (
                <span className="sq-badge sq-badge-ok" style={{ fontSize: 9 }}>DONE</span>
              )}
              {card.locked && (
                <span className="sq-badge sq-badge-neu" style={{ fontSize: 9 }}>LOCKED</span>
              )}
            </div>
            <span style={{ fontSize: 11, color: 'var(--muted)', textAlign: 'left', lineHeight: 1.5, fontWeight: 400, whiteSpace: 'normal', overflowWrap: 'break-word' }}>
              {card.desc}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
