'use client'

import type { Stage, WorkflowState } from '@/app/page'

interface Props {
  stage: Stage
  state: WorkflowState
  onNavigate: (s: Stage) => void
}

interface NavItem {
  id: Stage
  num: string
  label: string
  available: (s: WorkflowState) => boolean
  done: (s: WorkflowState) => boolean
}

const NAV_ITEMS: NavItem[] = [
  {
    id: 'OVERVIEW',
    num: '00',
    label: 'OVERVIEW',
    available: () => true,
    done: (s) => !!(s.datasetId),
  },
  {
    id: 'DATA',
    num: '01',
    label: 'DATA',
    available: () => true,
    done: (s) => !!(s.datasetId),
  },
  {
    id: 'TRAIN',
    num: '02',
    label: 'TRAIN',
    available: (s) => !!(s.datasetId),
    done: (s) => !!(s.modelId),
  },
  {
    id: 'EVALUATE',
    num: '03',
    label: 'EVALUATE',
    available: (s) => !!(s.modelId),
    done: (s) => !!(s.evaluationId),
  },
  {
    id: 'OPTIMIZE',
    num: '04',
    label: 'OPTIMIZE',
    available: (s) => !!(s.modelId),
    done: (s) => !!(s.quantizationId),
  },
  {
    id: 'RUNTIME',
    num: '05',
    label: 'RUNTIME',
    available: (s) => !!(s.modelId || s.quantizationId),
    done: () => false,
  },
  {
    id: 'BENCHMARK',
    num: '06',
    label: 'BENCHMARK',
    available: (s) => !!(s.modelId || s.quantizationId),
    done: (s) => !!(s.benchmarkId),
  },
]

export default function WorkflowNav({ stage, state, onNavigate }: Props) {
  return (
    <nav className="sq-nav">
      <div className="sq-nav-track">
        {NAV_ITEMS.map((item, i) => {
          const available = item.available(state)
          const done = item.done(state)
          const active = stage === item.id

          let dotClass = 'sq-nav-dot'
          if (active) dotClass += ' dot-on'
          else if (done) dotClass += ' dot-done'

          return (
            <div key={item.id} style={{ display: 'contents' }}>
              {i > 0 && <div className="sq-nav-sep" />}
              <button
                className={[
                  'sq-nav-btn',
                  active ? 'sq-nav-active' : '',
                ].join(' ')}
                onClick={() => onNavigate(item.id)}
              >
                <span className="sq-nav-num">{item.num}</span>
                <span className="sq-nav-name">{item.label}</span>
                <span className={dotClass} />
              </button>
            </div>
          )
        })}
      </div>
    </nav>
  )
}
