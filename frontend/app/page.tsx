'use client'

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

export interface WorkflowState {
  datasetId: string | null
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
