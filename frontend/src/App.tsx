import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  BarChart3,
  BookOpen,
  ChartNoAxesCombined,
  FlaskConical,
  LineChart,
  Radio,
  ScanSearch,
} from 'lucide-react'
import {
  ApiError,
  getExperiment,
  getExperiments,
  getHealth,
  getMeta,
  IS_DEMO_MODE,
} from './api/client'
import { DEMO_NOTICE } from './data/demo'
import { getDemoResearchCharts } from './data/demoCharts'
import type {
  ExperimentPage,
  ExperimentSummary,
  MetaResponse,
} from './types/api'
import './App.css'

const OverviewView = lazy(async () => ({
  default: (await import('./views/OverviewView')).OverviewView,
}))
const ScreenerView = lazy(async () => ({
  default: (await import('./views/ScreenerView')).ScreenerView,
}))
const PairResearchView = lazy(async () => ({
  default: (await import('./views/PairResearchView')).PairResearchView,
}))
const BacktestView = lazy(async () => ({
  default: (await import('./views/BacktestView')).BacktestView,
}))
const RobustnessView = lazy(async () => ({
  default: (await import('./views/RobustnessView')).RobustnessView,
}))
const MethodologyView = lazy(async () => ({
  default: (await import('./views/MethodologyView')).MethodologyView,
}))

type View = 'overview' | 'screener' | 'pair' | 'backtest' | 'robustness' | 'methodology'
type BackendState = 'checking' | 'online' | 'unavailable'
type ListState = 'loading' | 'ready' | 'error'
type DetailState = 'idle' | 'loading' | 'ready' | 'error'

const PAGE_LIMIT = 25
const DEFAULT_DEMO_PAIR = 'SYNTH_A / SYNTH_B'

const navigation: Array<{
  id: View
  label: string
  shortLabel: string
  icon: typeof LineChart
}> = [
  { id: 'overview', label: 'Overview', shortLabel: 'Overview', icon: LineChart },
  { id: 'screener', label: 'Screener', shortLabel: 'Screener', icon: ScanSearch },
  { id: 'pair', label: 'Pair Research', shortLabel: 'Pair Research', icon: Activity },
  { id: 'backtest', label: 'Backtest / OOS', shortLabel: 'Backtest', icon: ChartNoAxesCombined },
  { id: 'robustness', label: 'Robustness', shortLabel: 'Robustness', icon: BarChart3 },
  { id: 'methodology', label: 'Methodology', shortLabel: 'Methodology', icon: BookOpen },
]

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback
}

function App() {
  const [view, setView] = useState<View>('overview')
  const [backendState, setBackendState] = useState<BackendState>('checking')
  const [metadata, setMetadata] = useState<MetaResponse | null>(null)
  const [experimentPage, setExperimentPage] = useState<ExperimentPage | null>(null)
  const [listState, setListState] = useState<ListState>('loading')
  const [listError, setListError] = useState<string | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [selectedPairId, setSelectedPairId] = useState<string | null>(
    IS_DEMO_MODE ? DEFAULT_DEMO_PAIR : null,
  )
  const [experiment, setExperiment] = useState<ExperimentSummary | null>(null)
  const [detailState, setDetailState] = useState<DetailState>('idle')
  const [detailError, setDetailError] = useState<string | null>(null)

  const charts = useMemo(
    () => IS_DEMO_MODE && selectedPairId ? getDemoResearchCharts(selectedPairId) : null,
    [selectedPairId],
  )

  const loadExperiment = useCallback(async (
    runId: string,
    pairId?: string | null,
    navigateToPair = false,
  ) => {
    setSelectedRunId(runId)
    if (pairId) setSelectedPairId(pairId)
    setDetailState('loading')
    setDetailError(null)
    if (navigateToPair) setView('pair')
    try {
      const detail = await getExperiment(runId)
      setExperiment(detail)
      setSelectedPairId(pairId ?? detail.selected_pair?.pair_id ?? null)
      setDetailState('ready')
    } catch (error) {
      setExperiment(null)
      setDetailState('error')
      setDetailError(errorMessage(error, 'The selected experiment could not be loaded.'))
    }
  }, [])

  const loadExperiments = useCallback(async () => {
    setListState('loading')
    setListError(null)
    try {
      const page = await getExperiments({ limit: PAGE_LIMIT, offset: 0 })
      setExperimentPage(page)
      setListState('ready')
    } catch (error) {
      setListState('error')
      setListError(errorMessage(error, 'The experiment registry could not be loaded.'))
    }
  }, [])

  useEffect(() => {
    let current = true

    getHealth()
      .then((health) => {
        if (current) setBackendState(health.status === 'ok' ? 'online' : 'unavailable')
      })
      .catch(() => {
        if (current) setBackendState('unavailable')
      })

    getMeta()
      .then((meta) => {
        if (current) setMetadata(meta)
      })
      .catch(() => undefined)

    getExperiments({ limit: PAGE_LIMIT, offset: 0 })
      .then((page) => {
        if (!current) return
        setExperimentPage(page)
        setListState('ready')
        if (page.items.length === 0) return

        const first = page.items[0]
        setSelectedRunId(first.run_id)
        setSelectedPairId(first.selected_pair_id)
        setDetailState('loading')
        getExperiment(first.run_id)
          .then((detail) => {
            if (!current) return
            setExperiment(detail)
            setSelectedPairId(detail.selected_pair?.pair_id ?? first.selected_pair_id)
            setDetailState('ready')
          })
          .catch((error: unknown) => {
            if (!current) return
            setDetailState('error')
            setDetailError(errorMessage(error, 'The initial experiment could not be loaded.'))
          })
      })
      .catch((error: unknown) => {
        if (!current) return
        setListState('error')
        setListError(errorMessage(error, 'The experiment registry could not be loaded.'))
      })

    return () => {
      current = false
    }
  }, [])

  const handleSelectPair = useCallback((pairId: string, runId: string | null) => {
    setSelectedPairId(pairId)
    setView('pair')
    if (runId) {
      void loadExperiment(runId, pairId, true)
    } else {
      setSelectedRunId(null)
      setExperiment(null)
      setDetailError(null)
      setDetailState('ready')
    }
  }, [loadExperiment])

  const backendLabel = IS_DEMO_MODE
    ? 'Synthetic demo'
    : backendState === 'online'
      ? 'API online'
      : backendState === 'checking'
        ? 'Connecting'
        : 'API unavailable'
  const presentationState = IS_DEMO_MODE ? 'online' : backendState
  const viewLabel = navigation.find((item) => item.id === view)?.shortLabel ?? 'Overview'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <Radio size={19} strokeWidth={1.8} />
          </div>
          <div>
            <p className="brand-name">Stat-Arb Research</p>
            <p className="brand-caption">Institutional research workspace</p>
          </div>
        </div>

        <nav className="primary-nav" aria-label="Primary navigation">
          <p className="nav-eyebrow">Research workflow</p>
          {navigation.map(({ id, label, icon: Icon }, index) => (
            <button
              className={`nav-item${view === id ? ' is-active' : ''}`}
              type="button"
              key={id}
              onClick={() => setView(id)}
              aria-current={view === id ? 'page' : undefined}
              aria-label={label}
              title={label}
            >
              <span className="nav-index" aria-hidden="true">
                {String(index + 1).padStart(2, '0')}
              </span>
              <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
              <span className="nav-label">{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-context">
          <p>Active relationship</p>
          <strong>{selectedPairId ?? 'None selected'}</strong>
          <span>{experiment?.experiment_name ?? 'Select from screener'}</span>
        </div>

        <div className="sidebar-foot">
          <div className={`status-dot status-${presentationState}`} aria-hidden="true" />
          <div>
            <p>{backendLabel}</p>
            <span>{IS_DEMO_MODE ? 'Deterministic local fixtures' : 'Read-only research access'}</span>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <p className="topbar-kicker">Research workspace</p>
            <p className="topbar-title">{viewLabel}</p>
          </div>
          <div className="topbar-meta">
            <span className="topbar-datum">
              <small>Active pair</small>
              <strong>{selectedPairId ?? 'No pair selected'}</strong>
            </span>
            <span className="topbar-datum">
              <small>Pipeline</small>
              <strong>{metadata?.research_pipeline_version ?? '—'}</strong>
            </span>
            <span className={`health-pill health-${presentationState}`}>
              <span aria-hidden="true" />
              {backendLabel}
            </span>
          </div>
        </header>

        {IS_DEMO_MODE && (
          <div className="demo-banner" role="status">
            <FlaskConical size={14} aria-hidden="true" />
            <strong>Synthetic demo</strong>
            <span>{DEMO_NOTICE}</span>
          </div>
        )}

        <Suspense fallback={<div className="view-loading">Loading research workspace…</div>}>
          {view === 'overview' && (
            <OverviewView
              backendState={backendState}
              metadata={metadata}
              experiment={experiment}
              charts={charts}
              demoMode={IS_DEMO_MODE}
            />
          )}
          {view === 'screener' && (
            <ScreenerView
              page={experimentPage}
              state={listState}
              error={listError}
              demoMode={IS_DEMO_MODE}
              selectedPairId={selectedPairId}
              onRetry={() => void loadExperiments()}
              onSelectPair={handleSelectPair}
            />
          )}
          {view === 'pair' && (
            <PairResearchView
              pairId={selectedPairId}
              experiment={experiment}
              charts={charts}
              state={detailState}
              error={detailError}
              demoMode={IS_DEMO_MODE}
              onOpenScreener={() => setView('screener')}
              onRetry={() => selectedRunId && void loadExperiment(selectedRunId, selectedPairId, true)}
            />
          )}
          {view === 'backtest' && (
            <BacktestView experiment={experiment} charts={charts} demoMode={IS_DEMO_MODE} />
          )}
          {view === 'robustness' && (
            <RobustnessView experiment={experiment} charts={charts} demoMode={IS_DEMO_MODE} />
          )}
          {view === 'methodology' && <MethodologyView />}
        </Suspense>
      </main>
    </div>
  )
}

export default App
