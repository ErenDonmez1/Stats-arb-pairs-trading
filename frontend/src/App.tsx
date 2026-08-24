import { useCallback, useEffect, useState } from 'react'
import {
  Beaker,
  BookOpen,
  FlaskConical,
  LineChart,
  Radio,
} from 'lucide-react'
import {
  ApiError,
  getExperiment,
  getExperiments,
  getHealth,
  getMeta,
} from './api/client'
import type {
  ExperimentPage,
  ExperimentSummary,
  MetaResponse,
} from './types/api'
import { ExperimentsView } from './views/ExperimentsView'
import { MethodologyView } from './views/MethodologyView'
import { OverviewView } from './views/OverviewView'
import { ResearchDetailView } from './views/ResearchDetailView'
import './App.css'

type View = 'overview' | 'experiments' | 'detail' | 'methodology'
type BackendState = 'checking' | 'online' | 'unavailable'
type ListState = 'loading' | 'ready' | 'error'
type DetailState = 'idle' | 'loading' | 'ready' | 'error'

const PAGE_LIMIT = 10

const navigation: Array<{
  id: View
  label: string
  shortLabel: string
  icon: typeof LineChart
}> = [
  { id: 'overview', label: 'Overview', shortLabel: 'Overview', icon: LineChart },
  { id: 'experiments', label: 'Experiments', shortLabel: 'Experiments', icon: Beaker },
  { id: 'detail', label: 'Research detail', shortLabel: 'Detail', icon: FlaskConical },
  { id: 'methodology', label: 'About / Methodology', shortLabel: 'About', icon: BookOpen },
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
  const [experiment, setExperiment] = useState<ExperimentSummary | null>(null)
  const [detailState, setDetailState] = useState<DetailState>('idle')
  const [detailError, setDetailError] = useState<string | null>(null)

  const loadExperiments = useCallback(async (offset = 0) => {
    setListState('loading')
    setListError(null)
    try {
      const page = await getExperiments({ limit: PAGE_LIMIT, offset })
      setExperimentPage(page)
      setListState('ready')
    } catch (error) {
      setListState('error')
      setListError(errorMessage(error, 'The experiment registry could not be loaded.'))
    }
  }, [])

  const loadDetail = useCallback(async (runId: string) => {
    setSelectedRunId(runId)
    setExperiment(null)
    setDetailState('loading')
    setDetailError(null)
    setView('detail')
    try {
      const detail = await getExperiment(runId)
      setExperiment(detail)
      setDetailState('ready')
    } catch (error) {
      setDetailState('error')
      setDetailError(errorMessage(error, 'The selected experiment could not be loaded.'))
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

  const backendLabel =
    backendState === 'online'
      ? 'API Online'
      : backendState === 'checking'
        ? 'Connecting'
        : 'API Unavailable'

  const viewLabel = navigation.find((item) => item.id === view)?.shortLabel ?? 'Overview'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <Radio size={19} strokeWidth={1.8} />
          </div>
          <div>
            <p className="brand-name">Pairs Research</p>
            <p className="brand-caption">Quantitative systems</p>
          </div>
        </div>

        <nav className="primary-nav" aria-label="Primary navigation">
          <p className="nav-eyebrow">Workspace</p>
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              className={`nav-item${view === id ? ' is-active' : ''}`}
              type="button"
              key={id}
              onClick={() => setView(id)}
              aria-current={view === id ? 'page' : undefined}
              aria-label={label}
              title={label}
            >
              <Icon size={17} strokeWidth={1.8} aria-hidden="true" />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          <div className={`status-dot status-${backendState}`} aria-hidden="true" />
          <div>
            <p>{backendLabel}</p>
            <span>Read-only research access</span>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <p className="topbar-kicker">Research environment / {viewLabel}</p>
            <p className="topbar-title">Statistical Arbitrage</p>
          </div>
          <div className="topbar-meta">
            <span>Pipeline {metadata?.research_pipeline_version ?? '—'}</span>
            <span className={`health-pill health-${backendState}`}>
              <span aria-hidden="true" />
              {backendLabel}
            </span>
          </div>
        </header>

        {view === 'overview' && (
          <OverviewView
            backendState={backendState}
            metadata={metadata}
            experimentCount={experimentPage?.total ?? null}
            experimentsUnavailable={listState === 'error'}
          />
        )}
        {view === 'experiments' && (
          <ExperimentsView
            page={experimentPage}
            state={listState}
            error={listError}
            onRetry={() => void loadExperiments(experimentPage?.offset ?? 0)}
            onPage={(offset) => void loadExperiments(offset)}
            onSelect={(runId) => void loadDetail(runId)}
          />
        )}
        {view === 'detail' && (
          <ResearchDetailView
            experiment={experiment}
            state={detailState}
            error={detailError}
            onChooseExperiment={() => setView('experiments')}
            onRetry={() => selectedRunId && void loadDetail(selectedRunId)}
          />
        )}
        {view === 'methodology' && <MethodologyView />}
      </main>
    </div>
  )
}

export default App
