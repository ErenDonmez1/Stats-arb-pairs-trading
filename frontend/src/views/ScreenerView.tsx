import { AlertCircle, Filter, RefreshCw, Search } from 'lucide-react'
import { StatusBadge } from '../components/StatusBadge'
import {
  DEMO_PAIR_CANDIDATES,
  type DemoPairCandidate,
} from '../data/demoCharts'
import type { ExperimentPage } from '../types/api'
import { formatCount, formatRatio } from '../utils/format'

type LoadState = 'loading' | 'ready' | 'error'

interface ScreenerViewProps {
  page: ExperimentPage | null
  state: LoadState
  error: string | null
  demoMode: boolean
  selectedPairId: string | null
  onRetry: () => void
  onSelectPair: (pairId: string, runId: string | null) => void
}

interface ScreenerRow {
  pairId: string
  group: string
  screeningStatus: string
  halfLife: number | null
  hurst: number | null
  rank: number | null
  status: string
  runId: string | null
}

function fromDemo(candidate: DemoPairCandidate): ScreenerRow {
  return {
    pairId: candidate.pairId,
    group: candidate.group,
    screeningStatus: candidate.screeningStatus,
    halfLife: candidate.halfLife,
    hurst: candidate.hurst,
    rank: candidate.rank,
    status: candidate.status,
    runId: candidate.experimentRunId,
  }
}

export function ScreenerView({
  page,
  state,
  error,
  demoMode,
  selectedPairId,
  onRetry,
  onSelectPair,
}: ScreenerViewProps) {
  const rows: ScreenerRow[] = demoMode
    ? DEMO_PAIR_CANDIDATES.map(fromDemo)
    : (page?.items ?? []).map((item) => ({
        pairId: item.selected_pair_id ?? 'No selected pair',
        group: '—',
        screeningStatus: item.pipeline_status,
        halfLife: null,
        hurst: null,
        rank: null,
        status: item.selected_pair_id ? 'SELECTED' : 'UNAVAILABLE',
        runId: item.run_id,
      }))

  const selectedCount = rows.filter((row) => row.status === 'SELECTED').length

  return (
    <div className="content-wrap page-content screener-page">
      <header className="research-page-heading">
        <div>
          <p className="section-eyebrow">Formation workspace</p>
          <h1>Pair screener</h1>
          <p>
            Which economically related pairs survive cointegration, persistence,
            and mean-reversion diagnostics?
          </p>
        </div>
        <button className="quiet-button" type="button" onClick={onRetry}>
          <RefreshCw size={14} aria-hidden="true" /> Refresh registry
        </button>
      </header>

      <section className="screener-toolbar" aria-label="Screener context">
        <div className="screener-summary">
          <span><strong>{formatCount(rows.length)}</strong> candidates shown</span>
          <span><strong>{formatCount(selectedCount)}</strong> selected</span>
          <span><strong>{demoMode ? 'Grouped' : 'Persisted'}</strong> universe</span>
        </div>
        <div className="screener-mode">
          <Filter size={14} aria-hidden="true" /> Formation-only evidence
          {demoMode && <span className="demo-chip">Synthetic demo</span>}
        </div>
      </section>

      {state === 'loading' && !demoMode && (
        <div className="analytical-table loading-table" aria-live="polite" aria-busy="true">
          {Array.from({ length: 6 }).map((_, index) => <div className="loading-row" key={index} />)}
        </div>
      )}

      {state === 'error' && !demoMode && (
        <section className="state-panel" role="alert">
          <AlertCircle size={25} aria-hidden="true" />
          <h2>Experiment registry unavailable</h2>
          <p>{error ?? 'The persisted screening summaries could not be loaded.'}</p>
          <button className="quiet-button" type="button" onClick={onRetry}>Try again</button>
        </section>
      )}

      {(demoMode || state === 'ready') && rows.length === 0 && (
        <section className="state-panel">
          <Search size={25} aria-hidden="true" />
          <h2>No screening records</h2>
          <p>Persist an experiment summary before opening the real-data screener.</p>
        </section>
      )}

      {(demoMode || state === 'ready') && rows.length > 0 && (
        <div className="analytical-table">
          <table>
            <thead>
              <tr>
                <th scope="col">Pair</th>
                <th scope="col">Economic group</th>
                <th scope="col">Screening evidence</th>
                <th scope="col" className="numeric-cell">Half-life</th>
                <th scope="col" className="numeric-cell">Hurst</th>
                <th scope="col" className="numeric-cell">Rank</th>
                <th scope="col">Decision</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={`${row.runId ?? 'candidate'}-${row.pairId}`}
                  className={selectedPairId === row.pairId ? 'is-selected' : ''}
                  onClick={() => onSelectPair(row.pairId, row.runId)}
                >
                  <td>
                    <button
                      className="pair-cell-button"
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation()
                        onSelectPair(row.pairId, row.runId)
                      }}
                    >
                      <strong>{row.pairId}</strong>
                      <span>Open pair research →</span>
                    </button>
                  </td>
                  <td>{row.group}</td>
                  <td>{row.screeningStatus}</td>
                  <td className="numeric-cell mono-value">{formatRatio(row.halfLife)}</td>
                  <td className="numeric-cell mono-value">{formatRatio(row.hurst)}</td>
                  <td className="numeric-cell mono-value">{formatCount(row.rank)}</td>
                  <td><StatusBadge status={row.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="table-footnote">
        {demoMode
          ? 'Synthetic candidates are deterministic interface fixtures. Their statistics are not market evidence.'
          : 'The summary API does not persist rejected candidate histories; unavailable fields remain blank.'}
      </p>
    </div>
  )
}
