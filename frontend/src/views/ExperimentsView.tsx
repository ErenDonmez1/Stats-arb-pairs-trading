import { AlertCircle, ArrowLeft, ArrowRight, DatabaseZap, RefreshCw } from 'lucide-react'
import { StatusBadge } from '../components/StatusBadge'
import type { ExperimentPage } from '../types/api'
import { formatDate, formatPercent, formatRatio } from '../utils/format'

type LoadState = 'loading' | 'ready' | 'error'

interface ExperimentsViewProps {
  page: ExperimentPage | null
  state: LoadState
  error: string | null
  onRetry: () => void
  onPage: (offset: number) => void
  onSelect: (runId: string) => void
}

export function ExperimentsView({
  page,
  state,
  error,
  onRetry,
  onPage,
  onSelect,
}: ExperimentsViewProps) {
  const limit = page?.limit ?? 10
  const offset = page?.offset ?? 0
  const canGoBack = offset > 0
  const canGoForward = page ? offset + page.count < page.total : false

  return (
    <div className="content-wrap page-content">
      <header className="page-heading">
        <div>
          <p className="section-eyebrow">Experiment registry</p>
          <h1>Persisted research</h1>
          <p>Canonical summaries ordered newest first by the research database.</p>
        </div>
        <button className="secondary-button" type="button" onClick={onRetry}>
          <RefreshCw size={15} aria-hidden="true" />
          Refresh
        </button>
      </header>

      {state === 'loading' && (
        <div className="table-panel" aria-live="polite" aria-busy="true">
          <div className="loading-table">
            {Array.from({ length: 6 }).map((_, index) => (
              <div className="loading-row" key={index} />
            ))}
          </div>
        </div>
      )}

      {state === 'error' && (
        <section className="state-panel" role="alert">
          <AlertCircle size={25} aria-hidden="true" />
          <h2>Experiment registry unavailable</h2>
          <p>{error ?? 'The persisted experiment list could not be loaded.'}</p>
          <button className="primary-button" type="button" onClick={onRetry}>Try again</button>
        </section>
      )}

      {state === 'ready' && page?.items.length === 0 && (
        <section className="state-panel">
          <DatabaseZap size={27} aria-hidden="true" />
          <h2>No persisted experiments</h2>
          <p>
            Run the research pipeline from Python and persist its canonical summary;
            this dashboard never fabricates example performance.
          </p>
        </section>
      )}

      {state === 'ready' && page && page.items.length > 0 && (
        <>
          <div className="table-panel">
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Experiment</th>
                    <th scope="col">Run date</th>
                    <th scope="col">Status</th>
                    <th scope="col">Selected pair</th>
                    <th scope="col" className="numeric-cell">OOS Sharpe</th>
                    <th scope="col" className="numeric-cell">OOS return</th>
                    <th scope="col">Validation</th>
                    <th scope="col" aria-label="Open experiment" />
                  </tr>
                </thead>
                <tbody>
                  {page.items.map((experiment) => (
                    <tr key={experiment.run_id}>
                      <td>
                        <button
                          className="experiment-link"
                          type="button"
                          onClick={() => onSelect(experiment.run_id)}
                        >
                          <strong>{experiment.experiment_name}</strong>
                          <span>{experiment.run_id}</span>
                        </button>
                      </td>
                      <td>{formatDate(experiment.created_at)}</td>
                      <td><StatusBadge status={experiment.pipeline_status} /></td>
                      <td className="mono-value">{experiment.selected_pair_id ?? '—'}</td>
                      <td className="numeric-cell mono-value">
                        {formatRatio(experiment.walk_forward_calendar_oos_sharpe_ratio)}
                      </td>
                      <td className="numeric-cell mono-value">
                        {formatPercent(experiment.walk_forward_calendar_oos_total_return)}
                      </td>
                      <td><StatusBadge status={experiment.validation_stage} /></td>
                      <td>
                        <button
                          className="icon-button"
                          type="button"
                          onClick={() => onSelect(experiment.run_id)}
                          aria-label={`Open ${experiment.experiment_name}`}
                        >
                          <ArrowRight size={16} aria-hidden="true" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <footer className="pagination-bar">
            <p>
              Showing {offset + 1}–{offset + page.count} of {page.total}
            </p>
            <div>
              <button
                className="secondary-button"
                type="button"
                disabled={!canGoBack}
                onClick={() => onPage(Math.max(0, offset - limit))}
              >
                <ArrowLeft size={14} aria-hidden="true" /> Previous
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={!canGoForward}
                onClick={() => onPage(offset + limit)}
              >
                Next <ArrowRight size={14} aria-hidden="true" />
              </button>
            </div>
          </footer>
        </>
      )}
    </div>
  )
}
