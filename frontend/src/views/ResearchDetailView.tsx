import {
  AlertCircle,
  ArrowLeft,
  Braces,
  CheckCircle2,
  ChevronRight,
  FileWarning,
  FlaskConical,
  RefreshCw,
} from 'lucide-react'
import { DigestValue } from '../components/DigestValue'
import { MetricCard } from '../components/MetricCard'
import { StatusBadge } from '../components/StatusBadge'
import type { ExperimentSummary, JsonObject } from '../types/api'
import {
  formatCount,
  formatDate,
  formatLabel,
  formatPercent,
  formatRatio,
} from '../utils/format'

type LoadState = 'idle' | 'loading' | 'ready' | 'error'

interface ResearchDetailViewProps {
  experiment: ExperimentSummary | null
  state: LoadState
  error: string | null
  onChooseExperiment: () => void
  onRetry: () => void
}

function StageCard({
  label,
  status,
  detail,
}: {
  label: string
  status?: string | null
  detail?: string
}) {
  return (
    <article className="stage-card">
      <p>{label}</p>
      {status ? <StatusBadge status={status} /> : <strong>{detail ?? '—'}</strong>}
      {status && detail && <span>{detail}</span>}
    </article>
  )
}

function JsonPanel({ title, value }: { title: string; value: JsonObject }) {
  return (
    <details className="json-panel">
      <summary>
        <Braces size={15} aria-hidden="true" />
        {title}
        <ChevronRight size={15} className="summary-chevron" aria-hidden="true" />
      </summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  )
}

export function ResearchDetailView({
  experiment,
  state,
  error,
  onChooseExperiment,
  onRetry,
}: ResearchDetailViewProps) {
  if (state === 'idle') {
    return (
      <div className="content-wrap page-content">
        <section className="state-panel detail-empty">
          <FlaskConical size={29} aria-hidden="true" />
          <h1>Select a research experiment</h1>
          <p>Open a persisted experiment to inspect its screening, diagnostic, OOS and provenance summaries.</p>
          <button className="primary-button" type="button" onClick={onChooseExperiment}>
            Browse experiments
          </button>
        </section>
      </div>
    )
  }

  if (state === 'loading') {
    return (
      <div className="content-wrap page-content" aria-live="polite" aria-busy="true">
        <div className="detail-loading-block" />
        <div className="detail-loading-grid">
          {Array.from({ length: 8 }).map((_, index) => (
            <div className="detail-loading-card" key={index} />
          ))}
        </div>
      </div>
    )
  }

  if (state === 'error' || !experiment) {
    return (
      <div className="content-wrap page-content">
        <section className="state-panel" role="alert">
          <AlertCircle size={27} aria-hidden="true" />
          <h1>Research detail unavailable</h1>
          <p>{error ?? 'The selected experiment could not be loaded.'}</p>
          <div className="state-actions">
            <button className="secondary-button" type="button" onClick={onChooseExperiment}>
              <ArrowLeft size={14} aria-hidden="true" /> Experiments
            </button>
            <button className="primary-button" type="button" onClick={onRetry}>
              <RefreshCw size={14} aria-hidden="true" /> Retry
            </button>
          </div>
        </section>
      </div>
    )
  }

  const pair = experiment.selected_pair
  const diagnostic = experiment.diagnostic
  const walkForward = experiment.walk_forward
  const robustness = experiment.robustness
  const validation = experiment.validation

  return (
    <div className="content-wrap page-content detail-content">
      <button className="text-button" type="button" onClick={onChooseExperiment}>
        <ArrowLeft size={14} aria-hidden="true" /> Back to experiments
      </button>

      <header className="detail-identity">
        <div>
          <p className="section-eyebrow">Research detail / Persisted experiment</p>
          <div className="detail-title-line">
            <h1>{experiment.experiment_name}</h1>
            <StatusBadge status={experiment.pipeline_status} />
          </div>
          <p className="detail-run-id">{experiment.run_id}</p>
        </div>
        <div className="identity-aside">
          <p>Selected pair</p>
          <strong>{pair?.pair_id ?? '—'}</strong>
          <span>{formatDate(experiment.created_at)}</span>
        </div>
      </header>

      <section className="identity-grid" aria-label="Experiment identity">
        <div><span>Research digest</span><DigestValue value={experiment.research_content_digest} /></div>
        <div><span>Price digest</span><DigestValue value={experiment.price_content_digest} /></div>
        <div><span>Pipeline</span><strong>{experiment.research_pipeline_version}</strong></div>
        <div><span>Schema / Config</span><strong>{experiment.experiment_schema_version} / {experiment.configuration_snapshot_version}</strong></div>
      </section>

      <section className="detail-section" aria-labelledby="stages-title">
        <div className="detail-section-heading">
          <div>
            <p className="section-eyebrow">Pipeline status</p>
            <h2 id="stages-title">Research stages</h2>
          </div>
        </div>
        <div className="stage-grid">
          <StageCard
            label="Screening"
            detail={`${formatCount(experiment.screening.selected_count)} of ${formatCount(experiment.screening.candidate_count)} selected`}
          />
          <StageCard label="Diagnostic" status={diagnostic.stage} detail="Full-sample in-sample" />
          <StageCard label="Walk-forward" status={walkForward.stage} detail={formatLabel(walkForward.calendar_analytics_status)} />
          <StageCard label="Robustness" status={robustness.stage} detail={formatLabel(robustness.common_horizon_analytics_status)} />
          <StageCard label="Validation" status={validation.stage} detail={formatLabel(validation.overall_availability)} />
        </div>
      </section>

      <section className="detail-section" aria-labelledby="screening-title">
        <div className="detail-section-heading">
          <div>
            <p className="section-eyebrow">Formation evidence</p>
            <h2 id="screening-title">Pair screening</h2>
          </div>
          <p>{formatLabel(experiment.screening.selection_scope)}</p>
        </div>
        <div className="screening-layout">
          <article className="selected-pair-card">
            <p>Selected relationship</p>
            <strong>{pair?.symbol_y ?? '—'} <span>/</span> {pair?.symbol_x ?? '—'}</strong>
            <dl>
              <div><dt>Rank</dt><dd>{formatCount(pair?.rank)}</dd></div>
              <div><dt>Hedge beta</dt><dd>{formatRatio(pair?.beta)}</dd></div>
              <div><dt>Corrected p-value</dt><dd>{formatRatio(pair?.corrected_pvalue)}</dd></div>
              <div><dt>Half-life</dt><dd>{formatRatio(pair?.half_life)}</dd></div>
              <div><dt>Hurst</dt><dd>{formatRatio(pair?.hurst)}</dd></div>
            </dl>
          </article>
          <div className="screening-stats">
            <MetricCard label="Candidates" value={formatCount(experiment.screening.candidate_count)} />
            <MetricCard label="Selected" value={formatCount(experiment.screening.selected_count)} />
            <MetricCard label="Selection coverage" value={formatPercent(walkForward.selection_coverage)} />
            <MetricCard label="Selected rank" value={formatCount(pair?.rank)} />
          </div>
        </div>
      </section>

      <section className="detail-section diagnostic-section" aria-labelledby="diagnostic-title">
        <div className="detail-section-heading">
          <div>
            <p className="section-eyebrow">Diagnostic only</p>
            <h2 id="diagnostic-title">In-sample diagnostic</h2>
          </div>
          <p>Diagnostic research output — not out-of-sample evidence.</p>
        </div>
        <div className="research-metric-grid">
          <MetricCard label="Total return" value={formatPercent(diagnostic.total_return)} />
          <MetricCard label="Annualized return" value={formatPercent(diagnostic.annualized_return)} />
          <MetricCard label="Annualized volatility" value={formatPercent(diagnostic.annualized_volatility)} />
          <MetricCard label="Sharpe ratio" value={formatRatio(diagnostic.sharpe_ratio)} />
          <MetricCard label="Sortino ratio" value={formatRatio(diagnostic.sortino_ratio)} />
          <MetricCard label="Maximum drawdown" value={formatPercent(diagnostic.maximum_drawdown)} />
          <MetricCard label="Calmar ratio" value={formatRatio(diagnostic.calmar_ratio)} />
          <MetricCard label="Trades" value={formatCount(diagnostic.trade_count)} />
        </div>
      </section>

      <section className="detail-section oos-section" aria-labelledby="oos-title">
        <div className="detail-section-heading">
          <div>
            <p className="section-eyebrow">Calendar evaluation</p>
            <h2 id="oos-title">Calendar walk-forward OOS</h2>
          </div>
          <p>Calendar returns retain no-selection cash periods and unavailable observations.</p>
        </div>
        <div className="research-metric-grid">
          <MetricCard emphasis="oos" label="Total return" value={formatPercent(walkForward.calendar_oos_total_return)} />
          <MetricCard emphasis="oos" label="Annualized return" value={formatPercent(walkForward.calendar_oos_annualized_return)} />
          <MetricCard emphasis="oos" label="Annualized volatility" value={formatPercent(walkForward.calendar_oos_annualized_volatility)} />
          <MetricCard emphasis="oos" label="Sharpe ratio" value={formatRatio(walkForward.calendar_oos_sharpe_ratio)} />
          <MetricCard emphasis="oos" label="Sortino ratio" value={formatRatio(walkForward.calendar_oos_sortino_ratio)} />
          <MetricCard emphasis="oos" label="Maximum drawdown" value={formatPercent(walkForward.calendar_oos_maximum_drawdown)} />
          <MetricCard emphasis="oos" label="Calmar ratio" value={formatRatio(walkForward.calendar_oos_calmar_ratio)} />
          <MetricCard emphasis="oos" label="OOS observations" value={formatCount(walkForward.calendar_oos_report_observations)} />
        </div>
        <div className="oos-context-grid">
          <div><span>Folds completed</span><strong>{formatCount(walkForward.completed_fold_count)} / {formatCount(walkForward.fold_count)}</strong></div>
          <div><span>Selected observations</span><strong>{formatCount(walkForward.selected_oos_observations)}</strong></div>
          <div><span>No-selection observations</span><strong>{formatCount(walkForward.no_selection_oos_observations)}</strong></div>
          <div><span>Unavailable observations</span><strong>{formatCount(walkForward.unavailable_oos_observations)}</strong></div>
          <div><span>Selection coverage</span><strong>{formatPercent(walkForward.selection_coverage)}</strong></div>
        </div>
      </section>

      <section className="detail-section" aria-labelledby="validation-title">
        <div className="detail-section-heading">
          <div>
            <p className="section-eyebrow">Uncertainty analysis</p>
            <h2 id="validation-title">Robustness & validation</h2>
          </div>
        </div>
        <div className="validation-grid">
          <article className="validation-card">
            <div><h3>Parameter robustness</h3><StatusBadge status={robustness.stage} /></div>
            <dl>
              <div><dt>Scenarios</dt><dd>{formatCount(robustness.scenario_count)}</dd></div>
              <div><dt>Completed</dt><dd>{formatCount(robustness.completed_scenarios)}</dd></div>
              <div><dt>Common-horizon observations</dt><dd>{formatCount(robustness.common_horizon_observations)}</dd></div>
              <div><dt>Analytics status</dt><dd>{formatLabel(robustness.common_horizon_analytics_status)}</dd></div>
            </dl>
          </article>
          <article className="validation-card">
            <div><h3>Statistical validation</h3><StatusBadge status={validation.stage} /></div>
            <dl>
              <div><dt>Overall availability</dt><dd>{formatLabel(validation.overall_availability)}</dd></div>
              <div><dt>PSR probability</dt><dd>{formatPercent(validation.probabilistic_sharpe_probability)}</dd></div>
              <div><dt>Minimum track record</dt><dd>{formatCount(validation.minimum_track_record_observations)}</dd></div>
              <div><dt>Tested configurations</dt><dd>{formatCount(validation.multiple_testing_total_configurations)}</dd></div>
            </dl>
          </article>
        </div>
      </section>

      <section className="detail-section" aria-labelledby="provenance-title">
        <div className="detail-section-heading">
          <div>
            <p className="section-eyebrow">Reproducibility</p>
            <h2 id="provenance-title">Provenance & warnings</h2>
          </div>
        </div>

        {experiment.warnings.length > 0 ? (
          <div className="warnings-panel">
            <FileWarning size={18} aria-hidden="true" />
            <div>
              <strong>{experiment.warnings.length} persisted warning{experiment.warnings.length === 1 ? '' : 's'}</strong>
              <ul>{experiment.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
            </div>
          </div>
        ) : (
          <div className="warnings-panel no-warnings">
            <CheckCircle2 size={18} aria-hidden="true" />
            <p>No persisted experiment warnings.</p>
          </div>
        )}

        <div className="json-panel-stack">
          <JsonPanel title="Provenance" value={experiment.provenance} />
          <JsonPanel title="Configuration snapshot" value={experiment.configuration_snapshot} />
          <JsonPanel title="Experiment metadata" value={experiment.metadata} />
        </div>
      </section>
    </div>
  )
}
