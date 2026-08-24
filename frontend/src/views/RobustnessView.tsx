import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  ChartFrame,
  ChartUnavailable,
  ResearchTooltip,
} from '../components/ResearchChart'
import { CHART_COLOURS } from '../components/chartTheme'
import {
  DEMO_BOOTSTRAP_INTERVAL,
  DEMO_COST_STRESS,
  DEMO_ROBUSTNESS_GRID,
  type DemoResearchCharts,
} from '../data/demoCharts'
import type { ExperimentSummary } from '../types/api'
import { formatCount, formatLabel, formatPercent } from '../utils/format'
import { StatusBadge } from '../components/StatusBadge'

interface RobustnessViewProps {
  experiment: ExperimentSummary | null
  charts: DemoResearchCharts | null
  demoMode: boolean
}

const formationWindows = [252, 378, 504, 630]
const entryThresholds = [1.5, 2, 2.5, 3]

function heatColour(sharpe: number): string {
  const strength = Math.min(Math.abs(sharpe) / 0.3, 1)
  return sharpe >= 0
    ? `rgba(86, 183, 200, ${0.12 + strength * 0.58})`
    : `rgba(211, 109, 114, ${0.12 + strength * 0.58})`
}

export function RobustnessView({ experiment, charts, demoMode }: RobustnessViewProps) {
  if (!experiment) {
    return (
      <div className="content-wrap page-content">
        <section className="state-panel">
          <h1>No experiment selected</h1>
          <p>Select a pair before inspecting robustness and statistical validation.</p>
        </section>
      </div>
    )
  }

  const robustness = experiment.robustness
  const validation = experiment.validation
  const scenarioDistribution = DEMO_ROBUSTNESS_GRID.map((cell) => ({
    scenario: cell.scenarioId,
    sharpe: cell.sharpe,
    baseline: cell.baseline,
  }))

  return (
    <div className="content-wrap page-content robustness-page">
      <header className="research-page-heading">
        <div>
          <p className="section-eyebrow">Sensitivity and uncertainty</p>
          <h1>Robustness</h1>
          <p>How fragile are the conclusions across predefined scenarios and statistical diagnostics?</p>
        </div>
        <div className="heading-context">
          {demoMode && <span className="demo-chip">Synthetic demo</span>}
          <StatusBadge status={robustness.stage} />
          <span>{formatCount(robustness.scenario_count)} scenarios declared</span>
        </div>
      </header>

      <section className="robustness-principle">
        <strong>Diagnostic sensitivity, not OOS optimization.</strong>
        <p>The predefined baseline remains identified; no scenario is promoted as a winner.</p>
      </section>

      <div className="robustness-primary-grid">
        <ChartFrame
          eyebrow="Parameter surface"
          title="Robustness heatmap"
          description="Calendar OOS Sharpe across fixed entry and formation settings."
          badge={demoMode ? 'Synthetic demo robustness surface' : undefined}
          className="heatmap-frame"
        >
          {demoMode ? (
            <div className="heatmap-wrap">
              <div className="heatmap-axis-title">Entry z-score →</div>
              <div className="robustness-heatmap" role="img" aria-label="Synthetic scenario Sharpe heatmap">
                <div className="heatmap-corner">Formation</div>
                {entryThresholds.map((entry) => <div className="heatmap-column-label" key={entry}>{entry.toFixed(1)}</div>)}
                {formationWindows.flatMap((formation) => {
                  const row = DEMO_ROBUSTNESS_GRID.filter((cell) => cell.formationDays === formation)
                  return [
                    <div className="heatmap-row-label" key={`label-${formation}`}>{formation}d</div>,
                    ...row.map((cell) => (
                      <div
                        className={`heatmap-cell${cell.baseline ? ' is-baseline' : ''}`}
                        key={cell.scenarioId}
                        style={{ backgroundColor: heatColour(cell.sharpe) }}
                        title={`${cell.scenarioId}: Sharpe ${cell.sharpe.toFixed(2)}, return ${(cell.totalReturn * 100).toFixed(2)}%`}
                      >
                        <strong>{cell.sharpe.toFixed(2)}</strong>
                        {cell.baseline && <span>Baseline</span>}
                      </div>
                    )),
                  ]
                })}
              </div>
              <div className="heatmap-legend"><span>Lower</span><i /><span>Higher Sharpe</span></div>
            </div>
          ) : (
            <ChartUnavailable />
          )}
        </ChartFrame>

        <section className="robustness-summary-panel">
          <div className="panel-heading-compact">
            <div>
              <p className="section-eyebrow">Scenario accounting</p>
              <h2>Evaluation status</h2>
            </div>
          </div>
          <dl className="compact-definition-list">
            <div><dt>Baseline scenario</dt><dd>{robustness.baseline_scenario_id ?? '—'}</dd></div>
            <div><dt>Completed</dt><dd>{formatCount(robustness.completed_scenarios)}</dd></div>
            <div><dt>Analytics unavailable</dt><dd>{formatCount(robustness.analytically_unavailable_scenarios)}</dd></div>
            <div><dt>Invalid</dt><dd>{formatCount(robustness.invalid_scenarios)}</dd></div>
            <div><dt>Failed</dt><dd>{formatCount(robustness.failed_scenarios)}</dd></div>
            <div><dt>Common horizon</dt><dd>{formatCount(robustness.common_horizon_observations)} obs.</dd></div>
            <div><dt>Headline basis</dt><dd>{formatLabel(robustness.headline_metric_basis)}</dd></div>
          </dl>
        </section>
      </div>

      <div className="robustness-secondary-grid">
        <ChartFrame
          eyebrow="Cross-scenario dispersion"
          title="Scenario Sharpe distribution"
          description="Every predefined scenario remains represented; the baseline is marked separately."
          className="scenario-chart"
        >
          {demoMode ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={scenarioDistribution} margin={{ top: 10, right: 8, left: -10, bottom: 0 }}>
                <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                <XAxis dataKey="scenario" stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} interval={1} />
                <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} />
                <ReferenceLine y={0} stroke={CHART_COLOURS.muted} />
                <Tooltip content={<ResearchTooltip />} />
                <Bar dataKey="sharpe" name="Scenario Sharpe" radius={[2, 2, 0, 0]}>
                  {scenarioDistribution.map((scenario) => (
                    <Cell
                      key={scenario.scenario}
                      fill={scenario.baseline
                        ? CHART_COLOURS.accent
                        : scenario.sharpe >= 0
                          ? CHART_COLOURS.primary
                          : CHART_COLOURS.negative}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <ChartUnavailable compact />}
        </ChartFrame>

        <ChartFrame
          eyebrow="Execution sensitivity"
          title="Cost stress"
          description="Return response to progressively higher predefined transaction-cost assumptions."
          className="scenario-chart"
        >
          {demoMode ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={DEMO_COST_STRESS} margin={{ top: 10, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                <XAxis dataKey="costBps" unit=" bps" stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} />
                <ReferenceLine y={0} stroke={CHART_COLOURS.muted} />
                <Tooltip content={<ResearchTooltip valueFormatter={(value, name) => name.includes('Return') ? `${(value * 100).toFixed(2)}%` : value.toFixed(2)} />} />
                <Line type="monotone" dataKey="totalReturn" name="OOS return" stroke={CHART_COLOURS.primary} strokeWidth={2} dot={{ r: 3, fill: CHART_COLOURS.primary }} />
              </LineChart>
            </ResponsiveContainer>
          ) : <ChartUnavailable compact />}
        </ChartFrame>
      </div>

      <section className="validation-section">
        <header className="validation-section-heading">
          <div>
            <p className="section-eyebrow">Statistical validation</p>
            <h2>Uncertainty and consistency</h2>
          </div>
          <StatusBadge status={validation.overall_availability} />
        </header>

        <div className="validation-visual-grid">
          <article className="bootstrap-interval-panel">
            <p className="section-eyebrow">Moving-block bootstrap</p>
            <h3>{DEMO_BOOTSTRAP_INTERVAL.metric} interval</h3>
            {demoMode ? (
              <>
                <div className="interval-scale">
                  <span className="interval-line" style={{ left: '18%', width: '64%' }} />
                  <span className="interval-median" style={{ left: '55%' }} />
                  <span className="interval-observed" style={{ left: '57%' }} />
                </div>
                <div className="interval-labels">
                  <span>Lower<strong>{DEMO_BOOTSTRAP_INTERVAL.lower.toFixed(2)}</strong></span>
                  <span>Median<strong>{DEMO_BOOTSTRAP_INTERVAL.median.toFixed(2)}</strong></span>
                  <span>Observed<strong>{DEMO_BOOTSTRAP_INTERVAL.observed.toFixed(2)}</strong></span>
                  <span>Upper<strong>{DEMO_BOOTSTRAP_INTERVAL.upper.toFixed(2)}</strong></span>
                </div>
              </>
            ) : (
              <ChartUnavailable compact />
            )}
          </article>

          <article className="validation-indicator-panel">
            <p className="section-eyebrow">Sharpe evidence</p>
            <div className="large-indicator">
              <strong>{formatPercent(validation.probabilistic_sharpe_probability)}</strong>
              <span>Probabilistic Sharpe probability</span>
            </div>
            <div className="indicator-track"><span style={{ width: `${Math.max(0, Math.min(100, (validation.probabilistic_sharpe_probability ?? 0) * 100))}%` }} /></div>
            <p>Evidence that Sharpe exceeds its configured benchmark—not probability of profitability.</p>
          </article>

          <article className="validation-indicator-panel">
            <p className="section-eyebrow">Track record</p>
            <div className="large-indicator">
              <strong>{formatCount(validation.minimum_track_record_observations)}</strong>
              <span>Minimum observations estimated</span>
            </div>
            <dl className="compact-definition-list is-small">
              <div><dt>Fold consistency</dt><dd>{formatLabel(validation.fold_consistency_availability)}</dd></div>
              <div><dt>Primary availability</dt><dd>{formatLabel(validation.primary_availability)}</dd></div>
            </dl>
          </article>

          <article className="validation-indicator-panel">
            <p className="section-eyebrow">Multiplicity accounting</p>
            <div className="multiplicity-ratio">
              <strong>{formatCount(validation.multiple_testing_valid_pvalue_count)}</strong>
              <span>/ {formatCount(validation.multiple_testing_total_configurations)} valid p-values</span>
            </div>
            <dl className="compact-definition-list is-small">
              <div><dt>Eligible hypotheses</dt><dd>{formatCount(validation.multiple_testing_eligible_hypothesis_count)}</dd></div>
              <div><dt>Unavailable p-values</dt><dd>{formatCount(validation.multiple_testing_unavailable_pvalue_count)}</dd></div>
            </dl>
          </article>
        </div>

        {demoMode && charts && (
          <ChartFrame
            eyebrow="Fold consistency"
            title="Contribution by OOS fold"
            description="Adverse and no-selection folds remain visible in the diagnostic."
            className="validation-fold-chart"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={charts.foldReturns} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                <XAxis dataKey="fold" stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} />
                <ReferenceLine y={0} stroke={CHART_COLOURS.muted} />
                <Tooltip content={<ResearchTooltip valueFormatter={(value) => `${(value * 100).toFixed(2)}%`} />} />
                <Bar dataKey="return" name="Fold return" radius={[2, 2, 0, 0]}>
                  {charts.foldReturns.map((fold) => (
                    <Cell key={fold.fold} fill={fold.status === 'NO_SELECTION' ? CHART_COLOURS.muted : fold.return >= 0 ? CHART_COLOURS.positive : CHART_COLOURS.negative} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>
        )}
      </section>

      <p className="robustness-footnote">
        Tested dimensions: {robustness.tested_dimensions.join(', ') || '—'}. Untested material dimensions remain disclosed in provenance.
      </p>
    </div>
  )
}
