import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { DemoResearchCharts } from '../data/demoCharts'
import type { ExperimentSummary, MetaResponse } from '../types/api'
import { formatCount, formatPercent, formatRatio } from '../utils/format'
import {
  ChartFrame,
  ChartUnavailable,
  ResearchTooltip,
  SyntheticChartLabel,
} from '../components/ResearchChart'
import { CHART_COLOURS } from '../components/chartTheme'
import { StatusBadge } from '../components/StatusBadge'

type BackendState = 'checking' | 'online' | 'unavailable'

interface OverviewViewProps {
  backendState: BackendState
  metadata: MetaResponse | null
  experiment: ExperimentSummary | null
  charts: DemoResearchCharts | null
  demoMode: boolean
}

const architecture = [
  'Data',
  'Screening',
  'Signal',
  'Backtest',
  'Walk-forward',
  'Validation',
  'Persistence',
]

function monthTick(value: string): string {
  return value.slice(0, 7)
}

export function OverviewView({
  backendState,
  metadata,
  experiment,
  charts,
  demoMode,
}: OverviewViewProps) {
  const oos = experiment?.walk_forward
  const pipelineStages = experiment
    ? [
        ['Screening', experiment.screening.selected_count > 0 ? 'COMPLETED' : 'UNAVAILABLE'],
        ['Diagnostic', experiment.diagnostic.stage],
        ['Walk-forward', experiment.walk_forward.stage],
        ['Robustness', experiment.robustness.stage],
        ['Validation', experiment.validation.stage],
      ]
    : []

  return (
    <div className="content-wrap overview-page">
      <header className="research-page-heading overview-heading">
        <div>
          <p className="section-eyebrow">Quantitative research system</p>
          <h1>Stat-Arb Research Platform</h1>
          <p>
            Causal pairs-trading research, validation and portfolio-risk
            infrastructure.
          </p>
        </div>
        <div className="heading-context">
          {demoMode && <span className="demo-chip">Synthetic demo</span>}
          <span>Pipeline {metadata?.research_pipeline_version ?? '—'}</span>
          <span>API {demoMode ? 'bypassed' : backendState}</span>
        </div>
      </header>

      <ChartFrame
        eyebrow="Primary evidence"
        title="Calendar walk-forward OOS equity curve"
        description="Calendar-time evidence retains no-selection cash periods and preserves unavailable observations."
        badge={demoMode ? 'Synthetic demo series' : undefined}
        className="hero-chart"
      >
        {charts ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={charts.points} margin={{ top: 12, right: 12, left: -8, bottom: 0 }}>
              <defs>
                <linearGradient id="equity-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CHART_COLOURS.primary} stopOpacity={0.28} />
                  <stop offset="100%" stopColor={CHART_COLOURS.primary} stopOpacity={0.01} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
              <XAxis
                dataKey="date"
                tickFormatter={monthTick}
                stroke={CHART_COLOURS.axis}
                tickLine={false}
                axisLine={false}
                minTickGap={46}
              />
              <YAxis
                domain={['dataMin - 1', 'dataMax + 1']}
                stroke={CHART_COLOURS.axis}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value: number) => value.toFixed(0)}
              />
              <Tooltip content={<ResearchTooltip valueFormatter={(value) => value.toFixed(2)} />} />
              <Area
                type="monotone"
                dataKey="equity"
                name="OOS equity"
                stroke={CHART_COLOURS.primary}
                strokeWidth={2}
                fill="url(#equity-fill)"
                activeDot={{ r: 4, fill: CHART_COLOURS.primary, strokeWidth: 0 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <ChartUnavailable />
        )}
      </ChartFrame>

      <section className="metric-tape" aria-label="Primary OOS metrics">
        <div><span>OOS Sharpe</span><strong>{formatRatio(oos?.calendar_oos_sharpe_ratio)}</strong></div>
        <div><span>OOS return</span><strong>{formatPercent(oos?.calendar_oos_total_return)}</strong></div>
        <div><span>Max drawdown</span><strong>{formatPercent(oos?.calendar_oos_maximum_drawdown)}</strong></div>
        <div><span>Selection coverage</span><strong>{formatPercent(oos?.selection_coverage)}</strong></div>
        <div><span>OOS trades</span><strong>{formatCount(charts?.oosTradeCount)}</strong></div>
      </section>

      <div className="overview-analysis-grid">
        <ChartFrame
          eyebrow="Path risk"
          title="Drawdown"
          description="Peak-to-trough decline on the calendar OOS equity path."
          className="drawdown-chart"
        >
          {charts ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={charts.points} margin={{ top: 8, right: 10, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                <XAxis
                  dataKey="date"
                  tickFormatter={monthTick}
                  stroke={CHART_COLOURS.axis}
                  tickLine={false}
                  axisLine={false}
                  minTickGap={42}
                />
                <YAxis
                  stroke={CHART_COLOURS.axis}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`}
                />
                <Tooltip content={<ResearchTooltip valueFormatter={(value) => `${(value * 100).toFixed(2)}%`} />} />
                <Area
                  type="monotone"
                  dataKey="drawdown"
                  name="Drawdown"
                  stroke={CHART_COLOURS.negative}
                  fill={CHART_COLOURS.negative}
                  fillOpacity={0.16}
                  strokeWidth={1.6}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <ChartUnavailable compact />
          )}
        </ChartFrame>

        <section className="pipeline-panel">
          <div className="panel-heading-compact">
            <div>
              <p className="section-eyebrow">Research controls</p>
              <h2>Pipeline status</h2>
            </div>
            {demoMode && <SyntheticChartLabel />}
          </div>
          {pipelineStages.length > 0 ? (
            <div className="pipeline-status-list">
              {pipelineStages.map(([label, status], index) => (
                <div key={label}>
                  <span>{String(index + 1).padStart(2, '0')}</span>
                  <strong>{label}</strong>
                  <StatusBadge status={status} />
                </div>
              ))}
            </div>
          ) : (
            <div className="compact-empty">Select a persisted experiment to inspect stage status.</div>
          )}
        </section>
      </div>

      <section className="architecture-strip" aria-label="Research architecture">
        <span className="architecture-label">Research path</span>
        {architecture.map((step, index) => (
          <div key={step}>
            <strong>{step}</strong>
            {index < architecture.length - 1 && <span aria-hidden="true">→</span>}
          </div>
        ))}
      </section>
    </div>
  )
}
