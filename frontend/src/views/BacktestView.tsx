import { AlertTriangle } from 'lucide-react'
import {
  Area,
  AreaChart,
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
import type { DemoResearchCharts } from '../data/demoCharts'
import type { ExperimentSummary } from '../types/api'
import { formatCount, formatPercent, formatRatio } from '../utils/format'

interface BacktestViewProps {
  experiment: ExperimentSummary | null
  charts: DemoResearchCharts | null
  demoMode: boolean
}

function monthTick(value: string): string {
  return value.slice(0, 7)
}

export function BacktestView({ experiment, charts, demoMode }: BacktestViewProps) {
  if (!experiment) {
    return (
      <div className="content-wrap page-content">
        <section className="state-panel">
          <h1>No experiment selected</h1>
          <p>Select a persisted pair from the screener before opening OOS analysis.</p>
        </section>
      </div>
    )
  }

  const oos = experiment.walk_forward
  const diagnostic = experiment.diagnostic

  return (
    <div className="content-wrap page-content backtest-page">
      <header className="research-page-heading">
        <div>
          <p className="section-eyebrow">Execution-aware evidence</p>
          <h1>Backtest / OOS</h1>
          <p>How did the selected relationship behave in calendar walk-forward evaluation?</p>
        </div>
        <div className="heading-context">
          {demoMode && <span className="demo-chip">Synthetic demo</span>}
          <span>{experiment.selected_pair?.pair_id ?? 'No selected pair'}</span>
          <span>{oos.evaluated_start_label ?? '—'} → {oos.evaluated_end_label ?? '—'}</span>
        </div>
      </header>

      <div className="evidence-band">
        <div>
          <p className="section-eyebrow">Primary evidence</p>
          <strong>Calendar walk-forward OOS</strong>
        </div>
        <p>No-selection folds remain cash; unavailable calendar observations are not repaired.</p>
      </div>

      <ChartFrame
        eyebrow="Calendar performance"
        title="OOS equity curve"
        description="Indexed calendar wealth path under equal-capital-reset fold semantics."
        badge={demoMode ? 'Synthetic demo series' : undefined}
        className="backtest-equity-chart"
      >
        {charts ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={charts.points} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
              <defs>
                <linearGradient id="backtest-equity-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CHART_COLOURS.primary} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={CHART_COLOURS.primary} stopOpacity={0.01} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
              <XAxis dataKey="date" tickFormatter={monthTick} stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} minTickGap={46} />
              <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} domain={['dataMin - 1', 'dataMax + 1']} />
              <Tooltip content={<ResearchTooltip valueFormatter={(value) => value.toFixed(2)} />} />
              <Area type="monotone" dataKey="equity" name="OOS equity" stroke={CHART_COLOURS.primary} fill="url(#backtest-equity-fill)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <ChartUnavailable />
        )}
      </ChartFrame>

      <section className="performance-ledger" aria-label="Walk-forward performance metrics">
        <div><span>Total return</span><strong>{formatPercent(oos.calendar_oos_total_return)}</strong></div>
        <div><span>Annualized return</span><strong>{formatPercent(oos.calendar_oos_annualized_return)}</strong></div>
        <div><span>Volatility</span><strong>{formatPercent(oos.calendar_oos_annualized_volatility)}</strong></div>
        <div><span>Sharpe</span><strong>{formatRatio(oos.calendar_oos_sharpe_ratio)}</strong></div>
        <div><span>Sortino</span><strong>{formatRatio(oos.calendar_oos_sortino_ratio)}</strong></div>
        <div><span>Max drawdown</span><strong>{formatPercent(oos.calendar_oos_maximum_drawdown)}</strong></div>
        <div><span>Calmar</span><strong>{formatRatio(oos.calendar_oos_calmar_ratio)}</strong></div>
        <div><span>OOS trades</span><strong>{formatCount(charts?.oosTradeCount)}</strong></div>
        <div><span>Selection coverage</span><strong>{formatPercent(oos.selection_coverage)}</strong></div>
      </section>

      <div className="backtest-secondary-grid">
        <ChartFrame eyebrow="Path risk" title="Drawdown" className="compact-chart">
          {charts ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={charts.points} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                <XAxis dataKey="date" tickFormatter={monthTick} stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} minTickGap={38} />
                <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} />
                <Tooltip content={<ResearchTooltip valueFormatter={(value) => `${(value * 100).toFixed(2)}%`} />} />
                <Area type="monotone" dataKey="drawdown" name="Drawdown" stroke={CHART_COLOURS.negative} fill={CHART_COLOURS.negative} fillOpacity={0.17} strokeWidth={1.6} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <ChartUnavailable compact />}
        </ChartFrame>

        <ChartFrame eyebrow="Rolling behavior" title="Rolling Sharpe" className="compact-chart">
          {charts ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={charts.points} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                <XAxis dataKey="date" tickFormatter={monthTick} stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} minTickGap={38} />
                <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} />
                <ReferenceLine y={0} stroke={CHART_COLOURS.muted} />
                <Tooltip content={<ResearchTooltip />} />
                <Line type="monotone" dataKey="rollingSharpe" name="Rolling Sharpe" stroke={CHART_COLOURS.accent} strokeWidth={1.8} dot={false} connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : <ChartUnavailable compact />}
        </ChartFrame>
      </div>

      <div className="fold-analysis-grid">
        <ChartFrame
          eyebrow="Fold attribution"
          title="Fold returns"
          description="Zero-height no-selection folds are retained as cash, not removed from the calendar record."
          className="fold-chart"
        >
          {charts ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={charts.foldReturns} margin={{ top: 12, right: 8, left: -8, bottom: 0 }}>
                <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                <XAxis dataKey="fold" stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} />
                <ReferenceLine y={0} stroke={CHART_COLOURS.muted} />
                <Tooltip content={<ResearchTooltip valueFormatter={(value) => `${(value * 100).toFixed(2)}%`} />} />
                <Bar dataKey="return" name="Fold return" radius={[2, 2, 0, 0]}>
                  {charts.foldReturns.map((fold) => (
                    <Cell
                      key={fold.fold}
                      fill={fold.status === 'NO_SELECTION'
                        ? CHART_COLOURS.muted
                        : fold.return >= 0
                          ? CHART_COLOURS.positive
                          : CHART_COLOURS.negative}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <ChartUnavailable compact />}
        </ChartFrame>

        <section className="fold-status-panel">
          <div className="panel-heading-compact">
            <div>
              <p className="section-eyebrow">Calendar coverage</p>
              <h2>Fold status</h2>
            </div>
          </div>
          <dl className="compact-definition-list">
            <div><dt>Completed folds</dt><dd>{formatCount(oos.completed_fold_count)} / {formatCount(oos.fold_count)}</dd></div>
            <div><dt>No-selection folds</dt><dd>{formatCount(oos.no_selection_fold_count)}</dd></div>
            <div><dt>Insufficient-data folds</dt><dd>{formatCount(oos.insufficient_data_fold_count)}</dd></div>
            <div><dt>Selected observations</dt><dd>{formatCount(oos.selected_oos_observations)}</dd></div>
            <div><dt>No-selection observations</dt><dd>{formatCount(oos.no_selection_oos_observations)}</dd></div>
            <div><dt>Unavailable observations</dt><dd>{formatCount(oos.unavailable_oos_observations)}</dd></div>
          </dl>
        </section>
      </div>

      <section className="diagnostic-comparison">
        <header>
          <div>
            <p className="section-eyebrow">Secondary context</p>
            <h2>In-sample diagnostic</h2>
          </div>
          <div className="diagnostic-warning">
            <AlertTriangle size={15} aria-hidden="true" />
            Diagnostic only — not out-of-sample evidence.
          </div>
        </header>
        <div className="diagnostic-inline-metrics">
          <span>Total return<strong>{formatPercent(diagnostic.total_return)}</strong></span>
          <span>Sharpe<strong>{formatRatio(diagnostic.sharpe_ratio)}</strong></span>
          <span>Volatility<strong>{formatPercent(diagnostic.annualized_volatility)}</strong></span>
          <span>Max drawdown<strong>{formatPercent(diagnostic.maximum_drawdown)}</strong></span>
          <span>Trades<strong>{formatCount(diagnostic.trade_count)}</strong></span>
        </div>
      </section>
    </div>
  )
}
