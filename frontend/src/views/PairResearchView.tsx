import { AlertCircle, ArrowLeft, RefreshCw } from 'lucide-react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
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
import { StatusBadge } from '../components/StatusBadge'
import {
  DEMO_PAIR_CANDIDATES,
  type DemoResearchCharts,
} from '../data/demoCharts'
import type { ExperimentSummary } from '../types/api'
import { formatCount, formatPercent, formatRatio } from '../utils/format'

type LoadState = 'idle' | 'loading' | 'ready' | 'error'

interface PairResearchViewProps {
  pairId: string | null
  experiment: ExperimentSummary | null
  charts: DemoResearchCharts | null
  state: LoadState
  error: string | null
  demoMode: boolean
  onOpenScreener: () => void
  onRetry: () => void
}

function monthTick(value: string): string {
  return value.slice(0, 7)
}

export function PairResearchView({
  pairId,
  experiment,
  charts,
  state,
  error,
  demoMode,
  onOpenScreener,
  onRetry,
}: PairResearchViewProps) {
  if (!pairId) {
    return (
      <div className="content-wrap page-content">
        <section className="state-panel">
          <h1>Select a candidate pair</h1>
          <p>Choose a screening row to inspect its relationship diagnostics.</p>
          <button className="quiet-button" type="button" onClick={onOpenScreener}>Open screener</button>
        </section>
      </div>
    )
  }

  if (state === 'loading') {
    return (
      <div className="content-wrap page-content" aria-live="polite" aria-busy="true">
        <div className="detail-loading-block" />
        <div className="detail-loading-grid">
          {Array.from({ length: 4 }).map((_, index) => <div className="detail-loading-card" key={index} />)}
        </div>
      </div>
    )
  }

  if (state === 'error') {
    return (
      <div className="content-wrap page-content">
        <section className="state-panel" role="alert">
          <AlertCircle size={26} aria-hidden="true" />
          <h1>Pair research unavailable</h1>
          <p>{error ?? 'The persisted experiment could not be loaded.'}</p>
          <div className="state-actions">
            <button className="quiet-button" type="button" onClick={onOpenScreener}>
              <ArrowLeft size={14} aria-hidden="true" /> Screener
            </button>
            <button className="quiet-button is-accent" type="button" onClick={onRetry}>
              <RefreshCw size={14} aria-hidden="true" /> Retry
            </button>
          </div>
        </section>
      </div>
    )
  }

  const candidate = demoMode
    ? DEMO_PAIR_CANDIDATES.find((item) => item.pairId === pairId)
    : undefined
  const pair = experiment?.selected_pair?.pair_id === pairId ? experiment.selected_pair : null
  const status = candidate?.status ?? experiment?.pipeline_status ?? 'UNAVAILABLE'
  const events = charts?.points.filter((point) => point.event !== null) ?? []
  const entryLong = events.filter((point) => point.event === 'ENTER_LONG')
  const entryShort = events.filter((point) => point.event === 'ENTER_SHORT')
  const exits = events.filter((point) => point.event === 'EXIT')
  const chartUnavailableMessage = demoMode
    ? 'This rejected synthetic candidate does not have a detailed research-series fixture.'
    : 'Detailed time-series were not persisted for this experiment.'

  return (
    <div className="content-wrap page-content pair-research-page">
      <button className="text-button" type="button" onClick={onOpenScreener}>
        <ArrowLeft size={14} aria-hidden="true" /> Back to screener
      </button>

      <header className="pair-research-heading">
        <div>
          <p className="section-eyebrow">Relationship diagnostics</p>
          <div className="pair-title-row">
            <h1>{pairId}</h1>
            <StatusBadge status={status} />
            {demoMode && <span className="demo-chip">Synthetic demo</span>}
          </div>
          <p>Why is this pair statistically interesting, and how does the spread behave?</p>
        </div>
        <div className="pair-context-meta">
          <span>Economic group<strong>{candidate?.group ?? 'Not persisted'}</strong></span>
          <span>Run ID<strong>{experiment?.run_id ?? 'Candidate only'}</strong></span>
        </div>
      </header>

      <section className="research-facts" aria-label="Pair screening facts">
        <div><span>Corrected p-value</span><strong>{formatRatio(pair?.corrected_pvalue)}</strong></div>
        <div><span>Half-life</span><strong>{formatRatio(candidate?.halfLife ?? pair?.half_life)}</strong></div>
        <div><span>Hurst</span><strong>{formatRatio(candidate?.hurst ?? pair?.hurst)}</strong></div>
        <div><span>Hedge beta</span><strong>{formatRatio(pair?.beta)}</strong></div>
        <div><span>Formation rank</span><strong>{formatCount(candidate?.rank ?? pair?.rank)}</strong></div>
        <div><span>Screening result</span><strong>{candidate?.screeningStatus ?? status}</strong></div>
      </section>

      {charts ? (
        <>
          <div className="pair-chart-grid primary-pair-grid">
            <ChartFrame
              eyebrow="Relative price behavior"
              title="Normalized price relationship"
              description="Indexed synthetic prices reveal co-movement while retaining temporary relative dislocations."
              badge="Synthetic demo series"
              className="normalized-price-chart"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={charts.points} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
                  <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                  <XAxis dataKey="date" tickFormatter={monthTick} stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} minTickGap={42} />
                  <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} domain={['dataMin - 1', 'dataMax + 1']} />
                  <Tooltip content={<ResearchTooltip />} />
                  <Line type="monotone" dataKey="normalizedY" name={candidate?.symbolY ?? 'Asset Y'} stroke={CHART_COLOURS.primary} strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="normalizedX" name={candidate?.symbolX ?? 'Asset X'} stroke={CHART_COLOURS.secondary} strokeWidth={1.8} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartFrame>

            <ChartFrame
              eyebrow="Dynamic exposure"
              title="Hedge ratio"
              description="Causal posterior beta used only after its observation row."
              className="hedge-chart"
            >
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={charts.points} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                  <defs>
                    <linearGradient id="beta-fill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={CHART_COLOURS.accent} stopOpacity={0.24} />
                      <stop offset="100%" stopColor={CHART_COLOURS.accent} stopOpacity={0.01} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                  <XAxis dataKey="date" tickFormatter={(value: string) => value.slice(2, 7)} stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} minTickGap={36} />
                  <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} domain={['dataMin - 0.02', 'dataMax + 0.02']} />
                  <Tooltip content={<ResearchTooltip />} />
                  <Area type="stepAfter" dataKey="beta" name="Hedge beta" stroke={CHART_COLOURS.accent} fill="url(#beta-fill)" strokeWidth={1.8} />
                </AreaChart>
              </ResponsiveContainer>
            </ChartFrame>
          </div>

          <div className="pair-chart-grid secondary-pair-grid">
            <ChartFrame
              eyebrow="Mean-reverting residual"
              title="Log-price spread"
              description="Residual from the hedge relationship with its rolling reference mean."
              className="spread-chart"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={charts.points} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
                  <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                  <XAxis dataKey="date" tickFormatter={monthTick} stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} minTickGap={42} />
                  <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} tickFormatter={(value: number) => value.toFixed(2)} />
                  <Tooltip content={<ResearchTooltip valueFormatter={(value) => value.toFixed(4)} />} />
                  <Line type="monotone" dataKey="spread" name="Spread" stroke={CHART_COLOURS.primary} strokeWidth={1.8} dot={false} />
                  <Line type="monotone" dataKey="spreadMean" name="Rolling mean" stroke={CHART_COLOURS.muted} strokeDasharray="4 4" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartFrame>

            <ChartFrame
              eyebrow="Signal state"
              title="Causal z-score and decisions"
              description="Threshold decisions are generated at observation time and executed on a later row."
              className="zscore-chart"
            >
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={charts.points} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
                  <CartesianGrid stroke={CHART_COLOURS.grid} vertical={false} />
                  <XAxis dataKey="date" tickFormatter={monthTick} stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} minTickGap={42} />
                  <YAxis stroke={CHART_COLOURS.axis} tickLine={false} axisLine={false} domain={[-4, 4]} />
                  <Tooltip content={<ResearchTooltip />} />
                  <ReferenceLine y={charts.entryZ} stroke={CHART_COLOURS.accent} strokeDasharray="5 5" />
                  <ReferenceLine y={-charts.entryZ} stroke={CHART_COLOURS.accent} strokeDasharray="5 5" />
                  <ReferenceLine y={charts.exitZ} stroke={CHART_COLOURS.muted} strokeDasharray="3 5" />
                  <ReferenceLine y={-charts.exitZ} stroke={CHART_COLOURS.muted} strokeDasharray="3 5" />
                  <ReferenceLine y={charts.stopZ} stroke={CHART_COLOURS.negative} strokeDasharray="2 5" />
                  <ReferenceLine y={-charts.stopZ} stroke={CHART_COLOURS.negative} strokeDasharray="2 5" />
                  <Line type="monotone" dataKey="zscore" name="Z-score" stroke={CHART_COLOURS.primary} strokeWidth={1.8} dot={false} />
                  <Scatter data={entryLong} dataKey="eventValue" name="Enter long" fill={CHART_COLOURS.positive} />
                  <Scatter data={entryShort} dataKey="eventValue" name="Enter short" fill={CHART_COLOURS.negative} />
                  <Scatter data={exits} dataKey="eventValue" name="Exit" fill={CHART_COLOURS.accent} />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartFrame>
          </div>

          <div className="signal-legend" aria-label="Signal chart legend">
            <span><i className="legend-dot is-long" /> Long-spread entry</span>
            <span><i className="legend-dot is-short" /> Short-spread entry</span>
            <span><i className="legend-dot is-exit" /> Mean-reversion exit</span>
            <span>Entry ±{charts.entryZ.toFixed(1)}</span>
            <span>Exit ±{charts.exitZ.toFixed(1)}</span>
            <span>Stop ±{charts.stopZ.toFixed(1)}</span>
          </div>
        </>
      ) : (
        <ChartFrame
          eyebrow="Relationship histories"
          title="Detailed pair time-series"
          description="Summary statistics remain available above."
          className="full-unavailable-chart"
        >
          <ChartUnavailable message={chartUnavailableMessage} />
        </ChartFrame>
      )}

      {experiment && (
        <section className="pair-provenance-line">
          <span>Signal observation coverage <strong>{formatPercent(experiment.diagnostic.signal_observation_coverage)}</strong></span>
          <span>Finite beta rows <strong>{formatCount(experiment.diagnostic.finite_beta_rows)}</strong></span>
          <span>Execution policy <strong>{experiment.diagnostic.beta_execution_policy ?? '—'}</strong></span>
        </section>
      )}
    </div>
  )
}
