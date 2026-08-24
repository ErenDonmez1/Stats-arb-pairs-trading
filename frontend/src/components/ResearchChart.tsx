import type { ReactNode } from 'react'
import { AreaChart, DatabaseZap } from 'lucide-react'

interface ChartFrameProps {
  eyebrow: string
  title: string
  description?: string
  badge?: string
  className?: string
  children: ReactNode
}

export function ChartFrame({
  eyebrow,
  title,
  description,
  badge,
  className = '',
  children,
}: ChartFrameProps) {
  return (
    <section className={`chart-frame ${className}`.trim()}>
      <header className="chart-frame-header">
        <div>
          <p className="section-eyebrow">{eyebrow}</p>
          <h2>{title}</h2>
          {description && <p>{description}</p>}
        </div>
        {badge && <span className="chart-badge">{badge}</span>}
      </header>
      <div className="chart-frame-body">{children}</div>
    </section>
  )
}

interface TooltipEntry {
  name?: string
  value?: number | string
  color?: string
}

interface ResearchTooltipProps {
  active?: boolean
  label?: string | number
  payload?: readonly TooltipEntry[]
  valueFormatter?: (value: number, name: string) => string
}

export function ResearchTooltip({
  active,
  label,
  payload,
  valueFormatter,
}: ResearchTooltipProps) {
  if (!active || !payload?.length) return null

  return (
    <div className="research-tooltip">
      <p>{label}</p>
      {payload.map((entry) => {
        const name = entry.name ?? 'Value'
        const numeric = typeof entry.value === 'number' ? entry.value : Number(entry.value)
        const value = Number.isFinite(numeric)
          ? valueFormatter?.(numeric, name) ?? numeric.toFixed(2)
          : String(entry.value ?? '—')
        return (
          <div key={name}>
            <span style={{ backgroundColor: entry.color }} aria-hidden="true" />
            <small>{name}</small>
            <strong>{value}</strong>
          </div>
        )
      })}
    </div>
  )
}

export function ChartUnavailable({
  compact = false,
  title = 'Time-series unavailable',
  message = 'Detailed time-series were not persisted for this experiment.',
}: {
  compact?: boolean
  title?: string
  message?: string
}) {
  return (
    <div className={`chart-unavailable${compact ? ' is-compact' : ''}`}>
      <span className="chart-unavailable-icon" aria-hidden="true">
        {compact ? <AreaChart size={19} /> : <DatabaseZap size={23} />}
      </span>
      <div>
        <span className="chart-unavailable-kicker">Data availability</span>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
    </div>
  )
}

export function SyntheticChartLabel() {
  return <span className="synthetic-chart-label">Synthetic demo series</span>
}
