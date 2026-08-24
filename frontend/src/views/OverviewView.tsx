import {
  Activity,
  Archive,
  Boxes,
  Database,
  Gauge,
  GitBranch,
  Network,
  Server,
  ShieldCheck,
} from 'lucide-react'
import type { MetaResponse } from '../types/api'

type BackendState = 'checking' | 'online' | 'unavailable'

const architecture = [
  { label: 'Market data', icon: Database },
  { label: 'Pair screening', icon: Network },
  { label: 'Signal engine', icon: Activity },
  { label: 'Backtest', icon: Gauge },
  { label: 'Walk-forward', icon: GitBranch },
  { label: 'Robustness', icon: ShieldCheck },
  { label: 'DuckDB', icon: Archive },
  { label: 'FastAPI', icon: Server },
  { label: 'React', icon: Boxes },
]

interface OverviewViewProps {
  backendState: BackendState
  metadata: MetaResponse | null
  experimentCount: number | null
  experimentsUnavailable: boolean
  demoMode: boolean
}

export function OverviewView({
  backendState,
  metadata,
  experimentCount,
  experimentsUnavailable,
  demoMode,
}: OverviewViewProps) {
  const presentationState = demoMode ? 'online' : backendState

  return (
    <div className="content-wrap">
      <section className="hero-panel" aria-labelledby="overview-title">
        <div className="hero-copy">
          <p className="section-eyebrow">Research platform / Overview</p>
          <h1 id="overview-title">Stat-Arb Research Platform</h1>
          <p>
            Causal pairs-trading research, validation and portfolio-risk
            infrastructure.
          </p>
        </div>
        <div className="hero-index" aria-label="Platform classification">
          <span>01</span>
          <p>Offline research</p>
          <strong>Reproducible</strong>
        </div>
      </section>

      <section className="metric-strip" aria-label="Platform metadata">
        <article className="metric-card">
          <p>{demoMode ? 'Data mode' : 'Backend status'}</p>
          <strong className="metric-status">
            <span className={`status-dot status-${presentationState}`} aria-hidden="true" />
            {demoMode
              ? 'Synthetic demo'
              : backendState === 'online'
                ? 'Operational'
                : backendState === 'checking'
                  ? 'Checking'
                  : 'Unavailable'}
          </strong>
          <span>{demoMode ? 'Deterministic local fixtures' : 'FastAPI read layer'}</span>
        </article>
        <article className="metric-card">
          <p>{demoMode ? 'Demo experiments' : 'Persisted experiments'}</p>
          <strong>{experimentsUnavailable ? '—' : experimentCount ?? '—'}</strong>
          <span>
            {experimentsUnavailable
              ? 'Registry unavailable'
              : demoMode
                ? 'Synthetic UI examples'
                : 'Canonical summaries'}
          </span>
        </article>
        <article className="metric-card">
          <p>Experiment schema</p>
          <strong>{metadata?.experiment_schema_version ?? '—'}</strong>
          <span>Immutable contract</span>
        </article>
        <article className="metric-card">
          <p>Config snapshot</p>
          <strong>{metadata?.configuration_snapshot_version ?? '—'}</strong>
          <span>Versioned inputs</span>
        </article>
      </section>

      <section className="architecture-panel" aria-labelledby="architecture-title">
        <div className="section-heading">
          <div>
            <p className="section-eyebrow">System architecture</p>
            <h2 id="architecture-title">Research flow</h2>
          </div>
          <p>One causal path from validated marks to persisted evidence.</p>
        </div>

        <div className="architecture-flow">
          {architecture.map(({ label, icon: Icon }, index) => (
            <div className="architecture-step" key={label}>
              <div className="architecture-card">
                <span>{String(index + 1).padStart(2, '0')}</span>
                <Icon size={19} strokeWidth={1.7} aria-hidden="true" />
                <strong>{label}</strong>
              </div>
              {index < architecture.length - 1 && (
                <span className="flow-arrow" aria-hidden="true">→</span>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="overview-grid">
        <article className="brief-card">
          <p className="section-eyebrow">Research controls</p>
          <h3>Causal by construction</h3>
          <p>
            Formation-only screening, lagged decisions and observed-price
            execution policies keep diagnostic and OOS evidence separate.
          </p>
        </article>
        <article className="brief-card">
          <p className="section-eyebrow">Evidence standard</p>
          <h3>Unavailable stays unavailable</h3>
          <p>
            Persisted nulls, stage status and provenance are shown directly—
            without zero-filling or fabricated performance series.
          </p>
        </article>
      </section>
    </div>
  )
}
