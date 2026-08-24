import {
  Activity,
  BarChart3,
  Database,
  GitBranch,
  Network,
  Scale,
  Server,
  ShieldCheck,
} from 'lucide-react'

const researchPath = [
  {
    icon: Database,
    title: 'Observed market data',
    text: 'Request-aware ingestion, explicit cleaning and observed-versus-imputed provenance.',
  },
  {
    icon: Network,
    title: 'Formation-only screening',
    text: 'Economic groups, correlation pre-screening, Engle–Granger tests and FDR control.',
  },
  {
    icon: Activity,
    title: 'Causal signal formation',
    text: 'Prior-window estimation, stateful thresholds and future-invariance checks.',
  },
  {
    icon: Scale,
    title: 'Execution-aware accounting',
    text: 'Lagged decisions, costs, financing, borrow, rebalancing and reconciled ledgers.',
  },
  {
    icon: GitBranch,
    title: 'Walk-forward separation',
    text: 'Formation and trading windows remain explicit, including no-selection cash periods.',
  },
  {
    icon: BarChart3,
    title: 'Robustness diagnostics',
    text: 'Predefined scenario grids and uncertainty estimates without automatic OOS promotion.',
  },
  {
    icon: ShieldCheck,
    title: 'Portfolio risk controls',
    text: 'Static sleeves, exposure aggregation and causal risk-policy intent states.',
  },
  {
    icon: Server,
    title: 'Reproducible presentation',
    text: 'Immutable summaries, content digests, DuckDB persistence and a read-only API.',
  },
]

export function MethodologyView() {
  return (
    <div className="content-wrap page-content methodology-page">
      <header className="research-page-heading methodology-heading">
        <div>
          <p className="section-eyebrow">Research methodology</p>
          <h1>Evidence before narrative</h1>
          <p>
            A modular path from observed prices to explicitly scoped diagnostic,
            out-of-sample, and portfolio-risk evidence.
          </p>
        </div>
        <div className="methodology-principles">
          <span>Causal</span>
          <span>Reconciled</span>
          <span>Reproducible</span>
        </div>
      </header>

      <section className="methodology-ledger" aria-label="Research workflow">
        {researchPath.map(({ icon: Icon, title, text }, index) => (
          <article key={title}>
            <span className="method-index">{String(index + 1).padStart(2, '0')}</span>
            <Icon size={18} strokeWidth={1.7} aria-hidden="true" />
            <h2>{title}</h2>
            <p>{text}</p>
          </article>
        ))}
      </section>

      <div className="methodology-notes-grid">
        <section>
          <p className="section-eyebrow">Interpretation discipline</p>
          <h2>Diagnostic and OOS evidence stay distinct</h2>
          <p>
            Full-sample diagnostics are retained for research context. Calendar
            walk-forward results receive primary visual emphasis and preserve
            unavailable observations rather than selectively dropping them.
          </p>
        </section>
        <section>
          <p className="section-eyebrow">Scope</p>
          <h2>Research software, not a trading product</h2>
          <p>
            The system does not provide live execution, investment advice, or a
            claim of future profitability. Demo-mode results are fixed synthetic
            fixtures used only to demonstrate the interface.
          </p>
        </section>
      </div>
    </div>
  )
}
