import {
  BarChart3,
  Database,
  GitBranch,
  Network,
  Scale,
  Server,
  ShieldCheck,
  Workflow,
} from 'lucide-react'

const capabilities = [
  {
    icon: Network,
    title: 'Relationship research',
    text: 'Economic-group screening, Engle–Granger cointegration, FDR correction, half-life and Hurst diagnostics.',
  },
  {
    icon: Workflow,
    title: 'Dynamic estimation',
    text: 'Static and rolling OLS plus transparent Kalman-filter hedge estimation on validated log prices.',
  },
  {
    icon: GitBranch,
    title: 'Causal signals',
    text: 'Prior-window z-scores, explicit state transitions, cooldowns, stops and lagged execution decisions.',
  },
  {
    icon: Scale,
    title: 'Execution accounting',
    text: 'Position sizing, marked-to-market P&L, commissions, slippage, borrow, financing and trade ledgers.',
  },
  {
    icon: BarChart3,
    title: 'OOS evidence',
    text: 'Formation-only selection, calendar walk-forward returns, robustness grids and statistical uncertainty diagnostics.',
  },
  {
    icon: ShieldCheck,
    title: 'Portfolio controls',
    text: 'Static sleeve allocation, exposure aggregation, concentration analysis and deterministic portfolio risk states.',
  },
  {
    icon: Database,
    title: 'Reproducibility',
    text: 'DuckDB persistence, content digests, configuration snapshots, immutable experiment summaries and provenance.',
  },
  {
    icon: Server,
    title: 'Presentation layer',
    text: 'A read-only FastAPI contract and typed React client expose evidence without triggering research computations.',
  },
]

export function MethodologyView() {
  return (
    <div className="content-wrap page-content">
      <header className="page-heading methodology-heading">
        <div>
          <p className="section-eyebrow">About / Methodology</p>
          <h1>Evidence before narrative</h1>
          <p>
            A modular research system for studying relative-value behavior while
            keeping diagnostic, out-of-sample and provenance claims explicit.
          </p>
        </div>
      </header>

      <section className="methodology-grid" aria-label="Research capabilities">
        {capabilities.map(({ icon: Icon, title, text }, index) => (
          <article className="method-card" key={title}>
            <div>
              <span>{String(index + 1).padStart(2, '0')}</span>
              <Icon size={20} strokeWidth={1.7} aria-hidden="true" />
            </div>
            <h2>{title}</h2>
            <p>{text}</p>
          </article>
        ))}
      </section>

      <section className="disclaimer-panel">
        <p className="section-eyebrow">Scope</p>
        <h2>Research software, not a trading product</h2>
        <p>
          This project is research software and does not represent investment advice
          or a production trading system. Historical diagnostics and walk-forward
          results are not live performance and do not imply future profitability.
        </p>
      </section>
    </div>
  )
}
