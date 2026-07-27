# Statistical-Arbitrage Pairs-Trading Research System

This project is a Python research platform for discovering economically related
securities, modelling a stable relative-value relationship, and evaluating
mean-reversion trades with realistic out-of-sample controls. It is intended for
research and education, not investment advice or live trading.

## Planned architecture

The project uses a conventional `src` layout. Future milestones will add
independent modules for configuration, market-data ingestion and validation,
statistical pair screening, hedge-ratio estimation, signal generation,
transaction-cost-aware backtesting, portfolio analytics, and visualization.
A Colab-compatible notebook will orchestrate those tested modules rather than
duplicating their implementation.

```text
.
├── data/raw/             # Local provider downloads; contents are ignored
├── notebooks/            # Colab and exploratory notebooks
├── reports/              # Generated charts and research outputs
├── src/pairs_trading/    # Installable Python package
└── tests/                # Automated tests
```

Only package scaffolding is present in Milestone 1. Quantitative research
components will be introduced and tested in later milestones.

## Development setup

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Optional notebook and dashboard environments can be installed with
`.[notebook]` and `.[dashboard]`, respectively.

