# Statistical-Arbitrage Pairs-Trading Research System

This repository is an offline-testable Python research platform for screening
economically related securities, estimating relative-value spreads, generating
causal trading decisions, and evaluating pair and multi-pair portfolios. It is
for research and education, not investment advice or live trading.

## Architecture

The installable `src/pairs_trading` package separates typed configuration,
market-data validation, statistical estimation, cointegration screening,
signals, execution-aware backtesting, walk-forward evaluation, robustness and
uncertainty analysis, portfolio construction, and portfolio risk controls.
DuckDB persistence stores market prices, data-quality reports, and screening
results; notebooks and reports remain orchestration/output locations rather
than alternate implementations.

```text
.
|-- config.yaml             # Typed research defaults
|-- data/raw/               # Ignored provider cache/data
|-- notebooks/              # Colab and exploratory orchestration
|-- reports/                # Generated research outputs
|-- src/pairs_trading/      # Installable, tested Python package
`-- tests/                  # Offline unit and integration tests
```

## Configuration and screening

`load_config()` safely loads immutable dataclasses and rejects unknown keys.
The active screening settings are `min_observations`, `fdr_threshold`,
`max_half_life`, and `hurst_threshold`; `screening_kwargs_from_config()` is the
canonical adapter to the existing scalar `screen_pairs()` API. Formation-window
length is used by fold construction and is not passed as a screening threshold.
Legacy screening fields that are not implemented are rejected rather than
silently ignored.

## Data and persistence contracts

Yahoo Finance downloads use explicit adjusted-price arguments and exact
request-aware CSV cache metadata. Open-ended requests include the current UTC
calendar date in their cache identity, while explicit historical end dates are
stable. Cache metadata records the symbol universe, horizon, interval, source,
retrieval time, format version, and adjustment policy.

Cleaning treats nonfinite and nonpositive prices as unavailable and records a
cell-level `observed_price_mask`. Limited forward fills are valuation-only:
they must not authorize executions. DuckDB stores this Boolean provenance in
the same row and transaction as each price and reconstructs the aligned mask on
load. Existing databases without the provenance column are deliberately
incompatible until explicitly recreated or migrated; unknown historical
provenance is never assumed observed. Explicit symbol loads are strict by
default, with `strict=False` available when a complete requested column set
containing unavailable values is required.

`MarketDataLoader.clean()` computes coverage and complete-row filtering over
the entire frame supplied to it. A historical simulation must therefore make
universe-eligibility decisions from formation-only data; cleaning a
future-inclusive horizon before selection can introduce look-ahead.

DuckDB uses deterministic validated upserts for market and screening data and
an immutable registry for canonical experiment summaries. Full raw-dataset
snapshots and concurrent-writer coordination remain outside the current scope.

## Read-only research API

The FastAPI presentation layer exposes persisted experiment summaries without
running research or accepting SQL. It reads `data/research.duckdb` by default;
set `PAIRS_TRADING_DB_PATH` to select another server-controlled database. Local
React development at `http://localhost:5173` is allowed by default. Override
that allowlist with a comma-separated `PAIRS_TRADING_CORS_ORIGINS` value.

```bash
uvicorn pairs_trading.api:app --reload
```

Interactive OpenAPI documentation is available at `/docs`. The API persists
summary metrics and provenance, but not equity curves, drawdown series, trade
ledgers, bootstrap replicates, or other detailed time-series artifacts.

## React dashboard

The `frontend/` application is a responsive React and TypeScript dashboard for
the read-only research API. It presents experiment identity, screening,
in-sample diagnostics, calendar walk-forward OOS metrics, validation status,
and provenance without fabricating unavailable time-series data.

Start the backend from the repository root:

```bash
uvicorn pairs_trading.api:app --reload
```

Then start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The local dashboard is available at `http://localhost:5173`. It calls
`http://localhost:8000` by default; set `VITE_API_BASE_URL` in a frontend `.env`
file to use another API origin. See `frontend/.env.example`.

## Development setup

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Ordinary tests mock market providers and require no internet access. Optional
notebook and dashboard environments can be installed with `.[notebook]` and
`.[dashboard]`.
