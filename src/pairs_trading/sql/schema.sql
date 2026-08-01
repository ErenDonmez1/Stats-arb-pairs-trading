CREATE TABLE IF NOT EXISTS prices (
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    adjusted_close DOUBLE NOT NULL,
    source VARCHAR NOT NULL,
    loaded_at TIMESTAMP NOT NULL,
    CONSTRAINT prices_date_symbol_source_unique UNIQUE (date, symbol, source)
);

CREATE TABLE IF NOT EXISTS data_quality_reports (
    run_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    total_observations BIGINT NOT NULL,
    valid_observations BIGINT NOT NULL,
    missing_or_invalid BIGINT NOT NULL,
    non_positive BIGINT NOT NULL,
    coverage DOUBLE NOT NULL,
    stale_fraction DOUBLE,
    first_valid TIMESTAMP,
    last_valid TIMESTAMP,
    forward_filled BIGINT NOT NULL,
    retained BOOLEAN NOT NULL,
    loaded_at TIMESTAMP NOT NULL,
    CONSTRAINT data_quality_run_symbol_unique UNIQUE (run_id, symbol)
);
