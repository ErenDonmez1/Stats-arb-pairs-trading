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

CREATE TABLE IF NOT EXISTS pair_screening_results (
    run_id VARCHAR NOT NULL,
    formation_start DATE NOT NULL,
    formation_end DATE NOT NULL,
    symbol_y VARCHAR NOT NULL,
    symbol_x VARCHAR NOT NULL,
    group_name VARCHAR,
    observations BIGINT NOT NULL,
    alpha DOUBLE,
    beta DOUBLE,
    spread_standard_deviation DOUBLE,
    cointegration_statistic DOUBLE,
    cointegration_pvalue DOUBLE,
    corrected_pvalue DOUBLE,
    adf_statistic DOUBLE,
    adf_pvalue DOUBLE,
    half_life DOUBLE,
    hurst DOUBLE,
    selected BOOLEAN NOT NULL,
    rank BIGINT,
    rejection_reasons VARCHAR,
    loaded_at TIMESTAMP NOT NULL,
    half_life_was_infinite BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT screening_run_pair_unique UNIQUE (run_id, symbol_y, symbol_x),
    CONSTRAINT screening_canonical_pair CHECK (symbol_y < symbol_x),
    CONSTRAINT screening_non_negative_observations CHECK (observations >= 0),
    CONSTRAINT screening_rank_policy CHECK (
        (selected AND rank IS NOT NULL AND rank > 0)
        OR (NOT selected AND rank IS NULL)
    ),
    CONSTRAINT screening_infinite_half_life_policy CHECK (
        NOT half_life_was_infinite OR half_life IS NULL
    )
);
