CREATE TABLE IF NOT EXISTS prices (
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    adjusted_close DOUBLE NOT NULL,
    observed BOOLEAN NOT NULL,
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
    cointegration_critical_values VARCHAR NOT NULL,
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

CREATE TABLE IF NOT EXISTS research_experiments (
    run_id VARCHAR PRIMARY KEY,
    research_content_digest VARCHAR NOT NULL,
    price_content_digest VARCHAR NOT NULL,
    configuration_digest VARCHAR NOT NULL,
    research_pipeline_version VARCHAR NOT NULL,
    configuration_snapshot_version BIGINT NOT NULL,
    experiment_schema_version BIGINT NOT NULL,
    experiment_name VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    pipeline_status VARCHAR NOT NULL,
    configuration_snapshot VARCHAR NOT NULL,
    metadata VARCHAR NOT NULL,
    provenance VARCHAR NOT NULL,
    warnings VARCHAR NOT NULL,
    CONSTRAINT research_digest_lengths CHECK (
        length(research_content_digest) = 64
        AND length(price_content_digest) = 64
        AND length(configuration_digest) = 64
    ),
    CONSTRAINT research_supported_versions CHECK (
        configuration_snapshot_version > 0 AND experiment_schema_version > 0
    ),
    CONSTRAINT research_pipeline_status CHECK (pipeline_status = 'COMPLETED')
);

CREATE TABLE IF NOT EXISTS research_experiment_summaries (
    run_id VARCHAR PRIMARY KEY,
    selected_pair_id VARCHAR,
    symbol_y VARCHAR,
    symbol_x VARCHAR,
    selected_rank BIGINT,
    screening_candidate_count BIGINT NOT NULL,
    screening_selected_count BIGINT NOT NULL,
    screening_selection_scope VARCHAR NOT NULL,
    screening_ranking_policy VARCHAR NOT NULL,
    alpha DOUBLE,
    beta DOUBLE,
    cointegration_statistic DOUBLE,
    cointegration_pvalue DOUBLE,
    corrected_pvalue DOUBLE,
    half_life DOUBLE,
    hurst DOUBLE,
    analytics_stage VARCHAR NOT NULL,
    diagnostic_in_sample_total_return DOUBLE,
    diagnostic_in_sample_annualized_return DOUBLE,
    diagnostic_in_sample_annualized_volatility DOUBLE,
    diagnostic_in_sample_sharpe_ratio DOUBLE,
    diagnostic_in_sample_sortino_ratio DOUBLE,
    diagnostic_in_sample_maximum_drawdown DOUBLE,
    diagnostic_in_sample_calmar_ratio DOUBLE,
    diagnostic_in_sample_trade_count BIGINT,
    diagnostic_total_rows BIGINT,
    diagnostic_finite_beta_rows BIGINT,
    diagnostic_positive_execution_beta_rows BIGINT,
    diagnostic_non_positive_execution_beta_rows BIGINT,
    diagnostic_finite_signal_rows BIGINT,
    diagnostic_entry_execution_unavailable_rows_due_to_beta BIGINT,
    diagnostic_signal_observation_coverage DOUBLE,
    diagnostic_beta_execution_policy VARCHAR,
    walk_forward_stage VARCHAR NOT NULL,
    walk_forward_calendar_analytics_status VARCHAR,
    walk_forward_fold_count BIGINT,
    walk_forward_completed_fold_count BIGINT,
    walk_forward_no_selection_fold_count BIGINT,
    walk_forward_insufficient_data_fold_count BIGINT,
    walk_forward_scheduled_oos_observations BIGINT,
    walk_forward_scheduled_eligible_oos_observations BIGINT,
    walk_forward_selected_oos_observations BIGINT,
    walk_forward_no_selection_oos_observations BIGINT,
    walk_forward_unavailable_oos_observations BIGINT,
    walk_forward_selection_coverage DOUBLE,
    walk_forward_evaluated_start_label VARCHAR,
    walk_forward_evaluated_end_label VARCHAR,
    walk_forward_capital_policy VARCHAR,
    walk_forward_aggregate_return_policy VARCHAR,
    walk_forward_inactive_capital_policy VARCHAR,
    walk_forward_selection_coverage_denominator VARCHAR,
    walk_forward_aggregate_dollar_pnl_available BOOLEAN,
    walk_forward_aggregate_trade_dollar_metrics_available BOOLEAN,
    walk_forward_calendar_oos_total_return DOUBLE,
    walk_forward_calendar_oos_annualized_return DOUBLE,
    walk_forward_calendar_oos_annualized_volatility DOUBLE,
    walk_forward_calendar_oos_sharpe_ratio DOUBLE,
    walk_forward_calendar_oos_sortino_ratio DOUBLE,
    walk_forward_calendar_oos_maximum_drawdown DOUBLE,
    walk_forward_calendar_oos_calmar_ratio DOUBLE,
    walk_forward_calendar_oos_report_observations BIGINT,
    robustness_stage VARCHAR NOT NULL,
    robustness_baseline_scenario_id VARCHAR,
    robustness_scenario_count BIGINT,
    robustness_completed_scenarios BIGINT,
    robustness_analytically_unavailable_scenarios BIGINT,
    robustness_invalid_scenarios BIGINT,
    robustness_failed_scenarios BIGINT,
    robustness_common_horizon_structurally_available BOOLEAN,
    robustness_common_horizon_fully_observed BOOLEAN,
    robustness_common_horizon_analytics_available BOOLEAN,
    robustness_common_horizon_analytics_status VARCHAR,
    robustness_common_horizon_observations BIGINT,
    robustness_common_horizon_eligible_scenario_count BIGINT,
    robustness_headline_metric_basis VARCHAR,
    robustness_distribution_policy VARCHAR,
    robustness_tested_dimensions VARCHAR,
    robustness_untested_material_dimensions VARCHAR,
    robustness_provenance_warning_summary VARCHAR,
    validation_stage VARCHAR NOT NULL,
    validation_primary_availability VARCHAR,
    validation_overall_availability VARCHAR,
    probabilistic_sharpe_probability DOUBLE,
    minimum_track_record_observations BIGINT,
    fold_consistency_availability VARCHAR,
    multiple_testing_total_configurations BIGINT,
    multiple_testing_valid_comparable_configurations BIGINT,
    multiple_testing_valid_pvalue_count BIGINT,
    multiple_testing_unavailable_pvalue_count BIGINT,
    multiple_testing_eligible_hypothesis_count BIGINT,
    validation_purpose VARCHAR,
    validation_provenance_warning_summary VARCHAR,
    CONSTRAINT research_summary_experiment_fk FOREIGN KEY (run_id)
        REFERENCES research_experiments(run_id),
    CONSTRAINT research_selected_pair_shape CHECK (
        (selected_pair_id IS NULL AND symbol_y IS NULL AND symbol_x IS NULL
            AND selected_rank IS NULL)
        OR
        (selected_pair_id IS NOT NULL AND symbol_y IS NOT NULL
            AND symbol_x IS NOT NULL AND selected_rank IS NOT NULL
            AND selected_rank > 0)
    ),
    CONSTRAINT research_screening_counts CHECK (
        screening_candidate_count >= 0
        AND screening_selected_count >= 0
        AND screening_selected_count <= screening_candidate_count
    ),
    CONSTRAINT research_stage_values CHECK (
        analytics_stage IN ('COMPLETED', 'UNAVAILABLE')
        AND walk_forward_stage IN ('COMPLETED', 'UNAVAILABLE')
        AND robustness_stage IN ('COMPLETED', 'NOT_REQUESTED', 'UNAVAILABLE')
        AND validation_stage IN ('COMPLETED', 'NOT_REQUESTED', 'UNAVAILABLE')
    ),
    CONSTRAINT research_walk_forward_counts CHECK (
        walk_forward_stage <> 'COMPLETED'
        OR (
            walk_forward_fold_count >= 0
            AND walk_forward_completed_fold_count >= 0
            AND walk_forward_no_selection_fold_count >= 0
            AND walk_forward_insufficient_data_fold_count >= 0
            AND walk_forward_fold_count = walk_forward_completed_fold_count
                + walk_forward_no_selection_fold_count
                + walk_forward_insufficient_data_fold_count
            AND walk_forward_scheduled_oos_observations =
                walk_forward_scheduled_eligible_oos_observations
                + walk_forward_unavailable_oos_observations
            AND walk_forward_scheduled_eligible_oos_observations =
                walk_forward_selected_oos_observations
                + walk_forward_no_selection_oos_observations
        )
    ),
    CONSTRAINT research_robustness_counts CHECK (
        robustness_stage <> 'COMPLETED'
        OR robustness_scenario_count = robustness_completed_scenarios
            + robustness_analytically_unavailable_scenarios
            + robustness_invalid_scenarios
            + robustness_failed_scenarios
    ),
    CONSTRAINT research_multiple_testing_counts CHECK (
        validation_stage <> 'COMPLETED'
        OR multiple_testing_total_configurations =
            multiple_testing_valid_pvalue_count
            + multiple_testing_unavailable_pvalue_count
    )
);
