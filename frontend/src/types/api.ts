export interface HealthResponse {
  status: 'ok'
}

export interface MetaResponse {
  research_pipeline_version: string
  configuration_snapshot_version: number
  experiment_schema_version: number
}

export interface ExperimentListItem {
  run_id: string
  research_content_digest: string
  experiment_name: string
  created_at: string
  pipeline_status: string
  selected_pair_id: string | null
  diagnostic_in_sample_total_return: number | null
  diagnostic_in_sample_sharpe_ratio: number | null
  diagnostic_in_sample_maximum_drawdown: number | null
  diagnostic_in_sample_trade_count: number | null
  walk_forward_calendar_oos_total_return: number | null
  walk_forward_calendar_oos_sharpe_ratio: number | null
  walk_forward_calendar_oos_maximum_drawdown: number | null
  walk_forward_stage: string
  robustness_stage: string
  validation_stage: string
}

export interface ExperimentPage {
  items: ExperimentListItem[]
  limit: number
  offset: number
  count: number
  total: number
}

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

export type JsonObject = { [key: string]: JsonValue }

export interface SelectedPair {
  pair_id: string
  symbol_y: string
  symbol_x: string
  rank: number
  alpha: number | null
  beta: number | null
  cointegration_statistic: number | null
  cointegration_pvalue: number | null
  corrected_pvalue: number | null
  half_life: number | null
  hurst: number | null
}

export interface ScreeningSummary {
  candidate_count: number
  selected_count: number
  selection_scope: string
  ranking_policy: string
}

export interface DiagnosticSummary {
  stage: string
  scope: string
  total_return: number | null
  annualized_return: number | null
  annualized_volatility: number | null
  sharpe_ratio: number | null
  sortino_ratio: number | null
  maximum_drawdown: number | null
  calmar_ratio: number | null
  trade_count: number | null
  total_rows: number | null
  finite_beta_rows: number | null
  positive_execution_beta_rows: number | null
  non_positive_execution_beta_rows: number | null
  finite_signal_rows: number | null
  entry_execution_unavailable_rows_due_to_beta: number | null
  signal_observation_coverage: number | null
  beta_execution_policy: string | null
}

export interface WalkForwardSummary {
  stage: string
  calendar_analytics_status: string | null
  fold_count: number | null
  completed_fold_count: number | null
  no_selection_fold_count: number | null
  insufficient_data_fold_count: number | null
  scheduled_oos_observations: number | null
  scheduled_eligible_oos_observations: number | null
  selected_oos_observations: number | null
  no_selection_oos_observations: number | null
  unavailable_oos_observations: number | null
  selection_coverage: number | null
  evaluated_start_label: string | null
  evaluated_end_label: string | null
  capital_policy: string | null
  aggregate_return_policy: string | null
  inactive_capital_policy: string | null
  selection_coverage_denominator: string | null
  aggregate_dollar_pnl_available: boolean | null
  aggregate_trade_dollar_metrics_available: boolean | null
  calendar_oos_total_return: number | null
  calendar_oos_annualized_return: number | null
  calendar_oos_annualized_volatility: number | null
  calendar_oos_sharpe_ratio: number | null
  calendar_oos_sortino_ratio: number | null
  calendar_oos_maximum_drawdown: number | null
  calendar_oos_calmar_ratio: number | null
  calendar_oos_report_observations: number | null
}

export interface RobustnessSummary {
  stage: string
  baseline_scenario_id: string | null
  scenario_count: number | null
  completed_scenarios: number | null
  analytically_unavailable_scenarios: number | null
  invalid_scenarios: number | null
  failed_scenarios: number | null
  common_horizon_structurally_available: boolean | null
  common_horizon_fully_observed: boolean | null
  common_horizon_analytics_available: boolean | null
  common_horizon_analytics_status: string | null
  common_horizon_observations: number | null
  common_horizon_eligible_scenario_count: number | null
  headline_metric_basis: string | null
  distribution_policy: string | null
  tested_dimensions: string[]
  untested_material_dimensions: string[]
  provenance_warnings: string[]
}

export interface ValidationSummary {
  stage: string
  primary_availability: string | null
  overall_availability: string | null
  probabilistic_sharpe_probability: number | null
  minimum_track_record_observations: number | null
  fold_consistency_availability: string | null
  multiple_testing_total_configurations: number | null
  multiple_testing_valid_comparable_configurations: number | null
  multiple_testing_valid_pvalue_count: number | null
  multiple_testing_unavailable_pvalue_count: number | null
  multiple_testing_eligible_hypothesis_count: number | null
  purpose: string | null
  provenance_warnings: string[]
}

export interface ExperimentSummary {
  run_id: string
  research_content_digest: string
  price_content_digest: string
  research_pipeline_version: string
  configuration_snapshot_version: number
  experiment_schema_version: number
  experiment_name: string
  created_at: string
  configuration_digest: string
  pipeline_status: string
  selected_pair: SelectedPair | null
  screening: ScreeningSummary
  diagnostic: DiagnosticSummary
  walk_forward: WalkForwardSummary
  robustness: RobustnessSummary
  validation: ValidationSummary
  configuration_snapshot: JsonObject
  metadata: JsonObject
  provenance: JsonObject
  warnings: string[]
}
