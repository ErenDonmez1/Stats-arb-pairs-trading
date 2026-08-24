import type {
  ExperimentListItem,
  ExperimentPage,
  ExperimentSummary,
  HealthResponse,
  MetaResponse,
} from '../types/api'

export const DEMO_NOTICE =
  'Demo data — synthetic research output for interface demonstration.'

const SYNTHETIC_WARNING =
  'Synthetic demo output for interface demonstration only; it is not historical investment performance.'

const BASELINE_EXPERIMENT: ExperimentSummary = {
  run_id: 'demo-daily-baseline-2025-01',
  research_content_digest:
    '1a42d0e78a6d9e24d97b4f3a84952d45b833a7979c673f9677cb52d48da4e91c',
  price_content_digest:
    '4c93d7825894d9ab7e98922f405562269c036154123ebf670d3e4c8d44204a7e',
  research_pipeline_version: '10A.1',
  configuration_snapshot_version: 1,
  experiment_schema_version: 1,
  experiment_name: 'Synthetic Daily Baseline',
  created_at: '2025-01-15T16:30:00Z',
  configuration_digest:
    '0d549704d325407edacf437e6c4cedba6d9b811e57b371f2a2700927bc158885',
  pipeline_status: 'COMPLETED',
  selected_pair: {
    pair_id: 'SYNTH_A / SYNTH_B',
    symbol_y: 'SYNTH_A',
    symbol_x: 'SYNTH_B',
    rank: 1,
    alpha: 0.184,
    beta: 1.073,
    cointegration_statistic: -4.31,
    cointegration_pvalue: 0.0027,
    corrected_pvalue: 0.0162,
    half_life: 8.6,
    hurst: 0.38,
  },
  screening: {
    candidate_count: 15,
    selected_count: 2,
    selection_scope: 'formation_only_grouped_universe',
    ranking_policy: 'corrected_pvalue_then_half_life',
  },
  diagnostic: {
    stage: 'COMPLETED',
    scope: 'full_sample_diagnostic_only',
    total_return: 0.128,
    annualized_return: 0.081,
    annualized_volatility: 0.112,
    sharpe_ratio: 0.72,
    sortino_ratio: 1.06,
    maximum_drawdown: -0.071,
    calmar_ratio: 1.14,
    trade_count: 18,
    total_rows: 1260,
    finite_beta_rows: 1200,
    positive_execution_beta_rows: 1198,
    non_positive_execution_beta_rows: 2,
    finite_signal_rows: 1180,
    entry_execution_unavailable_rows_due_to_beta: 1,
    signal_observation_coverage: 0.9365,
    beta_execution_policy: 'posterior_beta_available_after_decision_row',
  },
  walk_forward: {
    stage: 'COMPLETED',
    calendar_analytics_status: 'AVAILABLE',
    fold_count: 9,
    completed_fold_count: 8,
    no_selection_fold_count: 1,
    insufficient_data_fold_count: 0,
    scheduled_oos_observations: 756,
    scheduled_eligible_oos_observations: 756,
    selected_oos_observations: 546,
    no_selection_oos_observations: 84,
    unavailable_oos_observations: 0,
    selection_coverage: 0.7222,
    evaluated_start_label: '2022-01-03',
    evaluated_end_label: '2024-12-31',
    capital_policy: 'equal_capital_reset',
    aggregate_return_policy: 'time_weighted_equal_capital_reset',
    inactive_capital_policy: 'zero_return_cash_for_no_selection_rows',
    selection_coverage_denominator: 'scheduled_eligible_oos_observations',
    aggregate_dollar_pnl_available: false,
    aggregate_trade_dollar_metrics_available: false,
    calendar_oos_total_return: 0.046,
    calendar_oos_annualized_return: 0.021,
    calendar_oos_annualized_volatility: 0.087,
    calendar_oos_sharpe_ratio: 0.24,
    calendar_oos_sortino_ratio: 0.33,
    calendar_oos_maximum_drawdown: -0.064,
    calendar_oos_calmar_ratio: 0.33,
    calendar_oos_report_observations: 756,
  },
  robustness: {
    stage: 'COMPLETED',
    baseline_scenario_id: 'baseline',
    scenario_count: 18,
    completed_scenarios: 17,
    analytically_unavailable_scenarios: 1,
    invalid_scenarios: 0,
    failed_scenarios: 0,
    common_horizon_structurally_available: true,
    common_horizon_fully_observed: true,
    common_horizon_analytics_available: true,
    common_horizon_analytics_status: 'AVAILABLE',
    common_horizon_observations: 504,
    common_horizon_eligible_scenario_count: 17,
    headline_metric_basis: 'calendar_oos_common_horizon',
    distribution_policy: 'predefined_scenarios_no_oos_selection',
    tested_dimensions: [
      'formation_window',
      'trading_window',
      'entry_z',
      'exit_z',
      'transaction_cost_bps',
    ],
    untested_material_dimensions: [
      'point_in_time_universe_membership',
      'market_impact_and_capacity',
    ],
    provenance_warnings: [SYNTHETIC_WARNING],
  },
  validation: {
    stage: 'COMPLETED',
    primary_availability: 'AVAILABLE',
    overall_availability: 'AVAILABLE',
    probabilistic_sharpe_probability: 0.68,
    minimum_track_record_observations: 410,
    fold_consistency_availability: 'AVAILABLE',
    multiple_testing_total_configurations: 18,
    multiple_testing_valid_comparable_configurations: 17,
    multiple_testing_valid_pvalue_count: 17,
    multiple_testing_unavailable_pvalue_count: 1,
    multiple_testing_eligible_hypothesis_count: 18,
    purpose: 'diagnostic_uncertainty_estimation_not_parameter_selection',
    provenance_warnings: [SYNTHETIC_WARNING],
  },
  configuration_snapshot: {
    data: {
      interval: '1d',
      min_coverage: 0.95,
      limited_forward_fill: 1,
    },
    screening: {
      min_observations: 252,
      fdr_threshold: 0.05,
    },
    strategy: {
      hedge_lookback: 60,
      zscore_lookback: 40,
      entry_z: 2.0,
      exit_z: 0.5,
      stop_z: 3.5,
    },
    synthetic_demo: true,
  },
  metadata: {
    data_classification: 'SYNTHETIC_DEMO',
    display_purpose: 'interface_demonstration',
    generated_from_market_data: false,
    deterministic_fixture: true,
  },
  provenance: {
    source: 'frontend_local_fixture',
    provider: null,
    synthetic_demo: true,
    causal_claim_scope: 'software_path_demonstration_only',
    performance_claim: 'none',
  },
  warnings: [
    SYNTHETIC_WARNING,
    'The caller-supplied universe is not represented as point-in-time survivorship-free.',
    'Execution costs are a research approximation and do not model market impact.',
  ],
}

const ROBUSTNESS_EXPERIMENT: ExperimentSummary = {
  ...BASELINE_EXPERIMENT,
  run_id: 'demo-cost-stress-2024-09',
  research_content_digest:
    '8d7c1f9be505d30f4cbd2ea56e194581633fc4cb5452c4df0e513b605a1e9102',
  price_content_digest:
    'a36560f8ad63f110d5210a4b22f22e26f003665a01f50ee3fb3b4a260bf326ea',
  experiment_name: 'Synthetic Robustness Example',
  created_at: '2024-09-30T16:30:00Z',
  configuration_digest:
    'b7d72d5bbf4197009f27abeaf593856034d9f2b71d72dbfeab4b76fb60af7208',
  selected_pair: {
    pair_id: 'SYNTH_C / SYNTH_D',
    symbol_y: 'SYNTH_C',
    symbol_x: 'SYNTH_D',
    rank: 1,
    alpha: -0.092,
    beta: 0.884,
    cointegration_statistic: -3.87,
    cointegration_pvalue: 0.0071,
    corrected_pvalue: 0.0355,
    half_life: 13.2,
    hurst: 0.44,
  },
  screening: {
    ...BASELINE_EXPERIMENT.screening,
    candidate_count: 10,
    selected_count: 1,
  },
  diagnostic: {
    ...BASELINE_EXPERIMENT.diagnostic,
    total_return: 0.094,
    annualized_return: 0.058,
    annualized_volatility: 0.124,
    sharpe_ratio: 0.47,
    sortino_ratio: 0.63,
    maximum_drawdown: -0.083,
    calmar_ratio: 0.7,
    trade_count: 14,
    positive_execution_beta_rows: 1200,
    non_positive_execution_beta_rows: 0,
    entry_execution_unavailable_rows_due_to_beta: 0,
  },
  walk_forward: {
    ...BASELINE_EXPERIMENT.walk_forward,
    completed_fold_count: 7,
    no_selection_fold_count: 2,
    selected_oos_observations: 462,
    no_selection_oos_observations: 168,
    selection_coverage: 0.6111,
    calendar_oos_total_return: -0.028,
    calendar_oos_annualized_return: -0.013,
    calendar_oos_annualized_volatility: 0.091,
    calendar_oos_sharpe_ratio: -0.14,
    calendar_oos_sortino_ratio: -0.18,
    calendar_oos_maximum_drawdown: -0.089,
    calendar_oos_calmar_ratio: -0.15,
  },
  robustness: {
    ...BASELINE_EXPERIMENT.robustness,
    scenario_count: 24,
    completed_scenarios: 21,
    analytically_unavailable_scenarios: 2,
    invalid_scenarios: 1,
    common_horizon_eligible_scenario_count: 21,
    provenance_warnings: [
      SYNTHETIC_WARNING,
      'Scenario metrics are diagnostics and are not used to promote an OOS winner.',
    ],
  },
  validation: {
    ...BASELINE_EXPERIMENT.validation,
    probabilistic_sharpe_probability: 0.41,
    minimum_track_record_observations: null,
    multiple_testing_total_configurations: 24,
    multiple_testing_valid_comparable_configurations: 21,
    multiple_testing_valid_pvalue_count: 21,
    multiple_testing_unavailable_pvalue_count: 3,
    multiple_testing_eligible_hypothesis_count: 24,
  },
  configuration_snapshot: {
    ...BASELINE_EXPERIMENT.configuration_snapshot,
    transaction_cost_stress_bps: [1, 3, 5, 8],
    synthetic_demo: true,
  },
  metadata: {
    ...BASELINE_EXPERIMENT.metadata,
    fixture_variant: 'cost_and_parameter_stress',
  },
  provenance: {
    ...BASELINE_EXPERIMENT.provenance,
    scenario_generation: 'fixed_before_synthetic_oos_evaluation',
  },
  warnings: [
    SYNTHETIC_WARNING,
    'This fixture deliberately includes weak OOS evidence to demonstrate neutral reporting.',
    'No scenario is identified as a winner, validated strategy, or deployment candidate.',
  ],
}

const DEMO_EXPERIMENTS: readonly ExperimentSummary[] = [
  BASELINE_EXPERIMENT,
  ROBUSTNESS_EXPERIMENT,
]

export const DEMO_HEALTH: HealthResponse = { status: 'ok' }

export const DEMO_META: MetaResponse = {
  research_pipeline_version: '10A.1',
  configuration_snapshot_version: 1,
  experiment_schema_version: 1,
}

function cloneDemo<T>(value: T): T {
  return structuredClone(value)
}

function toListItem(experiment: ExperimentSummary): ExperimentListItem {
  return {
    run_id: experiment.run_id,
    research_content_digest: experiment.research_content_digest,
    experiment_name: experiment.experiment_name,
    created_at: experiment.created_at,
    pipeline_status: experiment.pipeline_status,
    selected_pair_id: experiment.selected_pair?.pair_id ?? null,
    diagnostic_in_sample_total_return: experiment.diagnostic.total_return,
    diagnostic_in_sample_sharpe_ratio: experiment.diagnostic.sharpe_ratio,
    diagnostic_in_sample_maximum_drawdown: experiment.diagnostic.maximum_drawdown,
    diagnostic_in_sample_trade_count: experiment.diagnostic.trade_count,
    walk_forward_calendar_oos_total_return:
      experiment.walk_forward.calendar_oos_total_return,
    walk_forward_calendar_oos_sharpe_ratio:
      experiment.walk_forward.calendar_oos_sharpe_ratio,
    walk_forward_calendar_oos_maximum_drawdown:
      experiment.walk_forward.calendar_oos_maximum_drawdown,
    walk_forward_stage: experiment.walk_forward.stage,
    robustness_stage: experiment.robustness.stage,
    validation_stage: experiment.validation.stage,
  }
}

export function getDemoExperimentPage(limit: number, offset: number): ExperimentPage {
  const pageItems = DEMO_EXPERIMENTS.slice(offset, offset + limit).map(toListItem)
  return cloneDemo({
    items: pageItems,
    limit,
    offset,
    count: pageItems.length,
    total: DEMO_EXPERIMENTS.length,
  })
}

export function getDemoExperiment(runId: string): ExperimentSummary | null {
  const experiment = DEMO_EXPERIMENTS.find((item) => item.run_id === runId)
  return experiment ? cloneDemo(experiment) : null
}
