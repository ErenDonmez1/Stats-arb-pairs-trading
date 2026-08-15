"""Focused tests for Milestone 8C statistical evidence diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, norm, skew

import pairs_trading.robustness as robustness_module
import pairs_trading.validation as validation_module
from pairs_trading.analytics import (
    annualized_return,
    annualized_volatility,
    maximum_drawdown,
    sharpe_ratio,
    total_return,
)
from pairs_trading.robustness import (
    ParameterScenario,
    RobustnessResult,
    ScenarioStatus,
    generate_parameter_scenarios,
    run_sensitivity_analysis,
)
from pairs_trading.validation import (
    BootstrapMetricResult,
    ConfidenceInterval,
    FoldConsistencyMetrics,
    MultipleTestingDiagnostics,
    RegimePerformance,
    StatisticalValidationResult,
    ValidationAvailability,
    analyze_fold_consistency,
    analyze_regime_consistency,
    bootstrap_performance_metrics,
    build_statistical_validation_report,
    minimum_track_record_length,
    moving_block_bootstrap,
    multiple_testing_diagnostics,
    probabilistic_sharpe_ratio,
)
from pairs_trading.walkforward import (
    WalkForwardAnalyticsStatus,
    WalkForwardFold,
    WalkForwardFoldResult,
    WalkForwardResult,
    WalkForwardStatus,
)


def _returns(values: list[float] | None = None) -> pd.Series:
    if values is None:
        values = [0.012, -0.006, 0.009, -0.003, 0.011, 0.002, -0.004, 0.008]
    return pd.Series(values, index=pd.RangeIndex(len(values), name="row"), name="oos_return")


def _prices(rows: int = 100) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "AAA": 100.0 * np.exp(0.001 * positions),
            "BBB": 90.0 * np.exp(0.0008 * positions),
        },
        index=pd.RangeIndex(rows, name="row"),
    )


def _scenario(**overrides: Any) -> ParameterScenario:
    values: dict[str, Any] = {
        "scenario_id": "baseline",
        "entry_z": 1.0,
        "exit_z": 0.25,
        "stop_z": 3.0,
        "zscore_lookback": 5,
        "formation_window": 60,
        "trading_window": 8,
        "screening_min_observations": 50,
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "financing_rate": 0.0,
        "borrow_rate_y": 0.0,
        "borrow_rate_x": 0.0,
    }
    values.update(overrides)
    return ParameterScenario(**values)


def _fold_result(
    fold_id: int,
    status: WalkForwardStatus,
    values: list[float] | None,
    *,
    trade_count: int = 0,
) -> WalkForwardFoldResult:
    start = (fold_id - 1) * 4
    fold = WalkForwardFold(
        fold_id=fold_id,
        formation_start_position=start,
        formation_end_position=start + 3,
        trading_start_position=start + 4,
        trading_end_position=start + 7,
        formation_start_label=start,
        formation_end_label=start + 3,
        trading_start_label=start + 4,
        trading_end_label=start + 7,
    )
    series = (
        pd.Series(values, index=pd.RangeIndex(start + 4, start + 4 + len(values)), name="oos_return")
        if values is not None
        else pd.Series(dtype=float, name="oos_return")
    )
    return WalkForwardFoldResult(
        fold=fold,
        status=status,
        message=("unavailable" if status is WalkForwardStatus.INSUFFICIENT_DATA else None),
        candidates_screened=0,
        screening_results=(),
        selected_symbol_y=None,
        selected_symbol_x=None,
        screening_rank=None,
        corrected_pvalue=None,
        selected_screening_result=None,
        frozen_alpha=None,
        frozen_beta=None,
        backtest=None,
        performance_report=None,
        analytics_status=(
            WalkForwardAnalyticsStatus.AVAILABLE
            if status is WalkForwardStatus.COMPLETED
            else WalkForwardAnalyticsStatus.NOT_APPLICABLE
        ),
        analytics_error=None,
        oos_returns=series,
        formation_observations=4,
        trading_observations=4,
        trade_count=trade_count,
        starting_capital=100_000.0,
        ending_capital=100_000.0,
        eligible_symbols=("AAA", "BBB"),
        group_snapshot=None,
    )


def _walk_forward_result(
    calendar_values: list[float],
    *,
    folds: tuple[WalkForwardFoldResult, ...] = (),
    start: int = 60,
    universe_provenance: str = "fixture-universe",
    point_in_time_validated: bool = False,
) -> WalkForwardResult:
    index = pd.RangeIndex(start, start + len(calendar_values), name="row")
    calendar = pd.Series(calendar_values, index=index, name="calendar_oos_return")
    conditional = calendar.dropna().rename("conditional_oos_return")
    unavailable = int(calendar.isna().sum())
    available = len(calendar) - unavailable
    return WalkForwardResult(
        folds=folds,
        fold_count=len(folds),
        completed_fold_count=sum(item.status is WalkForwardStatus.COMPLETED for item in folds),
        no_selection_fold_count=sum(item.status is WalkForwardStatus.NO_SELECTION for item in folds),
        insufficient_data_fold_count=sum(item.status is WalkForwardStatus.INSUFFICIENT_DATA for item in folds),
        scheduled_oos_observations=len(calendar),
        scheduled_eligible_oos_observations=available,
        selected_oos_observations=available,
        no_selection_oos_observations=0,
        unavailable_oos_observations=unavailable,
        selection_coverage=(1.0 if available else float("nan")),
        conditional_oos_returns=conditional,
        calendar_oos_returns=calendar,
        conditional_performance_report=None,
        calendar_performance_report=None,
        conditional_analytics_status=WalkForwardAnalyticsStatus.UNAVAILABLE,
        calendar_analytics_status=WalkForwardAnalyticsStatus.UNAVAILABLE,
        conditional_analytics_error="fixture",
        calendar_analytics_error="fixture",
        capital_policy="equal_capital_reset",
        aggregate_return_policy="time_weighted_equal_capital_reset",
        inactive_capital_policy="zero_return_cash_for_no_selection_rows",
        selection_coverage_denominator="selected_plus_no_selection_scheduled_rows",
        aggregate_dollar_pnl_available=False,
        aggregate_trade_dollar_metrics_available=False,
        universe_provenance=universe_provenance,
        cleaning_provenance="fixture-cleaning",
        point_in_time_universe_validated=point_in_time_validated,
        provenance_warnings=("fixture provenance warning",),
        evaluated_start_position=start,
        evaluated_end_position=start + len(calendar) - 1,
        evaluated_start_label=index[0],
        evaluated_end_label=index[-1],
        discarded_terminal_rows=0,
    )


def _robustness_result(
    monkeypatch: pytest.MonkeyPatch,
    scenario_returns: dict[float, list[float]] | None = None,
    *,
    folds: tuple[WalkForwardFoldResult, ...] = (),
) -> RobustnessResult:
    if scenario_returns is None:
        scenario_returns = {1.0: _returns().tolist()}
    baseline = _scenario()
    scenarios = generate_parameter_scenarios(
        baseline,
        entry_z_values=sorted(scenario_returns),
    )

    def fake(
        prices: pd.DataFrame,
        formation_window: int,
        trading_window: int,
        **kwargs: Any,
    ) -> WalkForwardResult:
        return _walk_forward_result(
            scenario_returns[kwargs["entry_z"]],
            folds=(folds if kwargs["entry_z"] == 1.0 else ()),
        )

    monkeypatch.setattr(robustness_module, "run_walk_forward_analysis", fake)
    return run_sensitivity_analysis(_prices(), scenarios, "baseline")


def test_moving_block_bootstrap_uses_contiguous_source_blocks() -> None:
    source = _returns([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    samples = moving_block_bootstrap(source, 2, 20, random_seed=7)
    valid_blocks = {tuple(source.iloc[start : start + 2]) for start in range(5)}

    for sample in samples.to_numpy():
        assert tuple(sample[0:2]) in valid_blocks
        assert tuple(sample[2:4]) in valid_blocks
        assert tuple(sample[4:6]) in valid_blocks


def test_bootstrap_sample_length_matches_original() -> None:
    samples = moving_block_bootstrap(_returns(), 3, 11, random_seed=2)
    assert samples.shape == (11, 8)


def test_same_seed_produces_identical_bootstrap_samples() -> None:
    first = moving_block_bootstrap(_returns(), 3, 25, random_seed=42)
    second = moving_block_bootstrap(_returns(), 3, 25, random_seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_leave_point_estimates_unchanged() -> None:
    first = bootstrap_performance_metrics(_returns(), 2, 50, random_seed=1)
    second = bootstrap_performance_metrics(_returns(), 2, 50, random_seed=2)
    assert first.total_return.point_estimate == second.total_return.point_estimate
    assert first.sharpe_ratio.point_estimate == second.sharpe_ratio.point_estimate
    assert not first.bootstrap_samples.equals(second.bootstrap_samples)


@pytest.mark.parametrize("block_length", [0, -1, True, 1.5, "2", 9])
def test_invalid_block_lengths_are_rejected(block_length: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        moving_block_bootstrap(_returns(), block_length, 10)


@pytest.mark.parametrize("n_bootstrap", [0, -1, True, 1.5, "10"])
def test_invalid_bootstrap_counts_are_rejected(n_bootstrap: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        moving_block_bootstrap(_returns(), 2, n_bootstrap)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("random_seed", True),
        ("random_seed", -1),
        ("confidence_level", 0.0),
        ("confidence_level", 1.0),
        ("minimum_valid_fraction", 0.0),
        ("minimum_valid_fraction", 1.1),
    ],
)
def test_invalid_bootstrap_control_parameters_are_rejected(
    keyword: str,
    value: Any,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        bootstrap_performance_metrics(
            _returns(), 2, 10, **{keyword: value}
        )


def test_bootstrap_rejects_duplicate_index_and_infinite_returns() -> None:
    duplicate = _returns()
    duplicate.index = pd.Index([0, 0, 1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="unique index"):
        moving_block_bootstrap(duplicate, 2, 10)
    infinite = _returns()
    infinite.iloc[2] = np.inf
    with pytest.raises(ValueError, match="finite"):
        moving_block_bootstrap(infinite, 2, 10)


def test_missing_primary_returns_make_bootstrap_unavailable_without_dropping() -> None:
    source = _returns([0.01, np.nan, 0.02, np.nan])
    result = bootstrap_performance_metrics(source, 2, 20, random_seed=3)
    assert result.status is ValidationAvailability.UNAVAILABLE
    assert result.original_returns.isna().sum() == 2
    assert result.bootstrap_samples.empty
    assert result.total_return.valid_replicates == 0
    assert result.total_return.undefined_replicates == 20


def test_empty_primary_returns_produce_insufficient_bootstrap_result() -> None:
    source = pd.Series(dtype=float, name="calendar_oos_return")
    result = bootstrap_performance_metrics(source, 1, 10, random_seed=3)
    assert result.status is ValidationAvailability.INSUFFICIENT_DATA
    assert result.observations == 0
    assert result.bootstrap_samples.empty
    assert result.total_return.undefined_replicates == 10


def test_bootstrap_point_estimates_match_direct_analytics() -> None:
    source = _returns()
    result = bootstrap_performance_metrics(source, 2, 40, random_seed=4)
    assert result.total_return.point_estimate == pytest.approx(total_return(source))
    assert result.annualized_return.point_estimate == pytest.approx(annualized_return(source, 252))
    assert result.annualized_volatility.point_estimate == pytest.approx(annualized_volatility(source, 252))
    assert result.sharpe_ratio.point_estimate == pytest.approx(sharpe_ratio(source, 252))
    assert result.maximum_drawdown.point_estimate == pytest.approx(maximum_drawdown(source))


def test_percentile_interval_matches_deterministic_full_block_fixture() -> None:
    source = _returns()
    result = bootstrap_performance_metrics(source, len(source), 10, random_seed=5)
    metric = result.total_return
    assert isinstance(metric.interval, ConfidenceInterval)
    assert metric.bootstrap_median == pytest.approx(metric.point_estimate)
    assert metric.interval.lower == pytest.approx(metric.point_estimate)
    assert metric.interval.upper == pytest.approx(metric.point_estimate)


def test_bootstrap_valid_and_undefined_replicates_reconcile_with_catastrophe() -> None:
    source = _returns([-1.0, 0.02, 0.03, 0.01])
    result = bootstrap_performance_metrics(
        source, 1, 100, random_seed=9, minimum_valid_fraction=0.01
    )
    metric = result.total_return
    assert metric.valid_replicates + metric.undefined_replicates == 100
    assert metric.undefined_replicates > 0
    assert -1.0 in result.bootstrap_samples.to_numpy()
    assert result.original_returns.iloc[0] == -1.0


def test_too_few_valid_bootstrap_replicates_make_interval_unavailable() -> None:
    result = bootstrap_performance_metrics(
        _returns([-1.0, 0.02, 0.03, 0.01]),
        1,
        100,
        random_seed=9,
        minimum_valid_fraction=1.0,
    )
    assert result.total_return.status is ValidationAvailability.UNAVAILABLE
    assert result.total_return.interval is None


def test_bootstrap_uses_local_rng_without_changing_global_numpy_state() -> None:
    np.random.seed(1234)
    before = np.random.get_state()
    moving_block_bootstrap(_returns(), 2, 10, random_seed=17)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_psr_matches_manual_finite_sample_formula() -> None:
    source = _returns()
    result = probabilistic_sharpe_ratio(source, benchmark_sharpe=0.25)
    array = source.to_numpy()
    observed = float(array.mean() / array.std(ddof=1))
    benchmark = 0.25 / np.sqrt(252)
    gamma3 = float(skew(array, bias=False))
    gamma4 = float(kurtosis(array, fisher=False, bias=False))
    variance = 1.0 - gamma3 * observed + ((gamma4 - 1.0) / 4.0) * observed**2
    expected = norm.cdf((observed - benchmark) * np.sqrt(len(array) - 1) / np.sqrt(variance))
    assert result.status is ValidationAvailability.AVAILABLE
    assert result.probability == pytest.approx(expected)
    assert "per-period" in result.convention


def test_higher_psr_benchmark_reduces_probability() -> None:
    low = probabilistic_sharpe_ratio(_returns(), benchmark_sharpe=0.0)
    high = probabilistic_sharpe_ratio(_returns(), benchmark_sharpe=1.0)
    assert high.probability < low.probability


def test_constant_returns_produce_undefined_psr() -> None:
    result = probabilistic_sharpe_ratio(_returns([0.01] * 8))
    assert result.status is ValidationAvailability.UNAVAILABLE
    assert np.isnan(result.probability)


def test_missing_and_short_samples_have_explicit_psr_unavailability() -> None:
    missing = probabilistic_sharpe_ratio(_returns([0.01, np.nan, 0.02, 0.0]))
    short = probabilistic_sharpe_ratio(_returns([0.01, -0.01, 0.02]))
    assert missing.status is ValidationAvailability.UNAVAILABLE
    assert short.status is ValidationAvailability.INSUFFICIENT_DATA
    assert np.isnan(missing.probability)
    assert np.isnan(short.probability)


def test_invalid_psr_variance_term_is_handled_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation_module, "skew", lambda *args, **kwargs: 100.0)
    monkeypatch.setattr(validation_module, "kurtosis", lambda *args, **kwargs: 1.0)
    result = probabilistic_sharpe_ratio(_returns())
    assert result.status is ValidationAvailability.UNAVAILABLE
    assert np.isnan(result.probability)
    assert result.variance_term <= 0.0


def test_minimum_track_record_is_monotonic_in_confidence_level() -> None:
    source = _returns()
    lower = minimum_track_record_length(source, confidence_level=0.90)
    higher = minimum_track_record_length(source, confidence_level=0.99)
    assert lower.estimated_required_observations is not None
    assert higher.estimated_required_observations is not None
    assert higher.estimated_required_observations >= lower.estimated_required_observations


def test_minimum_track_record_matches_psr_rearrangement() -> None:
    source = _returns()
    psr = probabilistic_sharpe_ratio(source)
    result = minimum_track_record_length(source, confidence_level=0.95)
    difference = psr.observed_sharpe_per_period - psr.benchmark_sharpe_per_period
    expected = max(
        4,
        int(ceil(1.0 + psr.variance_term * (norm.ppf(0.95) / difference) ** 2)),
    )
    assert result.estimated_required_observations == expected


def test_sharpe_below_benchmark_does_not_fabricate_track_record() -> None:
    result = minimum_track_record_length(_returns(), benchmark_sharpe=20.0)
    assert result.status is ValidationAvailability.UNAVAILABLE
    assert result.estimated_required_observations is None
    assert result.sufficient_track_record is None


def _mixed_folds() -> tuple[WalkForwardFoldResult, ...]:
    return (
        _fold_result(1, WalkForwardStatus.COMPLETED, [0.10, 0.0, 0.0, 0.0], trade_count=1),
        _fold_result(2, WalkForwardStatus.COMPLETED, [-0.10, 0.0, 0.0, 0.0], trade_count=1),
        _fold_result(3, WalkForwardStatus.NO_SELECTION, None),
        _fold_result(4, WalkForwardStatus.INSUFFICIENT_DATA, None),
        _fold_result(5, WalkForwardStatus.COMPLETED, [0.20, 0.0, 0.0, 0.0], trade_count=2),
    )


def test_fold_consistency_counts_and_quantiles_retain_all_folds() -> None:
    folds = _mixed_folds()
    result = analyze_fold_consistency(_walk_forward_result(_returns().tolist(), folds=folds))
    expected_totals = np.array([0.10, -0.10, 0.0, 0.20])
    assert result.fold_count == 5
    assert result.observable_fold_count == 4
    assert result.positive_return_fold_count == 2
    assert result.negative_return_fold_count == 1
    assert result.zero_return_no_selection_fold_count == 1
    assert result.unavailable_fold_count == 1
    assert result.fraction_observable_folds_positive == 0.5
    assert result.median_fold_total_return == pytest.approx(np.quantile(expected_totals, 0.50))
    assert result.lower_quartile_fold_total_return == pytest.approx(np.quantile(expected_totals, 0.25))
    assert result.upper_quartile_fold_total_return == pytest.approx(np.quantile(expected_totals, 0.75))
    assert result.worst_fold_return == pytest.approx(-0.10)
    assert result.strongest_fold_return == pytest.approx(0.20)
    assert any(item.total_return < 0.0 for item in result.folds)


def test_fold_positive_concentration_and_ceiling_rule_are_exact() -> None:
    positive_folds = tuple(
        _fold_result(index + 1, WalkForwardStatus.COMPLETED, [value, 0.0, 0.0, 0.0])
        for index, value in enumerate([0.01, 0.02, 0.03, 0.04, 0.05, 0.15])
    )
    result = analyze_fold_consistency(_walk_forward_result(_returns().tolist(), folds=positive_folds))
    assert result.strongest_single_positive_fold_concentration == pytest.approx(0.15 / 0.30)
    assert result.strongest_twenty_percent_positive_fold_count == 2
    assert result.strongest_twenty_percent_positive_folds_concentration == pytest.approx(0.20 / 0.30)


def test_no_positive_fold_contribution_is_explicitly_undefined() -> None:
    folds = (
        _fold_result(1, WalkForwardStatus.COMPLETED, [-0.01, 0.0, 0.0, 0.0]),
        _fold_result(2, WalkForwardStatus.NO_SELECTION, None),
    )
    result = analyze_fold_consistency(_walk_forward_result(_returns().tolist(), folds=folds))
    assert np.isnan(result.strongest_single_positive_fold_concentration)
    assert np.isnan(result.strongest_twenty_percent_positive_folds_concentration)
    assert result.strongest_twenty_percent_positive_fold_count == 0


def test_regime_labels_require_exact_alignment() -> None:
    labels = pd.Series(["a"] * 8, index=pd.RangeIndex(1, 9))
    with pytest.raises(ValueError, match="exactly aligned"):
        analyze_regime_consistency(_returns(), labels)


@pytest.mark.parametrize("function", [analyze_fold_consistency, analyze_regime_consistency])
def test_fold_and_regime_analysis_reject_invalid_risk_free_rate(
    function: Any,
) -> None:
    if function is analyze_fold_consistency:
        arguments = (_walk_forward_result(_returns().tolist()),)
    else:
        source = _returns()
        arguments = (source, pd.Series(["a"] * len(source), index=source.index))
    with pytest.raises(ValueError, match="greater than -1"):
        function(*arguments, risk_free_rate=-1.0)


def test_regime_metrics_names_provenance_and_concentration_are_correct() -> None:
    source = _returns([0.04, 0.02, -0.01, -0.02, 0.03, 0.01])
    labels = pd.Series(["Expansion", "Expansion", "Stress", "Stress", "Recovery", "Recovery"], index=source.index)
    result = analyze_regime_consistency(source, labels)
    by_name = {item.regime_label: item for item in result.regimes}
    expansion = source.loc[labels == "Expansion"]
    assert [item.regime_label for item in result.regimes] == ["Expansion", "Stress", "Recovery"]
    assert by_name["Expansion"].total_return == pytest.approx(total_return(expansion))
    assert by_name["Expansion"].annualized_volatility == pytest.approx(annualized_volatility(expansion, 252))
    assert by_name["Expansion"].sharpe_ratio == pytest.approx(sharpe_ratio(expansion, 252))
    assert by_name["Expansion"].maximum_drawdown == pytest.approx(maximum_drawdown(expansion))
    assert result.worst_regime == "Stress"
    assert result.strongest_regime_positive_contribution_concentration == pytest.approx(0.06 / 0.10)
    assert not result.point_in_time_regime_labels_validated
    assert "caller-supplied" in result.provenance_warning
    assert not hasattr(result, "selected_regime")
    assert all(isinstance(item, RegimePerformance) for item in result.regimes)


def test_missing_regime_returns_are_unavailable_not_compressed() -> None:
    source = _returns([0.01, np.nan, 0.02, 0.0])
    labels = pd.Series(["a", "a", "b", "b"], index=source.index)
    result = analyze_regime_consistency(source, labels)
    assert result.status is ValidationAvailability.UNAVAILABLE
    assert result.returns.isna().sum() == 1
    assert result.regimes == ()


def test_multiple_testing_counts_mapping_and_adjustments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(
        monkeypatch,
        {
            1.0: [0.01, -0.01, 0.02, -0.005, 0.01, 0.0],
            1.5: [0.02, -0.01, 0.01, -0.005, 0.02, 0.0],
            2.0: [0.03, -0.01, 0.02, -0.005, 0.01, 0.0],
        },
    )
    probabilities = iter([0.99, 0.98, 0.90])

    def fake_psr(*args: Any, **kwargs: Any) -> Any:
        probability = next(probabilities)
        return replace(
            probabilistic_sharpe_ratio(_returns()),
            probability=probability,
        )

    monkeypatch.setattr(validation_module, "probabilistic_sharpe_ratio", fake_psr)
    result = multiple_testing_diagnostics(robustness, significance_level=0.05)
    raw = [item.raw_pvalue for item in result.scenario_pvalues]
    bonferroni = [item.bonferroni_adjusted_pvalue for item in result.scenario_pvalues]
    bh = [item.benjamini_hochberg_adjusted_pvalue for item in result.scenario_pvalues]
    assert result.total_tested_configurations == 3
    assert result.valid_pvalue_count == 3
    assert result.unavailable_pvalue_count == 0
    assert raw == pytest.approx([0.01, 0.02, 0.10])
    assert bonferroni == pytest.approx([0.03, 0.06, 0.30])
    assert bh == pytest.approx([0.03, 0.03, 0.10])
    ordered = sorted(zip(raw, bh))
    assert [value for _, value in ordered] == sorted(value for _, value in ordered)
    assert result.raw_threshold_exceedance_count == 2
    assert result.bonferroni_threshold_exceedance_count == 1
    assert result.benjamini_hochberg_discovery_count == 2
    assert result.baseline_scenario_id == robustness.baseline_scenario_id == "baseline"
    assert "dependent" in result.warning
    assert isinstance(result, MultipleTestingDiagnostics)


def test_invalid_unavailable_and_failed_scenarios_remain_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(
        monkeypatch,
        {1.0: _returns().tolist(), 1.5: _returns().tolist(), 2.0: _returns().tolist()},
    )
    scenarios = list(robustness.scenarios)
    scenarios[1] = replace(scenarios[1], status=ScenarioStatus.INVALID_CONFIGURATION)
    scenarios[2] = replace(scenarios[2], status=ScenarioStatus.FAILED)
    modified = replace(robustness, scenarios=tuple(scenarios))
    result = multiple_testing_diagnostics(modified)
    assert result.total_tested_configurations == 3
    assert result.invalid_configurations == 1
    assert result.failed_configurations == 1
    assert result.valid_pvalue_count == 1
    assert result.unavailable_pvalue_count == 2
    assert len(result.scenario_pvalues) == 3
    assert sum(np.isnan(item.raw_pvalue) for item in result.scenario_pvalues) == 2


def test_undefined_scenario_psr_remains_in_valid_test_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(
        monkeypatch,
        {
            1.0: _returns().tolist(),
            1.5: [0.01] * 8,
            2.0: [0.02, -0.01, 0.01, -0.004, 0.02, 0.0, -0.003, 0.01],
        },
    )
    result = multiple_testing_diagnostics(robustness)
    undefined = next(
        item
        for item in result.scenario_pvalues
        if item.psr_status is not ValidationAvailability.AVAILABLE
    )
    assert result.total_tested_configurations == 3
    assert result.valid_pvalue_count == 2
    assert result.bonferroni_valid_test_count == 2
    assert result.unavailable_pvalue_count == 1
    assert result.analytically_unavailable_configurations == 1
    assert np.isnan(undefined.raw_pvalue)
    assert len(result.scenario_pvalues) == 3


def test_report_is_immutable_defensive_and_composes_standalone_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds = (_fold_result(1, WalkForwardStatus.COMPLETED, [0.01, -0.005, 0.01, 0.0]),)
    robustness = _robustness_result(monkeypatch, folds=folds)
    labels = pd.Series(["a", "a", "b", "b", "a", "a", "b", "b"], index=robustness.baseline_result.common_horizon_returns.index)
    report = build_statistical_validation_report(
        robustness, block_length=2, n_bootstrap=40, random_seed=5, regime_labels=labels
    )
    standalone_bootstrap = bootstrap_performance_metrics(
        robustness.baseline_result.common_horizon_returns, 2, 40, random_seed=5
    )
    standalone_psr = probabilistic_sharpe_ratio(robustness.baseline_result.common_horizon_returns)
    assert report.bootstrap.total_return == standalone_bootstrap.total_return
    assert report.probabilistic_sharpe == standalone_psr
    assert report.multiple_testing.baseline_scenario_id == "baseline"
    assert report.purpose == "statistical_diagnostics_not_proof_of_future_profitability"
    assert any("finite historical sample" in warning for warning in report.validation_warnings)
    assert any("Statistical significance" in warning for warning in report.validation_warnings)
    assert "fixture provenance warning" in report.provenance_warnings
    assert validation_module.REGIME_PROVENANCE_WARNING in report.provenance_warnings
    assert isinstance(report, StatisticalValidationResult)
    with pytest.raises(FrozenInstanceError):
        report.observations = 0  # type: ignore[misc]
    before = report.primary_oos_returns.copy(deep=True)
    robustness.baseline_result.common_horizon_returns.iloc[:] = 99.0
    pd.testing.assert_series_equal(report.primary_oos_returns, before)
    report.bootstrap.bootstrap_samples.iloc[:] = 77.0
    assert not standalone_bootstrap.bootstrap_samples.eq(77.0).all().all()


def test_missing_common_calendar_prevents_primary_inferential_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(monkeypatch, {1.0: [0.01, np.nan, 0.02, 0.0]})
    report = build_statistical_validation_report(
        robustness, block_length=2, n_bootstrap=20, random_seed=3
    )
    assert report.primary_oos_returns.isna().sum() == 1
    assert report.availability is ValidationAvailability.UNAVAILABLE
    assert report.bootstrap.status is ValidationAvailability.UNAVAILABLE
    assert report.probabilistic_sharpe.status is ValidationAvailability.UNAVAILABLE
    assert report.minimum_track_record.status is ValidationAvailability.UNAVAILABLE


def test_fully_observed_catastrophic_return_remains_in_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(monkeypatch, {1.0: [-1.0, 0.02, 0.01, 0.0]})
    report = build_statistical_validation_report(
        robustness, block_length=1, n_bootstrap=30, random_seed=4
    )
    assert report.primary_oos_returns.iloc[0] == -1.0
    assert -1.0 in report.bootstrap.original_returns.to_numpy()
    assert report.bootstrap.total_return.status is ValidationAvailability.UNAVAILABLE


def test_repeated_complete_reports_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(monkeypatch)
    first = build_statistical_validation_report(
        robustness, block_length=2, n_bootstrap=30, random_seed=8
    )
    second = build_statistical_validation_report(
        robustness, block_length=2, n_bootstrap=30, random_seed=8
    )
    pd.testing.assert_frame_equal(first.bootstrap.bootstrap_samples, second.bootstrap.bootstrap_samples)
    pd.testing.assert_frame_equal(first.bootstrap.replicate_metrics, second.bootstrap.replicate_metrics)
    assert first.probabilistic_sharpe == second.probabilistic_sharpe
    assert first.minimum_track_record == second.minimum_track_record


def test_inputs_remain_unchanged() -> None:
    source = _returns()
    labels = pd.Series(["a", "b"] * 4, index=source.index)
    source_before = source.copy(deep=True)
    labels_before = labels.copy(deep=True)
    bootstrap_performance_metrics(source, 2, 20, random_seed=1)
    probabilistic_sharpe_ratio(source)
    minimum_track_record_length(source)
    analyze_regime_consistency(source, labels)
    pd.testing.assert_series_equal(source, source_before)
    pd.testing.assert_series_equal(labels, labels_before)


def test_no_automatic_strategy_promotion_api_exists() -> None:
    forbidden = (
        "best_parameters", "winner", "optimal", "approved", "validated",
        "production_ready", "invest", "deploy", "select_best_scenario",
    )
    for name in forbidden:
        assert not hasattr(validation_module, name)
    assert all(
        token not in StatisticalValidationResult.__dataclass_fields__
        for token in ("approved", "validated", "production_ready", "optimal")
    )
    assert isinstance(BootstrapMetricResult.__dataclass_fields__, dict)
    assert isinstance(FoldConsistencyMetrics.__dataclass_fields__, dict)
