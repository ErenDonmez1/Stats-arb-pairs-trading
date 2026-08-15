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


def test_full_length_block_is_not_primary_interval_evidence() -> None:
    source = _returns()
    result = bootstrap_performance_metrics(source, len(source), 10, random_seed=5)
    metric = result.total_return
    assert metric.interval is None
    assert metric.status is ValidationAvailability.UNAVAILABLE
    assert result.possible_block_starts == 1
    assert result.effective_unique_replicates == 1
    assert result.inference_design_status is ValidationAvailability.UNAVAILABLE
    assert np.isfinite(metric.point_estimate)


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


@pytest.mark.parametrize(
    "bad_value",
    ["0.01", True, pd.Timestamp("2024-01-01"), 0.01 + 0.02j],
)
def test_statistical_return_inputs_reject_coercible_non_real_values(
    bad_value: Any,
) -> None:
    source = pd.Series(
        [0.01, -0.01, bad_value, 0.02],
        index=pd.RangeIndex(4),
        dtype=object,
    )
    for function in (
        lambda: moving_block_bootstrap(source, 2, 10),
        lambda: bootstrap_performance_metrics(source, 2, 10),
        lambda: probabilistic_sharpe_ratio(source),
        lambda: minimum_track_record_length(source),
    ):
        with pytest.raises(ValueError, match="real numeric"):
            function()


@pytest.mark.parametrize(
    "values",
    [
        [0.01, -0.01, 0.02, 0.0],
        [np.float64(0.01), np.float32(-0.01), np.float64(0.02), np.float32(0.0)],
        [1, -1, 2, 0],
        [np.int64(1), np.int32(-1), np.int64(2), np.int32(0)],
    ],
)
def test_python_and_numpy_real_return_scalars_are_accepted(
    values: list[Any],
) -> None:
    source = pd.Series(values, index=pd.RangeIndex(4), dtype=object)
    result = moving_block_bootstrap(source, 2, 5, random_seed=3)
    assert result.shape == (5, 4)


def test_bootstrap_requires_monotonic_chronology_without_sorting() -> None:
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-05", "2024-01-09", "2024-01-15"]
    )
    source = pd.Series([0.01, -0.01, 0.02, 0.0], index=dates)
    original = source.copy(deep=True)
    accepted = moving_block_bootstrap(source, 2, 10, random_seed=2)
    assert accepted.shape == (10, 4)
    pd.testing.assert_series_equal(source, original)

    with pytest.raises(ValueError, match="monotonically increasing"):
        moving_block_bootstrap(source.iloc[::-1], 2, 10)
    shuffled = source.iloc[[0, 2, 1, 3]]
    with pytest.raises(ValueError, match="monotonically increasing"):
        bootstrap_performance_metrics(shuffled, 2, 10)
    assert list(shuffled.index) == list(source.index[[0, 2, 1, 3]])


def test_degenerate_bootstrap_designs_do_not_create_primary_intervals() -> None:
    source = _returns()
    single_replicate = bootstrap_performance_metrics(source, 2, 1, random_seed=4)
    assert single_replicate.total_return.interval is None
    assert single_replicate.inference_design_status is ValidationAvailability.UNAVAILABLE

    singleton = bootstrap_performance_metrics(
        pd.Series([0.01], index=pd.RangeIndex(1)),
        1,
        20,
        random_seed=4,
    )
    assert singleton.status is ValidationAvailability.INSUFFICIENT_DATA
    assert singleton.total_return.interval is None
    assert np.isfinite(singleton.total_return.point_estimate)


def test_n_minus_one_blocks_can_retain_genuine_interval_diversity() -> None:
    source = _returns([0.01, -0.02, 0.03, -0.01, 0.02])
    result = bootstrap_performance_metrics(
        source,
        len(source) - 1,
        100,
        random_seed=19,
    )
    assert result.possible_block_starts == 2
    assert result.effective_unique_replicates >= 2
    assert result.inference_design_status is ValidationAvailability.AVAILABLE
    assert result.total_return.status is ValidationAvailability.AVAILABLE
    assert result.total_return.interval is not None


def test_undefined_replicates_create_only_conditional_secondary_interval() -> None:
    source = _returns([0.0, 0.0, 0.01, 0.01])
    result = bootstrap_performance_metrics(
        source,
        1,
        200,
        random_seed=12,
        minimum_valid_fraction=0.01,
    )
    metric = result.sharpe_ratio
    assert metric.valid_replicates + metric.undefined_replicates == 200
    assert metric.undefined_replicates > 0
    assert metric.interval is None
    assert metric.status is ValidationAvailability.UNAVAILABLE
    assert metric.conditional_status is ValidationAvailability.AVAILABLE
    assert metric.conditional_interval is not None
    assert metric.conditional_bootstrap_median is not None
    assert not metric.primary_interval_is_unconditional

    stricter = bootstrap_performance_metrics(
        source,
        1,
        200,
        random_seed=12,
        minimum_valid_fraction=0.95,
    )
    assert stricter.sharpe_ratio.interval is None
    assert stricter.sharpe_ratio.status is ValidationAvailability.UNAVAILABLE


def test_unexpected_bootstrap_metric_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_unexpectedly(*args: Any, **kwargs: Any) -> float:
        raise ValueError("injected programming defect")

    monkeypatch.setattr(validation_module, "total_return", fail_unexpectedly)
    with pytest.raises(ValueError, match="injected programming defect"):
        bootstrap_performance_metrics(_returns(), 2, 10)


def test_expected_catastrophic_bootstrap_metric_is_recorded_not_raised() -> None:
    result = bootstrap_performance_metrics(
        _returns([-1.0, 0.01, 0.02, 0.0]),
        1,
        50,
        random_seed=6,
        minimum_valid_fraction=0.01,
    )
    assert result.total_return.undefined_replicates > 0
    assert result.original_returns.iloc[0] == -1.0


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


@pytest.mark.parametrize("catastrophic", [-1.0, -1.0001, -2.0])
def test_psr_retains_but_rejects_catastrophic_returns(
    catastrophic: float,
) -> None:
    source = _returns([0.01, catastrophic, 0.02, -0.01, 0.0])
    original = source.copy(deep=True)
    result = probabilistic_sharpe_ratio(source)
    assert result.status is ValidationAvailability.UNAVAILABLE
    assert result.observations == len(source)
    assert np.isnan(result.probability)
    assert np.isnan(result.statistic)
    assert np.isnan(result.one_sided_pvalue)
    assert "at or below -100%" in str(result.error)
    pd.testing.assert_series_equal(source, original)


def test_psr_exposes_numerically_stable_survival_tail() -> None:
    rng = np.random.default_rng(314)
    source = pd.Series(rng.normal(0.01, 0.005, 100), name="oos_return")
    result = probabilistic_sharpe_ratio(source)
    assert result.status is ValidationAvailability.AVAILABLE
    assert result.statistic > 8.0
    assert result.one_sided_pvalue == pytest.approx(
        norm.sf(result.statistic),
        rel=1e-12,
    )
    assert result.one_sided_pvalue > 0.0
    assert 1.0 - result.probability == 0.0


@pytest.mark.parametrize("base", [0.01, 0.50])
def test_psr_near_zero_volatility_matches_core_analytics(base: float) -> None:
    source = _returns(
        [base, base + 2e-13, base - 2e-13, base + 1e-13, base - 1e-13]
    )
    assert np.isnan(sharpe_ratio(source, 252))
    result = probabilistic_sharpe_ratio(source)
    assert result.status is ValidationAvailability.UNAVAILABLE
    assert np.isnan(result.probability)


def test_psr_supports_genuinely_nonzero_volatility() -> None:
    source = _returns([0.01, 0.012, 0.008, 0.011, 0.009, 0.013])
    assert np.isfinite(sharpe_ratio(source, 252))
    assert probabilistic_sharpe_ratio(source).status is ValidationAvailability.AVAILABLE


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


def test_minimum_track_record_near_benchmark_is_unbounded_not_exception() -> None:
    source = _returns()
    observed = probabilistic_sharpe_ratio(source).observed_sharpe_annualized
    benchmark = float(np.nextafter(observed, -np.inf))
    result = minimum_track_record_length(source, benchmark_sharpe=benchmark)
    assert result.estimated_required_observations is None
    assert result.sufficient_track_record is None
    assert result.status is ValidationAvailability.UNAVAILABLE


def test_minimum_track_record_grows_as_sharpe_advantage_shrinks() -> None:
    source = _returns()
    observed = probabilistic_sharpe_ratio(source).observed_sharpe_annualized
    farther = minimum_track_record_length(source, benchmark_sharpe=observed - 1.0)
    nearer = minimum_track_record_length(source, benchmark_sharpe=observed - 0.1)
    assert farther.estimated_required_observations is not None
    assert nearer.estimated_required_observations is not None
    assert nearer.estimated_required_observations > farther.estimated_required_observations


def test_extreme_finite_psr_moments_cannot_overflow_track_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = probabilistic_sharpe_ratio(_returns())
    extreme = replace(
        base,
        observed_sharpe_per_period=0.1,
        observed_sharpe_annualized=0.1 * np.sqrt(252),
        benchmark_sharpe_per_period=0.0,
        benchmark_sharpe_annualized=0.0,
        variance_term=np.finfo(float).max / 4.0,
    )
    monkeypatch.setattr(
        validation_module,
        "probabilistic_sharpe_ratio",
        lambda *args, **kwargs: extreme,
    )
    result = minimum_track_record_length(_returns())
    assert result.status is ValidationAvailability.UNAVAILABLE
    assert result.estimated_required_observations is None
    assert "unbounded" in str(result.error)


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


def test_singleton_fold_retains_return_but_has_insufficient_risk_analytics() -> None:
    folds = (_fold_result(1, WalkForwardStatus.COMPLETED, [0.05]),)
    result = analyze_fold_consistency(
        _walk_forward_result(_returns().tolist(), folds=folds)
    )
    fold = result.folds[0]
    assert fold.observations == 1
    assert fold.total_return == pytest.approx(0.05)
    assert np.isnan(fold.annualized_volatility)
    assert np.isnan(fold.sharpe_ratio)
    assert fold.availability is ValidationAvailability.INSUFFICIENT_DATA
    assert result.status is ValidationAvailability.INSUFFICIENT_DATA
    assert result.analytically_available_fold_count == 0
    assert result.insufficient_data_fold_count == 1


def test_catastrophic_fold_remains_explicit_and_cannot_improve_summaries() -> None:
    folds = (
        _fold_result(1, WalkForwardStatus.COMPLETED, [-1.0, 0.0, 0.0, 0.0]),
        _fold_result(2, WalkForwardStatus.COMPLETED, [0.10, 0.0, 0.0, 0.0]),
    )
    result = analyze_fold_consistency(
        _walk_forward_result(_returns().tolist(), folds=folds)
    )
    catastrophic = result.folds[0]
    assert catastrophic.catastrophic
    assert not catastrophic.invalid_return
    assert catastrophic.availability is ValidationAvailability.UNAVAILABLE
    assert result.catastrophic_fold_count == 1
    assert result.catastrophic_fold_ids == (1,)
    assert result.negative_return_fold_count == 1
    assert result.fraction_observable_folds_positive == pytest.approx(0.5)
    assert not result.summaries_complete
    assert np.isnan(result.worst_fold_return)
    assert np.isnan(result.strongest_fold_return)
    assert np.isnan(result.strongest_single_positive_fold_concentration)


def test_fold_return_below_minus_one_is_explicitly_invalid() -> None:
    folds = (_fold_result(1, WalkForwardStatus.COMPLETED, [-1.01, 0.0]),)
    result = analyze_fold_consistency(
        _walk_forward_result(_returns().tolist(), folds=folds)
    )
    assert result.invalid_return_fold_count == 1
    assert result.folds[0].invalid_return
    assert "below -100%" in str(result.folds[0].error)
    assert not result.summaries_complete


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


def test_singleton_regime_retains_total_return_but_is_insufficient() -> None:
    source = _returns([0.01, -0.02, 0.03])
    labels = pd.Series(["singleton", "other", "other"], index=source.index)
    result = analyze_regime_consistency(source, labels)
    singleton = result.regimes[0]
    assert singleton.observations == 1
    assert singleton.total_return == pytest.approx(0.01)
    assert np.isnan(singleton.annualized_volatility)
    assert np.isnan(singleton.sharpe_ratio)
    assert singleton.availability is ValidationAvailability.INSUFFICIENT_DATA
    assert result.status is ValidationAvailability.INSUFFICIENT_DATA


def test_adequately_populated_regimes_are_available() -> None:
    source = _returns([0.01, -0.02, 0.03, -0.01, 0.02, -0.005])
    labels = pd.Series(["a", "a", "a", "b", "b", "b"], index=source.index)
    result = analyze_regime_consistency(source, labels)
    assert result.status is ValidationAvailability.AVAILABLE
    assert all(
        regime.availability is ValidationAvailability.AVAILABLE
        for regime in result.regimes
    )
    assert result.contribution_basis == "arithmetic_return_sum_diagnostic"
    assert "not compounded" in result.contribution_warning


def test_empty_aligned_regime_inputs_return_structured_insufficient_data() -> None:
    source = pd.Series(dtype=float, name="oos_return")
    labels = pd.Series(dtype=object, index=source.index, name="regime")
    result = analyze_regime_consistency(source, labels)
    assert result.status is ValidationAvailability.INSUFFICIENT_DATA
    assert result.observations == 0
    assert result.regime_count == 0
    assert result.regimes == ()


@pytest.mark.parametrize("label", [["nested"], {"nested": 1}, {"nested"}])
def test_mutable_nested_regime_labels_are_rejected(label: Any) -> None:
    source = _returns([0.01, -0.01])
    labels = pd.Series([label, "stable"], index=source.index, dtype=object)
    with pytest.raises(ValueError, match="immutable scalar"):
        analyze_regime_consistency(source, labels)


def test_string_and_numeric_regime_labels_are_supported() -> None:
    source = _returns([0.01, -0.01, 0.02, -0.02])
    labels = pd.Series(["text", "text", np.int64(2), np.int64(2)], index=source.index)
    result = analyze_regime_consistency(source, labels)
    assert [regime.regime_label for regime in result.regimes] == ["text", 2]


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
        statistic = float(norm.ppf(probability))
        return replace(
            probabilistic_sharpe_ratio(_returns()),
            probability=probability,
            statistic=statistic,
            one_sided_pvalue=float(norm.sf(statistic)),
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
    assert result.bonferroni_valid_test_count == 3
    assert result.eligible_hypothesis_count == 3
    assert result.finite_pvalue_count == 2
    assert result.unavailable_eligible_hypothesis_count == 1
    assert result.family_size == 3
    assert result.unavailable_pvalue_count == 1
    assert result.analytically_unavailable_configurations == 1
    assert np.isnan(undefined.raw_pvalue)
    assert len(result.scenario_pvalues) == 3
    finite = [
        item for item in result.scenario_pvalues if np.isfinite(item.raw_pvalue)
    ]
    assert all(
        item.bonferroni_adjusted_pvalue
        == pytest.approx(min(item.raw_pvalue * 3, 1.0))
        for item in finite
    )
    assert (
        result.eligible_hypothesis_count
        == result.finite_pvalue_count
        + result.unavailable_eligible_hypothesis_count
    )
    assert (
        result.total_tested_configurations
        == result.eligible_hypothesis_count
        + result.invalid_configurations
        + result.failed_configurations
    )


def test_undefined_eligible_hypothesis_cannot_improve_adjustments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smaller = _robustness_result(
        monkeypatch,
        {
            1.0: _returns().tolist(),
            2.0: [0.02, -0.01, 0.01, -0.004, 0.02, 0.0, -0.003, 0.01],
        },
    )
    small_result = multiple_testing_diagnostics(smaller)
    larger = _robustness_result(
        monkeypatch,
        {
            1.0: _returns().tolist(),
            1.5: [0.01] * 8,
            2.0: [0.02, -0.01, 0.01, -0.004, 0.02, 0.0, -0.003, 0.01],
        },
    )
    large_result = multiple_testing_diagnostics(larger)
    small_by_id = {item.scenario_id: item for item in small_result.scenario_pvalues}
    large_by_id = {item.scenario_id: item for item in large_result.scenario_pvalues}
    for scenario_id in set(small_by_id).intersection(large_by_id):
        assert (
            large_by_id[scenario_id].bonferroni_adjusted_pvalue
            >= small_by_id[scenario_id].bonferroni_adjusted_pvalue
        )
        assert (
            large_by_id[scenario_id].benjamini_hochberg_adjusted_pvalue
            >= small_by_id[scenario_id].benjamini_hochberg_adjusted_pvalue
        )
    unavailable = next(
        item for item in large_result.scenario_pvalues
        if item.psr_status is not ValidationAvailability.AVAILABLE
    )
    assert np.isnan(unavailable.raw_pvalue)
    assert np.isnan(unavailable.bonferroni_adjusted_pvalue)
    assert np.isnan(unavailable.benjamini_hochberg_adjusted_pvalue)


def test_catastrophic_scenarios_never_receive_pvalues_or_discoveries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(
        monkeypatch,
        {
            1.0: [-1.0, 0.02, 0.01, 0.0, 0.01, -0.01, 0.02, 0.0],
            1.5: _returns().tolist(),
        },
    )
    result = multiple_testing_diagnostics(robustness, significance_level=0.05)
    baseline = next(
        item for item in result.scenario_pvalues if item.scenario_id == "baseline"
    )
    assert result.baseline_scenario_id == "baseline"
    assert baseline.psr_status is ValidationAvailability.UNAVAILABLE
    assert np.isnan(baseline.raw_pvalue)
    assert np.isnan(baseline.bonferroni_adjusted_pvalue)
    assert np.isnan(baseline.benjamini_hochberg_adjusted_pvalue)
    assert result.eligible_hypothesis_count == 2
    assert result.unavailable_eligible_hypothesis_count >= 1
    assert result.raw_threshold_exceedance_count <= result.finite_pvalue_count
    assert result.bonferroni_threshold_exceedance_count <= result.finite_pvalue_count
    assert result.benjamini_hochberg_discovery_count <= result.finite_pvalue_count


def test_duplicate_scenario_ids_are_rejected_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(
        monkeypatch,
        {1.0: _returns().tolist(), 1.5: _returns().tolist()},
    )
    duplicate = replace(
        robustness.scenarios[1],
        scenario=replace(robustness.scenarios[1].scenario, scenario_id="baseline"),
    )
    malformed = replace(
        robustness,
        scenarios=(robustness.scenarios[0], duplicate),
    )
    with pytest.raises(ValueError, match="Scenario IDs must be unique"):
        multiple_testing_diagnostics(malformed)


def test_multiple_testing_uses_stable_survival_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(monkeypatch)
    statistic = 9.0
    precise_tail = float(norm.sf(statistic))

    def fake_psr(*args: Any, **kwargs: Any) -> Any:
        return replace(
            probabilistic_sharpe_ratio(_returns()),
            probability=float(norm.cdf(statistic)),
            statistic=statistic,
            one_sided_pvalue=precise_tail,
        )

    monkeypatch.setattr(validation_module, "probabilistic_sharpe_ratio", fake_psr)
    result = multiple_testing_diagnostics(robustness)
    row = result.scenario_pvalues[0]
    assert row.raw_pvalue == precise_tail
    assert row.raw_pvalue > 0.0
    assert row.bonferroni_adjusted_pvalue == precise_tail
    assert row.benjamini_hochberg_adjusted_pvalue == precise_tail


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
    assert report.probabilistic_sharpe.status is ValidationAvailability.UNAVAILABLE
    baseline = next(
        item
        for item in report.multiple_testing.scenario_pvalues
        if item.scenario_id == "baseline"
    )
    assert np.isnan(baseline.raw_pvalue)
    assert np.isnan(baseline.bonferroni_adjusted_pvalue)
    assert np.isnan(baseline.benjamini_hochberg_adjusted_pvalue)


def test_empty_regime_evidence_composes_structured_unavailable_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(monkeypatch)
    empty = pd.Series(dtype=float, name="common_horizon_return")
    scenario = replace(
        robustness.scenarios[0],
        common_horizon_returns=empty,
        common_horizon_structurally_available=False,
        common_horizon_fully_observed=False,
        common_horizon_analytics_status=(
            robustness_module.MetricAvailabilityStatus.NOT_APPLICABLE
        ),
    )
    modified = replace(
        robustness,
        scenarios=(scenario,),
        common_horizon_index=empty.index,
        common_horizon_observations=0,
        common_horizon_scenario_count=0,
        common_horizon_available=False,
        common_horizon_structurally_available=False,
        common_horizon_fully_observed=False,
        common_horizon_analytics_available=False,
        common_horizon_analytics_status=(
            robustness_module.MetricAvailabilityStatus.NOT_APPLICABLE
        ),
    )
    labels = pd.Series(dtype=object, index=empty.index, name="regime")
    report = build_statistical_validation_report(
        modified,
        block_length=1,
        n_bootstrap=10,
        regime_labels=labels,
    )
    assert report.availability is ValidationAvailability.UNAVAILABLE
    assert report.regime_consistency is not None
    assert (
        report.regime_consistency.status
        is ValidationAvailability.INSUFFICIENT_DATA
    )
    assert report.regime_consistency.observations == 0


def test_bootstrap_interpretation_exposes_inference_assumptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robustness = _robustness_result(monkeypatch)
    report = build_statistical_validation_report(
        robustness,
        block_length=2,
        n_bootstrap=20,
    )
    interpretation = report.bootstrap.interpretation.lower()
    warnings = " ".join(report.validation_warnings).lower()
    assert "stationarity/mixing" in interpretation
    assert "caller-declared block length" in interpretation
    assert "row adjacency" in interpretation
    assert "not guarantees" in interpretation
    assert "stationarity/mixing" in warnings
    assert "future performance" in warnings


def test_nested_upstream_mutation_cannot_change_completed_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds = (
        _fold_result(1, WalkForwardStatus.COMPLETED, [0.01, -0.005, 0.01, 0.0]),
    )
    robustness = _robustness_result(
        monkeypatch,
        {1.0: _returns().tolist(), 1.5: _returns().tolist()},
        folds=folds,
    )
    report = build_statistical_validation_report(
        robustness,
        block_length=2,
        n_bootstrap=20,
        random_seed=9,
    )
    fold_snapshot = report.fold_consistency.folds
    pvalue_snapshot = report.multiple_testing.scenario_pvalues
    primary_snapshot = report.primary_oos_returns.copy(deep=True)

    baseline_walk_forward = robustness.baseline_result.walk_forward_result
    assert baseline_walk_forward is not None
    baseline_walk_forward.folds[0].oos_returns.iloc[:] = 88.0
    robustness.baseline_result.common_horizon_returns.iloc[:] = 77.0
    robustness.scenarios[1].common_horizon_returns.iloc[:] = 66.0

    assert report.fold_consistency.folds == fold_snapshot
    assert report.multiple_testing.scenario_pvalues == pvalue_snapshot
    pd.testing.assert_series_equal(report.primary_oos_returns, primary_snapshot)


def test_real_walkforward_robustness_validation_path_is_deterministic() -> None:
    scenarios = generate_parameter_scenarios(
        _scenario(),
        entry_z_values=[1.0, 1.5],
    )
    first_robustness = run_sensitivity_analysis(
        _prices(84),
        scenarios,
        "baseline",
        groups={"economic_pair": ("AAA", "BBB")},
    )
    second_robustness = run_sensitivity_analysis(
        _prices(84),
        scenarios,
        "baseline",
        groups={"economic_pair": ("AAA", "BBB")},
    )
    first = build_statistical_validation_report(
        first_robustness,
        block_length=2,
        n_bootstrap=20,
        random_seed=23,
    )
    second = build_statistical_validation_report(
        second_robustness,
        block_length=2,
        n_bootstrap=20,
        random_seed=23,
    )
    assert len(first_robustness.scenarios) == 2
    assert first.multiple_testing.total_tested_configurations == 2
    assert first.multiple_testing.family_size == 2
    assert first.multiple_testing.baseline_scenario_id == "baseline"
    assert (
        first.multiple_testing.eligible_hypothesis_count
        == first.multiple_testing.finite_pvalue_count
        + first.multiple_testing.unavailable_eligible_hypothesis_count
    )
    pd.testing.assert_series_equal(
        first_robustness.baseline_result.common_horizon_returns,
        second_robustness.baseline_result.common_horizon_returns,
    )
    pd.testing.assert_frame_equal(
        first.bootstrap.bootstrap_samples,
        second.bootstrap.bootstrap_samples,
    )
    for original, repeated in zip(
        first.multiple_testing.scenario_pvalues,
        second.multiple_testing.scenario_pvalues,
    ):
        assert original.scenario_id == repeated.scenario_id
        assert original.scenario_status == repeated.scenario_status
        assert original.psr_status is repeated.psr_status
        assert original.error == repeated.error
        np.testing.assert_allclose(
            [
                original.raw_pvalue,
                original.bonferroni_adjusted_pvalue,
                original.benjamini_hochberg_adjusted_pvalue,
            ],
            [
                repeated.raw_pvalue,
                repeated.bonferroni_adjusted_pvalue,
                repeated.benjamini_hochberg_adjusted_pvalue,
            ],
            equal_nan=True,
        )


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
