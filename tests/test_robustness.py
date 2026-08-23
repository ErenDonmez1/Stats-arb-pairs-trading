"""Focused tests for diagnostic parameter robustness and sensitivity analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.analytics import calculate_core_metrics, calculate_drawdown_metrics
from pairs_trading.data import make_synthetic_universe
from pairs_trading.robustness import (
    LocalNeighbor,
    MetricAvailabilityStatus,
    MetricDistribution,
    MetricFractionSummary,
    ParameterAxisMetadata,
    ParameterScenario,
    RobustnessResult,
    ScenarioResult,
    ScenarioStatus,
    SensitivitySummary,
    generate_parameter_scenarios,
    run_parameter_scenario,
    run_sensitivity_analysis,
    sensitivity_table,
    summarize_sensitivity,
)
from pairs_trading.screening import PairScreeningResult
from pairs_trading.walkforward import (
    WalkForwardFold,
    WalkForwardAnalyticsStatus,
    WalkForwardResult,
    WalkForwardReturnReport,
)
import pairs_trading.robustness as robustness_module
import pairs_trading.walkforward as walkforward_module


def _baseline(**overrides: Any) -> ParameterScenario:
    values: dict[str, Any] = {
        "scenario_id": "baseline",
        "entry_z": 1.0,
        "exit_z": 0.25,
        "stop_z": 3.0,
        "zscore_lookback": 5,
        "formation_window": 60,
        "trading_window": 8,
        "screening_min_observations": 50,
        "commission_bps": 2.0,
        "slippage_bps": 1.0,
        "financing_rate": 0.02,
        "borrow_rate_y": 0.03,
        "borrow_rate_x": 0.03,
    }
    values.update(overrides)
    return ParameterScenario(**values)


def _prices(rows: int = 84) -> pd.DataFrame:
    positions = np.arange(rows, dtype=float)
    price_x = 100.0 * np.exp(0.0005 * positions)
    spread = np.resize(np.array([-0.01, -0.005, 0.0, 0.005, 0.01]), rows)
    for start in range(50, rows, 8):
        spread[start : min(start + 3, rows)] = np.array([-0.08, -0.07, 0.0])[
            : min(3, rows - start)
        ]
    return pd.DataFrame(
        {
            "AAA": price_x * np.exp(spread),
            "BBB": price_x,
            "CCC": 70.0 * np.exp(0.0003 * positions + 0.002 * np.sin(positions)),
        },
        index=pd.RangeIndex(rows, name="row"),
    )


def _screening_result(observations: int = 60) -> PairScreeningResult:
    return PairScreeningResult(
        symbol_y="AAA",
        symbol_x="BBB",
        group=None,
        observations=observations,
        alpha=0.0,
        beta=1.0,
        spread_standard_deviation=0.01,
        cointegration_statistic=-4.0,
        cointegration_pvalue=0.005,
        corrected_pvalue=0.01,
        cointegration_critical_values={"1%": -3.9, "5%": -3.3, "10%": -3.0},
        adf_statistic=-4.2,
        adf_pvalue=0.004,
        half_life=4.0,
        hurst=0.3,
        selected=True,
        rank=1,
        rejection_reasons=(),
    )


def _return_report(returns: pd.Series) -> WalkForwardReturnReport | None:
    valid = returns.dropna()
    if len(valid) < 2 or valid.le(-1.0).any():
        return None
    core = calculate_core_metrics(returns, 252)
    drawdown = calculate_drawdown_metrics(returns, 252)
    return WalkForwardReturnReport(
        core=core,
        drawdown=drawdown,
        report_observations=core.observations,
        periods_per_year=252,
    )


def _walk_forward_result(
    calendar_values: list[float],
    *,
    index: pd.Index | None = None,
    conditional_values: list[float] | None = None,
    selected_observations: int | None = None,
    evaluated_start: int = 60,
) -> WalkForwardResult:
    if index is None:
        index = pd.RangeIndex(evaluated_start, evaluated_start + len(calendar_values))
    calendar = pd.Series(calendar_values, index=index, name="calendar_oos_return")
    if conditional_values is None:
        conditional = calendar.dropna().copy(deep=True)
    else:
        conditional = pd.Series(
            conditional_values,
            index=index[: len(conditional_values)],
            name="conditional_oos_return",
        )
    calendar_report = _return_report(calendar)
    conditional_report = _return_report(conditional)
    unavailable = int(calendar.isna().sum())
    available = len(calendar) - unavailable
    selected = available if selected_observations is None else selected_observations
    no_selection = max(available - selected, 0)
    eligible = selected + no_selection
    return WalkForwardResult(
        folds=(),
        fold_count=1,
        completed_fold_count=1 if selected else 0,
        no_selection_fold_count=1 if no_selection else 0,
        insufficient_data_fold_count=1 if unavailable else 0,
        scheduled_oos_observations=len(calendar),
        scheduled_eligible_oos_observations=eligible,
        selected_oos_observations=selected,
        no_selection_oos_observations=no_selection,
        unavailable_oos_observations=unavailable,
        selection_coverage=float(selected / eligible) if eligible else float("nan"),
        conditional_oos_returns=conditional,
        calendar_oos_returns=calendar,
        conditional_performance_report=conditional_report,
        calendar_performance_report=calendar_report if not unavailable else None,
        conditional_analytics_status=(
            WalkForwardAnalyticsStatus.AVAILABLE
            if conditional_report is not None
            else WalkForwardAnalyticsStatus.UNAVAILABLE
        ),
        calendar_analytics_status=(
            WalkForwardAnalyticsStatus.AVAILABLE
            if calendar_report is not None and not unavailable
            else WalkForwardAnalyticsStatus.UNAVAILABLE
        ),
        conditional_analytics_error=None,
        calendar_analytics_error=(
            None if calendar_report is not None and not unavailable else "unavailable"
        ),
        capital_policy="equal_capital_reset",
        aggregate_return_policy="time_weighted_equal_capital_reset",
        inactive_capital_policy="zero_return_cash_for_no_selection_rows",
        selection_coverage_denominator="selected_plus_no_selection_scheduled_rows",
        aggregate_dollar_pnl_available=False,
        aggregate_trade_dollar_metrics_available=False,
        universe_provenance="fixture",
        cleaning_provenance="fixture",
        point_in_time_universe_validated=False,
        provenance_warnings=("fixture",),
        evaluated_start_position=evaluated_start,
        evaluated_end_position=evaluated_start + len(calendar) - 1,
        evaluated_start_label=index[0],
        evaluated_end_label=index[-1],
        discarded_terminal_rows=0,
    )


def _install_fake_walk_forward(
    monkeypatch: pytest.MonkeyPatch,
    factory: Any | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(
        prices: pd.DataFrame,
        formation_window: int,
        trading_window: int,
        **kwargs: Any,
    ) -> WalkForwardResult:
        calls.append(
            {
                "prices": prices.copy(deep=True),
                "formation_window": formation_window,
                "trading_window": trading_window,
                **kwargs,
            }
        )
        if factory is not None:
            return factory(prices, formation_window, trading_window, kwargs)
        start = formation_window
        values = [0.01, -0.005, 0.002, 0.0]
        return _walk_forward_result(
            values,
            index=prices.index[start : start + len(values)],
            evaluated_start=start,
        )

    monkeypatch.setattr(robustness_module, "run_walk_forward_analysis", fake)
    return calls


@dataclass(frozen=True)
class _CostBacktest:
    accounting: pd.DataFrame


@dataclass(frozen=True)
class _CostFold:
    trade_count: int
    backtest: _CostBacktest | None


def test_cartesian_grid_and_scenario_ids_are_deterministic() -> None:
    baseline = _baseline()
    kwargs = {
        "entry_z_values": [1.5, 1.0],
        "commission_bps_values": [5.0, 2.0],
    }

    first = generate_parameter_scenarios(baseline, **kwargs)
    second = generate_parameter_scenarios(baseline, **kwargs)

    assert len(first) == 4
    assert first == second
    assert [scenario.scenario_id for scenario in first] == sorted(
        scenario.scenario_id for scenario in first
    )
    assert len({scenario.scenario_id for scenario in first}) == 4


def test_grid_order_does_not_depend_on_set_or_list_iteration() -> None:
    baseline = _baseline()

    from_lists = generate_parameter_scenarios(
        baseline,
        entry_z_values=[1.5, 1.0],
        commission_bps_values=[5.0, 2.0],
    )
    from_sets = generate_parameter_scenarios(
        baseline,
        entry_z_values={1.0, 1.5},
        commission_bps_values={2.0, 5.0},
    )

    assert from_lists == from_sets


@pytest.mark.parametrize("values_name", ["entry_z_values", "formation_window_values"])
def test_empty_parameter_lists_are_rejected(values_name: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        generate_parameter_scenarios(_baseline(), **{values_name: []})


def test_duplicate_parameter_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        generate_parameter_scenarios(_baseline(), entry_z_values=[1.0, 1.0])


def test_distinct_ids_cannot_duplicate_parameter_configuration_before_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_walk_forward(monkeypatch)
    duplicate = replace(_baseline(), scenario_id="same-parameters")

    with pytest.raises(ValueError, match="parameter tuples must be unique"):
        run_sensitivity_analysis(
            _prices(),
            (_baseline(), duplicate),
            "baseline",
        )

    assert calls == []


def test_summary_rejects_duplicate_parameter_configurations() -> None:
    metrics = robustness_module.PerformanceMetrics(
        0.1, 0.1, 0.2, 1.0, 1.0, -0.1, 1.0, 4
    )

    def record(scenario: ParameterScenario) -> ScenarioResult:
        return ScenarioResult(
            scenario=scenario,
            status=ScenarioStatus.COMPLETED,
            error=None,
            walk_forward_result=None,
            calendar_metrics=metrics,
            conditional_metrics=None,
            common_horizon_metrics=metrics,
            common_horizon_error=None,
            calendar_oos_returns=pd.Series([0.0, 0.0]),
            conditional_oos_returns=pd.Series(dtype=float),
            common_horizon_returns=pd.Series([0.0, 0.0]),
            evaluated_start_position=0,
            evaluated_end_position=1,
            scheduled_oos_observations=2,
            available_oos_observations=2,
            selected_oos_observations=2,
            unavailable_oos_observations=0,
            selection_coverage=1.0,
            trade_count=0,
            total_transaction_cost=0.0,
            periods_per_year=252,
            risk_free_rate=0.0,
        )

    with pytest.raises(ValueError, match="parameter tuples must be unique"):
        summarize_sensitivity(
            [record(_baseline()), record(replace(_baseline(), scenario_id="other"))],
            "baseline",
        )


def test_scenario_limit_is_enforced_before_any_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_walk_forward(monkeypatch)

    with pytest.raises(ValueError, match="exceeding max_scenarios"):
        generate_parameter_scenarios(
            _baseline(),
            entry_z_values=[1.0, 1.5],
            commission_bps_values=[2.0, 5.0],
            max_scenarios=3,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("formation_window_values", [60, 60.5]),
        ("trading_window_values", [8, True]),
        ("zscore_lookback_values", [5, "6"]),
    ],
)
def test_invalid_integer_grid_values_are_rejected(name: str, values: list[Any]) -> None:
    with pytest.raises(TypeError):
        generate_parameter_scenarios(_baseline(), **{name: values})


def test_baseline_is_explicitly_present_and_identified() -> None:
    baseline = _baseline(scenario_id="declared-baseline")
    scenarios = generate_parameter_scenarios(
        baseline,
        entry_z_values=[1.0, 1.5],
    )

    matching = [scenario for scenario in scenarios if scenario == baseline]
    assert matching == [baseline]
    assert matching[0].scenario_id == "declared-baseline"


def test_invalid_threshold_combinations_are_retained_and_not_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[0.2, 1.0],
    )
    invalid = next(scenario for scenario in scenarios if scenario.entry_z == 0.2)

    result = run_parameter_scenario(_prices(), invalid)

    assert result.status is ScenarioStatus.INVALID_CONFIGURATION
    assert "entry_z must be greater" in str(result.error)
    assert calls == []


def test_invalid_scenarios_remain_visible_in_sensitivity_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[0.2, 1.0],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    assert len(result.scenarios) == 2
    assert result.summary.invalid_scenarios == 1
    assert any(
        scenario.status is ScenarioStatus.INVALID_CONFIGURATION
        for scenario in result.scenarios
    )


def test_each_valid_scenario_calls_walk_forward_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5],
        commission_bps_values=[2.0, 5.0],
    )

    run_sensitivity_analysis(_prices(), scenarios, "baseline")

    assert len(calls) == 4
    assert len({id(call["prices"]) for call in calls}) == 4


def test_calendar_returns_are_primary_and_no_selection_cash_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(*args: Any) -> WalkForwardResult:
        return _walk_forward_result(
            [0.10, 0.0, 0.0, 0.0],
            conditional_values=[0.10],
            selected_observations=1,
        )

    _install_fake_walk_forward(monkeypatch, factory)

    result = run_parameter_scenario(_prices(), _baseline())

    assert result.status is ScenarioStatus.COMPLETED
    assert result.calendar_metrics is not None
    assert result.calendar_metrics.total_return == pytest.approx(0.10)
    assert len(result.calendar_oos_returns) == 4
    assert result.calendar_oos_returns.iloc[1:].eq(0.0).all()
    assert len(result.conditional_oos_returns) == 1
    assert result.selection_coverage == pytest.approx(0.25)


def test_native_and_common_horizon_drawdown_include_first_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(
        monkeypatch,
        lambda *args: _walk_forward_result([-0.20, 0.25]),
    )

    result = run_sensitivity_analysis(
        _prices(),
        (_baseline(),),
        "baseline",
    )
    baseline = result.baseline_result

    assert baseline.calendar_metrics is not None
    assert baseline.common_horizon_metrics is not None
    assert baseline.calendar_metrics.maximum_drawdown == pytest.approx(-0.20)
    assert baseline.common_horizon_metrics.maximum_drawdown == pytest.approx(-0.20)


def test_insufficient_data_rows_remain_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(
        monkeypatch,
        lambda *args: _walk_forward_result([0.01, np.nan, 0.0, np.nan]),
    )

    result = run_parameter_scenario(_prices(), _baseline())

    assert result.status is ScenarioStatus.ANALYTICS_UNAVAILABLE
    assert result.unavailable_oos_observations == 2
    assert result.available_oos_observations == 2
    assert result.calendar_oos_returns.isna().sum() == 2


def test_partial_calendar_never_produces_primary_or_headline_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        values = (
            [0.10, np.nan, np.nan, np.nan]
            if kwargs["entry_z"] == 1.0
            else [0.02, 0.0, 0.0, 0.0]
        )
        return _walk_forward_result(values)

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")
    partial = result.baseline_result

    assert partial.calendar_oos_returns.tolist()[0] == pytest.approx(0.10)
    assert partial.calendar_oos_returns.isna().sum() == 3
    assert partial.scheduled_oos_observations == 4
    assert partial.available_oos_observations == 1
    assert partial.unavailable_oos_observations == 3
    assert partial.calendar_metrics is None
    assert (
        partial.calendar_metrics_status
        is MetricAvailabilityStatus.UNAVAILABLE_PARTIAL_CALENDAR
    )
    assert result.summary.scenario_count == 2
    assert result.summary.unavailable_scenarios == 1
    assert not result.summary.headline_metrics_available
    assert result.summary.metric_distributions.total_return.observations == 0
    assert result.summary.total_return_fraction.eligible_scenarios == 2
    assert result.summary.total_return_fraction.defined_scenarios == 0
    assert result.summary.total_return_fraction.undefined_scenarios == 2
    assert np.isnan(result.summary.fraction_positive_total_return)


def test_one_row_calendar_retains_raw_return_but_analytics_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(
        monkeypatch,
        lambda *args: _walk_forward_result([0.03]),
    )

    result = run_parameter_scenario(_prices(), _baseline())

    assert result.calendar_oos_returns.tolist() == [0.03]
    assert result.calendar_metrics is None
    assert (
        result.calendar_metrics_status
        is MetricAvailabilityStatus.UNAVAILABLE_INSUFFICIENT_OBSERVATIONS
    )


def test_observed_only_partial_metrics_are_explicitly_secondary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(
        monkeypatch,
        lambda *args: _walk_forward_result([0.10, 0.02, np.nan, np.nan]),
    )

    result = run_parameter_scenario(_prices(), _baseline())

    assert result.calendar_metrics is None
    assert result.available_observations_metrics is not None
    assert result.available_observations_metrics.total_return == pytest.approx(0.122)


def test_all_unavailable_rows_produce_explicit_no_valid_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(
        monkeypatch,
        lambda *args: _walk_forward_result([np.nan, np.nan, np.nan]),
    )

    result = run_parameter_scenario(_prices(), _baseline())

    assert result.status is ScenarioStatus.NO_VALID_FOLDS
    assert result.calendar_metrics is None
    assert result.available_oos_observations == 0
    assert result.calendar_oos_returns.isna().all()


def test_scenario_native_horizon_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)

    result = run_parameter_scenario(_prices(), _baseline())

    assert result.evaluated_start_position == 60
    assert result.evaluated_end_position == 63
    assert result.scheduled_oos_observations == 4


def test_formation_window_changes_expose_different_native_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        formation_window_values=[50, 60],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    starts = {item.scenario.formation_window: item.evaluated_start_position for item in result.scenarios}
    assert starts == {50: 50, 60: 60}


def test_common_horizon_uses_structural_scheduled_index_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        if formation == 50:
            return _walk_forward_result(
                [0.01, 0.02, 0.03, 0.04],
                index=pd.Index([50, 51, 52, 53]),
                evaluated_start=50,
            )
        return _walk_forward_result(
            [0.05, np.nan, 0.06, 0.07],
            index=pd.Index([52, 53, 54, 55]),
            evaluated_start=60,
        )

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        formation_window_values=[50, 60],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    assert result.common_horizon_index.tolist() == [52, 53]
    assert result.common_horizon_structurally_available
    assert not result.common_horizon_fully_observed
    assert result.common_horizon_available is False
    partial = next(
        item for item in result.scenarios if item.scenario.formation_window == 60
    )
    complete = next(
        item for item in result.scenarios if item.scenario.formation_window == 50
    )
    assert partial.common_horizon_returns.index.tolist() == [52, 53]
    assert np.isnan(partial.common_horizon_returns.loc[53])
    assert partial.common_horizon_metrics is None
    assert complete.common_horizon_metrics is not None
    assert result.summary.metric_distributions.total_return.observations == 0


def test_common_horizon_metrics_use_exact_multirow_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        start = 50 if formation == 50 else 52
        return _walk_forward_result(
            [0.01, 0.02, 0.03, 0.04],
            index=pd.RangeIndex(start, start + 4),
            evaluated_start=formation,
        )

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        formation_window_values=[50, 60],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    assert result.common_horizon_index.tolist() == [52, 53]
    assert result.common_horizon_available
    for item in result.scenarios:
        assert item.common_horizon_returns.index.tolist() == [52, 53]
        assert item.common_horizon_metrics is not None


def test_headline_distributions_use_common_not_opposing_native_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        if formation == 50:
            return _walk_forward_result(
                [-0.50, 0.0, 0.10, 0.10],
                index=pd.RangeIndex(50, 54),
                evaluated_start=50,
            )
        return _walk_forward_result(
            [0.10, 0.10, 0.10, 0.10],
            index=pd.RangeIndex(52, 56),
            evaluated_start=60,
        )

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        formation_window_values=[50, 60],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    native_totals = {
        item.scenario.formation_window: item.calendar_metrics.total_return
        for item in result.scenarios
        if item.calendar_metrics is not None
    }
    assert native_totals[50] < 0.0 < native_totals[60]
    assert result.summary.headline_metric_basis == (
        "structural_common_scheduled_horizon"
    )
    assert result.summary.metric_distributions.total_return.median == pytest.approx(
        0.21
    )
    assert result.summary.native_metric_distributions.total_return.median != (
        result.summary.metric_distributions.total_return.median
    )


def test_loss_date_remains_in_structural_common_horizon_when_peer_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        values = [-0.40, 0.01, 0.01] if formation == 50 else [np.nan, 0.01, 0.01]
        return _walk_forward_result(
            values,
            index=pd.Index([52, 53, 54]),
            evaluated_start=formation,
        )

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        formation_window_values=[50, 60],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    assert result.common_horizon_index.tolist() == [52, 53, 54]
    losing = next(
        item for item in result.scenarios if item.scenario.formation_window == 50
    )
    missing = next(
        item for item in result.scenarios if item.scenario.formation_window == 60
    )
    assert losing.common_horizon_returns.loc[52] == pytest.approx(-0.40)
    assert np.isnan(missing.common_horizon_returns.loc[52])
    assert result.common_horizon_structurally_available
    assert not result.common_horizon_analytics_available


def test_one_row_common_horizon_is_structural_but_analytically_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        start = 50 if formation == 50 else 53
        return _walk_forward_result(
            [0.01, 0.02, 0.03, 0.04],
            index=pd.RangeIndex(start, start + 4),
            evaluated_start=formation,
        )

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        formation_window_values=[50, 60],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    assert result.common_horizon_index.tolist() == [53]
    assert result.common_horizon_structurally_available
    assert result.common_horizon_fully_observed
    assert not result.common_horizon_analytics_available
    assert (
        result.common_horizon_analytics_status
        is MetricAvailabilityStatus.UNAVAILABLE_INSUFFICIENT_OBSERVATIONS
    )
    assert result.summary.metric_distributions.total_return.observations == 0


def test_baseline_metrics_equal_direct_baseline_walk_forward_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    baseline = _baseline()
    scenarios = generate_parameter_scenarios(baseline, entry_z_values=[1.0, 1.5])

    direct = run_parameter_scenario(_prices(), baseline)
    combined = run_sensitivity_analysis(_prices(), scenarios, baseline.scenario_id)

    assert combined.baseline_result.calendar_metrics == direct.calendar_metrics
    assert combined.summary.baseline_metrics == direct.calendar_metrics


def test_performance_cannot_change_predefined_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        value = 0.50 if kwargs["entry_z"] == 1.5 else -0.05
        return _walk_forward_result([value, 0.0, 0.0, 0.0])

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    assert result.baseline_scenario_id == "baseline"
    assert result.baseline_result.scenario.entry_z == 1.0
    assert result.baseline_result.calendar_metrics is not None
    assert result.baseline_result.calendar_metrics.total_return == pytest.approx(-0.05)


def test_summary_medians_quartiles_and_positive_fraction_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = {1.0: -0.10, 1.5: 0.10, 2.0: 0.30}

    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        return _walk_forward_result([returns[kwargs["entry_z"]], 0.0, 0.0, 0.0])

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5, 2.0],
    )

    summary = run_sensitivity_analysis(_prices(), scenarios, "baseline").summary
    distribution = summary.metric_distributions.total_return

    assert distribution.median == pytest.approx(0.10)
    assert distribution.lower_quartile == pytest.approx(0.0)
    assert distribution.upper_quartile == pytest.approx(0.20)
    assert summary.fraction_positive_total_return == pytest.approx(2 / 3)


def test_positive_sharpe_fraction_excludes_undefined_values() -> None:
    defined = robustness_module.PerformanceMetrics(0.1, 0.1, 0.2, 1.0, 1.0, -0.1, 1.0, 4)
    undefined = replace(defined, sharpe_ratio=float("nan"), total_return=-0.1)
    baseline = _baseline()
    other = replace(baseline, scenario_id="other", entry_z=1.5)
    invalid_scenario = replace(baseline, scenario_id="invalid", entry_z=2.0)
    failed_scenario = replace(baseline, scenario_id="failed", entry_z=2.5)

    def scenario_result(scenario: ParameterScenario, metrics: Any) -> ScenarioResult:
        return ScenarioResult(
            scenario=scenario,
            status=ScenarioStatus.COMPLETED,
            error=None,
            walk_forward_result=None,
            calendar_metrics=metrics,
            conditional_metrics=None,
                common_horizon_metrics=metrics,
                common_horizon_error=None,
            calendar_oos_returns=pd.Series([0.0, 0.0]),
            conditional_oos_returns=pd.Series(dtype=float),
                common_horizon_returns=pd.Series([0.0, 0.0]),
            evaluated_start_position=0,
            evaluated_end_position=1,
            scheduled_oos_observations=2,
            available_oos_observations=2,
            selected_oos_observations=2,
            unavailable_oos_observations=0,
            selection_coverage=1.0,
            trade_count=0,
            total_transaction_cost=0.0,
            periods_per_year=252,
            risk_free_rate=0.0,
        )

    invalid = replace(
        scenario_result(invalid_scenario, None),
        status=ScenarioStatus.INVALID_CONFIGURATION,
    )
    failed = replace(
        scenario_result(failed_scenario, None),
        status=ScenarioStatus.FAILED,
    )
    summary = summarize_sensitivity(
        [
            scenario_result(baseline, defined),
            scenario_result(other, undefined),
            invalid,
            failed,
        ],
        "baseline",
    )

    assert summary.fraction_positive_sharpe == 1.0
    assert summary.metric_distributions.sharpe_ratio.observations == 1
    assert isinstance(summary.sharpe_fraction, MetricFractionSummary)
    assert summary.sharpe_fraction.eligible_scenarios == 2
    assert summary.sharpe_fraction.defined_scenarios == 1
    assert summary.sharpe_fraction.undefined_scenarios == 1
    assert summary.sharpe_fraction.positive_scenarios == 1
    assert summary.sharpe_fraction.invalid_scenarios == 1
    assert summary.sharpe_fraction.failed_scenarios == 1
    assert summary.sharpe_fraction.positive_fraction_defined == 1.0
    assert summary.sharpe_fraction.positive_fraction_all_eligible == 0.5


def test_baseline_neighbors_differ_by_exactly_one_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5],
        commission_bps_values=[2.0, 5.0],
    )

    summary = run_sensitivity_analysis(_prices(), scenarios, "baseline").summary

    assert summary.baseline_neighbor_count == 2
    assert summary.neighbor_metric_distributions.total_return.observations == 2
    assert summary.one_parameter_variant_count == 2
    assert summary.variant_metric_distributions.total_return.observations == 2


def test_neighbor_sign_and_positive_sharpe_summary_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        value = 0.01 if kwargs["entry_z"] in {1.0, 1.5} else -0.01
        return _walk_forward_result([value, -0.002, 0.003, 0.0])

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5, 2.0],
    )

    summary = run_sensitivity_analysis(_prices(), scenarios, "baseline").summary

    assert summary.baseline_neighbor_count == 2
    assert summary.fraction_neighbors_same_total_return_sign == pytest.approx(0.5)
    assert 0.0 <= summary.fraction_neighbors_positive_sharpe <= 1.0


def test_only_immediately_adjacent_one_parameter_variants_are_local_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5, 2.5],
    )

    summary = run_sensitivity_analysis(_prices(), scenarios, "baseline").summary

    assert summary.one_parameter_variant_count == 2
    assert summary.local_neighbor_count == 1
    neighbor = summary.local_neighbors[0]
    assert isinstance(neighbor, LocalNeighbor)
    assert neighbor.changed_parameter == "entry_z"
    assert neighbor.baseline_value == 1.0
    assert neighbor.neighbor_value == 1.5
    assert neighbor.absolute_distance == pytest.approx(0.5)
    assert neighbor.relative_distance == pytest.approx(0.5)


def test_local_neighbors_include_deterministic_lower_and_upper_adjacent_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    baseline = _baseline(entry_z=1.5)
    scenarios = generate_parameter_scenarios(
        baseline,
        entry_z_values=[1.0, 1.5, 2.0, 2.75],
    )

    summary = run_sensitivity_analysis(_prices(), scenarios, "baseline").summary

    assert [neighbor.neighbor_value for neighbor in summary.local_neighbors] == [
        1.0,
        2.0,
    ]
    assert all(
        neighbor.absolute_distance == pytest.approx(0.5)
        for neighbor in summary.local_neighbors
    )
    assert all(
        neighbor.relative_distance == pytest.approx(1.0 / 3.0)
        for neighbor in summary.local_neighbors
    )


def test_grid_density_and_dimension_scope_metadata_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5, 2.5],
        commission_bps_values=[2.0, 5.0],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")
    axes = {axis.parameter: axis for axis in result.axis_metadata}

    assert result.distribution_policy == "equal_weight_per_tested_grid_point"
    assert "not confidence intervals" in result.grid_density_warning
    assert isinstance(axes["entry_z"], ParameterAxisMetadata)
    assert axes["entry_z"].tested_values == (1.0, 1.5, 2.5)
    assert axes["entry_z"].count == 3
    assert axes["entry_z"].numeric_spacing == (0.5, 1.0)
    assert set(result.tested_dimensions) == {"entry_z", "commission_bps"}
    assert "entry_z" not in result.untested_material_dimensions
    assert "commission_bps" not in result.untested_material_dimensions
    assert "fdr_threshold" in result.untested_material_dimensions
    assert "execution_lag" in result.untested_material_dimensions
    assert "hedge_estimation_policy" in result.untested_material_dimensions


def test_denser_grid_changes_equal_weight_distribution_transparently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        return _walk_forward_result([kwargs["entry_z"] / 100.0, 0.0, 0.0])

    _install_fake_walk_forward(monkeypatch, factory)
    sparse = run_sensitivity_analysis(
        _prices(),
        generate_parameter_scenarios(
            _baseline(), entry_z_values=[1.0, 2.0]
        ),
        "baseline",
    )
    dense = run_sensitivity_analysis(
        _prices(),
        generate_parameter_scenarios(
            _baseline(), entry_z_values=[1.0, 1.5, 1.75, 2.0]
        ),
        "baseline",
    )

    assert sparse.summary.metric_distributions.total_return.median != (
        dense.summary.metric_distributions.total_return.median
    )
    assert sparse.distribution_policy == dense.distribution_policy
    assert next(
        axis.count for axis in dense.axis_metadata if axis.parameter == "entry_z"
    ) == 4


def test_cost_axes_are_forwarded_without_changing_cost_formulas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        commission_bps_values=[2.0, 8.0],
        slippage_bps_values=[1.0, 4.0],
    )

    run_sensitivity_analysis(_prices(), scenarios, "baseline")

    forwarded = {(call["commission_bps"], call["slippage_bps"]) for call in calls}
    assert forwarded == {(2.0, 1.0), (2.0, 4.0), (8.0, 1.0), (8.0, 4.0)}


def test_higher_real_costs_do_not_reduce_recorded_transaction_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        walkforward_module,
        "screen_pairs",
        lambda frame, groups=None, **kwargs: (_screening_result(len(frame)),),
    )
    baseline = _baseline(commission_bps=0.0, slippage_bps=0.0)
    scenarios = generate_parameter_scenarios(
        baseline,
        commission_bps_values=[0.0, 10.0],
    )

    result = run_sensitivity_analysis(
        _prices(68),
        scenarios,
        "baseline",
        walk_forward_kwargs={
            "target_gross_notional": 10_000.0,
            "initial_capital": 100_000.0,
            "fixed_commission_per_leg": 0.0,
            "max_half_life": 100.0,
            "hurst_threshold": 0.7,
        },
    )
    costs = {item.scenario.commission_bps: item.total_transaction_cost for item in result.scenarios}

    assert costs[10.0] >= costs[0.0]
    assert costs[10.0] > 0.0


@pytest.mark.parametrize(
    ("costs", "expected_total", "known", "unknown"),
    [
        ([1.25, 2.75], 4.0, 2, 0),
        ([1.25, np.nan], np.nan, 1, 1),
        ([1.25, np.inf], np.nan, 1, 1),
    ],
)
def test_equal_capital_reset_transaction_cost_total_requires_complete_components(
    monkeypatch: pytest.MonkeyPatch,
    costs: list[float],
    expected_total: float,
    known: int,
    unknown: int,
) -> None:
    fold = _CostFold(
        trade_count=0,
        backtest=_CostBacktest(
            pd.DataFrame({"transaction_cost": costs}, dtype=float)
        ),
    )
    produced = replace(_walk_forward_result([0.01, 0.0]), folds=(fold,))
    _install_fake_walk_forward(monkeypatch, lambda *args: produced)

    result = run_parameter_scenario(_prices(), _baseline())

    if np.isnan(expected_total):
        assert np.isnan(result.equal_capital_reset_fold_transaction_cost_total)
    else:
        assert result.equal_capital_reset_fold_transaction_cost_total == pytest.approx(
            expected_total
        )
    assert result.transaction_cost_known_components == known
    assert result.transaction_cost_unknown_components == unknown
    assert result.transaction_cost_basis == "equal_capital_reset_fold_dollar_total"


def test_formation_window_variation_reruns_screening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths: list[int] = []

    def screen(frame: pd.DataFrame, groups: Any = None, **kwargs: Any) -> tuple[PairScreeningResult, ...]:
        lengths.append(len(frame))
        return (_screening_result(len(frame)),)

    monkeypatch.setattr(walkforward_module, "screen_pairs", screen)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        formation_window_values=[50, 60],
    )

    run_sensitivity_analysis(
        _prices(68),
        scenarios,
        "baseline",
        walk_forward_kwargs={"target_gross_notional": 10_000.0, "initial_capital": 100_000.0},
    )

    assert 50 in lengths
    assert 60 in lengths


def test_trading_mutation_does_not_change_formation_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[pd.DataFrame] = []

    def screen(frame: pd.DataFrame, groups: Any = None, **kwargs: Any) -> tuple[PairScreeningResult, ...]:
        calls.append(frame.copy(deep=True))
        return (_screening_result(len(frame)),)

    monkeypatch.setattr(walkforward_module, "screen_pairs", screen)
    prices = _prices(68)
    changed = prices.copy(deep=True)
    changed.iloc[60:] *= 2.0

    original = run_parameter_scenario(prices, _baseline())
    modified = run_parameter_scenario(changed, _baseline())

    pd.testing.assert_frame_equal(calls[0], calls[1])
    assert original.walk_forward_result is not None
    assert modified.walk_forward_result is not None
    assert original.walk_forward_result.folds[0].selected_screening_result == (
        modified.walk_forward_result.folds[0].selected_screening_result
    )


def test_future_mutation_cannot_change_earlier_scenario_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        walkforward_module,
        "screen_pairs",
        lambda frame, groups=None, **kwargs: (_screening_result(len(frame)),),
    )
    prices = _prices(76)
    changed = prices.copy(deep=True)
    changed.iloc[68:] *= 1.8

    original = run_parameter_scenario(prices, _baseline())
    modified = run_parameter_scenario(changed, _baseline())

    assert original.walk_forward_result is not None
    assert modified.walk_forward_result is not None
    first = original.walk_forward_result.folds[0]
    changed_first = modified.walk_forward_result.folds[0]
    assert first.selected_screening_result == changed_first.selected_screening_result
    assert first.backtest is not None and changed_first.backtest is not None
    pd.testing.assert_frame_equal(first.backtest.accounting, changed_first.backtest.accounting)


def test_sensitivity_does_not_mutate_prices_or_parameter_iterables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    prices = _prices()
    before = prices.copy(deep=True)
    values = [1.0, 1.5]
    scenarios = generate_parameter_scenarios(_baseline(), entry_z_values=values)

    run_sensitivity_analysis(prices, scenarios, "baseline")

    pd.testing.assert_frame_equal(prices, before)
    assert values == [1.0, 1.5]


def test_future_prices_cannot_change_fixed_scenario_ids_or_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(_baseline(), entry_z_values=[1.0, 1.5])
    changed = _prices()
    changed.iloc[68:] *= 5.0

    original = run_sensitivity_analysis(_prices(), scenarios, "baseline")
    modified = run_sensitivity_analysis(changed, scenarios, "baseline")

    assert [item.scenario for item in original.scenarios] == [
        item.scenario for item in modified.scenarios
    ]
    assert original.baseline_scenario == modified.baseline_scenario == _baseline()


def test_shared_kwargs_cannot_override_predefined_scenario_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_walk_forward(monkeypatch)

    with pytest.raises(ValueError, match="scenario-controlled"):
        run_parameter_scenario(
            _prices(),
            _baseline(),
            walk_forward_kwargs={"entry_z": 9.0},
        )

    assert calls == []


def test_generator_grid_values_are_consumed_once_without_mutating_scenarios() -> None:
    values = (value for value in [1.5, 1.0])
    baseline = _baseline()

    scenarios = generate_parameter_scenarios(baseline, entry_z_values=values)

    assert {scenario.entry_z for scenario in scenarios} == {1.0, 1.5}
    assert baseline.entry_z == 1.0


def test_repeated_sensitivity_runs_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(_baseline(), entry_z_values=[1.0, 1.5])

    first = run_sensitivity_analysis(_prices(), scenarios, "baseline")
    second = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    pd.testing.assert_frame_equal(sensitivity_table(first), sensitivity_table(second))
    assert [item.status for item in first.scenarios] == [item.status for item in second.scenarios]
    for original, repeated in zip(first.scenarios, second.scenarios):
        pd.testing.assert_series_equal(original.calendar_oos_returns, repeated.calendar_oos_returns)


def test_stateful_group_provider_is_snapshotted_by_fold_independent_of_scenario_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5],
    )

    def execute(order: tuple[ParameterScenario, ...]) -> tuple[list[Any], int]:
        provider_calls = 0
        observed_snapshots: list[Any] = []

        def provider(fold: WalkForwardFold) -> dict[str, list[str]]:
            nonlocal provider_calls
            provider_calls += 1
            symbols = ["AAA", "BBB"] if provider_calls % 2 else ["AAA", "CCC"]
            return {"group": symbols}

        def factory(
            prices: pd.DataFrame,
            formation: int,
            trading: int,
            kwargs: Any,
        ) -> WalkForwardResult:
            fold = robustness_module.generate_walk_forward_folds(
                prices.index,
                formation,
                trading,
                step_size=trading,
                minimum_observations=50,
            )[0]
            snapshot = kwargs["groups"](fold)
            observed_snapshots.append(snapshot)
            assert isinstance(snapshot["group"], tuple)
            return _walk_forward_result([0.01, 0.0, 0.0, 0.0])

        _install_fake_walk_forward(monkeypatch, factory)
        run_sensitivity_analysis(_prices(), order, "baseline", groups=provider)
        return observed_snapshots, provider_calls

    forward, forward_calls = execute(scenarios)
    reverse, reverse_calls = execute(tuple(reversed(scenarios)))

    assert forward == reverse
    assert forward[0] == forward[1]
    # Three complete folds exist, and each boundary is materialized once per run.
    assert forward_calls == reverse_calls == 3


def test_group_provider_snapshot_owns_and_normalizes_caller_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_symbols = ["BBB", "AAA"]
    observed: list[tuple[str, ...]] = []

    def provider(fold: WalkForwardFold) -> dict[str, list[str]]:
        return {"group": shared_symbols}

    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        fold = robustness_module.generate_walk_forward_folds(
            prices.index,
            formation,
            trading,
            step_size=trading,
            minimum_observations=50,
        )[0]
        observed.append(kwargs["groups"](fold)["group"])
        shared_symbols[:] = ["CCC"]
        return _walk_forward_result([0.01, 0.0, 0.0, 0.0])

    _install_fake_walk_forward(monkeypatch, factory)
    run_sensitivity_analysis(
        _prices(),
        generate_parameter_scenarios(_baseline(), entry_z_values=[1.0, 1.5]),
        "baseline",
        groups=provider,
    )

    assert observed == [("AAA", "BBB"), ("AAA", "BBB")]


def test_provenance_is_promoted_and_conflicts_are_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        prices: pd.DataFrame,
        formation: int,
        trading: int,
        kwargs: Any,
    ) -> WalkForwardResult:
        baseline = kwargs["entry_z"] == 1.0
        return replace(
            _walk_forward_result([0.01, 0.0, 0.0, 0.0]),
            universe_provenance="validated-feed" if baseline else "caller-feed",
            cleaning_provenance="clean-v1" if baseline else "clean-v2",
            point_in_time_universe_validated=baseline,
            provenance_warnings=("caller warning",),
        )

    _install_fake_walk_forward(monkeypatch, factory)
    scenarios = generate_parameter_scenarios(
        _baseline(),
        entry_z_values=[1.0, 1.5],
    )

    result = run_sensitivity_analysis(_prices(), scenarios, "baseline")

    assert result.universe_provenance == ("caller-feed", "validated-feed")
    assert result.cleaning_provenance == ("clean-v1", "clean-v2")
    assert not result.point_in_time_universe_validated
    assert any("conflicting universe" in warning for warning in result.provenance_warnings)
    assert any("conflicting cleaning" in warning for warning in result.provenance_warnings)
    assert any("conservatively not validated" in warning for warning in result.provenance_warnings)


@pytest.mark.parametrize(
    ("factory", "field_name", "value"),
    [
        (_baseline, "entry_z", 9.0),
    ],
)
def test_parameter_scenario_is_immutable(factory: Any, field_name: str, value: Any) -> None:
    scenario = factory()
    with pytest.raises(FrozenInstanceError):
        setattr(scenario, field_name, value)


def test_scenario_summary_and_result_are_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    result = run_sensitivity_analysis(_prices(), (_baseline(),), "baseline")

    with pytest.raises(FrozenInstanceError):
        result.scenarios[0].status = ScenarioStatus.FAILED  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.summary.scenario_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.purpose = "optimization"  # type: ignore[misc]


def test_result_pandas_objects_are_defensive_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    produced = _walk_forward_result([0.01, 0.02, -0.01, 0.0])
    _install_fake_walk_forward(monkeypatch, lambda *args: produced)

    result = run_parameter_scenario(_prices(), _baseline())
    before = result.calendar_oos_returns.copy(deep=True)
    produced.calendar_oos_returns.iloc[:] = 99.0

    pd.testing.assert_series_equal(result.calendar_oos_returns, before)


def test_sensitivity_table_is_scenario_id_ordered_without_winner_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)
    scenarios = generate_parameter_scenarios(_baseline(), entry_z_values=[1.0, 1.5])
    reversed_scenarios = tuple(reversed(scenarios))
    result = run_sensitivity_analysis(_prices(), reversed_scenarios, "baseline")

    table = sensitivity_table(result)

    assert table["scenario_id"].tolist() == sorted(table["scenario_id"])
    assert "headline_common_total_return" in table
    assert "native_calendar_total_return" in table
    assert "equal_capital_reset_fold_transaction_cost_total" in table
    assert table["transaction_cost_basis"].eq(
        "equal_capital_reset_fold_dollar_total"
    ).all()
    assert not any(
        token in column.upper()
        for column in table.columns
        for token in ("BEST", "WINNER", "OPTIMAL")
    )
    for forbidden in ("best_parameters", "optimize", "maximize_sharpe", "select_best_scenario"):
        assert not hasattr(robustness_module, forbidden)


def test_purpose_and_warning_explicitly_prohibit_oos_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)

    result = run_sensitivity_analysis(_prices(), (_baseline(),), "baseline")

    assert result.purpose == "diagnostic_sensitivity_not_parameter_optimization"
    assert "must not" in result.warning
    assert "highest OOS" in result.warning


def test_scenario_analytics_unavailability_preserves_raw_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = [-1.0, 0.02, 0.0]
    _install_fake_walk_forward(
        monkeypatch,
        lambda *args: _walk_forward_result(raw),
    )

    result = run_parameter_scenario(_prices(), _baseline())

    assert result.status is ScenarioStatus.ANALYTICS_UNAVAILABLE
    assert result.calendar_metrics is None
    assert (
        result.calendar_metrics_status
        is MetricAvailabilityStatus.UNAVAILABLE_INVALID_RETURNS
    )
    assert result.calendar_oos_returns.tolist() == raw

    combined = run_sensitivity_analysis(
        _prices(),
        (_baseline(),),
        "baseline",
    )
    assert combined.common_horizon_structurally_available
    assert combined.common_horizon_fully_observed
    assert not combined.common_horizon_analytics_available
    assert (
        combined.common_horizon_analytics_status
        is MetricAvailabilityStatus.UNAVAILABLE_INVALID_RETURNS
    )
    assert combined.baseline_result.common_horizon_returns.tolist() == raw


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad data"),
        ValueError("causal accounting invariant"),
        TypeError("programming defect"),
        RuntimeError("invariant"),
    ],
)
def test_unexpected_walk_forward_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(
        robustness_module,
        "run_walk_forward_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    with pytest.raises(type(error), match=str(error)):
        run_parameter_scenario(_prices(), _baseline())


def test_high_cost_insolvency_cannot_disappear_as_a_failed_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robustness_module,
        "run_walk_forward_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Net equity must remain strictly positive after costs")
        ),
    )
    scenarios = generate_parameter_scenarios(
        _baseline(),
        commission_bps_values=[2.0, 10_000.0],
    )

    with pytest.raises(ValueError, match="strictly positive after costs"):
        run_sensitivity_analysis(_prices(), scenarios, "baseline")


def test_unexpected_common_analytics_value_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    produced = _walk_forward_result([0.01, 0.0, 0.0])
    _install_fake_walk_forward(monkeypatch, lambda *args: produced)
    monkeypatch.setattr(
        robustness_module,
        "calculate_core_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("unexpected analytics invariant")
        ),
    )

    with pytest.raises(ValueError, match="unexpected analytics invariant"):
        run_sensitivity_analysis(_prices(), (_baseline(),), "baseline")


def test_summary_structures_have_typed_distribution_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)

    result = run_sensitivity_analysis(_prices(), (_baseline(),), "baseline")

    assert isinstance(result, RobustnessResult)
    assert isinstance(result.summary, SensitivitySummary)
    assert isinstance(result.summary.metric_distributions.total_return, MetricDistribution)


def test_real_synthetic_walk_forward_hardening_regression_is_deterministic() -> None:
    prices, groups = make_synthetic_universe(n_days=400, seed=123)
    baseline = _baseline(
        formation_window=320,
        trading_window=40,
        screening_min_observations=100,
        zscore_lookback=20,
        commission_bps=0.0,
        slippage_bps=0.0,
        financing_rate=0.0,
        borrow_rate_y=0.0,
        borrow_rate_x=0.0,
    )
    scenarios = generate_parameter_scenarios(
        baseline,
        formation_window_values=[300, 320],
    )

    def point_in_time_groups(
        fold: WalkForwardFold,
    ) -> dict[str, tuple[str, ...]]:
        # First formation window runs normally; later folds are explicitly
        # unavailable because their point-in-time universe has no candidate pair.
        return groups if fold.formation_start_position == 0 else {"empty": ()}

    kwargs = {
        "fdr_threshold": 0.05,
        "max_half_life": 100.0,
        "hurst_threshold": 0.7,
        "target_gross_notional": 10_000.0,
        "initial_capital": 100_000.0,
    }

    first = run_sensitivity_analysis(
        prices,
        scenarios,
        "baseline",
        groups=point_in_time_groups,
        walk_forward_kwargs=kwargs,
    )
    second = run_sensitivity_analysis(
        prices,
        tuple(reversed(scenarios)),
        "baseline",
        groups=point_in_time_groups,
        walk_forward_kwargs=kwargs,
    )

    assert first.common_horizon_index.equals(second.common_horizon_index)
    assert first.common_horizon_observations == 60
    assert first.common_horizon_structurally_available
    assert not first.common_horizon_fully_observed
    assert not first.common_horizon_analytics_available
    assert first.summary.total_return_fraction.eligible_scenarios == 2
    assert first.summary.total_return_fraction.defined_scenarios == 0
    assert first.summary.total_return_fraction.undefined_scenarios == 2
    assert first.universe_provenance == (
        "per_fold_groups_caller_supplied_unvalidated",
    )
    assert not first.point_in_time_universe_validated
    pd.testing.assert_frame_equal(sensitivity_table(first), sensitivity_table(second))
    for original, repeated in zip(first.scenarios, second.scenarios):
        assert original.scenario == repeated.scenario
        pd.testing.assert_series_equal(
            original.calendar_oos_returns,
            repeated.calendar_oos_returns,
        )
