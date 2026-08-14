"""Focused tests for diagnostic parameter robustness and sensitivity analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.analytics import calculate_core_metrics, calculate_drawdown_metrics
from pairs_trading.robustness import (
    MetricDistribution,
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


def test_common_horizon_uses_only_available_index_intersection(
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

    assert result.common_horizon_index.tolist() == [52]
    assert result.common_horizon_available is False
    assert all(item.common_horizon_metrics is None for item in result.scenarios)


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

    def scenario_result(scenario: ParameterScenario, metrics: Any) -> ScenarioResult:
        return ScenarioResult(
            scenario=scenario,
            status=ScenarioStatus.COMPLETED,
            error=None,
            walk_forward_result=None,
            calendar_metrics=metrics,
            conditional_metrics=None,
            common_horizon_metrics=None,
            common_horizon_error="fixture",
            calendar_oos_returns=pd.Series([0.0, 0.0]),
            conditional_oos_returns=pd.Series(dtype=float),
            common_horizon_returns=pd.Series(dtype=float),
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

    summary = summarize_sensitivity(
        [scenario_result(baseline, defined), scenario_result(other, undefined)],
        "baseline",
    )

    assert summary.fraction_positive_sharpe == 1.0
    assert summary.metric_distributions.sharpe_ratio.observations == 1


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

    assert first.summary == second.summary
    assert [item.status for item in first.scenarios] == [item.status for item in second.scenarios]
    for original, repeated in zip(first.scenarios, second.scenarios):
        pd.testing.assert_series_equal(original.calendar_oos_returns, repeated.calendar_oos_returns)


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
    assert result.calendar_metrics is not None
    assert result.calendar_metrics.total_return == -1.0
    assert np.isnan(result.calendar_metrics.sharpe_ratio)
    assert result.calendar_oos_returns.tolist() == raw


def test_expected_scenario_failure_is_recorded_but_invariant_error_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robustness_module,
        "run_walk_forward_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad data")),
    )
    failed = run_parameter_scenario(_prices(), _baseline())
    assert failed.status is ScenarioStatus.FAILED
    assert "bad data" in str(failed.error)

    monkeypatch.setattr(
        robustness_module,
        "run_walk_forward_analysis",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("invariant")),
    )
    with pytest.raises(RuntimeError, match="invariant"):
        run_parameter_scenario(_prices(), _baseline())


def test_summary_structures_have_typed_distribution_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_walk_forward(monkeypatch)

    result = run_sensitivity_analysis(_prices(), (_baseline(),), "baseline")

    assert isinstance(result, RobustnessResult)
    assert isinstance(result.summary, SensitivitySummary)
    assert isinstance(result.summary.metric_distributions.total_return, MetricDistribution)
