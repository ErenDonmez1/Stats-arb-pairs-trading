"""Focused tests for Milestone 9B deterministic portfolio risk controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pairs_trading.portfolio_risk as risk_module
from pairs_trading.portfolio import (
    PairPortfolioInput,
    PortfolioAvailability,
    run_multi_pair_portfolio,
)
from pairs_trading.portfolio_risk import (
    PortfolioRiskBreach,
    PortfolioRiskLimits,
    PortfolioRiskResult,
    PortfolioRiskSummary,
    RiskAction,
    RiskControlStatus,
    RiskState,
    apply_portfolio_risk_policy,
    build_portfolio_risk_schedule,
    evaluate_portfolio_risk,
    run_portfolio_risk_controls,
    summarize_portfolio_risk,
    validate_portfolio_risk_limits,
)
from pairs_trading.walkforward import WalkForwardAnalyticsStatus, WalkForwardResult


def _index(rows: int = 4) -> pd.RangeIndex:
    return pd.RangeIndex(rows, name="row")


def _pair_input(
    pair_id: str,
    returns: list[float],
    *,
    index: pd.Index | None = None,
    active: list[bool | None] | None = None,
    market_value_y: list[float] | None = None,
    market_value_x: list[float] | None = None,
    source_capital_basis: float | None = 1_000.0,
) -> PairPortfolioInput:
    symbol_y, symbol_x = pair_id.split("|")
    row_index = _index(len(returns)) if index is None else index
    active_values = [False] * len(returns) if active is None else active
    values_y = [0.0] * len(returns) if market_value_y is None else market_value_y
    values_x = [0.0] * len(returns) if market_value_x is None else market_value_x
    market_y = np.asarray(values_y, dtype=float)
    market_x = np.asarray(values_x, dtype=float)
    exposure = pd.DataFrame(
        {
            "market_value_y": market_y,
            "market_value_x": market_x,
            "gross_exposure": np.abs(market_y) + np.abs(market_x),
            "long_exposure": np.maximum(market_y, 0.0)
            + np.maximum(market_x, 0.0),
            "short_exposure": np.maximum(-market_y, 0.0)
            + np.maximum(-market_x, 0.0),
            "net_exposure": market_y + market_x,
        },
        index=row_index,
    )
    return PairPortfolioInput(
        pair_id=pair_id,
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        calendar_returns=pd.Series(returns, index=row_index, name="pair_return"),
        active_state=pd.Series(active_values, index=row_index, dtype="boolean"),
        execution_rows=pd.Series(False, index=row_index, dtype=bool),
        source_exposure=exposure,
        source_capital_basis=source_capital_basis,
        source_type="risk-fixture",
        capital_policy="standalone_pair_capital",
        point_in_time_universe_validated=False,
        universe_provenance="caller-supplied",
        cleaning_provenance="caller-supplied",
        provenance_warnings=("risk fixture provenance is unvalidated",),
    )


def _portfolio(*, index: pd.Index | None = None, reverse: bool = False):
    row_index = _index() if index is None else index
    first = _pair_input(
        "AAA|BBB",
        [0.0, 0.20, -0.25, 0.0],
        index=row_index,
        active=[False, True, True, False],
        market_value_y=[0.0, 600.0, 900.0, 900.0],
        market_value_x=[0.0, -400.0, -600.0, -600.0],
    )
    second = _pair_input(
        "CCC|DDD",
        [0.0, 0.0, 0.0, 0.0],
        index=row_index,
        active=[False, False, True, True],
        market_value_y=[0.0, 500.0, 500.0, 0.0],
        market_value_x=[0.0, -500.0, -500.0, 0.0],
    )
    items = [(first.pair_id, first), (second.pair_id, second)]
    if reverse:
        items.reverse()
    return run_multi_pair_portfolio(items, 10_000.0)


def _single_pair_portfolio(returns: list[float]):
    pair = _pair_input(
        "AAA|BBB",
        returns,
        active=[True] * len(returns),
    )
    return run_multi_pair_portfolio({pair.pair_id: pair}, 1_000.0)


def _walk_forward_no_selection(index: pd.Index) -> WalkForwardResult:
    calendar = pd.Series(0.0, index=index, name="calendar_oos_return")
    return WalkForwardResult(
        folds=(),
        fold_count=1,
        completed_fold_count=0,
        no_selection_fold_count=1,
        insufficient_data_fold_count=0,
        scheduled_oos_observations=len(index),
        scheduled_eligible_oos_observations=len(index),
        selected_oos_observations=0,
        no_selection_oos_observations=len(index),
        unavailable_oos_observations=0,
        selection_coverage=0.0,
        conditional_oos_returns=pd.Series(dtype=float, name="conditional_oos_return"),
        calendar_oos_returns=calendar,
        conditional_performance_report=None,
        calendar_performance_report=None,
        conditional_analytics_status=WalkForwardAnalyticsStatus.UNAVAILABLE,
        calendar_analytics_status=WalkForwardAnalyticsStatus.AVAILABLE,
        conditional_analytics_error="No selected pair.",
        calendar_analytics_error=None,
        capital_policy="equal_capital_reset",
        aggregate_return_policy="time_weighted_equal_capital_reset",
        inactive_capital_policy="zero_return_cash_for_no_selection_rows",
        selection_coverage_denominator="selected_plus_no_selection_scheduled_rows",
        aggregate_dollar_pnl_available=False,
        aggregate_trade_dollar_metrics_available=False,
        universe_provenance="static_groups_caller_supplied_unvalidated",
        cleaning_provenance="caller-supplied-cleaning",
        point_in_time_universe_validated=False,
        provenance_warnings=("No-selection walk-forward fixture.",),
        evaluated_start_position=0,
        evaluated_end_position=len(index) - 1,
        evaluated_start_label=index[0],
        evaluated_end_label=index[-1],
        discarded_terminal_rows=0,
    )


def _status(result: PortfolioRiskResult, control: str, row: int) -> RiskControlStatus:
    return result.control_statuses[control].iloc[row]


def _invoke_direct_status_consumer(
    consumer: str,
    result: PortfolioRiskResult,
    statuses: pd.DataFrame,
) -> pd.DataFrame | PortfolioRiskSummary:
    if consumer == "schedule":
        return build_portfolio_risk_schedule(
            result.observed_metrics,
            statuses,
            result.observations,
            result.policy_state,
            result.requested_actions,
            result.risk_schedule["risk_action_intent"],
            result.risk_schedule["action_effective_position"],
            result.risk_schedule["action_effective_label"],
        )
    if consumer == "summary":
        return summarize_portfolio_risk(
            result.observed_metrics,
            statuses,
            result.breach_records,
            result.requested_actions,
            result.summary.configured_controls,
        )
    raise AssertionError(f"Unknown test consumer {consumer!r}.")


def test_empty_limits_are_explicitly_not_configured() -> None:
    result = run_portfolio_risk_controls(_portfolio(), PortfolioRiskLimits())
    assert result.control_statuses.map(
        lambda value: value is RiskControlStatus.NOT_CONFIGURED
    ).all(axis=None)
    assert all(
        observation.overall_status is RiskControlStatus.NOT_CONFIGURED
        for observation in result.observations
    )
    assert result.summary.configured_controls == ()
    assert result.policy_state.map(lambda value: value is RiskState.NORMAL).all()


def test_gross_exposure_below_equal_and_above_limit_boundaries() -> None:
    portfolio = _portfolio()
    baseline = run_portfolio_risk_controls(portfolio)
    maximum = baseline.observed_metrics["gross_exposure_ratio"].max()
    equal = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_gross_exposure_ratio=maximum),
    )
    assert not equal.control_statuses["gross_exposure"].eq(
        RiskControlStatus.BREACH
    ).any()
    high = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_gross_exposure_ratio=maximum + 1.0),
    )
    assert high.control_statuses["gross_exposure"].eq(
        RiskControlStatus.WITHIN_LIMIT
    ).all()
    low = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_gross_exposure_ratio=1.0),
    )
    assert _status(low, "gross_exposure", 2) is RiskControlStatus.BREACH


def test_missing_gross_exposure_is_unevaluable_not_safe() -> None:
    portfolio = _portfolio()
    aggregate = portfolio.aggregate_exposures.copy(deep=True)
    aggregate.loc[2, "gross_exposure_ratio"] = np.nan
    changed = replace(portfolio, aggregate_exposures=aggregate)
    result = run_portfolio_risk_controls(
        changed,
        PortfolioRiskLimits(max_gross_exposure_ratio=2.0),
    )
    assert _status(result, "gross_exposure", 2) is RiskControlStatus.UNEVALUABLE
    assert result.observations[2].unevaluable_control_count == 1
    assert result.observations[2].overall_status is RiskControlStatus.UNEVALUABLE
    assert result.policy_state.iloc[2] is RiskState.NORMAL
    assert any(
        "does not override an UNEVALUABLE overall status" in warning
        for warning in result.warnings
    )


def test_net_exposure_uses_absolute_value_and_equality_is_within_limit() -> None:
    portfolio = _portfolio()
    result = run_portfolio_risk_controls(portfolio)
    expected = (
        portfolio.aggregate_exposures["total_net_exposure"].abs()
        / portfolio.portfolio_equity
    )
    pd.testing.assert_series_equal(
        result.observed_metrics["abs_net_exposure_ratio"],
        expected.rename("abs_net_exposure_ratio"),
    )
    limit = float(expected.max())
    boundary = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_abs_net_exposure_ratio=limit),
    )
    assert not boundary.control_statuses["abs_net_exposure"].eq(
        RiskControlStatus.BREACH
    ).any()


def test_long_and_short_exposure_controls_use_portfolio_equity() -> None:
    portfolio = _portfolio()
    baseline = run_portfolio_risk_controls(portfolio)
    expected_long = (
        portfolio.aggregate_exposures["total_long_exposure"]
        / portfolio.portfolio_equity
    )
    expected_short = (
        portfolio.aggregate_exposures["total_short_exposure"]
        / portfolio.portfolio_equity
    )
    pd.testing.assert_series_equal(
        baseline.observed_metrics["long_exposure_ratio"],
        expected_long.rename("long_exposure_ratio"),
    )
    pd.testing.assert_series_equal(
        baseline.observed_metrics["short_exposure_ratio"],
        expected_short.rename("short_exposure_ratio"),
    )
    boundary = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(
            max_long_exposure_ratio=float(expected_long.max()),
            max_short_exposure_ratio=float(expected_short.max()),
        ),
    )
    assert not boundary.control_statuses[["long_exposure", "short_exposure"]].map(
        lambda value: value is RiskControlStatus.BREACH
    ).any(axis=None)


def test_pair_concentration_uses_current_drifted_equity_weights() -> None:
    portfolio = _portfolio()
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_pair_equity_weight=0.54),
    )
    expected = portfolio.pair_current_equity_weights.max(axis=1)
    pd.testing.assert_series_equal(
        result.observed_metrics["largest_pair_equity_weight"],
        expected.rename("largest_pair_equity_weight"),
    )
    assert expected.iloc[1] > 0.5
    assert dict(portfolio.allocation_policy.pair_weights) == {
        "AAA|BBB": 0.5,
        "CCC|DDD": 0.5,
    }
    assert _status(result, "pair_concentration", 1) is RiskControlStatus.BREACH
    assert _status(result, "pair_concentration", 2) is RiskControlStatus.WITHIN_LIMIT


def test_symbol_concentration_uses_unnetted_sleeve_gross_over_equity() -> None:
    portfolio = _portfolio()
    result = run_portfolio_risk_controls(portfolio)
    unnetted = portfolio.symbol_exposures.xs(
        "unnetted_sleeve_gross_market_value",
        level="metric",
        axis=1,
    )
    expected = unnetted.max(axis=1) / portfolio.portfolio_equity
    pd.testing.assert_series_equal(
        result.observed_metrics[
            "largest_symbol_unnetted_gross_exposure_ratio"
        ],
        expected.rename("largest_symbol_unnetted_gross_exposure_ratio"),
    )
    assert result.symbol_gross_measure == "unnetted_sleeve_gross_market_value"


def test_missing_any_symbol_exposure_makes_concentration_unevaluable() -> None:
    portfolio = _portfolio()
    symbols = portfolio.symbol_exposures.copy(deep=True)
    symbols.loc[2, ("AAA", "unnetted_sleeve_gross_market_value")] = np.nan
    result = run_portfolio_risk_controls(
        replace(portfolio, symbol_exposures=symbols),
        PortfolioRiskLimits(max_symbol_gross_exposure_ratio=1.0),
    )
    assert _status(result, "symbol_concentration", 2) is RiskControlStatus.UNEVALUABLE


def test_active_pair_count_uses_upstream_state_not_nonzero_return() -> None:
    portfolio = _portfolio()
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_active_pairs=1),
    )
    np.testing.assert_allclose(
        result.observed_metrics["active_pair_count"],
        portfolio.portfolio_schedule["active_pair_count"],
    )
    assert _status(result, "active_pairs", 2) is RiskControlStatus.BREACH
    assert portfolio.portfolio_returns.iloc[2] != 0.0
    assert result.observed_metrics["active_pair_count"].iloc[3] == 1.0
    assert portfolio.portfolio_returns.iloc[3] == 0.0


def test_nonzero_return_does_not_imply_active_pair() -> None:
    pair = _pair_input("AAA|BBB", [0.0, 0.2, 0.0, 0.0], active=[False] * 4)
    portfolio = run_multi_pair_portfolio({pair.pair_id: pair}, 1_000.0)
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_active_pairs=1),
    )
    assert result.observed_metrics["active_pair_count"].eq(0.0).all()
    assert result.control_statuses["active_pairs"].eq(
        RiskControlStatus.WITHIN_LIMIT
    ).all()


def test_missing_active_state_makes_control_unevaluable() -> None:
    portfolio = _portfolio()
    schedule = portfolio.portfolio_schedule.copy(deep=True)
    schedule.loc[1, "active_pair_count_available"] = False
    result = run_portfolio_risk_controls(
        replace(portfolio, portfolio_schedule=schedule),
        PortfolioRiskLimits(max_active_pairs=2),
    )
    assert np.isnan(result.observed_metrics["active_pair_count"].iloc[1])
    assert _status(result, "active_pairs", 1) is RiskControlStatus.UNEVALUABLE


def test_drawdown_uses_causal_running_portfolio_equity_peak() -> None:
    portfolio = _portfolio()
    result = run_portfolio_risk_controls(portfolio)
    expected_peak = portfolio.portfolio_equity.cummax().clip(
        lower=portfolio.initial_capital
    )
    expected = portfolio.portfolio_equity / expected_peak - 1.0
    pd.testing.assert_series_equal(
        result.observed_metrics["running_equity_peak"],
        expected_peak.rename("running_equity_peak"),
    )
    pd.testing.assert_series_equal(
        result.observed_metrics["drawdown"],
        expected.rename("drawdown"),
    )


def test_first_row_drawdown_uses_initial_capital_and_exact_boundaries() -> None:
    portfolio = _single_pair_portfolio([-0.20, 0.0, 0.0])
    assert portfolio.initial_capital == pytest.approx(1_000.0)
    assert portfolio.portfolio_equity.iloc[0] == pytest.approx(800.0)

    breached = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_drawdown=0.10),
    )
    assert breached.observed_metrics["running_equity_peak"].iloc[0] == pytest.approx(
        1_000.0
    )
    assert breached.observed_metrics["drawdown"].iloc[0] == pytest.approx(-0.20)
    assert breached.observed_metrics["drawdown_magnitude"].iloc[0] == pytest.approx(
        0.20
    )
    assert _status(breached, "drawdown", 0) is RiskControlStatus.BREACH
    assert breached.risk_schedule.index.equals(portfolio.portfolio_equity.index)
    assert len(breached.risk_schedule) == len(portfolio.portfolio_equity)

    equal = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_drawdown=0.20),
    )
    assert _status(equal, "drawdown", 0) is RiskControlStatus.WITHIN_LIMIT


def test_first_row_total_loss_reports_full_drawdown() -> None:
    portfolio = _single_pair_portfolio([-1.0, 0.0, 0.0])
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_drawdown=0.99),
    )
    assert portfolio.portfolio_equity.iloc[0] == pytest.approx(0.0)
    assert result.observed_metrics["running_equity_peak"].iloc[0] == pytest.approx(
        1_000.0
    )
    assert result.observed_metrics["drawdown"].iloc[0] == pytest.approx(-1.0)
    assert result.observed_metrics["drawdown_magnitude"].iloc[0] == pytest.approx(
        1.0
    )
    assert _status(result, "drawdown", 0) is RiskControlStatus.BREACH
    assert result.policy_state.iloc[0] is RiskState.TERMINAL


def test_first_gain_and_later_new_high_update_the_causal_peak() -> None:
    first_gain = run_portfolio_risk_controls(
        _single_pair_portfolio([0.10, 0.0]),
    )
    assert first_gain.observed_metrics["running_equity_peak"].iloc[0] == pytest.approx(
        1_100.0
    )
    assert first_gain.observed_metrics["drawdown"].iloc[0] == pytest.approx(0.0)

    later_peak = run_portfolio_risk_controls(
        _single_pair_portfolio([0.0, 0.10, -0.20]),
    )
    assert later_peak.observed_metrics["running_equity_peak"].tolist() == pytest.approx(
        [1_000.0, 1_100.0, 1_100.0]
    )
    assert later_peak.observed_metrics["drawdown"].iloc[2] == pytest.approx(-0.20)


def test_drawdown_is_future_invariant_without_exposing_an_initial_row() -> None:
    original_portfolio = _single_pair_portfolio([-0.20, 0.10, -0.05])
    modified_portfolio = _single_pair_portfolio([-0.20, 0.10, 0.50])
    original = run_portfolio_risk_controls(original_portfolio)
    modified = run_portfolio_risk_controls(modified_portfolio)

    pd.testing.assert_frame_equal(
        original.observed_metrics.iloc[:2],
        modified.observed_metrics.iloc[:2],
    )
    assert original.observed_metrics.index.equals(original_portfolio.portfolio_equity.index)
    assert modified.observed_metrics.index.equals(modified_portfolio.portfolio_equity.index)
    assert len(original.observed_metrics) == len(original_portfolio.portfolio_equity)


def test_drawdown_equality_is_within_and_stricter_limit_breaches() -> None:
    portfolio = _portfolio()
    magnitude = float(
        run_portfolio_risk_controls(portfolio)
        .observed_metrics["drawdown_magnitude"]
        .max()
    )
    equal = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_drawdown=magnitude),
    )
    assert not equal.control_statuses["drawdown"].eq(RiskControlStatus.BREACH).any()
    breached = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_drawdown=magnitude / 2.0),
    )
    assert _status(breached, "drawdown", 2) is RiskControlStatus.BREACH


def test_minimum_equity_ratio_equality_and_breach_are_exact() -> None:
    portfolio = _portfolio()
    minimum = float((portfolio.portfolio_equity / portfolio.initial_capital).min())
    equal = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(min_portfolio_equity_ratio=minimum),
    )
    assert not equal.control_statuses["minimum_equity"].eq(
        RiskControlStatus.BREACH
    ).any()
    breached = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(min_portfolio_equity_ratio=minimum + 0.01),
    )
    assert _status(breached, "minimum_equity", 2) is RiskControlStatus.BREACH
    assert breached.observed_metrics["portfolio_equity_ratio"].iloc[2] == pytest.approx(
        portfolio.portfolio_equity.iloc[2] / portfolio.initial_capital
    )


def test_missing_equity_breaks_all_equity_dependent_controls_permanently() -> None:
    portfolio = _portfolio()
    equity = portfolio.portfolio_equity.copy(deep=True)
    equity.iloc[2] = np.nan
    changed = replace(portfolio, portfolio_equity=equity)
    limits = PortfolioRiskLimits(
        max_gross_exposure_ratio=2.0,
        max_abs_net_exposure_ratio=2.0,
        max_pair_equity_weight=1.0,
        max_symbol_gross_exposure_ratio=2.0,
        max_drawdown=0.5,
        min_portfolio_equity_ratio=0.5,
    )
    result = run_portfolio_risk_controls(changed, limits)
    dependent = (
        "gross_exposure",
        "abs_net_exposure",
        "pair_concentration",
        "symbol_concentration",
        "drawdown",
        "minimum_equity",
    )
    for control in dependent:
        assert _status(result, control, 2) is RiskControlStatus.UNEVALUABLE
        assert _status(result, control, 3) is RiskControlStatus.UNEVALUABLE
    assert np.isnan(result.observed_metrics["running_equity_peak"].iloc[2:]).all()
    assert np.isnan(result.observed_metrics["drawdown"].iloc[2:]).all()


def test_insolvent_sleeve_is_visible_and_optional_limit_breaches() -> None:
    survivor = _pair_input("AAA|BBB", [0.0, 0.0, 0.0, 0.0])
    insolvent = _pair_input("CCC|DDD", [0.0, -1.0, 2.0, 0.0])
    portfolio = run_multi_pair_portfolio(
        {survivor.pair_id: survivor, insolvent.pair_id: insolvent},
        1_000.0,
    )
    assert portfolio.availability is PortfolioAvailability.PARTIALLY_AVAILABLE
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_insolvent_sleeves=0),
    )
    assert result.observed_metrics["insolvent_sleeve_count"].iloc[1] == 1.0
    assert result.observed_metrics["insolvent_pair_ids"].iloc[1] == ("CCC|DDD",)
    assert _status(result, "insolvent_sleeves", 1) is RiskControlStatus.BREACH
    assert _status(result, "insolvent_sleeves", 3) is RiskControlStatus.BREACH


def test_control_counts_reconcile_on_every_row() -> None:
    portfolio = _portfolio()
    symbols = portfolio.symbol_exposures.copy(deep=True)
    symbols.loc[1, ("AAA", "unnetted_sleeve_gross_market_value")] = np.nan
    result = run_portfolio_risk_controls(
        replace(portfolio, symbol_exposures=symbols),
        PortfolioRiskLimits(
            max_gross_exposure_ratio=1.0,
            max_symbol_gross_exposure_ratio=1.0,
            max_active_pairs=1,
        ),
    )
    for observation in result.observations:
        assert observation.configured_control_count == (
            observation.evaluated_control_count
            + observation.unevaluable_control_count
        )
        assert observation.breach_control_count <= observation.evaluated_control_count


def test_breach_dominates_unevaluable_and_unevaluable_dominates_safe() -> None:
    portfolio = _portfolio()
    symbols = portfolio.symbol_exposures.copy(deep=True)
    symbols.loc[2, ("AAA", "unnetted_sleeve_gross_market_value")] = np.nan
    changed = replace(portfolio, symbol_exposures=symbols)
    breach = run_portfolio_risk_controls(
        changed,
        PortfolioRiskLimits(
            max_gross_exposure_ratio=1.0,
            max_symbol_gross_exposure_ratio=1.0,
        ),
    )
    assert breach.observations[2].overall_status is RiskControlStatus.BREACH
    unevaluable = run_portfolio_risk_controls(
        changed,
        PortfolioRiskLimits(
            max_gross_exposure_ratio=2.0,
            max_symbol_gross_exposure_ratio=1.0,
        ),
    )
    assert unevaluable.observations[2].overall_status is RiskControlStatus.UNEVALUABLE


def test_breach_records_and_summary_order_counts_and_extrema_are_exact() -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(
            max_gross_exposure_ratio=1.0,
            max_pair_equity_weight=0.54,
            max_active_pairs=1,
        ),
    )
    keys = [(item.row_position, item.control_name) for item in result.breach_records]
    expected_order = {name: i for i, name in enumerate(result.control_statuses.columns)}
    assert keys == sorted(keys, key=lambda item: (item[0], expected_order[item[1]]))
    assert result.summary.first_breach_position == 1
    assert result.summary.first_breach_label == result.risk_schedule.index[1]
    assert result.summary.last_breach_position == 2
    assert result.summary.last_breach_label == result.risk_schedule.index[2]
    assert result.summary.total_breach_events == len(result.breach_records)
    assert dict(result.summary.breach_count_by_control) == {
        control: int(result.control_statuses[control].eq(RiskControlStatus.BREACH).sum())
        for control in result.summary.configured_controls
    }
    assert result.summary.maximum_observed_gross_exposure_ratio == pytest.approx(
        result.observed_metrics["gross_exposure_ratio"].max()
    )
    assert result.summary.maximum_observed_pair_weight == pytest.approx(
        result.observed_metrics["largest_pair_equity_weight"].max()
    )
    assert result.summary.minimum_equity_ratio == pytest.approx(
        result.observed_metrics["portfolio_equity_ratio"].min()
    )


def test_never_evaluable_metric_summary_remains_nan() -> None:
    portfolio = _portfolio()
    aggregate = portfolio.aggregate_exposures.copy(deep=True)
    aggregate["gross_exposure_ratio"] = np.nan
    result = run_portfolio_risk_controls(
        replace(portfolio, aggregate_exposures=aggregate),
        PortfolioRiskLimits(max_gross_exposure_ratio=2.0),
    )
    assert np.isnan(result.summary.maximum_observed_gross_exposure_ratio)
    assert dict(result.summary.evaluated_count_by_control)["gross_exposure"] == 0
    assert dict(result.summary.unevaluable_count_by_control)["gross_exposure"] == len(
        result.risk_schedule
    )


def test_none_action_never_changes_portfolio_accounting() -> None:
    portfolio = _portfolio()
    equity_before = portfolio.portfolio_equity.copy(deep=True)
    returns_before = portfolio.portfolio_returns.copy(deep=True)
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_gross_exposure_ratio=1.0),
    )
    assert result.requested_actions.map(lambda value: value is RiskAction.NONE).all()
    pd.testing.assert_series_equal(portfolio.portfolio_equity, equity_before)
    pd.testing.assert_series_equal(portfolio.portfolio_returns, returns_before)
    assert result.action_execution_policy == "intent_only_no_position_or_accounting_replay"


def test_halt_intent_activates_next_row_and_is_sticky() -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(
            max_pair_equity_weight=0.54,
            breach_action=RiskAction.HALT_NEW_ENTRIES,
        ),
    )
    assert result.risk_schedule["risk_action_intent"].iloc[1] is RiskAction.HALT_NEW_ENTRIES
    assert result.policy_state.iloc[1] is RiskState.NORMAL
    assert result.risk_schedule["action_effective_position"].iloc[1] == 2
    assert result.requested_actions.iloc[2] is RiskAction.HALT_NEW_ENTRIES
    assert result.policy_state.iloc[2] is RiskState.ENTRY_HALTED
    assert result.policy_state.iloc[3] is RiskState.ENTRY_HALTED
    assert not result.risk_schedule["action_executed"].any()
    record = next(
        item
        for item in result.breach_records
        if item.row_position == 1 and item.control_name == "pair_concentration"
    )
    assert record.requested_action is RiskAction.HALT_NEW_ENTRIES
    assert record.action_effective_position == 2
    assert result.requested_actions.iloc[2] is record.requested_action


def test_same_row_breach_cannot_rewrite_already_observed_row() -> None:
    portfolio = _portfolio()
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(
            max_pair_equity_weight=0.54,
            breach_action="HALT_NEW_ENTRIES",
        ),
    )
    assert result.policy_state.iloc[1] is RiskState.NORMAL
    assert result.requested_actions.iloc[1] is RiskAction.NONE
    assert result.risk_schedule["portfolio_equity_ratio"].iloc[1] == pytest.approx(
        portfolio.portfolio_equity.iloc[1] / portfolio.initial_capital
    )


def test_initial_row_breach_action_is_effective_only_from_row_one() -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(
            max_pair_equity_weight=0.40,
            breach_action=RiskAction.HALT_NEW_ENTRIES,
        ),
    )
    assert _status(result, "pair_concentration", 0) is RiskControlStatus.BREACH
    assert result.policy_state.iloc[0] is RiskState.NORMAL
    assert result.requested_actions.iloc[0] is RiskAction.NONE
    assert result.risk_schedule["action_effective_position"].iloc[0] == 1
    assert result.requested_actions.iloc[1] is RiskAction.HALT_NEW_ENTRIES
    assert result.policy_state.iloc[1] is RiskState.ENTRY_HALTED


def test_final_row_breach_has_no_future_effective_action() -> None:
    portfolio = _portfolio()
    # Active-pair count equals one only on the final row after the earlier row-2
    # count is made unavailable, so this isolates final-row action timing.
    schedule = portfolio.portfolio_schedule.copy(deep=True)
    schedule.loc[:2, "active_pair_count"] = 0
    changed = replace(portfolio, portfolio_schedule=schedule)
    result = run_portfolio_risk_controls(
        changed,
        PortfolioRiskLimits(
            max_active_pairs=0 + 1,
            breach_action=RiskAction.HALT_NEW_ENTRIES,
        ),
    )
    # Equality is within limit, so force a final breach with a zero insolvency cap.
    insolvency = changed.pair_insolvency_state.copy(deep=True)
    insolvency.iloc[-1, 0] = True
    final = run_portfolio_risk_controls(
        replace(changed, pair_insolvency_state=insolvency),
        PortfolioRiskLimits(
            max_insolvent_sleeves=0,
            breach_action=RiskAction.HALT_NEW_ENTRIES,
        ),
    )
    assert result.summary.total_breach_events == 0
    assert _status(final, "insolvent_sleeves", -1) is RiskControlStatus.BREACH
    record = final.breach_records[-1]
    assert record.row_position == len(final.risk_schedule) - 1
    assert final.risk_schedule["risk_action_intent"].iloc[-1] is RiskAction.NONE
    assert record.requested_action is RiskAction.NONE
    assert record.action_effective_position is None
    assert record.action_effective_label is None
    assert final.requested_actions.map(lambda value: value is RiskAction.NONE).all()


@pytest.mark.parametrize(
    "configured_action",
    (RiskAction.LIQUIDATE_ALL, RiskAction.HALT_NEW_ENTRIES),
)
def test_terminal_breach_record_contains_no_unemitted_action(
    configured_action: RiskAction,
) -> None:
    result = run_portfolio_risk_controls(
        _single_pair_portfolio([-1.0, 0.0]),
        PortfolioRiskLimits(
            max_drawdown=0.50,
            breach_action=configured_action,
        ),
    )
    record = next(
        item
        for item in result.breach_records
        if item.row_position == 0 and item.control_name == "drawdown"
    )
    assert result.policy_state.iloc[0] is RiskState.TERMINAL
    assert result.risk_schedule["risk_action_intent"].iloc[0] is RiskAction.NONE
    assert record.requested_action is RiskAction.NONE
    assert record.action_effective_position is None
    assert record.action_effective_label is None
    assert not record.action_executed


def test_breach_records_match_emitted_and_effective_action_series() -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(
            max_pair_equity_weight=0.54,
            breach_action=RiskAction.LIQUIDATE_ALL,
        ),
    )
    for record in result.breach_records:
        emitted = result.risk_schedule["risk_action_intent"].iloc[
            record.row_position
        ]
        assert record.requested_action is emitted
        if record.action_effective_position is None:
            assert record.requested_action is RiskAction.NONE
        else:
            assert (
                result.requested_actions.iloc[record.action_effective_position]
                is record.requested_action
            )


def test_liquidation_is_causal_required_intent_never_fake_execution() -> None:
    portfolio = _portfolio()
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(
            max_pair_equity_weight=0.54,
            breach_action=RiskAction.LIQUIDATE_ALL,
        ),
    )
    assert result.policy_state.iloc[1] is RiskState.NORMAL
    assert result.policy_state.iloc[2] is RiskState.LIQUIDATION_REQUIRED
    assert result.requested_actions.iloc[2] is RiskAction.LIQUIDATE_ALL
    assert all(not record.action_executed for record in result.breach_records)
    assert not result.risk_schedule["action_executed"].any()
    pd.testing.assert_series_equal(portfolio.portfolio_equity, _portfolio().portfolio_equity)


def test_later_compliance_does_not_erase_breach_or_sticky_state() -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(
            max_pair_equity_weight=0.54,
            breach_action=RiskAction.HALT_NEW_ENTRIES,
        ),
    )
    assert _status(result, "pair_concentration", 1) is RiskControlStatus.BREACH
    assert _status(result, "pair_concentration", 2) is RiskControlStatus.WITHIN_LIMIT
    assert any(record.row_position == 1 for record in result.breach_records)
    assert result.policy_state.iloc[2] is RiskState.ENTRY_HALTED


def test_portfolio_catastrophe_produces_sticky_terminal_state() -> None:
    pair = _pair_input("AAA|BBB", [0.0, -1.0, 2.0, 0.0])
    portfolio = run_multi_pair_portfolio({pair.pair_id: pair}, 1_000.0)
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(min_portfolio_equity_ratio=0.5),
    )
    assert result.policy_state.iloc[0] is RiskState.NORMAL
    assert result.policy_state.iloc[1] is RiskState.TERMINAL
    assert result.policy_state.iloc[2] is RiskState.TERMINAL
    assert result.policy_state.iloc[3] is RiskState.TERMINAL
    assert result.summary.terminal_state_reached
    assert _status(result, "minimum_equity", 1) is RiskControlStatus.BREACH
    assert _status(result, "minimum_equity", 2) is RiskControlStatus.UNEVALUABLE


def test_future_returns_exposures_and_missingness_cannot_change_prior_risk() -> None:
    portfolio = _portfolio()
    limits = PortfolioRiskLimits(
        max_gross_exposure_ratio=1.0,
        max_pair_equity_weight=0.54,
        max_drawdown=0.1,
        breach_action=RiskAction.HALT_NEW_ENTRIES,
    )
    original = run_portfolio_risk_controls(portfolio, limits)
    equity = portfolio.portfolio_equity.copy(deep=True)
    equity.iloc[-1] = np.nan
    aggregate = portfolio.aggregate_exposures.copy(deep=True)
    aggregate.iloc[-1, aggregate.columns.get_loc("gross_exposure_ratio")] = np.nan
    weights = portfolio.pair_current_equity_weights.copy(deep=True)
    weights.iloc[-1] = np.nan
    changed = replace(
        portfolio,
        portfolio_equity=equity,
        aggregate_exposures=aggregate,
        pair_current_equity_weights=weights,
    )
    modified = run_portfolio_risk_controls(changed, limits)
    pd.testing.assert_frame_equal(
        original.observed_metrics.iloc[:-1], modified.observed_metrics.iloc[:-1]
    )
    pd.testing.assert_frame_equal(
        original.control_statuses.iloc[:-1], modified.control_statuses.iloc[:-1]
    )
    pd.testing.assert_series_equal(
        original.policy_state.iloc[:-1], modified.policy_state.iloc[:-1]
    )
    assert tuple(record for record in original.breach_records if record.row_position < 3) == tuple(
        record for record in modified.breach_records if record.row_position < 3
    )


def test_timezone_aware_and_non_datetime_indices_are_preserved() -> None:
    dates = pd.date_range("2024-01-01", periods=4, tz="Europe/London", name="date")
    dated = run_portfolio_risk_controls(_portfolio(index=dates))
    assert dated.risk_schedule.index.equals(dates)
    assert dated.risk_schedule.index.tz == dates.tz
    labels = pd.Index(["z", "a", "m", "b"], name="observation")
    non_datetime = run_portfolio_risk_controls(_portfolio(index=labels))
    assert non_datetime.risk_schedule.index.equals(labels)


def test_duplicate_or_misaligned_portfolio_indices_are_rejected() -> None:
    portfolio = _portfolio()
    equity = portfolio.portfolio_equity.copy(deep=True)
    equity.index = pd.Index([0, 0, 2, 3])
    with pytest.raises(ValueError, match="unique"):
        run_portfolio_risk_controls(replace(portfolio, portfolio_equity=equity))
    returns = portfolio.portfolio_returns.copy(deep=True)
    returns.index = pd.RangeIndex(1, 5)
    with pytest.raises(ValueError, match="align exactly"):
        run_portfolio_risk_controls(replace(portfolio, portfolio_returns=returns))


@pytest.mark.parametrize(
    "name",
    [
        "max_gross_exposure_ratio",
        "max_abs_net_exposure_ratio",
        "max_pair_equity_weight",
        "max_symbol_gross_exposure_ratio",
        "max_long_exposure_ratio",
        "max_short_exposure_ratio",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, True, np.nan, np.inf, "1", pd.Series([1.0])])
def test_invalid_positive_ratio_limits_are_rejected(name: str, value: Any) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        PortfolioRiskLimits(**{name: value})


@pytest.mark.parametrize("name", ["max_drawdown", "min_portfolio_equity_ratio"])
@pytest.mark.parametrize("value", [0.0, -0.1, 1.01, True, np.nan, np.inf, "0.5"])
def test_invalid_unit_interval_limits_are_rejected(name: str, value: Any) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        PortfolioRiskLimits(**{name: value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", np.nan, np.inf])
def test_invalid_max_active_pairs_is_rejected(value: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="max_active_pairs"):
        PortfolioRiskLimits(max_active_pairs=value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "2", np.nan, np.inf])
def test_invalid_max_insolvent_sleeves_is_rejected(value: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="max_insolvent_sleeves"):
        PortfolioRiskLimits(max_insolvent_sleeves=value)


def test_limits_and_public_result_structures_are_frozen() -> None:
    limits = PortfolioRiskLimits(max_gross_exposure_ratio=2.0)
    result = run_portfolio_risk_controls(_portfolio(), limits)
    with pytest.raises(FrozenInstanceError):
        limits.max_gross_exposure_ratio = 3.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.breach_records = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.summary.observations = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.observations[0].row_position = 1  # type: ignore[misc]
    assert isinstance(result, PortfolioRiskResult)
    assert isinstance(result.summary, PortfolioRiskSummary)
    validate_portfolio_risk_limits(limits)


def test_breach_structure_is_frozen() -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(max_gross_exposure_ratio=1.0),
    )
    breach = result.breach_records[0]
    assert isinstance(breach, PortfolioRiskBreach)
    with pytest.raises(FrozenInstanceError):
        breach.observed_value = 0.0  # type: ignore[misc]


def test_result_pandas_objects_are_defensive_and_input_is_unchanged() -> None:
    portfolio = _portfolio()
    equity_before = portfolio.portfolio_equity.copy(deep=True)
    exposures_before = portfolio.aggregate_exposures.copy(deep=True)
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(max_gross_exposure_ratio=1.0),
    )
    observed_before = result.observed_metrics.copy(deep=True)
    statuses_before = result.control_statuses.copy(deep=True)
    portfolio.portfolio_equity.iloc[:] = 999.0
    portfolio.aggregate_exposures.loc[
        portfolio.aggregate_exposures.index[0], "gross_exposure_ratio"
    ] = np.nan
    pd.testing.assert_frame_equal(result.observed_metrics, observed_before)
    pd.testing.assert_frame_equal(result.control_statuses, statuses_before)
    # The risk layer did not mutate either input during evaluation.
    assert not equity_before.equals(portfolio.portfolio_equity)
    assert not exposures_before.equals(portfolio.aggregate_exposures)
    result.risk_schedule.loc[
        result.risk_schedule.index[0], "gross_exposure_ratio"
    ] = np.nan
    pd.testing.assert_frame_equal(result.observed_metrics, observed_before)
    pd.testing.assert_frame_equal(result.control_statuses, statuses_before)


def test_evaluation_does_not_mutate_portfolio_or_limits() -> None:
    portfolio = _portfolio()
    limits = PortfolioRiskLimits(max_gross_exposure_ratio=1.0)
    equity_before = portfolio.portfolio_equity.copy(deep=True)
    schedule_before = portfolio.portfolio_schedule.copy(deep=True)
    limits_before = replace(limits)
    run_portfolio_risk_controls(portfolio, limits)
    pd.testing.assert_series_equal(portfolio.portfolio_equity, equity_before)
    pd.testing.assert_frame_equal(portfolio.portfolio_schedule, schedule_before)
    assert limits == limits_before


def test_repeated_runs_and_pair_mapping_order_are_deterministic() -> None:
    limits = PortfolioRiskLimits(
        max_gross_exposure_ratio=1.0,
        max_pair_equity_weight=0.54,
        breach_action=RiskAction.HALT_NEW_ENTRIES,
    )
    first = run_portfolio_risk_controls(_portfolio(), limits)
    second = run_portfolio_risk_controls(_portfolio(), limits)
    reversed_order = run_portfolio_risk_controls(_portfolio(reverse=True), limits)
    for other in (second, reversed_order):
        pd.testing.assert_frame_equal(first.risk_schedule, other.risk_schedule)
        pd.testing.assert_frame_equal(first.control_statuses, other.control_statuses)
        assert first.breach_records == other.breach_records
        assert first.summary == other.summary


def test_provenance_is_conservative_and_synthetic_warning_survives() -> None:
    index = _index()
    pair = _pair_input("AAA|BBB", [0.0] * 4, index=index)
    walkforward = _walk_forward_no_selection(index)
    portfolio = run_multi_pair_portfolio(
        {pair.pair_id: pair, "CCC|DDD": walkforward},
        10_000.0,
    )
    result = run_portfolio_risk_controls(portfolio)
    assert result.contains_synthetic_reset_sources
    assert result.upstream_self_financing_interpretation == (
        portfolio.self_financing_interpretation
    )
    assert result.source_path_provenance == portfolio.source_path_provenance
    assert any("equal-capital-reset" in warning for warning in result.provenance_warnings)
    assert any("unnetted pair-sleeve" in warning for warning in result.warnings)
    assert result.upstream_portfolio_availability == portfolio.availability.value


def test_real_portfolio_integration_has_causal_drift_breach_without_rewrite() -> None:
    portfolio = _portfolio()
    equity_before = portfolio.portfolio_equity.copy(deep=True)
    result = run_portfolio_risk_controls(
        portfolio,
        PortfolioRiskLimits(
            max_pair_equity_weight=0.54,
            max_gross_exposure_ratio=1.0,
            max_drawdown=0.10,
            min_portfolio_equity_ratio=0.90,
            breach_action=RiskAction.HALT_NEW_ENTRIES,
        ),
    )
    assert result.observed_metrics["largest_pair_equity_weight"].iloc[1] > 0.5
    assert _status(result, "pair_concentration", 1) is RiskControlStatus.BREACH
    assert _status(result, "gross_exposure", 2) is RiskControlStatus.BREACH
    assert _status(result, "drawdown", 2) is RiskControlStatus.BREACH
    assert result.policy_state.iloc[1] is RiskState.NORMAL
    assert result.policy_state.iloc[2] is RiskState.ENTRY_HALTED
    pd.testing.assert_series_equal(portfolio.portfolio_equity, equity_before)
    assert not result.risk_schedule["action_executed"].any()


def test_direct_evaluation_and_policy_functions_preserve_contracts() -> None:
    portfolio = _portfolio()
    limits = PortfolioRiskLimits(
        max_gross_exposure_ratio=1.0,
        breach_action=RiskAction.HALT_NEW_ENTRIES,
    )
    observed, statuses, observations = evaluate_portfolio_risk(portfolio, limits)
    state, requested, intent, positions, labels = apply_portfolio_risk_policy(
        statuses,
        observed["terminal"],
        limits.breach_action,
    )
    assert len(observations) == len(portfolio.portfolio_equity)
    assert state.index.equals(portfolio.portfolio_equity.index)
    assert requested.index.equals(state.index)
    assert intent.index.equals(state.index)
    assert positions.index.equals(state.index)
    assert labels.index.equals(state.index)


def test_direct_policy_rejects_malformed_status_values() -> None:
    portfolio = _portfolio()
    observed, statuses, _ = evaluate_portfolio_risk(
        portfolio,
        PortfolioRiskLimits(max_gross_exposure_ratio=1.0),
    )
    malformed = statuses.copy(deep=True)
    malformed.iloc[0, 0] = "BREACH"
    with pytest.raises(TypeError, match="RiskControlStatus"):
        apply_portfolio_risk_policy(
            malformed,
            observed["terminal"],
            RiskAction.NONE,
        )


@pytest.mark.parametrize("consumer", ("schedule", "summary"))
def test_direct_status_consumers_accept_valid_enum_frames(consumer: str) -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(max_gross_exposure_ratio=1.0),
    )
    output = _invoke_direct_status_consumer(
        consumer,
        result,
        result.control_statuses.copy(deep=True),
    )
    if consumer == "schedule":
        assert isinstance(output, pd.DataFrame)
        assert output.index.equals(result.observed_metrics.index)
    else:
        assert isinstance(output, PortfolioRiskSummary)
        assert output.observations == len(result.observed_metrics)


@pytest.mark.parametrize("consumer", ("schedule", "summary"))
@pytest.mark.parametrize(
    ("invalid_status", "error_type", "message"),
    (
        pytest.param(
            "BREACH",
            TypeError,
            "RiskControlStatus",
            id="arbitrary-string",
        ),
        pytest.param(np.nan, ValueError, "missing values", id="missing-status"),
        pytest.param(
            object(),
            TypeError,
            "RiskControlStatus",
            id="unknown-object",
        ),
    ),
)
def test_direct_status_consumers_reject_malformed_values(
    consumer: str,
    invalid_status: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(max_gross_exposure_ratio=1.0),
    )
    malformed = result.control_statuses.copy(deep=True)
    malformed.iloc[0, 0] = invalid_status
    with pytest.raises(error_type, match=message):
        _invoke_direct_status_consumer(consumer, result, malformed)


@pytest.mark.parametrize("consumer", ("schedule", "summary"))
@pytest.mark.parametrize("malformation", ("missing-column", "misaligned-row"))
def test_direct_status_consumers_reject_incomplete_or_misaligned_frames(
    consumer: str,
    malformation: str,
) -> None:
    result = run_portfolio_risk_controls(
        _portfolio(),
        PortfolioRiskLimits(max_gross_exposure_ratio=1.0),
    )
    if malformation == "missing-column":
        malformed = result.control_statuses.drop(columns="gross_exposure")
        message = "columns"
    else:
        malformed = result.control_statuses.iloc[:-1].copy(deep=True)
        message = "align exactly"
    with pytest.raises(ValueError, match=message):
        _invoke_direct_status_consumer(consumer, result, malformed)


def test_malformed_synthetic_provenance_boolean_is_rejected() -> None:
    with pytest.raises(TypeError, match="contains_synthetic_reset_sources"):
        run_portfolio_risk_controls(
            replace(_portfolio(), contains_synthetic_reset_sources="False"),
        )


def test_no_optimizer_selection_or_execution_api_exists() -> None:
    forbidden = (
        "optimize",
        "optimise",
        "kelly",
        "risk_parity",
        "volatility_target",
        "best",
        "winner",
        "rank",
        "reallocate",
        "rebalance",
        "execute_liquidation",
        "order_route",
    )
    public_names = tuple(name.lower() for name in risk_module.__all__)
    assert all(
        token not in public_name
        for public_name in public_names
        for token in forbidden
    )
