"""Focused tests for Milestone 9A static multi-pair portfolio construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pairs_trading.portfolio as portfolio_module
from pairs_trading.backtest import BacktestResult, run_pair_backtest
from pairs_trading.portfolio import (
    AllocationMethod,
    PairAllocation,
    PairPortfolioInput,
    PortfolioAllocationPolicy,
    PortfolioAvailability,
    PortfolioResult,
    allocate_pair_capital,
    calculate_portfolio_returns,
    normalize_pair_weights,
    run_multi_pair_portfolio,
    validate_pair_results,
)
from pairs_trading.walkforward import (
    WalkForwardAnalyticsStatus,
    WalkForwardResult,
)


def _index(rows: int = 4) -> pd.RangeIndex:
    return pd.RangeIndex(rows, name="row")


def _pair_input(
    pair_id: str = "AAA|BBB",
    returns: list[float] | None = None,
    *,
    index: pd.Index | None = None,
    active: list[bool | None] | None = None,
    execution: list[bool] | None = None,
    market_value_y: list[float] | None = None,
    market_value_x: list[float] | None = None,
    source_capital_basis: float | None = 1_000.0,
    point_in_time_validated: bool = False,
    capital_policy: str = "standalone_pair_capital",
) -> PairPortfolioInput:
    symbol_y, symbol_x = pair_id.split("|")
    values = [0.0, 0.01, -0.005, 0.0] if returns is None else returns
    row_index = _index(len(values)) if index is None else index
    active_values = (
        [False, True, True, False]
        if active is None and len(values) == 4
        else ([False] * len(values) if active is None else active)
    )
    execution_values = (
        [False, True, False, True]
        if execution is None and len(values) == 4
        else ([False] * len(values) if execution is None else execution)
    )
    values_y = (
        [0.0, 600.0, 660.0, 0.0]
        if market_value_y is None and len(values) == 4
        else ([0.0] * len(values) if market_value_y is None else market_value_y)
    )
    values_x = (
        [0.0, -400.0, -400.0, 0.0]
        if market_value_x is None and len(values) == 4
        else ([0.0] * len(values) if market_value_x is None else market_value_x)
    )
    market_y = np.asarray(values_y, dtype=float)
    market_x = np.asarray(values_x, dtype=float)
    exposure = pd.DataFrame(
        {
            "market_value_y": market_y,
            "market_value_x": market_x,
            "gross_exposure": np.abs(market_y) + np.abs(market_x),
            "long_exposure": np.maximum(market_y, 0.0) + np.maximum(market_x, 0.0),
            "short_exposure": np.maximum(-market_y, 0.0) + np.maximum(-market_x, 0.0),
            "net_exposure": market_y + market_x,
        },
        index=row_index,
    )
    return PairPortfolioInput(
        pair_id=pair_id,
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        calendar_returns=pd.Series(values, index=row_index, name="pair_return"),
        active_state=pd.Series(active_values, index=row_index, dtype="boolean"),
        execution_rows=pd.Series(execution_values, index=row_index, dtype=bool),
        source_exposure=exposure,
        source_capital_basis=source_capital_basis,
        source_type="fixture",
        capital_policy=capital_policy,
        point_in_time_universe_validated=point_in_time_validated,
        universe_provenance=(
            "point-in-time-fixture" if point_in_time_validated else "caller-supplied"
        ),
        cleaning_provenance="fixture-cleaning-unvalidated",
        provenance_warnings=("fixture provenance limitation",),
    )


def _two_pairs(**second_overrides: Any) -> dict[str, PairPortfolioInput]:
    first = _pair_input()
    second_values: dict[str, Any] = {
        "pair_id": "CCC|DDD",
        "returns": [0.0, -0.02, 0.01, 0.005],
        "market_value_y": [0.0, 200.0, 220.0, 0.0],
        "market_value_x": [0.0, -800.0, -800.0, 0.0],
    }
    second_values.update(second_overrides)
    second = _pair_input(**second_values)
    return {first.pair_id: first, second.pair_id: second}


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
        provenance_warnings=("No-selection fixture provenance.",),
        evaluated_start_position=0,
        evaluated_end_position=len(index) - 1,
        evaluated_start_label=index[0],
        evaluated_end_label=index[-1],
        discarded_terminal_rows=0,
    )


def _real_backtest(
    price_y_values: list[float],
    price_x_values: list[float],
    zscores: list[float],
    *,
    initial_capital: float,
    target_notional: float,
) -> BacktestResult:
    index = pd.date_range("2024-01-02", periods=len(zscores), freq="D", tz="UTC")
    return run_pair_backtest(
        pd.Series(price_y_values, index=index, name="Y"),
        pd.Series(price_x_values, index=index, name="X"),
        1.0,
        target_notional,
        zscore=pd.Series(zscores, index=index, name="zscore"),
        initial_capital=initial_capital,
        entry_z=1.0,
        exit_z=0.25,
        stop_z=3.0,
        execution_lag=1,
        commission_bps=2.0,
        slippage_bps=1.0,
        force_liquidation=True,
    )


def test_equal_weight_allocation_and_concentration_are_exact() -> None:
    result = run_multi_pair_portfolio(_two_pairs(), 10_000.0)
    weights = {item.pair_id: item.weight for item in result.pair_allocations}
    assert weights == {"AAA|BBB": 0.5, "CCC|DDD": 0.5}
    assert all(item.allocated_capital == 5_000.0 for item in result.pair_allocations)
    assert result.cash_weight == 0.0
    assert result.metrics.largest_pair_weight == 0.5
    assert result.metrics.allocation_hhi == pytest.approx(0.5)
    assert result.metrics.effective_number_of_pairs == pytest.approx(2.0)


def test_fixed_weights_are_preserved_and_create_cash_sleeve() -> None:
    weights = {"CCC|DDD": 0.20, "AAA|BBB": 0.60}
    before = dict(weights)
    result = run_multi_pair_portfolio(
        _two_pairs(),
        10_000.0,
        allocation_method=AllocationMethod.FIXED_WEIGHT,
        fixed_weights=weights,
    )
    assert result.allocation_policy.pair_weights == (
        ("AAA|BBB", 0.60),
        ("CCC|DDD", 0.20),
    )
    assert result.cash_weight == pytest.approx(0.20)
    assert result.cash_capital == pytest.approx(2_000.0)
    assert result.metrics.allocation_hhi == pytest.approx(0.40)
    assert result.metrics.effective_number_of_pairs == pytest.approx(2.5)
    assert weights == before


def test_explicit_weight_normalization_is_opt_in() -> None:
    normalized = normalize_pair_weights(
        {"AAA|BBB": 2.0, "CCC|DDD": 1.0},
        ["CCC|DDD", "AAA|BBB"],
    )
    assert tuple(pair_id for pair_id, _ in normalized) == ("AAA|BBB", "CCC|DDD")
    assert tuple(weight for _, weight in normalized) == pytest.approx(
        (2.0 / 3.0, 1.0 / 3.0)
    )
    result = run_multi_pair_portfolio(
        _two_pairs(),
        10_000.0,
        allocation_method="FIXED_WEIGHT",
        fixed_weights={"AAA|BBB": 2.0, "CCC|DDD": 1.0},
        normalize_weights=True,
    )
    assert result.cash_weight == pytest.approx(0.0)
    assert result.allocation_policy.weights_normalized


@pytest.mark.parametrize(
    "weights",
    [
        {"AAA|BBB": 0.8, "CCC|DDD": 0.3},
        {"AAA|BBB": -0.1, "CCC|DDD": 0.5},
        {"AAA|BBB": True, "CCC|DDD": 0.5},
        {"AAA|BBB": np.inf, "CCC|DDD": 0.5},
        {"AAA|BBB": np.nan, "CCC|DDD": 0.5},
    ],
)
def test_invalid_fixed_weights_are_rejected(weights: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        run_multi_pair_portfolio(
            _two_pairs(),
            10_000.0,
            allocation_method="FIXED_WEIGHT",
            fixed_weights=weights,
        )


@pytest.mark.parametrize("capital", [0.0, -1.0, True, np.nan, np.inf, "1000"])
def test_initial_capital_validation_is_strict(capital: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="portfolio_initial_capital"):
        run_multi_pair_portfolio(_two_pairs(), capital)


def test_pair_order_does_not_change_allocation_or_output() -> None:
    pairs = _two_pairs()
    first = run_multi_pair_portfolio(pairs, 10_000.0)
    second = run_multi_pair_portfolio(
        list(reversed(tuple(pairs.items()))),
        10_000.0,
    )
    assert first.pair_ids == second.pair_ids == ("AAA|BBB", "CCC|DDD")
    assert first.pair_allocations == second.pair_allocations
    pd.testing.assert_series_equal(first.portfolio_returns, second.portfolio_returns)
    pd.testing.assert_frame_equal(
        first.pair_return_contributions,
        second.pair_return_contributions,
    )


def test_portfolio_return_contribution_and_cash_identities_are_exact() -> None:
    result = run_multi_pair_portfolio(
        _two_pairs(),
        10_000.0,
        allocation_method="FIXED_WEIGHT",
        fixed_weights={"AAA|BBB": 0.6, "CCC|DDD": 0.2},
    )
    expected = (
        0.6 * result.pair_sleeve_returns["AAA|BBB"]
        + 0.2 * result.pair_sleeve_returns["CCC|DDD"]
    )
    pd.testing.assert_series_equal(
        result.portfolio_returns,
        expected.rename("portfolio_return"),
    )
    reconciled = result.pair_return_contributions.sum(axis=1) + result.portfolio_schedule[
        "cash_return_contribution"
    ]
    np.testing.assert_allclose(reconciled, result.portfolio_returns)
    assert result.portfolio_schedule["cash_return_contribution"].eq(0.0).all()


def test_portfolio_equity_uses_geometric_compounding() -> None:
    result = run_multi_pair_portfolio(_two_pairs(), 10_000.0)
    expected = 10_000.0 * (1.0 + result.portfolio_returns).cumprod()
    pd.testing.assert_series_equal(
        result.portfolio_equity,
        expected.rename("portfolio_equity"),
    )


def test_raw_pair_dollar_pnl_is_not_used_for_portfolio_return() -> None:
    pair = _pair_input(returns=[0.0, 0.01, 0.0, 0.0])
    changed_exposure = replace(
        pair,
        source_exposure=pair.source_exposure * 1_000_000.0,
    )
    original = run_multi_pair_portfolio({pair.pair_id: pair}, 10_000.0)
    changed = run_multi_pair_portfolio(
        {changed_exposure.pair_id: changed_exposure},
        10_000.0,
    )
    pd.testing.assert_series_equal(original.portfolio_returns, changed.portfolio_returns)


def test_static_weights_do_not_change_after_winners_losers_or_flat_sleeves() -> None:
    pairs = _two_pairs(
        returns=[0.0, 0.0, 0.0, 0.0],
        active=[False, False, False, False],
    )
    result = run_multi_pair_portfolio(pairs, 10_000.0)
    assert result.allocation_policy.pair_weights == (
        ("AAA|BBB", 0.5),
        ("CCC|DDD", 0.5),
    )
    expected = 0.5 * pairs["AAA|BBB"].calendar_returns
    pd.testing.assert_series_equal(
        result.portfolio_returns,
        expected.rename("portfolio_return"),
    )
    assert result.portfolio_schedule.loc[1, "active_pair_count"] == 1


def test_walkforward_no_selection_sleeve_remains_zero_return_cash() -> None:
    index = _index()
    active_pair = _pair_input(index=index)
    no_selection = _walk_forward_no_selection(index)
    result = run_multi_pair_portfolio(
        {"AAA|BBB": active_pair, "CCC|DDD": no_selection},
        10_000.0,
    )
    expected = 0.5 * active_pair.calendar_returns
    pd.testing.assert_series_equal(
        result.portfolio_returns,
        expected.rename("portfolio_return"),
    )
    assert dict(result.allocation_policy.pair_weights)["CCC|DDD"] == 0.5
    assert "equal_capital_reset" in result.pair_capital_policies


def test_missing_positive_weight_return_blocks_row_without_renormalization() -> None:
    pairs = _two_pairs(returns=[0.0, -0.02, np.nan, 0.005])
    result = run_multi_pair_portfolio(pairs, 10_000.0)
    assert np.isnan(result.portfolio_returns.iloc[2])
    assert result.unavailable_rows.iloc[2]["unavailable_pair_ids"] == ("CCC|DDD",)
    assert result.pair_return_contributions["AAA|BBB"].iloc[2] == pytest.approx(-0.0025)
    assert np.isnan(result.pair_return_contributions["CCC|DDD"].iloc[2])
    assert result.availability is PortfolioAvailability.PARTIALLY_AVAILABLE
    assert result.allocation_policy.pair_weights == (
        ("AAA|BBB", 0.5),
        ("CCC|DDD", 0.5),
    )


def test_missing_zero_weight_pair_does_not_block_portfolio_return() -> None:
    pairs = _two_pairs(returns=[np.nan] * 4)
    result = run_multi_pair_portfolio(
        pairs,
        10_000.0,
        allocation_method="FIXED_WEIGHT",
        fixed_weights={"AAA|BBB": 1.0, "CCC|DDD": 0.0},
    )
    pd.testing.assert_series_equal(
        result.portfolio_returns,
        pairs["AAA|BBB"].calendar_returns.rename("portfolio_return"),
    )


def test_catastrophic_pair_return_is_preserved_and_weighted() -> None:
    pairs = _two_pairs(returns=[0.0, -1.0, 0.0, 0.0])
    diversified = run_multi_pair_portfolio(pairs, 10_000.0)
    assert diversified.pair_sleeve_returns["CCC|DDD"].iloc[1] == -1.0
    assert diversified.pair_return_contributions["CCC|DDD"].iloc[1] == -0.5
    assert diversified.portfolio_returns.iloc[1] > -1.0

    catastrophic = run_multi_pair_portfolio(
        pairs,
        10_000.0,
        allocation_method="FIXED_WEIGHT",
        fixed_weights={"AAA|BBB": 0.0, "CCC|DDD": 1.0},
    )
    assert catastrophic.portfolio_returns.iloc[1] == -1.0
    assert catastrophic.portfolio_equity.iloc[1] == 0.0
    assert catastrophic.availability is PortfolioAvailability.UNAVAILABLE
    assert catastrophic.metrics.catastrophic_return_row_count == 1


def test_misaligned_duplicate_and_nonmonotonic_datetime_indices_are_rejected() -> None:
    pairs = _two_pairs()
    misaligned = replace(
        pairs["CCC|DDD"],
        calendar_returns=pairs["CCC|DDD"].calendar_returns.set_axis(pd.RangeIndex(1, 5)),
        active_state=pairs["CCC|DDD"].active_state.set_axis(pd.RangeIndex(1, 5)),
        execution_rows=pairs["CCC|DDD"].execution_rows.set_axis(pd.RangeIndex(1, 5)),
        source_exposure=pairs["CCC|DDD"].source_exposure.set_axis(pd.RangeIndex(1, 5)),
    )
    with pytest.raises(ValueError, match="exact declared portfolio index"):
        run_multi_pair_portfolio({"AAA|BBB": pairs["AAA|BBB"], "CCC|DDD": misaligned}, 10_000.0)

    duplicate_index = pd.Index([0, 0, 1, 2])
    duplicate = _pair_input(index=duplicate_index)
    with pytest.raises(ValueError, match="unique"):
        run_multi_pair_portfolio({"AAA|BBB": duplicate}, 10_000.0)

    dates = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"])
    shuffled = _pair_input(index=dates)
    with pytest.raises(ValueError, match="monotonically increasing"):
        run_multi_pair_portfolio({"AAA|BBB": shuffled}, 10_000.0)


def test_timezone_aware_index_is_preserved_exactly() -> None:
    index = pd.date_range("2024-01-01", periods=4, tz="Europe/London", name="date")
    pair = _pair_input(index=index)
    result = run_multi_pair_portfolio({"AAA|BBB": pair}, 10_000.0)
    assert result.portfolio_returns.index.equals(index)
    assert result.portfolio_returns.index.tz == index.tz
    assert result.portfolio_returns.index.name == "date"


def test_non_datetime_index_preserves_declared_order() -> None:
    index = pd.Index(["z", "a", "m", "b"], name="observation")
    pair = _pair_input(index=index)
    result = run_multi_pair_portfolio({"AAA|BBB": pair}, 10_000.0)
    assert result.portfolio_returns.index.equals(index)


def test_source_capital_basis_scales_pair_and_aggregate_exposure() -> None:
    result = run_multi_pair_portfolio(_two_pairs(), 10_000.0)
    first = next(item for item in result.pair_allocations if item.pair_id == "AAA|BBB")
    second = next(item for item in result.pair_allocations if item.pair_id == "CCC|DDD")
    assert first.exposure_scaling_factor == pytest.approx(5.0)
    assert second.exposure_scaling_factor == pytest.approx(5.0)
    row = 1
    assert result.pair_exposures.loc[row, ("AAA|BBB", "gross_exposure")] == pytest.approx(5_000.0)
    assert result.pair_exposures.loc[row, ("CCC|DDD", "gross_exposure")] == pytest.approx(5_000.0)
    assert result.aggregate_exposures.loc[row, "total_gross_exposure"] == pytest.approx(10_000.0)
    assert result.aggregate_exposures.loc[row, "total_long_exposure"] == pytest.approx(4_000.0)
    assert result.aggregate_exposures.loc[row, "total_short_exposure"] == pytest.approx(6_000.0)
    assert result.aggregate_exposures.loc[row, "total_net_exposure"] == pytest.approx(-2_000.0)


def test_unknown_capital_basis_blocks_exposure_but_not_returns() -> None:
    pairs = _two_pairs(source_capital_basis=None)
    result = run_multi_pair_portfolio(pairs, 10_000.0)
    assert result.portfolio_returns.notna().all()
    assert result.aggregate_exposures["total_gross_exposure"].isna().all()
    assert result.metrics.exposure_unavailable_pair_count == 1
    assert result.metrics.exposure_unavailable_row_count == len(result.portfolio_returns)
    assert result.availability is PortfolioAvailability.PARTIALLY_AVAILABLE


def test_missing_exposure_row_is_transparent_without_corrupting_returns() -> None:
    pair = _pair_input()
    exposure = pair.source_exposure.copy(deep=True)
    exposure.iloc[2] = np.nan
    changed = replace(pair, source_exposure=exposure)
    result = run_multi_pair_portfolio({"AAA|BBB": changed}, 1_000.0)
    assert result.portfolio_returns.notna().all()
    assert not bool(result.aggregate_exposures.loc[2, "exposure_available"])
    assert result.metrics.exposure_unavailable_row_count == 1
    assert result.availability is PortfolioAvailability.PARTIALLY_AVAILABLE


def test_symbol_overlap_is_aggregated_without_cross_pair_gross_netting() -> None:
    first = _pair_input()
    second = _pair_input(
        pair_id="AAA|CCC",
        returns=[0.0, 0.0, 0.0, 0.0],
        market_value_y=[0.0, 200.0, 220.0, 0.0],
        market_value_x=[0.0, -800.0, -800.0, 0.0],
    )
    result = run_multi_pair_portfolio(
        {"AAA|BBB": first, "AAA|CCC": second},
        10_000.0,
    )
    assert result.metrics.shared_symbols == ("AAA",)
    assert result.symbol_exposures.loc[1, ("AAA", "net_market_value")] == pytest.approx(4_000.0)
    assert result.symbol_exposures.loc[1, ("AAA", "gross_market_value")] == pytest.approx(4_000.0)
    assert result.symbol_exposures.loc[1, ("AAA", "gross_exposure_fraction")] == pytest.approx(0.4)
    assert result.metrics.largest_symbol_gross_exposure_fraction == pytest.approx(
        result.aggregate_exposures["largest_symbol_gross_exposure_fraction"].max()
    )
    assert any("Shared symbols" in warning for warning in result.warnings)


def test_exposure_ratio_uses_current_portfolio_equity() -> None:
    pair = _pair_input(returns=[0.0, 0.10, 0.0, 0.0])
    result = run_multi_pair_portfolio({"AAA|BBB": pair}, 1_000.0)
    assert result.portfolio_equity.iloc[1] == pytest.approx(1_100.0)
    assert result.aggregate_exposures.loc[1, "gross_exposure_ratio"] == pytest.approx(1_000.0 / 1_100.0)


def test_execution_row_leverage_breach_is_rejected() -> None:
    pair = _pair_input(returns=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="initial/execution row"):
        run_multi_pair_portfolio(
            {"AAA|BBB": pair},
            1_000.0,
            max_total_gross_exposure_ratio=0.9,
        )


def test_natural_exposure_drift_breach_is_reported_without_rebalancing() -> None:
    pair = _pair_input(
        returns=[0.0, 0.0, 0.0, 0.0],
        execution=[False, True, False, True],
        market_value_y=[0.0, 600.0, 1_100.0, 0.0],
        market_value_x=[0.0, -400.0, -400.0, 0.0],
    )
    result = run_multi_pair_portfolio(
        {"AAA|BBB": pair},
        1_000.0,
        max_total_gross_exposure_ratio=1.2,
    )
    assert not result.portfolio_schedule.loc[1, "gross_exposure_limit_breach"]
    assert result.portfolio_schedule.loc[2, "gross_exposure_limit_breach"]
    assert not result.portfolio_schedule.loc[2, "execution_row"]
    assert result.metrics.gross_exposure_limit_breach_count == 1
    assert result.pair_allocations[0].weight == 1.0


@pytest.mark.parametrize("limit", [0.0, -1.0, True, np.nan, np.inf, "2"])
def test_invalid_gross_exposure_limits_are_rejected(limit: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="max_total_gross_exposure_ratio"):
        run_multi_pair_portfolio(
            {"AAA|BBB": _pair_input()},
            1_000.0,
            max_total_gross_exposure_ratio=limit,
        )


def test_future_changes_do_not_alter_prior_rows_or_static_weights() -> None:
    pairs = _two_pairs()
    original = run_multi_pair_portfolio(pairs, 10_000.0)
    changed_returns = pairs["CCC|DDD"].calendar_returns.copy(deep=True)
    changed_returns.iloc[3] = -0.90
    changed = replace(pairs["CCC|DDD"], calendar_returns=changed_returns)
    modified = run_multi_pair_portfolio(
        {"AAA|BBB": pairs["AAA|BBB"], "CCC|DDD": changed},
        10_000.0,
    )
    pd.testing.assert_series_equal(
        original.portfolio_returns.iloc[:3],
        modified.portfolio_returns.iloc[:3],
    )
    assert original.allocation_policy == modified.allocation_policy


def test_missing_future_row_does_not_change_prior_allocations_or_returns() -> None:
    pairs = _two_pairs()
    original = run_multi_pair_portfolio(pairs, 10_000.0)
    changed_returns = pairs["CCC|DDD"].calendar_returns.copy(deep=True)
    changed_returns.iloc[-1] = np.nan
    changed = replace(pairs["CCC|DDD"], calendar_returns=changed_returns)
    modified = run_multi_pair_portfolio(
        {"AAA|BBB": pairs["AAA|BBB"], "CCC|DDD": changed},
        10_000.0,
    )
    pd.testing.assert_series_equal(
        original.portfolio_returns.iloc[:-1],
        modified.portfolio_returns.iloc[:-1],
    )
    assert original.pair_allocations == modified.pair_allocations


def test_pair_inputs_weights_and_declared_index_are_not_mutated() -> None:
    pairs = _two_pairs()
    input_snapshots = {
        pair_id: (
            source.calendar_returns.copy(deep=True),
            source.active_state.copy(deep=True),
            source.source_exposure.copy(deep=True),
        )
        for pair_id, source in pairs.items()
    }
    weights = {"AAA|BBB": 0.6, "CCC|DDD": 0.2}
    weights_before = dict(weights)
    index = pairs["AAA|BBB"].calendar_returns.index.copy()
    index_before = index.copy()
    run_multi_pair_portfolio(
        pairs,
        10_000.0,
        allocation_method="FIXED_WEIGHT",
        fixed_weights=weights,
        portfolio_index=index,
    )
    for pair_id, source in pairs.items():
        returns, active, exposure = input_snapshots[pair_id]
        pd.testing.assert_series_equal(source.calendar_returns, returns)
        pd.testing.assert_series_equal(source.active_state, active)
        pd.testing.assert_frame_equal(source.source_exposure, exposure)
    assert weights == weights_before
    assert index.equals(index_before)


def test_result_pandas_objects_are_defensive_copies() -> None:
    pair = _pair_input()
    result = run_multi_pair_portfolio({"AAA|BBB": pair}, 10_000.0)
    returns_before = result.portfolio_returns.copy(deep=True)
    contribution_before = result.pair_return_contributions.copy(deep=True)
    pair.calendar_returns.iloc[:] = 99.0
    pair.source_exposure.iloc[:] = 99.0
    pd.testing.assert_series_equal(result.portfolio_returns, returns_before)
    pd.testing.assert_frame_equal(
        result.pair_return_contributions,
        contribution_before,
    )


def test_repeated_portfolio_runs_are_deterministic() -> None:
    first = run_multi_pair_portfolio(_two_pairs(), 10_000.0)
    second = run_multi_pair_portfolio(_two_pairs(), 10_000.0)
    assert first.pair_allocations == second.pair_allocations
    assert first.metrics == second.metrics
    pd.testing.assert_frame_equal(first.portfolio_schedule, second.portfolio_schedule)
    pd.testing.assert_frame_equal(first.symbol_exposures, second.symbol_exposures)


def test_provenance_summary_remains_conservative() -> None:
    pairs = _two_pairs(point_in_time_validated=True)
    result = run_multi_pair_portfolio(pairs, 10_000.0)
    assert not result.all_pair_universes_point_in_time_validated
    assert not result.pair_indices_differ
    assert "standalone_pair_capital" in result.pair_capital_policies
    assert any("caller-supplied" in warning for warning in result.provenance_warnings)


def test_public_result_structures_are_frozen() -> None:
    result = run_multi_pair_portfolio(_two_pairs(), 10_000.0)
    allocation = result.pair_allocations[0]
    policy = result.allocation_policy
    with pytest.raises(FrozenInstanceError):
        allocation.weight = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.cash_weight = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.initial_capital = 0.0  # type: ignore[misc]
    assert isinstance(allocation, PairAllocation)
    assert isinstance(policy, PortfolioAllocationPolicy)
    assert isinstance(result, PortfolioResult)


def test_real_backtest_outputs_integrate_returns_costs_and_scaled_exposures() -> None:
    first = _real_backtest(
        [100.0, 101.0, 103.0, 102.0, 101.0],
        [100.0, 100.0, 100.0, 100.0, 100.0],
        [-1.5, -1.2, -0.8, 0.0, 0.0],
        initial_capital=10_000.0,
        target_notional=2_000.0,
    )
    second = _real_backtest(
        [50.0, 50.5, 50.5, 50.0, 50.0],
        [75.0, 75.0, 75.0, 75.0, 75.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        initial_capital=20_000.0,
        target_notional=4_000.0,
    )
    pair_results = {"AAA|BBB": first, "AAA|CCC": second}
    first_run = run_multi_pair_portfolio(
        pair_results,
        30_000.0,
        source_capital_bases={"AAA|BBB": 10_000.0, "AAA|CCC": 20_000.0},
    )
    second_run = run_multi_pair_portfolio(
        pair_results,
        30_000.0,
        source_capital_bases={"AAA|BBB": 10_000.0, "AAA|CCC": 20_000.0},
    )
    expected = 0.5 * first.accounting["net_return_after_carry"] + 0.5 * second.accounting[
        "net_return_after_carry"
    ]
    pd.testing.assert_series_equal(
        first_run.portfolio_returns,
        expected.rename("portfolio_return"),
    )
    assert first_run.pair_allocations[0].exposure_scaling_factor == pytest.approx(1.5)
    assert first_run.pair_allocations[1].exposure_scaling_factor == pytest.approx(0.75)
    assert first_run.portfolio_schedule["active_pair_count"].max() == 1
    assert dict(first_run.allocation_policy.pair_weights)["AAA|CCC"] == 0.5
    assert first.accounting["transaction_cost"].sum() > 0.0
    pd.testing.assert_frame_equal(
        first_run.portfolio_schedule,
        second_run.portfolio_schedule,
    )


def test_no_optimizer_or_strategy_promotion_api_exists() -> None:
    forbidden = (
        "optimize",
        "optimise",
        "markowitz",
        "kelly",
        "risk_parity",
        "best_pair",
        "winner",
        "select_best",
        "approved",
        "production_ready",
        "deploy",
    )
    public_names = tuple(name.lower() for name in portfolio_module.__all__)
    assert all(
        token not in public_name
        for public_name in public_names
        for token in forbidden
    )
