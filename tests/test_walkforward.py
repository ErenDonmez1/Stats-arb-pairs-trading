"""Focused tests for causal walk-forward out-of-sample evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.analytics import calculate_core_metrics
from pairs_trading.data import OBSERVED_PRICE_MASK_ATTR, make_synthetic_universe
from pairs_trading.screening import PairScreeningResult
import pairs_trading.walkforward as walkforward_module
from pairs_trading.walkforward import (
    WalkForwardAnalyticsStatus,
    WalkForwardFold,
    WalkForwardFoldResult,
    WalkForwardResult,
    WalkForwardStatus,
    generate_walk_forward_folds,
    run_walk_forward_analysis,
    run_walk_forward_fold,
)


def _prices(rows: int = 84, *, datetime_index: bool = False) -> pd.DataFrame:
    """Return deterministic positive prices with repeatable OOS spread shocks."""
    positions = np.arange(rows, dtype=float)
    price_x = 100.0 * np.exp(0.0005 * positions)
    base_pattern = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    spread = np.resize(base_pattern, rows).astype(float)
    for start in range(60, rows, 8):
        spread[start : min(start + 3, rows)] = [-0.08, -0.07, 0.0][
            : max(0, min(3, rows - start))
        ]
    price_y = price_x * np.exp(spread)
    comparison = 70.0 * np.exp(0.0003 * positions + 0.002 * np.sin(positions))
    index: pd.Index
    if datetime_index:
        index = pd.date_range("2020-01-01", periods=rows, freq="D", name="date")
    else:
        index = pd.RangeIndex(rows, name="row")
    return pd.DataFrame(
        {"AAA": price_y, "BBB": price_x, "CCC": comparison},
        index=index,
    )


def _screening_result(
    *,
    symbol_y: str = "AAA",
    symbol_x: str = "BBB",
    selected: bool = True,
    rank: int | None = 1,
    alpha: float | None = 0.0,
    beta: float | None = 1.0,
    corrected_pvalue: float | None = 0.01,
) -> PairScreeningResult:
    return PairScreeningResult(
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        group="sector",
        observations=60,
        alpha=alpha,
        beta=beta,
        spread_standard_deviation=0.01,
        cointegration_statistic=-4.0,
        cointegration_pvalue=0.005,
        corrected_pvalue=corrected_pvalue,
        cointegration_critical_values={"1%": -3.9, "5%": -3.3, "10%": -3.0},
        adf_statistic=-4.2,
        adf_pvalue=0.004,
        half_life=4.0,
        hurst=0.3,
        selected=selected,
        rank=rank if selected else None,
        rejection_reasons=() if selected else ("not_selected",),
    )


def _install_screening(
    monkeypatch: pytest.MonkeyPatch,
    results: tuple[PairScreeningResult, ...] | None = None,
) -> list[pd.DataFrame]:
    calls: list[pd.DataFrame] = []
    returned = results or (_screening_result(),)

    def fake_screen_pairs(
        prices: pd.DataFrame,
        groups: Any = None,
        **kwargs: Any,
    ) -> tuple[PairScreeningResult, ...]:
        calls.append(prices.copy(deep=True))
        return returned

    monkeypatch.setattr(walkforward_module, "screen_pairs", fake_screen_pairs)
    return calls


def _fold(prices: pd.DataFrame, fold_number: int = 0) -> WalkForwardFold:
    return generate_walk_forward_folds(prices.index, 60, 8)[fold_number]


def _run_fold(
    prices: pd.DataFrame,
    fold: WalkForwardFold,
    **overrides: Any,
) -> WalkForwardFoldResult:
    parameters: dict[str, Any] = {
        "screening_min_observations": 50,
        "zscore_lookback": 5,
        "entry_z": 1.0,
        "exit_z": 0.25,
        "stop_z": 50.0,
        "target_gross_notional": 10_000.0,
        "initial_capital": 100_000.0,
        "execution_lag": 1,
        "commission_bps": 5.0,
        "fixed_commission_per_leg": 0.25,
        "slippage_bps": 3.0,
        "borrow_rate_y": 0.05,
        "borrow_rate_x": 0.05,
        "financing_rate": 0.03,
        "periods_per_year": 252,
        "force_liquidation": True,
    }
    parameters.update(overrides)
    return run_walk_forward_fold(prices, fold, **parameters)


def _run_analysis(
    prices: pd.DataFrame,
    *,
    formation_window: int = 60,
    trading_window: int = 8,
    **overrides: Any,
) -> WalkForwardResult:
    parameters: dict[str, Any] = {
        "screening_min_observations": 50,
        "zscore_lookback": 5,
        "entry_z": 1.0,
        "exit_z": 0.25,
        "stop_z": 50.0,
        "target_gross_notional": 10_000.0,
        "initial_capital": 100_000.0,
        "execution_lag": 1,
        "commission_bps": 5.0,
        "fixed_commission_per_leg": 0.25,
        "slippage_bps": 3.0,
        "borrow_rate_y": 0.05,
        "borrow_rate_x": 0.05,
        "financing_rate": 0.03,
        "periods_per_year": 252,
        "force_liquidation": True,
    }
    parameters.update(overrides)
    return run_walk_forward_analysis(
        prices,
        formation_window,
        trading_window,
        **parameters,
    )


def test_fold_boundaries_are_generated_by_inclusive_row_position() -> None:
    folds = generate_walk_forward_folds(pd.RangeIndex(14), 5, 3)

    assert len(folds) == 3
    assert (
        folds[0].formation_start_position,
        folds[0].formation_end_position,
        folds[0].trading_start_position,
        folds[0].trading_end_position,
    ) == (0, 4, 5, 7)
    assert (
        folds[1].formation_start_position,
        folds[1].formation_end_position,
        folds[1].trading_start_position,
        folds[1].trading_end_position,
    ) == (3, 7, 8, 10)


def test_formation_and_trading_ranges_never_overlap_within_fold() -> None:
    folds = generate_walk_forward_folds(pd.RangeIndex(30), 8, 4, step_size=2)

    for fold in folds:
        assert fold.formation_end_position < fold.trading_start_position
        assert fold.trading_start_position == fold.formation_end_position + 1


def test_default_step_size_equals_trading_window() -> None:
    folds = generate_walk_forward_folds(pd.RangeIndex(30), 8, 4)

    starts = [fold.trading_start_position for fold in folds]
    assert np.diff(starts).tolist() == [4] * (len(starts) - 1)


def test_custom_step_size_advances_both_fixed_windows() -> None:
    folds = generate_walk_forward_folds(
        pd.RangeIndex(30),
        8,
        3,
        step_size=5,
    )

    assert folds[1].formation_start_position == 5
    assert folds[1].formation_end_position == 12
    assert folds[1].trading_start_position == 13


def test_expanding_formation_keeps_start_and_advances_end() -> None:
    folds = generate_walk_forward_folds(
        pd.RangeIndex(30),
        8,
        4,
        expanding=True,
    )

    assert folds[0].formation_start_position == 0
    assert folds[1].formation_start_position == 0
    assert folds[0].formation_end_position == 7
    assert folds[1].formation_end_position == 11


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("formation_window", True),
        ("formation_window", 0),
        ("formation_window", 3.5),
        ("trading_window", False),
        ("trading_window", -1),
        ("step_size", "2"),
        ("step_size", 0),
        ("minimum_observations", 1.5),
        ("expanding", "yes"),
    ],
)
def test_invalid_fold_parameters_are_rejected(name: str, value: Any) -> None:
    kwargs: dict[str, Any] = {
        "formation_window": 5,
        "trading_window": 3,
    }
    if name in kwargs:
        kwargs[name] = value
    else:
        kwargs[name] = value

    with pytest.raises((TypeError, ValueError), match=name):
        generate_walk_forward_folds(pd.RangeIndex(20), **kwargs)


def test_minimum_observations_cannot_exceed_formation_window() -> None:
    with pytest.raises(ValueError, match="minimum_observations"):
        generate_walk_forward_folds(
            pd.RangeIndex(20),
            5,
            3,
            minimum_observations=6,
        )


def test_too_short_index_is_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="too short"):
        generate_walk_forward_folds(pd.RangeIndex(7), 5, 3)


def test_datetime_fold_labels_are_preserved() -> None:
    index = pd.date_range("2024-01-01", periods=12, tz="UTC", name="date")

    fold = generate_walk_forward_folds(index, 5, 3)[0]

    assert fold.formation_start_label == index[0]
    assert fold.formation_end_label == index[4]
    assert fold.trading_start_label == index[5]
    assert fold.trading_end_label == index[7]


def test_non_datetime_nonmonotonic_labels_are_preserved_by_position() -> None:
    index = pd.Index(["z", "a", "q", "b", "x", "c", "w", "d"])

    fold = generate_walk_forward_folds(index, 5, 3)[0]

    assert fold.formation_start_label == "z"
    assert fold.formation_end_label == "x"
    assert fold.trading_start_label == "c"
    assert fold.trading_end_label == "d"


def test_nonmonotonic_datetime_index_is_rejected() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"])

    with pytest.raises(ValueError, match="monotonically increasing"):
        generate_walk_forward_folds(index, 1, 1)


def test_duplicate_fold_index_is_rejected() -> None:
    index = pd.Index(["a", "b", "b", "c"])

    with pytest.raises(ValueError, match="unique"):
        generate_walk_forward_folds(index, 2, 1)


def test_pair_screening_receives_formation_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    calls = _install_screening(monkeypatch)
    fold = _fold(prices)

    result = _run_fold(prices, fold)

    assert result.status is WalkForwardStatus.COMPLETED
    assert len(calls) == 1
    pd.testing.assert_frame_equal(
        calls[0],
        prices.iloc[:60],
    )


def test_trading_prices_cannot_change_selection_or_frozen_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)
    fold = _fold(prices)
    changed = prices.copy(deep=True)
    changed.iloc[fold.trading_start_position :, 0] *= 1.25

    original = _run_fold(prices, fold)
    modified = _run_fold(changed, fold)

    assert original.selected_symbol_y == modified.selected_symbol_y
    assert original.selected_symbol_x == modified.selected_symbol_x
    assert original.frozen_alpha == modified.frozen_alpha
    assert original.frozen_beta == modified.frozen_beta
    assert original.selected_screening_result == modified.selected_screening_result


def test_data_after_fold_cannot_change_formation_screening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    calls = _install_screening(monkeypatch)
    fold = _fold(prices)
    changed = prices.copy(deep=True)
    changed.iloc[fold.trading_end_position + 1 :] *= 1.4

    _run_fold(prices, fold)
    _run_fold(changed, fold)

    pd.testing.assert_frame_equal(calls[0], calls[1])


def test_rank_one_pair_is_selected_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank_two = _screening_result(symbol_x="CCC", rank=2, corrected_pvalue=0.02)
    rank_one = _screening_result(rank=1, corrected_pvalue=0.01)
    _install_screening(monkeypatch, (rank_two, rank_one))
    prices = _prices()

    result = _run_fold(prices, _fold(prices))

    assert result.status is WalkForwardStatus.COMPLETED
    assert result.selected_symbol_y == "AAA"
    assert result.selected_symbol_x == "BBB"
    assert result.screening_rank == 1


def test_no_passing_pair_produces_explicit_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected = _screening_result(selected=False, rank=None, corrected_pvalue=0.8)
    _install_screening(monkeypatch, (rejected,))
    prices = _prices()

    result = _run_fold(prices, _fold(prices))

    assert result.status is WalkForwardStatus.NO_SELECTION
    assert result.candidates_screened == 1
    assert result.backtest is None
    assert result.performance_report is None
    assert result.trade_count == 0
    assert result.ending_capital == result.starting_capital


def test_insufficient_formation_data_is_an_explicit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(60)
    calls = _install_screening(monkeypatch)
    fold = generate_walk_forward_folds(prices.index, 49, 5)[0]

    result = _run_fold(prices, fold)

    assert result.status is WalkForwardStatus.INSUFFICIENT_DATA
    assert result.backtest is None
    assert calls == []


def test_formation_rows_generate_no_oos_accounting_or_ledger_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)
    fold = _fold(prices)

    result = _run_fold(prices, fold)

    assert result.backtest is not None
    expected_index = prices.index[60:68]
    assert result.backtest.accounting.index.equals(expected_index)
    assert result.backtest.positions.index.equals(expected_index)
    assert not result.backtest.accounting.index.isin(prices.index[:60]).any()
    if not result.backtest.ledger.empty:
        assert result.backtest.ledger["entry_index"].isin(expected_index).all()
        assert result.backtest.ledger["exit_index"].isin(expected_index).all()


def test_each_completed_fold_begins_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    result = _run_fold(prices, _fold(prices))

    assert result.backtest is not None
    assert result.backtest.positions["executed_state"].iat[0] == "FLAT"
    assert result.backtest.positions["units_y"].iat[0] == 0.0
    assert result.backtest.positions["units_x"].iat[0] == 0.0


def test_completed_fold_finishes_flat_under_force_liquidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    result = _run_fold(prices, _fold(prices), force_liquidation=True)

    assert result.status is WalkForwardStatus.COMPLETED
    assert result.backtest is not None
    assert result.backtest.forced_liquidation_requested
    assert result.backtest.positions["executed_state"].iat[-1] == "FLAT"
    assert result.backtest.positions["units_y"].iat[-1] == 0.0
    assert result.backtest.positions["units_x"].iat[-1] == 0.0


def test_execution_lag_remains_active_inside_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    result = _run_fold(prices, _fold(prices), execution_lag=1)

    assert result.backtest is not None
    entry_decisions = np.flatnonzero(
        result.backtest.signals["event"].isin(["ENTER_LONG", "ENTER_SHORT"])
    )
    entry_executions = np.flatnonzero(
        result.backtest.positions["execution_event"].isin(
            ["ENTER_LONG", "ENTER_SHORT"]
        )
    )
    assert len(entry_decisions) >= 1
    assert len(entry_executions) >= 1
    assert entry_executions[0] >= entry_decisions[0] + 1
    assert result.backtest.execution_lag == 1


def test_transaction_costs_remain_active_inside_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    result = _run_fold(prices, _fold(prices))

    assert result.backtest is not None
    assert result.backtest.accounting["transaction_cost"].sum() > 0.0
    assert result.backtest.accounting["commission_cost"].sum() > 0.0
    assert result.backtest.accounting["slippage_cost"].sum() > 0.0


def test_borrow_and_financing_costs_remain_active_inside_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    result = _run_fold(prices, _fold(prices))

    assert result.backtest is not None
    assert result.backtest.accounting["borrow_cost"].sum() > 0.0
    assert result.backtest.accounting["financing_cost"].sum() > 0.0
    assert result.backtest.accounting["carry_cost"].sum() > 0.0


def test_performance_report_uses_trading_returns_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    result = _run_fold(prices, _fold(prices), risk_free_rate=0.04)

    assert result.performance_report is not None
    assert result.backtest is not None
    valid_returns = result.backtest.accounting["net_return_after_carry"].notna().sum()
    assert result.performance_report.report_observations == valid_returns
    assert result.performance_report.report_observations <= result.trading_observations
    assert result.performance_report.core == calculate_core_metrics(
        result.backtest.accounting["net_return_after_carry"],
        252,
        0.04,
    )


def test_fold_status_metadata_and_outputs_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)
    fold = _fold(prices)

    first = _run_fold(prices, fold)
    second = _run_fold(prices, fold)

    assert first.status == second.status
    assert first.fold == second.fold
    assert first.selected_screening_result == second.selected_screening_result
    assert first.frozen_alpha == second.frozen_alpha
    assert first.frozen_beta == second.frozen_beta
    assert first.ending_capital == pytest.approx(second.ending_capital)
    assert first.backtest is not None and second.backtest is not None
    pd.testing.assert_frame_equal(first.backtest.accounting, second.backtest.accounting)
    pd.testing.assert_frame_equal(first.backtest.ledger, second.backtest.ledger)


def test_trading_future_mutation_leaves_earlier_outputs_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)
    fold = _fold(prices)
    cutoff_position = fold.trading_start_position + 3
    changed = prices.copy(deep=True)
    changed.iloc[cutoff_position + 1 : fold.trading_end_position + 1, 0] *= 1.35

    original = _run_fold(prices, fold)
    modified = _run_fold(changed, fold)

    assert original.backtest is not None and modified.backtest is not None
    prefix_rows = cutoff_position - fold.trading_start_position + 1
    pd.testing.assert_frame_equal(
        original.backtest.signals.iloc[:prefix_rows],
        modified.backtest.signals.iloc[:prefix_rows],
    )
    pd.testing.assert_frame_equal(
        original.backtest.positions.iloc[:prefix_rows],
        modified.backtest.positions.iloc[:prefix_rows],
    )
    pd.testing.assert_frame_equal(
        original.backtest.accounting.iloc[:prefix_rows],
        modified.backtest.accounting.iloc[:prefix_rows],
    )
    original_closed = original.backtest.ledger.loc[
        original.backtest.ledger["exit_row"] < prefix_rows
    ]
    modified_closed = modified.backtest.ledger.loc[
        modified.backtest.ledger["exit_row"] < prefix_rows
    ]
    pd.testing.assert_frame_equal(original_closed, modified_closed)


def test_fold_execution_does_not_mutate_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    prices.attrs["source"] = "fixture"
    before = prices.copy(deep=True)
    _install_screening(monkeypatch)

    _run_fold(prices, _fold(prices))

    pd.testing.assert_frame_equal(prices, before)
    assert prices.attrs == before.attrs


def test_invalid_run_parameters_are_rejected_even_before_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch, (_screening_result(selected=False),))

    with pytest.raises(ValueError, match="initial_capital"):
        _run_fold(prices, _fold(prices), initial_capital=0.0)
    with pytest.raises(TypeError, match="execution_lag"):
        _run_fold(prices, _fold(prices), execution_lag=True)
    with pytest.raises(ValueError, match="entry_z"):
        _run_fold(prices, _fold(prices), entry_z=0.2, exit_z=0.25)


def test_later_fold_data_does_not_change_earlier_fold_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)
    changed = prices.copy(deep=True)
    changed.iloc[68:] *= 1.2

    original = _run_analysis(prices)
    modified = _run_analysis(changed)

    first = original.folds[0]
    changed_first = modified.folds[0]
    assert first.selected_screening_result == changed_first.selected_screening_result
    assert first.frozen_beta == changed_first.frozen_beta
    assert first.backtest is not None and changed_first.backtest is not None
    pd.testing.assert_frame_equal(
        first.backtest.signals,
        changed_first.backtest.signals,
    )
    pd.testing.assert_frame_equal(
        first.backtest.positions,
        changed_first.backtest.positions,
    )
    pd.testing.assert_frame_equal(
        first.backtest.accounting,
        changed_first.backtest.accounting,
    )
    pd.testing.assert_frame_equal(first.backtest.ledger, changed_first.backtest.ledger)


def test_overlapping_trading_windows_are_rejected_for_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    calls = _install_screening(monkeypatch)

    with pytest.raises(ValueError, match="overlapping trading windows"):
        _run_analysis(prices, step_size=4)

    assert calls == []


def test_non_overlapping_oos_returns_concatenate_in_fold_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    result = _run_analysis(prices)

    expected = pd.concat(
        [
            fold.backtest.accounting["net_return_after_carry"]
            for fold in result.folds
            if fold.status is WalkForwardStatus.COMPLETED
            and fold.backtest is not None
        ]
    ).rename("oos_return")
    pd.testing.assert_series_equal(result.oos_returns, expected)
    assert result.oos_returns.index.is_monotonic_increasing
    assert result.oos_returns.index.is_unique


def test_aggregate_metrics_use_concatenated_returns_not_fold_averages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    result = _run_analysis(prices)

    assert result.overall_performance_report is not None
    expected = calculate_core_metrics(result.oos_returns, 252)
    assert result.overall_performance_report.core == expected
    fold_sharpes = [
        fold.performance_report.core.sharpe_ratio
        for fold in result.folds
        if fold.performance_report is not None
    ]
    assert result.overall_performance_report.core.sharpe_ratio != pytest.approx(
        np.mean(fold_sharpes)
    )
    assert result.aggregate_trade_dollar_metrics_available is False
    assert not hasattr(result.overall_performance_report, "trades")


def test_no_selection_fold_is_cash_in_calendar_but_not_conditional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    calls = 0

    def alternating_screen(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[PairScreeningResult, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return (_screening_result(),)
        return (_screening_result(selected=False, corrected_pvalue=0.8),)

    monkeypatch.setattr(walkforward_module, "screen_pairs", alternating_screen)

    result = _run_analysis(prices)

    assert result.fold_count == 2
    assert result.completed_fold_count == 1
    assert result.no_selection_fold_count == 1
    assert result.total_oos_observations == 8
    assert len(result.oos_returns) == 8
    assert len(result.conditional_oos_returns) == 8
    assert len(result.calendar_oos_returns) == 16
    assert result.calendar_oos_returns.iloc[8:].eq(0.0).all()
    assert result.no_selection_oos_observations == 8
    assert result.selection_coverage == pytest.approx(0.5)
    assert result.folds[1].backtest is None
    assert result.folds[1].trade_count == 0
    assert result.calendar_performance_report is not None
    assert result.conditional_performance_report is not None
    assert result.calendar_performance_report.core == calculate_core_metrics(
        result.calendar_oos_returns,
        252,
    )
    assert result.conditional_performance_report.core == calculate_core_metrics(
        result.conditional_oos_returns,
        252,
    )


def test_walk_forward_status_counts_are_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    calls = 0

    def alternating_screen(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[PairScreeningResult, ...]:
        nonlocal calls
        calls += 1
        return (
            _screening_result()
            if calls == 1
            else _screening_result(selected=False, corrected_pvalue=0.8),
        )

    monkeypatch.setattr(walkforward_module, "screen_pairs", alternating_screen)

    result = _run_analysis(prices)

    assert result.fold_count == 2
    assert result.completed_fold_count == 1
    assert result.no_selection_fold_count == 1
    assert result.insufficient_data_fold_count == 0


def test_one_row_executed_fold_is_retained_when_analytics_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(61)
    _install_screening(monkeypatch)

    result = _run_analysis(prices, trading_window=1)

    assert result.completed_fold_count == 1
    assert result.folds[0].status is WalkForwardStatus.COMPLETED
    assert result.folds[0].analytics_status is WalkForwardAnalyticsStatus.UNAVAILABLE
    assert result.folds[0].performance_report is None
    assert len(result.folds[0].oos_returns) == 1
    assert len(result.conditional_oos_returns) == 1
    assert len(result.calendar_oos_returns) == 1


def test_performance_report_failure_preserves_execution_returns_and_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(68)
    _install_screening(monkeypatch)

    def fail_report(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("forced reporting failure")

    monkeypatch.setattr(walkforward_module, "build_performance_report", fail_report)

    result = _run_analysis(prices)
    fold = result.folds[0]

    assert fold.status is WalkForwardStatus.COMPLETED
    assert fold.analytics_status is WalkForwardAnalyticsStatus.FAILED
    assert fold.performance_report is None
    assert "forced reporting failure" in str(fold.analytics_error)
    assert fold.backtest is not None
    assert len(fold.backtest.ledger) == 1
    assert len(result.conditional_oos_returns) == 8
    assert result.selected_oos_observations == 8


def test_catastrophic_return_is_retained_and_aggregate_analytics_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(68)
    _install_screening(monkeypatch)
    original_runner = walkforward_module.run_pair_backtest

    def catastrophic_runner(*args: Any, **kwargs: Any) -> Any:
        backtest = original_runner(*args, **kwargs)
        accounting = backtest.accounting.copy(deep=True)
        accounting.loc[accounting.index[-1], "net_return_after_carry"] = -1.0
        return replace(backtest, accounting=accounting)

    monkeypatch.setattr(
        walkforward_module,
        "run_pair_backtest",
        catastrophic_runner,
    )

    result = _run_analysis(prices)

    assert result.folds[0].status is WalkForwardStatus.COMPLETED
    assert result.folds[0].analytics_status is WalkForwardAnalyticsStatus.UNAVAILABLE
    assert result.conditional_oos_returns.iat[-1] == -1.0
    assert result.calendar_oos_returns.iat[-1] == -1.0
    assert result.conditional_performance_report is None
    assert result.calendar_performance_report is None
    assert result.conditional_analytics_status is WalkForwardAnalyticsStatus.UNAVAILABLE
    assert "-100%" in str(result.conditional_analytics_error)


def test_insufficient_data_rows_are_unavailable_not_cash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(68)
    prices.loc[prices.index[:60], ["BBB", "CCC"]] = np.nan
    calls = _install_screening(monkeypatch)

    result = _run_analysis(prices)

    assert calls == []
    assert result.insufficient_data_fold_count == 1
    assert result.unavailable_oos_observations == 8
    assert result.no_selection_oos_observations == 0
    assert result.conditional_oos_returns.empty
    assert result.calendar_oos_returns.isna().all()
    assert result.calendar_performance_report is None
    assert result.calendar_analytics_status is WalkForwardAnalyticsStatus.UNAVAILABLE


def test_aggregate_analysis_rejects_gapped_windows_before_screening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(datetime_index=True)
    calls = _install_screening(monkeypatch)

    with pytest.raises(ValueError, match="calendar gaps"):
        _run_analysis(prices, step_size=12)

    assert calls == []


def test_aggregate_analysis_accepts_exactly_contiguous_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    _install_screening(monkeypatch)

    result = _run_analysis(prices, step_size=8)

    assert result.fold_count == 2
    assert result.calendar_oos_returns.index.equals(prices.index[60:76])


def test_equal_capital_reset_return_policy_does_not_claim_dollar_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    _install_screening(monkeypatch)
    folds = generate_walk_forward_folds(prices.index, 60, 8)
    templates = tuple(_run_fold(prices, fold) for fold in folds)
    fold_returns = (-0.50, 0.10)
    fold_dollar_pnl = (-50_000.0, 10_000.0)
    calls = 0

    def fixed_fold(*args: Any, **kwargs: Any) -> WalkForwardFoldResult:
        nonlocal calls
        fold = args[1]
        template = templates[calls]
        scheduled_index = prices.index[
            fold.trading_start_position : fold.trading_end_position + 1
        ]
        returns = pd.Series(0.0, index=scheduled_index, name="oos_return")
        returns.iat[0] = fold_returns[calls]
        assert template.backtest is not None
        accounting = template.backtest.accounting.copy(deep=True)
        accounting["net_pnl_after_carry"] = 0.0
        accounting.iloc[0, accounting.columns.get_loc("net_pnl_after_carry")] = (
            fold_dollar_pnl[calls]
        )
        backtest = replace(template.backtest, accounting=accounting)
        ending_capital = 100_000.0 + fold_dollar_pnl[calls]
        calls += 1
        return replace(
            template,
            fold=fold,
            backtest=backtest,
            oos_returns=returns,
            performance_report=None,
            analytics_status=WalkForwardAnalyticsStatus.UNAVAILABLE,
            analytics_error="fixture",
            ending_capital=ending_capital,
        )

    monkeypatch.setattr(walkforward_module, "run_walk_forward_fold", fixed_fold)

    result = _run_analysis(prices)

    assert result.capital_policy == "equal_capital_reset"
    assert result.aggregate_return_policy == "time_weighted_equal_capital_reset"
    assert result.aggregate_dollar_pnl_available is False
    assert result.aggregate_trade_dollar_metrics_available is False
    assert result.calendar_performance_report is not None
    assert result.calendar_performance_report.core.total_return == pytest.approx(-0.45)
    raw_fold_pnl = sum(
        float(fold.backtest.accounting["net_pnl_after_carry"].sum())
        for fold in result.folds
        if fold.backtest is not None
    )
    assert raw_fold_pnl == pytest.approx(-40_000.0)
    assert 100_000.0 * result.calendar_performance_report.core.total_return == pytest.approx(
        -45_000.0
    )


def test_future_missingness_cannot_change_earlier_formation_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    prices["LATE"] = np.nan
    prices.loc[prices.index[60:], "LATE"] = np.linspace(20.0, 22.0, 16)
    changed = prices.copy(deep=True)
    changed.loc[changed.index[68:], "LATE"] = np.nan
    calls = _install_screening(monkeypatch)
    fold = generate_walk_forward_folds(prices.index, 60, 8)[0]

    original = _run_fold(prices, fold)
    modified = _run_fold(changed, fold)

    assert original.eligible_symbols == modified.eligible_symbols
    assert "LATE" not in original.eligible_symbols
    assert "LATE" not in calls[0].columns
    pd.testing.assert_frame_equal(calls[0], calls[1])


def test_future_observed_mask_values_cannot_invalidate_earlier_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    mask = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    prices.attrs[OBSERVED_PRICE_MASK_ATTR] = mask
    changed = prices.copy(deep=True)
    changed_mask = mask.copy(deep=True).astype(object)
    changed_mask.iloc[68:, :] = np.nan
    changed.attrs[OBSERVED_PRICE_MASK_ATTR] = changed_mask
    _install_screening(monkeypatch)
    fold = generate_walk_forward_folds(prices.index, 60, 8)[0]

    original = _run_fold(prices, fold)
    modified = _run_fold(changed, fold)

    assert original.status is WalkForwardStatus.COMPLETED
    assert modified.status is WalkForwardStatus.COMPLETED
    assert original.backtest is not None and modified.backtest is not None
    pd.testing.assert_frame_equal(original.backtest.accounting, modified.backtest.accounting)


def test_later_listed_symbol_becomes_eligible_only_in_later_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(140)
    prices["LATE"] = np.nan
    prices.loc[prices.index[60:], "LATE"] = np.linspace(30.0, 35.0, 80)
    calls = _install_screening(monkeypatch)
    folds = generate_walk_forward_folds(prices.index, 60, 10)

    early = _run_fold(prices, folds[0])
    later = _run_fold(prices, folds[6])

    assert "LATE" not in early.eligible_symbols
    assert "LATE" in later.eligible_symbols
    assert "LATE" not in calls[0].columns
    assert "LATE" in calls[1].columns


def test_per_fold_group_provider_changes_only_subsequent_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    seen: list[dict[str, tuple[str, ...]]] = []

    def provider(fold: WalkForwardFold) -> dict[str, list[str]]:
        return {
            "sector": ["AAA", "BBB"] if fold.fold_id == 1 else ["AAA", "CCC"]
        }

    def screen(
        formation: pd.DataFrame,
        groups: Any = None,
        **kwargs: Any,
    ) -> tuple[PairScreeningResult, ...]:
        snapshot = {group: tuple(symbols) for group, symbols in groups.items()}
        seen.append(snapshot)
        symbols = snapshot["sector"]
        return (_screening_result(symbol_y=symbols[0], symbol_x=symbols[1]),)

    monkeypatch.setattr(walkforward_module, "screen_pairs", screen)

    result = _run_analysis(prices, groups=provider)

    assert seen == [
        {"sector": ("AAA", "BBB")},
        {"sector": ("AAA", "CCC")},
    ]
    assert result.folds[0].group_snapshot == seen[0]
    assert result.folds[1].group_snapshot == seen[1]
    assert result.universe_provenance == "per_fold_groups_caller_supplied_unvalidated"
    assert result.point_in_time_universe_validated is False


def test_static_generator_groups_are_materialized_once_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    symbols = (symbol for symbol in ["CCC", "AAA", "BBB"])
    groups: dict[str, Any] = {"sector": symbols}
    original_value = groups["sector"]
    seen: list[tuple[str, ...]] = []

    def screen(
        formation: pd.DataFrame,
        snapshot: Any = None,
        **kwargs: Any,
    ) -> tuple[PairScreeningResult, ...]:
        seen.append(tuple(snapshot["sector"]))
        return (_screening_result(),)

    monkeypatch.setattr(walkforward_module, "screen_pairs", screen)

    result = _run_analysis(prices, groups=groups)

    assert seen == [("AAA", "BBB", "CCC"), ("AAA", "BBB", "CCC")]
    assert groups["sector"] is original_value
    assert result.folds[0].group_snapshot == result.folds[1].group_snapshot
    assert result.universe_provenance == "static_groups_caller_supplied_unvalidated"
    assert result.point_in_time_universe_validated is False
    assert any("Static groups" in warning for warning in result.provenance_warnings)


@pytest.mark.parametrize(
    "members",
    [
        ["CCC", "AAA", "BBB"],
        ("CCC", "AAA", "BBB"),
        {"CCC", "AAA", "BBB"},
    ],
)
def test_group_iterable_order_does_not_change_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    members: Any,
) -> None:
    prices = _prices(68)
    _install_screening(monkeypatch)

    result = _run_analysis(prices, groups={"sector": members})

    assert result.folds[0].group_snapshot == {
        "sector": ("AAA", "BBB", "CCC")
    }


def test_result_frames_do_not_alias_source_or_intermediate_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(68)
    _install_screening(monkeypatch)
    original_runner = walkforward_module.run_pair_backtest
    captured: list[Any] = []

    def capture_runner(*args: Any, **kwargs: Any) -> Any:
        backtest = original_runner(*args, **kwargs)
        captured.append(backtest)
        return backtest

    monkeypatch.setattr(walkforward_module, "run_pair_backtest", capture_runner)
    result = _run_analysis(prices)
    conditional_before = result.conditional_oos_returns.copy(deep=True)
    accounting_before = result.folds[0].backtest.accounting.copy(deep=True)  # type: ignore[union-attr]

    prices.iloc[:, :] = 1.0
    captured[0].accounting["net_return_after_carry"] = 999.0

    pd.testing.assert_series_equal(result.conditional_oos_returns, conditional_before)
    assert result.folds[0].backtest is not None
    pd.testing.assert_frame_equal(result.folds[0].backtest.accounting, accounting_before)


def test_fold_results_have_independent_mutable_pandas_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(76)
    _install_screening(monkeypatch)
    result = _run_analysis(prices)
    assert result.folds[0].backtest is not None
    assert result.folds[1].backtest is not None
    second_before = result.folds[1].backtest.accounting.copy(deep=True)

    result.folds[0].backtest.accounting["net_return_after_carry"] = -777.0

    pd.testing.assert_frame_equal(result.folds[1].backtest.accounting, second_before)


@pytest.mark.parametrize(
    ("rows", "expected_discarded"),
    [(76, 0), (79, 3)],
)
def test_terminal_horizon_metadata_reports_discarded_rows(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    expected_discarded: int,
) -> None:
    prices = _prices(rows)
    _install_screening(monkeypatch)

    result = _run_analysis(prices)

    assert result.fold_count == 2
    assert result.evaluated_start_position == 60
    assert result.evaluated_end_position == 75
    assert result.evaluated_start_label == prices.index[60]
    assert result.evaluated_end_label == prices.index[75]
    assert result.discarded_terminal_rows == expected_discarded


def test_real_screening_is_formation_only_under_future_mutation() -> None:
    prices, groups = make_synthetic_universe(n_days=400, seed=123)
    fold = generate_walk_forward_folds(prices.index, 320, 40)[0]
    changed = prices.copy(deep=True)
    changed.iloc[fold.trading_start_position :, :] *= np.linspace(
        1.1,
        1.8,
        len(changed) - fold.trading_start_position,
    )[:, None]
    parameters = {
        "groups": groups,
        "screening_min_observations": 100,
        "fdr_threshold": 0.05,
        "max_half_life": 100.0,
        "hurst_threshold": 0.7,
        "zscore_lookback": 20,
        "entry_z": 1.0,
        "exit_z": 0.25,
        "stop_z": 50.0,
        "target_gross_notional": 10_000.0,
        "initial_capital": 100_000.0,
    }

    original = run_walk_forward_fold(prices, fold, **parameters)
    modified = run_walk_forward_fold(changed, fold, **parameters)

    assert original.status is WalkForwardStatus.COMPLETED
    assert modified.status is WalkForwardStatus.COMPLETED
    assert original.selected_symbol_y == modified.selected_symbol_y
    assert original.selected_symbol_x == modified.selected_symbol_x
    assert original.screening_rank == modified.screening_rank
    assert original.corrected_pvalue == modified.corrected_pvalue
    assert original.frozen_alpha == modified.frozen_alpha
    assert original.frozen_beta == modified.frozen_beta
    assert original.screening_results == modified.screening_results


def test_repeated_walk_forward_runs_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)

    first = _run_analysis(prices)
    second = _run_analysis(prices)

    assert first.fold_count == second.fold_count
    assert first.completed_fold_count == second.completed_fold_count
    pd.testing.assert_series_equal(first.oos_returns, second.oos_returns)
    for original, repeated in zip(first.folds, second.folds):
        assert original.status == repeated.status
        assert original.selected_screening_result == repeated.selected_screening_result
        assert original.backtest is not None and repeated.backtest is not None
        pd.testing.assert_frame_equal(
            original.backtest.accounting,
            repeated.backtest.accounting,
        )


def test_datetime_analysis_preserves_oos_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(datetime_index=True)
    _install_screening(monkeypatch)

    result = _run_analysis(prices)

    expected = prices.index[60:]
    assert result.oos_returns.index.equals(expected)
    assert result.oos_returns.index.tz == expected.tz


def test_non_datetime_row_order_is_supported_without_sorting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices(68)
    labels = pd.Index([f"label-{position:02d}" for position in range(68)][::-1])
    prices.index = labels
    calls = _install_screening(monkeypatch)

    result = _run_analysis(prices)

    assert result.completed_fold_count == 1
    assert result.oos_returns.index.equals(labels[60:68])
    assert calls[0].iloc[:, 0].tolist() == prices.iloc[:60, 0].tolist()


def test_walk_forward_fold_is_immutable() -> None:
    fold = generate_walk_forward_folds(pd.RangeIndex(10), 5, 3)[0]

    with pytest.raises(FrozenInstanceError):
        fold.fold_id = 99  # type: ignore[misc]


def test_walk_forward_fold_result_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)
    result = _run_fold(prices, _fold(prices))

    with pytest.raises(FrozenInstanceError):
        result.status = WalkForwardStatus.NO_SELECTION  # type: ignore[misc]


def test_walk_forward_result_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    _install_screening(monkeypatch)
    result = _run_analysis(prices)

    with pytest.raises(FrozenInstanceError):
        result.fold_count = 0  # type: ignore[misc]
