"""Focused tests for causal walk-forward out-of-sample evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.analytics import calculate_core_metrics
from pairs_trading.screening import PairScreeningResult
import pairs_trading.walkforward as walkforward_module
from pairs_trading.walkforward import (
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


def _run_analysis(prices: pd.DataFrame, **overrides: Any) -> WalkForwardResult:
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
    return run_walk_forward_analysis(prices, 60, 8, **parameters)


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

    with pytest.raises(ValueError, match="Overlapping trading windows"):
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
    assert result.overall_performance_report.trades.trades == sum(
        fold.trade_count
        for fold in result.folds
        if fold.status is WalkForwardStatus.COMPLETED
    )


def test_no_selection_fold_adds_no_fabricated_zero_returns(
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
