"""Focused tests for Milestone 6A execution scheduling and pair sizing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.backtest import (
    BacktestResult,
    LedgerReconciliation,
    PairUnits,
    TradeRecord,
    TradeExitReason,
    apply_execution_costs,
    build_trade_ledger,
    build_net_pnl_schedule,
    build_pnl_schedule,
    build_position_schedule,
    calculate_pair_units,
    calculate_borrow_costs,
    calculate_financing_costs,
    calculate_position_pnl,
    calculate_rebalancing_costs,
    calculate_strategy_returns,
    calculate_transaction_costs,
    build_financed_pnl_schedule,
    force_liquidate_open_position,
    lag_trade_decisions,
    reconcile_trade_ledger,
    run_pair_backtest,
    validate_backtest_invariants,
)
from pairs_trading.data import MarketDataLoader, OBSERVED_PRICE_MASK_ATTR


def _signals_from_events(
    events: list[str],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Build the state/event subset emitted by generate_trade_signals()."""
    current_state = "FLAT"
    states: list[str] = []
    for event in events:
        if event == "ENTER_LONG":
            current_state = "LONG_SPREAD"
        elif event == "ENTER_SHORT":
            current_state = "SHORT_SPREAD"
        elif event.startswith("EXIT_"):
            current_state = "FLAT"
        states.append(current_state)

    if index is None:
        index = pd.RangeIndex(len(events), name="row")
    return pd.DataFrame({"state": states, "event": events}, index=index)


def _market_inputs(
    events: list[str],
    *,
    index: pd.Index | None = None,
    beta: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Return aligned deterministic prices, signals, and dynamic beta."""
    if index is None:
        index = pd.RangeIndex(len(events), name="row")
    count = len(events)
    price_y = pd.Series(
        100.0 + 10.0 * np.arange(count),
        index=index,
        name="Y",
    )
    price_x = pd.Series(
        50.0 + 5.0 * np.arange(count),
        index=index,
        name="X",
    )
    signals = _signals_from_events(events, index)
    hedge_ratio = pd.Series(beta, index=index, name="beta", dtype=float)
    return price_y, price_x, signals, hedge_ratio


def _schedule(
    events: list[str],
    *,
    target: float = 12_000.0,
    execution_lag: int = 1,
    index: pd.Index | None = None,
    hedge_ratio: float | pd.Series = 2.0,
    price_y: pd.Series | None = None,
    price_x: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a schedule with compact defaults used throughout this module."""
    default_y, default_x, signals, _ = _market_inputs(events, index=index)
    return build_position_schedule(
        default_y if price_y is None else price_y,
        default_x if price_x is None else price_x,
        signals,
        hedge_ratio,
        target,
        execution_lag,
    )


def test_build_position_schedule_returns_required_schema_and_preserves_index() -> None:
    index = pd.Index(["third", "first", "second"], name="observation")
    result = _schedule(["ENTER_LONG", "NONE", "NONE"], index=index)

    assert result.columns.tolist() == [
        "decision_state",
        "executed_state",
        "decision_event",
        "execution_event",
        "hedge_ratio",
        "price_y",
        "price_x",
        "observed_y",
        "observed_x",
        "units_y",
        "units_x",
        "notional_y",
        "notional_x",
        "gross_exposure",
        "net_exposure",
    ]
    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert not any(
        term in result.columns
        for term in ("pnl", "return", "cost", "borrow", "trade", "ledger")
    )


def test_lag_trade_decisions_uses_row_lag_without_same_row_execution() -> None:
    signals = _signals_from_events(["ENTER_LONG", "NONE", "EXIT_STOP"])

    result = lag_trade_decisions(signals)

    assert result["decision_event"].tolist() == [
        "ENTER_LONG",
        "NONE",
        "EXIT_STOP",
    ]
    assert result["due_event"].tolist() == ["NONE", "ENTER_LONG", "NONE"]
    assert result["due_state"].tolist() == [None, "LONG_SPREAD", "LONG_SPREAD"]


def test_default_one_row_lag_has_no_same_row_execution() -> None:
    result = _schedule(["ENTER_LONG", "NONE", "NONE"])

    assert result["executed_state"].tolist() == [
        "FLAT",
        "LONG_SPREAD",
        "LONG_SPREAD",
    ]
    assert result["execution_event"].tolist() == ["NONE", "ENTER_LONG", "NONE"]
    assert result.loc[0, "units_y"] == 0.0
    assert result.loc[0, "units_x"] == 0.0


def test_lag_is_applied_by_row_position_for_lag_two() -> None:
    index = pd.Index([30, 10, 40, 20], name="sequence")
    signals = _signals_from_events(
        ["ENTER_LONG", "NONE", "NONE", "EXIT_TIME"],
        index,
    )

    result = lag_trade_decisions(signals, execution_lag=np.int64(2))

    assert result["due_event"].tolist() == ["NONE", "NONE", "ENTER_LONG", "NONE"]
    pd.testing.assert_index_equal(result.index, index, exact=True)


def test_final_row_entry_remains_unexecuted() -> None:
    result = _schedule(["NONE", "NONE", "ENTER_LONG"])

    assert result["decision_event"].iloc[-1] == "ENTER_LONG"
    assert result["executed_state"].eq("FLAT").all()
    assert result["execution_event"].eq("NONE").all()
    assert result[["units_y", "units_x"]].eq(0.0).all().all()


def test_long_spread_units_follow_signed_hedge_ratio_weights() -> None:
    units = calculate_pair_units("LONG_SPREAD", 200.0, 25.0, 2.0, 12_000.0)

    assert isinstance(units, PairUnits)
    assert units.units_y == pytest.approx(20.0)
    assert units.units_x == pytest.approx(-320.0)
    notional_y = units.units_y * 200.0
    notional_x = units.units_x * 25.0
    assert abs(notional_y) + abs(notional_x) == pytest.approx(12_000.0)
    assert abs(notional_x / notional_y) == pytest.approx(2.0)


def test_short_spread_units_follow_signed_hedge_ratio_weights() -> None:
    units = calculate_pair_units("SHORT_SPREAD", 200.0, 25.0, 2.0, 12_000.0)

    assert units.units_y == pytest.approx(-20.0)
    assert units.units_x == pytest.approx(320.0)
    assert abs(units.units_y * 200.0) + abs(units.units_x * 25.0) == pytest.approx(
        12_000.0
    )


def test_extreme_finite_hedge_ratio_preserves_both_pair_legs() -> None:
    units = calculate_pair_units(
        "LONG_SPREAD",
        1.0,
        1.0,
        np.finfo(np.float64).max,
        12_000.0,
    )

    assert np.isfinite(units).all()
    assert units.units_y > 0.0
    assert units.units_x < 0.0


def test_flat_position_has_zero_units_and_exposures() -> None:
    units = calculate_pair_units("FLAT", 100.0, 50.0, 2.0, 12_000.0)
    result = _schedule(["NONE", "NONE"])

    assert units == (0.0, 0.0)
    assert result[["units_y", "units_x"]].eq(0.0).all().all()
    assert result[["gross_exposure", "net_exposure"]].eq(0.0).all().all()


@pytest.mark.parametrize(
    ("entry_event", "expected_y_sign", "expected_x_sign"),
    [("ENTER_LONG", 1, -1), ("ENTER_SHORT", -1, 1)],
)
def test_execution_row_gross_exposure_equals_target_and_has_expected_signs(
    entry_event: str,
    expected_y_sign: int,
    expected_x_sign: int,
) -> None:
    result = _schedule([entry_event, "NONE", "NONE"])
    execution = result.iloc[1]

    assert np.sign(execution["units_y"]) == expected_y_sign
    assert np.sign(execution["units_x"]) == expected_x_sign
    assert execution["gross_exposure"] == pytest.approx(12_000.0)
    assert abs(execution["notional_x"] / execution["notional_y"]) == pytest.approx(
        2.0
    )


def test_units_use_execution_row_prices_not_decision_row_prices() -> None:
    index = pd.RangeIndex(3)
    price_y = pd.Series([10.0, 200.0, 300.0], index=index)
    price_x = pd.Series([500.0, 25.0, 30.0], index=index)
    result = _schedule(
        ["ENTER_LONG", "NONE", "NONE"],
        price_y=price_y,
        price_x=price_x,
    )

    assert result.loc[1, "units_y"] == pytest.approx(4_000.0 / 200.0)
    assert result.loc[1, "units_x"] == pytest.approx(-8_000.0 / 25.0)
    assert result.loc[1, "units_y"] != pytest.approx(4_000.0 / 10.0)


def test_scalar_hedge_ratio_is_broadcast() -> None:
    result = _schedule(["ENTER_LONG", "NONE", "NONE"], hedge_ratio=1.5)

    assert result["hedge_ratio"].tolist() == [1.5, 1.5, 1.5]
    assert result.loc[1, "gross_exposure"] == pytest.approx(12_000.0)


def test_dynamic_hedge_ratio_uses_latest_prior_available_value() -> None:
    index = pd.RangeIndex(3)
    beta = pd.Series([0.25, 3.0, 8.0], index=index)
    result = _schedule(
        ["ENTER_LONG", "NONE", "NONE"],
        index=index,
        hedge_ratio=beta,
    )

    execution = result.loc[1]
    assert abs(execution["notional_x"] / execution["notional_y"]) == pytest.approx(
        0.25
    )


def test_future_hedge_ratios_do_not_change_earlier_units() -> None:
    index = pd.RangeIndex(5)
    beta = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)
    changed = beta.copy(deep=True)
    changed.iloc[2:] = [30.0, 40.0, 50.0]

    original = _schedule(
        ["ENTER_LONG", "NONE", "NONE", "NONE", "NONE"],
        index=index,
        hedge_ratio=beta,
    )
    modified = _schedule(
        ["ENTER_LONG", "NONE", "NONE", "NONE", "NONE"],
        index=index,
        hedge_ratio=changed,
    )

    pd.testing.assert_frame_equal(original.iloc[:2], modified.iloc[:2])


def test_holdings_persist_without_rebalancing_between_executions() -> None:
    index = pd.RangeIndex(5)
    price_y = pd.Series([100.0, 100.0, 150.0, 80.0, 120.0], index=index)
    price_x = pd.Series([50.0, 50.0, 40.0, 70.0, 60.0], index=index)
    beta = pd.Series([1.0, 2.0, 8.0, 0.5, 4.0], index=index)
    result = _schedule(
        ["ENTER_LONG", "NONE", "NONE", "NONE", "NONE"],
        index=index,
        price_y=price_y,
        price_x=price_x,
        hedge_ratio=beta,
    )

    assert result.loc[1:, "units_y"].nunique() == 1
    assert result.loc[1:, "units_x"].nunique() == 1
    assert result.loc[1:, "execution_event"].tolist() == [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "NONE",
    ]
    assert result.loc[1:, "gross_exposure"].nunique() > 1


def test_exit_decision_remains_held_until_lagged_close() -> None:
    result = _schedule(["ENTER_LONG", "NONE", "EXIT_STOP", "NONE"])

    assert result["executed_state"].tolist() == [
        "FLAT",
        "LONG_SPREAD",
        "LONG_SPREAD",
        "FLAT",
    ]
    assert result["execution_event"].tolist() == [
        "NONE",
        "ENTER_LONG",
        "NONE",
        "EXIT_STOP",
    ]
    assert result.loc[3, ["units_y", "units_x", "gross_exposure"]].tolist() == [
        0.0,
        0.0,
        0.0,
    ]


def test_execution_event_occurs_only_when_executed_state_changes() -> None:
    result = _schedule(
        ["NONE", "ENTER_LONG", "NONE", "EXIT_TIME", "NONE"],
        execution_lag=1,
    )

    changes = result["executed_state"].ne(result["executed_state"].shift(fill_value="FLAT"))
    assert result.loc[changes, "execution_event"].ne("NONE").all()
    assert result.loc[~changes, "execution_event"].eq("NONE").all()


@pytest.mark.parametrize("missing_input", ["price_y", "price_x", "hedge_ratio"])
def test_missing_due_input_defers_entry_to_next_fully_valid_row(
    missing_input: str,
) -> None:
    events = ["ENTER_LONG", "NONE", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    inputs = {"price_y": price_y, "price_x": price_x, "hedge_ratio": beta}
    inputs[missing_input] = inputs[missing_input].copy(deep=True)
    missing_row = 0 if missing_input == "hedge_ratio" else 1
    inputs[missing_input].iloc[missing_row] = np.nan

    result = build_position_schedule(
        inputs["price_y"],
        inputs["price_x"],
        signals,
        inputs["hedge_ratio"],
        12_000.0,
    )

    assert result.loc[1, "executed_state"] == "FLAT"
    assert result.loc[1, "execution_event"] == "NONE"
    assert result.loc[2, "executed_state"] == "LONG_SPREAD"
    assert result.loc[2, "execution_event"] == "ENTER_LONG"
    assert result.loc[2, "gross_exposure"] == pytest.approx(12_000.0)


def test_deferred_order_at_data_end_remains_unexecuted() -> None:
    events = ["ENTER_LONG", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    price_y.iloc[1] = np.nan

    result = build_position_schedule(price_y, price_x, signals, beta, 12_000.0)

    assert result["executed_state"].eq("FLAT").all()
    assert result["execution_event"].eq("NONE").all()


def test_none_decisions_do_not_expire_a_pending_order() -> None:
    events = ["ENTER_LONG", "NONE", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    price_y.iloc[1:3] = np.nan

    result = build_position_schedule(price_y, price_x, signals, beta, 12_000.0)

    assert result["execution_event"].tolist() == [
        "NONE",
        "NONE",
        "NONE",
        "ENTER_LONG",
    ]


@pytest.mark.parametrize("missing_input", ["price_y", "price_x", "hedge_ratio"])
def test_missing_due_input_defers_exit_to_next_fully_valid_row(
    missing_input: str,
) -> None:
    events = ["ENTER_LONG", "NONE", "EXIT_STOP", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    inputs = {"price_y": price_y, "price_x": price_x, "hedge_ratio": beta}
    inputs[missing_input] = inputs[missing_input].copy(deep=True)
    missing_row = 2 if missing_input == "hedge_ratio" else 3
    inputs[missing_input].iloc[missing_row] = np.nan

    result = build_position_schedule(
        inputs["price_y"],
        inputs["price_x"],
        signals,
        inputs["hedge_ratio"],
        12_000.0,
    )

    assert result.loc[3, "executed_state"] == "LONG_SPREAD"
    assert result.loc[3, "execution_event"] == "NONE"
    assert result.loc[4, "executed_state"] == "FLAT"
    assert result.loc[4, "execution_event"] == "EXIT_STOP"
    assert result.loc[4, ["units_y", "units_x"]].tolist() == [0.0, 0.0]


def test_deferred_exit_and_opposite_entry_execute_on_distinct_rows() -> None:
    events = [
        "ENTER_LONG",
        "NONE",
        "EXIT_MEAN_REVERSION",
        "ENTER_SHORT",
        "NONE",
        "NONE",
    ]
    price_y, price_x, signals, beta = _market_inputs(events)
    price_y.iloc[3] = np.nan

    result = build_position_schedule(price_y, price_x, signals, beta, 12_000.0)

    assert result.loc[3, "executed_state"] == "LONG_SPREAD"
    assert result.loc[3, "execution_event"] == "NONE"
    assert result.loc[4, "executed_state"] == "FLAT"
    assert result.loc[4, "execution_event"] == "EXIT_MEAN_REVERSION"
    assert result.loc[5, "executed_state"] == "SHORT_SPREAD"
    assert result.loc[5, "execution_event"] == "ENTER_SHORT"


def test_missing_mark_preserves_units_without_backfilling_exposure() -> None:
    events = ["ENTER_LONG", "NONE", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    price_y.iloc[2] = np.nan

    result = build_position_schedule(price_y, price_x, signals, beta, 12_000.0)

    assert result.loc[2, "units_y"] == result.loc[1, "units_y"]
    assert result.loc[2, "units_x"] == result.loc[1, "units_x"]
    assert pd.isna(result.loc[2, "notional_y"])
    assert np.isfinite(result.loc[2, "notional_x"])
    assert pd.isna(result.loc[2, "gross_exposure"])
    assert pd.isna(result.loc[2, "net_exposure"])


def test_forward_filled_price_cannot_trigger_entry() -> None:
    events = ["ENTER_LONG", "NONE", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    observed_y = pd.Series([True, False, True, True], index=price_y.index)
    observed_x = pd.Series(True, index=price_x.index)

    result = build_position_schedule(
        price_y,
        price_x,
        signals,
        beta,
        12_000.0,
        observed_y=observed_y,
        observed_x=observed_x,
    )

    assert result.loc[1, "executed_state"] == "FLAT"
    assert result.loc[1, "execution_event"] == "NONE"
    assert not bool(result.loc[1, "observed_y"])
    assert result.loc[2, "execution_event"] == "ENTER_LONG"


def test_cleaned_price_provenance_automatically_blocks_entry() -> None:
    dates = pd.bdate_range("2025-01-02", periods=4)
    raw = pd.DataFrame(
        {
            "Y": [100.0, np.nan, 102.0, 103.0],
            "X": [50.0, 51.0, 52.0, 53.0],
        },
        index=dates,
    )
    clean, _ = MarketDataLoader.clean(
        raw,
        min_coverage=0.75,
        max_forward_fill=1,
        min_observations=4,
    )
    signals = _signals_from_events(["ENTER_LONG", "NONE", "NONE", "NONE"], dates)

    result = build_position_schedule(
        clean["Y"],
        clean["X"],
        signals,
        1.0,
        2_000.0,
    )

    assert not bool(result.loc[dates[1], "observed_y"])
    assert result.loc[dates[1], "execution_event"] == "NONE"
    assert result.loc[dates[2], "execution_event"] == "ENTER_LONG"


def test_forward_filled_price_cannot_trigger_exit() -> None:
    events = ["ENTER_LONG", "NONE", "EXIT_STOP", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    observed_y = pd.Series([True, True, True, False, True], index=price_y.index)

    result = build_position_schedule(
        price_y,
        price_x,
        signals,
        beta,
        12_000.0,
        observed_y=observed_y,
    )

    assert result.loc[3, "executed_state"] == "LONG_SPREAD"
    assert result.loc[3, "execution_event"] == "NONE"
    assert result.loc[4, "executed_state"] == "FLAT"
    assert result.loc[4, "execution_event"] == "EXIT_STOP"


def test_stale_price_may_value_holdings_but_cannot_execute() -> None:
    events = ["ENTER_LONG", "NONE", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    observed_y = pd.Series([True, True, False, True], index=price_y.index)
    schedule = build_position_schedule(
        price_y,
        price_x,
        signals,
        beta,
        12_000.0,
        observed_y=observed_y,
    )

    accounting = build_pnl_schedule(
        schedule,
        price_y,
        price_x,
        initial_capital=20_000.0,
    )

    assert not bool(schedule.loc[2, "observed_y"])
    assert schedule.loc[2, "execution_event"] == "NONE"
    assert schedule.loc[2, "units_y"] == schedule.loc[1, "units_y"]
    assert schedule.loc[2, "notional_y"] == pytest.approx(
        schedule.loc[2, "units_y"] * price_y.loc[2]
    )
    assert accounting.loc[2, "market_value_y"] == pytest.approx(
        schedule.loc[2, "units_y"] * price_y.loc[2]
    )
    assert np.isfinite(accounting.loc[2, "gross_pnl"])


def test_due_entry_executes_before_same_row_flat_decision_matures() -> None:
    events = ["ENTER_LONG", "EXIT_MEAN_REVERSION", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)

    result = build_position_schedule(
        price_y,
        price_x,
        signals,
        beta,
        12_000.0,
    )

    assert result.loc[1, "decision_state"] == "FLAT"
    assert result.loc[1, "execution_event"] == "ENTER_LONG"
    assert result.loc[1, "executed_state"] == "LONG_SPREAD"
    assert result.loc[2, "execution_event"] == "EXIT_MEAN_REVERSION"
    assert result.loc[2, "executed_state"] == "FLAT"


def test_deferred_entry_remains_live_before_flat_decision_matures() -> None:
    events = ["ENTER_LONG", "NONE", "EXIT_MEAN_REVERSION", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    observed_y = pd.Series([True, False, True, True], index=price_y.index)

    result = build_position_schedule(
        price_y,
        price_x,
        signals,
        beta,
        12_000.0,
        observed_y=observed_y,
    )

    assert result.loc[1, "execution_event"] == "NONE"
    assert result.loc[2, "decision_state"] == "FLAT"
    assert result.loc[2, "execution_event"] == "ENTER_LONG"
    assert result.loc[2, "executed_state"] == "LONG_SPREAD"
    assert result.loc[3, "execution_event"] == "EXIT_MEAN_REVERSION"


def test_deferred_entry_is_cancelled_when_flat_decision_becomes_due() -> None:
    events = ["ENTER_LONG", "EXIT_MEAN_REVERSION", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    observed_y = pd.Series([True, False, True, True], index=price_y.index)

    result = build_position_schedule(
        price_y,
        price_x,
        signals,
        beta,
        12_000.0,
        observed_y=observed_y,
    )

    assert result.loc[1, "execution_event"] == "NONE"
    assert result.loc[2, "decision_event"] == "NONE"
    assert result.loc[2, "execution_event"] == "NONE"
    assert result["executed_state"].eq("FLAT").all()


def test_pending_entry_cancelled_before_fill_has_no_cost_or_trade() -> None:
    events = ["ENTER_LONG", "EXIT_MEAN_REVERSION", "NONE", "NONE"]
    price_y, price_x, signals, _ = _market_inputs(events)
    observed_y = pd.Series([True, False, True, True], index=price_y.index)

    result = run_pair_backtest(
        price_y,
        price_x,
        1.0,
        12_000.0,
        signals=signals,
        observed_y=observed_y,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=1.0,
    )

    assert result.positions["executed_state"].eq("FLAT").all()
    assert result.positions["execution_event"].eq("NONE").all()
    assert result.accounting["transaction_cost"].eq(0.0).all()
    assert result.ledger.empty


def test_non_finite_marked_exposure_is_rejected() -> None:
    events = ["ENTER_LONG", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    price_y.iloc[2] = np.finfo(np.float64).max

    with pytest.raises(ValueError, match="Marked leg notionals"):
        build_position_schedule(price_y, price_x, signals, beta, 12_000.0)


@pytest.mark.parametrize(
    "invalid",
    [0.0, -1.0, np.inf, -np.inf, "100", True, np.bool_(False)],
)
def test_invalid_price_observations_are_rejected(invalid: Any) -> None:
    events = ["NONE", "NONE"]
    price_y, price_x, signals, _ = _market_inputs(events)
    price_y = price_y.astype(object)
    price_y.iloc[0] = invalid

    with pytest.raises((TypeError, ValueError), match="price_y"):
        build_position_schedule(price_y, price_x, signals, 2.0, 12_000.0)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "2", 0.0, -1.0, np.nan, np.inf, -np.inf, 1 + 2j],
)
def test_invalid_scalar_hedge_ratios_are_rejected(invalid: Any) -> None:
    price_y, price_x, signals, _ = _market_inputs(["NONE", "NONE"])

    with pytest.raises((TypeError, ValueError), match="hedge_ratio"):
        build_position_schedule(price_y, price_x, signals, invalid, 12_000.0)


@pytest.mark.parametrize("invalid", [True, "2", 0.0, -1.0, np.inf, -np.inf])
def test_invalid_dynamic_hedge_ratio_observations_are_rejected(invalid: Any) -> None:
    price_y, price_x, signals, beta = _market_inputs(["NONE", "NONE"])
    beta = beta.astype(object)
    beta.iloc[0] = invalid

    with pytest.raises((TypeError, ValueError), match="hedge_ratio"):
        build_position_schedule(price_y, price_x, signals, beta, 12_000.0)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "1000", 0.0, -1.0, np.nan, np.inf, -np.inf, 1 + 2j],
)
def test_invalid_target_gross_notional_is_rejected(invalid: Any) -> None:
    price_y, price_x, signals, _ = _market_inputs(["NONE", "NONE"])

    with pytest.raises((TypeError, ValueError), match="target_gross_notional"):
        build_position_schedule(price_y, price_x, signals, 2.0, invalid)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), 0, -1, 1.0, 1.5, "1", np.nan, None],

)
def test_invalid_execution_lags_are_rejected(invalid: Any) -> None:
    signals = _signals_from_events(["NONE", "NONE"])

    with pytest.raises((TypeError, ValueError), match="execution_lag"):
        lag_trade_decisions(signals, invalid)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), 0, -1, 1.0, 1.5, "1", np.nan, None],
)
def test_dynamic_hedge_ratio_lag_must_be_a_positive_integer(invalid: Any) -> None:
    price_y, price_x, signals, beta = _market_inputs(["NONE", "NONE"])

    with pytest.raises((TypeError, ValueError), match="hedge_ratio_lag"):
        build_position_schedule(
            price_y,
            price_x,
            signals,
            beta,
            12_000.0,
            hedge_ratio_lag=invalid,
        )


@pytest.mark.parametrize("mismatched", ["price_x", "signals", "hedge_ratio"])
def test_mismatched_or_reordered_indices_are_rejected(mismatched: str) -> None:
    events = ["ENTER_LONG", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    reversed_index = price_y.index[::-1]
    if mismatched == "price_x":
        price_x = price_x.set_axis(reversed_index)
    elif mismatched == "signals":
        signals = signals.set_axis(reversed_index)
    else:
        beta = beta.set_axis(reversed_index)

    with pytest.raises(ValueError, match="index"):
        build_position_schedule(price_y, price_x, signals, beta, 12_000.0)


@pytest.mark.parametrize("duplicated", ["price_y", "price_x", "signals", "hedge_ratio"])
def test_duplicate_indices_are_rejected_for_every_indexed_input(duplicated: str) -> None:
    events = ["ENTER_LONG", "NONE", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    duplicate_index = pd.Index([0, 0, 1])
    if duplicated == "price_y":
        price_y = price_y.set_axis(duplicate_index)
    elif duplicated == "price_x":
        price_x = price_x.set_axis(duplicate_index)
    elif duplicated == "signals":
        signals = signals.set_axis(duplicate_index)
    else:
        beta = beta.set_axis(duplicate_index)

    with pytest.raises(ValueError, match="unique index"):
        build_position_schedule(price_y, price_x, signals, beta, 12_000.0)


def test_backtest_inputs_are_not_mutated() -> None:
    events = ["ENTER_LONG", "NONE", "EXIT_STOP", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    originals = tuple(item.copy(deep=True) for item in (price_y, price_x, signals, beta))

    build_position_schedule(price_y, price_x, signals, beta, 12_000.0)

    pd.testing.assert_series_equal(price_y, originals[0])
    pd.testing.assert_series_equal(price_x, originals[1])
    pd.testing.assert_frame_equal(signals, originals[2])
    pd.testing.assert_series_equal(beta, originals[3])


def test_repeated_position_schedules_are_deterministic() -> None:
    events = ["ENTER_LONG", "NONE", "EXIT_TIME", "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)

    first = build_position_schedule(price_y, price_x, signals, beta, 12_000.0)
    second = build_position_schedule(price_y, price_x, signals, beta, 12_000.0)

    pd.testing.assert_frame_equal(first, second)


def test_non_datetime_nonmonotonic_index_is_supported_in_row_order() -> None:
    index = pd.Index(["z", "a", "m", "b"], name="sequence")
    result = _schedule(
        ["ENTER_SHORT", "NONE", "EXIT_STOP", "NONE"],
        index=index,
    )

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result["execution_event"].tolist() == [
        "NONE",
        "ENTER_SHORT",
        "NONE",
        "EXIT_STOP",
    ]


def test_timezone_aware_datetime_index_metadata_is_preserved() -> None:
    index = pd.date_range(
        "2025-03-27",
        periods=4,
        freq="h",
        tz="Europe/London",
        name="decision_time",
    )
    result = _schedule(
        ["ENTER_LONG", "NONE", "EXIT_TIME", "NONE"],
        index=index,
    )

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result.index.tz == index.tz
    assert result.index.freq == index.freq
    assert result.index.name == index.name


def test_strictly_future_changes_do_not_affect_schedule_prefix() -> None:
    index = pd.RangeIndex(7)
    original_events = [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "NONE",
        "EXIT_TIME",
        "NONE",
        "NONE",
    ]
    changed_events = [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "EXIT_STOP",
        "ENTER_SHORT",
        "NONE",
        "NONE",
    ]
    original_y, original_x, original_signals, original_beta = _market_inputs(
        original_events,
        index=index,
    )
    changed_y = original_y.copy(deep=True)
    changed_x = original_x.copy(deep=True)
    changed_beta = original_beta.copy(deep=True)
    changed_y.iloc[3:] *= 5.0
    changed_x.iloc[3:] *= 0.5
    changed_beta.iloc[3:] = [5.0, 6.0, 7.0, 8.0]
    changed_signals = _signals_from_events(changed_events, index)

    original = build_position_schedule(
        original_y,
        original_x,
        original_signals,
        original_beta,
        12_000.0,
    )
    changed = build_position_schedule(
        changed_y,
        changed_x,
        changed_signals,
        changed_beta,
        12_000.0,
    )

    pd.testing.assert_frame_equal(original.iloc[:3], changed.iloc[:3])


@pytest.mark.parametrize(
    "signals",
    [
        pd.DataFrame({"state": ["FLAT"]}),
        pd.DataFrame({"event": ["NONE"]}),
        pd.DataFrame({"state": ["LONG_SPREAD"], "event": ["NONE"]}),
        pd.DataFrame({"state": ["SHORT_SPREAD"], "event": ["ENTER_LONG"]}),
    ],
)
def test_malformed_signal_contract_is_rejected(signals: pd.DataFrame) -> None:
    index = signals.index
    price_y = pd.Series(100.0, index=index)
    price_x = pd.Series(50.0, index=index)

    with pytest.raises((TypeError, ValueError), match="signals|transition|required"):
        build_position_schedule(price_y, price_x, signals, 2.0, 12_000.0)


def _accounting_case(
    *,
    events: list[str] | None = None,
    y_values: list[float] | None = None,
    x_values: list[float] | None = None,
    index: pd.Index | None = None,
    initial_capital: float = 10_000.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return a simple long-spread schedule and its 6B accounting output."""
    if events is None:
        events = ["ENTER_LONG", "NONE", "EXIT_TIME", "NONE"]
    if y_values is None:
        y_values = [100.0, 100.0, 110.0, 105.0]
    if x_values is None:
        x_values = [50.0, 50.0, 45.0, 40.0]
    if index is None:
        index = pd.RangeIndex(len(events), name="row")
    price_y = pd.Series(y_values, index=index, name="Y", dtype=float)
    price_x = pd.Series(x_values, index=index, name="X", dtype=float)
    signals = _signals_from_events(events, index)
    schedule = build_position_schedule(
        price_y,
        price_x,
        signals,
        hedge_ratio=1.0,
        target_gross_notional=2_000.0,
    )
    accounting = build_pnl_schedule(
        schedule,
        price_y,
        price_x,
        initial_capital=initial_capital,
    )
    return schedule, price_y, price_x, accounting


def test_build_pnl_schedule_returns_required_schema() -> None:
    _, _, _, result = _accounting_case()

    assert result.columns.tolist() == [
        "price_y",
        "price_x",
        "units_y",
        "units_x",
        "market_value_y",
        "market_value_x",
        "gross_exposure",
        "net_exposure",
        "long_exposure",
        "short_exposure",
        "pnl_y",
        "pnl_x",
        "gross_pnl",
        "realised_pnl",
        "unrealised_pnl",
        "cumulative_realised_pnl",
        "cumulative_gross_pnl",
        "portfolio_equity",
        "strategy_return",
    ]
    assert not any(
        name in result.columns
        for name in ("transaction_cost", "slippage", "borrow_fee", "trade_id")
    )


def test_first_row_pnl_and_return_are_zero() -> None:
    _, _, _, result = _accounting_case()

    assert result.loc[0, ["pnl_y", "pnl_x", "gross_pnl"]].tolist() == [
        0.0,
        0.0,
        0.0,
    ]
    assert result.loc[0, "strategy_return"] == 0.0


def test_long_y_and_short_x_pnl_use_prior_row_units_with_correct_signs() -> None:
    schedule, _, _, result = _accounting_case()

    assert schedule.loc[1, "units_y"] == pytest.approx(10.0)
    assert schedule.loc[1, "units_x"] == pytest.approx(-20.0)
    assert result.loc[2, "pnl_y"] == pytest.approx(100.0)
    assert result.loc[2, "pnl_x"] == pytest.approx(100.0)
    assert result.loc[2, "gross_pnl"] == pytest.approx(200.0)


def test_entry_execution_earns_no_prior_interval_and_starts_next_interval() -> None:
    schedule, _, _, result = _accounting_case()

    assert schedule.loc[1, "execution_event"] == "ENTER_LONG"
    assert result.loc[1, "gross_pnl"] == 0.0
    assert result.loc[2, "gross_pnl"] == pytest.approx(200.0)


def test_exit_execution_keeps_final_prior_position_interval_pnl() -> None:
    schedule, _, _, result = _accounting_case()

    assert schedule.loc[3, "execution_event"] == "EXIT_TIME"
    assert schedule.loc[3, ["units_y", "units_x"]].tolist() == [0.0, 0.0]
    assert result.loc[3, "pnl_y"] == pytest.approx(-50.0)
    assert result.loc[3, "pnl_x"] == pytest.approx(100.0)
    assert result.loc[3, "gross_pnl"] == pytest.approx(50.0)


def test_flat_positions_produce_zero_pnl_even_when_prices_move() -> None:
    events = ["NONE", "NONE", "NONE", "NONE"]
    _, _, _, result = _accounting_case(
        events=events,
        y_values=[100.0, 120.0, 80.0, 140.0],
        x_values=[50.0, 30.0, 70.0, 20.0],
    )

    assert result[["pnl_y", "pnl_x", "gross_pnl"]].eq(0.0).all().all()


def test_gross_and_cumulative_pnl_equal_expected_leg_accounting() -> None:
    _, _, _, result = _accounting_case()

    pd.testing.assert_series_equal(
        result["gross_pnl"],
        (result["pnl_y"] + result["pnl_x"]).rename("gross_pnl"),
    )
    assert result["cumulative_gross_pnl"].tolist() == pytest.approx(
        [0.0, 0.0, 200.0, 250.0]
    )


def test_market_values_and_exposure_decomposition_use_current_holdings() -> None:
    schedule, _, _, result = _accounting_case()
    row = result.loc[2]

    assert row["market_value_y"] == pytest.approx(
        schedule.loc[2, "units_y"] * result.loc[2, "price_y"]
    )
    assert row["market_value_x"] == pytest.approx(
        schedule.loc[2, "units_x"] * result.loc[2, "price_x"]
    )
    assert row["market_value_y"] == pytest.approx(1_100.0)
    assert row["market_value_x"] == pytest.approx(-900.0)
    assert row["gross_exposure"] == pytest.approx(2_000.0)
    assert row["net_exposure"] == pytest.approx(200.0)
    assert row["long_exposure"] == pytest.approx(1_100.0)
    assert row["short_exposure"] == pytest.approx(900.0)


def test_flat_exposures_are_exact_zero_without_negative_zero() -> None:
    _, _, _, result = _accounting_case()

    flat = result.loc[[0, 3], [
        "market_value_y",
        "market_value_x",
        "gross_exposure",
        "net_exposure",
        "long_exposure",
        "short_exposure",
    ]]
    assert flat.eq(0.0).all().all()
    assert not np.signbit(flat.to_numpy(dtype=float)).any()


def test_missing_current_and_prior_marks_while_holding_make_interval_pnl_nan() -> None:
    events = ["ENTER_LONG", "NONE", "NONE", "NONE", "NONE"]
    _, _, _, result = _accounting_case(
        events=events,
        y_values=[100.0, 100.0, np.nan, 110.0, 115.0],
        x_values=[50.0, 50.0, 48.0, 46.0, 44.0],
    )

    assert pd.isna(result.loc[2, "pnl_y"])
    assert pd.isna(result.loc[2, "gross_pnl"])
    assert pd.isna(result.loc[3, "pnl_y"])
    assert pd.isna(result.loc[3, "gross_pnl"])
    assert np.isfinite(result.loc[4, "pnl_y"])
    assert np.isfinite(result.loc[4, "gross_pnl"])


def test_missing_marks_while_flat_do_not_fabricate_pnl_or_exposure() -> None:
    events = ["NONE", "NONE", "NONE", "NONE"]
    index = pd.RangeIndex(4, name="row")
    price_y = pd.Series([100.0, np.nan, 110.0, np.nan], index=index)
    price_x = pd.Series([np.nan, 50.0, np.nan, 55.0], index=index)
    signals = _signals_from_events(events, index)
    schedule = build_position_schedule(price_y, price_x, signals, 1.0, 2_000.0)

    result = build_pnl_schedule(schedule, price_y, price_x, 10_000.0)

    assert result[["pnl_y", "pnl_x", "gross_pnl"]].eq(0.0).all().all()
    assert result[
        [
            "market_value_y",
            "market_value_x",
            "gross_exposure",
            "net_exposure",
            "long_exposure",
            "short_exposure",
        ]
    ].eq(0.0).all().all()


def test_missing_marks_are_not_filled_dropped_or_silently_omitted() -> None:
    index = pd.Index(["a", "c", "b", "e", "d"], name="sequence")
    events = ["ENTER_LONG", "NONE", "NONE", "NONE", "NONE"]
    _, _, _, result = _accounting_case(
        events=events,
        y_values=[100.0, 100.0, np.nan, 110.0, 115.0],
        x_values=[50.0, 50.0, 48.0, 46.0, 44.0],
        index=index,
    )

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert pd.isna(result.loc["b", "price_y"])
    assert pd.isna(result.loc["b", "gross_pnl"])
    assert pd.isna(result.loc["e", "gross_pnl"])
    assert np.isfinite(result.loc["d", "gross_pnl"])
    assert pd.isna(result.loc["d", "cumulative_gross_pnl"])
    assert pd.isna(result.loc["d", "portfolio_equity"])


def test_equity_and_returns_follow_cumulative_and_prior_equity_policy() -> None:
    _, _, _, result = _accounting_case()

    assert result["portfolio_equity"].tolist() == pytest.approx(
        [10_000.0, 10_000.0, 10_200.0, 10_250.0]
    )
    assert result["strategy_return"].tolist() == pytest.approx(
        [0.0, 0.0, 200.0 / 10_000.0, 50.0 / 10_200.0]
    )


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "10000", 0.0, -1.0, np.nan, np.inf, -np.inf],
)
def test_invalid_initial_capital_is_rejected(invalid: Any) -> None:
    schedule, price_y, price_x, _ = _accounting_case()

    with pytest.raises((TypeError, ValueError), match="initial_capital"):
        build_pnl_schedule(schedule, price_y, price_x, invalid)


def test_non_positive_prior_equity_is_rejected_before_return_division() -> None:
    gross_pnl = pd.Series([0.0, -100.0, 1.0], name="gross_pnl")

    with pytest.raises(ValueError, match="Prior portfolio equity must be positive"):
        calculate_strategy_returns(gross_pnl, initial_capital=100.0)


def test_realised_and_unrealised_pnl_follow_open_close_policy() -> None:
    _, _, _, result = _accounting_case()

    assert result["unrealised_pnl"].tolist() == pytest.approx(
        [0.0, 0.0, 200.0, 0.0]
    )
    assert result["realised_pnl"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 250.0]
    )
    assert result["cumulative_realised_pnl"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 250.0]
    )


def test_multiple_trades_realise_separately_without_unrealised_leakage() -> None:
    events = [
        "ENTER_LONG",
        "NONE",
        "EXIT_TIME",
        "ENTER_SHORT",
        "NONE",
        "EXIT_STOP",
        "NONE",
    ]
    _, _, _, result = _accounting_case(
        events=events,
        y_values=[100.0, 100.0, 110.0, 105.0, 105.0, 100.0, 95.0],
        x_values=[50.0, 50.0, 45.0, 40.0, 40.0, 42.0, 44.0],
    )

    assert result.loc[3, "realised_pnl"] == pytest.approx(250.0)
    assert result.loc[4, "unrealised_pnl"] == 0.0
    assert result.loc[6, "realised_pnl"] > 0.0
    assert result.loc[6, "unrealised_pnl"] == 0.0
    assert result.loc[6, "cumulative_realised_pnl"] == pytest.approx(
        result.loc[6, "cumulative_gross_pnl"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_column",
        "boolean_units",
        "flat_nonzero",
        "first_row_entry",
        "unannounced_resize",
    ],
)
def test_malformed_position_schedules_are_rejected(mutation: str) -> None:
    schedule, price_y, price_x, _ = _accounting_case()
    malformed = schedule.copy(deep=True)
    if mutation == "missing_column":
        malformed = malformed.drop(columns="units_x")
    elif mutation == "boolean_units":
        malformed["units_y"] = malformed["units_y"].astype(object)
        malformed.loc[0, "units_y"] = True
    elif mutation == "flat_nonzero":
        malformed.loc[0, "units_y"] = 1.0
    elif mutation == "first_row_entry":
        malformed.loc[0, "executed_state"] = "LONG_SPREAD"
        malformed.loc[0, "execution_event"] = "ENTER_LONG"
        malformed.loc[0, "units_y"] = 10.0
        malformed.loc[0, "units_x"] = -20.0
    else:
        malformed.loc[2, "units_y"] *= 2.0

    with pytest.raises((TypeError, ValueError)):
        build_pnl_schedule(malformed, price_y, price_x, 10_000.0)


@pytest.mark.parametrize("mismatch", ["price_y", "price_x"])
def test_misaligned_accounting_price_indices_are_rejected(mismatch: str) -> None:
    schedule, price_y, price_x, _ = _accounting_case()
    if mismatch == "price_y":
        price_y = price_y.set_axis(price_y.index[::-1])
    else:
        price_x = price_x.set_axis(price_x.index[::-1])

    with pytest.raises(ValueError, match="index"):
        build_pnl_schedule(schedule, price_y, price_x, 10_000.0)


@pytest.mark.parametrize("duplicated", ["schedule", "price_y", "price_x"])
def test_duplicate_accounting_indices_are_rejected(duplicated: str) -> None:
    schedule, price_y, price_x, _ = _accounting_case()
    duplicate_index = pd.Index([0, 0, 1, 2])
    if duplicated == "schedule":
        schedule = schedule.set_axis(duplicate_index)
    elif duplicated == "price_y":
        price_y = price_y.set_axis(duplicate_index)
    else:
        price_x = price_x.set_axis(duplicate_index)

    with pytest.raises(ValueError, match="unique index"):
        build_pnl_schedule(schedule, price_y, price_x, 10_000.0)


def test_pnl_accounting_does_not_mutate_inputs_and_is_deterministic() -> None:
    schedule, price_y, price_x, _ = _accounting_case()
    schedule_before = schedule.copy(deep=True)
    y_before = price_y.copy(deep=True)
    x_before = price_x.copy(deep=True)

    first = build_pnl_schedule(schedule, price_y, price_x, 10_000.0)
    second = build_pnl_schedule(schedule, price_y, price_x, 10_000.0)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(schedule, schedule_before)
    pd.testing.assert_series_equal(price_y, y_before)
    pd.testing.assert_series_equal(price_x, x_before)


def test_non_datetime_accounting_index_is_preserved() -> None:
    index = pd.Index(["z", "a", "m", "b"], name="accounting_order")
    _, _, _, result = _accounting_case(index=index)

    pd.testing.assert_index_equal(result.index, index, exact=True)


def test_timezone_aware_accounting_index_metadata_is_preserved() -> None:
    index = pd.date_range(
        "2026-03-27",
        periods=4,
        freq="h",
        tz="Europe/London",
        name="mark_time",
    )
    _, _, _, result = _accounting_case(index=index)

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result.index.tz == index.tz
    assert result.index.freq == index.freq
    assert result.index.name == index.name


def test_future_accounting_changes_do_not_alter_earlier_outputs() -> None:
    index = pd.RangeIndex(7)
    original_events = [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "NONE",
        "EXIT_TIME",
        "NONE",
        "NONE",
    ]
    changed_events = [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "EXIT_STOP",
        "ENTER_SHORT",
        "NONE",
        "NONE",
    ]
    original_y, original_x, original_signals, beta = _market_inputs(
        original_events,
        index=index,
        beta=1.0,
    )
    changed_y = original_y.copy(deep=True)
    changed_x = original_x.copy(deep=True)
    changed_y.iloc[3:] *= 1.5
    changed_x.iloc[3:] *= 0.75
    original_schedule = build_position_schedule(
        original_y,
        original_x,
        original_signals,
        beta,
        2_000.0,
    )
    changed_schedule = build_position_schedule(
        changed_y,
        changed_x,
        _signals_from_events(changed_events, index),
        beta,
        2_000.0,
    )

    original = build_pnl_schedule(
        original_schedule,
        original_y,
        original_x,
        10_000.0,
    )
    changed = build_pnl_schedule(
        changed_schedule,
        changed_y,
        changed_x,
        10_000.0,
    )

    pd.testing.assert_frame_equal(original.iloc[:3], changed.iloc[:3])


def test_public_pnl_and_return_functions_match_composed_schedule() -> None:
    schedule, price_y, price_x, combined = _accounting_case()

    pnl = calculate_position_pnl(schedule, price_y, price_x)
    returns = calculate_strategy_returns(pnl["gross_pnl"], 10_000.0)

    pd.testing.assert_frame_equal(
        pnl,
        combined[pnl.columns],
    )
    pd.testing.assert_frame_equal(
        returns,
        combined[returns.columns],
    )


def _flat_price_round_trip(
    *,
    entry_event: str = "ENTER_LONG",
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    fixed_commission_per_leg: float = 2.0,
    index: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return a constant-price round trip with transparent execution costs."""
    events = [entry_event, "NONE", "EXIT_TIME", "NONE"]
    if index is None:
        index = pd.RangeIndex(4, name="row")
    price_y = pd.Series(100.0, index=index, name="Y")
    price_x = pd.Series(50.0, index=index, name="X")
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        hedge_ratio=1.0,
        target_gross_notional=2_000.0,
    )
    result = build_net_pnl_schedule(
        schedule,
        price_y,
        price_x,
        initial_capital=10_000.0,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        fixed_commission_per_leg=fixed_commission_per_leg,
    )
    return schedule, price_y, price_x, result


def test_build_net_pnl_schedule_returns_required_schema() -> None:
    _, _, _, result = _flat_price_round_trip()

    assert result.columns.tolist() == [
        "price_y",
        "price_x",
        "units_y",
        "units_x",
        "delta_units_y",
        "delta_units_x",
        "traded_notional_y",
        "traded_notional_x",
        "commission_y",
        "commission_x",
        "fixed_commission_y",
        "fixed_commission_x",
        "commission_cost",
        "slippage_y",
        "slippage_x",
        "slippage_cost",
        "transaction_cost",
        "cumulative_transaction_cost",
        "gross_pnl",
        "net_pnl",
        "cumulative_gross_pnl",
        "cumulative_net_pnl",
        "portfolio_equity",
        "net_portfolio_equity",
        "strategy_return",
        "net_strategy_return",
    ]
    assert not any(
        name in result.columns
        for name in ("borrow_fee", "financing_cost", "margin_call", "trade_id")
    )


def test_zero_cost_parameters_reproduce_gross_pnl_equity_and_returns() -> None:
    schedule, price_y, price_x, gross = _accounting_case()

    result = build_net_pnl_schedule(
        schedule,
        price_y,
        price_x,
        initial_capital=10_000.0,
    )

    assert result["transaction_cost"].eq(0.0).all()
    pd.testing.assert_series_equal(
        result["net_pnl"].rename("gross_pnl"),
        gross["gross_pnl"],
    )
    pd.testing.assert_series_equal(
        result["net_portfolio_equity"].rename("portfolio_equity"),
        gross["portfolio_equity"],
    )
    pd.testing.assert_series_equal(
        result["net_strategy_return"].rename("strategy_return"),
        gross["strategy_return"],
    )


def test_costs_occur_only_on_actual_unit_changes_not_decision_rows() -> None:
    schedule, _, _, result = _flat_price_round_trip()

    assert schedule.loc[0, "decision_event"] == "ENTER_LONG"
    assert schedule.loc[0, "execution_event"] == "NONE"
    assert result["transaction_cost"].tolist() == pytest.approx(
        [0.0, 7.0, 0.0, 7.0]
    )
    assert result.loc[2, ["delta_units_y", "delta_units_x"]].tolist() == [
        0.0,
        0.0,
    ]


@pytest.mark.parametrize(
    ("entry_event", "expected_delta_y", "expected_delta_x"),
    [("ENTER_LONG", 10.0, -20.0), ("ENTER_SHORT", -10.0, 20.0)],
)
def test_long_and_short_entries_pay_commission_on_both_absolute_traded_legs(
    entry_event: str,
    expected_delta_y: float,
    expected_delta_x: float,
) -> None:
    _, _, _, result = _flat_price_round_trip(
        entry_event=entry_event,
        slippage_bps=0.0,
        fixed_commission_per_leg=0.0,
    )
    entry = result.loc[1]

    assert entry["delta_units_y"] == pytest.approx(expected_delta_y)
    assert entry["delta_units_x"] == pytest.approx(expected_delta_x)
    assert entry["traded_notional_y"] == pytest.approx(1_000.0)
    assert entry["traded_notional_x"] == pytest.approx(1_000.0)
    assert entry["commission_y"] == pytest.approx(1.0)
    assert entry["commission_x"] == pytest.approx(1.0)


def test_exit_pays_commission_and_fixed_fee_once_per_closing_leg() -> None:
    _, _, _, result = _flat_price_round_trip(slippage_bps=0.0)
    exit_row = result.loc[3]

    assert exit_row["delta_units_y"] == pytest.approx(-10.0)
    assert exit_row["delta_units_x"] == pytest.approx(20.0)
    assert exit_row["commission_y"] == pytest.approx(1.0)
    assert exit_row["commission_x"] == pytest.approx(1.0)
    assert exit_row["fixed_commission_y"] == pytest.approx(2.0)
    assert exit_row["fixed_commission_x"] == pytest.approx(2.0)
    assert exit_row["commission_cost"] == pytest.approx(6.0)


def test_unchanged_legs_pay_no_variable_or_fixed_commission() -> None:
    _, _, _, result = _flat_price_round_trip()
    unchanged = result.loc[2]

    assert unchanged[
        [
            "traded_notional_y",
            "traded_notional_x",
            "commission_y",
            "commission_x",
            "fixed_commission_y",
            "fixed_commission_x",
            "transaction_cost",
        ]
    ].eq(0.0).all()


def test_bps_commission_and_slippage_use_absolute_traded_notional() -> None:
    _, _, _, result = _flat_price_round_trip(
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=0.0,
    )

    for row_number in (1, 3):
        row = result.loc[row_number]
        assert row["commission_y"] == pytest.approx(1.0)
        assert row["commission_x"] == pytest.approx(1.0)
        assert row["slippage_y"] == pytest.approx(0.5)
        assert row["slippage_x"] == pytest.approx(0.5)
        assert row["slippage_cost"] == pytest.approx(1.0)
        assert row["transaction_cost"] == pytest.approx(3.0)


def test_slippage_is_adverse_for_buys_and_sells_on_open_and_close() -> None:
    _, _, _, result = _flat_price_round_trip(
        commission_bps=0.0,
        slippage_bps=10.0,
        fixed_commission_per_leg=0.0,
    )

    assert result.loc[1, "delta_units_y"] > 0.0
    assert result.loc[1, "delta_units_x"] < 0.0
    assert result.loc[3, "delta_units_y"] < 0.0
    assert result.loc[3, "delta_units_x"] > 0.0
    assert result.loc[[1, 3], ["slippage_y", "slippage_x"]].gt(0.0).all().all()
    assert result.loc[[1, 3], "net_pnl"].lt(0.0).all()


def test_transaction_cost_equals_commission_plus_slippage() -> None:
    _, _, _, result = _flat_price_round_trip()

    expected = result["commission_cost"] + result["slippage_cost"]
    pd.testing.assert_series_equal(
        result["transaction_cost"],
        expected.rename("transaction_cost"),
    )


def test_net_pnl_and_cumulative_cost_accounting_are_exact() -> None:
    _, _, _, result = _flat_price_round_trip()

    expected_net = result["gross_pnl"] - result["transaction_cost"]
    pd.testing.assert_series_equal(result["net_pnl"], expected_net.rename("net_pnl"))
    assert result["cumulative_transaction_cost"].tolist() == pytest.approx(
        [0.0, 7.0, 7.0, 14.0]
    )
    assert result["cumulative_net_pnl"].tolist() == pytest.approx(
        [0.0, -7.0, -7.0, -14.0]
    )


def test_net_equity_and_returns_use_cumulative_net_pnl_and_prior_net_equity() -> None:
    _, _, _, result = _flat_price_round_trip()

    assert result["net_portfolio_equity"].tolist() == pytest.approx(
        [10_000.0, 9_993.0, 9_993.0, 9_986.0]
    )
    assert result["net_strategy_return"].tolist() == pytest.approx(
        [0.0, -7.0 / 10_000.0, 0.0, -7.0 / 9_993.0]
    )


def test_flat_price_round_trip_loses_exactly_total_execution_costs() -> None:
    _, _, _, result = _flat_price_round_trip()

    assert result["gross_pnl"].eq(0.0).all()
    assert result.loc[3, "cumulative_net_pnl"] == pytest.approx(-14.0)
    assert result.loc[3, "cumulative_net_pnl"] == pytest.approx(
        -result.loc[3, "cumulative_transaction_cost"]
    )


def test_larger_bps_assumptions_produce_larger_costs() -> None:
    _, _, _, low = _flat_price_round_trip(
        commission_bps=1.0,
        slippage_bps=1.0,
        fixed_commission_per_leg=0.0,
    )
    _, _, _, high = _flat_price_round_trip(
        commission_bps=10.0,
        slippage_bps=10.0,
        fixed_commission_per_leg=0.0,
    )

    assert high["transaction_cost"].sum() > low["transaction_cost"].sum()


@pytest.mark.parametrize(
    "parameter",
    ["commission_bps", "slippage_bps", "fixed_commission_per_leg"],
)
def test_negative_cost_parameters_are_rejected(parameter: str) -> None:
    schedule, price_y, price_x, _ = _accounting_case()

    with pytest.raises(ValueError, match=parameter):
        calculate_transaction_costs(
            schedule,
            price_y,
            price_x,
            **{parameter: -0.01},
        )


@pytest.mark.parametrize(
    "parameter",
    ["commission_bps", "slippage_bps", "fixed_commission_per_leg"],
)
@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "1.0", np.nan, np.inf, -np.inf],
)
def test_cost_parameters_reject_booleans_strings_nan_and_infinity(
    parameter: str,
    invalid: Any,
) -> None:
    schedule, price_y, price_x, _ = _accounting_case()

    with pytest.raises((TypeError, ValueError), match=parameter):
        calculate_transaction_costs(
            schedule,
            price_y,
            price_x,
            **{parameter: invalid},
        )


@pytest.mark.parametrize("missing_leg", ["price_y", "price_x"])
def test_trade_rows_with_missing_execution_prices_are_rejected(
    missing_leg: str,
) -> None:
    schedule, price_y, price_x, _ = _flat_price_round_trip()
    if missing_leg == "price_y":
        price_y = price_y.copy(deep=True)
        price_y.iloc[1] = np.nan
    else:
        price_x = price_x.copy(deep=True)
        price_x.iloc[1] = np.nan

    with pytest.raises(ValueError, match="execution price is missing"):
        calculate_transaction_costs(schedule, price_y, price_x, 10.0)


def test_missing_gross_pnl_remains_unknown_instead_of_cost_only_loss() -> None:
    events = ["ENTER_LONG", "NONE", "EXIT_TIME", "NONE"]
    index = pd.RangeIndex(4, name="row")
    price_y = pd.Series([100.0, 100.0, np.nan, 100.0], index=index)
    price_x = pd.Series(50.0, index=index)
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        1.0,
        2_000.0,
    )

    result = build_net_pnl_schedule(
        schedule,
        price_y,
        price_x,
        10_000.0,
        commission_bps=10.0,
    )

    assert pd.isna(result.loc[2, "gross_pnl"])
    assert pd.isna(result.loc[2, "net_pnl"])
    assert result.loc[3, "transaction_cost"] > 0.0
    assert pd.isna(result.loc[3, "gross_pnl"])
    assert pd.isna(result.loc[3, "net_pnl"])
    assert pd.isna(result.loc[3, "cumulative_net_pnl"])
    assert pd.isna(result.loc[3, "net_portfolio_equity"])


def test_first_row_external_cost_uses_initial_capital_return_denominator() -> None:
    pnl = pd.DataFrame({"gross_pnl": [0.0, 0.0]})
    costs = pd.Series([5.0, 0.0], name="transaction_cost")

    result = apply_execution_costs(pnl, costs, initial_capital=1_000.0)

    assert result.loc[0, "net_pnl"] == pytest.approx(-5.0)
    assert result.loc[0, "net_portfolio_equity"] == pytest.approx(995.0)
    assert result.loc[0, "net_strategy_return"] == pytest.approx(-0.005)


def test_cost_accounting_does_not_mutate_inputs_and_is_deterministic() -> None:
    schedule, price_y, price_x, gross = _accounting_case()
    schedule_before = schedule.copy(deep=True)
    y_before = price_y.copy(deep=True)
    x_before = price_x.copy(deep=True)
    gross_before = gross.copy(deep=True)

    first = build_net_pnl_schedule(
        schedule,
        price_y,
        price_x,
        10_000.0,
        commission_bps=3.0,
        slippage_bps=2.0,
        fixed_commission_per_leg=0.5,
    )
    second = build_net_pnl_schedule(
        schedule,
        price_y,
        price_x,
        10_000.0,
        commission_bps=3.0,
        slippage_bps=2.0,
        fixed_commission_per_leg=0.5,
    )

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(schedule, schedule_before)
    pd.testing.assert_series_equal(price_y, y_before)
    pd.testing.assert_series_equal(price_x, x_before)
    pd.testing.assert_frame_equal(gross, gross_before)


@pytest.mark.parametrize("mismatch", ["price_y", "price_x"])
def test_misaligned_cost_price_indices_are_rejected(mismatch: str) -> None:
    schedule, price_y, price_x, _ = _accounting_case()
    if mismatch == "price_y":
        price_y = price_y.set_axis(price_y.index[::-1])
    else:
        price_x = price_x.set_axis(price_x.index[::-1])

    with pytest.raises(ValueError, match="index"):
        calculate_transaction_costs(schedule, price_y, price_x)


@pytest.mark.parametrize("duplicated", ["schedule", "price_y", "price_x"])
def test_duplicate_cost_indices_are_rejected(duplicated: str) -> None:
    schedule, price_y, price_x, _ = _accounting_case()
    duplicate_index = pd.Index([0, 0, 1, 2])
    if duplicated == "schedule":
        schedule = schedule.set_axis(duplicate_index)
    elif duplicated == "price_y":
        price_y = price_y.set_axis(duplicate_index)
    else:
        price_x = price_x.set_axis(duplicate_index)

    with pytest.raises(ValueError, match="unique index"):
        calculate_transaction_costs(schedule, price_y, price_x)


def test_non_datetime_cost_index_is_preserved() -> None:
    index = pd.Index(["z", "a", "m", "b"], name="execution_order")
    _, _, _, result = _flat_price_round_trip(index=index)

    pd.testing.assert_index_equal(result.index, index, exact=True)


def test_timezone_aware_cost_index_metadata_is_preserved() -> None:
    index = pd.date_range(
        "2026-10-23",
        periods=4,
        freq="h",
        tz="Europe/London",
        name="execution_time",
    )
    _, _, _, result = _flat_price_round_trip(index=index)

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result.index.tz == index.tz
    assert result.index.freq == index.freq
    assert result.index.name == index.name


def test_future_execution_changes_do_not_alter_earlier_cost_rows() -> None:
    index = pd.RangeIndex(7)
    original_events = [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "NONE",
        "EXIT_TIME",
        "NONE",
        "NONE",
    ]
    changed_events = [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "EXIT_STOP",
        "ENTER_SHORT",
        "NONE",
        "NONE",
    ]
    original_y, original_x, original_signals, beta = _market_inputs(
        original_events,
        index=index,
        beta=1.0,
    )
    changed_y = original_y.copy(deep=True)
    changed_x = original_x.copy(deep=True)
    changed_y.iloc[3:] *= 1.25
    changed_x.iloc[3:] *= 0.8
    original_schedule = build_position_schedule(
        original_y,
        original_x,
        original_signals,
        beta,
        2_000.0,
    )
    changed_schedule = build_position_schedule(
        changed_y,
        changed_x,
        _signals_from_events(changed_events, index),
        beta,
        2_000.0,
    )

    original = build_net_pnl_schedule(
        original_schedule,
        original_y,
        original_x,
        10_000.0,
        commission_bps=4.0,
        slippage_bps=3.0,
        fixed_commission_per_leg=0.25,
    )
    changed = build_net_pnl_schedule(
        changed_schedule,
        changed_y,
        changed_x,
        10_000.0,
        commission_bps=4.0,
        slippage_bps=3.0,
        fixed_commission_per_leg=0.25,
    )

    pd.testing.assert_frame_equal(original.iloc[:3], changed.iloc[:3])


def test_public_cost_functions_match_composed_net_schedule() -> None:
    schedule, price_y, price_x, combined = _flat_price_round_trip()
    gross = build_pnl_schedule(schedule, price_y, price_x, 10_000.0)
    costs = calculate_transaction_costs(
        schedule,
        price_y,
        price_x,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=2.0,
    )
    net = apply_execution_costs(gross, costs, 10_000.0)

    pd.testing.assert_frame_equal(costs, combined[costs.columns])
    pd.testing.assert_frame_equal(net, combined[net.columns])


def _constant_price_carry_case(
    *,
    entry_event: str = "ENTER_LONG",
    borrow_rate_y: float | pd.Series = 0.0,
    borrow_rate_x: float | pd.Series = 0.0,
    financing_rate: float = 0.0,
    periods_per_year: int = 252,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    index: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return an entry, exit, and post-exit row at constant prices."""
    events = [entry_event, "NONE", "EXIT_TIME", "NONE", "NONE"]
    if index is None:
        index = pd.RangeIndex(5, name="row")
    price_y = pd.Series(100.0, index=index, name="Y")
    price_x = pd.Series(50.0, index=index, name="X")
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        hedge_ratio=1.0,
        target_gross_notional=2_000.0,
    )
    result = build_financed_pnl_schedule(
        schedule,
        price_y,
        price_x,
        initial_capital=10_000.0,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        borrow_rate_y=borrow_rate_y,
        borrow_rate_x=borrow_rate_x,
        financing_rate=financing_rate,
        periods_per_year=periods_per_year,
    )
    return schedule, price_y, price_x, result


def _rebalancing_case(
    beta_values: list[float],
    *,
    threshold: float = 0.5,
    rebalance: bool = True,
    y_values: list[float] | None = None,
    x_values: list[float] | None = None,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fixed_commission_per_leg: float = 0.0,
    index: pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Return an open long-spread schedule with optional dynamic rebalancing."""
    count = len(beta_values)
    if index is None:
        index = pd.RangeIndex(count, name="row")
    if y_values is None:
        y_values = [100.0] * count
    if x_values is None:
        x_values = [50.0] * count
    price_y = pd.Series(y_values, index=index, name="Y")
    price_x = pd.Series(x_values, index=index, name="X")
    beta = pd.Series(beta_values, index=index, name="beta")
    events = ["ENTER_LONG"] + ["NONE"] * (count - 1)
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        beta,
        2_000.0,
    )
    result = build_financed_pnl_schedule(
        schedule,
        price_y,
        price_x,
        initial_capital=10_000.0,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        fixed_commission_per_leg=fixed_commission_per_leg,
        rebalance=rebalance,
        hedge_ratio=beta,
        target_gross_notional=2_000.0,
        rebalance_threshold=threshold,
    )
    return schedule, price_y, price_x, beta, result


def test_financed_schedule_preserves_existing_6c_schema_and_adds_6d_fields() -> None:
    _, _, _, result = _constant_price_carry_case()

    expected_additions = {
        "borrow_cost_y",
        "borrow_cost_x",
        "borrow_cost",
        "financing_cost",
        "carry_cost",
        "cumulative_borrow_cost",
        "cumulative_financing_cost",
        "cumulative_carry_cost",
        "rebalance",
        "rebalance_beta",
        "rebalance_delta_units_y",
        "rebalance_delta_units_x",
        "rebalance_decision_row",
        "net_pnl_after_carry",
        "cumulative_net_pnl_after_carry",
        "net_equity_after_carry",
        "net_return_after_carry",
    }
    assert set(result.columns[:26]) == {
        "price_y",
        "price_x",
        "units_y",
        "units_x",
        "delta_units_y",
        "delta_units_x",
        "traded_notional_y",
        "traded_notional_x",
        "commission_y",
        "commission_x",
        "fixed_commission_y",
        "fixed_commission_x",
        "commission_cost",
        "slippage_y",
        "slippage_x",
        "slippage_cost",
        "transaction_cost",
        "cumulative_transaction_cost",
        "gross_pnl",
        "net_pnl",
        "cumulative_gross_pnl",
        "cumulative_net_pnl",
        "portfolio_equity",
        "net_portfolio_equity",
        "strategy_return",
        "net_strategy_return",
    }
    assert expected_additions.issubset(result.columns)


def test_zero_borrow_and_financing_reproduce_6c_net_accounting() -> None:
    schedule, price_y, price_x, financed = _constant_price_carry_case(
        commission_bps=4.0,
        slippage_bps=3.0,
    )
    existing = build_net_pnl_schedule(
        schedule,
        price_y,
        price_x,
        10_000.0,
        commission_bps=4.0,
        slippage_bps=3.0,
    )

    pd.testing.assert_frame_equal(financed[existing.columns], existing)
    pd.testing.assert_series_equal(
        financed["net_pnl_after_carry"].rename("net_pnl"),
        existing["net_pnl"],
    )
    assert financed["carry_cost"].eq(0.0).all()


def test_first_row_entry_and_flat_rows_have_zero_borrow_and_financing() -> None:
    _, _, _, result = _constant_price_carry_case(
        borrow_rate_x=0.252,
        financing_rate=0.252,
    )

    assert result.loc[0, ["borrow_cost", "financing_cost"]].tolist() == [0.0, 0.0]
    assert result.loc[1, ["borrow_cost", "financing_cost"]].tolist() == [0.0, 0.0]
    assert result.loc[4, ["borrow_cost", "financing_cost"]].tolist() == [0.0, 0.0]


def test_long_spread_charges_only_short_x_borrow_using_prior_exposure() -> None:
    _, _, _, result = _constant_price_carry_case(
        borrow_rate_y=0.504,
        borrow_rate_x=0.252,
    )

    assert result["borrow_cost_y"].eq(0.0).all()
    assert result["borrow_cost_x"].tolist() == pytest.approx(
        [0.0, 0.0, 1.0, 1.0, 0.0]
    )


def test_short_spread_charges_only_short_y_borrow() -> None:
    _, _, _, result = _constant_price_carry_case(
        entry_event="ENTER_SHORT",
        borrow_rate_y=0.504,
        borrow_rate_x=0.252,
    )

    assert result["borrow_cost_y"].tolist() == pytest.approx(
        [0.0, 0.0, 2.0, 2.0, 0.0]
    )
    assert result["borrow_cost_x"].eq(0.0).all()


def test_exit_row_pays_final_borrow_and_no_fee_follows_exit() -> None:
    schedule, _, _, result = _constant_price_carry_case(borrow_rate_x=0.252)

    assert schedule.loc[3, "execution_event"] == "EXIT_TIME"
    assert result.loc[3, "borrow_cost"] == pytest.approx(1.0)
    assert result.loc[4, "borrow_cost"] == 0.0


def test_dynamic_borrow_rates_use_prior_row_rate_without_future_leakage() -> None:
    index = pd.RangeIndex(5, name="row")
    rates = pd.Series([0.0, 0.252, 0.504, 9.0, 9.0], index=index)
    _, _, _, result = _constant_price_carry_case(
        borrow_rate_x=rates,
        index=index,
    )

    assert result["borrow_cost_x"].tolist() == pytest.approx(
        [0.0, 0.0, 1.0, 2.0, 0.0]
    )


def test_very_high_finite_borrow_rate_is_valid() -> None:
    _, _, _, result = _constant_price_carry_case(borrow_rate_x=5.04)

    assert result.loc[2, "borrow_cost_x"] == pytest.approx(20.0)


@pytest.mark.parametrize("rate_name", ["borrow_rate_y", "borrow_rate_x"])
@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "0.1", -0.1, np.nan, np.inf, -np.inf],
)
def test_invalid_borrow_rates_are_rejected(rate_name: str, invalid: Any) -> None:
    _, _, _, gross = _accounting_case()

    with pytest.raises((TypeError, ValueError), match=rate_name):
        calculate_borrow_costs(gross, **{rate_name: invalid})


def test_financing_uses_prior_gross_exposure_and_includes_exit_interval() -> None:
    schedule, _, _, result = _constant_price_carry_case(financing_rate=0.252)

    assert result["financing_cost"].tolist() == pytest.approx(
        [0.0, 0.0, 2.0, 2.0, 0.0]
    )
    assert schedule.loc[3, "execution_event"] == "EXIT_TIME"


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "0.1", -0.1, np.nan, np.inf, -np.inf],
)
def test_invalid_financing_rates_are_rejected(invalid: Any) -> None:
    _, _, _, gross = _accounting_case()

    with pytest.raises((TypeError, ValueError), match="financing_rate"):
        calculate_financing_costs(gross, invalid)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), 0, -1, 252.0, 1.5, "252", np.nan],
)
def test_invalid_periods_per_year_are_rejected(invalid: Any) -> None:
    _, _, _, gross = _accounting_case()

    with pytest.raises((TypeError, ValueError), match="periods_per_year"):
        calculate_borrow_costs(gross, periods_per_year=invalid)
    with pytest.raises((TypeError, ValueError), match="periods_per_year"):
        calculate_financing_costs(gross, periods_per_year=invalid)


def test_carry_cost_and_financed_net_accounting_are_exact() -> None:
    _, _, _, result = _constant_price_carry_case(
        borrow_rate_x=0.252,
        financing_rate=0.252,
    )

    assert result["carry_cost"].tolist() == pytest.approx(
        [0.0, 0.0, 3.0, 3.0, 0.0]
    )
    expected = result["gross_pnl"] - result["transaction_cost"] - result["carry_cost"]
    pd.testing.assert_series_equal(
        result["net_pnl_after_carry"],
        expected.rename("net_pnl_after_carry"),
    )
    assert result["cumulative_carry_cost"].tolist() == pytest.approx(
        [0.0, 0.0, 3.0, 6.0, 6.0]
    )
    assert result["cumulative_net_pnl_after_carry"].tolist() == pytest.approx(
        [0.0, 0.0, -3.0, -6.0, -6.0]
    )
    assert result["net_equity_after_carry"].tolist() == pytest.approx(
        [10_000.0, 10_000.0, 9_997.0, 9_994.0, 9_994.0]
    )
    assert result["net_return_after_carry"].tolist() == pytest.approx(
        [0.0, 0.0, -3.0 / 10_000.0, -3.0 / 9_997.0, 0.0]
    )


def test_rebalance_false_preserves_original_units_exactly() -> None:
    schedule, _, _, _, result = _rebalancing_case(
        [1.0, 1.0, 2.0, 3.0, 4.0],
        rebalance=False,
    )

    pd.testing.assert_series_equal(result["units_y"], schedule["units_y"])
    pd.testing.assert_series_equal(result["units_x"], schedule["units_x"])
    assert not result["rebalance"].any()


def test_dynamic_beta_threshold_crossing_rebalances_without_state_change() -> None:
    schedule, _, _, _, result = _rebalancing_case([1.0, 1.0, 2.0, 2.0, 2.0])

    assert bool(result.loc[3, "rebalance"])
    assert result.loc[3, "rebalance_beta"] == pytest.approx(2.0)
    assert result.loc[3, "rebalance_decision_row"] == 2
    assert schedule.loc[3, "executed_state"] == "LONG_SPREAD"
    assert result.loc[3, "units_y"] == pytest.approx(2_000.0 / 3.0 / 100.0)
    assert result.loc[3, "units_x"] == pytest.approx(-4_000.0 / 3.0 / 50.0)
    assert result.loc[3, "units_y"] != schedule.loc[3, "units_y"]


def test_beta_shock_cannot_rebalance_same_row_and_records_later_execution() -> None:
    _, _, _, beta, result = _rebalancing_case(
        [1.0, 1.0, 3.0, 3.0, 3.0],
    )

    shock_row = 2
    execution_row = 3
    assert beta.loc[shock_row] == 3.0
    assert not bool(result.loc[shock_row, "rebalance"])
    assert bool(result.loc[execution_row, "rebalance"])
    assert result.loc[execution_row, "rebalance_beta"] == 3.0
    assert result.loc[execution_row, "rebalance_decision_row"] == shock_row


def test_forward_filled_price_defers_rebalance_until_observed_row() -> None:
    index = pd.RangeIndex(5, name="rebalance_observation")
    price_y = pd.Series(100.0, index=index, name="Y")
    price_x = pd.Series(50.0, index=index, name="X")
    beta = pd.Series([1.0, 1.0, 2.0, 2.0, 2.0], index=index, name="beta")
    observed_y = pd.Series([True, True, True, False, True], index=index)
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(["ENTER_LONG"] + ["NONE"] * 4, index),
        beta,
        2_000.0,
        observed_y=observed_y,
    )

    result = calculate_rebalancing_costs(
        schedule,
        price_y,
        price_x,
        beta,
        2_000.0,
        rebalance=True,
        rebalance_threshold=0.5,
    )

    assert not bool(result.loc[3, "rebalance"])
    assert bool(result.loc[4, "rebalance"])
    assert result.loc[4, "rebalance_decision_row"] == 3


def test_price_metadata_false_blocks_rebalance_despite_true_schedule_mask() -> None:
    schedule, price_y, price_x, beta, _ = _rebalancing_case(
        [1.0, 1.0, 2.0, 2.0, 2.0],
    )
    marked_y = price_y.copy(deep=True)
    metadata_mask = pd.Series(True, index=marked_y.index)
    metadata_mask.loc[3] = False
    marked_y.attrs[OBSERVED_PRICE_MASK_ATTR] = metadata_mask

    result = calculate_rebalancing_costs(
        schedule,
        marked_y,
        price_x,
        beta,
        2_000.0,
        rebalance=True,
        rebalance_threshold=0.5,
    )

    assert bool(schedule.loc[3, "observed_y"])
    assert not bool(metadata_mask.loc[3])
    assert not bool(result.loc[3, "rebalance"])
    assert bool(result.loc[4, "rebalance"])


def test_beta_change_below_threshold_does_not_rebalance() -> None:
    schedule, _, _, _, result = _rebalancing_case(
        [1.0, 1.0, 1.1, 1.1, 1.1],
        threshold=0.2,
    )

    assert not result["rebalance"].any()
    pd.testing.assert_series_equal(result["units_y"], schedule["units_y"])


def test_flat_positions_never_rebalance() -> None:
    index = pd.RangeIndex(4)
    prices_y = pd.Series(100.0, index=index)
    prices_x = pd.Series(50.0, index=index)
    beta = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    schedule = build_position_schedule(
        prices_y,
        prices_x,
        _signals_from_events(["NONE"] * 4, index),
        beta,
        2_000.0,
    )

    result = calculate_rebalancing_costs(
        schedule,
        prices_y,
        prices_x,
        beta,
        2_000.0,
        rebalance=True,
    )

    assert not result["rebalance"].any()
    assert result[["units_y", "units_x", "transaction_cost"]].eq(0.0).all().all()


def test_rebalance_uses_current_prices_and_beta_and_emits_expected_deltas() -> None:
    schedule, _, _, _, result = _rebalancing_case(
        [1.0, 1.0, 2.0, 2.0, 2.0],
        y_values=[100.0, 100.0, 150.0, 200.0, 200.0],
        x_values=[50.0, 50.0, 40.0, 25.0, 25.0],
    )

    desired_y = (2_000.0 / 3.0) / 200.0
    desired_x = -(4_000.0 / 3.0) / 25.0
    assert result.loc[3, "units_y"] == pytest.approx(desired_y)
    assert result.loc[3, "units_x"] == pytest.approx(desired_x)
    assert result.loc[3, "rebalance_delta_units_y"] == pytest.approx(
        desired_y - schedule.loc[2, "units_y"]
    )
    assert result.loc[3, "rebalance_delta_units_x"] == pytest.approx(
        desired_x - schedule.loc[2, "units_x"]
    )


def test_rebalance_transaction_uses_normal_commission_and_slippage_only() -> None:
    _, _, _, _, result = _rebalancing_case(
        [1.0, 1.0, 2.0, 2.0, 2.0],
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=2.0,
    )
    row = result.loc[3]

    expected_commission = (
        row["traded_notional_y"] + row["traded_notional_x"]
    ) * 10.0 / 10_000.0 + 4.0
    expected_slippage = (
        row["traded_notional_y"] + row["traded_notional_x"]
    ) * 5.0 / 10_000.0
    assert row["commission_cost"] == pytest.approx(expected_commission)
    assert row["slippage_cost"] == pytest.approx(expected_slippage)
    assert row["transaction_cost"] == pytest.approx(
        expected_commission + expected_slippage
    )


def test_missing_beta_defers_and_reconsiders_rebalance_on_later_valid_row() -> None:
    schedule, _, _, beta, result = _rebalancing_case(
        [1.0, 1.0, np.nan, 2.0, 2.0],
    )

    assert pd.isna(beta.loc[2])
    assert not bool(result.loc[2, "rebalance"])
    assert result.loc[2, "units_y"] == schedule.loc[2, "units_y"]
    assert not bool(result.loc[3, "rebalance"])
    assert bool(result.loc[4, "rebalance"])
    assert result.loc[4, "rebalance_beta"] == pytest.approx(2.0)


def test_future_beta_changes_do_not_leak_into_earlier_rebalancing() -> None:
    _, _, _, _, original = _rebalancing_case([1.0, 1.0, 1.1, 2.0, 3.0])
    _, _, _, _, changed = _rebalancing_case([1.0, 1.0, 1.1, 20.0, 30.0])

    pd.testing.assert_frame_equal(original.iloc[:3], changed.iloc[:3])


@pytest.mark.parametrize("invalid", [np.bool_(True), 1, 0, "true", None])
def test_rebalance_requires_actual_bool(invalid: Any) -> None:
    schedule, price_y, price_x, beta, _ = _rebalancing_case(
        [1.0, 1.0, 2.0],
        rebalance=False,
    )

    with pytest.raises(TypeError, match="rebalance"):
        calculate_rebalancing_costs(
            schedule,
            price_y,
            price_x,
            beta,
            2_000.0,
            rebalance=invalid,
        )


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "0.1", -0.1, np.nan, np.inf, -np.inf],
)
def test_invalid_rebalance_threshold_is_rejected(invalid: Any) -> None:
    schedule, price_y, price_x, beta, _ = _rebalancing_case(
        [1.0, 1.0, 2.0],
        rebalance=False,
    )

    with pytest.raises((TypeError, ValueError), match="rebalance_threshold"):
        calculate_rebalancing_costs(
            schedule,
            price_y,
            price_x,
            beta,
            2_000.0,
            rebalance=True,
            rebalance_threshold=invalid,
        )


def test_missing_required_prior_exposure_propagates_carry_and_net_unknown() -> None:
    events = ["ENTER_LONG", "NONE", "EXIT_TIME", "NONE", "NONE"]
    index = pd.RangeIndex(5, name="row")
    price_y = pd.Series(100.0, index=index)
    price_x = pd.Series([50.0, 50.0, np.nan, 50.0, 50.0], index=index)
    beta = pd.Series(1.0, index=index)
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        beta,
        2_000.0,
    )

    result = build_financed_pnl_schedule(
        schedule,
        price_y,
        price_x,
        10_000.0,
        borrow_rate_x=0.252,
        financing_rate=0.252,
    )

    assert pd.isna(result.loc[3, "borrow_cost"])
    assert pd.isna(result.loc[3, "financing_cost"])
    assert pd.isna(result.loc[3, "carry_cost"])
    assert pd.isna(result.loc[3, "gross_pnl"])
    assert pd.isna(result.loc[3, "net_pnl_after_carry"])
    assert pd.isna(result.loc[4, "cumulative_net_pnl_after_carry"])
    assert pd.isna(result.loc[4, "net_equity_after_carry"])


def test_financed_inputs_are_immutable_and_output_is_deterministic() -> None:
    schedule, price_y, price_x, beta, _ = _rebalancing_case(
        [1.0, 1.0, 2.0, 2.0],
        rebalance=False,
    )
    schedule_before = schedule.copy(deep=True)
    y_before = price_y.copy(deep=True)
    x_before = price_x.copy(deep=True)
    beta_before = beta.copy(deep=True)

    kwargs = {
        "initial_capital": 10_000.0,
        "commission_bps": 3.0,
        "slippage_bps": 2.0,
        "borrow_rate_x": 0.3,
        "financing_rate": 0.2,
        "rebalance": True,
        "hedge_ratio": beta,
        "target_gross_notional": 2_000.0,
        "rebalance_threshold": 0.5,
    }
    first = build_financed_pnl_schedule(schedule, price_y, price_x, **kwargs)
    second = build_financed_pnl_schedule(schedule, price_y, price_x, **kwargs)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(schedule, schedule_before)
    pd.testing.assert_series_equal(price_y, y_before)
    pd.testing.assert_series_equal(price_x, x_before)
    pd.testing.assert_series_equal(beta, beta_before)


def test_misaligned_dynamic_beta_and_borrow_rate_indices_are_rejected() -> None:
    schedule, price_y, price_x, beta, _ = _rebalancing_case(
        [1.0, 1.0, 2.0, 2.0],
        rebalance=False,
    )
    reversed_beta = beta.set_axis(beta.index[::-1])
    gross = build_pnl_schedule(schedule, price_y, price_x, 10_000.0)

    with pytest.raises(ValueError, match="hedge_ratio index"):
        calculate_rebalancing_costs(
            schedule,
            price_y,
            price_x,
            reversed_beta,
            2_000.0,
            rebalance=True,
        )
    with pytest.raises(ValueError, match="borrow_rate_x index"):
        calculate_borrow_costs(gross, borrow_rate_x=reversed_beta)


def test_duplicate_dynamic_rate_index_is_rejected() -> None:
    _, _, _, gross = _accounting_case()
    rate = pd.Series([0.1, 0.1, 0.1, 0.1], index=[0, 0, 1, 2])

    with pytest.raises(ValueError, match="unique index"):
        calculate_borrow_costs(gross, borrow_rate_x=rate)


def test_non_datetime_financed_index_is_preserved() -> None:
    index = pd.Index(["z", "a", "m", "b", "q"], name="carry_order")
    _, _, _, result = _constant_price_carry_case(index=index)

    pd.testing.assert_index_equal(result.index, index, exact=True)


def test_timezone_aware_financed_index_metadata_is_preserved() -> None:
    index = pd.date_range(
        "2026-10-23",
        periods=5,
        freq="h",
        tz="Europe/London",
        name="carry_time",
    )
    _, _, _, result = _constant_price_carry_case(index=index)

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result.index.tz == index.tz
    assert result.index.freq == index.freq
    assert result.index.name == index.name


def test_future_price_beta_rate_and_state_changes_do_not_affect_prior_rows() -> None:
    index = pd.RangeIndex(7)
    original_events = [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "NONE",
        "EXIT_TIME",
        "NONE",
        "NONE",
    ]
    changed_events = [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "EXIT_STOP",
        "ENTER_SHORT",
        "NONE",
        "NONE",
    ]
    original_y = pd.Series(100.0, index=index)
    original_x = pd.Series(50.0, index=index)
    original_beta = pd.Series([1.0, 1.0, 1.1, 2.0, 2.0, 2.0, 2.0], index=index)
    changed_y = original_y.copy(deep=True)
    changed_x = original_x.copy(deep=True)
    changed_beta = original_beta.copy(deep=True)
    changed_y.iloc[3:] *= 1.5
    changed_x.iloc[3:] *= 0.75
    changed_beta.iloc[3:] = [20.0, 20.0, 20.0, 20.0]
    original_rates = pd.Series(0.2, index=index)
    changed_rates = original_rates.copy(deep=True)
    changed_rates.iloc[3:] = 5.0
    original_schedule = build_position_schedule(
        original_y,
        original_x,
        _signals_from_events(original_events, index),
        original_beta,
        2_000.0,
    )
    changed_schedule = build_position_schedule(
        changed_y,
        changed_x,
        _signals_from_events(changed_events, index),
        changed_beta,
        2_000.0,
    )
    common = {
        "initial_capital": 10_000.0,
        "borrow_rate_x": original_rates,
        "financing_rate": 0.1,
        "rebalance": True,
        "target_gross_notional": 2_000.0,
        "rebalance_threshold": 0.5,
    }
    original = build_financed_pnl_schedule(
        original_schedule,
        original_y,
        original_x,
        hedge_ratio=original_beta,
        **common,
    )
    changed = build_financed_pnl_schedule(
        changed_schedule,
        changed_y,
        changed_x,
        hedge_ratio=changed_beta,
        **{**common, "borrow_rate_x": changed_rates},
    )

    pd.testing.assert_frame_equal(original.iloc[:3], changed.iloc[:3])


def _completed_trade_case(
    *,
    side: str = "LONG",
    index: pd.Index | None = None,
    beta_values: list[float] | None = None,
    rebalance: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Return one fully financed completed trade and its ledger."""
    entry_event = "ENTER_LONG" if side == "LONG" else "ENTER_SHORT"
    events = [entry_event, "NONE", "NONE", "EXIT_TIME", "NONE", "NONE"]
    if index is None:
        index = pd.RangeIndex(len(events), name="ledger_row")
    price_y = pd.Series(
        [100.0, 100.0, 102.0, 104.0, 105.0, 106.0],
        index=index,
        name="Y",
    )
    price_x = pd.Series(
        [50.0, 50.0, 49.0, 48.0, 47.0, 46.0],
        index=index,
        name="X",
    )
    if beta_values is None:
        beta_values = [1.0] * len(events)
    beta = pd.Series(beta_values, index=index, name="beta")
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        beta,
        target_gross_notional=2_000.0,
    )
    accounting = build_financed_pnl_schedule(
        schedule,
        price_y,
        price_x,
        initial_capital=10_000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=0.25,
        borrow_rate_y=0.252,
        borrow_rate_x=0.504,
        financing_rate=0.126,
        periods_per_year=252,
        rebalance=rebalance,
        hedge_ratio=beta,
        target_gross_notional=2_000.0,
        rebalance_threshold=0.5,
    )
    ledger = build_trade_ledger(accounting, schedule)
    return schedule, price_y, price_x, beta, accounting, ledger


def _forced_liquidation_case(
    *,
    force_liquidation: bool = True,
    index: pd.Index | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Return original/possibly forced schedules, accounting, and ledger."""
    events = ["ENTER_LONG", "NONE", "NONE", "NONE", "NONE"]
    if index is None:
        index = pd.RangeIndex(len(events), name="forced_row")
    price_y = pd.Series(
        [100.0, 100.0, 101.0, 103.0, 106.0],
        index=index,
        name="Y",
    )
    price_x = pd.Series(
        [50.0, 50.0, 49.0, 48.0, 47.0],
        index=index,
        name="X",
    )
    beta = pd.Series(1.0, index=index, name="beta")
    original = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        beta,
        target_gross_notional=2_000.0,
    )
    closed = force_liquidate_open_position(
        original,
        price_y,
        price_x,
        beta,
        force_liquidation=force_liquidation,
    )
    accounting = build_financed_pnl_schedule(
        closed,
        price_y,
        price_x,
        initial_capital=10_000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=0.25,
        borrow_rate_x=0.252,
        financing_rate=0.126,
        periods_per_year=252,
    )
    ledger = build_trade_ledger(accounting, closed)
    return original, closed, price_y, price_x, beta, accounting, ledger


def test_completed_long_trade_has_complete_lifecycle_and_endpoint_fields() -> None:
    schedule, _, _, beta, accounting, ledger = _completed_trade_case()

    assert len(ledger) == 1
    trade = ledger.iloc[0]
    assert trade["trade_id"] == 1
    assert trade["side"] == "LONG_SPREAD"
    assert trade["entry_row"] == 1
    assert trade["exit_row"] == 4
    assert trade["entry_index"] == accounting.index[1]
    assert trade["exit_index"] == accounting.index[4]
    assert trade["entry_event"] == "ENTER_LONG"
    assert trade["exit_event"] == "EXIT_TIME"
    assert trade["exit_reason"] == TradeExitReason.TIME.value
    assert trade["entry_price_y"] == accounting["price_y"].iat[1]
    assert trade["entry_price_x"] == accounting["price_x"].iat[1]
    assert trade["exit_price_y"] == accounting["price_y"].iat[4]
    assert trade["exit_price_x"] == accounting["price_x"].iat[4]
    assert trade["entry_units_y"] == accounting["units_y"].iat[1]
    assert trade["entry_units_x"] == accounting["units_x"].iat[1]
    assert trade["exit_units_y"] == accounting["units_y"].iat[3]
    assert trade["exit_units_x"] == accounting["units_x"].iat[3]
    assert trade["entry_hedge_ratio"] == beta.iat[1]
    assert trade["exit_hedge_ratio"] == beta.iat[4]
    assert trade["holding_period_rows"] == 3
    assert trade["entry_gross_notional"] == pytest.approx(2_000.0)
    assert not trade["forced_exit"]
    assert schedule["executed_state"].iat[4] == "FLAT"


def test_completed_short_trade_produces_one_short_ledger_record() -> None:
    _, _, _, _, accounting, ledger = _completed_trade_case(side="SHORT")

    assert len(ledger) == 1
    assert ledger.loc[0, "side"] == "SHORT_SPREAD"
    assert ledger.loc[0, "entry_event"] == "ENTER_SHORT"
    assert ledger.loc[0, "entry_units_y"] < 0.0
    assert ledger.loc[0, "entry_units_x"] > 0.0
    assert ledger.loc[0, "exit_units_y"] == accounting["units_y"].iat[3]
    assert ledger.loc[0, "exit_units_x"] == accounting["units_x"].iat[3]


def test_multiple_trades_receive_sequential_deterministic_ids() -> None:
    events = [
        "ENTER_LONG",
        "NONE",
        "EXIT_TIME",
        "NONE",
        "ENTER_SHORT",
        "NONE",
        "NONE",
        "EXIT_STOP",
        "NONE",
        "NONE",
    ]
    index = pd.RangeIndex(len(events), name="multi_row")
    price_y = pd.Series(100.0 + np.arange(len(events)), index=index)
    price_x = pd.Series(50.0 - 0.5 * np.arange(len(events)), index=index)
    beta = pd.Series(1.0, index=index)
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        beta,
        2_000.0,
    )
    accounting = build_financed_pnl_schedule(schedule, price_y, price_x, 10_000.0)

    first = build_trade_ledger(accounting, schedule)
    second = build_trade_ledger(accounting, schedule)

    assert first["trade_id"].tolist() == [1, 2]
    assert first["side"].tolist() == ["LONG_SPREAD", "SHORT_SPREAD"]
    assert first["entry_row"].tolist() == [1, 5]
    assert first["exit_row"].tolist() == [3, 8]
    pd.testing.assert_frame_equal(first, second)


def test_trade_cost_pnl_carry_and_return_attribution_are_exact() -> None:
    _, _, _, _, accounting, ledger = _completed_trade_case()
    trade = ledger.iloc[0]
    attributed = accounting.iloc[1:5]

    assert trade["gross_pnl"] == pytest.approx(attributed["gross_pnl"].sum())
    assert trade["commission_cost"] == pytest.approx(
        attributed["commission_cost"].sum()
    )
    assert trade["slippage_cost"] == pytest.approx(
        attributed["slippage_cost"].sum()
    )
    assert trade["transaction_cost"] == pytest.approx(
        attributed["transaction_cost"].sum()
    )
    assert trade["borrow_cost"] == pytest.approx(attributed["borrow_cost"].sum())
    assert trade["financing_cost"] == pytest.approx(
        attributed["financing_cost"].sum()
    )
    assert trade["carry_cost"] == pytest.approx(
        trade["borrow_cost"] + trade["financing_cost"]
    )
    assert trade["total_cost"] == pytest.approx(
        trade["transaction_cost"] + trade["carry_cost"]
    )
    assert trade["net_pnl"] == pytest.approx(
        trade["gross_pnl"] - trade["total_cost"]
    )
    assert trade["return_on_entry_gross_notional"] == pytest.approx(
        trade["net_pnl"] / trade["entry_gross_notional"]
    )


def test_rebalance_stays_in_one_trade_and_its_cost_is_attributed() -> None:
    _, _, _, _, accounting, ledger = _completed_trade_case(
        beta_values=[1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        rebalance=True,
    )

    assert accounting["rebalance"].sum() == 1
    rebalance_row = int(np.flatnonzero(accounting["rebalance"].to_numpy())[0])
    assert rebalance_row == 3
    assert len(ledger) == 1
    assert ledger.loc[0, "transaction_cost"] == pytest.approx(
        accounting.loc[1:4, "transaction_cost"].sum()
    )
    assert accounting["transaction_cost"].iat[rebalance_row] > 0.0
    assert ledger.loc[0, "transaction_cost"] > (
        accounting["transaction_cost"].iat[1]
        + accounting["transaction_cost"].iat[4]
    )


def test_flat_rows_between_or_after_trades_do_not_leak_costs() -> None:
    schedule, _, _, _, accounting, expected = _completed_trade_case()
    altered = accounting.copy(deep=True)
    final_row = len(altered) - 1
    altered.iat[
        final_row, altered.columns.get_loc("commission_cost")
    ] += 7.0
    altered.iat[
        final_row, altered.columns.get_loc("transaction_cost")
    ] += 7.0
    altered.iat[
        final_row, altered.columns.get_loc("net_pnl_after_carry")
    ] -= 7.0
    altered.iat[
        final_row,
        altered.columns.get_loc("cumulative_net_pnl_after_carry"),
    ] -= 7.0

    actual = build_trade_ledger(altered, schedule)

    pd.testing.assert_frame_equal(actual, expected)


def test_open_final_trade_is_omitted_when_liquidation_is_disabled() -> None:
    original, unchanged, _, _, _, accounting, ledger = _forced_liquidation_case(
        force_liquidation=False
    )

    pd.testing.assert_frame_equal(unchanged, original)
    assert ledger.empty
    assert accounting["units_y"].iat[-1] != 0.0
    assert accounting["units_x"].iat[-1] != 0.0


def test_forced_liquidation_closes_final_row_and_marks_completed_trade() -> None:
    original, closed, _, _, _, accounting, ledger = _forced_liquidation_case()

    assert original["executed_state"].iat[-1] == "LONG_SPREAD"
    assert closed["executed_state"].iat[-1] == "FLAT"
    assert closed["execution_event"].iat[-1] == "FORCED_EXIT"
    assert closed["units_y"].iat[-1] == 0.0
    assert closed["units_x"].iat[-1] == 0.0
    assert accounting["units_y"].iat[-1] == 0.0
    assert accounting["units_x"].iat[-1] == 0.0
    assert len(ledger) == 1
    assert bool(ledger.loc[0, "forced_exit"])
    assert ledger.loc[0, "exit_event"] == "FORCED_EXIT"
    assert ledger.loc[0, "exit_reason"] == TradeExitReason.END_OF_BACKTEST.value


def test_forced_close_charges_normal_costs_and_preserves_final_interval_pnl() -> None:
    original, closed, price_y, price_x, _, forced, ledger = (
        _forced_liquidation_case()
    )
    open_accounting = build_financed_pnl_schedule(
        original,
        price_y,
        price_x,
        initial_capital=10_000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=0.25,
        borrow_rate_x=0.252,
        financing_rate=0.126,
        periods_per_year=252,
    )

    assert forced["gross_pnl"].iat[-1] == pytest.approx(
        open_accounting["gross_pnl"].iat[-1]
    )
    assert forced["transaction_cost"].iat[-1] > 0.0
    assert forced["delta_units_y"].iat[-1] == pytest.approx(
        -closed["units_y"].iat[-2]
    )
    assert forced["delta_units_x"].iat[-1] == pytest.approx(
        -closed["units_x"].iat[-2]
    )
    expected_net = (
        ledger.loc[0, "gross_pnl"]
        - ledger.loc[0, "transaction_cost"]
        - ledger.loc[0, "carry_cost"]
    )
    assert ledger.loc[0, "net_pnl"] == pytest.approx(expected_net)
    assert ledger.loc[0, "total_cost"] == pytest.approx(
        ledger.loc[0, "transaction_cost"] + ledger.loc[0, "carry_cost"]
    )


@pytest.mark.parametrize("missing_input", ["price_y", "price_x", "hedge_ratio"])
def test_missing_final_market_input_prevents_forced_liquidation(
    missing_input: str,
) -> None:
    events = ["ENTER_LONG", "NONE", "NONE", "NONE"]
    index = pd.RangeIndex(len(events))
    price_y = pd.Series([100.0, 100.0, 101.0, 102.0], index=index)
    price_x = pd.Series([50.0, 50.0, 49.0, 48.0], index=index)
    beta = pd.Series(1.0, index=index)
    if missing_input == "price_y":
        price_y.iat[-1] = np.nan
    elif missing_input == "price_x":
        price_x.iat[-1] = np.nan
    else:
        beta.iat[-2] = np.nan
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        beta,
        2_000.0,
    )

    with pytest.raises(ValueError, match="Final prices|Final hedge ratio"):
        force_liquidate_open_position(schedule)


def test_forward_filled_final_price_cannot_force_liquidation() -> None:
    original, _, price_y, price_x, beta, _, _ = _forced_liquidation_case(
        force_liquidation=False
    )
    observed_y = pd.Series(True, index=price_y.index)
    observed_y.iat[-1] = False

    with pytest.raises(ValueError, match="genuine observed"):
        force_liquidate_open_position(
            original,
            price_y,
            price_x,
            beta,
            observed_y=observed_y,
        )


def test_price_metadata_false_blocks_forced_close_despite_true_schedule_mask() -> None:
    original, _, price_y, price_x, beta, _, _ = _forced_liquidation_case(
        force_liquidation=False
    )
    marked_y = price_y.copy(deep=True)
    metadata_mask = pd.Series(True, index=marked_y.index)
    metadata_mask.iat[-1] = False
    marked_y.attrs[OBSERVED_PRICE_MASK_ATTR] = metadata_mask

    assert bool(original["observed_y"].iat[-1])
    with pytest.raises(ValueError, match="genuine observed"):
        force_liquidate_open_position(
            original,
            marked_y,
            price_x,
            beta,
        )


def test_explicit_false_mask_blocks_forced_close_when_other_sources_are_true() -> None:
    original, _, price_y, price_x, beta, _, _ = _forced_liquidation_case(
        force_liquidation=False
    )
    metadata_mask = pd.Series(True, index=price_y.index)
    price_y.attrs[OBSERVED_PRICE_MASK_ATTR] = metadata_mask
    explicit_mask = metadata_mask.copy(deep=True)
    explicit_mask.iat[-1] = False

    with pytest.raises(ValueError, match="genuine observed"):
        force_liquidate_open_position(
            original,
            price_y,
            price_x,
            beta,
            observed_y=explicit_mask,
        )


def test_all_true_provenance_sources_allow_forced_close() -> None:
    original, _, price_y, price_x, beta, _, _ = _forced_liquidation_case(
        force_liquidation=False
    )
    all_true = pd.Series(True, index=price_y.index)
    price_y.attrs[OBSERVED_PRICE_MASK_ATTR] = all_true.copy(deep=True)

    result = force_liquidate_open_position(
        original,
        price_y,
        price_x,
        beta,
        observed_y=all_true,
    )

    assert bool(original["observed_y"].iat[-1])
    assert result["execution_event"].iat[-1] == "FORCED_EXIT"
    assert result["executed_state"].iat[-1] == "FLAT"


def test_misaligned_price_metadata_mask_is_rejected() -> None:
    price_y, price_x, signals, _ = _market_inputs(["ENTER_LONG", "NONE", "NONE"])
    price_y.attrs[OBSERVED_PRICE_MASK_ATTR] = pd.Series(
        True,
        index=price_y.index[::-1],
    )

    with pytest.raises(ValueError, match="price metadata index"):
        build_position_schedule(
            price_y,
            price_x,
            signals,
            1.0,
            2_000.0,
        )


@pytest.mark.parametrize("entry_event", ["ENTER_LONG", "ENTER_SHORT"])
def test_standalone_forced_liquidation_suppresses_terminal_entry(
    entry_event: str,
) -> None:
    events = [entry_event, "NONE"]
    price_y, price_x, signals, beta = _market_inputs(events)
    schedule = build_position_schedule(
        price_y,
        price_x,
        signals,
        beta,
        2_000.0,
    )

    suppressed = force_liquidate_open_position(
        schedule,
        price_y,
        price_x,
        beta,
    )
    accounting = build_financed_pnl_schedule(
        suppressed,
        price_y,
        price_x,
        10_000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=1.0,
    )
    ledger = build_trade_ledger(accounting, suppressed)

    assert schedule["execution_event"].iat[-1] == entry_event
    pd.testing.assert_frame_equal(suppressed.iloc[:-1], schedule.iloc[:-1])
    assert suppressed["executed_state"].iat[-1] == "FLAT"
    assert suppressed["execution_event"].iat[-1] == "NONE"
    assert suppressed[["units_y", "units_x"]].iloc[-1].eq(0.0).all()
    assert not suppressed["execution_event"].eq("FORCED_EXIT").any()
    assert accounting["transaction_cost"].eq(0.0).all()
    assert ledger.empty


def test_fully_closed_ledger_reconciles_to_final_after_carry_accounting() -> None:
    schedule, _, _, _, accounting, ledger = _completed_trade_case()

    result = reconcile_trade_ledger(ledger, accounting, schedule)

    assert isinstance(result, LedgerReconciliation)
    assert result.status == "RECONCILED"
    assert result.completed_trade_count == 1
    assert not result.has_open_trade
    assert result.fully_reconcilable
    assert result.completed_totals_match
    assert result.final_accounting_match is True
    assert result.gross_pnl_match is True
    assert result.transaction_cost_match is True
    assert result.carry_cost_match is True
    assert result.net_pnl_match is True
    assert result.ledger_net_pnl == pytest.approx(
        accounting["cumulative_net_pnl_after_carry"].iat[-1]
    )


def test_forced_liquidation_reconciles_closing_cost_with_final_equity() -> None:
    _, closed, _, _, _, accounting, ledger = _forced_liquidation_case()

    result = reconcile_trade_ledger(ledger, accounting, closed)

    assert result.status == "RECONCILED"
    assert result.final_accounting_match is True
    assert accounting["net_equity_after_carry"].iat[-1] == pytest.approx(
        10_000.0 + ledger["net_pnl"].sum()
    )


def test_open_trade_reconciliation_reports_incomplete_accounting_separately() -> None:
    _, open_schedule, _, _, _, accounting, ledger = _forced_liquidation_case(
        force_liquidation=False
    )

    result = reconcile_trade_ledger(ledger, accounting, open_schedule)

    assert result.status == "OPEN_TRADE"
    assert result.has_open_trade
    assert result.completed_trade_count == 0
    assert result.completed_totals_match
    assert result.final_accounting_match is None
    assert result.open_trade_transaction_cost > 0.0
    assert result.open_trade_net_pnl == pytest.approx(
        accounting["net_pnl_after_carry"].iloc[1:].sum()
    )


def test_unknown_completed_trade_pnl_remains_nan_and_is_not_reconciled() -> None:
    events = ["ENTER_LONG", "NONE", "NONE", "EXIT_TIME", "NONE", "NONE"]
    index = pd.RangeIndex(len(events))
    price_y = pd.Series([100.0, 100.0, np.nan, 103.0, 104.0, 105.0], index=index)
    price_x = pd.Series([50.0, 50.0, 49.0, 48.0, 47.0, 46.0], index=index)
    beta = pd.Series(1.0, index=index)
    schedule = build_position_schedule(
        price_y,
        price_x,
        _signals_from_events(events, index),
        beta,
        2_000.0,
    )
    accounting = build_financed_pnl_schedule(
        schedule,
        price_y,
        price_x,
        10_000.0,
        commission_bps=5.0,
        borrow_rate_x=0.1,
        financing_rate=0.05,
    )
    ledger = build_trade_ledger(accounting, schedule)

    assert len(ledger) == 1
    assert np.isnan(ledger.loc[0, "gross_pnl"])
    assert np.isnan(ledger.loc[0, "net_pnl"])
    assert np.isfinite(ledger.loc[0, "transaction_cost"])
    result = reconcile_trade_ledger(ledger, accounting, schedule)
    assert result.status == "UNKNOWN_ACCOUNTING"
    assert not result.fully_reconcilable
    assert result.gross_pnl_match is None
    assert result.net_pnl_match is None
    assert result.final_accounting_match is None


def test_ledger_rejects_malformed_executed_state_transitions() -> None:
    schedule, _, _, _, accounting, _ = _completed_trade_case()
    malformed = schedule.copy(deep=True)
    malformed.iat[2, malformed.columns.get_loc("execution_event")] = "ENTER_SHORT"

    with pytest.raises(ValueError, match="Invalid|changed|transition"):
        build_trade_ledger(accounting, malformed)


def test_ledger_rejects_duplicate_and_misaligned_indices() -> None:
    schedule, _, _, _, accounting, _ = _completed_trade_case()
    duplicated = accounting.copy(deep=True)
    duplicated.index = pd.Index([0, 0, 1, 2, 3, 4])
    reordered = schedule.iloc[::-1].copy()

    with pytest.raises(ValueError, match="unique index"):
        build_trade_ledger(duplicated, schedule)
    with pytest.raises(ValueError, match="position_schedule index"):
        build_trade_ledger(accounting, reordered)


@pytest.mark.parametrize("invalid", [np.bool_(True), 1, 0, "true", None])
def test_forced_liquidation_requires_actual_bool(invalid: Any) -> None:
    original, _, price_y, price_x, beta, _, _ = _forced_liquidation_case(
        force_liquidation=False
    )

    with pytest.raises(TypeError, match="force_liquidation"):
        force_liquidate_open_position(
            original,
            price_y,
            price_x,
            beta,
            force_liquidation=invalid,
        )


def test_ledger_and_liquidation_are_immutable_and_deterministic() -> None:
    schedule, price_y, price_x, beta, accounting, _ = _completed_trade_case()
    schedule_before = schedule.copy(deep=True)
    accounting_before = accounting.copy(deep=True)

    first = build_trade_ledger(accounting, schedule)
    second = build_trade_ledger(accounting, schedule)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(schedule, schedule_before)
    pd.testing.assert_frame_equal(accounting, accounting_before)

    _, open_schedule, _, _, _, _, _ = _forced_liquidation_case(
        force_liquidation=False
    )
    open_before = open_schedule.copy(deep=True)
    liquidated_first = force_liquidate_open_position(
        open_schedule, price_y.iloc[:5], price_x.iloc[:5], beta.iloc[:5]
    )
    liquidated_second = force_liquidate_open_position(
        open_schedule, price_y.iloc[:5], price_x.iloc[:5], beta.iloc[:5]
    )
    pd.testing.assert_frame_equal(liquidated_first, liquidated_second)
    pd.testing.assert_frame_equal(open_schedule, open_before)


def test_trade_record_and_reconciliation_structures_are_frozen() -> None:
    schedule, _, _, _, accounting, ledger = _completed_trade_case()
    record = TradeRecord(**ledger.iloc[0].to_dict())
    reconciliation = reconcile_trade_ledger(ledger, accounting, schedule)

    with pytest.raises(FrozenInstanceError):
        record.trade_id = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        reconciliation.status = "MISMATCH"  # type: ignore[misc]


@pytest.mark.parametrize(
    "index",
    [
        pd.Index(["z", "a", "m", "b", "q", "x"], name="ledger_order"),
        pd.date_range(
            "2026-10-25",
            periods=6,
            freq="h",
            tz="Europe/London",
            name="ledger_time",
        ),
    ],
)
def test_ledger_preserves_non_datetime_and_timezone_aware_index_values(
    index: pd.Index,
) -> None:
    _, _, _, _, _, ledger = _completed_trade_case(index=index)

    assert ledger.loc[0, "entry_index"] == index[1]
    assert ledger.loc[0, "exit_index"] == index[4]
    if isinstance(index, pd.DatetimeIndex):
        assert str(ledger.loc[0, "entry_index"].tz) == str(index.tz)
        assert str(ledger.loc[0, "exit_index"].tz) == str(index.tz)


def test_future_rows_do_not_change_an_already_completed_trade_record() -> None:
    events = [
        "ENTER_LONG",
        "NONE",
        "EXIT_TIME",
        "NONE",
        "NONE",
        "NONE",
        "NONE",
        "NONE",
    ]
    index = pd.RangeIndex(len(events))
    original_y = pd.Series([100.0, 100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0], index=index)
    original_x = pd.Series([50.0, 50.0, 49.0, 48.0, 47.0, 46.0, 45.0, 44.0], index=index)
    original_beta = pd.Series(1.0, index=index)
    changed_y = original_y.copy(deep=True)
    changed_x = original_x.copy(deep=True)
    changed_beta = original_beta.copy(deep=True)
    changed_y.iloc[4:] *= 10.0
    changed_x.iloc[4:] *= 0.1
    changed_beta.iloc[4:] = 5.0
    changed_events = events.copy()
    changed_events[4:] = ["ENTER_SHORT", "NONE", "EXIT_STOP", "NONE"]

    original_schedule = build_position_schedule(
        original_y,
        original_x,
        _signals_from_events(events, index),
        original_beta,
        2_000.0,
    )
    changed_schedule = build_position_schedule(
        changed_y,
        changed_x,
        _signals_from_events(changed_events, index),
        changed_beta,
        2_000.0,
    )
    original_accounting = build_financed_pnl_schedule(
        original_schedule,
        original_y,
        original_x,
        10_000.0,
        commission_bps=5.0,
        financing_rate=0.05,
    )
    changed_accounting = build_financed_pnl_schedule(
        changed_schedule,
        changed_y,
        changed_x,
        10_000.0,
        commission_bps=5.0,
        financing_rate=0.05,
    )

    original_ledger = build_trade_ledger(original_accounting, original_schedule)
    changed_ledger = build_trade_ledger(changed_accounting, changed_schedule)

    pd.testing.assert_series_equal(
        original_ledger.iloc[0],
        changed_ledger.iloc[0],
        check_names=False,
    )


def _integrated_case(
    *,
    side: str = "LONG",
    complete: bool = True,
    force_liquidation: bool = False,
    rebalance: bool = False,
    index: pd.Index | None = None,
    missing_price_row: int | None = None,
) -> tuple[dict[str, Any], BacktestResult]:
    """Run a compact deterministic end-to-end pair backtest."""
    count = 6 if complete else 5
    if index is None:
        index = pd.RangeIndex(count, name="integrated_row")
    price_y = pd.Series(
        [100.0, 100.0, 101.0, 103.0, 105.0, 106.0][:count],
        index=index,
        name="Y",
    )
    price_x = pd.Series(
        [50.0, 50.0, 49.0, 48.0, 47.0, 46.0][:count],
        index=index,
        name="X",
    )
    if missing_price_row is not None:
        price_y.iat[missing_price_row] = np.nan
    if side == "LONG":
        z_values = [0.0, -2.5, -1.0, -0.2, 0.0, 0.0][:count]
    else:
        z_values = [0.0, 2.5, 1.0, 0.2, 0.0, 0.0][:count]
    if not complete:
        z_values = [0.0, -2.5, -1.0, -1.0, -1.0]
    zscore = pd.Series(z_values, index=index, name="zscore")
    if rebalance:
        beta_values = [1.0, 1.0, 2.0, 2.0, 2.0, 2.0][:count]
    else:
        beta_values = [1.0] * count
    beta = pd.Series(beta_values, index=index, name="beta")
    borrow_rate_y = pd.Series(0.252, index=index, name="borrow_y")
    borrow_rate_x = pd.Series(0.504, index=index, name="borrow_x")
    inputs: dict[str, Any] = {
        "price_y": price_y,
        "price_x": price_x,
        "zscore": zscore,
        "hedge_ratio": beta,
        "borrow_rate_y": borrow_rate_y,
        "borrow_rate_x": borrow_rate_x,
    }
    result = run_pair_backtest(
        price_y,
        price_x,
        beta,
        2_000.0,
        zscore=zscore,
        initial_capital=10_000.0,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        execution_lag=1,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=0.25,
        borrow_rate_y=borrow_rate_y,
        borrow_rate_x=borrow_rate_x,
        financing_rate=0.126,
        periods_per_year=252,
        rebalance=rebalance,
        rebalance_threshold=0.5,
        force_liquidation=force_liquidation,
    )
    return inputs, result


def test_integrated_long_trade_obeys_lag_pnl_and_cost_timing() -> None:
    _, result = _integrated_case()

    assert result.signals["event"].tolist() == [
        "NONE",
        "ENTER_LONG",
        "NONE",
        "EXIT_MEAN_REVERSION",
        "NONE",
        "NONE",
    ]
    assert result.positions["execution_event"].tolist() == [
        "NONE",
        "NONE",
        "ENTER_LONG",
        "NONE",
        "EXIT_MEAN_REVERSION",
        "NONE",
    ]
    assert result.accounting["gross_pnl"].iat[2] == 0.0
    expected_exit_pnl = (
        result.positions["units_y"].iat[3]
        * (
            result.accounting["price_y"].iat[4]
            - result.accounting["price_y"].iat[3]
        )
        + result.positions["units_x"].iat[3]
        * (
            result.accounting["price_x"].iat[4]
            - result.accounting["price_x"].iat[3]
        )
    )
    assert result.accounting["gross_pnl"].iat[4] == pytest.approx(
        expected_exit_pnl
    )
    assert np.flatnonzero(
        result.accounting["transaction_cost"].to_numpy() > 0.0
    ).tolist() == [2, 4]
    assert result.accounting["borrow_cost_y"].eq(0.0).all()
    assert result.accounting["borrow_cost_x"].iloc[[3, 4]].gt(0.0).all()
    assert result.accounting["financing_cost"].iat[2] == 0.0
    assert result.accounting["financing_cost"].iloc[[3, 4]].gt(0.0).all()
    assert len(result.ledger) == 1
    assert result.ledger.loc[0, "side"] == "LONG_SPREAD"
    assert result.reconciliation.status == "RECONCILED"


def test_integrated_short_trade_charges_only_short_y_borrow() -> None:
    _, result = _integrated_case(side="SHORT")

    assert result.positions["execution_event"].iat[2] == "ENTER_SHORT"
    assert result.positions["execution_event"].iat[4] == "EXIT_MEAN_REVERSION"
    assert result.accounting["borrow_cost_x"].eq(0.0).all()
    assert result.accounting["borrow_cost_y"].iloc[[3, 4]].gt(0.0).all()
    assert len(result.ledger) == 1
    assert result.ledger.loc[0, "side"] == "SHORT_SPREAD"
    assert result.reconciliation.status == "RECONCILED"


def test_precomputed_signals_and_generated_signals_compose_identically() -> None:
    inputs, generated = _integrated_case()

    precomputed = run_pair_backtest(
        inputs["price_y"],
        inputs["price_x"],
        inputs["hedge_ratio"],
        2_000.0,
        signals=generated.signals,
        initial_capital=10_000.0,
        execution_lag=1,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=0.25,
        borrow_rate_y=inputs["borrow_rate_y"],
        borrow_rate_x=inputs["borrow_rate_x"],
        financing_rate=0.126,
        periods_per_year=252,
    )

    pd.testing.assert_frame_equal(precomputed.signals, generated.signals)
    pd.testing.assert_frame_equal(precomputed.positions, generated.positions)
    pd.testing.assert_frame_equal(precomputed.accounting, generated.accounting)
    pd.testing.assert_frame_equal(precomputed.ledger, generated.ledger)
    assert precomputed.reconciliation == generated.reconciliation


def test_integrated_rebalance_stays_in_trade_and_uses_existing_costs() -> None:
    _, result = _integrated_case(rebalance=True)

    assert result.positions["rebalance"].sum() == 1
    rebalance_row = int(
        np.flatnonzero(result.positions["rebalance"].to_numpy())[0]
    )
    assert rebalance_row == 3
    assert result.positions["execution_event"].iat[rebalance_row] == "NONE"
    assert result.positions["executed_state"].iat[rebalance_row] == "LONG_SPREAD"
    assert result.accounting["transaction_cost"].iat[rebalance_row] > 0.0
    assert len(result.ledger) == 1
    assert result.ledger.loc[0, "transaction_cost"] == pytest.approx(
        result.accounting["transaction_cost"].iloc[2:5].sum()
    )


def test_integrated_forced_liquidation_changes_only_final_row_and_equity() -> None:
    _, open_result = _integrated_case(complete=False, force_liquidation=False)
    _, forced_result = _integrated_case(complete=False, force_liquidation=True)

    pd.testing.assert_frame_equal(
        forced_result.positions.iloc[:-1],
        open_result.positions.iloc[:-1],
    )
    pd.testing.assert_frame_equal(
        forced_result.accounting.iloc[:-1],
        open_result.accounting.iloc[:-1],
    )
    assert open_result.positions["executed_state"].iat[-1] == "LONG_SPREAD"
    assert open_result.reconciliation.status == "OPEN_TRADE"
    assert not open_result.forced_liquidation_applied
    assert forced_result.positions["executed_state"].iat[-1] == "FLAT"
    assert forced_result.positions["execution_event"].iat[-1] == "FORCED_EXIT"
    assert forced_result.positions[["units_y", "units_x"]].iloc[-1].eq(0.0).all()
    assert forced_result.forced_liquidation_applied
    assert forced_result.accounting["gross_pnl"].iat[-1] == pytest.approx(
        open_result.accounting["gross_pnl"].iat[-1]
    )
    closing_cost = forced_result.accounting["transaction_cost"].iat[-1]
    assert closing_cost > 0.0
    assert forced_result.accounting["net_equity_after_carry"].iat[-1] == pytest.approx(
        open_result.accounting["net_equity_after_carry"].iat[-1] - closing_cost
    )
    assert forced_result.reconciliation.status == "RECONCILED"
    assert bool(forced_result.ledger.loc[0, "forced_exit"])


def test_terminal_entry_is_suppressed_when_forced_liquidation_is_enabled() -> None:
    events = ["NONE", "ENTER_LONG", "NONE"]
    price_y, price_x, signals, _ = _market_inputs(events)

    result = run_pair_backtest(
        price_y,
        price_x,
        1.0,
        2_000.0,
        signals=signals,
        force_liquidation=True,
        commission_bps=10.0,
        slippage_bps=5.0,
    )

    assert result.positions["executed_state"].eq("FLAT").all()
    assert result.positions["execution_event"].eq("NONE").all()
    assert result.accounting["transaction_cost"].eq(0.0).all()
    assert result.ledger.empty
    assert result.forced_liquidation_requested
    assert not result.forced_liquidation_applied


def test_runner_exposes_research_mode_causality_and_sizing_warnings() -> None:
    _, result = _integrated_case()
    metadata = result.research_metadata

    assert metadata.upstream_inputs_assumed_causal
    assert not metadata.upstream_provenance_validated
    assert "does not validate" in metadata.warning
    assert "genuine observed" in metadata.price_policy
    assert "after 1 row" in metadata.hedge_ratio_policy
    assert metadata.sizing_policy == "beta_weighted_gross_notional"
    assert "not dollar-neutral unless beta == 1" in metadata.dollar_neutrality_note


def test_integrated_missing_held_mark_reports_unknown_accounting() -> None:
    _, result = _integrated_case(missing_price_row=3)

    assert np.isnan(result.accounting["gross_pnl"].iat[3])
    assert np.isnan(result.accounting["net_pnl_after_carry"].iat[3])
    assert np.isnan(result.ledger.loc[0, "gross_pnl"])
    assert np.isnan(result.ledger.loc[0, "net_pnl"])
    assert result.reconciliation.status == "UNKNOWN_ACCOUNTING"


def test_integrated_accounting_and_exposure_invariants_hold() -> None:
    _, result = _integrated_case(rebalance=True)
    accounting = result.accounting

    np.testing.assert_allclose(
        accounting["gross_pnl"],
        accounting["pnl_y"] + accounting["pnl_x"],
    )
    np.testing.assert_allclose(
        accounting["transaction_cost"],
        accounting["commission_cost"] + accounting["slippage_cost"],
    )
    np.testing.assert_allclose(
        accounting["carry_cost"],
        accounting["borrow_cost"] + accounting["financing_cost"],
    )
    np.testing.assert_allclose(
        accounting["net_pnl_after_carry"],
        accounting["gross_pnl"]
        - accounting["transaction_cost"]
        - accounting["carry_cost"],
    )
    assert accounting["cumulative_transaction_cost"].diff().dropna().ge(0.0).all()
    assert accounting["cumulative_carry_cost"].diff().dropna().ge(0.0).all()
    flat = result.positions["executed_state"].eq("FLAT")
    assert result.positions.loc[flat, ["units_y", "units_x"]].eq(0.0).all().all()
    assert accounting[
        ["gross_exposure", "long_exposure", "short_exposure"]
    ].dropna().ge(0.0).all().all()
    validate_backtest_invariants(result)


@pytest.mark.parametrize(
    "column",
    ["gross_pnl", "transaction_cost", "carry_cost", "net_pnl_after_carry"],
)
def test_invariant_validator_rejects_corrupted_accounting_identity(
    column: str,
) -> None:
    _, result = _integrated_case()
    corrupted = result.accounting.copy(deep=True)
    corrupted.iat[3, corrupted.columns.get_loc(column)] += 1.0

    with pytest.raises(ValueError, match="invariant failed"):
        validate_backtest_invariants(replace(result, accounting=corrupted))


@pytest.mark.parametrize(
    "corruption",
    ["decreasing_cost", "negative_exposure", "flat_units", "ledger_net"],
)
def test_invariant_validator_rejects_state_cost_and_ledger_corruption(
    corruption: str,
) -> None:
    _, result = _integrated_case()
    changed = result
    if corruption == "decreasing_cost":
        accounting = result.accounting.copy(deep=True)
        accounting.iat[
            4, accounting.columns.get_loc("cumulative_transaction_cost")
        ] = -1.0
        changed = replace(result, accounting=accounting)
    elif corruption == "negative_exposure":
        accounting = result.accounting.copy(deep=True)
        accounting.iat[2, accounting.columns.get_loc("gross_exposure")] = -1.0
        changed = replace(result, accounting=accounting)
    elif corruption == "flat_units":
        positions = result.positions.copy(deep=True)
        positions.iat[0, positions.columns.get_loc("units_y")] = 1.0
        changed = replace(result, positions=positions)
    else:
        ledger = result.ledger.copy(deep=True)
        ledger.iat[0, ledger.columns.get_loc("net_pnl")] += 1.0
        changed = replace(result, ledger=ledger)

    with pytest.raises(ValueError, match="invariant|first position-schedule"):
        validate_backtest_invariants(changed)


@pytest.mark.parametrize("future_change", ["prices", "zscore", "beta", "rates"])
def test_integrated_outputs_are_invariant_to_strictly_future_changes(
    future_change: str,
) -> None:
    index = pd.RangeIndex(8, name="causal_row")
    price_y = pd.Series(
        [100.0, 100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        index=index,
    )
    price_x = pd.Series(
        [50.0, 50.0, 49.0, 48.0, 47.0, 46.0, 45.0, 44.0],
        index=index,
    )
    zscore = pd.Series([-2.5, -1.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)
    beta = pd.Series(1.0, index=index)
    borrow_y = pd.Series(0.2, index=index)
    borrow_x = pd.Series(0.3, index=index)
    changed_y = price_y.copy(deep=True)
    changed_x = price_x.copy(deep=True)
    changed_z = zscore.copy(deep=True)
    changed_beta = beta.copy(deep=True)
    changed_borrow_y = borrow_y.copy(deep=True)
    changed_borrow_x = borrow_x.copy(deep=True)
    cutoff = 3
    if future_change == "prices":
        changed_y.iloc[cutoff + 1 :] *= 10.0
        changed_x.iloc[cutoff + 1 :] *= 0.1
    elif future_change == "zscore":
        changed_z.iloc[cutoff + 1 :] = [2.5, 1.0, 0.2, 0.0]
    elif future_change == "beta":
        changed_beta.iloc[cutoff + 1 :] = 5.0
    else:
        changed_borrow_y.iloc[cutoff + 1 :] = 4.0
        changed_borrow_x.iloc[cutoff + 1 :] = 6.0

    common = {
        "target_gross_notional": 2_000.0,
        "initial_capital": 10_000.0,
        "commission_bps": 5.0,
        "slippage_bps": 2.0,
        "financing_rate": 0.05,
        "rebalance": True,
        "rebalance_threshold": 0.5,
    }
    original = run_pair_backtest(
        price_y,
        price_x,
        beta,
        zscore=zscore,
        borrow_rate_y=borrow_y,
        borrow_rate_x=borrow_x,
        **common,
    )
    changed = run_pair_backtest(
        changed_y,
        changed_x,
        changed_beta,
        zscore=changed_z,
        borrow_rate_y=changed_borrow_y,
        borrow_rate_x=changed_borrow_x,
        **common,
    )

    pd.testing.assert_frame_equal(
        original.signals.iloc[: cutoff + 1],
        changed.signals.iloc[: cutoff + 1],
    )
    pd.testing.assert_frame_equal(
        original.positions.iloc[: cutoff + 1],
        changed.positions.iloc[: cutoff + 1],
    )
    pd.testing.assert_frame_equal(
        original.accounting.iloc[: cutoff + 1],
        changed.accounting.iloc[: cutoff + 1],
    )
    completed_original = original.ledger.loc[original.ledger["exit_row"] <= cutoff]
    completed_changed = changed.ledger.loc[changed.ledger["exit_row"] <= cutoff]
    pd.testing.assert_frame_equal(
        completed_original.reset_index(drop=True),
        completed_changed.reset_index(drop=True),
    )


def test_integrated_runs_are_deterministic_defensive_and_input_immutable() -> None:
    inputs, first = _integrated_case(rebalance=True)
    before = {
        name: value.copy(deep=True)
        for name, value in inputs.items()
        if isinstance(value, (pd.Series, pd.DataFrame))
    }
    second = run_pair_backtest(
        inputs["price_y"],
        inputs["price_x"],
        inputs["hedge_ratio"],
        2_000.0,
        zscore=inputs["zscore"],
        initial_capital=10_000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        fixed_commission_per_leg=0.25,
        borrow_rate_y=inputs["borrow_rate_y"],
        borrow_rate_x=inputs["borrow_rate_x"],
        financing_rate=0.126,
        periods_per_year=252,
        rebalance=True,
        rebalance_threshold=0.5,
    )

    pd.testing.assert_frame_equal(first.signals, second.signals)
    pd.testing.assert_frame_equal(first.positions, second.positions)
    pd.testing.assert_frame_equal(first.accounting, second.accounting)
    pd.testing.assert_frame_equal(first.ledger, second.ledger)
    assert first.reconciliation == second.reconciliation
    for name, expected in before.items():
        pd.testing.assert_series_equal(inputs[name], expected)

    first.positions.iat[0, first.positions.columns.get_loc("units_y")] = 99.0
    assert second.positions["units_y"].iat[0] == 0.0
    with pytest.raises(FrozenInstanceError):
        first.execution_lag = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_mode",
    ["both_sources", "neither_source", "malformed_signals", "misaligned_beta", "duplicate_index"],
)
def test_integrated_input_contract_rejects_invalid_sources_and_indices(
    invalid_mode: str,
) -> None:
    index = pd.RangeIndex(5)
    price_y = pd.Series(100.0, index=index)
    price_x = pd.Series(50.0, index=index)
    zscore = pd.Series([0.0, -2.5, -1.0, -0.2, 0.0], index=index)
    beta = pd.Series(1.0, index=index)
    kwargs: dict[str, Any] = {"zscore": zscore}
    if invalid_mode == "both_sources":
        kwargs["signals"] = _signals_from_events(["NONE"] * 5, index)
    elif invalid_mode == "neither_source":
        kwargs = {}
    elif invalid_mode == "malformed_signals":
        kwargs = {"signals": pd.DataFrame({"event": ["NONE"] * 5}, index=index)}
    elif invalid_mode == "misaligned_beta":
        beta = beta.iloc[::-1]
    else:
        duplicate = pd.Index([0, 0, 1, 2, 3])
        price_y.index = duplicate
        price_x.index = duplicate
        zscore.index = duplicate
        beta.index = duplicate

    with pytest.raises((TypeError, ValueError), match="Exactly one|missing|index|unique"):
        run_pair_backtest(
            price_y,
            price_x,
            beta,
            2_000.0,
            **kwargs,
        )


@pytest.mark.parametrize(
    "index",
    [
        pd.Index(["z", "a", "m", "b", "q", "x"], name="integrated_order"),
        pd.date_range(
            "2026-10-25",
            periods=6,
            freq="h",
            tz="Europe/London",
            name="integrated_time",
        ),
    ],
)
def test_integrated_output_preserves_arbitrary_index_and_timezone(
    index: pd.Index,
) -> None:
    _, result = _integrated_case(index=index)

    pd.testing.assert_index_equal(result.signals.index, index, exact=True)
    pd.testing.assert_index_equal(result.positions.index, index, exact=True)
    pd.testing.assert_index_equal(result.accounting.index, index, exact=True)
    assert result.ledger.loc[0, "entry_index"] == index[2]
    assert result.ledger.loc[0, "exit_index"] == index[4]
