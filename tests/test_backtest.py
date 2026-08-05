"""Focused tests for Milestone 6A execution scheduling and pair sizing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.backtest import (
    PairUnits,
    build_position_schedule,
    calculate_pair_units,
    lag_trade_decisions,
)


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


def test_dynamic_hedge_ratio_uses_execution_row_value() -> None:
    index = pd.RangeIndex(3)
    beta = pd.Series([0.25, 3.0, 8.0], index=index)
    result = _schedule(
        ["ENTER_LONG", "NONE", "NONE"],
        index=index,
        hedge_ratio=beta,
    )

    execution = result.loc[1]
    assert abs(execution["notional_x"] / execution["notional_y"]) == pytest.approx(
        3.0
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
    inputs[missing_input].iloc[1] = np.nan

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
    inputs[missing_input].iloc[3] = np.nan

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
