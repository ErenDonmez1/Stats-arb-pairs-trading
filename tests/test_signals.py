"""Tests for causal rolling spread standardisation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.signals import (
    ExitReason,
    PositionState,
    TradeEvent,
    generate_trade_signals,
    rolling_zscore,
    standardise_spread,
)


def _spread(periods: int = 12) -> pd.Series:
    """Return a simple non-degenerate spread with a labelled index."""
    index = pd.Index([f"row-{position}" for position in range(periods)], name="row")
    return pd.Series(np.arange(1.0, periods + 1.0), index=index, name="spread")


def test_rolling_zscore_preserves_exact_index_name_and_float_dtype() -> None:
    spread = _spread()

    result = rolling_zscore(spread, lookback=4)

    pd.testing.assert_index_equal(result.index, spread.index, exact=True)
    assert result.name == "zscore"
    assert pd.api.types.is_float_dtype(result.dtype)


def test_first_lookback_rows_are_nan_and_first_post_warmup_row_is_finite() -> None:
    lookback = 4

    frame = standardise_spread(_spread(), lookback=lookback)
    result = rolling_zscore(_spread(), lookback=lookback)

    assert result.iloc[:lookback].isna().all()
    assert np.isfinite(result.iloc[lookback])
    statistics = ["rolling_mean", "rolling_std", "zscore"]
    assert frame.iloc[:lookback][statistics].isna().all().all()
    assert frame.iloc[lookback][statistics].notna().all()


def test_first_valid_zscore_uses_exactly_preceding_lookback_observations() -> None:
    spread = pd.Series([1.0, 2.0, 3.0, 4.0])
    prior = spread.iloc[:3]
    expected = (spread.iloc[3] - prior.mean()) / prior.std(ddof=1)

    result = rolling_zscore(spread, lookback=3)

    assert result.iloc[3] == pytest.approx(expected)


def test_rolling_zscore_matches_manual_lagged_sample_calculation() -> None:
    spread = pd.Series([1.0, -1.0, 2.0, 0.5, 4.0, -2.0, 3.0])
    lookback = 3
    expected = pd.Series(np.nan, index=spread.index, name="zscore")
    for position in range(lookback, len(spread)):
        prior = spread.iloc[position - lookback : position]
        expected.iloc[position] = (
            spread.iloc[position] - prior.mean()
        ) / prior.std(ddof=1)

    result = rolling_zscore(spread, lookback=lookback)

    pd.testing.assert_series_equal(result, expected)


def test_standardise_spread_excludes_current_value_from_rolling_statistics() -> None:
    spread = pd.Series([2.0, 4.0, 8.0, 100.0, 7.0])
    prior = spread.iloc[:3]

    result = standardise_spread(spread, lookback=3)

    assert result.loc[3, "rolling_mean"] == pytest.approx(prior.mean())
    assert result.loc[3, "rolling_std"] == pytest.approx(prior.std(ddof=1))
    assert result.loc[3, "zscore"] == pytest.approx(
        (spread.iloc[3] - prior.mean()) / prior.std(ddof=1)
    )


def test_standardisation_is_invariant_to_strictly_future_changes() -> None:
    spread = pd.Series(np.sin(np.arange(30, dtype=float) / 3.0))
    cutoff = 18
    changed = spread.copy(deep=True)
    changed.iloc[cutoff + 1 :] = np.linspace(100.0, 1_000.0, len(changed) - cutoff - 1)

    original_result = standardise_spread(spread, lookback=6)
    changed_result = standardise_spread(changed, lookback=6)

    pd.testing.assert_frame_equal(
        original_result.iloc[: cutoff + 1],
        changed_result.iloc[: cutoff + 1],
    )


def test_large_current_shock_does_not_contaminate_its_own_window() -> None:
    baseline = pd.Series([1.0, 2.0, 4.0, 5.0, 6.0, 7.0])
    shocked = baseline.copy(deep=True)
    target = 3
    shocked.iloc[target] = 10_000.0

    baseline_result = standardise_spread(baseline, lookback=3)
    shocked_result = standardise_spread(shocked, lookback=3)

    assert shocked_result.loc[target, "rolling_mean"] == pytest.approx(
        baseline_result.loc[target, "rolling_mean"]
    )
    assert shocked_result.loc[target, "rolling_std"] == pytest.approx(
        baseline_result.loc[target, "rolling_std"]
    )
    assert shocked_result.loc[target, "zscore"] != pytest.approx(
        baseline_result.loc[target, "zscore"]
    )
    assert shocked_result.loc[target + 1, "rolling_mean"] != pytest.approx(
        baseline_result.loc[target + 1, "rolling_mean"]
    )


def test_standardisation_does_not_mutate_input_series() -> None:
    spread = _spread()
    before = spread.copy(deep=True)

    standardise_spread(spread, lookback=4)
    rolling_zscore(spread, lookback=4)

    pd.testing.assert_series_equal(spread, before)


@pytest.mark.parametrize(
    "lookback",
    [True, False, 0, -1, 3.0, 3.5, "3", None],
)
def test_invalid_lookbacks_are_rejected(lookback: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="lookback"):
        rolling_zscore(_spread(), lookback=lookback)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ddof",
    [True, False, -1, 3, 4, 1.5, "1", None, np.nan],
)
def test_invalid_ddof_values_are_rejected(ddof: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="ddof"):
        rolling_zscore(_spread(), lookback=3, ddof=ddof)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid",
    [
        pd.DataFrame({"spread": [1.0, 2.0, 3.0]}),
        [1.0, 2.0, 3.0],
        np.array([1.0, 2.0, 3.0]),
        (1.0, 2.0, 3.0),
        1.0,
        None,
    ],
)
def test_dataframe_and_non_series_inputs_are_rejected(invalid: Any) -> None:
    with pytest.raises(TypeError, match="Series"):
        rolling_zscore(invalid, lookback=2)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", ["1.0", True, np.bool_(False), 1.0 + 2.0j])
def test_non_numeric_and_boolean_spread_values_are_rejected(invalid: Any) -> None:
    spread = _spread(6).astype(object)
    spread.iloc[1] = invalid

    with pytest.raises(ValueError, match="numeric"):
        rolling_zscore(spread, lookback=3)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_positive_and_negative_infinity_are_rejected(invalid: float) -> None:
    spread = _spread(6)
    spread.iloc[1] = invalid

    with pytest.raises(ValueError, match="finite"):
        rolling_zscore(spread, lookback=3)


def test_unrepresentable_numeric_spread_value_is_rejected_cleanly() -> None:
    spread = _spread(6).astype(object)
    spread.iloc[1] = 10**400

    with pytest.raises(ValueError, match="representable as floats"):
        rolling_zscore(spread, lookback=3)


def test_duplicate_index_is_rejected() -> None:
    spread = pd.Series([1.0, np.nan, 3.0, 4.0], index=["a", "b", "b", "c"])

    with pytest.raises(ValueError, match="unique index"):
        rolling_zscore(spread, lookback=2)


def test_missing_current_observation_produces_missing_zscore() -> None:
    spread = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])

    result = standardise_spread(spread, lookback=3)

    assert np.isfinite(result.loc[3, "rolling_mean"])
    assert np.isfinite(result.loc[3, "rolling_std"])
    assert pd.isna(result.loc[3, "zscore"])


def test_missing_prior_observation_invalidates_each_affected_window() -> None:
    spread = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])

    result = standardise_spread(spread, lookback=3)

    assert result.loc[3:5, ["rolling_mean", "rolling_std", "zscore"]].isna().all().all()
    assert result.loc[6, ["rolling_mean", "rolling_std", "zscore"]].notna().all()


def test_missing_values_are_not_filled_dropped_or_reordered() -> None:
    index = pd.Index(["g", "a", "f", "b", "e", "c", "d", "h"], name="event")
    spread = pd.Series(
        [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
        index=index,
    )

    result = standardise_spread(spread, lookback=3)

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert len(result) == len(spread)
    pd.testing.assert_series_equal(
        result["spread"], spread.rename("spread"), check_dtype=False
    )
    assert result["zscore"].iloc[3:7].isna().all()
    assert np.isfinite(result["zscore"].iloc[7])


def test_zero_variance_prior_window_returns_nan_not_infinity() -> None:
    spread = pd.Series([5.0, 5.0, 5.0, 10.0, 11.0])

    result = standardise_spread(spread, lookback=3)

    assert result.loc[3, "rolling_std"] == pytest.approx(0.0)
    assert pd.isna(result.loc[3, "zscore"])
    assert not np.isinf(result["zscore"].to_numpy()).any()


def test_near_zero_variance_prior_window_returns_nan_under_documented_threshold() -> None:
    spread = pd.Series([0.0, 1e-13, -1e-13, 2.0, 3.0])

    result = standardise_spread(spread, lookback=3)

    assert 0.0 < result.loc[3, "rolling_std"] <= 1e-12
    assert pd.isna(result.loc[3, "zscore"])
    assert not np.isinf(result["zscore"].to_numpy()).any()


def test_repeated_standardisation_calls_are_identical() -> None:
    spread = _spread()

    first_frame = standardise_spread(spread, lookback=4)
    second_frame = standardise_spread(spread, lookback=4)
    first_zscore = rolling_zscore(spread, lookback=4)
    second_zscore = rolling_zscore(spread, lookback=4)

    pd.testing.assert_frame_equal(first_frame, second_frame)
    pd.testing.assert_series_equal(first_zscore, second_zscore)


def test_unique_non_datetime_nonmonotonic_index_is_preserved() -> None:
    index = pd.Index([9, 2, 7, 1, 5, 3], name="sequence")
    spread = pd.Series([1.0, 4.0, 2.0, 8.0, 3.0, 6.0], index=index)

    result = standardise_spread(spread, lookback=3)

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result.index.tolist() == [9, 2, 7, 1, 5, 3]


def test_timezone_aware_datetime_index_and_metadata_are_preserved() -> None:
    index = pd.date_range(
        "2024-03-28",
        periods=8,
        freq="D",
        tz="Europe/London",
        name="timestamp",
    )
    spread = pd.Series(np.arange(8, dtype=float), index=index)

    frame = standardise_spread(spread, lookback=3)
    zscore = rolling_zscore(spread, lookback=3)

    pd.testing.assert_index_equal(frame.index, index, exact=True)
    pd.testing.assert_index_equal(zscore.index, index, exact=True)
    assert frame.index.tz == index.tz
    assert frame.index.freq == index.freq


@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_structurally_short_inputs_are_rejected(length: int) -> None:
    spread = pd.Series(np.arange(length, dtype=float))

    with pytest.raises(ValueError, match=r"lookback \+ 1"):
        rolling_zscore(spread, lookback=3)


def test_standardise_spread_returns_expected_columns_and_matches_series_api() -> None:
    spread = _spread()

    frame = standardise_spread(spread, lookback=4)
    zscore = rolling_zscore(spread, lookback=4)

    assert frame.columns.tolist() == [
        "spread",
        "rolling_mean",
        "rolling_std",
        "zscore",
    ]
    pd.testing.assert_index_equal(frame.index, spread.index, exact=True)
    pd.testing.assert_series_equal(frame["zscore"], zscore)


@pytest.mark.parametrize("method", ["expanding", "Rolling", "", None, 1])
def test_standardise_spread_rejects_unknown_methods(method: Any) -> None:
    with pytest.raises(ValueError, match="method"):
        standardise_spread(_spread(), lookback=3, method=method)  # type: ignore[arg-type]


def test_ddof_zero_matches_manual_population_standard_deviation() -> None:
    spread = pd.Series([1.0, 2.0, 4.0, 8.0])
    prior = spread.iloc[:3]
    expected = (spread.iloc[3] - prior.mean()) / prior.std(ddof=0)

    result = rolling_zscore(spread, lookback=3, ddof=0)

    assert result.iloc[3] == pytest.approx(expected)


def test_lookback_one_is_supported_with_population_ddof() -> None:
    result = rolling_zscore(_spread(4), lookback=1, ddof=0)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1:].isna().all()


def test_numpy_integer_parameters_are_supported() -> None:
    result = rolling_zscore(
        _spread(),
        lookback=np.int64(3),  # type: ignore[arg-type]
        ddof=np.int64(1),  # type: ignore[arg-type]
    )

    assert np.isfinite(result.iloc[3])


def test_nullable_float_missing_values_follow_missing_policy() -> None:
    spread = pd.Series([1.0, 2.0, 3.0, pd.NA, 5.0, 6.0, 7.0, 8.0], dtype="Float64")

    result = standardise_spread(spread, lookback=3)

    assert result["spread"].dtype == np.dtype("float64")
    assert pd.isna(result.loc[3, "zscore"])
    assert result.loc[4:6, "zscore"].isna().all()
    assert np.isfinite(result.loc[7, "zscore"])


def _trade_signals(
    values: list[Any] | pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate signals with standard test thresholds."""
    zscore = values if isinstance(values, pd.Series) else pd.Series(values)
    parameters = {
        "entry_z": 2.0,
        "exit_z": 0.5,
        "stop_z": 3.0,
    }
    parameters.update(kwargs)
    return generate_trade_signals(zscore, **parameters)


def test_position_state_is_immutable_and_has_required_members() -> None:
    assert {state.name: int(state) for state in PositionState} == {
        "FLAT": 0,
        "LONG_SPREAD": 1,
        "SHORT_SPREAD": -1,
    }

    with pytest.raises(AttributeError):
        PositionState.FLAT.value = 2  # type: ignore[misc]


def test_trade_event_and_exit_reason_vocabularies_are_explicit() -> None:
    assert {event.value for event in TradeEvent} == {
        "NONE",
        "ENTER_LONG",
        "ENTER_SHORT",
        "EXIT_MEAN_REVERSION",
        "EXIT_STOP",
        "EXIT_TIME",
    }
    assert {reason.value for reason in ExitReason} == {
        "NONE",
        "MEAN_REVERSION",
        "STOP",
        "TIME",
    }


def test_generate_trade_signals_returns_exact_schema_and_dtypes() -> None:
    result = _trade_signals([0.0, -2.0, -1.0, 0.0])

    assert result.columns.tolist() == [
        "zscore",
        "state",
        "position",
        "entry_long",
        "entry_short",
        "exit",
        "stop",
        "time_exit",
        "event",
        "exit_reason",
        "holding_period",
        "cooldown_remaining",
    ]
    assert result["zscore"].dtype == np.dtype("float64")
    assert result["position"].dtype == np.dtype("int8")
    assert result["holding_period"].dtype == np.dtype("int64")
    assert result["cooldown_remaining"].dtype == np.dtype("int64")
    for column in ("entry_long", "entry_short", "exit", "stop", "time_exit"):
        assert result[column].dtype == np.dtype("bool")
    assert not any(
        term in result.columns
        for term in ("pnl", "shares", "quantity", "cost", "return", "fill")
    )


def test_long_entry_hold_and_mean_reversion_exit_use_inclusive_boundaries() -> None:
    result = _trade_signals([0.0, -2.0, -2.5, -0.5])

    assert result["state"].tolist() == [
        "FLAT",
        "LONG_SPREAD",
        "LONG_SPREAD",
        "FLAT",
    ]
    assert result["position"].tolist() == [0, 1, 1, 0]
    assert result["event"].tolist() == [
        "NONE",
        "ENTER_LONG",
        "NONE",
        "EXIT_MEAN_REVERSION",
    ]
    assert result["entry_long"].tolist() == [False, True, False, False]
    assert result["exit"].tolist() == [False, False, False, True]
    assert result["exit_reason"].iloc[-1] == "MEAN_REVERSION"
    assert result["holding_period"].tolist() == [0, 0, 1, 2]


def test_short_entry_hold_and_mean_reversion_exit_use_inclusive_boundaries() -> None:
    result = _trade_signals([0.0, 2.0, 2.5, 0.5])

    assert result["state"].tolist() == [
        "FLAT",
        "SHORT_SPREAD",
        "SHORT_SPREAD",
        "FLAT",
    ]
    assert result["position"].tolist() == [0, -1, -1, 0]
    assert result["event"].tolist() == [
        "NONE",
        "ENTER_SHORT",
        "NONE",
        "EXIT_MEAN_REVERSION",
    ]
    assert result["entry_short"].tolist() == [False, True, False, False]
    assert result["exit_reason"].iloc[-1] == "MEAN_REVERSION"
    assert result["holding_period"].tolist() == [0, 0, 1, 2]


def test_long_stop_uses_inclusive_boundary_and_sets_consistent_flags() -> None:
    result = _trade_signals([-2.0, -3.0], cooldown_period=2)

    assert result["event"].tolist() == ["ENTER_LONG", "EXIT_STOP"]
    assert result["position"].tolist() == [1, 0]
    assert bool(result.loc[1, "exit"])
    assert bool(result.loc[1, "stop"])
    assert not bool(result.loc[1, "time_exit"])
    assert result.loc[1, "exit_reason"] == "STOP"
    assert result.loc[1, "cooldown_remaining"] == 2
    assert result["holding_period"].tolist() == [0, 1]


def test_short_stop_uses_inclusive_boundary_and_sets_consistent_flags() -> None:
    result = _trade_signals([2.0, 3.0])

    assert result["event"].tolist() == ["ENTER_SHORT", "EXIT_STOP"]
    assert result["position"].tolist() == [-1, 0]
    assert bool(result.loc[1, "exit"])
    assert bool(result.loc[1, "stop"])
    assert result.loc[1, "exit_reason"] == "STOP"
    assert result["holding_period"].tolist() == [0, 1]


def test_flat_extreme_value_enters_without_same_row_stop() -> None:
    result = _trade_signals([-4.0])

    assert result.loc[0, "event"] == "ENTER_LONG"
    assert result.loc[0, "state"] == "LONG_SPREAD"
    assert not bool(result.loc[0, "stop"])
    assert not bool(result.loc[0, "exit"])


def test_no_repeated_entry_events_while_position_is_open() -> None:
    result = _trade_signals([-2.0, -2.2, -2.8])

    assert result["event"].tolist() == ["ENTER_LONG", "NONE", "NONE"]
    assert result["entry_long"].sum() == 1
    assert result["position"].tolist() == [1, 1, 1]


def test_opposite_threshold_exits_without_same_row_reversal() -> None:
    result = _trade_signals([-2.0, 2.0, 2.0])

    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_MEAN_REVERSION",
        "ENTER_SHORT",
    ]
    assert result["position"].tolist() == [1, 0, -1]
    assert not bool(result.loc[1, "entry_short"])
    assert bool(result.loc[2, "entry_short"])


def test_short_position_exits_before_a_later_long_entry() -> None:
    result = _trade_signals([2.0, -2.0, -2.0])

    assert result["event"].tolist() == [
        "ENTER_SHORT",
        "EXIT_MEAN_REVERSION",
        "ENTER_LONG",
    ]
    assert result["position"].tolist() == [-1, 0, 1]
    assert not bool(result.loc[1, "entry_long"])
    assert bool(result.loc[2, "entry_long"])


def test_stop_exit_precedes_time_exit() -> None:
    result = _trade_signals([-2.0, -3.0], max_holding_period=1)

    assert result.loc[1, "event"] == "EXIT_STOP"
    assert result.loc[1, "exit_reason"] == "STOP"
    assert bool(result.loc[1, "stop"])
    assert not bool(result.loc[1, "time_exit"])


def test_time_exit_precedes_mean_reversion_when_limit_is_reached() -> None:
    result = _trade_signals([-2.0, -0.5], max_holding_period=1)

    assert result.loc[1, "event"] == "EXIT_TIME"
    assert result.loc[1, "exit_reason"] == "TIME"
    assert bool(result.loc[1, "time_exit"])


def test_entry_row_holding_period_is_zero_and_later_rows_advance_it() -> None:
    result = _trade_signals([-2.0, -1.5, -1.0])

    assert result["holding_period"].tolist() == [0, 1, 2]


def test_maximum_holding_period_forces_exit_on_exact_subsequent_row() -> None:
    result = _trade_signals(
        [-2.0, -1.5, -1.0, -1.0],
        max_holding_period=3,
        cooldown_period=2,
    )

    assert result["position"].tolist() == [1, 1, 1, 0]
    assert result["holding_period"].tolist() == [0, 1, 2, 3]
    assert result["event"].tolist() == [
        "ENTER_LONG",
        "NONE",
        "NONE",
        "EXIT_TIME",
    ]
    assert result.loc[3, "cooldown_remaining"] == 2


def test_maximum_holding_period_one_exits_on_row_after_entry() -> None:
    result = _trade_signals([-2.0, -1.5], max_holding_period=1)

    assert result["event"].tolist() == ["ENTER_LONG", "EXIT_TIME"]
    assert result["holding_period"].tolist() == [0, 1]


def test_short_maximum_holding_period_emits_terminal_age_on_exit() -> None:
    result = _trade_signals(
        [2.0, 1.5, 1.0],
        max_holding_period=2,
    )

    assert result["position"].tolist() == [-1, -1, 0]
    assert result["event"].tolist() == ["ENTER_SHORT", "NONE", "EXIT_TIME"]
    assert result["holding_period"].tolist() == [0, 1, 2]


def test_flat_row_after_exit_resets_holding_period_to_zero() -> None:
    result = _trade_signals([-2.0, 0.0, 0.0])

    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_MEAN_REVERSION",
        "NONE",
    ]
    assert result["holding_period"].tolist() == [0, 1, 0]


def test_exit_row_does_not_count_toward_cooldown() -> None:
    result = _trade_signals([-2.0, 0.0], cooldown_period=2)

    assert result["event"].tolist() == ["ENTER_LONG", "EXIT_MEAN_REVERSION"]
    assert result.loc[1, "cooldown_remaining"] == 2


def test_cooldown_blocks_exact_number_of_later_rows() -> None:
    result = _trade_signals([-2.0, 0.0, 2.0, 2.0, 2.0], cooldown_period=2)

    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_MEAN_REVERSION",
        "NONE",
        "NONE",
        "ENTER_SHORT",
    ]
    assert result["cooldown_remaining"].tolist() == [0, 2, 2, 1, 0]
    assert result["position"].tolist() == [1, 0, 0, 0, -1]


def test_missing_zscore_while_flat_remains_flat() -> None:
    result = _trade_signals([np.nan, np.nan])

    assert result["state"].tolist() == ["FLAT", "FLAT"]
    assert result["event"].tolist() == ["NONE", "NONE"]
    assert result["holding_period"].tolist() == [0, 0]


def test_missing_zscore_retains_open_position_and_advances_holding_period() -> None:
    result = _trade_signals([-2.0, np.nan, -1.0])

    assert result["state"].tolist() == [
        "LONG_SPREAD",
        "LONG_SPREAD",
        "LONG_SPREAD",
    ]
    assert result["holding_period"].tolist() == [0, 1, 2]
    assert result["event"].tolist() == ["ENTER_LONG", "NONE", "NONE"]


def test_missing_zscore_retains_short_position_and_advances_holding_period() -> None:
    result = _trade_signals([2.0, np.nan, 1.0])

    assert result["state"].tolist() == [
        "SHORT_SPREAD",
        "SHORT_SPREAD",
        "SHORT_SPREAD",
    ]
    assert result["holding_period"].tolist() == [0, 1, 2]
    assert result["event"].tolist() == ["ENTER_SHORT", "NONE", "NONE"]


def test_maximum_holding_exit_can_occur_on_missing_zscore() -> None:
    result = _trade_signals(
        [-2.0, np.nan, np.nan],
        max_holding_period=2,
    )

    assert result["event"].tolist() == ["ENTER_LONG", "NONE", "EXIT_TIME"]
    assert pd.isna(result.loc[2, "zscore"])
    assert result.loc[2, "state"] == "FLAT"
    assert result.loc[2, "holding_period"] == 2


def test_missing_rows_advance_cooldown() -> None:
    result = _trade_signals(
        [-2.0, 0.0, np.nan, 2.0, 2.0],
        cooldown_period=2,
    )

    assert result["cooldown_remaining"].tolist() == [0, 2, 2, 1, 0]
    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_MEAN_REVERSION",
        "NONE",
        "NONE",
        "ENTER_SHORT",
    ]


def test_cooldown_after_stop_exit_blocks_configured_rows() -> None:
    result = _trade_signals(
        [-2.0, -3.0, 2.0, 2.0, 2.0],
        cooldown_period=2,
    )

    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_STOP",
        "NONE",
        "NONE",
        "ENTER_SHORT",
    ]
    assert result["cooldown_remaining"].tolist() == [0, 2, 2, 1, 0]


def test_cooldown_after_time_exit_blocks_configured_rows() -> None:
    result = _trade_signals(
        [-2.0, -1.0, 2.0, 2.0, 2.0],
        max_holding_period=1,
        cooldown_period=2,
    )

    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_TIME",
        "NONE",
        "NONE",
        "ENTER_SHORT",
    ]
    assert result["cooldown_remaining"].tolist() == [0, 2, 2, 1, 0]


def test_cooldown_period_one_blocks_exactly_the_next_row() -> None:
    result = _trade_signals([-2.0, 0.0, 2.0, 2.0], cooldown_period=1)

    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_MEAN_REVERSION",
        "NONE",
        "ENTER_SHORT",
    ]
    assert result["cooldown_remaining"].tolist() == [0, 1, 1, 0]


def test_zero_cooldown_remaining_means_current_row_is_entry_eligible() -> None:
    result = _trade_signals([-2.0, 0.0, 2.0, 2.0, 2.0], cooldown_period=2)

    zero_after_cooldown = result.iloc[4]
    assert zero_after_cooldown["cooldown_remaining"] == 0
    assert zero_after_cooldown["event"] == "ENTER_SHORT"
    assert bool(zero_after_cooldown["entry_short"])


def test_missing_trade_signal_rows_are_preserved_without_filling_or_reordering() -> None:
    index = pd.Index(["c", "a", "d", "b"], name="decision")
    zscore = pd.Series([-2.0, np.nan, -1.0, 0.0], index=index, name="input")

    result = _trade_signals(zscore)

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert len(result) == len(zscore)
    assert pd.isna(result.loc["a", "zscore"])
    assert result.loc["a", "state"] == "LONG_SPREAD"


def test_explicit_hold_missing_policy_matches_default() -> None:
    zscore = pd.Series([-2.0, np.nan, -1.0, 0.0])

    default = _trade_signals(zscore)
    explicit = _trade_signals(zscore, missing_policy="hold")

    pd.testing.assert_frame_equal(default, explicit)


@pytest.mark.parametrize(
    ("entry_z", "exit_z", "stop_z"),
    [
        (0.5, 0.5, 3.0),
        (0.4, 0.5, 3.0),
        (2.0, -0.1, 3.0),
        (2.0, 0.5, 2.0),
        (2.0, 0.5, 1.5),
    ],
)
def test_trade_threshold_relationships_are_strictly_validated(
    entry_z: float,
    exit_z: float,
    stop_z: float,
) -> None:
    with pytest.raises(ValueError):
        generate_trade_signals(pd.Series([0.0]), entry_z, exit_z, stop_z)


@pytest.mark.parametrize("field", ["entry_z", "exit_z", "stop_z"])
@pytest.mark.parametrize(
    "invalid",
    [True, False, np.bool_(True), np.bool_(False), "2.0", 1.0 + 2.0j, None],
)
def test_trade_thresholds_reject_invalid_numeric_types(
    field: str,
    invalid: Any,
) -> None:
    parameters: dict[str, Any] = {
        "entry_z": 2.0,
        "exit_z": 0.5,
        "stop_z": 3.0,
    }
    parameters[field] = invalid

    with pytest.raises(TypeError, match=field):
        generate_trade_signals(pd.Series([0.0]), **parameters)


@pytest.mark.parametrize("field", ["entry_z", "exit_z", "stop_z"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_trade_thresholds_reject_nan_and_infinity(
    field: str,
    invalid: float,
) -> None:
    parameters = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 3.0}
    parameters[field] = invalid

    with pytest.raises(ValueError, match=field):
        generate_trade_signals(pd.Series([0.0]), **parameters)


def test_integer_and_floating_trade_thresholds_are_accepted() -> None:
    result = generate_trade_signals(
        pd.Series([-2.0, 0.0, 3.0]),
        entry_z=2,
        exit_z=0.5,
        stop_z=3,
    )

    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_MEAN_REVERSION",
        "ENTER_SHORT",
    ]


def test_zero_exit_threshold_is_accepted() -> None:
    result = generate_trade_signals(
        pd.Series([-2.0, 0.0]),
        entry_z=2.0,
        exit_z=0.0,
        stop_z=3.0,
    )

    assert result["event"].tolist() == ["ENTER_LONG", "EXIT_MEAN_REVERSION"]


@pytest.mark.parametrize("field", ["max_holding_period", "cooldown_period"])
@pytest.mark.parametrize(
    "invalid",
    [True, False, np.bool_(True), np.bool_(False), 0, -1, 1.0, 1.5, "2", np.nan],
)
def test_optional_trade_periods_reject_invalid_values(
    field: str,
    invalid: Any,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        _trade_signals([0.0], **{field: invalid})


def test_numpy_integer_optional_trade_periods_are_accepted() -> None:
    result = _trade_signals(
        [-2.0, -1.0, -1.0],
        max_holding_period=np.int64(2),  # type: ignore[arg-type]
        cooldown_period=np.int64(1),  # type: ignore[arg-type]
    )

    assert result.loc[2, "event"] == "EXIT_TIME"
    assert result.loc[2, "cooldown_remaining"] == 1


@pytest.mark.parametrize("field", ["max_holding_period", "cooldown_period"])
def test_optional_trade_periods_reject_values_above_int64_range(field: str) -> None:
    with pytest.raises(ValueError, match="int64 range"):
        _trade_signals([0.0], **{field: np.iinfo(np.int64).max + 1})


@pytest.mark.parametrize("policy", ["drop", "fill", "Hold", "", None, 1])
def test_unknown_missing_trade_signal_policy_is_rejected(policy: Any) -> None:
    with pytest.raises(ValueError, match="missing_policy"):
        _trade_signals([0.0], missing_policy=policy)


@pytest.mark.parametrize(
    "invalid",
    [
        pd.DataFrame({"zscore": [0.0]}),
        [0.0],
        np.array([0.0]),
        (0.0,),
        0.0,
        None,
    ],
)
def test_generate_trade_signals_requires_a_series(invalid: Any) -> None:
    with pytest.raises(TypeError, match="Series"):
        generate_trade_signals(  # type: ignore[arg-type]
            invalid,
            entry_z=2.0,
            exit_z=0.5,
            stop_z=3.0,
        )


@pytest.mark.parametrize("invalid", ["2.0", True, np.bool_(False), 1.0 + 2.0j])
def test_non_numeric_trade_zscores_are_rejected(invalid: Any) -> None:
    zscore = pd.Series([0.0, invalid, 1.0], dtype=object)

    with pytest.raises(ValueError, match="numeric"):
        _trade_signals(zscore)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_infinite_trade_zscores_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _trade_signals(pd.Series([0.0, invalid]))


def test_unrepresentable_numeric_trade_zscore_is_rejected_cleanly() -> None:
    zscore = pd.Series([0.0, 10**400], dtype=object)

    with pytest.raises(ValueError, match="representable as floats"):
        _trade_signals(zscore)


def test_duplicate_trade_zscore_index_is_rejected() -> None:
    zscore = pd.Series([0.0, np.nan], index=["same", "same"])

    with pytest.raises(ValueError, match="unique index"):
        _trade_signals(zscore)


def test_nullable_float_trade_zscores_follow_hold_policy() -> None:
    zscore = pd.Series([-2.0, pd.NA, -1.0, 0.0], dtype="Float64")

    result = _trade_signals(zscore)

    assert result["zscore"].dtype == np.dtype("float64")
    assert result["state"].tolist() == [
        "LONG_SPREAD",
        "LONG_SPREAD",
        "LONG_SPREAD",
        "FLAT",
    ]
    assert pd.isna(result.loc[1, "zscore"])


def test_nonmonotonic_trade_index_is_processed_in_original_row_order() -> None:
    index = pd.Index([9, 2, 7], name="sequence")
    zscore = pd.Series([-2.0, 0.0, 2.0], index=index)

    result = _trade_signals(zscore)

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result["event"].tolist() == [
        "ENTER_LONG",
        "EXIT_MEAN_REVERSION",
        "ENTER_SHORT",
    ]


def test_timezone_trade_index_metadata_is_preserved() -> None:
    index = pd.date_range(
        "2024-03-28",
        periods=5,
        freq="D",
        tz="Europe/London",
        name="decision_time",
    )
    zscore = pd.Series([-2.0, -1.0, 0.0, 2.0, 1.0], index=index)

    result = _trade_signals(zscore)

    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result.index.tz == index.tz
    assert result.index.freq == index.freq


def test_generate_trade_signals_does_not_mutate_input() -> None:
    zscore = pd.Series([-2.0, np.nan, -1.0, 0.0], name="causal_zscore")
    before = zscore.copy(deep=True)

    _trade_signals(zscore, max_holding_period=3, cooldown_period=2)

    pd.testing.assert_series_equal(zscore, before)


def test_repeated_trade_signal_calls_are_deterministic() -> None:
    zscore = pd.Series([-2.0, np.nan, -1.0, 0.0, 2.0, 3.0])

    first = _trade_signals(zscore, max_holding_period=3, cooldown_period=1)
    second = _trade_signals(zscore, max_holding_period=3, cooldown_period=1)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    ("values", "kwargs", "row", "event", "reason", "stop", "time_exit"),
    [
        ([-2.0, 0.0], {}, 1, "EXIT_MEAN_REVERSION", "MEAN_REVERSION", False, False),
        ([-2.0, -3.0], {}, 1, "EXIT_STOP", "STOP", True, False),
        ([-2.0, -1.0], {"max_holding_period": 1}, 1, "EXIT_TIME", "TIME", False, True),
    ],
)
def test_exit_events_reasons_and_flags_are_consistent(
    values: list[float],
    kwargs: dict[str, Any],
    row: int,
    event: str,
    reason: str,
    stop: bool,
    time_exit: bool,
) -> None:
    result = _trade_signals(values, **kwargs)

    assert bool(result.loc[row, "exit"])
    assert bool(result.loc[row, "stop"]) is stop
    assert bool(result.loc[row, "time_exit"]) is time_exit
    assert result.loc[row, "event"] == event
    assert result.loc[row, "exit_reason"] == reason


def test_strictly_future_zscore_changes_do_not_change_output_through_cutoff() -> None:
    zscore = pd.Series([-2.0, -1.5, np.nan, -1.0, 0.0, 2.0, 2.5, 3.0])
    cutoff = 4
    changed = zscore.copy(deep=True)
    changed.iloc[cutoff + 1 :] = [-10.0, 10.0, -10.0]

    original = _trade_signals(
        zscore,
        max_holding_period=4,
        cooldown_period=2,
    )
    modified = _trade_signals(
        changed,
        max_holding_period=4,
        cooldown_period=2,
    )

    pd.testing.assert_frame_equal(
        original.iloc[: cutoff + 1],
        modified.iloc[: cutoff + 1],
    )


def test_empty_zscore_returns_typed_empty_signal_frame() -> None:
    index = pd.DatetimeIndex([], tz="UTC", name="decision_time")
    result = _trade_signals(pd.Series([], index=index, dtype=float))

    assert result.empty
    pd.testing.assert_index_equal(result.index, index, exact=True)
    assert result.columns.tolist() == [
        "zscore",
        "state",
        "position",
        "entry_long",
        "entry_short",
        "exit",
        "stop",
        "time_exit",
        "event",
        "exit_reason",
        "holding_period",
        "cooldown_remaining",
    ]
    assert result["position"].dtype == np.dtype("int8")
    assert result["exit"].dtype == np.dtype("bool")
