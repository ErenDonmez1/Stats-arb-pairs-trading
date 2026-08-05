"""Tests for causal rolling spread standardisation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.signals import rolling_zscore, standardise_spread


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
