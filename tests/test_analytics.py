"""Focused tests for Milestone 7A core performance and risk metrics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from pairs_trading.analytics import (
    CorePerformanceMetrics,
    annualized_return,
    annualized_volatility,
    calculate_core_metrics,
    downside_deviation,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)


def test_total_return_uses_geometric_compounding_not_arithmetic_sum() -> None:
    returns = pd.Series([0.10, -0.10])

    result = total_return(returns)

    assert result == pytest.approx(1.10 * 0.90 - 1.0)
    assert result != pytest.approx(returns.sum())


def test_positive_returns_produce_correct_total_return() -> None:
    returns = pd.Series([0.02, 0.03, 0.01])

    assert total_return(returns) == pytest.approx(
        np.prod(1.0 + returns.to_numpy()) - 1.0
    )


def test_mixed_returns_compound_correctly() -> None:
    returns = pd.Series([0.15, -0.07, 0.04, -0.02])

    assert total_return(returns) == pytest.approx(
        (1.15 * 0.93 * 1.04 * 0.98) - 1.0
    )


def test_annualized_return_uses_geometric_formula() -> None:
    returns = pd.Series([0.01, 0.02, -0.005, 0.015])
    periods_per_year = 12
    compounded = np.prod(1.0 + returns.to_numpy()) - 1.0
    expected = (1.0 + compounded) ** (periods_per_year / len(returns)) - 1.0

    assert annualized_return(returns, periods_per_year) == pytest.approx(expected)


def test_annualized_volatility_uses_sample_standard_deviation() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.005])

    result = annualized_volatility(returns, 12)

    assert result == pytest.approx(returns.std(ddof=1) * np.sqrt(12))
    assert result != pytest.approx(returns.std(ddof=0) * np.sqrt(12))


def test_annualized_volatility_scales_by_square_root_time() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.005])

    monthly = annualized_volatility(returns, 12)
    daily = annualized_volatility(returns, 252)

    assert daily / monthly == pytest.approx(np.sqrt(252 / 12))


def test_sharpe_zero_risk_free_rate_matches_manual_calculation() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.005])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(12)

    assert sharpe_ratio(returns, 12) == pytest.approx(expected)


def test_sharpe_nonzero_risk_free_rate_uses_geometric_period_rate() -> None:
    returns = pd.Series([0.01, -0.005, 0.02, 0.015])
    annual_risk_free = 0.06
    periods = 12
    period_rate = (1.0 + annual_risk_free) ** (1.0 / periods) - 1.0
    excess = returns.to_numpy() - period_rate
    expected = np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods)

    result = sharpe_ratio(returns, periods, annual_risk_free)

    assert result == pytest.approx(expected)


def test_constant_excess_returns_produce_nan_sharpe() -> None:
    periods = 12
    annual_risk_free = 0.06
    period_rate = (1.0 + annual_risk_free) ** (1.0 / periods) - 1.0
    returns = pd.Series([period_rate] * 5)

    assert np.isnan(sharpe_ratio(returns, periods, annual_risk_free))


def test_effectively_zero_excess_volatility_produces_nan_sharpe() -> None:
    returns = pd.Series([0.01, 0.01 + 1e-15, 0.01 - 1e-15])

    assert np.isnan(sharpe_ratio(returns, 252))


def test_downside_deviation_matches_lower_partial_manual_calculation() -> None:
    returns = pd.Series([0.02, -0.01, 0.005, -0.03])
    periods = 12
    annual_risk_free = 0.03
    target = (1.0 + annual_risk_free) ** (1.0 / periods) - 1.0
    downside = np.minimum(returns.to_numpy() - target, 0.0)
    expected = np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods)

    result = downside_deviation(returns, periods, annual_risk_free)

    assert result == pytest.approx(expected)


def test_sortino_matches_manual_lower_partial_calculation() -> None:
    returns = pd.Series([0.02, -0.01, 0.005, -0.03])
    periods = 12
    annual_risk_free = 0.03
    target = (1.0 + annual_risk_free) ** (1.0 / periods) - 1.0
    excess = returns.to_numpy() - target
    period_downside = np.sqrt(np.mean(np.square(np.minimum(excess, 0.0))))
    expected = np.mean(excess) / period_downside * np.sqrt(periods)

    result = sortino_ratio(returns, periods, annual_risk_free)

    assert result == pytest.approx(expected)


def test_no_downside_observations_produce_nan_sortino() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])

    assert downside_deviation(returns, 12) == 0.0
    assert np.isnan(sortino_ratio(returns, 12))


def test_missing_returns_are_dropped_for_metric_estimation() -> None:
    returns = pd.Series([0.01, np.nan, -0.02, np.nan, 0.03])
    expected = returns.dropna()

    assert total_return(returns) == pytest.approx(total_return(expected))
    assert annualized_return(returns, 12) == pytest.approx(
        annualized_return(expected, 12)
    )
    assert annualized_volatility(returns, 12) == pytest.approx(
        annualized_volatility(expected, 12)
    )


def test_missing_returns_are_not_filled_or_interpolated() -> None:
    returns = pd.Series([0.01, np.nan, -0.02, 0.03])
    dropped = returns.dropna()
    forward_filled = returns.ffill()

    result = annualized_volatility(returns, 12)

    assert result == pytest.approx(annualized_volatility(dropped, 12))
    assert result != pytest.approx(annualized_volatility(forward_filled, 12))


def test_core_observation_count_uses_only_valid_returns() -> None:
    returns = pd.Series([0.01, np.nan, -0.02, 0.03, np.nan])

    result = calculate_core_metrics(returns, 12)

    assert result.observations == 3


def test_unsupported_missing_policy_is_rejected() -> None:
    returns = pd.Series([0.01, -0.02])

    with pytest.raises(ValueError, match="missing_policy"):
        calculate_core_metrics(returns, 12, missing_policy="fill")


@pytest.mark.parametrize("invalid_return", [-1.0, -1.01])
@pytest.mark.parametrize(
    "metric",
    [
        lambda series: total_return(series),
        lambda series: annualized_return(series, 12),
        lambda series: calculate_core_metrics(series, 12),
    ],
)
def test_geometric_metrics_reject_returns_at_or_below_total_loss(
    invalid_return: float,
    metric: Callable[[pd.Series], Any],
) -> None:
    returns = pd.Series([0.01, invalid_return, 0.02])

    with pytest.raises(ValueError, match="exceed -1"):
        metric(returns)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), 0, -1, 12.0, 1.5, "12", np.nan, None],
)
def test_invalid_periods_per_year_values_are_rejected(invalid: Any) -> None:
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises((TypeError, ValueError), match="periods_per_year"):
        calculate_core_metrics(returns, invalid)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "0.01", -1.0, -2.0, np.nan, np.inf, -np.inf],
)
@pytest.mark.parametrize(
    "metric",
    [sharpe_ratio, downside_deviation, sortino_ratio],
)
def test_invalid_risk_free_rate_types_and_values_are_rejected(
    invalid: Any,
    metric: Callable[..., float],
) -> None:
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises((TypeError, ValueError), match="risk_free_rate"):
        metric(returns, 12, invalid)


@pytest.mark.parametrize(
    "metric",
    [
        lambda value: total_return(value),
        lambda value: annualized_return(value, 12),
        lambda value: annualized_volatility(value, 12),
        lambda value: sharpe_ratio(value, 12),
        lambda value: downside_deviation(value, 12),
        lambda value: sortino_ratio(value, 12),
        lambda value: calculate_core_metrics(value, 12),
    ],
)
def test_dataframe_inputs_are_rejected(metric: Callable[[Any], Any]) -> None:
    frame = pd.DataFrame({"return": [0.01, -0.02]})

    with pytest.raises(TypeError, match="pandas Series"):
        metric(frame)


@pytest.mark.parametrize(
    "invalid",
    [[0.01, -0.02], np.array([0.01, -0.02]), (0.01, -0.02), 0.01],
)
def test_non_series_inputs_are_rejected(invalid: Any) -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        calculate_core_metrics(invalid, 12)


@pytest.mark.parametrize("invalid", ["bad", True, np.bool_(False), 1 + 2j])
def test_non_numeric_non_missing_returns_are_rejected(invalid: Any) -> None:
    returns = pd.Series([0.01, invalid, -0.02], dtype=object)

    with pytest.raises(ValueError, match="real numeric"):
        calculate_core_metrics(returns, 12)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_infinite_returns_are_rejected(invalid: float) -> None:
    returns = pd.Series([0.01, invalid, -0.02])

    with pytest.raises(ValueError, match="finite"):
        calculate_core_metrics(returns, 12)


def test_duplicate_return_indices_are_rejected() -> None:
    returns = pd.Series([0.01, -0.02, 0.03], index=[0, 0, 1])

    with pytest.raises(ValueError, match="unique index"):
        calculate_core_metrics(returns, 12)


def test_input_series_is_unchanged() -> None:
    index = pd.Index(["third", "first", "second", "fourth"], name="sequence")
    returns = pd.Series([0.01, np.nan, -0.02, 0.03], index=index, name="net_return")
    returns.attrs["source"] = "backtest"
    before = returns.copy(deep=True)

    calculate_core_metrics(returns, 12)

    pd.testing.assert_series_equal(returns, before)
    assert returns.attrs == before.attrs


def test_repeated_calls_are_deterministic() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.005])

    first = calculate_core_metrics(returns, 12, 0.02)
    second = calculate_core_metrics(returns, 12, 0.02)

    assert first == second


def test_non_datetime_nonmonotonic_index_is_supported_without_reordering() -> None:
    index = pd.Index(["z", "a", "m", "b"], name="sequence")
    returns = pd.Series([0.01, -0.02, 0.03, -0.005], index=index)

    result = calculate_core_metrics(returns, 12)

    assert result.observations == 4
    assert total_return(returns) == pytest.approx(
        np.prod(1.0 + returns.to_numpy()) - 1.0
    )


def test_timezone_aware_datetime_index_is_supported() -> None:
    index = pd.date_range(
        "2025-03-27",
        periods=4,
        freq="h",
        tz="Europe/London",
        name="timestamp",
    )
    returns = pd.Series([0.01, -0.02, 0.03, -0.005], index=index)

    result = calculate_core_metrics(returns, 24)

    assert result.observations == 4
    assert result.periods_per_year == 24


def test_core_performance_metrics_is_immutable() -> None:
    result = calculate_core_metrics(pd.Series([0.01, -0.02, 0.03]), 12)

    with pytest.raises(FrozenInstanceError):
        result.total_return = 0.0  # type: ignore[misc]


def test_calculate_core_metrics_matches_individual_public_functions() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.005, np.nan])
    periods = 12
    annual_risk_free = 0.025

    result = calculate_core_metrics(returns, periods, annual_risk_free)

    assert isinstance(result, CorePerformanceMetrics)
    assert result.total_return == pytest.approx(total_return(returns))
    assert result.annualized_return == pytest.approx(
        annualized_return(returns, periods)
    )
    assert result.annualized_volatility == pytest.approx(
        annualized_volatility(returns, periods)
    )
    assert result.sharpe_ratio == pytest.approx(
        sharpe_ratio(returns, periods, annual_risk_free)
    )
    assert result.downside_deviation == pytest.approx(
        downside_deviation(returns, periods, annual_risk_free)
    )
    assert result.sortino_ratio == pytest.approx(
        sortino_ratio(returns, periods, annual_risk_free)
    )


def test_prefix_metrics_ignore_strictly_later_observations() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.005, 0.02, -0.01])
    changed = returns.copy(deep=True)
    cutoff = 3
    changed.iloc[cutoff + 1 :] = [0.80, -0.70]

    original_prefix = calculate_core_metrics(returns.iloc[: cutoff + 1], 12)
    changed_prefix = calculate_core_metrics(changed.iloc[: cutoff + 1], 12)

    assert original_prefix == changed_prefix


def test_insufficient_valid_observations_are_rejected() -> None:
    returns = pd.Series([np.nan, 0.01, np.nan])

    with pytest.raises(ValueError, match="require at least 2"):
        annualized_volatility(returns, 12)
    with pytest.raises(ValueError, match="require at least 2"):
        sharpe_ratio(returns, 12)
    with pytest.raises(ValueError, match="require at least 2"):
        calculate_core_metrics(returns, 12)


def test_all_missing_returns_are_rejected() -> None:
    returns = pd.Series([np.nan, np.nan])

    with pytest.raises(ValueError, match="require at least 1"):
        total_return(returns)
