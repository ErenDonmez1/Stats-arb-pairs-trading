"""Focused tests for Milestone 7A core performance and risk metrics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from pairs_trading.analytics import (
    CorePerformanceMetrics,
    DrawdownEpisode,
    DrawdownMetrics,
    TradePerformanceMetrics,
    average_holding_period,
    average_loser,
    average_winner,
    annualized_return,
    annualized_volatility,
    calculate_drawdown_metrics,
    calculate_core_metrics,
    calculate_trade_metrics,
    calmar_ratio,
    downside_deviation,
    drawdown_duration,
    drawdown_episodes,
    drawdown_series,
    equity_curve_from_returns,
    maximum_drawdown,
    median_holding_period,
    payoff_ratio,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    trade_expectancy,
    turnover,
    win_rate,
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


def _recovered_drawdown_returns(index: pd.Index | None = None) -> pd.Series:
    """Return a path with peak row 1, trough row 3, and recovery row 4."""
    values = [0.0, 0.10, -0.20, -0.10, 0.40, 0.02]
    if index is None:
        index = pd.RangeIndex(len(values), name="drawdown_row")
    return pd.Series(values, index=index, name="strategy_return")


def test_equity_curve_uses_geometric_compounding() -> None:
    returns = pd.Series([0.10, -0.20, 0.05], index=["a", "b", "c"])
    expected = pd.Series(
        [1.10, 1.10 * 0.80, 1.10 * 0.80 * 1.05],
        index=returns.index,
        name="equity_curve",
    )

    pd.testing.assert_series_equal(equity_curve_from_returns(returns), expected)


def test_initial_equity_scales_entire_curve() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])

    unit_curve = equity_curve_from_returns(returns)
    scaled_curve = equity_curve_from_returns(returns, initial_equity=250.0)

    pd.testing.assert_series_equal(scaled_curve, unit_curve * 250.0)


def test_new_equity_highs_produce_zero_drawdown_and_no_episode() -> None:
    returns = pd.Series([0.0, 0.10, 0.02, 0.03])

    result = drawdown_series(returns)

    assert result.eq(0.0).all()
    assert drawdown_episodes(returns) == ()


def test_underwater_equity_produces_negative_drawdown() -> None:
    returns = _recovered_drawdown_returns()
    result = drawdown_series(returns)

    assert result.loc[2] < 0.0
    assert result.loc[3] < result.loc[2]
    assert result.loc[4] == 0.0


def test_drawdown_series_matches_manual_running_peak_calculation() -> None:
    returns = _recovered_drawdown_returns()
    equity = equity_curve_from_returns(returns)
    expected = (equity / equity.cummax() - 1.0).rename("drawdown")

    pd.testing.assert_series_equal(drawdown_series(returns), expected)


def test_maximum_drawdown_matches_manual_negative_decimal() -> None:
    returns = _recovered_drawdown_returns()
    expected = (0.792 / 1.10) - 1.0

    result = maximum_drawdown(returns)

    assert result == pytest.approx(expected)
    assert result == pytest.approx(-0.28)
    assert result < 0.0


def test_maximum_drawdown_peak_trough_and_recovery_are_identified() -> None:
    returns = _recovered_drawdown_returns()

    result = calculate_drawdown_metrics(returns, periods_per_year=12)

    assert result.maximum_drawdown_start == 1
    assert result.maximum_drawdown_trough == 3
    assert result.maximum_drawdown_recovery == 4
    assert result.maximum_drawdown_duration == 3


def test_unrecovered_maximum_drawdown_has_no_recovery() -> None:
    returns = pd.Series([0.0, 0.10, -0.10, -0.05, -0.02])

    result = calculate_drawdown_metrics(returns, periods_per_year=12)

    assert result.maximum_drawdown_start == 1
    assert result.maximum_drawdown_trough == 4
    assert result.maximum_drawdown_recovery is None


def test_multiple_drawdown_episodes_are_identified() -> None:
    returns = pd.Series([0.0, 0.10, -0.10, 0.12, -0.20, 0.30])

    episodes = drawdown_episodes(returns)

    assert len(episodes) == 2
    assert [(episode.peak_index, episode.start_index) for episode in episodes] == [
        (1, 2),
        (3, 4),
    ]
    assert [episode.recovery_index for episode in episodes] == [3, 5]
    assert all(episode.recovered for episode in episodes)


def test_first_maximum_drawdown_tie_is_selected_deterministically() -> None:
    returns = pd.Series([0.0, -0.10, 0.12, -0.10, 0.12])

    result = calculate_drawdown_metrics(returns, periods_per_year=12)

    assert result.maximum_drawdown_start == 0
    assert result.maximum_drawdown_trough == 1
    assert result.maximum_drawdown_recovery == 2


def test_first_episode_trough_tie_is_selected_deterministically() -> None:
    returns = pd.Series([0.0, -0.20, 0.0, 0.30])

    episode = drawdown_episodes(returns)[0]

    assert episode.trough_index == 1
    assert episode.maximum_drawdown == pytest.approx(-0.20)


def test_recovered_duration_counts_peak_to_recovery_positions() -> None:
    episode = drawdown_episodes(_recovered_drawdown_returns())[0]

    assert episode.peak_index == 1
    assert episode.recovery_index == 4
    assert episode.duration == 3


def test_unrecovered_duration_counts_peak_to_final_position() -> None:
    returns = pd.Series([0.0, 0.10, -0.10, -0.05, -0.02])
    episode = drawdown_episodes(returns)[0]

    assert episode.peak_index == 1
    assert episode.recovery_index is None
    assert episode.duration == 3


def test_longest_drawdown_duration_considers_unrecovered_episode() -> None:
    returns = pd.Series([0.0, -0.10, 0.20, -0.05, -0.05, -0.05])

    episodes = drawdown_episodes(returns)

    assert [episode.duration for episode in episodes] == [2, 3]
    assert drawdown_duration(returns) == 3
    assert calculate_drawdown_metrics(returns, 12).longest_drawdown_duration == 3


def test_current_drawdown_matches_final_drawdown_value() -> None:
    returns = pd.Series([0.0, 0.10, -0.10, -0.05])

    result = calculate_drawdown_metrics(returns, 12)

    assert result.current_drawdown == pytest.approx(drawdown_series(returns).iat[-1])
    assert result.current_drawdown < 0.0


def test_calmar_matches_annualized_return_over_absolute_drawdown() -> None:
    returns = _recovered_drawdown_returns()
    periods = 12
    expected = annualized_return(returns, periods) / abs(maximum_drawdown(returns))

    assert calmar_ratio(returns, periods) == pytest.approx(expected)


def test_zero_drawdown_returns_nan_calmar() -> None:
    returns = pd.Series([0.0, 0.01, 0.02, 0.03])

    assert maximum_drawdown(returns) == 0.0
    assert np.isnan(calmar_ratio(returns, 12))


def test_near_zero_drawdown_returns_nan_calmar() -> None:
    returns = pd.Series([0.0, -5e-13, 1e-12])

    assert maximum_drawdown(returns) < 0.0
    assert abs(maximum_drawdown(returns)) < 1e-12
    assert np.isnan(calmar_ratio(returns, 12))


def test_drawdown_missing_returns_follow_drop_policy_and_compress_duration() -> None:
    index = pd.Index(["peak", "missing", "trough", "recovery"], name="event")
    returns = pd.Series([0.0, np.nan, -0.10, 0.12], index=index)

    equity = equity_curve_from_returns(returns)
    episode = drawdown_episodes(returns)[0]

    assert equity.index.tolist() == ["peak", "trough", "recovery"]
    assert episode.peak_index == "peak"
    assert episode.start_index == "trough"
    assert episode.recovery_index == "recovery"
    assert episode.duration == 2


def test_drawdown_missing_returns_are_not_filled_or_interpolated() -> None:
    returns = pd.Series([0.0, np.nan, -0.10, 0.12])

    dropped = drawdown_series(returns.dropna())
    actual = drawdown_series(returns)

    pd.testing.assert_series_equal(actual, dropped)
    assert len(actual) == 3


def test_drawdown_rejects_unsupported_missing_policy() -> None:
    with pytest.raises(ValueError, match="missing_policy"):
        calculate_drawdown_metrics(
            pd.Series([0.0, -0.10, 0.12]),
            12,
            missing_policy="fill",
        )


@pytest.mark.parametrize("invalid_return", [-1.0, -1.01])
@pytest.mark.parametrize(
    "metric",
    [
        lambda series: equity_curve_from_returns(series),
        lambda series: drawdown_series(series),
        lambda series: maximum_drawdown(series),
        lambda series: calculate_drawdown_metrics(series, 12),
    ],
)
def test_drawdown_metrics_reject_total_or_greater_capital_loss(
    invalid_return: float,
    metric: Callable[[pd.Series], Any],
) -> None:
    returns = pd.Series([0.0, invalid_return, 0.10])

    with pytest.raises(ValueError, match="exceed -1"):
        metric(returns)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), 0.0, -1.0, "1", np.nan, np.inf, -np.inf],
)
def test_invalid_initial_equity_is_rejected(invalid: Any) -> None:
    returns = pd.Series([0.0, -0.10, 0.12])

    with pytest.raises((TypeError, ValueError), match="initial_equity"):
        equity_curve_from_returns(returns, initial_equity=invalid)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), 0, -1, 12.0, 1.5, "12", np.nan, None],
)
def test_drawdown_metrics_reject_invalid_periods_per_year(invalid: Any) -> None:
    returns = pd.Series([0.0, -0.10, 0.12])

    with pytest.raises((TypeError, ValueError), match="periods_per_year"):
        calculate_drawdown_metrics(returns, invalid)


@pytest.mark.parametrize(
    "metric",
    [
        lambda value: equity_curve_from_returns(value),
        lambda value: drawdown_series(value),
        lambda value: drawdown_episodes(value),
        lambda value: maximum_drawdown(value),
        lambda value: drawdown_duration(value),
        lambda value: calmar_ratio(value, 12),
        lambda value: calculate_drawdown_metrics(value, 12),
    ],
)
def test_drawdown_functions_reject_dataframe_inputs(
    metric: Callable[[Any], Any],
) -> None:
    frame = pd.DataFrame({"return": [0.0, -0.10, 0.12]})

    with pytest.raises(TypeError, match="pandas Series"):
        metric(frame)


@pytest.mark.parametrize("invalid", [[0.0, -0.10], np.array([0.0, -0.10]), 0.1])
def test_drawdown_functions_reject_non_series_inputs(invalid: Any) -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        calculate_drawdown_metrics(invalid, 12)


@pytest.mark.parametrize("invalid", ["bad", True, np.bool_(False), 1 + 2j])
def test_drawdown_rejects_non_numeric_returns(invalid: Any) -> None:
    returns = pd.Series([0.0, invalid, -0.10], dtype=object)

    with pytest.raises(ValueError, match="real numeric"):
        calculate_drawdown_metrics(returns, 12)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_drawdown_rejects_infinite_returns(invalid: float) -> None:
    returns = pd.Series([0.0, invalid, -0.10])

    with pytest.raises(ValueError, match="finite"):
        calculate_drawdown_metrics(returns, 12)


def test_drawdown_rejects_duplicate_indices() -> None:
    returns = pd.Series([0.0, -0.10, 0.12], index=[0, 0, 1])

    with pytest.raises(ValueError, match="unique index"):
        calculate_drawdown_metrics(returns, 12)


def test_drawdown_analytics_do_not_mutate_input_series() -> None:
    returns = _recovered_drawdown_returns()
    returns.attrs["source"] = "backtest"
    before = returns.copy(deep=True)

    calculate_drawdown_metrics(returns, 12, initial_equity=100.0)

    pd.testing.assert_series_equal(returns, before)
    assert returns.attrs == before.attrs


def test_drawdown_analytics_are_deterministic() -> None:
    returns = _recovered_drawdown_returns()

    first_metrics = calculate_drawdown_metrics(returns, 12)
    second_metrics = calculate_drawdown_metrics(returns, 12)

    assert first_metrics == second_metrics
    assert drawdown_episodes(returns) == drawdown_episodes(returns)
    pd.testing.assert_series_equal(
        equity_curve_from_returns(returns),
        equity_curve_from_returns(returns),
    )


def test_non_datetime_index_and_labels_are_preserved() -> None:
    index = pd.Index(["p0", "p1", "p2", "p3", "p4", "p5"], name="sequence")
    returns = _recovered_drawdown_returns(index)

    equity = equity_curve_from_returns(returns)
    drawdowns = drawdown_series(returns)
    episode = drawdown_episodes(returns)[0]

    pd.testing.assert_index_equal(equity.index, index, exact=True)
    pd.testing.assert_index_equal(drawdowns.index, index, exact=True)
    assert episode.peak_index == "p1"
    assert episode.trough_index == "p3"
    assert episode.recovery_index == "p4"


def test_timezone_aware_drawdown_index_values_are_preserved() -> None:
    index = pd.date_range(
        "2025-03-27",
        periods=6,
        freq="h",
        tz="Europe/London",
        name="timestamp",
    )
    returns = _recovered_drawdown_returns(index)

    equity = equity_curve_from_returns(returns)
    drawdowns = drawdown_series(returns)
    episode = drawdown_episodes(returns)[0]

    pd.testing.assert_index_equal(equity.index, index, exact=True)
    pd.testing.assert_index_equal(drawdowns.index, index, exact=True)
    assert episode.peak_index == index[1]
    assert episode.recovery_index == index[4]


def test_drawdown_metrics_is_immutable() -> None:
    result = calculate_drawdown_metrics(_recovered_drawdown_returns(), 12)

    with pytest.raises(FrozenInstanceError):
        result.maximum_drawdown = 0.0  # type: ignore[misc]


def test_drawdown_episode_is_immutable() -> None:
    episode = drawdown_episodes(_recovered_drawdown_returns())[0]

    with pytest.raises(FrozenInstanceError):
        episode.duration = 0  # type: ignore[misc]


def test_calculate_drawdown_metrics_matches_public_functions() -> None:
    returns = _recovered_drawdown_returns()
    periods = 12

    result = calculate_drawdown_metrics(returns, periods)
    episodes = drawdown_episodes(returns)
    maximum_episode = min(episodes, key=lambda episode: episode.maximum_drawdown)

    assert isinstance(result, DrawdownMetrics)
    assert result.maximum_drawdown == pytest.approx(maximum_drawdown(returns))
    assert result.maximum_drawdown_start == maximum_episode.peak_index
    assert result.maximum_drawdown_trough == maximum_episode.trough_index
    assert result.maximum_drawdown_recovery == maximum_episode.recovery_index
    assert result.maximum_drawdown_duration == maximum_episode.duration
    assert result.longest_drawdown_duration == drawdown_duration(returns)
    assert result.calmar_ratio == pytest.approx(calmar_ratio(returns, periods))
    assert result.observations == len(equity_curve_from_returns(returns))
    assert result.current_drawdown == pytest.approx(drawdown_series(returns).iat[-1])


def test_running_peak_is_non_decreasing_and_equity_is_positive() -> None:
    equity = equity_curve_from_returns(_recovered_drawdown_returns())
    running_peak = equity.cummax()

    assert equity.gt(0.0).all()
    assert running_peak.diff().dropna().ge(0.0).all()


def test_drawdown_never_becomes_positive() -> None:
    drawdowns = drawdown_series(_recovered_drawdown_returns())

    assert drawdowns.le(0.0).all()
    assert drawdowns.max() == 0.0


def test_drawdown_episode_structural_invariants_hold() -> None:
    returns = pd.Series([0.0, 0.10, -0.10, 0.12, -0.20, 0.30, -0.05])
    equity = equity_curve_from_returns(returns)
    drawdowns = drawdown_series(returns)
    positions = {label: position for position, label in enumerate(returns.index)}

    for episode in drawdown_episodes(returns):
        peak_position = positions[episode.peak_index]
        trough_position = positions[episode.trough_index]
        end_position = (
            len(returns) - 1
            if episode.recovery_index is None
            else positions[episode.recovery_index]
        )
        assert peak_position < trough_position <= end_position
        assert episode.maximum_drawdown == pytest.approx(
            drawdowns.loc[episode.trough_index]
        )
        assert isinstance(episode.duration, int)
        assert episode.duration >= 0
        assert episode.duration == end_position - peak_position
        if episode.recovered:
            assert episode.recovery_index is not None
            assert equity.loc[episode.recovery_index] >= equity.loc[episode.peak_index]
        else:
            assert episode.recovery_index is None


def test_maximum_drawdown_equals_drawdown_series_minimum() -> None:
    returns = _recovered_drawdown_returns()

    assert maximum_drawdown(returns) == pytest.approx(drawdown_series(returns).min())


def test_drawdown_prefix_is_unchanged_by_strictly_later_returns() -> None:
    returns = pd.Series([0.0, 0.10, -0.20, -0.10, 0.40, 0.02, -0.03])
    changed = returns.copy(deep=True)
    cutoff = 3
    changed.iloc[cutoff + 1 :] = [0.90, -0.80, 0.70]

    original_prefix = returns.iloc[: cutoff + 1]
    changed_prefix = changed.iloc[: cutoff + 1]

    pd.testing.assert_series_equal(
        equity_curve_from_returns(original_prefix),
        equity_curve_from_returns(changed_prefix),
    )
    pd.testing.assert_series_equal(
        drawdown_series(original_prefix),
        drawdown_series(changed_prefix),
    )
    assert calculate_drawdown_metrics(
        original_prefix,
        12,
    ) == calculate_drawdown_metrics(changed_prefix, 12)


def _trade_ledger(
    net_pnl: list[Any] | None = None,
    holding_periods: list[Any] | None = None,
) -> pd.DataFrame:
    """Return a minimal completed-trade ledger accepted by 7C analytics."""
    if net_pnl is None:
        net_pnl = [100.0, -50.0, 0.0, np.nan]
    if holding_periods is None:
        holding_periods = [3, 5, 4, 7][: len(net_pnl)]
    count = len(net_pnl)
    return pd.DataFrame(
        {
            "trade_id": np.arange(1, count + 1),
            "net_pnl": net_pnl,
            "holding_period_rows": holding_periods,
            "entry_gross_notional": np.full(count, 1_000.0),
        },
        index=pd.Index([f"trade-{number}" for number in range(count)], name="row"),
    )


def _turnover_accounting() -> pd.DataFrame:
    """Return entry, hold, rebalance, and exit traded-notional rows."""
    return pd.DataFrame(
        {
            "traded_notional_y": [1_000.0, 0.0, 200.0, 1_000.0],
            "traded_notional_x": [500.0, 0.0, 100.0, 500.0],
        },
        index=pd.Index(["entry", "hold", "rebalance", "exit"], name="event"),
    )


def test_win_rate_matches_manual_known_trade_classification() -> None:
    ledger = _trade_ledger()

    result = calculate_trade_metrics(ledger)

    assert result.trades == 4
    assert result.known_pnl_trades == 3
    assert result.unknown_pnl_trades == 1
    assert result.winning_trades == 1
    assert result.losing_trades == 1
    assert result.breakeven_trades == 1
    assert result.win_rate == pytest.approx(1.0 / 3.0)
    assert win_rate(ledger) == pytest.approx(1.0 / 3.0)


def test_breakeven_tolerance_is_scale_aware_and_deterministic() -> None:
    ledger = _trade_ledger([5e-13, -5e-13, 2e-12, -2e-12])

    first = calculate_trade_metrics(ledger)
    second = calculate_trade_metrics(ledger)

    assert first.winning_trades == 1
    assert first.losing_trades == 1
    assert first.breakeven_trades == 2
    assert first.winning_trades == second.winning_trades
    assert first.losing_trades == second.losing_trades
    assert first.breakeven_trades == second.breakeven_trades


def test_average_winner_is_correct() -> None:
    ledger = _trade_ledger([100.0, 50.0, -25.0, 0.0])

    assert average_winner(ledger) == pytest.approx(75.0)


def test_no_winners_returns_nan_average_winner() -> None:
    ledger = _trade_ledger([-20.0, 0.0])

    assert np.isnan(average_winner(ledger))


def test_average_loser_is_correct_and_negative() -> None:
    ledger = _trade_ledger([100.0, -50.0, -30.0, 0.0])

    result = average_loser(ledger)

    assert result == pytest.approx(-40.0)
    assert result < 0.0


def test_no_losers_returns_nan_average_loser() -> None:
    ledger = _trade_ledger([20.0, 0.0])

    assert np.isnan(average_loser(ledger))


def test_payoff_ratio_matches_manual_calculation() -> None:
    ledger = _trade_ledger([100.0, 50.0, -50.0, -25.0])

    assert payoff_ratio(ledger) == pytest.approx(75.0 / 37.5)


def test_no_loser_denominator_returns_nan_payoff_ratio() -> None:
    ledger = _trade_ledger([100.0, 50.0, 0.0])

    assert np.isnan(payoff_ratio(ledger))


def test_expectancy_matches_probability_weighted_manual_calculation() -> None:
    ledger = _trade_ledger([100.0, -50.0, 0.0])
    expected = (1.0 / 3.0 * 100.0) + (1.0 / 3.0 * -50.0)

    assert trade_expectancy(ledger) == pytest.approx(expected)


def test_expectancy_matches_direct_mean_when_breakeven_is_zero() -> None:
    ledger = _trade_ledger([100.0, -50.0, 0.0])

    assert trade_expectancy(ledger) == pytest.approx(ledger["net_pnl"].mean())


def test_profit_factor_and_gross_profit_loss_match_manual_values() -> None:
    ledger = _trade_ledger([100.0, 50.0, -40.0, -10.0])

    result = calculate_trade_metrics(ledger)

    assert result.gross_profit == pytest.approx(150.0)
    assert result.gross_loss == pytest.approx(50.0)
    assert result.profit_factor == pytest.approx(3.0)
    assert profit_factor(ledger) == pytest.approx(3.0)


def test_no_losing_trades_returns_nan_profit_factor() -> None:
    ledger = _trade_ledger([100.0, 50.0, 0.0])

    assert np.isnan(profit_factor(ledger))
    assert not np.isinf(profit_factor(ledger))


def test_average_and_median_holding_period_are_correct() -> None:
    ledger = _trade_ledger()

    assert average_holding_period(ledger) == pytest.approx(4.75)
    assert median_holding_period(ledger) == pytest.approx(4.5)


def test_holding_periods_include_unknown_pnl_trades() -> None:
    ledger = _trade_ledger([100.0, np.nan], [2, 10])

    assert average_holding_period(ledger) == pytest.approx(6.0)
    assert median_holding_period(ledger) == pytest.approx(6.0)


@pytest.mark.parametrize("invalid", [True, np.bool_(False), -1, 1.5, 2.0, "2", np.nan])
def test_invalid_holding_periods_are_rejected(invalid: Any) -> None:
    ledger = _trade_ledger([10.0], [invalid])

    with pytest.raises(ValueError, match="holding_period_rows"):
        calculate_trade_metrics(ledger)


def test_unknown_pnl_is_excluded_and_never_classified_as_breakeven() -> None:
    ledger = _trade_ledger([100.0, -50.0, 0.0, np.nan])

    result = calculate_trade_metrics(ledger)

    assert result.known_pnl_trades == 3
    assert result.unknown_pnl_trades == 1
    assert result.breakeven_trades == 1
    assert result.winning_trades + result.losing_trades + result.breakeven_trades == 3


def test_all_unknown_pnl_returns_nan_denominator_statistics() -> None:
    ledger = _trade_ledger([np.nan, np.nan], [2, 5])

    result = calculate_trade_metrics(ledger)

    assert result.trades == 2
    assert result.known_pnl_trades == 0
    assert result.unknown_pnl_trades == 2
    assert result.winning_trades == result.losing_trades == result.breakeven_trades == 0
    for value in (
        result.win_rate,
        result.average_winner,
        result.average_loser,
        result.payoff_ratio,
        result.expectancy,
        result.profit_factor,
    ):
        assert np.isnan(value)


def test_total_net_pnl_is_unknown_if_any_trade_pnl_is_unknown() -> None:
    unknown = calculate_trade_metrics(_trade_ledger())
    known = calculate_trade_metrics(_trade_ledger([100.0, -50.0, 0.0]))

    assert np.isnan(unknown.total_net_pnl)
    assert known.total_net_pnl == pytest.approx(50.0)


def test_empty_ledger_returns_documented_empty_metrics() -> None:
    ledger = _trade_ledger([], [])

    result = calculate_trade_metrics(ledger)

    assert result.trades == 0
    assert result.known_pnl_trades == 0
    assert result.unknown_pnl_trades == 0
    assert result.winning_trades == result.losing_trades == result.breakeven_trades == 0
    assert result.gross_profit == 0.0
    assert result.gross_loss == 0.0
    assert result.total_net_pnl == 0.0
    for value in (
        result.win_rate,
        result.average_winner,
        result.average_loser,
        result.payoff_ratio,
        result.expectancy,
        result.profit_factor,
        result.average_holding_period,
        result.median_holding_period,
        result.turnover,
    ):
        assert np.isnan(value)


def test_duplicate_trade_ids_are_rejected() -> None:
    ledger = _trade_ledger([10.0, -5.0])
    ledger["trade_id"] = [7, 7]

    with pytest.raises(ValueError, match="trade_id.*unique"):
        calculate_trade_metrics(ledger)


@pytest.mark.parametrize(
    "missing_column",
    ["trade_id", "net_pnl", "holding_period_rows", "entry_gross_notional"],
)
def test_missing_required_trade_ledger_columns_are_rejected(
    missing_column: str,
) -> None:
    ledger = _trade_ledger().drop(columns=missing_column)

    with pytest.raises(ValueError, match="missing required columns"):
        calculate_trade_metrics(ledger)


@pytest.mark.parametrize("invalid", [True, "10", np.inf, -np.inf])
def test_invalid_known_net_pnl_values_are_rejected(invalid: Any) -> None:
    ledger = _trade_ledger([invalid])

    with pytest.raises(ValueError, match="net_pnl"):
        calculate_trade_metrics(ledger)


@pytest.mark.parametrize("invalid", [True, "1000", 0.0, -1.0, np.nan, np.inf])
def test_invalid_entry_gross_notional_values_are_rejected(invalid: Any) -> None:
    ledger = _trade_ledger([10.0])
    ledger["entry_gross_notional"] = ledger["entry_gross_notional"].astype(object)
    ledger.loc["trade-0", "entry_gross_notional"] = invalid

    with pytest.raises(ValueError, match="entry_gross_notional"):
        calculate_trade_metrics(ledger)


def test_trade_ledger_dataframe_and_index_contracts_are_enforced() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        calculate_trade_metrics([1, 2])

    ledger = _trade_ledger([10.0, -5.0])
    ledger.index = pd.Index([0, 0])
    with pytest.raises(ValueError, match="unique index"):
        calculate_trade_metrics(ledger)


def test_trade_metrics_do_not_mutate_input_ledger() -> None:
    ledger = _trade_ledger()
    ledger.attrs["source"] = "backtest-ledger"
    before = ledger.copy(deep=True)

    calculate_trade_metrics(ledger)

    pd.testing.assert_frame_equal(ledger, before)
    assert ledger.attrs == before.attrs


def test_trade_metrics_are_deterministic() -> None:
    ledger = _trade_ledger([100.0, -50.0, 0.0])
    accounting = _turnover_accounting()

    first = calculate_trade_metrics(
        ledger,
        accounting=accounting,
        initial_capital=10_000.0,
    )
    second = calculate_trade_metrics(
        ledger,
        accounting=accounting,
        initial_capital=10_000.0,
    )

    assert first == second


def test_exact_turnover_uses_total_traded_notional_over_initial_capital() -> None:
    accounting = _turnover_accounting()
    expected = accounting[["traded_notional_y", "traded_notional_x"]].sum().sum()

    assert turnover(accounting, 10_000.0) == pytest.approx(expected / 10_000.0)


def test_entry_exit_and_rebalance_notional_all_contribute_to_turnover() -> None:
    accounting = _turnover_accounting()

    total = turnover(accounting, 10_000.0)
    without_exit = turnover(accounting.iloc[:-1], 10_000.0)
    without_rebalance = turnover(accounting.drop(index="rebalance"), 10_000.0)

    assert total > without_exit
    assert total > without_rebalance


def test_zero_traded_notional_produces_zero_turnover() -> None:
    accounting = pd.DataFrame(
        {"traded_notional_y": [0.0, 0.0], "traded_notional_x": [0.0, 0.0]}
    )

    assert turnover(accounting, 10_000.0) == 0.0


@pytest.mark.parametrize("invalid", [True, "10", -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", ["traded_notional_y", "traded_notional_x"])
def test_invalid_traded_notional_values_are_rejected(
    invalid: Any,
    column: str,
) -> None:
    accounting = _turnover_accounting().astype(object)
    accounting.loc["entry", column] = invalid

    with pytest.raises(ValueError, match=column):
        turnover(accounting, 10_000.0)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), 0.0, -1.0, "10000", np.nan, np.inf, -np.inf],
)
def test_invalid_turnover_initial_capital_is_rejected(invalid: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="initial_capital"):
        turnover(_turnover_accounting(), invalid)


def test_turnover_dataframe_contract_is_enforced() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        turnover([1, 2], 10_000.0)

    missing = _turnover_accounting().drop(columns="traded_notional_x")
    with pytest.raises(ValueError, match="missing required columns"):
        turnover(missing, 10_000.0)

    duplicated = _turnover_accounting()
    duplicated.index = pd.Index([0, 0, 1, 2])
    with pytest.raises(ValueError, match="unique index"):
        turnover(duplicated, 10_000.0)


def test_turnover_does_not_mutate_accounting_dataframe() -> None:
    accounting = _turnover_accounting()
    accounting.attrs["source"] = "financed-accounting"
    before = accounting.copy(deep=True)

    turnover(accounting, 10_000.0)

    pd.testing.assert_frame_equal(accounting, before)
    assert accounting.attrs == before.attrs


def test_trade_performance_metrics_is_immutable() -> None:
    result = calculate_trade_metrics(_trade_ledger([100.0, -50.0]))

    assert isinstance(result, TradePerformanceMetrics)
    with pytest.raises(FrozenInstanceError):
        result.trades = 0  # type: ignore[misc]


def test_calculate_trade_metrics_matches_individual_public_functions() -> None:
    ledger = _trade_ledger([100.0, -50.0, 0.0])
    accounting = _turnover_accounting()

    result = calculate_trade_metrics(
        ledger,
        accounting=accounting,
        initial_capital=10_000.0,
    )

    assert result.win_rate == pytest.approx(win_rate(ledger))
    assert result.average_winner == pytest.approx(average_winner(ledger))
    assert result.average_loser == pytest.approx(average_loser(ledger))
    assert result.payoff_ratio == pytest.approx(payoff_ratio(ledger))
    assert result.expectancy == pytest.approx(trade_expectancy(ledger))
    assert result.profit_factor == pytest.approx(profit_factor(ledger))
    assert result.average_holding_period == pytest.approx(
        average_holding_period(ledger)
    )
    assert result.median_holding_period == pytest.approx(
        median_holding_period(ledger)
    )
    assert result.turnover == pytest.approx(turnover(accounting, 10_000.0))


def test_turnover_inputs_must_be_supplied_together() -> None:
    ledger = _trade_ledger([100.0, -50.0])

    with pytest.raises(ValueError, match="supplied together"):
        calculate_trade_metrics(ledger, accounting=_turnover_accounting())
    with pytest.raises(ValueError, match="supplied together"):
        calculate_trade_metrics(ledger, initial_capital=10_000.0)


def test_later_trades_do_not_change_explicit_ledger_prefix_metrics() -> None:
    ledger = _trade_ledger([100.0, -50.0, 20.0, -10.0])
    changed = ledger.copy(deep=True)
    changed.loc["trade-2":, "net_pnl"] = [2_000.0, -1_000.0]
    prefix = ledger.iloc[:2]
    changed_prefix = changed.iloc[:2]
    accounting_prefix = _turnover_accounting().iloc[:2]

    original = calculate_trade_metrics(
        prefix,
        accounting=accounting_prefix,
        initial_capital=10_000.0,
    )
    modified = calculate_trade_metrics(
        changed_prefix,
        accounting=accounting_prefix,
        initial_capital=10_000.0,
    )

    assert original == modified
