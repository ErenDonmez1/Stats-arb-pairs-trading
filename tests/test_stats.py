"""Tests for static and causal rolling OLS spread estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading.stats import kalman_ols_spread, ols_spread, rolling_ols_spread


def known_relationship(
    periods: int = 80,
    alpha: float = 0.35,
    beta: float = 1.4,
) -> tuple[pd.Series, pd.Series]:
    """Return exact positive prices with a known log-linear relationship."""
    index = pd.bdate_range("2022-01-03", periods=periods)
    log_x = np.linspace(3.5, 4.5, periods)
    x = pd.Series(np.exp(log_x), index=index, name="x_price")
    y = pd.Series(np.exp(alpha + beta * log_x), index=index, name="y_price")
    return y, x


def noisy_relationship(
    periods: int = 100,
    seed: int = 21,
) -> tuple[pd.Series, pd.Series]:
    """Return deterministic prices with a varying rolling relationship."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=periods)
    log_x = 4.0 + np.cumsum(rng.normal(0.001, 0.015, periods))
    residual = rng.normal(0.0, 0.01, periods)
    log_y = 0.2 + 1.15 * log_x + residual
    return (
        pd.Series(np.exp(log_y), index=index, name="y_price"),
        pd.Series(np.exp(log_x), index=index, name="x_price"),
    )


def dynamic_relationship(
    periods: int = 600,
    seed: int = 17,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return deterministic prices with a slowly changing true hedge ratio."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-02", periods=periods)
    phase = np.linspace(0.0, 6.0 * np.pi, periods)
    log_x = 3.5 + 0.8 * np.sin(phase) + rng.normal(0.0, 0.04, periods)
    true_beta = 1.15 + 0.18 * np.sin(phase / 3.0)
    true_alpha = 0.25 + 0.04 * np.cos(phase / 4.0)
    log_y = true_alpha + true_beta * log_x + rng.normal(0.0, 0.006, periods)
    return (
        pd.Series(np.exp(log_y), index=index, name="y_price"),
        pd.Series(np.exp(log_x), index=index, name="x_price"),
        pd.Series(true_beta, index=index, name="true_beta"),
    )


def test_static_ols_recovers_known_alpha_and_beta() -> None:
    y, x = known_relationship()

    _, alpha, beta = ols_spread(y, x)

    assert alpha == pytest.approx(0.35, abs=1e-10)
    assert beta == pytest.approx(1.4, abs=1e-10)


def test_static_spread_matches_log_price_definition() -> None:
    y, x = noisy_relationship()

    spread, alpha, beta = ols_spread(y, x)
    expected = np.log(y) - alpha - beta * np.log(x)

    pd.testing.assert_series_equal(
        spread, expected.rename("spread"), check_exact=False, atol=1e-12
    )


def test_missing_observations_are_aligned_on_shared_index() -> None:
    index = pd.bdate_range("2024-01-01", periods=7)
    x = pd.Series(np.exp(np.linspace(2.0, 2.6, 6)), index=index[1:])
    y = pd.Series(
        np.exp(0.1 + 1.2 * np.linspace(2.0, 2.6, 7)),
        index=index,
    )
    y.iloc[3] = np.nan

    spread, _, _ = ols_spread(y, x)

    expected_index = index[1:].difference(pd.DatetimeIndex([index[3]]), sort=False)
    assert spread.index.equals(expected_index)


@pytest.mark.parametrize("invalid", [0.0, -1.0, "not-a-price", np.inf])
def test_invalid_prices_are_rejected(invalid: object) -> None:
    y, x = known_relationship(periods=10)
    y = y.astype(object)
    y.iloc[4] = invalid

    with pytest.raises(ValueError):
        ols_spread(y, x)
    with pytest.raises(ValueError):
        rolling_ols_spread(y, x, lookback=4)
    with pytest.raises(ValueError):
        kalman_ols_spread(y, x)


def test_estimators_do_not_mutate_inputs() -> None:
    y, x = noisy_relationship(periods=30)
    y_before = y.copy(deep=True)
    x_before = x.copy(deep=True)

    ols_spread(y, x)
    rolling_ols_spread(y, x, lookback=10)

    pd.testing.assert_series_equal(y, y_before)
    pd.testing.assert_series_equal(x, x_before)


def test_rolling_output_has_expected_columns() -> None:
    y, x = noisy_relationship(periods=30)

    result = rolling_ols_spread(y, x, lookback=10)

    assert list(result.columns) == ["alpha", "beta", "spread"]
    assert result.index.equals(y.index)


def test_rolling_warmup_values_remain_nan() -> None:
    y, x = noisy_relationship(periods=30)
    lookback = 8

    result = rolling_ols_spread(y, x, lookback=lookback)

    assert result.iloc[:lookback].isna().all().all()
    assert result.iloc[lookback].notna().all()


def test_rolling_estimate_uses_only_observations_through_previous_date() -> None:
    y, x = noisy_relationship(periods=40)
    lookback = 12
    target_position = 20

    result = rolling_ols_spread(y, x, lookback=lookback)
    prior_y = y.iloc[target_position - lookback : target_position]
    prior_x = x.iloc[target_position - lookback : target_position]
    _, expected_alpha, expected_beta = ols_spread(prior_y, prior_x)
    target_date = y.index[target_position]

    assert result.loc[target_date, "alpha"] == pytest.approx(
        expected_alpha, abs=1e-10
    )
    assert result.loc[target_date, "beta"] == pytest.approx(
        expected_beta, abs=1e-10
    )


def test_future_price_changes_do_not_alter_earlier_rolling_estimates() -> None:
    y, x = noisy_relationship(periods=60)
    lookback = 15
    cutoff_position = 40
    changed_y = y.copy()
    changed_x = x.copy()
    changed_y.iloc[cutoff_position + 1 :] *= 3.0
    changed_x.iloc[cutoff_position + 1 :] *= 0.4

    original = rolling_ols_spread(y, x, lookback=lookback)
    changed = rolling_ols_spread(changed_y, changed_x, lookback=lookback)

    pd.testing.assert_frame_equal(
        original.iloc[: cutoff_position + 1],
        changed.iloc[: cutoff_position + 1],
    )


@pytest.mark.parametrize("lookback", [True, 1, 0, -2, 5.5, "5"])
def test_invalid_lookbacks_are_rejected(lookback: object) -> None:
    y, x = noisy_relationship(periods=20)

    with pytest.raises(ValueError, match="lookback"):
        rolling_ols_spread(y, x, lookback=lookback)  # type: ignore[arg-type]


def test_insufficient_observations_are_rejected() -> None:
    y, x = known_relationship(periods=5)

    with pytest.raises(ValueError, match=r"lookback \+ 1"):
        rolling_ols_spread(y, x, lookback=5)
    with pytest.raises(ValueError, match="at least three"):
        ols_spread(y.iloc[:2], x.iloc[:2])


def test_constant_explanatory_series_has_explicit_variance_policy() -> None:
    index = pd.bdate_range("2024-01-01", periods=12)
    x = pd.Series(100.0, index=index)
    y = pd.Series(np.exp(np.linspace(4.0, 4.2, len(index))), index=index)

    with pytest.raises(ValueError, match="non-zero variance"):
        ols_spread(y, x)

    rolling = rolling_ols_spread(y, x, lookback=5)
    assert rolling.isna().all().all()


def test_kalman_output_has_expected_columns_and_aligned_index() -> None:
    y, x = noisy_relationship(periods=30)

    result = kalman_ols_spread(y, x)

    assert list(result.columns) == [
        "alpha",
        "beta",
        "spread",
        "innovation",
        "innovation_variance",
    ]
    assert result.index.equals(y.index)


def test_kalman_estimates_are_finite_from_first_update() -> None:
    y, x = noisy_relationship(periods=50)

    result = kalman_ols_spread(y, x)

    assert np.isfinite(result.to_numpy()).all()
    assert result["innovation_variance"].gt(0).all()


def test_kalman_does_not_mutate_inputs() -> None:
    y, x = noisy_relationship(periods=40)
    y_before = y.copy(deep=True)
    x_before = x.copy(deep=True)

    kalman_ols_spread(y, x)

    pd.testing.assert_series_equal(y, y_before)
    pd.testing.assert_series_equal(x, x_before)


def test_kalman_is_deterministic() -> None:
    y, x = noisy_relationship(periods=70)

    first = kalman_ols_spread(y, x)
    second = kalman_ols_spread(y, x)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    "parameter",
    [
        "transition_variance",
        "observation_variance",
        "initial_state_covariance",
    ],
)
@pytest.mark.parametrize(
    "invalid",
    [True, "0.1", 0.0, -1.0, np.nan, np.inf],
)
def test_invalid_kalman_parameters_are_rejected(
    parameter: str,
    invalid: object,
) -> None:
    y, x = noisy_relationship(periods=20)

    with pytest.raises(ValueError, match=parameter):
        kalman_ols_spread(y, x, **{parameter: invalid})  # type: ignore[arg-type]


def test_kalman_future_data_does_not_change_estimates_through_cutoff() -> None:
    y, x = noisy_relationship(periods=100)
    cutoff = 60
    changed_y = y.copy()
    changed_x = x.copy()
    changed_y.iloc[cutoff + 1 :] *= 4.0
    changed_x.iloc[cutoff + 1 :] *= 0.3

    original = kalman_ols_spread(y, x)
    changed = kalman_ols_spread(changed_y, changed_x)

    pd.testing.assert_frame_equal(
        original.iloc[: cutoff + 1],
        changed.iloc[: cutoff + 1],
    )


def test_kalman_recovers_slowly_varying_hedge_ratio() -> None:
    y, x, true_beta = dynamic_relationship()

    result = kalman_ols_spread(
        y,
        x,
        transition_variance=2e-5,
        observation_variance=4e-5,
        initial_state_covariance=5.0,
    )
    comparison = pd.concat([result["beta"], true_beta], axis=1).iloc[100:]
    mean_absolute_error = (
        comparison["beta"] - comparison["true_beta"]
    ).abs().mean()

    assert mean_absolute_error < 0.08
    assert comparison.corr().iloc[0, 1] > 0.85


def test_kalman_spread_matches_posterior_state_definition() -> None:
    y, x = noisy_relationship(periods=60)

    result = kalman_ols_spread(y, x)
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    expected = (
        np.log(aligned["y"])
        - result["alpha"]
        - result["beta"] * np.log(aligned["x"])
    )

    pd.testing.assert_series_equal(
        result["spread"],
        expected.rename("spread"),
        check_exact=False,
        atol=1e-12,
    )


def test_kalman_first_innovation_and_variance_match_initial_policy() -> None:
    y, x = noisy_relationship(periods=20)
    observation_variance = 0.02
    initial_covariance = 3.0

    result = kalman_ols_spread(
        y,
        x,
        transition_variance=0.01,
        observation_variance=observation_variance,
        initial_state_covariance=initial_covariance,
    )
    first_log_x = float(np.log(x.iloc[0]))
    first_log_y = float(np.log(y.iloc[0]))
    design = np.array([1.0, first_log_x])
    expected_innovation = first_log_y - first_log_x
    expected_variance = float(
        design @ (initial_covariance * np.eye(2)) @ design
        + observation_variance
    )

    assert result["innovation"].iloc[0] == pytest.approx(expected_innovation)
    assert result["innovation_variance"].iloc[0] == pytest.approx(
        expected_variance
    )


def test_kalman_remains_stable_for_long_series() -> None:
    y, x = noisy_relationship(periods=5_000)

    result = kalman_ols_spread(
        y,
        x,
        transition_variance=1e-6,
        observation_variance=1e-4,
    )

    assert len(result) == 5_000
    assert np.isfinite(result.to_numpy()).all()
    assert result["innovation_variance"].gt(0).all()


def test_kalman_rejects_constant_explanatory_series() -> None:
    index = pd.bdate_range("2024-01-01", periods=20)
    x = pd.Series(100.0, index=index)
    y = pd.Series(np.exp(np.linspace(4.0, 4.3, len(index))), index=index)

    with pytest.raises(ValueError, match="non-zero variance"):
        kalman_ols_spread(y, x)


def test_kalman_missing_observations_use_shared_complete_index() -> None:
    y, x = noisy_relationship(periods=30)
    y.iloc[5] = np.nan
    x = x.iloc[2:].copy()
    x.iloc[8] = np.nan

    result = kalman_ols_spread(y, x)
    expected_index = pd.concat(
        [y.rename("y"), x.rename("x")], axis=1, join="inner"
    ).dropna().index

    assert result.index.equals(expected_index)
