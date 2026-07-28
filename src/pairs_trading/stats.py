"""Static and causal rolling OLS spread estimation."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _align_and_validate_prices(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    """Align two price series and validate their usable shared observations."""
    if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
        raise TypeError("y and x must both be pandas Series.")
    if not y.index.is_unique or not x.index.is_unique:
        raise ValueError("y and x must have unique indexes.")

    # Copies and explicit names keep the caller's objects and metadata untouched.
    aligned = pd.concat(
        [y.copy(deep=True).rename("y"), x.copy(deep=True).rename("x")],
        axis=1,
        join="inner",
        sort=False,
    ).dropna(how="any")
    if aligned.empty:
        raise ValueError("y and x have no complete shared observations.")

    for name in ("y", "x"):
        values = aligned[name]
        numeric = values.map(
            lambda value: isinstance(value, Real) and not isinstance(value, bool)
        )
        if not bool(numeric.all()):
            raise ValueError(f"{name} prices must be numeric.")

    aligned = aligned.astype(float)
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("y and x prices must be finite.")
    if aligned.le(0).any().any():
        raise ValueError("y and x prices must be strictly positive.")
    return aligned


def ols_spread(
    y: pd.Series,
    x: pd.Series,
) -> tuple[pd.Series, float, float]:
    """Estimate a static log-price spread with an intercept.

    The fitted relationship is ``log(y) = alpha + beta * log(x) + spread``.
    """
    aligned = _align_and_validate_prices(y, x)
    if len(aligned) < 3:
        raise ValueError("Static OLS requires at least three aligned observations.")

    log_y = np.log(aligned["y"]).rename("y")
    log_x = np.log(aligned["x"]).rename("x")
    if log_x.nunique(dropna=False) < 2:
        raise ValueError("Static OLS requires non-zero variance in log(x).")

    design = pd.DataFrame({"const": 1.0, "x": log_x}, index=aligned.index)
    fitted = sm.OLS(log_y, design, missing="raise").fit()
    alpha = float(fitted.params["const"])
    beta = float(fitted.params["x"])
    spread = (log_y - alpha - beta * log_x).rename("spread")
    return spread, alpha, beta


def rolling_ols_spread(
    y: pd.Series,
    x: pd.Series,
    lookback: int = 120,
) -> pd.DataFrame:
    """Estimate a causal rolling log-price spread with an intercept.

    At timestamp ``t``, the parameters use exactly ``lookback`` complete,
    aligned observations ending at ``t-1``. The current prices are evaluated
    using those prior parameters. Warm-up rows and zero-variance explanatory
    windows return NaN for alpha, beta, and spread.
    """
    if type(lookback) is not int or lookback < 2:
        raise ValueError("lookback must be an integer of at least 2.")

    aligned = _align_and_validate_prices(y, x)
    if len(aligned) <= lookback:
        raise ValueError(
            "Rolling OLS requires at least lookback + 1 aligned observations."
        )

    log_y = np.log(aligned["y"]).rename("y")
    log_x = np.log(aligned["x"]).rename("x")
    prior_y = log_y.shift(1)
    prior_x = log_x.shift(1)

    mean_y = prior_y.rolling(lookback, min_periods=lookback).mean()
    mean_x = prior_x.rolling(lookback, min_periods=lookback).mean()
    covariance = prior_x.rolling(
        lookback, min_periods=lookback
    ).cov(prior_y)
    variance = prior_x.rolling(
        lookback, min_periods=lookback
    ).var()
    usable_variance = variance.where(variance > 0)

    beta = (covariance / usable_variance).rename("beta")
    alpha = (mean_y - beta * mean_x).rename("alpha")
    spread = (log_y - alpha - beta * log_x).rename("spread")
    return pd.concat([alpha, beta, spread], axis=1)


def kalman_ols_spread(
    y: pd.Series,
    x: pd.Series,
    transition_variance: float = 1e-5,
    observation_variance: float = 1e-3,
    initial_state_covariance: float = 1.0,
) -> pd.DataFrame:
    """Estimate a causal dynamic log-price relationship with a Kalman filter.

    The state is ``[alpha, beta]`` with a random-walk transition and initial
    mean ``[0, 1]``. The first observation updates that explicit prior without
    adding transition noise. Later predictions add
    ``transition_variance * I`` before observing the next value.

    ``innovation`` is the observation minus its prior-state prediction, and
    ``innovation_variance`` includes both predicted state uncertainty and
    observation noise. Reported alpha, beta, and spread are posterior values,
    so row ``t`` uses information through ``t`` and never future observations.
    """
    parameters = {
        "transition_variance": transition_variance,
        "observation_variance": observation_variance,
        "initial_state_covariance": initial_state_covariance,
    }
    validated: dict[str, float] = {}
    for name, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be numeric.")
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite.")
        if numeric_value <= 0:
            raise ValueError(f"{name} must be strictly positive.")
        validated[name] = numeric_value

    aligned = _align_and_validate_prices(y, x)
    if len(aligned) < 2:
        raise ValueError("Kalman OLS requires at least two aligned observations.")

    log_y = np.log(aligned["y"]).to_numpy(dtype=float)
    log_x = np.log(aligned["x"]).to_numpy(dtype=float)
    if np.ptp(log_x) == 0:
        raise ValueError("Kalman OLS requires non-zero variance in log(x).")

    transition_covariance = validated["transition_variance"] * np.eye(2)
    observation_noise = validated["observation_variance"]
    state = np.array([0.0, 1.0], dtype=float)
    covariance = validated["initial_state_covariance"] * np.eye(2)
    identity = np.eye(2)
    rows: list[tuple[float, float, float, float, float]] = []

    for position, (observed_y, observed_x) in enumerate(zip(log_y, log_x)):
        predicted_state = state
        predicted_covariance = (
            covariance
            if position == 0
            else covariance + transition_covariance
        )
        design = np.array([1.0, observed_x], dtype=float)
        innovation = float(observed_y - design @ predicted_state)
        innovation_variance = float(
            design @ predicted_covariance @ design + observation_noise
        )
        if not np.isfinite(innovation_variance) or innovation_variance <= 0:
            raise FloatingPointError(
                "Kalman innovation variance became non-positive or non-finite."
            )

        gain = predicted_covariance @ design / innovation_variance
        state = predicted_state + gain * innovation

        # Joseph form is stable under floating-point roundoff and preserves
        # positive semidefiniteness better than P - KHP.
        residual_operator = identity - np.outer(gain, design)
        covariance = (
            residual_operator
            @ predicted_covariance
            @ residual_operator.T
            + observation_noise * np.outer(gain, gain)
        )
        covariance = 0.5 * (covariance + covariance.T)
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise FloatingPointError("Kalman state or covariance became non-finite.")

        spread = float(observed_y - design @ state)
        rows.append(
            (
                float(state[0]),
                float(state[1]),
                spread,
                innovation,
                innovation_variance,
            )
        )

    return pd.DataFrame(
        rows,
        index=aligned.index,
        columns=[
            "alpha",
            "beta",
            "spread",
            "innovation",
            "innovation_variance",
        ],
    )
