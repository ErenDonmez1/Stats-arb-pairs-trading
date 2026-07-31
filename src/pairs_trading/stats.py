"""Hedge-ratio estimation and diagnostics for an already-created spread."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


ADF_MIN_OBSERVATIONS = 50
HALF_LIFE_MIN_OBSERVATIONS = 30
DEFAULT_HURST_MIN_LAG = 2
DEFAULT_HURST_MAX_LAG = 20
_NEAR_DEGENERATE_RELATIVE_RANGE = 1e-12


@dataclass(frozen=True)
class ADFTestResult:
    """Immutable Augmented Dickey-Fuller test output."""

    statistic: float
    pvalue: float
    lags: int
    observations: int
    critical_values: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "critical_values",
            MappingProxyType(dict(self.critical_values)),
        )


@dataclass(frozen=True)
class SpreadDiagnostics:
    """Immutable stationarity and mean-reversion diagnostic summary."""

    adf_statistic: float
    adf_pvalue: float
    adf_lags: int
    adf_observations: int
    adf_critical_values: Mapping[str, float]
    half_life: float
    hurst: float

    def __post_init__(self) -> None:
        # frozen=True is shallow; copy and wrap the mapping for deep immutability.
        object.__setattr__(
            self,
            "adf_critical_values",
            MappingProxyType(dict(self.adf_critical_values)),
        )


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


def _validate_spread(
    spread: pd.Series,
    *,
    minimum_observations: int,
    diagnostic_name: str,
) -> pd.Series:
    """Return complete float observations after strict spread validation."""
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be exactly a one-dimensional pandas Series.")

    values = spread.copy(deep=True).dropna()
    if len(values) < minimum_observations:
        raise ValueError(
            f"{diagnostic_name} requires at least "
            f"{minimum_observations} complete observations."
        )

    numeric = values.map(
        lambda value: isinstance(value, Real) and not isinstance(value, bool)
    )
    if not bool(numeric.all()):
        raise ValueError("spread observations must be numeric.")

    values = values.astype(float)
    array = values.to_numpy()
    if not np.isfinite(array).all():
        raise ValueError("spread observations must be finite.")

    value_range = float(np.ptp(array))
    scale = max(float(np.max(np.abs(array))), 1.0)
    if value_range <= _NEAR_DEGENERATE_RELATIVE_RANGE * scale:
        raise ValueError("spread must not be constant or near-degenerate.")
    return values


def adf_test(spread: pd.Series) -> ADFTestResult:
    """Run an ADF stationarity test on at least 50 complete observations.

    The fixed specification is ``regression='c'`` and ``autolag='AIC'``.
    Statsmodels estimation errors are intentionally allowed to propagate.
    """
    values = _validate_spread(
        spread,
        minimum_observations=ADF_MIN_OBSERVATIONS,
        diagnostic_name="ADF",
    )
    statistic, pvalue, lags, observations, critical_values, _ = adfuller(
        values.to_numpy(),
        regression="c",
        autolag="AIC",
    )
    return ADFTestResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        lags=int(lags),
        observations=int(observations),
        critical_values={
            str(level): float(value) for level, value in critical_values.items()
        },
    )


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life from a lagged-level regression.

    Fits ``delta_t = intercept + lambda * spread_(t-1) + error_t`` using at
    least 30 complete observations. ``lambda >= 0`` is classified as not
    mean-reverting and returns positive infinity.
    """
    values = _validate_spread(
        spread,
        minimum_observations=HALF_LIFE_MIN_OBSERVATIONS,
        diagnostic_name="Half-life",
    )
    regression = pd.concat(
        [
            values.diff().rename("delta_spread"),
            values.shift(1).rename("lagged_spread"),
        ],
        axis=1,
    ).dropna()
    design = pd.DataFrame(
        {
            "const": 1.0,
            "lagged_spread": regression["lagged_spread"],
        },
        index=regression.index,
    )
    fitted = sm.OLS(regression["delta_spread"], design, missing="raise").fit()
    mean_reversion_coefficient = float(fitted.params["lagged_spread"])
    if mean_reversion_coefficient >= 0:
        return float("inf")
    return float(-np.log(2.0) / mean_reversion_coefficient)


def estimate_hurst(
    spread: pd.Series,
    min_lag: int = DEFAULT_HURST_MIN_LAG,
    max_lag: int = DEFAULT_HURST_MAX_LAG,
) -> float:
    """Estimate Hurst scaling from lagged-difference dispersion.

    For every integer lag in the inclusive range ``[min_lag, max_lag]``, this
    computes ``std(spread[t] - spread[t-lag])``. The Hurst estimate is the
    fitted slope in ``log(std difference) = const + H * log(lag)``.

    ``H < 0.5`` suggests mean reversion, approximately ``0.5`` suggests
    random-walk-like behaviour, and ``H > 0.5`` suggests persistence.
    """
    if type(min_lag) is not int or type(max_lag) is not int:
        raise ValueError("min_lag and max_lag must be non-boolean integers.")
    if min_lag < 1:
        raise ValueError("min_lag must be at least 1.")
    if max_lag <= min_lag:
        raise ValueError("max_lag must be greater than min_lag.")

    minimum_observations = max_lag + 2
    values = _validate_spread(
        spread,
        minimum_observations=minimum_observations,
        diagnostic_name="Hurst estimation",
    ).to_numpy()
    lags = np.arange(min_lag, max_lag + 1, dtype=int)
    dispersions = np.array(
        [
            np.std(values[lag:] - values[:-lag], ddof=1)
            for lag in lags
        ],
        dtype=float,
    )
    if not np.isfinite(dispersions).all() or np.any(dispersions <= 0):
        raise ValueError("Hurst lagged-difference dispersions are degenerate.")

    design = pd.DataFrame(
        {
            "const": 1.0,
            "log_lag": np.log(lags.astype(float)),
        }
    )
    response = pd.Series(np.log(dispersions), name="log_dispersion")
    fitted = sm.OLS(response, design, missing="raise").fit()
    hurst = float(fitted.params["log_lag"])
    if not np.isfinite(hurst):
        raise ValueError("Hurst estimation produced a non-finite result.")
    return hurst


def diagnose_spread(
    spread: pd.Series,
    min_lag: int = DEFAULT_HURST_MIN_LAG,
    max_lag: int = DEFAULT_HURST_MAX_LAG,
) -> SpreadDiagnostics:
    """Compute ADF, half-life, and Hurst diagnostics for one spread."""
    adf = adf_test(spread)
    return SpreadDiagnostics(
        adf_statistic=adf.statistic,
        adf_pvalue=adf.pvalue,
        adf_lags=adf.lags,
        adf_observations=adf.observations,
        adf_critical_values=adf.critical_values,
        half_life=estimate_half_life(spread),
        hurst=estimate_hurst(spread, min_lag=min_lag, max_lag=max_lag),
    )
