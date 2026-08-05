"""Causal spread standardisation without trading-position state.

Missing rows are preserved in place.  A z-score is defined only when the
current spread is present and every observation in its fixed-size prior window
is present.  Values are never filled, interpolated, dropped, or reordered.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


# A prior standard deviation at or below this absolute number of spread units
# is treated as numerically indistinguishable from zero.  The raw standard
# deviation remains visible in standardise_spread(); only division is masked.
_NEAR_ZERO_STANDARD_DEVIATION = 1e-12

__all__ = ["rolling_zscore", "standardise_spread"]


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer.")
    normalised = int(value)
    if normalised <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalised


def _valid_ddof(value: Any, lookback: int) -> int:
    """Validate degrees of freedom for a fixed window of ``lookback`` rows."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("ddof must be a non-Boolean integer.")
    normalised = int(value)
    if normalised < 0 or normalised >= lookback:
        raise ValueError("ddof must satisfy 0 <= ddof < lookback.")
    return normalised


def _validated_spread(
    spread: pd.Series,
    lookback: Any,
    ddof: Any,
) -> tuple[pd.Series, int, int]:
    """Return an independent float spread after strict structural validation."""
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series.")
    if not spread.index.is_unique:
        raise ValueError("spread must have a unique index.")

    normalised_lookback = _positive_integer(lookback, "lookback")
    normalised_ddof = _valid_ddof(ddof, normalised_lookback)
    if len(spread) <= normalised_lookback:
        raise ValueError(
            "spread must contain at least lookback + 1 rows to evaluate one "
            "post-warm-up observation."
        )

    copied = spread.copy(deep=True)
    non_missing = copied.loc[copied.notna()]
    numeric = non_missing.map(
        lambda value: isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if not bool(numeric.all()):
        raise ValueError("Non-missing spread observations must be real numeric values.")

    try:
        array = copied.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Non-missing spread observations must be representable as floats."
        ) from exc
    values = pd.Series(
        array,
        index=spread.index,
        name="spread",
        dtype=float,
    )
    finite_values = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_values).all():
        raise ValueError("Non-missing spread observations must be finite.")
    return values, normalised_lookback, normalised_ddof


def standardise_spread(
    spread: pd.Series,
    lookback: int = 60,
    *,
    method: str = "rolling",
    ddof: int = 1,
) -> pd.DataFrame:
    """Return causal rolling distribution statistics and spread z-scores.

    At row ``t``, ``rolling_mean`` and ``rolling_std`` use exactly ``lookback``
    rows ending at ``t-1``.  The current spread is excluded from both.  The
    default ``ddof=1`` produces the sample standard deviation; valid values
    satisfy ``0 <= ddof < lookback``.

    The first ``lookback`` rows are warm-up rows and remain missing.  Any
    missing value in a prior fixed-size window makes that row's mean, standard
    deviation, and z-score missing.  A missing current value makes its z-score
    missing without changing its prior-window statistics.

    Standard deviations less than or equal to ``1e-12`` spread units are
    considered near zero.  Their raw values remain in ``rolling_std`` for
    auditability, while the corresponding z-score is missing rather than zero
    or infinity.

    Only ``method="rolling"`` is supported in Milestone 5A.  The explicit
    selector keeps the public interface extensible without silently applying
    an unrequested standardisation policy.
    """
    if not isinstance(method, str) or method != "rolling":
        raise ValueError("method must be 'rolling'.")

    values, normalised_lookback, normalised_ddof = _validated_spread(
        spread,
        lookback,
        ddof,
    )
    prior = values.shift(1)
    rolling = prior.rolling(
        window=normalised_lookback,
        min_periods=normalised_lookback,
    )
    rolling_mean = rolling.mean().rename("rolling_mean")
    rolling_std = rolling.std(ddof=normalised_ddof).rename("rolling_std")
    usable_std = rolling_std.where(
        np.isfinite(rolling_std)
        & rolling_std.gt(_NEAR_ZERO_STANDARD_DEVIATION)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        zscore = ((values - rolling_mean) / usable_std).rename("zscore")
    zscore = zscore.where(np.isfinite(zscore))

    result = pd.concat(
        [values, rolling_mean, rolling_std, zscore],
        axis=1,
        copy=False,
    )
    # Reuse the caller's index object so names, timezone, frequency, and custom
    # ordering metadata remain intact.
    result.index = spread.index
    return result.loc[:, ["spread", "rolling_mean", "rolling_std", "zscore"]]


def rolling_zscore(
    spread: pd.Series,
    lookback: int = 60,
    *,
    ddof: int = 1,
) -> pd.Series:
    """Return a causal rolling z-score named ``zscore``.

    This is the compact Series interface to :func:`standardise_spread`; both
    functions therefore share exactly the same validation and missing-value
    policies.
    """
    result = standardise_spread(
        spread,
        lookback,
        method="rolling",
        ddof=ddof,
    )["zscore"].copy()
    result.name = "zscore"
    return result
