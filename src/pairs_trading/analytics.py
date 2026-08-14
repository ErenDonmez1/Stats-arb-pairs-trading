"""Return, risk, and drawdown metrics for an already-computed return series.

Trade statistics, benchmark analysis, portfolio aggregation, persistence, and
plotting remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


__all__ = [
    "CorePerformanceMetrics",
    "DrawdownEpisode",
    "DrawdownMetrics",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "downside_deviation",
    "sortino_ratio",
    "calculate_core_metrics",
    "equity_curve_from_returns",
    "drawdown_series",
    "drawdown_episodes",
    "maximum_drawdown",
    "drawdown_duration",
    "calmar_ratio",
    "calculate_drawdown_metrics",
]


NEAR_ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CorePerformanceMetrics:
    """Immutable summary of core strategy return and risk statistics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    downside_deviation: float
    sortino_ratio: float
    observations: int
    periods_per_year: int


@dataclass(frozen=True)
class DrawdownEpisode:
    """One immutable observation-count drawdown episode.

    ``peak_index`` is the most recent peak before the first underwater row,
    while ``start_index`` is that first underwater row.  ``duration`` is the
    recovery row position minus the peak row position, or the final row
    position minus the peak row position when unrecovered.
    """

    peak_index: Any
    start_index: Any
    trough_index: Any
    recovery_index: Any | None
    maximum_drawdown: float
    duration: int
    recovered: bool


@dataclass(frozen=True)
class DrawdownMetrics:
    """Immutable summary of drawdown severity, timing, duration, and Calmar."""

    maximum_drawdown: float
    maximum_drawdown_start: Any | None
    maximum_drawdown_trough: Any | None
    maximum_drawdown_recovery: Any | None
    maximum_drawdown_duration: int
    longest_drawdown_duration: int
    calmar_ratio: float
    observations: int
    current_drawdown: float
    underwater_observations: int


def _validated_returns(
    returns: pd.Series,
    *,
    missing_policy: str,
    min_observations: int,
) -> pd.Series:
    """Copy, drop missing observations, and validate a return Series.

    ``missing_policy='drop'`` removes NaN-like values without filling,
    interpolation, sorting, or index changes to the remaining observations.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if not returns.index.is_unique:
        raise ValueError("returns must have a unique index.")
    if missing_policy != "drop":
        raise ValueError("missing_policy must be 'drop'.")

    copied = returns.copy(deep=True)
    valid = copied.loc[copied.notna()]
    numeric = valid.map(
        lambda value: isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if not bool(numeric.all()):
        raise ValueError("Non-missing returns must be real numeric values.")

    try:
        values = valid.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Non-missing returns must be representable as floats."
        ) from exc
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Non-missing returns must be finite.")
    if len(values) < min_observations:
        raise ValueError(
            "Insufficient valid return observations: "
            f"received {len(values)}; require at least {min_observations}."
        )
    return values


def _validated_periods_per_year(periods_per_year: Any) -> int:
    """Return a positive, non-Boolean integer annualization frequency."""
    if isinstance(periods_per_year, (bool, np.bool_)) or not isinstance(
        periods_per_year,
        Integral,
    ):
        raise TypeError("periods_per_year must be a non-Boolean integer.")
    result = int(periods_per_year)
    if result <= 0:
        raise ValueError("periods_per_year must be strictly positive.")
    return result


def _validated_risk_free_rate(risk_free_rate: Any) -> float:
    """Return a finite annual risk-free rate strictly greater than -100%."""
    if isinstance(risk_free_rate, (bool, np.bool_)) or not isinstance(
        risk_free_rate,
        Real,
    ):
        raise TypeError("risk_free_rate must be a non-Boolean real number.")
    result = float(risk_free_rate)
    if not np.isfinite(result):
        raise ValueError("risk_free_rate must be finite.")
    if result <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")
    return result


def _validated_initial_equity(initial_equity: Any) -> float:
    """Return a finite, strictly positive, non-Boolean initial equity."""
    if isinstance(initial_equity, (bool, np.bool_)) or not isinstance(
        initial_equity,
        Real,
    ):
        raise TypeError("initial_equity must be a non-Boolean real number.")
    result = float(initial_equity)
    if not np.isfinite(result):
        raise ValueError("initial_equity must be finite.")
    if result <= 0.0:
        raise ValueError("initial_equity must be strictly positive.")
    return result


def _period_risk_free_rate(risk_free_rate: Any, periods_per_year: Any) -> float:
    """Convert an annual rate to its geometric per-period equivalent."""
    annual_rate = _validated_risk_free_rate(risk_free_rate)
    periods = _validated_periods_per_year(periods_per_year)
    return float(np.expm1(np.log1p(annual_rate) / periods))


def _is_effectively_zero(value: float, reference: np.ndarray) -> bool:
    """Apply the documented relative near-zero policy to a denominator.

    A denominator is effectively zero when its absolute value is no greater
    than ``NEAR_ZERO_TOLERANCE * max(1, max(abs(reference)))``.
    """
    scale = max(1.0, float(np.max(np.abs(reference))))
    return abs(value) <= NEAR_ZERO_TOLERANCE * scale


def _require_compoundable(returns: pd.Series) -> None:
    """Reject total losses and returns below -100% before compounding."""
    if bool(returns.le(-1.0).any()):
        raise ValueError("Geometric compounding requires every return to exceed -1.")


def total_return(
    returns: pd.Series,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return ``product(1 + r_t) - 1`` after dropping missing observations.

    Missing returns are dropped and never filled or interpolated.  At least one
    valid observation is required, and every compounded return must exceed
    -100%.
    """
    values = _validated_returns(
        returns,
        missing_policy=missing_policy,
        min_observations=1,
    )
    _require_compoundable(values)
    log_growth = float(np.log1p(values.to_numpy(dtype=float)).sum())
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.expm1(log_growth))
    if not np.isfinite(result):
        raise ValueError("Geometrically compounded total return is not finite.")
    return result


def annualized_return(
    returns: pd.Series,
    periods_per_year: int,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return the geometrically annualized strategy return.

    The convention is ``(1 + total_return) ** (periods_per_year / n) - 1``.
    Missing values are dropped without filling, and frequency is never inferred
    from the index.
    """
    periods = _validated_periods_per_year(periods_per_year)
    values = _validated_returns(
        returns,
        missing_policy=missing_policy,
        min_observations=1,
    )
    compounded = total_return(values, missing_policy="drop")
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(
            np.expm1(np.log1p(compounded) * periods / len(values))
        )
    if not np.isfinite(result):
        raise ValueError("Annualized return is not finite.")
    return result


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return sample volatility (``ddof=1``) scaled by square-root time.

    Missing values are dropped without filling.  At least two valid return
    observations are required.
    """
    periods = _validated_periods_per_year(periods_per_year)
    values = _validated_returns(
        returns,
        missing_policy=missing_policy,
        min_observations=2,
    )
    sample_volatility = float(values.std(ddof=1))
    return sample_volatility * float(np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return the annualized arithmetic Sharpe ratio.

    The annual risk-free rate is converted geometrically to a per-period rate.
    The mean and sample standard deviation (``ddof=1``) of per-period excess
    returns are used, followed by square-root-time annualization.  Missing
    observations are dropped.  An effectively zero excess-return volatility
    returns NaN rather than infinity.
    """
    periods = _validated_periods_per_year(periods_per_year)
    values = _validated_returns(
        returns,
        missing_policy=missing_policy,
        min_observations=2,
    )
    period_rate = _period_risk_free_rate(risk_free_rate, periods)
    excess = values.to_numpy(dtype=float) - period_rate
    sample_volatility = float(np.std(excess, ddof=1))
    if _is_effectively_zero(sample_volatility, excess):
        return float("nan")
    return float(np.mean(excess) / sample_volatility * np.sqrt(periods))


def downside_deviation(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return annualized lower-partial deviation around the risk-free target.

    The annual risk-free rate is converted to a per-period target.  For every
    valid observation, ``downside_t = min(return_t - target_t, 0)``.  The
    reported value is ``sqrt(mean(downside_t**2)) * sqrt(periods_per_year)``.
    Missing values are dropped without filling or interpolation.
    """
    periods = _validated_periods_per_year(periods_per_year)
    values = _validated_returns(
        returns,
        missing_policy=missing_policy,
        min_observations=1,
    )
    period_rate = _period_risk_free_rate(risk_free_rate, periods)
    downside = np.minimum(values.to_numpy(dtype=float) - period_rate, 0.0)
    period_deviation = float(np.sqrt(np.mean(np.square(downside))))
    return period_deviation * float(np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return the annualized Sortino ratio using lower-partial deviation.

    The numerator is the mean per-period excess return over the geometrically
    converted risk-free target.  The denominator is
    ``sqrt(mean(min(excess_t, 0)**2))``.  Their ratio is scaled by
    ``sqrt(periods_per_year)``.  Missing values are dropped.  An effectively
    zero downside denominator returns NaN rather than infinity.
    """
    periods = _validated_periods_per_year(periods_per_year)
    values = _validated_returns(
        returns,
        missing_policy=missing_policy,
        min_observations=1,
    )
    period_rate = _period_risk_free_rate(risk_free_rate, periods)
    excess = values.to_numpy(dtype=float) - period_rate
    downside = np.minimum(excess, 0.0)
    period_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if _is_effectively_zero(period_deviation, excess):
        return float("nan")
    return float(np.mean(excess) / period_deviation * np.sqrt(periods))


def calculate_core_metrics(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    *,
    missing_policy: str = "drop",
) -> CorePerformanceMetrics:
    """Calculate the complete immutable Milestone 7A metric summary.

    Every public metric receives the same original Series and explicit missing
    policy.  Missing observations are dropped independently and never filled;
    the reported observation count is the number of valid returns.  At least
    two valid observations are required because volatility and Sharpe use
    sample standard deviation.
    """
    periods = _validated_periods_per_year(periods_per_year)
    valid = _validated_returns(
        returns,
        missing_policy=missing_policy,
        min_observations=2,
    )
    return CorePerformanceMetrics(
        total_return=total_return(returns, missing_policy=missing_policy),
        annualized_return=annualized_return(
            returns,
            periods,
            missing_policy=missing_policy,
        ),
        annualized_volatility=annualized_volatility(
            returns,
            periods,
            missing_policy=missing_policy,
        ),
        sharpe_ratio=sharpe_ratio(
            returns,
            periods,
            risk_free_rate,
            missing_policy=missing_policy,
        ),
        downside_deviation=downside_deviation(
            returns,
            periods,
            risk_free_rate,
            missing_policy=missing_policy,
        ),
        sortino_ratio=sortino_ratio(
            returns,
            periods,
            risk_free_rate,
            missing_policy=missing_policy,
        ),
        observations=len(valid),
        periods_per_year=periods,
    )


def equity_curve_from_returns(
    returns: pd.Series,
    initial_equity: float = 1.0,
    *,
    missing_policy: str = "drop",
) -> pd.Series:
    """Return geometrically compounded equity on the retained return index.

    The curve is ``initial_equity * cumprod(1 + return_t)`` and is named
    ``equity_curve``.  Under ``missing_policy='drop'``, missing observations are
    removed without filling or interpolation.  Their removal compresses
    observation time while retaining the exact labels and order of valid rows.
    At least one valid return is required, and every return must exceed -100%.
    """
    capital = _validated_initial_equity(initial_equity)
    values = _validated_returns(
        returns,
        missing_policy=missing_policy,
        min_observations=1,
    )
    _require_compoundable(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        equity_values = capital * np.cumprod(1.0 + values.to_numpy(dtype=float))
    if not np.isfinite(equity_values).all() or bool((equity_values <= 0.0).any()):
        raise ValueError("Compounded equity must remain finite and strictly positive.")
    result = pd.Series(
        equity_values,
        index=values.index,
        name="equity_curve",
        dtype=float,
    )
    result.index = values.index
    return result


def drawdown_series(
    returns: pd.Series,
    initial_equity: float = 1.0,
    *,
    missing_policy: str = "drop",
) -> pd.Series:
    """Return negative-or-zero equity drawdowns on the retained return index.

    Drawdown is ``equity_t / running_peak_t - 1``.  New highs are zero and
    underwater observations are negative.  Missing returns follow the 7A drop
    policy, which compresses observation time without fabricating values.
    """
    equity = equity_curve_from_returns(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    running_peak = equity.cummax()
    result = (equity / running_peak - 1.0).rename("drawdown")
    if bool(result.gt(NEAR_ZERO_TOLERANCE).any()):
        raise RuntimeError("Drawdown cannot be positive relative to its running peak.")
    result = result.mask(result.gt(0.0), 0.0).astype(float)
    result.index = equity.index
    return result


def drawdown_episodes(
    returns: pd.Series,
    initial_equity: float = 1.0,
    *,
    missing_policy: str = "drop",
) -> tuple[DrawdownEpisode, ...]:
    """Return deterministic drawdown episodes in retained observation order.

    An episode begins on the first row below the most recent peak and recovers
    on the first later row whose equity reaches or exceeds that peak.  The first
    row attaining an episode's most negative drawdown is its trough.  Equal
    peak observations before an episode update the associated peak to the most
    recent one.  Duration is measured from peak row position to recovery row
    position, or to the final row for an unrecovered episode.
    """
    equity = equity_curve_from_returns(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    drawdowns = drawdown_series(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    equity_values = equity.to_numpy(dtype=float)
    drawdown_values = drawdowns.to_numpy(dtype=float)
    index = equity.index

    peak_position = 0
    peak_equity = float(equity_values[0])
    underwater = False
    start_position = 0
    trough_position = 0
    trough_drawdown = 0.0
    episodes: list[DrawdownEpisode] = []

    for position in range(1, len(equity_values)):
        current_equity = float(equity_values[position])
        current_drawdown = float(drawdown_values[position])
        if not underwater:
            if current_equity >= peak_equity:
                peak_position = position
                peak_equity = current_equity
                continue
            underwater = True
            start_position = position
            trough_position = position
            trough_drawdown = current_drawdown
            continue

        if current_equity >= peak_equity:
            episodes.append(
                DrawdownEpisode(
                    peak_index=index[peak_position],
                    start_index=index[start_position],
                    trough_index=index[trough_position],
                    recovery_index=index[position],
                    maximum_drawdown=trough_drawdown,
                    duration=int(position - peak_position),
                    recovered=True,
                )
            )
            underwater = False
            peak_position = position
            peak_equity = current_equity
            continue

        if current_drawdown < trough_drawdown:
            trough_position = position
            trough_drawdown = current_drawdown

    if underwater:
        final_position = len(equity_values) - 1
        episodes.append(
            DrawdownEpisode(
                peak_index=index[peak_position],
                start_index=index[start_position],
                trough_index=index[trough_position],
                recovery_index=None,
                maximum_drawdown=trough_drawdown,
                duration=int(final_position - peak_position),
                recovered=False,
            )
        )
    return tuple(episodes)


def maximum_drawdown(
    returns: pd.Series,
    initial_equity: float = 1.0,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return the most negative value of the drawdown Series as a float."""
    drawdowns = drawdown_series(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    return float(drawdowns.min())


def drawdown_duration(
    returns: pd.Series,
    initial_equity: float = 1.0,
    *,
    missing_policy: str = "drop",
) -> int:
    """Return the longest recovered or unrecovered duration in observations."""
    episodes = drawdown_episodes(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    return max((episode.duration for episode in episodes), default=0)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int,
    initial_equity: float = 1.0,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return annualized return divided by absolute maximum drawdown.

    Annualized return is delegated to :func:`annualized_return`.  Maximum
    drawdown uses negative decimals, so its absolute value is the denominator.
    A zero or effectively zero denominator under the 7A relative tolerance
    returns NaN rather than infinity.
    """
    annual_return = annualized_return(
        returns,
        periods_per_year,
        missing_policy=missing_policy,
    )
    drawdowns = drawdown_series(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    max_drawdown = float(drawdowns.min())
    if _is_effectively_zero(
        max_drawdown,
        drawdowns.to_numpy(dtype=float),
    ):
        return float("nan")
    return float(annual_return / abs(max_drawdown))


def calculate_drawdown_metrics(
    returns: pd.Series,
    periods_per_year: int,
    initial_equity: float = 1.0,
    *,
    missing_policy: str = "drop",
) -> DrawdownMetrics:
    """Compose immutable maximum-drawdown, duration, and Calmar metrics.

    Missing values are dropped without filling.  Consequently, episode
    durations count retained observations rather than original rows or calendar
    time.  When there is no drawdown episode, timing labels are None and both
    duration fields are zero.
    """
    equity = equity_curve_from_returns(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    drawdowns = drawdown_series(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    episodes = drawdown_episodes(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    max_drawdown = maximum_drawdown(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    longest_duration = drawdown_duration(
        returns,
        initial_equity,
        missing_policy=missing_policy,
    )
    calmar = calmar_ratio(
        returns,
        periods_per_year,
        initial_equity,
        missing_policy=missing_policy,
    )

    maximum_episode = min(
        episodes,
        key=lambda episode: episode.maximum_drawdown,
        default=None,
    )
    return DrawdownMetrics(
        maximum_drawdown=max_drawdown,
        maximum_drawdown_start=(
            None if maximum_episode is None else maximum_episode.peak_index
        ),
        maximum_drawdown_trough=(
            None if maximum_episode is None else maximum_episode.trough_index
        ),
        maximum_drawdown_recovery=(
            None if maximum_episode is None else maximum_episode.recovery_index
        ),
        maximum_drawdown_duration=(
            0 if maximum_episode is None else maximum_episode.duration
        ),
        longest_drawdown_duration=longest_duration,
        calmar_ratio=calmar,
        observations=len(equity),
        current_drawdown=float(drawdowns.iat[-1]),
        underwater_observations=int(drawdowns.lt(0.0).sum()),
    )
