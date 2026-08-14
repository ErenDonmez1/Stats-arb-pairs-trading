"""Return, drawdown, trade, rolling, and benchmark research analytics.

Portfolio aggregation, persistence, and plotting remain outside this module.
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
    "TradePerformanceMetrics",
    "BenchmarkMetrics",
    "StrategyPerformanceReport",
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
    "win_rate",
    "average_winner",
    "average_loser",
    "payoff_ratio",
    "trade_expectancy",
    "profit_factor",
    "average_holding_period",
    "median_holding_period",
    "turnover",
    "calculate_trade_metrics",
    "rolling_volatility",
    "rolling_sharpe_ratio",
    "rolling_sortino_ratio",
    "rolling_drawdown",
    "benchmark_beta",
    "benchmark_alpha",
    "tracking_error",
    "information_ratio",
    "correlation_to_benchmark",
    "calculate_benchmark_metrics",
    "build_performance_report",
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


@dataclass(frozen=True)
class TradePerformanceMetrics:
    """Immutable completed-trade statistics and optional exact turnover."""

    trades: int
    known_pnl_trades: int
    unknown_pnl_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    average_winner: float
    average_loser: float
    payoff_ratio: float
    expectancy: float
    profit_factor: float
    average_holding_period: float
    median_holding_period: float
    gross_profit: float
    gross_loss: float
    total_net_pnl: float
    turnover: float


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Immutable single-benchmark comparison statistics."""

    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    correlation: float
    observations: int
    periods_per_year: int


@dataclass(frozen=True)
class StrategyPerformanceReport:
    """Immutable composition of strategy, trade, and benchmark analytics."""

    core: CorePerformanceMetrics
    drawdown: DrawdownMetrics
    trades: TradePerformanceMetrics
    benchmark: BenchmarkMetrics | None
    report_observations: int
    periods_per_year: int


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


_REQUIRED_TRADE_LEDGER_COLUMNS = {
    "trade_id",
    "net_pnl",
    "holding_period_rows",
    "entry_gross_notional",
}
_REQUIRED_TURNOVER_COLUMNS = {"traded_notional_y", "traded_notional_x"}


def _validated_trade_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive, validated copy of a completed-trade ledger.

    ``net_pnl`` is the only required field permitted to be missing.  Missing
    P&L remains unknown; it is never converted to zero.  Ledger row order is
    preserved.
    """
    if not isinstance(ledger, pd.DataFrame):
        raise TypeError("ledger must be a pandas DataFrame.")
    if not ledger.index.is_unique:
        raise ValueError("ledger must have a unique index.")
    missing = _REQUIRED_TRADE_LEDGER_COLUMNS.difference(ledger.columns)
    if missing:
        raise ValueError(f"ledger is missing required columns: {sorted(missing)}.")

    result = ledger.copy(deep=True)
    trade_ids = result["trade_id"]
    if bool(trade_ids.isna().any()):
        raise ValueError("ledger trade_id values must not be missing.")
    if bool(trade_ids.duplicated().any()):
        raise ValueError("ledger trade_id values must be unique.")

    pnl = result["net_pnl"]
    known_pnl = pnl.loc[pnl.notna()]
    numeric_pnl = known_pnl.map(
        lambda value: isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if not bool(numeric_pnl.all()):
        raise ValueError("Known ledger net_pnl values must be real numeric values.")
    try:
        pnl_values = pnl.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Known ledger net_pnl values must be representable as floats."
        ) from exc
    if not np.isfinite(pnl_values[~np.isnan(pnl_values)]).all():
        raise ValueError("Known ledger net_pnl values must be finite.")

    holding = result["holding_period_rows"]
    valid_holding = holding.map(
        lambda value: isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and int(value) >= 0
    )
    if not bool(valid_holding.all()):
        raise ValueError(
            "ledger holding_period_rows values must be non-negative, "
            "non-Boolean integers."
        )

    notionals = result["entry_gross_notional"]
    numeric_notional = notionals.map(
        lambda value: isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if not bool(numeric_notional.all()):
        raise ValueError(
            "ledger entry_gross_notional values must be real numeric values."
        )
    try:
        notional_values = notionals.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "ledger entry_gross_notional values must be representable as floats."
        ) from exc
    if not np.isfinite(notional_values).all() or bool((notional_values <= 0.0).any()):
        raise ValueError(
            "ledger entry_gross_notional values must be finite and strictly positive."
        )

    result["net_pnl"] = pd.Series(
        pnl_values,
        index=result.index,
        dtype=float,
    )
    result["holding_period_rows"] = pd.Series(
        [int(value) for value in holding],
        index=result.index,
        dtype="int64",
    )
    result["entry_gross_notional"] = pd.Series(
        notional_values,
        index=result.index,
        dtype=float,
    )
    result.index = ledger.index
    return result


def _classified_trade_pnl(
    ledger: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return known, winning, losing, and breakeven P&L arrays plus tolerance."""
    known = ledger["net_pnl"].dropna().to_numpy(dtype=float)
    scale = max(1.0, float(np.max(np.abs(known)))) if len(known) else 1.0
    tolerance = NEAR_ZERO_TOLERANCE * scale
    winners = known[known > tolerance]
    losers = known[known < -tolerance]
    breakeven = known[np.abs(known) <= tolerance]
    return known, winners, losers, breakeven, tolerance


def win_rate(ledger: pd.DataFrame) -> float:
    """Return winning known-P&L trades divided by all known-P&L trades.

    Classification uses ``NEAR_ZERO_TOLERANCE * max(1, max(abs(net_pnl)))``.
    Unknown P&L is excluded.  With no known trades, the result is NaN.
    """
    validated = _validated_trade_ledger(ledger)
    known, winners, _, _, _ = _classified_trade_pnl(validated)
    if not len(known):
        return float("nan")
    return float(len(winners) / len(known))


def average_winner(ledger: pd.DataFrame) -> float:
    """Return mean winning-trade net P&L, or NaN when there are no winners."""
    validated = _validated_trade_ledger(ledger)
    _, winners, _, _, _ = _classified_trade_pnl(validated)
    return float(np.mean(winners)) if len(winners) else float("nan")


def average_loser(ledger: pd.DataFrame) -> float:
    """Return negative mean losing-trade net P&L, or NaN with no losers."""
    validated = _validated_trade_ledger(ledger)
    _, _, losers, _, _ = _classified_trade_pnl(validated)
    return float(np.mean(losers)) if len(losers) else float("nan")


def payoff_ratio(ledger: pd.DataFrame) -> float:
    """Return average winner divided by absolute average loser.

    Missing winner or loser groups and effectively zero loser denominators
    produce NaN rather than zero or infinity.
    """
    validated = _validated_trade_ledger(ledger)
    known, winners, losers, _, _ = _classified_trade_pnl(validated)
    if not len(winners) or not len(losers):
        return float("nan")
    winner_mean = float(np.mean(winners))
    loser_mean = float(np.mean(losers))
    if _is_effectively_zero(loser_mean, known):
        return float("nan")
    return float(winner_mean / abs(loser_mean))


def trade_expectancy(ledger: pd.DataFrame) -> float:
    """Return probability-weighted expected P&L per known trade.

    Winner and loser probabilities use all known-P&L trades as the denominator.
    Breakeven trades contribute exactly zero, and unknown P&L is excluded.  No
    known trades produces NaN.
    """
    validated = _validated_trade_ledger(ledger)
    known, winners, losers, _, _ = _classified_trade_pnl(validated)
    if not len(known):
        return float("nan")
    winner_term = (
        len(winners) / len(known) * float(np.mean(winners))
        if len(winners)
        else 0.0
    )
    loser_term = (
        len(losers) / len(known) * float(np.mean(losers))
        if len(losers)
        else 0.0
    )
    return float(winner_term + loser_term)


def profit_factor(ledger: pd.DataFrame) -> float:
    """Return gross winning P&L divided by absolute gross losing P&L.

    No losing trades or an effectively zero gross loss produces NaN rather than
    positive infinity.
    """
    validated = _validated_trade_ledger(ledger)
    known, winners, losers, _, _ = _classified_trade_pnl(validated)
    gross_profit = float(np.sum(winners)) if len(winners) else 0.0
    gross_loss = abs(float(np.sum(losers))) if len(losers) else 0.0
    if not len(losers) or _is_effectively_zero(gross_loss, known):
        return float("nan")
    return float(gross_profit / gross_loss)


def average_holding_period(ledger: pd.DataFrame) -> float:
    """Return mean holding-period rows across all trades, including unknown P&L."""
    validated = _validated_trade_ledger(ledger)
    if validated.empty:
        return float("nan")
    return float(validated["holding_period_rows"].mean())


def median_holding_period(ledger: pd.DataFrame) -> float:
    """Return median holding-period rows across all trades, including unknown P&L."""
    validated = _validated_trade_ledger(ledger)
    if validated.empty:
        return float("nan")
    return float(validated["holding_period_rows"].median())


def _validated_initial_capital(initial_capital: Any) -> float:
    """Return finite positive capital for turnover normalization."""
    if isinstance(initial_capital, (bool, np.bool_)) or not isinstance(
        initial_capital,
        Real,
    ):
        raise TypeError("initial_capital must be a non-Boolean real number.")
    result = float(initial_capital)
    if not np.isfinite(result):
        raise ValueError("initial_capital must be finite.")
    if result <= 0.0:
        raise ValueError("initial_capital must be strictly positive.")
    return result


def turnover(accounting: pd.DataFrame, initial_capital: float) -> float:
    """Return exact non-annualized traded notional divided by initial capital.

    Every row contributes ``traded_notional_y + traded_notional_x``.  This
    includes entries, exits, and hedge rebalances already represented by the
    accounting schedule.
    """
    if not isinstance(accounting, pd.DataFrame):
        raise TypeError("accounting must be a pandas DataFrame.")
    if not accounting.index.is_unique:
        raise ValueError("accounting must have a unique index.")
    missing = _REQUIRED_TURNOVER_COLUMNS.difference(accounting.columns)
    if missing:
        raise ValueError(
            f"accounting is missing required columns: {sorted(missing)}."
        )
    capital = _validated_initial_capital(initial_capital)
    copied = accounting.copy(deep=True)
    arrays: list[np.ndarray] = []
    for column in ("traded_notional_y", "traded_notional_x"):
        values = copied[column]
        numeric = values.map(
            lambda value: isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
        )
        if not bool(numeric.all()):
            raise ValueError(f"accounting {column} values must be real numeric values.")
        try:
            array = values.to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"accounting {column} values must be representable as floats."
            ) from exc
        if not np.isfinite(array).all() or bool((array < 0.0).any()):
            raise ValueError(
                f"accounting {column} values must be finite and non-negative."
            )
        arrays.append(array)
    with np.errstate(over="ignore", invalid="ignore"):
        total_traded_notional = float(np.sum(arrays[0]) + np.sum(arrays[1]))
        result = float(total_traded_notional / capital)
    if not np.isfinite(total_traded_notional) or not np.isfinite(result):
        raise ValueError("Turnover and total traded notional must remain finite.")
    return result


def calculate_trade_metrics(
    ledger: pd.DataFrame,
    *,
    accounting: pd.DataFrame | None = None,
    initial_capital: float | None = None,
) -> TradePerformanceMetrics:
    """Compose immutable trade statistics and optional exact turnover.

    ``trades`` counts all ledger rows.  Classification statistics use only
    known P&L, while holding periods use every trade.  If any P&L is unknown,
    ``total_net_pnl`` is NaN.  Turnover is NaN when both optional turnover
    inputs are omitted; otherwise both accounting and initial capital are
    required.
    """
    if (accounting is None) != (initial_capital is None):
        raise ValueError(
            "accounting and initial_capital must be supplied together for turnover."
        )
    validated = _validated_trade_ledger(ledger)
    known, winners, losers, breakeven, _ = _classified_trade_pnl(validated)
    unknown_count = int(validated["net_pnl"].isna().sum())
    gross_profit = float(np.sum(winners)) if len(winners) else 0.0
    gross_loss = abs(float(np.sum(losers))) if len(losers) else 0.0
    total_net_pnl = (
        float("nan")
        if unknown_count
        else (float(np.sum(known)) if len(known) else 0.0)
    )
    turnover_value = (
        float("nan")
        if accounting is None
        else turnover(accounting, initial_capital)
    )
    return TradePerformanceMetrics(
        trades=len(validated),
        known_pnl_trades=len(known),
        unknown_pnl_trades=unknown_count,
        winning_trades=len(winners),
        losing_trades=len(losers),
        breakeven_trades=len(breakeven),
        win_rate=win_rate(ledger),
        average_winner=average_winner(ledger),
        average_loser=average_loser(ledger),
        payoff_ratio=payoff_ratio(ledger),
        expectancy=trade_expectancy(ledger),
        profit_factor=profit_factor(ledger),
        average_holding_period=average_holding_period(ledger),
        median_holding_period=median_holding_period(ledger),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        total_net_pnl=total_net_pnl,
        turnover=turnover_value,
    )


def _validated_numeric_series_with_gaps(
    values: pd.Series,
    *,
    argument_name: str,
) -> pd.Series:
    """Return a float copy while retaining every original row and missing value."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"{argument_name} must be a pandas Series.")
    if not values.index.is_unique:
        raise ValueError(f"{argument_name} must have a unique index.")

    copied = values.copy(deep=True)
    present_mask = copied.notna()
    present = copied.loc[present_mask]
    numeric = present.map(
        lambda value: isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if not bool(numeric.all()):
        raise ValueError(
            f"Non-missing {argument_name} values must be real numeric values."
        )
    try:
        present_values = present.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Non-missing {argument_name} values must be representable as floats."
        ) from exc
    if not np.isfinite(present_values).all():
        raise ValueError(f"Non-missing {argument_name} values must be finite.")

    result_values = np.full(len(copied), np.nan, dtype=float)
    result_values[present_mask.to_numpy(dtype=bool)] = present_values
    result = pd.Series(
        result_values,
        index=values.index,
        name=values.name,
        dtype=float,
    )
    result.attrs = values.attrs.copy()
    result.index = values.index
    return result


def _validated_rolling_window(
    window: Any,
    *,
    observations: int,
    minimum: int,
) -> int:
    """Return a valid row-count rolling window for one metric."""
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral):
        raise TypeError("window must be a non-Boolean integer.")
    result = int(window)
    if result < minimum:
        qualifier = "strictly positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"window must be {qualifier}.")
    if observations < result:
        raise ValueError(
            "Insufficient rows for rolling window: "
            f"received {observations}; require at least {result}."
        )
    return result


def _finalize_rolling_result(
    result: pd.Series,
    source: pd.Series,
    *,
    name: str,
) -> pd.Series:
    """Restore exact index metadata and reject non-finite emitted metrics."""
    result = result.astype(float).rename(name)
    result.index = source.index
    emitted = result.loc[result.notna()].to_numpy(dtype=float)
    if not np.isfinite(emitted).all():
        raise ValueError(f"Calculated {name} values must remain finite or NaN.")
    return result


def rolling_volatility(
    returns: pd.Series,
    window: int,
    periods_per_year: int,
) -> pd.Series:
    """Return causal annualized sample volatility on complete row windows.

    Unlike full-sample metrics, rolling output never drops rows.  Each value at
    row ``t`` uses rows ``t-window+1`` through ``t``.  A missing value anywhere
    in that window leaves the result missing; no observation is filled.
    """
    values = _validated_numeric_series_with_gaps(
        returns,
        argument_name="returns",
    )
    periods = _validated_periods_per_year(periods_per_year)
    lookback = _validated_rolling_window(
        window,
        observations=len(values),
        minimum=2,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.rolling(lookback, min_periods=lookback).std(ddof=1)
        result = result * float(np.sqrt(periods))
    return _finalize_rolling_result(result, returns, name="rolling_volatility")


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Return causal rolling Sharpe ratios on complete row windows.

    The annual risk-free rate is converted to its geometric per-period
    equivalent.  Sample excess-return volatility uses ``ddof=1``.  Complete
    windows with an effectively zero denominator emit NaN, never infinity.
    """
    values = _validated_numeric_series_with_gaps(
        returns,
        argument_name="returns",
    )
    periods = _validated_periods_per_year(periods_per_year)
    lookback = _validated_rolling_window(
        window,
        observations=len(values),
        minimum=2,
    )
    period_rate = _period_risk_free_rate(risk_free_rate, periods)
    excess = values - period_rate
    rolling = excess.rolling(lookback, min_periods=lookback)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        denominator = rolling.std(ddof=1)
        scale = excess.abs().rolling(lookback, min_periods=lookback).max()
        scale = scale.clip(lower=1.0)
        result = rolling.mean() / denominator * float(np.sqrt(periods))
    result = result.mask(
        denominator.abs() <= NEAR_ZERO_TOLERANCE * scale,
        np.nan,
    )
    return _finalize_rolling_result(result, returns, name="rolling_sharpe_ratio")


def rolling_sortino_ratio(
    returns: pd.Series,
    window: int,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Return causal rolling Sortino ratios using 7A lower-partial deviation.

    Every complete window uses ``sqrt(mean(min(excess, 0)**2))``.  Missing rows
    are retained, poison only windows containing them, and are never filled.
    """
    values = _validated_numeric_series_with_gaps(
        returns,
        argument_name="returns",
    )
    periods = _validated_periods_per_year(periods_per_year)
    lookback = _validated_rolling_window(
        window,
        observations=len(values),
        minimum=1,
    )
    period_rate = _period_risk_free_rate(risk_free_rate, periods)
    excess = values - period_rate
    downside_squared = excess.clip(upper=0.0).pow(2)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        numerator = excess.rolling(lookback, min_periods=lookback).mean()
        denominator = np.sqrt(
            downside_squared.rolling(lookback, min_periods=lookback).mean()
        )
        scale = excess.abs().rolling(lookback, min_periods=lookback).max()
        scale = scale.clip(lower=1.0)
        result = numerator / denominator * float(np.sqrt(periods))
    result = result.mask(
        denominator.abs() <= NEAR_ZERO_TOLERANCE * scale,
        np.nan,
    )
    return _finalize_rolling_result(result, returns, name="rolling_sortino_ratio")


def rolling_drawdown(
    returns: pd.Series,
    initial_equity: float = 1.0,
) -> pd.Series:
    """Return current causal drawdown while preserving the complete row index.

    This is current drawdown, not maximum drawdown over a lookback.  Valid
    observations follow :func:`drawdown_series` exactly.  A missing current
    return emits NaN at that row; later valid observations resume the causal
    equity path without inventing a return for the missing row.
    """
    values = _validated_numeric_series_with_gaps(
        returns,
        argument_name="returns",
    )
    retained = values.loc[values.notna()]
    if retained.empty:
        raise ValueError("rolling_drawdown requires at least one valid return.")
    retained_drawdown = drawdown_series(
        retained,
        initial_equity,
        missing_policy="drop",
    )
    result = retained_drawdown.reindex(values.index).rename("rolling_drawdown")
    result.index = returns.index
    return result.astype(float)


def _validated_benchmark_pair(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    missing_policy: str,
    min_observations: int = 2,
) -> pd.DataFrame:
    """Return pairwise-valid strategy and benchmark returns on an exact index."""
    if missing_policy != "drop":
        raise ValueError("missing_policy must be 'drop'.")
    strategy = _validated_numeric_series_with_gaps(
        strategy_returns,
        argument_name="strategy_returns",
    )
    benchmark = _validated_numeric_series_with_gaps(
        benchmark_returns,
        argument_name="benchmark_returns",
    )
    if not strategy_returns.index.equals(benchmark_returns.index):
        raise ValueError(
            "strategy_returns and benchmark_returns must have matching exact "
            "indices and row order."
        )
    paired_mask = strategy.notna() & benchmark.notna()
    paired = pd.DataFrame(
        {
            "strategy": strategy.loc[paired_mask].to_numpy(dtype=float),
            "benchmark": benchmark.loc[paired_mask].to_numpy(dtype=float),
        },
        index=strategy.index[paired_mask],
        dtype=float,
    )
    if len(paired) < min_observations:
        raise ValueError(
            "Insufficient paired return observations: "
            f"received {len(paired)}; require at least {min_observations}."
        )
    return paired


def _sample_standard_deviation(values: np.ndarray, *, label: str) -> float:
    """Return a finite sample standard deviation for validated observations."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.std(values, ddof=1))
    if not np.isfinite(result):
        raise ValueError(f"{label} sample standard deviation must remain finite.")
    return result


def benchmark_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return single-benchmark return beta from sample covariance/variance.

    This market beta is unrelated to the pair hedge ratio estimated by the
    statistical model.  Pairwise-missing rows are dropped without filling.
    Effectively constant benchmark returns produce NaN.
    """
    paired = _validated_benchmark_pair(
        strategy_returns,
        benchmark_returns,
        missing_policy=missing_policy,
    )
    strategy = paired["strategy"].to_numpy(dtype=float)
    benchmark = paired["benchmark"].to_numpy(dtype=float)
    benchmark_std = _sample_standard_deviation(
        benchmark,
        label="Benchmark return",
    )
    if _is_effectively_zero(benchmark_std, benchmark):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = float(np.cov(strategy, benchmark, ddof=1)[0, 1])
        variance = float(np.var(benchmark, ddof=1))
        result = float(covariance / variance)
    if not np.isfinite(result):
        raise ValueError("Benchmark beta must remain finite or NaN.")
    return result


def benchmark_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return arithmetic annualized Jensen-style alpha for one benchmark.

    The annual risk-free rate is converted geometrically to a per-period rate.
    Periodic alpha is mean strategy excess return less beta times mean benchmark
    excess return, then arithmetically multiplied by ``periods_per_year``.
    """
    periods = _validated_periods_per_year(periods_per_year)
    period_rate = _period_risk_free_rate(risk_free_rate, periods)
    paired = _validated_benchmark_pair(
        strategy_returns,
        benchmark_returns,
        missing_policy=missing_policy,
    )
    beta = benchmark_beta(
        strategy_returns,
        benchmark_returns,
        missing_policy=missing_policy,
    )
    if np.isnan(beta):
        return float("nan")
    strategy_excess = paired["strategy"].to_numpy(dtype=float) - period_rate
    benchmark_excess = paired["benchmark"].to_numpy(dtype=float) - period_rate
    with np.errstate(over="ignore", invalid="ignore"):
        periodic_alpha = float(
            np.mean(strategy_excess) - beta * np.mean(benchmark_excess)
        )
        result = float(periodic_alpha * periods)
    if not np.isfinite(result):
        raise ValueError("Annualized benchmark alpha must remain finite or NaN.")
    return result


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return annualized sample volatility of pairwise-valid active returns."""
    periods = _validated_periods_per_year(periods_per_year)
    paired = _validated_benchmark_pair(
        strategy_returns,
        benchmark_returns,
        missing_policy=missing_policy,
    )
    active = (
        paired["strategy"].to_numpy(dtype=float)
        - paired["benchmark"].to_numpy(dtype=float)
    )
    active_std = _sample_standard_deviation(active, label="Active return")
    result = float(active_std * np.sqrt(periods))
    if not np.isfinite(result):
        raise ValueError("Tracking error must remain finite.")
    return result


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return annualized mean active return divided by sample active volatility."""
    periods = _validated_periods_per_year(periods_per_year)
    paired = _validated_benchmark_pair(
        strategy_returns,
        benchmark_returns,
        missing_policy=missing_policy,
    )
    active = (
        paired["strategy"].to_numpy(dtype=float)
        - paired["benchmark"].to_numpy(dtype=float)
    )
    active_std = _sample_standard_deviation(active, label="Active return")
    if _is_effectively_zero(active_std, active):
        return float("nan")
    result = float(np.mean(active) / active_std * np.sqrt(periods))
    if not np.isfinite(result):
        raise ValueError("Information ratio must remain finite or NaN.")
    return result


def correlation_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    missing_policy: str = "drop",
) -> float:
    """Return pairwise Pearson correlation, or NaN for constant inputs."""
    paired = _validated_benchmark_pair(
        strategy_returns,
        benchmark_returns,
        missing_policy=missing_policy,
    )
    strategy = paired["strategy"].to_numpy(dtype=float)
    benchmark = paired["benchmark"].to_numpy(dtype=float)
    strategy_std = _sample_standard_deviation(strategy, label="Strategy return")
    benchmark_std = _sample_standard_deviation(
        benchmark,
        label="Benchmark return",
    )
    if _is_effectively_zero(strategy_std, strategy) or _is_effectively_zero(
        benchmark_std,
        benchmark,
    ):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.corrcoef(strategy, benchmark)[0, 1])
    if not np.isfinite(result):
        raise ValueError("Benchmark correlation must remain finite or NaN.")
    return result


def calculate_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    *,
    missing_policy: str = "drop",
) -> BenchmarkMetrics:
    """Compose immutable single-benchmark metrics from paired observations."""
    periods = _validated_periods_per_year(periods_per_year)
    paired = _validated_benchmark_pair(
        strategy_returns,
        benchmark_returns,
        missing_policy=missing_policy,
    )
    return BenchmarkMetrics(
        beta=benchmark_beta(
            strategy_returns,
            benchmark_returns,
            missing_policy=missing_policy,
        ),
        alpha=benchmark_alpha(
            strategy_returns,
            benchmark_returns,
            periods,
            risk_free_rate,
            missing_policy=missing_policy,
        ),
        tracking_error=tracking_error(
            strategy_returns,
            benchmark_returns,
            periods,
            missing_policy=missing_policy,
        ),
        information_ratio=information_ratio(
            strategy_returns,
            benchmark_returns,
            periods,
            missing_policy=missing_policy,
        ),
        correlation=correlation_to_benchmark(
            strategy_returns,
            benchmark_returns,
            missing_policy=missing_policy,
        ),
        observations=len(paired),
        periods_per_year=periods,
    )


def build_performance_report(
    returns: pd.Series,
    ledger: pd.DataFrame,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    accounting: pd.DataFrame | None = None,
    initial_capital: float | None = None,
    benchmark_returns: pd.Series | None = None,
) -> StrategyPerformanceReport:
    """Compose existing core, drawdown, trade, and optional benchmark metrics.

    Turnover remains optional but requires both an accounting DataFrame and an
    explicit initial capital.  No capital base or annualization frequency is
    inferred.  All nested summaries are calculated from the supplied inputs;
    external later observations have no effect unless they are supplied.
    """
    if (accounting is None) != (initial_capital is None):
        raise ValueError(
            "accounting and initial_capital must be supplied together for turnover."
        )
    periods = _validated_periods_per_year(periods_per_year)
    core = calculate_core_metrics(
        returns,
        periods,
        risk_free_rate,
        missing_policy="drop",
    )
    drawdown = calculate_drawdown_metrics(
        returns,
        periods,
        missing_policy="drop",
    )
    trades = calculate_trade_metrics(
        ledger,
        accounting=accounting,
        initial_capital=initial_capital,
    )
    benchmark = (
        None
        if benchmark_returns is None
        else calculate_benchmark_metrics(
            returns,
            benchmark_returns,
            periods,
            risk_free_rate,
            missing_policy="drop",
        )
    )

    if core.observations != drawdown.observations:
        raise RuntimeError("Core and drawdown observation counts are inconsistent.")
    if trades.trades != len(ledger):
        raise RuntimeError("Trade summary count is inconsistent with the ledger.")
    if core.periods_per_year != periods:
        raise RuntimeError("Core annualization frequency is inconsistent.")
    if benchmark is not None and benchmark.periods_per_year != periods:
        raise RuntimeError("Benchmark annualization frequency is inconsistent.")
    if benchmark is not None and benchmark_returns is not None:
        paired = _validated_benchmark_pair(
            returns,
            benchmark_returns,
            missing_policy="drop",
        )
        if benchmark.observations != len(paired):
            raise RuntimeError("Benchmark paired observation count is inconsistent.")

    return StrategyPerformanceReport(
        core=core,
        drawdown=drawdown,
        trades=trades,
        benchmark=benchmark,
        report_observations=core.observations,
        periods_per_year=periods,
    )
