"""Causal row-position walk-forward evaluation for one selected pair per fold.

Formation data is used for pair selection and frozen static parameters only.
Signals, execution, accounting, and reported returns are restricted to the
subsequent trading interval.  Milestone 8A intentionally resets every fold to
flat with the same explicit capital base.  Aggregate analysis requires
contiguous, non-overlapping OOS windows and reports calendar and conditional
percentage-return views separately.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .analytics import (
    CorePerformanceMetrics,
    DrawdownMetrics,
    StrategyPerformanceReport,
    build_performance_report,
    calculate_core_metrics,
    calculate_drawdown_metrics,
)
from .backtest import BacktestResult, run_pair_backtest
from .data import OBSERVED_PRICE_MASK_ATTR
from .screening import PairScreeningResult, screen_pairs
from .signals import rolling_zscore
from .stats import ADF_MIN_OBSERVATIONS


__all__ = [
    "WalkForwardStatus",
    "WalkForwardAnalyticsStatus",
    "WalkForwardFold",
    "WalkForwardFoldResult",
    "WalkForwardReturnReport",
    "WalkForwardResult",
    "generate_walk_forward_folds",
    "run_walk_forward_fold",
    "run_walk_forward_analysis",
]


class WalkForwardStatus(str, Enum):
    """Deterministic terminal status for one walk-forward fold."""

    COMPLETED = "COMPLETED"
    NO_SELECTION = "NO_SELECTION"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class WalkForwardAnalyticsStatus(str, Enum):
    """Availability of analytics calculated after an execution outcome."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class WalkForwardFold:
    """Inclusive row-position boundaries and original labels for one fold."""

    fold_id: int
    formation_start_position: int
    formation_end_position: int
    trading_start_position: int
    trading_end_position: int
    formation_start_label: Any
    formation_end_label: Any
    trading_start_label: Any
    trading_end_label: Any


@dataclass(frozen=True)
class WalkForwardFoldResult:
    """Frozen wrapper around one formation/OOS evaluation.

    Successful execution and optional analytics have separate statuses.  The
    pandas objects held here are independent deep copies, but remain mutable
    objects owned by the caller; freezing prevents field replacement rather
    than making pandas internals immutable.
    """

    fold: WalkForwardFold
    status: WalkForwardStatus
    message: str | None
    candidates_screened: int
    screening_results: tuple[PairScreeningResult, ...]
    selected_symbol_y: str | None
    selected_symbol_x: str | None
    screening_rank: int | None
    corrected_pvalue: float | None
    selected_screening_result: PairScreeningResult | None
    frozen_alpha: float | None
    frozen_beta: float | None
    backtest: BacktestResult | None
    performance_report: StrategyPerformanceReport | None
    analytics_status: WalkForwardAnalyticsStatus
    analytics_error: str | None
    oos_returns: pd.Series
    formation_observations: int
    trading_observations: int
    trade_count: int
    starting_capital: float
    ending_capital: float
    eligible_symbols: tuple[str, ...]
    group_snapshot: Mapping[str, tuple[str, ...]] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "screening_results", tuple(self.screening_results))
        object.__setattr__(self, "eligible_symbols", tuple(self.eligible_symbols))
        if self.group_snapshot is not None:
            copied_groups = {
                str(group): tuple(symbols)
                for group, symbols in self.group_snapshot.items()
            }
            object.__setattr__(
                self,
                "group_snapshot",
                MappingProxyType(copied_groups),
            )
        copied_returns = self.oos_returns.copy(deep=True)
        copied_returns.name = "oos_return"
        object.__setattr__(self, "oos_returns", copied_returns)
        if self.backtest is not None:
            object.__setattr__(self, "backtest", _copy_backtest(self.backtest))

    @property
    def execution_status(self) -> WalkForwardStatus:
        """Explicit compatibility name for the fold execution status."""
        return self.status


@dataclass(frozen=True)
class WalkForwardReturnReport:
    """Return-only analytics without cross-fold dollar trade aggregation."""

    core: CorePerformanceMetrics
    drawdown: DrawdownMetrics
    report_observations: int
    periods_per_year: int


@dataclass(frozen=True)
class WalkForwardResult:
    """Frozen walk-forward result with explicit calendar and conditional views.

    Stored pandas objects are independently owned deep copies.  They remain
    caller-mutable; the dataclass itself only prevents field replacement.
    """

    folds: tuple[WalkForwardFoldResult, ...]
    fold_count: int
    completed_fold_count: int
    no_selection_fold_count: int
    insufficient_data_fold_count: int
    scheduled_oos_observations: int
    scheduled_eligible_oos_observations: int
    selected_oos_observations: int
    no_selection_oos_observations: int
    unavailable_oos_observations: int
    selection_coverage: float
    conditional_oos_returns: pd.Series
    calendar_oos_returns: pd.Series
    conditional_performance_report: WalkForwardReturnReport | None
    calendar_performance_report: WalkForwardReturnReport | None
    conditional_analytics_status: WalkForwardAnalyticsStatus
    calendar_analytics_status: WalkForwardAnalyticsStatus
    conditional_analytics_error: str | None
    calendar_analytics_error: str | None
    capital_policy: str
    aggregate_return_policy: str
    inactive_capital_policy: str
    selection_coverage_denominator: str
    aggregate_dollar_pnl_available: bool
    aggregate_trade_dollar_metrics_available: bool
    universe_provenance: str
    cleaning_provenance: str
    point_in_time_universe_validated: bool
    provenance_warnings: tuple[str, ...]
    evaluated_start_position: int
    evaluated_end_position: int
    evaluated_start_label: Any
    evaluated_end_label: Any
    discarded_terminal_rows: int

    def __post_init__(self) -> None:
        owned_folds = tuple(replace(fold) for fold in self.folds)
        object.__setattr__(self, "folds", owned_folds)
        conditional = self.conditional_oos_returns.copy(deep=True)
        conditional.name = "conditional_oos_return"
        calendar = self.calendar_oos_returns.copy(deep=True)
        calendar.name = "calendar_oos_return"
        object.__setattr__(self, "conditional_oos_returns", conditional)
        object.__setattr__(self, "calendar_oos_returns", calendar)
        object.__setattr__(self, "provenance_warnings", tuple(self.provenance_warnings))

    @property
    def total_oos_observations(self) -> int:
        """Compatibility alias for selected/executed OOS observations."""
        return self.selected_oos_observations

    @property
    def oos_returns(self) -> pd.Series:
        """Compatibility copy of conditional returns under the former name."""
        result = self.conditional_oos_returns.copy(deep=True)
        result.name = "oos_return"
        return result

    @property
    def overall_performance_report(self) -> WalkForwardReturnReport | None:
        """Compatibility alias for the calendar-time return report."""
        return self.calendar_performance_report


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    """Return an integer at or above an explicit positive minimum."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _optional_positive_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_real(value: Any, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _nonnegative_real(value: Any, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _validated_index(index: pd.Index) -> pd.Index:
    """Validate chronological structure without sorting or relabelling callers."""
    if not isinstance(index, pd.Index):
        raise TypeError("index must be a pandas Index.")
    if not index.is_unique:
        raise ValueError("index must be unique.")
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be monotonically increasing.")
    return index


def _validated_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.shape[1] < 2:
        raise ValueError("prices must contain at least two symbol columns.")
    if prices.columns.duplicated().any():
        raise ValueError("prices must have unique symbol columns.")
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in prices):
        raise ValueError("price column names must be non-empty strings.")
    _validated_index(prices.index)
    result = prices.copy(deep=True)
    result.attrs = deepcopy(prices.attrs)
    return result


def _copy_backtest(backtest: BacktestResult) -> BacktestResult:
    """Return a BacktestResult whose pandas frames share no mutable storage."""
    return replace(
        backtest,
        signals=backtest.signals.copy(deep=True),
        positions=backtest.positions.copy(deep=True),
        accounting=backtest.accounting.copy(deep=True),
        ledger=backtest.ledger.copy(deep=True),
    )


def _normalize_groups(
    groups: Mapping[str, Iterable[str]] | None,
    available_symbols: Iterable[str],
) -> Mapping[str, tuple[str, ...]] | None:
    """Materialize a group mapping once into deterministic immutable tuples."""
    if groups is None:
        return None
    if not isinstance(groups, Mapping):
        raise TypeError("groups must be a mapping or a per-fold callable.")

    available = set(available_symbols)
    invalid_groups = [
        group for group in groups if not isinstance(group, str) or not group.strip()
    ]
    if invalid_groups:
        raise ValueError("group names must be non-empty strings.")

    normalized: dict[str, tuple[str, ...]] = {}
    assigned: dict[str, str] = {}
    for group in sorted(groups):
        configured = groups[group]
        if isinstance(configured, (str, bytes)) or not isinstance(
            configured, Iterable
        ):
            raise ValueError(f"Group {group!r} symbols must be an iterable.")
        materialized = tuple(configured)
        if any(
            not isinstance(symbol, str) or not symbol.strip()
            for symbol in materialized
        ):
            raise ValueError(f"Group {group!r} contains an invalid symbol.")
        if len(materialized) != len(set(materialized)):
            raise ValueError(f"Group {group!r} contains duplicate symbols.")
        unknown = sorted(set(materialized).difference(available))
        if unknown:
            raise ValueError(f"Group {group!r} contains unknown symbols: {unknown}.")
        for symbol in materialized:
            if symbol in assigned:
                raise ValueError(
                    f"Symbol {symbol!r} belongs to both {assigned[symbol]!r} "
                    f"and {group!r}."
                )
            assigned[symbol] = group
        normalized[group] = tuple(sorted(materialized))
    return MappingProxyType(normalized)


def _formation_eligible_symbols(
    formation: pd.DataFrame,
    minimum_observations: int,
) -> tuple[str, ...]:
    """Choose symbols using usable formation observations and no future rows."""
    eligible: list[str] = []
    for symbol in formation.columns:
        series = formation[symbol]
        usable = 0
        malformed = False
        for value in series.loc[series.notna()]:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or float(value) <= 0.0
            ):
                malformed = True
                break
            usable += 1
        if not malformed and usable >= minimum_observations:
            eligible.append(symbol)
    return tuple(eligible)


def _formation_group_snapshot(
    groups: Mapping[str, tuple[str, ...]] | None,
    eligible_symbols: tuple[str, ...],
) -> Mapping[str, tuple[str, ...]] | None:
    """Restrict a validated group snapshot to formation-eligible symbols."""
    if groups is None:
        return None
    eligible = set(eligible_symbols)
    return MappingProxyType(
        {
            group: tuple(symbol for symbol in symbols if symbol in eligible)
            for group, symbols in groups.items()
        }
    )


def _has_candidate_pair(
    eligible_symbols: tuple[str, ...],
    groups: Mapping[str, tuple[str, ...]] | None,
) -> bool:
    if groups is None:
        return len(eligible_symbols) >= 2
    return any(len(symbols) >= 2 for symbols in groups.values())


def _return_analytics(
    returns: pd.Series,
    *,
    periods_per_year: int,
    risk_free_rate: float,
) -> tuple[
    WalkForwardReturnReport | None,
    WalkForwardAnalyticsStatus,
    str | None,
]:
    """Build aggregate return-only analytics without changing membership."""
    valid = returns.dropna()
    if len(valid) < 2:
        return (
            None,
            WalkForwardAnalyticsStatus.UNAVAILABLE,
            "At least two available return observations are required.",
        )
    if bool(valid.le(-1.0).any()):
        return (
            None,
            WalkForwardAnalyticsStatus.UNAVAILABLE,
            "Returns at or below -100% cannot be geometrically compounded.",
        )
    try:
        core = calculate_core_metrics(
            returns,
            periods_per_year,
            risk_free_rate,
            missing_policy="drop",
        )
        drawdown = calculate_drawdown_metrics(
            returns,
            periods_per_year,
            missing_policy="drop",
        )
    except Exception as exc:  # Reporting must never remove executed observations.
        return (
            None,
            WalkForwardAnalyticsStatus.FAILED,
            f"{type(exc).__name__}: {exc}",
        )
    return (
        WalkForwardReturnReport(
            core=core,
            drawdown=drawdown,
            report_observations=core.observations,
            periods_per_year=periods_per_year,
        ),
        WalkForwardAnalyticsStatus.AVAILABLE,
        None,
    )


def generate_walk_forward_folds(
    index: pd.Index,
    formation_window: int,
    trading_window: int,
    *,
    step_size: int | None = None,
    expanding: bool = False,
    minimum_observations: int | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Generate complete, adjacent formation/trading folds by row position.

    Positions are zero-based and inclusive.  In fixed mode both windows advance
    by ``step_size``.  In expanding mode the formation start remains zero while
    its end advances.  ``step_size`` defaults to ``trading_window``; smaller
    values explicitly create overlapping trading windows and larger values
    create gaps.  Fold generation permits both; aggregate analysis rejects both.
    """
    validated_index = _validated_index(index)
    formation = _positive_integer(formation_window, "formation_window")
    trading = _positive_integer(trading_window, "trading_window")
    step = (
        trading
        if step_size is None
        else _positive_integer(step_size, "step_size")
    )
    if not isinstance(expanding, (bool, np.bool_)):
        raise TypeError("expanding must be Boolean.")
    minimum = _optional_positive_integer(
        minimum_observations,
        "minimum_observations",
    )
    if minimum is not None and formation < minimum:
        raise ValueError(
            "formation_window must be at least minimum_observations."
        )

    folds: list[WalkForwardFold] = []
    offset = 0
    while True:
        formation_start = 0 if bool(expanding) else offset
        formation_end = formation - 1 + offset
        trading_start = formation_end + 1
        trading_end = trading_start + trading - 1
        if trading_end >= len(validated_index):
            break
        folds.append(
            WalkForwardFold(
                fold_id=len(folds) + 1,
                formation_start_position=formation_start,
                formation_end_position=formation_end,
                trading_start_position=trading_start,
                trading_end_position=trading_end,
                formation_start_label=validated_index[formation_start],
                formation_end_label=validated_index[formation_end],
                trading_start_label=validated_index[trading_start],
                trading_end_label=validated_index[trading_end],
            )
        )
        offset += step

    if not folds:
        raise ValueError(
            "The supplied index is too short to generate one complete "
            "formation and trading fold."
        )
    return tuple(folds)


def _validate_fold_against_prices(
    fold: WalkForwardFold,
    prices: pd.DataFrame,
) -> None:
    if not isinstance(fold, WalkForwardFold):
        raise TypeError("fold must be a WalkForwardFold.")
    _positive_integer(fold.fold_id, "fold.fold_id")
    positions = (
        fold.formation_start_position,
        fold.formation_end_position,
        fold.trading_start_position,
        fold.trading_end_position,
    )
    if any(
        isinstance(position, (bool, np.bool_))
        or not isinstance(position, Integral)
        for position in positions
    ):
        raise ValueError("fold positions must be non-Boolean integers.")
    formation_start, formation_end, trading_start, trading_end = map(int, positions)
    if formation_start < 0 or formation_end < formation_start:
        raise ValueError("fold formation positions are invalid.")
    if formation_end >= trading_start:
        raise ValueError("formation must end before trading starts.")
    if trading_start != formation_end + 1:
        raise ValueError("trading must begin immediately after formation.")
    if trading_end < trading_start or trading_end >= len(prices):
        raise ValueError("fold trading positions are outside the price data.")
    expected_labels = (
        prices.index[formation_start],
        prices.index[formation_end],
        prices.index[trading_start],
        prices.index[trading_end],
    )
    supplied_labels = (
        fold.formation_start_label,
        fold.formation_end_label,
        fold.trading_start_label,
        fold.trading_end_label,
    )
    labels_match = all(
        pd.Series([expected], dtype=object).equals(
            pd.Series([supplied], dtype=object)
        )
        for expected, supplied in zip(expected_labels, supplied_labels)
    )
    if not labels_match:
        raise ValueError("fold boundary labels do not match the price index.")


def _validate_run_parameters(
    *,
    screening_min_observations: Any,
    fdr_threshold: Any,
    max_half_life: Any,
    hurst_threshold: Any,
    zscore_lookback: Any,
    entry_z: Any,
    exit_z: Any,
    stop_z: Any,
    max_holding_period: Any,
    cooldown_period: Any,
    target_gross_notional: Any,
    initial_capital: Any,
    execution_lag: Any,
    commission_bps: Any,
    fixed_commission_per_leg: Any,
    slippage_bps: Any,
    borrow_rate_y: Any,
    borrow_rate_x: Any,
    financing_rate: Any,
    periods_per_year: Any,
    risk_free_rate: Any,
    force_liquidation: Any,
) -> dict[str, Any]:
    """Validate parameters even when a fold ultimately selects no pair."""
    minimum = _positive_integer(
        screening_min_observations,
        "screening_min_observations",
        minimum=ADF_MIN_OBSERVATIONS,
    )
    fdr = _finite_real(fdr_threshold, "fdr_threshold")
    if not 0.0 <= fdr <= 1.0:
        raise ValueError("fdr_threshold must lie in [0, 1].")
    maximum_half_life = _positive_real(max_half_life, "max_half_life")
    hurst = _finite_real(hurst_threshold, "hurst_threshold")
    lookback = _positive_integer(zscore_lookback, "zscore_lookback", minimum=2)
    entry = _positive_real(entry_z, "entry_z")
    exit_threshold = _nonnegative_real(exit_z, "exit_z")
    stop = _positive_real(stop_z, "stop_z")
    if entry <= exit_threshold:
        raise ValueError("entry_z must be greater than exit_z.")
    if stop <= entry:
        raise ValueError("stop_z must be greater than entry_z.")
    maximum_holding = _optional_positive_integer(
        max_holding_period,
        "max_holding_period",
    )
    cooldown = _optional_positive_integer(cooldown_period, "cooldown_period")
    target = _positive_real(target_gross_notional, "target_gross_notional")
    capital = _positive_real(initial_capital, "initial_capital")
    lag = _positive_integer(execution_lag, "execution_lag")
    commission = _nonnegative_real(commission_bps, "commission_bps")
    fixed_commission = _nonnegative_real(
        fixed_commission_per_leg,
        "fixed_commission_per_leg",
    )
    slippage = _nonnegative_real(slippage_bps, "slippage_bps")
    borrow_y = _nonnegative_real(borrow_rate_y, "borrow_rate_y")
    borrow_x = _nonnegative_real(borrow_rate_x, "borrow_rate_x")
    financing = _nonnegative_real(financing_rate, "financing_rate")
    periods = _positive_integer(periods_per_year, "periods_per_year")
    risk_free = _finite_real(risk_free_rate, "risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")
    if not isinstance(force_liquidation, (bool, np.bool_)):
        raise TypeError("force_liquidation must be Boolean.")
    return {
        "screening_min_observations": minimum,
        "fdr_threshold": fdr,
        "max_half_life": maximum_half_life,
        "hurst_threshold": hurst,
        "zscore_lookback": lookback,
        "entry_z": entry,
        "exit_z": exit_threshold,
        "stop_z": stop,
        "max_holding_period": maximum_holding,
        "cooldown_period": cooldown,
        "target_gross_notional": target,
        "initial_capital": capital,
        "execution_lag": lag,
        "commission_bps": commission,
        "fixed_commission_per_leg": fixed_commission,
        "slippage_bps": slippage,
        "borrow_rate_y": borrow_y,
        "borrow_rate_x": borrow_x,
        "financing_rate": financing,
        "periods_per_year": periods,
        "risk_free_rate": risk_free,
        "force_liquidation": bool(force_liquidation),
    }


def _screening_frame(formation: pd.DataFrame) -> pd.DataFrame:
    """Give screening a monotonic copy without reordering non-time labels."""
    result = formation.copy(deep=True)
    result.attrs = {}
    if (
        not isinstance(result.index, pd.DatetimeIndex)
        and not result.index.is_monotonic_increasing
    ):
        result.index = pd.RangeIndex(len(result), name="formation_row")
    return result


def _validated_pair_prices(
    frame: pd.DataFrame,
    symbol_y: str,
    symbol_x: str,
) -> tuple[pd.Series, pd.Series]:
    """Return aligned float prices, preserving missing rows for existing policies."""
    result: list[pd.Series] = []
    for symbol in (symbol_y, symbol_x):
        copied = frame[symbol].copy(deep=True)
        present = copied.loc[copied.notna()]
        numeric = present.map(
            lambda value: isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
        )
        if not bool(numeric.all()):
            raise ValueError(
                f"Non-missing {symbol} prices must be real numeric values."
            )
        try:
            array = copied.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{symbol} prices must be representable as floats."
            ) from exc
        finite = array[~np.isnan(array)]
        if not np.isfinite(finite).all() or bool((finite <= 0.0).any()):
            raise ValueError(
                f"Non-missing {symbol} prices must be finite and strictly positive."
            )
        series = pd.Series(array, index=frame.index, name=symbol, dtype=float)
        series.attrs = {}
        result.append(series)
    return result[0], result[1]


def _trading_observed_masks(
    prices: pd.DataFrame,
    fold: WalkForwardFold,
    symbol_y: str,
    symbol_x: str,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Slice optional cleaned-price provenance to the exact OOS interval."""
    if OBSERVED_PRICE_MASK_ATTR not in prices.attrs:
        return None, None
    mask = prices.attrs[OBSERVED_PRICE_MASK_ATTR]
    if not isinstance(mask, pd.DataFrame):
        raise TypeError("price observed-mask metadata must be a pandas DataFrame.")
    if not mask.index.equals(prices.index):
        raise ValueError("price observed-mask metadata index must match prices.")
    missing_columns = {symbol_y, symbol_x}.difference(mask.columns)
    if missing_columns:
        raise ValueError(
            "price observed-mask metadata is missing selected symbols: "
            f"{sorted(missing_columns)}."
        )
    start = fold.trading_start_position
    stop = fold.trading_end_position + 1
    sliced = mask.loc[:, [symbol_y, symbol_x]].iloc[start:stop].copy(deep=True)
    if sliced.isna().any().any():
        raise ValueError(
            "price observed-mask metadata must not contain missing values "
            "inside the fold trading interval."
        )
    valid = sliced.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not bool(valid.all().all()):
        raise TypeError(
            "price observed-mask metadata must contain Boolean values inside "
            "the fold trading interval."
        )
    sliced = sliced.astype(bool)
    return sliced[symbol_y].copy(deep=True), sliced[symbol_x].copy(deep=True)


def _empty_fold_result(
    fold: WalkForwardFold,
    *,
    status: WalkForwardStatus,
    message: str,
    screening_results: tuple[PairScreeningResult, ...],
    selected: PairScreeningResult | None,
    formation_observations: int,
    trading_observations: int,
    starting_capital: float,
    eligible_symbols: tuple[str, ...] = (),
    group_snapshot: Mapping[str, tuple[str, ...]] | None = None,
) -> WalkForwardFoldResult:
    return WalkForwardFoldResult(
        fold=fold,
        status=status,
        message=message,
        candidates_screened=len(screening_results),
        screening_results=screening_results,
        selected_symbol_y=None if selected is None else selected.symbol_y,
        selected_symbol_x=None if selected is None else selected.symbol_x,
        screening_rank=None if selected is None else selected.rank,
        corrected_pvalue=None if selected is None else selected.corrected_pvalue,
        selected_screening_result=selected,
        frozen_alpha=None if selected is None else selected.alpha,
        frozen_beta=None if selected is None else selected.beta,
        backtest=None,
        performance_report=None,
        analytics_status=WalkForwardAnalyticsStatus.NOT_APPLICABLE,
        analytics_error=None,
        oos_returns=pd.Series(dtype=float, name="oos_return"),
        formation_observations=formation_observations,
        trading_observations=trading_observations,
        trade_count=0,
        starting_capital=starting_capital,
        ending_capital=starting_capital,
        eligible_symbols=eligible_symbols,
        group_snapshot=group_snapshot,
    )


def run_walk_forward_fold(
    prices: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    groups: Mapping[str, Iterable[str]] | None = None,
    screening_min_observations: int = 100,
    fdr_threshold: float = 0.05,
    max_half_life: float = 60.0,
    hurst_threshold: float = 0.5,
    zscore_lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_period: int | None = None,
    cooldown_period: int | None = None,
    target_gross_notional: float = 100_000.0,
    initial_capital: float = 1_000_000.0,
    execution_lag: int = 1,
    commission_bps: float = 0.0,
    fixed_commission_per_leg: float = 0.0,
    slippage_bps: float = 0.0,
    borrow_rate_y: float = 0.0,
    borrow_rate_x: float = 0.0,
    financing_rate: float = 0.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    force_liquidation: bool = True,
) -> WalkForwardFoldResult:
    """Select on formation data and run one independent OOS pair backtest.

    The rank-1 identity and its static log-price OLS alpha/beta are frozen from
    screening before trading.  Formation spread observations seed the causal
    z-score lookback, but signal generation starts flat on the trading slice and
    only trading rows are passed to :func:`run_pair_backtest`.
    """
    validated_prices = _validated_prices(prices)
    _validate_fold_against_prices(fold, validated_prices)
    parameters = _validate_run_parameters(
        screening_min_observations=screening_min_observations,
        fdr_threshold=fdr_threshold,
        max_half_life=max_half_life,
        hurst_threshold=hurst_threshold,
        zscore_lookback=zscore_lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        max_holding_period=max_holding_period,
        cooldown_period=cooldown_period,
        target_gross_notional=target_gross_notional,
        initial_capital=initial_capital,
        execution_lag=execution_lag,
        commission_bps=commission_bps,
        fixed_commission_per_leg=fixed_commission_per_leg,
        slippage_bps=slippage_bps,
        borrow_rate_y=borrow_rate_y,
        borrow_rate_x=borrow_rate_x,
        financing_rate=financing_rate,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        force_liquidation=force_liquidation,
    )
    formation = validated_prices.iloc[
        fold.formation_start_position : fold.formation_end_position + 1
    ].copy(deep=True)
    trading = validated_prices.iloc[
        fold.trading_start_position : fold.trading_end_position + 1
    ].copy(deep=True)
    # The full-frame observed mask is sliced explicitly below.  Keeping that
    # DataFrame-valued attr on computational slices can make pandas concat try
    # to compare masks with an ambiguous elementwise truth value.
    formation.attrs = {}
    trading.attrs = {}
    normalized_groups = _normalize_groups(groups, validated_prices.columns)
    eligible_symbols = _formation_eligible_symbols(
        formation,
        parameters["screening_min_observations"],
    )
    group_snapshot = _formation_group_snapshot(
        normalized_groups,
        eligible_symbols,
    )
    if len(formation) < parameters["screening_min_observations"]:
        return _empty_fold_result(
            fold,
            status=WalkForwardStatus.INSUFFICIENT_DATA,
            message="Formation rows are below screening_min_observations.",
            screening_results=(),
            selected=None,
            formation_observations=len(formation),
            trading_observations=len(trading),
            starting_capital=parameters["initial_capital"],
            eligible_symbols=eligible_symbols,
            group_snapshot=group_snapshot,
        )

    if not _has_candidate_pair(eligible_symbols, group_snapshot):
        return _empty_fold_result(
            fold,
            status=WalkForwardStatus.INSUFFICIENT_DATA,
            message=(
                "Formation-local usable history produced fewer than two "
                "candidate symbols within an allowed group."
            ),
            screening_results=(),
            selected=None,
            formation_observations=len(formation),
            trading_observations=len(trading),
            starting_capital=parameters["initial_capital"],
            eligible_symbols=eligible_symbols,
            group_snapshot=group_snapshot,
        )

    screening_results = tuple(
        screen_pairs(
            _screening_frame(formation.loc[:, list(eligible_symbols)]),
            group_snapshot,
            min_observations=parameters["screening_min_observations"],
            fdr_threshold=parameters["fdr_threshold"],
            max_half_life=parameters["max_half_life"],
            hurst_threshold=parameters["hurst_threshold"],
        )
    )
    selected = next(
        (
            result
            for result in screening_results
            if result.selected and result.rank == 1
        ),
        None,
    )
    if selected is None:
        return _empty_fold_result(
            fold,
            status=WalkForwardStatus.NO_SELECTION,
            message="Formation screening selected no rank-1 pair.",
            screening_results=screening_results,
            selected=None,
            formation_observations=len(formation),
            trading_observations=len(trading),
            starting_capital=parameters["initial_capital"],
            eligible_symbols=eligible_symbols,
            group_snapshot=group_snapshot,
        )

    if (
        selected.alpha is None
        or selected.beta is None
        or not np.isfinite(float(selected.alpha))
        or not np.isfinite(float(selected.beta))
        or float(selected.beta) <= 0.0
    ):
        return _empty_fold_result(
            fold,
            status=WalkForwardStatus.INSUFFICIENT_DATA,
            message="Selected pair lacks a finite positive frozen hedge ratio.",
            screening_results=screening_results,
            selected=selected,
            formation_observations=len(formation),
            trading_observations=len(trading),
            starting_capital=parameters["initial_capital"],
            eligible_symbols=eligible_symbols,
            group_snapshot=group_snapshot,
        )

    symbol_y = selected.symbol_y
    symbol_x = selected.symbol_x
    if symbol_y not in eligible_symbols or symbol_x not in eligible_symbols:
        return _empty_fold_result(
            fold,
            status=WalkForwardStatus.INSUFFICIENT_DATA,
            message="Selected pair is not formation-eligible in this fold.",
            screening_results=screening_results,
            selected=selected,
            formation_observations=len(formation),
            trading_observations=len(trading),
            starting_capital=parameters["initial_capital"],
            eligible_symbols=eligible_symbols,
            group_snapshot=group_snapshot,
        )
    history = pd.concat(
        [formation.loc[:, [symbol_y, symbol_x]], trading.loc[:, [symbol_y, symbol_x]]],
        axis=0,
        copy=True,
    )
    try:
        history_y, history_x = _validated_pair_prices(history, symbol_y, symbol_x)
        spread = (
            np.log(history_y)
            - float(selected.alpha)
            - float(selected.beta) * np.log(history_x)
        ).rename("spread")
        if len(spread) <= parameters["zscore_lookback"]:
            raise ValueError(
                "Formation plus trading history is shorter than z-score warm-up."
            )
        complete_zscore = rolling_zscore(
            spread,
            parameters["zscore_lookback"],
            ddof=1,
        )
        trading_zscore = complete_zscore.iloc[-len(trading) :].copy(deep=True)
        trading_y, trading_x = _validated_pair_prices(trading, symbol_y, symbol_x)
        observed_y, observed_x = _trading_observed_masks(
            validated_prices,
            fold,
            symbol_y,
            symbol_x,
        )
    except ValueError as exc:
        return _empty_fold_result(
            fold,
            status=WalkForwardStatus.INSUFFICIENT_DATA,
            message=f"Trading inputs could not be constructed: {exc}",
            screening_results=screening_results,
            selected=selected,
            formation_observations=len(formation),
            trading_observations=len(trading),
            starting_capital=parameters["initial_capital"],
            eligible_symbols=eligible_symbols,
            group_snapshot=group_snapshot,
        )

    backtest = run_pair_backtest(
        trading_y,
        trading_x,
        float(selected.beta),
        parameters["target_gross_notional"],
        zscore=trading_zscore,
        initial_capital=parameters["initial_capital"],
        entry_z=parameters["entry_z"],
        exit_z=parameters["exit_z"],
        stop_z=parameters["stop_z"],
        max_holding_period=parameters["max_holding_period"],
        cooldown_period=parameters["cooldown_period"],
        missing_policy="hold",
        execution_lag=parameters["execution_lag"],
        commission_bps=parameters["commission_bps"],
        fixed_commission_per_leg=parameters["fixed_commission_per_leg"],
        slippage_bps=parameters["slippage_bps"],
        borrow_rate_y=parameters["borrow_rate_y"],
        borrow_rate_x=parameters["borrow_rate_x"],
        financing_rate=parameters["financing_rate"],
        periods_per_year=parameters["periods_per_year"],
        rebalance=False,
        force_liquidation=parameters["force_liquidation"],
        observed_y=observed_y,
        observed_x=observed_x,
    )
    oos_returns = backtest.accounting["net_return_after_carry"].copy(deep=True)
    report: StrategyPerformanceReport | None = None
    analytics_status = WalkForwardAnalyticsStatus.UNAVAILABLE
    analytics_error: str | None = None
    valid_oos_returns = oos_returns.dropna()
    if len(valid_oos_returns) < 2:
        analytics_error = "At least two available OOS returns are required."
    elif bool(valid_oos_returns.le(-1.0).any()):
        analytics_error = (
            "OOS returns at or below -100% cannot be geometrically compounded."
        )
    else:
        try:
            report = build_performance_report(
                oos_returns,
                backtest.ledger,
                periods_per_year=parameters["periods_per_year"],
                risk_free_rate=parameters["risk_free_rate"],
                accounting=backtest.accounting,
                initial_capital=parameters["initial_capital"],
            )
        except Exception as exc:  # Execution membership survives reporting errors.
            analytics_status = WalkForwardAnalyticsStatus.FAILED
            analytics_error = f"{type(exc).__name__}: {exc}"
        else:
            analytics_status = WalkForwardAnalyticsStatus.AVAILABLE

    ending_capital = float(backtest.accounting["net_equity_after_carry"].iat[-1])
    return WalkForwardFoldResult(
        fold=fold,
        status=WalkForwardStatus.COMPLETED,
        message=None,
        candidates_screened=len(screening_results),
        screening_results=screening_results,
        selected_symbol_y=symbol_y,
        selected_symbol_x=symbol_x,
        screening_rank=selected.rank,
        corrected_pvalue=selected.corrected_pvalue,
        selected_screening_result=selected,
        frozen_alpha=float(selected.alpha),
        frozen_beta=float(selected.beta),
        backtest=backtest,
        performance_report=report,
        analytics_status=analytics_status,
        analytics_error=analytics_error,
        oos_returns=oos_returns,
        formation_observations=len(formation),
        trading_observations=len(trading),
        trade_count=len(backtest.ledger),
        starting_capital=parameters["initial_capital"],
        ending_capital=ending_capital,
        eligible_symbols=eligible_symbols,
        group_snapshot=group_snapshot,
    )


def _validated_aggregate_step(
    trading_window: Any,
    step_size: Any,
) -> int:
    """Require contiguous non-overlapping windows for calendar aggregation."""
    trading = _positive_integer(trading_window, "trading_window")
    step = trading if step_size is None else _positive_integer(step_size, "step_size")
    if step < trading:
        raise ValueError(
            "Aggregate walk-forward analysis rejects overlapping trading "
            "windows: step_size must equal trading_window."
        )
    if step > trading:
        raise ValueError(
            "Aggregate walk-forward analysis rejects calendar gaps: "
            "step_size must equal trading_window."
        )
    return step


def run_walk_forward_analysis(
    prices: pd.DataFrame,
    formation_window: int,
    trading_window: int,
    *,
    step_size: int | None = None,
    expanding: bool = False,
    minimum_observations: int | None = None,
    groups: (
        Mapping[str, Iterable[str]]
        | Callable[[WalkForwardFold], Mapping[str, Iterable[str]] | None]
        | None
    ) = None,
    screening_min_observations: int = 100,
    fdr_threshold: float = 0.05,
    max_half_life: float = 60.0,
    hurst_threshold: float = 0.5,
    zscore_lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_period: int | None = None,
    cooldown_period: int | None = None,
    target_gross_notional: float = 100_000.0,
    initial_capital: float = 1_000_000.0,
    execution_lag: int = 1,
    commission_bps: float = 0.0,
    fixed_commission_per_leg: float = 0.0,
    slippage_bps: float = 0.0,
    borrow_rate_y: float = 0.0,
    borrow_rate_x: float = 0.0,
    financing_rate: float = 0.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    force_liquidation: bool = True,
) -> WalkForwardResult:
    """Run independent equal-capital folds over a contiguous OOS horizon.

    Conditional returns contain executed folds only.  Calendar returns retain
    every scheduled row: NO_SELECTION rows earn an explicit cash return of
    zero, while pre-execution INSUFFICIENT_DATA rows remain unavailable (NaN).
    Aggregate reports contain percentage-return analytics only because each
    fold resets capital; per-fold ledgers retain their own dollar attribution.

    A static group mapping is caller-supplied and not point-in-time verified.
    A callable receives each fold and may supply a formation-date snapshot, but
    its upstream provenance is still the caller's responsibility.
    """
    validated_prices = _validated_prices(prices)
    risk_free = _finite_real(risk_free_rate, "risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")
    periods = _positive_integer(periods_per_year, "periods_per_year")
    aggregate_step = _validated_aggregate_step(trading_window, step_size)
    folds = generate_walk_forward_folds(
        validated_prices.index,
        formation_window,
        trading_window,
        step_size=aggregate_step,
        expanding=expanding,
        minimum_observations=minimum_observations,
    )

    group_provider: (
        Callable[[WalkForwardFold], Mapping[str, Iterable[str]] | None] | None
    ) = None
    normalized_static_groups: Mapping[str, tuple[str, ...]] | None = None
    if groups is None:
        universe_provenance = "price_columns_caller_supplied_unvalidated"
    elif isinstance(groups, Mapping):
        normalized_static_groups = _normalize_groups(
            groups,
            validated_prices.columns,
        )
        universe_provenance = "static_groups_caller_supplied_unvalidated"
    elif callable(groups):
        group_provider = groups
        universe_provenance = "per_fold_groups_caller_supplied_unvalidated"
    else:
        raise TypeError("groups must be a mapping, per-fold callable, or None.")

    result_items: list[WalkForwardFoldResult] = []
    for fold in folds:
        fold_groups = normalized_static_groups
        if group_provider is not None:
            fold_groups = _normalize_groups(
                group_provider(fold),
                validated_prices.columns,
            )
        result_items.append(run_walk_forward_fold(
            validated_prices,
            fold,
            groups=fold_groups,
            screening_min_observations=screening_min_observations,
            fdr_threshold=fdr_threshold,
            max_half_life=max_half_life,
            hurst_threshold=hurst_threshold,
            zscore_lookback=zscore_lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_holding_period=max_holding_period,
            cooldown_period=cooldown_period,
            target_gross_notional=target_gross_notional,
            initial_capital=initial_capital,
            execution_lag=execution_lag,
            commission_bps=commission_bps,
            fixed_commission_per_leg=fixed_commission_per_leg,
            slippage_bps=slippage_bps,
            borrow_rate_y=borrow_rate_y,
            borrow_rate_x=borrow_rate_x,
            financing_rate=financing_rate,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free,
            force_liquidation=force_liquidation,
        ))
    results = tuple(result_items)
    completed = tuple(
        result for result in results if result.status is WalkForwardStatus.COMPLETED
    )
    conditional_parts: list[pd.Series] = []
    calendar_parts: list[pd.Series] = []
    selected_observations = 0
    no_selection_observations = 0
    unavailable_observations = 0
    for result in results:
        start = result.fold.trading_start_position
        stop = result.fold.trading_end_position + 1
        scheduled_index = validated_prices.index[start:stop]
        if result.status is WalkForwardStatus.COMPLETED:
            if not result.oos_returns.index.equals(scheduled_index):
                raise RuntimeError(
                    "Executed fold OOS returns do not match its trading index."
                )
            owned = result.oos_returns.copy(deep=True)
            conditional_parts.append(owned)
            calendar_parts.append(owned.copy(deep=True))
            selected_observations += len(scheduled_index)
        elif result.status is WalkForwardStatus.NO_SELECTION:
            calendar_parts.append(
                pd.Series(0.0, index=scheduled_index, dtype=float)
            )
            no_selection_observations += len(scheduled_index)
        elif result.status is WalkForwardStatus.INSUFFICIENT_DATA:
            calendar_parts.append(
                pd.Series(np.nan, index=scheduled_index, dtype=float)
            )
            unavailable_observations += len(scheduled_index)
        else:  # pragma: no cover - Enum exhaustiveness guard.
            raise RuntimeError(f"Unsupported fold status: {result.status!r}.")

    if conditional_parts:
        conditional_returns = pd.concat(
            conditional_parts,
            axis=0,
            copy=True,
        ).rename("conditional_oos_return")
    else:
        conditional_returns = pd.Series(
            dtype=float,
            name="conditional_oos_return",
        )
    calendar_returns = pd.concat(
        calendar_parts,
        axis=0,
        copy=True,
    ).rename("calendar_oos_return")
    if not conditional_returns.index.is_unique:
        raise RuntimeError("Contiguous folds produced duplicate conditional labels.")
    if not calendar_returns.index.is_unique:
        raise RuntimeError("Contiguous folds produced duplicate calendar labels.")

    conditional_report, conditional_status, conditional_error = _return_analytics(
        conditional_returns,
        periods_per_year=periods,
        risk_free_rate=risk_free,
    )
    if unavailable_observations:
        calendar_report = None
        calendar_status = WalkForwardAnalyticsStatus.UNAVAILABLE
        calendar_error = (
            "Calendar analytics are unavailable because scheduled rows have "
            "INSUFFICIENT_DATA rather than investable cash returns."
        )
    else:
        calendar_report, calendar_status, calendar_error = _return_analytics(
            calendar_returns,
            periods_per_year=periods,
            risk_free_rate=risk_free,
        )

    scheduled_observations = len(calendar_returns)
    scheduled_eligible_observations = (
        selected_observations + no_selection_observations
    )
    selection_coverage = (
        float(selected_observations / scheduled_eligible_observations)
        if scheduled_eligible_observations
        else float("nan")
    )
    first_fold = folds[0]
    last_fold = folds[-1]
    discarded_terminal_rows = len(validated_prices) - (
        last_fold.trading_end_position + 1
    )
    provenance_warnings = [
        "Upstream cleaning, listing history, delistings, and survivorship "
        "filters are caller-supplied and are not audited by walkforward.py."
    ]
    if isinstance(groups, Mapping):
        provenance_warnings.append(
            "Static groups are reused across folds and are caller-assumed, not "
            "independently point-in-time verified."
        )
    elif group_provider is not None:
        provenance_warnings.append(
            "Per-fold group snapshots are structurally validated, but their "
            "upstream causal provenance remains caller-supplied."
        )

    return WalkForwardResult(
        folds=results,
        fold_count=len(results),
        completed_fold_count=len(completed),
        no_selection_fold_count=sum(
            result.status is WalkForwardStatus.NO_SELECTION for result in results
        ),
        insufficient_data_fold_count=sum(
            result.status is WalkForwardStatus.INSUFFICIENT_DATA for result in results
        ),
        scheduled_oos_observations=scheduled_observations,
        scheduled_eligible_oos_observations=scheduled_eligible_observations,
        selected_oos_observations=selected_observations,
        no_selection_oos_observations=no_selection_observations,
        unavailable_oos_observations=unavailable_observations,
        selection_coverage=selection_coverage,
        conditional_oos_returns=conditional_returns,
        calendar_oos_returns=calendar_returns,
        conditional_performance_report=conditional_report,
        calendar_performance_report=calendar_report,
        conditional_analytics_status=conditional_status,
        calendar_analytics_status=calendar_status,
        conditional_analytics_error=conditional_error,
        calendar_analytics_error=calendar_error,
        capital_policy="equal_capital_reset",
        aggregate_return_policy="time_weighted_equal_capital_reset",
        inactive_capital_policy="zero_return_cash_for_no_selection_rows",
        selection_coverage_denominator=(
            "selected_plus_no_selection_scheduled_rows"
        ),
        aggregate_dollar_pnl_available=False,
        aggregate_trade_dollar_metrics_available=False,
        universe_provenance=universe_provenance,
        cleaning_provenance="caller_supplied_prices_unvalidated",
        point_in_time_universe_validated=False,
        provenance_warnings=tuple(provenance_warnings),
        evaluated_start_position=first_fold.trading_start_position,
        evaluated_end_position=last_fold.trading_end_position,
        evaluated_start_label=first_fold.trading_start_label,
        evaluated_end_label=last_fold.trading_end_label,
        discarded_terminal_rows=discarded_terminal_rows,
    )
