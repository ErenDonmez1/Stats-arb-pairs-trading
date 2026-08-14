"""Causal row-position walk-forward evaluation for one selected pair per fold.

Formation data is used for pair selection and frozen static parameters only.
Signals, execution, accounting, and reported returns are restricted to the
subsequent trading interval.  Milestone 8A intentionally resets every fold to
flat with the same explicit capital base and rejects overlapping OOS windows
when constructing an aggregate report.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from .analytics import StrategyPerformanceReport, build_performance_report
from .backtest import BacktestResult, run_pair_backtest
from .data import OBSERVED_PRICE_MASK_ATTR
from .screening import PairScreeningResult, screen_pairs
from .signals import rolling_zscore
from .stats import ADF_MIN_OBSERVATIONS


__all__ = [
    "WalkForwardStatus",
    "WalkForwardFold",
    "WalkForwardFoldResult",
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
    """Immutable wrapper around one explicit formation/OOS evaluation."""

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
    formation_observations: int
    trading_observations: int
    trade_count: int
    starting_capital: float
    ending_capital: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "screening_results", tuple(self.screening_results))


@dataclass(frozen=True)
class WalkForwardResult:
    """Immutable fold collection and non-overlapping aggregate OOS report."""

    folds: tuple[WalkForwardFoldResult, ...]
    fold_count: int
    completed_fold_count: int
    no_selection_fold_count: int
    insufficient_data_fold_count: int
    total_oos_observations: int
    oos_returns: pd.Series
    overall_performance_report: StrategyPerformanceReport | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "folds", tuple(self.folds))
        copied_returns = self.oos_returns.copy(deep=True)
        copied_returns.name = "oos_return"
        object.__setattr__(self, "oos_returns", copied_returns)


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
    return prices.copy(deep=True)


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
    values explicitly create overlapping trading windows, which fold generation
    permits but aggregate analysis rejects.
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
    relevant = mask.loc[:, [symbol_y, symbol_x]]
    if relevant.isna().any().any():
        raise ValueError(
            "price observed-mask metadata must not contain missing values."
        )
    valid = relevant.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not bool(valid.all().all()):
        raise TypeError("price observed-mask metadata must contain Boolean values.")
    start = fold.trading_start_position
    stop = fold.trading_end_position + 1
    sliced = relevant.iloc[start:stop].astype(bool)
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
    backtest: BacktestResult | None = None,
) -> WalkForwardFoldResult:
    ending_capital = starting_capital
    trade_count = 0
    if backtest is not None:
        trade_count = len(backtest.ledger)
        if len(backtest.accounting):
            ending_capital = float(
                backtest.accounting["net_equity_after_carry"].iat[-1]
            )
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
        backtest=backtest,
        performance_report=None,
        formation_observations=formation_observations,
        trading_observations=trading_observations,
        trade_count=trade_count,
        starting_capital=starting_capital,
        ending_capital=ending_capital,
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
        )

    screening_results = tuple(
        screen_pairs(
            _screening_frame(formation),
            groups,
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
        )

    symbol_y = selected.symbol_y
    symbol_x = selected.symbol_x
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
            prices,
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
    try:
        report = build_performance_report(
            oos_returns,
            backtest.ledger,
            periods_per_year=parameters["periods_per_year"],
            risk_free_rate=parameters["risk_free_rate"],
            accounting=backtest.accounting,
            initial_capital=parameters["initial_capital"],
        )
    except ValueError as exc:
        return _empty_fold_result(
            fold,
            status=WalkForwardStatus.INSUFFICIENT_DATA,
            message=f"OOS analytics could not be calculated: {exc}",
            screening_results=screening_results,
            selected=selected,
            formation_observations=len(formation),
            trading_observations=len(trading),
            starting_capital=parameters["initial_capital"],
            backtest=backtest,
        )

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
        formation_observations=len(formation),
        trading_observations=len(trading),
        trade_count=len(backtest.ledger),
        starting_capital=parameters["initial_capital"],
        ending_capital=ending_capital,
    )


def _require_non_overlapping_trading_windows(
    folds: tuple[WalkForwardFold, ...],
) -> None:
    for previous, current in zip(folds, folds[1:]):
        if current.trading_start_position <= previous.trading_end_position:
            raise ValueError(
                "Overlapping trading windows are not supported for aggregate "
                "OOS reporting."
            )


def _namespace_completed_ledgers(
    results: tuple[WalkForwardFoldResult, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    template: pd.DataFrame | None = None
    for result in results:
        if result.status is not WalkForwardStatus.COMPLETED or result.backtest is None:
            continue
        ledger = result.backtest.ledger.copy(deep=True)
        if template is None:
            template = ledger.iloc[:0].copy(deep=True)
        if ledger.empty:
            continue
        ledger["trade_id"] = [
            f"fold-{result.fold.fold_id}:trade-{trade_id}"
            for trade_id in ledger["trade_id"]
        ]
        ledger.index = pd.MultiIndex.from_arrays(
            [
                np.full(len(ledger), result.fold.fold_id, dtype=int),
                np.arange(len(ledger), dtype=int),
            ],
            names=["fold_id", "fold_trade_row"],
        )
        frames.append(ledger)
    if frames:
        return pd.concat(frames, axis=0, copy=True)
    if template is not None:
        return template
    raise RuntimeError("Cannot build a ledger without a completed fold.")


def run_walk_forward_analysis(
    prices: pd.DataFrame,
    formation_window: int,
    trading_window: int,
    *,
    step_size: int | None = None,
    expanding: bool = False,
    minimum_observations: int | None = None,
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
) -> WalkForwardResult:
    """Run independent folds and aggregate non-overlapping OOS return rows.

    Completed fold returns are concatenated in fold order; Sharpe and drawdown
    are then recomputed from that combined Series rather than averaged across
    folds.  NO_SELECTION and INSUFFICIENT_DATA folds remain visible but add no
    fabricated zero returns.  Fold trade IDs are namespaced for combined trade
    statistics; aggregate turnover is intentionally left undefined because
    each fold resets its capital base.
    """
    validated_prices = _validated_prices(prices)
    risk_free = _finite_real(risk_free_rate, "risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")
    folds = generate_walk_forward_folds(
        validated_prices.index,
        formation_window,
        trading_window,
        step_size=step_size,
        expanding=expanding,
        minimum_observations=minimum_observations,
    )
    _require_non_overlapping_trading_windows(folds)

    results = tuple(
        run_walk_forward_fold(
            validated_prices,
            fold,
            groups=groups,
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
        )
        for fold in folds
    )
    completed = tuple(
        result for result in results if result.status is WalkForwardStatus.COMPLETED
    )
    return_parts = [
        result.backtest.accounting["net_return_after_carry"].copy(deep=True)
        for result in completed
        if result.backtest is not None
    ]
    if return_parts:
        oos_returns = pd.concat(return_parts, axis=0, copy=True).rename("oos_return")
        if not oos_returns.index.is_unique:
            raise RuntimeError("Non-overlapping folds produced duplicate OOS labels.")
    else:
        oos_returns = pd.Series(dtype=float, name="oos_return")

    overall_report: StrategyPerformanceReport | None = None
    if len(oos_returns.dropna()) >= 2 and completed:
        combined_ledger = _namespace_completed_ledgers(results)
        overall_report = build_performance_report(
            oos_returns,
            combined_ledger,
            periods_per_year=_positive_integer(periods_per_year, "periods_per_year"),
            risk_free_rate=risk_free,
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
        total_oos_observations=len(oos_returns),
        oos_returns=oos_returns,
        overall_performance_report=overall_report,
    )
