"""Deterministic multi-pair portfolio construction for research outputs.

The module combines already-completed pair strategies with static, caller-
declared capital weights.  It does not select pairs, recycle inactive capital,
optimise weights, or recreate pair-level execution and accounting logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .walkforward import WalkForwardResult


__all__ = [
    "AllocationMethod",
    "PortfolioAvailability",
    "PairPortfolioInput",
    "PairAllocation",
    "PortfolioAllocationPolicy",
    "PortfolioPosition",
    "PortfolioMetrics",
    "PortfolioResult",
    "validate_pair_results",
    "normalize_pair_weights",
    "allocate_pair_capital",
    "calculate_portfolio_returns",
    "calculate_portfolio_exposure_metrics",
    "build_portfolio_schedule",
    "run_multi_pair_portfolio",
]


PairIdentifier = str | tuple[str, str]
_WEIGHT_TOLERANCE = 1e-12
_EXPOSURE_COLUMNS = (
    "market_value_y",
    "market_value_x",
    "gross_exposure",
    "long_exposure",
    "short_exposure",
    "net_exposure",
)
_RETURN_COLUMN = "net_return_after_carry"
_PORTFOLIO_WARNING = (
    "Static sleeve weights are predefined and never changed using OOS outcomes. "
    "Inactive, no-selection, unavailable, losing, and catastrophic sleeves are "
    "not reallocated or removed."
)


class AllocationMethod(str, Enum):
    """Supported deterministic capital-allocation rules."""

    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    FIXED_WEIGHT = "FIXED_WEIGHT"


class PortfolioAvailability(str, Enum):
    """Availability of the composed portfolio research output."""

    AVAILABLE = "AVAILABLE"
    PARTIALLY_AVAILABLE = "PARTIALLY_AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class PairPortfolioInput:
    """Defensively owned canonical data extracted from one pair result."""

    pair_id: str
    symbol_y: str
    symbol_x: str
    calendar_returns: pd.Series
    active_state: pd.Series
    execution_rows: pd.Series
    source_exposure: pd.DataFrame
    source_capital_basis: float | None
    source_type: str
    capital_policy: str
    point_in_time_universe_validated: bool
    universe_provenance: str
    cleaning_provenance: str
    provenance_warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calendar_returns",
            self.calendar_returns.copy(deep=True),
        )
        object.__setattr__(self, "active_state", self.active_state.copy(deep=True))
        object.__setattr__(
            self,
            "execution_rows",
            self.execution_rows.copy(deep=True),
        )
        object.__setattr__(
            self,
            "source_exposure",
            self.source_exposure.copy(deep=True),
        )
        object.__setattr__(
            self,
            "provenance_warnings",
            tuple(self.provenance_warnings),
        )


@dataclass(frozen=True)
class PairAllocation:
    """Static capital budget and exposure scaling for one pair sleeve."""

    pair_id: str
    symbol_y: str
    symbol_x: str
    weight: float
    allocated_capital: float
    source_capital_basis: float | None
    exposure_scaling_factor: float | None
    exposure_available: bool


@dataclass(frozen=True)
class PortfolioAllocationPolicy:
    """Immutable declaration of the portfolio's static allocation policy."""

    method: AllocationMethod
    pair_weights: tuple[tuple[str, float], ...]
    weights_normalized: bool
    cash_weight: float
    cash_return: float
    max_total_gross_exposure_ratio: float | None
    numerical_tolerance: float = _WEIGHT_TOLERANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_weights", tuple(self.pair_weights))


@dataclass(frozen=True)
class PortfolioPosition:
    """Descriptive position availability for one statically allocated sleeve."""

    pair_id: str
    allocation_weight: float
    allocated_capital: float
    active_state_available: bool
    active_observations: int
    exposure_available: bool


@dataclass(frozen=True)
class PortfolioMetrics:
    """Concentration, availability, and exposure diagnostics."""

    pair_count: int
    allocated_pair_count: int
    largest_pair_weight: float
    allocation_hhi: float
    effective_number_of_pairs: float
    cash_weight: float
    unavailable_return_row_count: int
    catastrophic_return_row_count: int
    exposure_available_pair_count: int
    exposure_unavailable_pair_count: int
    exposure_unavailable_row_count: int
    maximum_gross_exposure_ratio: float
    gross_exposure_limit_breach_count: int
    shared_symbols: tuple[str, ...]
    largest_symbol_gross_exposure_fraction: float


@dataclass(frozen=True)
class PortfolioResult:
    """Independently owned static multi-pair research portfolio output.

    The wrapper is frozen and all pandas objects are defensive copies.  The
    returned pandas objects remain mutable by the result holder; freezing does
    not make pandas internals recursively immutable.
    """

    pair_ids: tuple[str, ...]
    allocation_policy: PortfolioAllocationPolicy
    pair_allocations: tuple[PairAllocation, ...]
    pair_positions: tuple[PortfolioPosition, ...]
    portfolio_schedule: pd.DataFrame
    portfolio_returns: pd.Series
    portfolio_equity: pd.Series
    pair_sleeve_returns: pd.DataFrame
    pair_return_contributions: pd.DataFrame
    pair_exposures: pd.DataFrame
    aggregate_exposures: pd.DataFrame
    symbol_exposures: pd.DataFrame
    unavailable_rows: pd.DataFrame
    metrics: PortfolioMetrics
    availability: PortfolioAvailability
    initial_capital: float
    cash_weight: float
    cash_capital: float
    all_pair_universes_point_in_time_validated: bool
    pair_capital_policies: tuple[str, ...]
    pair_indices_differ: bool
    provenance_warnings: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_ids", tuple(self.pair_ids))
        object.__setattr__(self, "pair_allocations", tuple(self.pair_allocations))
        object.__setattr__(self, "pair_positions", tuple(self.pair_positions))
        for field_name in (
            "portfolio_schedule",
            "pair_sleeve_returns",
            "pair_return_contributions",
            "pair_exposures",
            "aggregate_exposures",
            "symbol_exposures",
            "unavailable_rows",
        ):
            object.__setattr__(
                self,
                field_name,
                getattr(self, field_name).copy(deep=True),
            )
        object.__setattr__(
            self,
            "portfolio_returns",
            self.portfolio_returns.copy(deep=True),
        )
        object.__setattr__(
            self,
            "portfolio_equity",
            self.portfolio_equity.copy(deep=True),
        )
        object.__setattr__(
            self,
            "pair_capital_policies",
            tuple(self.pair_capital_policies),
        )
        object.__setattr__(
            self,
            "provenance_warnings",
            tuple(self.provenance_warnings),
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _normalise_symbol(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError(f"{name} must not be blank.")
    if "|" in symbol:
        raise ValueError(f"{name} must not contain '|'.")
    return symbol


def _normalise_pair_identifier(value: PairIdentifier) -> tuple[str, str, str]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Tuple pair identifiers must contain exactly two symbols.")
        symbol_y = _normalise_symbol(value[0], "symbol_y")
        symbol_x = _normalise_symbol(value[1], "symbol_x")
    elif isinstance(value, str):
        parts = value.split("|")
        if len(parts) != 2:
            raise ValueError(
                "String pair identifiers must use the stable 'SYMBOL_Y|SYMBOL_X' form."
            )
        symbol_y = _normalise_symbol(parts[0], "symbol_y")
        symbol_x = _normalise_symbol(parts[1], "symbol_x")
    else:
        raise TypeError("Pair identifiers must be two-symbol tuples or 'Y|X' strings.")
    if symbol_y == symbol_x:
        raise ValueError("A pair identifier must contain two different symbols.")
    return f"{symbol_y}|{symbol_x}", symbol_y, symbol_x


def _validate_index(index: pd.Index, name: str) -> pd.Index:
    if not isinstance(index, pd.Index):
        raise TypeError(f"{name} must be a pandas Index.")
    if not index.is_unique:
        raise ValueError(f"{name} must be unique.")
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        raise ValueError(f"{name} DatetimeIndex must be monotonically increasing.")
    return index.copy()


def _validate_returns(series: pd.Series, pair_id: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{pair_id} calendar returns must be a pandas Series.")
    _validate_index(series.index, f"{pair_id} return index")
    nonmissing = series.loc[series.notna()].tolist()
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in nonmissing
    ):
        raise ValueError(
            f"{pair_id} non-missing returns must be non-Boolean real numbers."
        )
    result = series.astype(float).copy(deep=True)
    finite = result.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError(f"{pair_id} non-missing returns must be finite.")
    result.name = "pair_return"
    return result


def _normalise_capital_bases(
    values: Mapping[PairIdentifier, Any] | None,
) -> dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("source_capital_bases must be a mapping or None.")
    result: dict[str, float] = {}
    for identifier, capital in values.items():
        pair_id, _, _ = _normalise_pair_identifier(identifier)
        if pair_id in result:
            raise ValueError("source_capital_bases contains duplicate normalized IDs.")
        result[pair_id] = _finite_positive(
            capital,
            f"source_capital_bases[{pair_id!r}]",
        )
    return result


def _active_from_backtest(result: BacktestResult, index: pd.Index) -> pd.Series:
    positions = result.positions.copy(deep=True)
    if not positions.index.equals(index):
        raise ValueError("Backtest position index must exactly match accounting index.")
    if "executed_state" in positions.columns:
        active = positions["executed_state"].astype(str).ne("FLAT")
    elif {"units_y", "units_x"}.issubset(positions.columns):
        active = positions[["units_y", "units_x"]].fillna(0.0).ne(0.0).any(axis=1)
    else:
        active = pd.Series(pd.NA, index=index, dtype="boolean")
    active = active.astype("boolean")
    active.name = "active"
    return active


def _execution_rows_from_backtest(
    result: BacktestResult,
    index: pd.Index,
) -> pd.Series:
    positions = result.positions.copy(deep=True)
    execution = pd.Series(False, index=index, dtype=bool, name="execution_row")
    if "execution_event" in positions.columns:
        events = positions["execution_event"].astype(str)
        execution |= ~events.isin(("NONE", "", "nan", "None"))
    if "rebalance" in positions.columns:
        execution |= positions["rebalance"].fillna(False).astype(bool)
    return execution


def _source_from_backtest(
    pair_id: str,
    symbol_y: str,
    symbol_x: str,
    result: BacktestResult,
    capital_basis: float | None,
) -> PairPortfolioInput:
    accounting = result.accounting.copy(deep=True)
    if _RETURN_COLUMN not in accounting.columns:
        raise ValueError(
            f"{pair_id} BacktestResult accounting lacks {_RETURN_COLUMN!r}."
        )
    returns = _validate_returns(accounting[_RETURN_COLUMN], pair_id)
    if not result.signals.index.equals(returns.index):
        raise ValueError("Backtest signal index must exactly match accounting index.")
    active = _active_from_backtest(result, returns.index)
    execution = _execution_rows_from_backtest(result, returns.index)
    if set(_EXPOSURE_COLUMNS).issubset(accounting.columns):
        exposure = accounting.loc[:, _EXPOSURE_COLUMNS].copy(deep=True)
        for column in _EXPOSURE_COLUMNS:
            exposure[column] = pd.to_numeric(exposure[column], errors="raise")
        finite = exposure.to_numpy(dtype=float)
        if np.isinf(finite).any():
            raise ValueError(f"{pair_id} exposure data must not contain infinity.")
    else:
        exposure = pd.DataFrame(np.nan, index=returns.index, columns=_EXPOSURE_COLUMNS)
    return PairPortfolioInput(
        pair_id=pair_id,
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        calendar_returns=returns,
        active_state=active,
        execution_rows=execution,
        source_exposure=exposure,
        source_capital_basis=capital_basis,
        source_type="BacktestResult",
        capital_policy="standalone_pair_capital",
        point_in_time_universe_validated=False,
        universe_provenance="not_exposed_by_backtest_result",
        cleaning_provenance="not_exposed_by_backtest_result",
        provenance_warnings=(result.research_metadata.warning,),
    )


def _source_from_walk_forward(
    pair_id: str,
    symbol_y: str,
    symbol_x: str,
    result: WalkForwardResult,
    capital_basis: float | None,
) -> PairPortfolioInput:
    returns = _validate_returns(result.calendar_oos_returns, pair_id)
    active = pd.Series(pd.NA, index=returns.index, dtype="boolean", name="active")
    execution = pd.Series(False, index=returns.index, dtype=bool, name="execution_row")
    exposure = pd.DataFrame(np.nan, index=returns.index, columns=_EXPOSURE_COLUMNS)
    warnings = tuple(
        dict.fromkeys(
            (
                *result.provenance_warnings,
                "WalkForwardResult cross-fold positions and exposures are not "
                "aggregated by Milestone 9A; return aggregation remains available.",
            )
        )
    )
    return PairPortfolioInput(
        pair_id=pair_id,
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        calendar_returns=returns,
        active_state=active,
        execution_rows=execution,
        source_exposure=exposure,
        source_capital_basis=capital_basis,
        source_type="WalkForwardResult",
        capital_policy=result.capital_policy,
        point_in_time_universe_validated=result.point_in_time_universe_validated,
        universe_provenance=result.universe_provenance,
        cleaning_provenance=result.cleaning_provenance,
        provenance_warnings=warnings,
    )


def _materialise_pair_items(
    pair_results: (
        Mapping[PairIdentifier, BacktestResult | WalkForwardResult | PairPortfolioInput]
        | Iterable[
            tuple[
                PairIdentifier,
                BacktestResult | WalkForwardResult | PairPortfolioInput,
            ]
        ]
    ),
) -> tuple[tuple[PairIdentifier, BacktestResult | WalkForwardResult | PairPortfolioInput], ...]:
    if isinstance(pair_results, Mapping):
        items = tuple(pair_results.items())
    elif isinstance(pair_results, Iterable) and not isinstance(
        pair_results,
        (str, bytes),
    ):
        items = tuple(pair_results)
    else:
        raise TypeError("pair_results must be a mapping or iterable of pairs.")
    if not items:
        raise ValueError("pair_results must contain at least one pair sleeve.")
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("pair_results iterable entries must be (pair_id, result).")
    return items


def validate_pair_results(
    pair_results: (
        Mapping[PairIdentifier, BacktestResult | WalkForwardResult | PairPortfolioInput]
        | Iterable[
            tuple[
                PairIdentifier,
                BacktestResult | WalkForwardResult | PairPortfolioInput,
            ]
        ]
    ),
    *,
    source_capital_bases: Mapping[PairIdentifier, Any] | None = None,
    portfolio_index: pd.Index | None = None,
) -> tuple[PairPortfolioInput, ...]:
    """Normalize pair outputs and require an exact shared calendar index.

    No sorting, filling, interpolation, calendar compression, or outcome-based
    exclusion occurs.  Pair identifiers alone are sorted deterministically.
    """
    items = _materialise_pair_items(pair_results)
    capital_bases = _normalise_capital_bases(source_capital_bases)
    normalized: list[PairPortfolioInput] = []
    seen_ids: set[str] = set()
    for identifier, result in items:
        pair_id, symbol_y, symbol_x = _normalise_pair_identifier(identifier)
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate normalized pair identifier {pair_id!r}.")
        seen_ids.add(pair_id)
        capital_basis = capital_bases.get(pair_id)
        if isinstance(result, BacktestResult):
            source = _source_from_backtest(
                pair_id,
                symbol_y,
                symbol_x,
                result,
                capital_basis,
            )
        elif isinstance(result, WalkForwardResult):
            source = _source_from_walk_forward(
                pair_id,
                symbol_y,
                symbol_x,
                result,
                capital_basis,
            )
        elif isinstance(result, PairPortfolioInput):
            if result.pair_id != pair_id:
                raise ValueError(
                    "PairPortfolioInput.pair_id must match its normalized mapping key."
                )
            if (result.symbol_y, result.symbol_x) != (symbol_y, symbol_x):
                raise ValueError(
                    "PairPortfolioInput symbols must match its normalized mapping key."
                )
            if capital_basis is not None and result.source_capital_basis not in {
                None,
                capital_basis,
            }:
                raise ValueError("Conflicting source capital bases were supplied.")
            effective_basis = (
                capital_basis
                if capital_basis is not None
                else result.source_capital_basis
            )
            if effective_basis is not None:
                effective_basis = _finite_positive(
                    effective_basis,
                    f"{pair_id} source_capital_basis",
                )
            validated_returns = _validate_returns(
                result.calendar_returns,
                pair_id,
            )
            if not isinstance(result.active_state, pd.Series):
                raise TypeError(f"{pair_id} active_state must be a pandas Series.")
            if not isinstance(result.execution_rows, pd.Series):
                raise TypeError(f"{pair_id} execution_rows must be a pandas Series.")
            if not isinstance(result.source_exposure, pd.DataFrame):
                raise TypeError(f"{pair_id} source_exposure must be a DataFrame.")
            if not set(_EXPOSURE_COLUMNS).issubset(result.source_exposure.columns):
                raise ValueError(f"{pair_id} source_exposure lacks required columns.")
            exposure = result.source_exposure.loc[:, _EXPOSURE_COLUMNS].copy(deep=True)
            for column in _EXPOSURE_COLUMNS:
                exposure[column] = pd.to_numeric(exposure[column], errors="raise")
            if np.isinf(exposure.to_numpy(dtype=float)).any():
                raise ValueError(f"{pair_id} exposure data must not contain infinity.")
            source = PairPortfolioInput(
                **{
                    **result.__dict__,
                    "calendar_returns": validated_returns,
                    "active_state": result.active_state.astype("boolean"),
                    "execution_rows": result.execution_rows.astype(bool),
                    "source_exposure": exposure,
                    "source_capital_basis": effective_basis,
                }
            )
        else:
            raise TypeError(
                "Pair results must be BacktestResult, WalkForwardResult, or "
                "PairPortfolioInput values."
            )
        normalized.append(source)
    unknown_capital_ids = sorted(set(capital_bases).difference(seen_ids))
    if unknown_capital_ids:
        raise ValueError(
            f"source_capital_bases contains unknown pair IDs: {unknown_capital_ids}."
        )
    ordered = tuple(sorted(normalized, key=lambda item: item.pair_id))
    declared_index = (
        _validate_index(portfolio_index, "portfolio_index")
        if portfolio_index is not None
        else ordered[0].calendar_returns.index.copy()
    )
    for source in ordered:
        if not source.calendar_returns.index.equals(declared_index):
            raise ValueError(
                "All pair results must share the exact declared portfolio index "
                "and row order."
            )
        if not source.active_state.index.equals(declared_index):
            raise ValueError(f"{source.pair_id} active-state index is misaligned.")
        if not source.execution_rows.index.equals(declared_index):
            raise ValueError(f"{source.pair_id} execution-row index is misaligned.")
        if not source.source_exposure.index.equals(declared_index):
            raise ValueError(f"{source.pair_id} exposure index is misaligned.")
    return tuple(
        PairPortfolioInput(**source.__dict__)
        for source in ordered
    )


def _normalise_method(value: AllocationMethod | str) -> AllocationMethod:
    if isinstance(value, AllocationMethod):
        return value
    if not isinstance(value, str):
        raise TypeError("allocation_method must be an AllocationMethod or string.")
    try:
        return AllocationMethod(value.strip().upper())
    except ValueError as exc:
        raise ValueError(f"Unknown allocation_method {value!r}.") from exc


def _validated_fixed_weights(
    fixed_weights: Mapping[PairIdentifier, Any],
    pair_ids: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(fixed_weights, Mapping):
        raise TypeError("fixed_weights must be a mapping for FIXED_WEIGHT allocation.")
    normalized: dict[str, float] = {}
    for identifier, weight in fixed_weights.items():
        pair_id, _, _ = _normalise_pair_identifier(identifier)
        if pair_id in normalized:
            raise ValueError("fixed_weights contains duplicate normalized pair IDs.")
        normalized[pair_id] = _finite_nonnegative(
            weight,
            f"fixed_weights[{pair_id!r}]",
        )
    expected = set(pair_ids)
    actual = set(normalized)
    if actual != expected:
        raise ValueError(
            "fixed_weights must contain exactly one weight for every declared pair; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}."
        )
    return normalized


def normalize_pair_weights(
    weights: Mapping[PairIdentifier, Any],
    pair_ids: Iterable[PairIdentifier],
) -> tuple[tuple[str, float], ...]:
    """Explicitly normalize non-negative declared weights to sum to one."""
    normalized_ids = tuple(
        sorted(_normalise_pair_identifier(value)[0] for value in pair_ids)
    )
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("pair_ids contain duplicate normalized identifiers.")
    values = _validated_fixed_weights(weights, normalized_ids)
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError("At least one pair weight must be positive to normalize.")
    return tuple((pair_id, values[pair_id] / total) for pair_id in normalized_ids)


def allocate_pair_capital(
    pair_inputs: Iterable[PairPortfolioInput],
    portfolio_initial_capital: Any,
    *,
    allocation_method: AllocationMethod | str = AllocationMethod.EQUAL_WEIGHT,
    fixed_weights: Mapping[PairIdentifier, Any] | None = None,
    normalize_weights: bool = False,
) -> tuple[PairAllocation, ...]:
    """Create deterministic static pair-sleeve capital allocations."""
    sources = tuple(sorted(tuple(pair_inputs), key=lambda item: item.pair_id))
    if not sources or any(not isinstance(item, PairPortfolioInput) for item in sources):
        raise TypeError("pair_inputs must contain PairPortfolioInput values.")
    pair_ids = tuple(item.pair_id for item in sources)
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_inputs must have unique pair IDs.")
    capital = _finite_positive(portfolio_initial_capital, "portfolio_initial_capital")
    method = _normalise_method(allocation_method)
    if method is AllocationMethod.EQUAL_WEIGHT:
        if fixed_weights is not None:
            raise ValueError("fixed_weights must be omitted for EQUAL_WEIGHT allocation.")
        if normalize_weights:
            raise ValueError("normalize_weights is only meaningful for fixed weights.")
        weights = {pair_id: 1.0 / len(sources) for pair_id in pair_ids}
    else:
        if fixed_weights is None:
            raise ValueError("fixed_weights are required for FIXED_WEIGHT allocation.")
        weights = _validated_fixed_weights(fixed_weights, pair_ids)
        if normalize_weights:
            weights = dict(normalize_pair_weights(fixed_weights, pair_ids))
        total = float(sum(weights.values()))
        if total > 1.0 + _WEIGHT_TOLERANCE:
            raise ValueError("Fixed pair weights must sum to at most one.")
    allocations: list[PairAllocation] = []
    for source in sources:
        weight = float(weights[source.pair_id])
        allocated = capital * weight
        scale = (
            allocated / source.source_capital_basis
            if source.source_capital_basis is not None
            else None
        )
        exposure_available = bool(
            weight == 0.0
            or (
                scale is not None
                and set(_EXPOSURE_COLUMNS).issubset(source.source_exposure.columns)
                and not source.source_exposure.empty
                and bool(source.source_exposure.notna().any().any())
            )
        )
        allocations.append(
            PairAllocation(
                pair_id=source.pair_id,
                symbol_y=source.symbol_y,
                symbol_x=source.symbol_x,
                weight=weight,
                allocated_capital=allocated,
                source_capital_basis=source.source_capital_basis,
                exposure_scaling_factor=scale,
                exposure_available=exposure_available,
            )
        )
    return tuple(allocations)


def calculate_portfolio_returns(
    pair_inputs: Iterable[PairPortfolioInput],
    allocations: Iterable[PairAllocation],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return static weighted contributions without missing-row renormalization."""
    sources = tuple(sorted(tuple(pair_inputs), key=lambda item: item.pair_id))
    if not sources:
        raise ValueError("pair_inputs must not be empty.")
    index = sources[0].calendar_returns.index.copy()
    if any(not source.calendar_returns.index.equals(index) for source in sources):
        raise ValueError("pair_inputs must share the exact same return index.")
    allocation_tuple = tuple(allocations)
    allocation_by_id = {item.pair_id: item for item in allocation_tuple}
    if len(allocation_by_id) != len(allocation_tuple):
        raise ValueError("allocations must have unique pair IDs.")
    if set(allocation_by_id) != {item.pair_id for item in sources}:
        raise ValueError("allocations must match pair_inputs exactly.")
    sleeve_returns = pd.DataFrame(
        {
            source.pair_id: source.calendar_returns.to_numpy(dtype=float)
            for source in sources
        },
        index=index,
    )
    contributions = pd.DataFrame(index=index, columns=sleeve_returns.columns, dtype=float)
    unavailable_ids: list[tuple[str, ...]] = []
    reasons: list[str | None] = []
    portfolio_values = np.empty(len(index), dtype=float)
    for row_number in range(len(index)):
        missing: list[str] = []
        row_total = 0.0
        for source in sources:
            weight = allocation_by_id[source.pair_id].weight
            pair_return = sleeve_returns[source.pair_id].iat[row_number]
            column_position = contributions.columns.get_loc(source.pair_id)
            if weight > 0.0 and not np.isfinite(pair_return):
                contributions.iat[row_number, column_position] = np.nan
                missing.append(source.pair_id)
            elif weight == 0.0:
                contributions.iat[row_number, column_position] = 0.0
            else:
                contribution = weight * pair_return
                contributions.iat[row_number, column_position] = contribution
                row_total += contribution
        if missing:
            portfolio_values[row_number] = np.nan
            unavailable_ids.append(tuple(sorted(missing)))
            reasons.append("Positive-weight pair return unavailable; weights were not renormalized.")
        else:
            portfolio_values[row_number] = row_total
            unavailable_ids.append(())
            reasons.append(None)
    portfolio_returns = pd.Series(
        portfolio_values,
        index=index,
        name="portfolio_return",
    )
    cash_contribution = pd.Series(0.0, index=index, name="cash_return_contribution")
    unavailable = pd.DataFrame(
        {
            "unavailable_pair_ids": unavailable_ids,
            "reason": reasons,
        },
        index=index,
    )
    return contributions, portfolio_returns, cash_contribution, unavailable


def _compound_equity(returns: pd.Series, initial_capital: float) -> pd.Series:
    equity = np.empty(len(returns), dtype=float)
    running = initial_capital
    available = True
    for row_number, value in enumerate(returns.to_numpy(dtype=float)):
        if not available or not np.isfinite(value):
            available = False
            equity[row_number] = np.nan
            continue
        running *= 1.0 + value
        equity[row_number] = running if np.isfinite(running) else np.nan
        if not np.isfinite(running):
            available = False
    return pd.Series(equity, index=returns.index.copy(), name="portfolio_equity")


def calculate_portfolio_exposure_metrics(
    pair_inputs: Iterable[PairPortfolioInput],
    allocations: Iterable[PairAllocation],
    portfolio_equity: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...], pd.Series]:
    """Scale source exposures and aggregate pair and symbol diagnostics."""
    sources = tuple(sorted(tuple(pair_inputs), key=lambda item: item.pair_id))
    index = portfolio_equity.index.copy()
    if not sources:
        raise ValueError("pair_inputs must not be empty.")
    if any(not source.calendar_returns.index.equals(index) for source in sources):
        raise ValueError("pair inputs and portfolio_equity must align exactly.")
    allocation_tuple = tuple(allocations)
    allocation_by_id = {item.pair_id: item for item in allocation_tuple}
    if len(allocation_by_id) != len(allocation_tuple):
        raise ValueError("allocations must have unique pair IDs.")
    if set(allocation_by_id) != {source.pair_id for source in sources}:
        raise ValueError("allocations must match pair_inputs exactly.")
    pair_columns = pd.MultiIndex.from_product(
        ([item.pair_id for item in sources], (*_EXPOSURE_COLUMNS, "gross_exposure_fraction")),
        names=("pair_id", "metric"),
    )
    pair_exposures = pd.DataFrame(np.nan, index=index, columns=pair_columns)
    symbols_to_pairs: dict[str, set[str]] = {}
    for source in sources:
        symbols_to_pairs.setdefault(source.symbol_y, set()).add(source.pair_id)
        symbols_to_pairs.setdefault(source.symbol_x, set()).add(source.pair_id)
        allocation = allocation_by_id[source.pair_id]
        if allocation.weight == 0.0:
            scaled = pd.DataFrame(0.0, index=index, columns=_EXPOSURE_COLUMNS)
        elif allocation.exposure_scaling_factor is None:
            continue
        else:
            scaled = source.source_exposure.loc[:, _EXPOSURE_COLUMNS].astype(float)
            scaled = scaled * allocation.exposure_scaling_factor
        for column in _EXPOSURE_COLUMNS:
            pair_exposures[(source.pair_id, column)] = scaled[column]
        denominator = portfolio_equity.where(portfolio_equity > 0.0)
        pair_exposures[(source.pair_id, "gross_exposure_fraction")] = (
            scaled["gross_exposure"] / denominator
        )

    aggregate = pd.DataFrame(
        np.nan,
        index=index,
        columns=(
            "total_gross_exposure",
            "total_long_exposure",
            "total_short_exposure",
            "total_net_exposure",
            "gross_exposure_ratio",
            "largest_symbol_gross_exposure_fraction",
            "exposure_available",
        ),
    )
    positive_allocations = tuple(
        allocation for allocation in allocation_tuple if allocation.weight > 0.0
    )
    all_exposure_sources_known = all(
        allocation.exposure_available for allocation in positive_allocations
    )
    if all_exposure_sources_known:
        gross_frame = pair_exposures.xs("gross_exposure", level="metric", axis=1)
        long_frame = pair_exposures.xs("long_exposure", level="metric", axis=1)
        short_frame = pair_exposures.xs("short_exposure", level="metric", axis=1)
        net_frame = pair_exposures.xs("net_exposure", level="metric", axis=1)
        rows_known = ~gross_frame.isna().any(axis=1)
        aggregate.loc[rows_known, "total_gross_exposure"] = gross_frame.loc[rows_known].sum(axis=1)
        aggregate.loc[rows_known, "total_long_exposure"] = long_frame.loc[rows_known].sum(axis=1)
        aggregate.loc[rows_known, "total_short_exposure"] = short_frame.loc[rows_known].sum(axis=1)
        aggregate.loc[rows_known, "total_net_exposure"] = net_frame.loc[rows_known].sum(axis=1)
        aggregate.loc[rows_known, "gross_exposure_ratio"] = (
            aggregate.loc[rows_known, "total_gross_exposure"]
            / portfolio_equity.loc[rows_known].where(portfolio_equity.loc[rows_known] > 0.0)
        )
        aggregate["exposure_available"] = rows_known.astype(bool)
    else:
        aggregate["exposure_available"] = False

    symbols = tuple(sorted(symbols_to_pairs))
    symbol_columns = pd.MultiIndex.from_product(
        (symbols, ("net_market_value", "gross_market_value", "gross_exposure_fraction")),
        names=("symbol", "metric"),
    )
    symbol_exposures = pd.DataFrame(np.nan, index=index, columns=symbol_columns)
    if all_exposure_sources_known:
        for symbol in symbols:
            signed_legs: list[pd.Series] = []
            gross_legs: list[pd.Series] = []
            for source in sources:
                allocation = allocation_by_id[source.pair_id]
                if allocation.weight == 0.0:
                    scale = 0.0
                else:
                    scale = allocation.exposure_scaling_factor
                if scale is None:
                    continue
                if source.symbol_y == symbol:
                    leg = source.source_exposure["market_value_y"].astype(float) * scale
                    signed_legs.append(leg)
                    gross_legs.append(leg.abs())
                if source.symbol_x == symbol:
                    leg = source.source_exposure["market_value_x"].astype(float) * scale
                    signed_legs.append(leg)
                    gross_legs.append(leg.abs())
            signed_frame = pd.concat(signed_legs, axis=1)
            gross_leg_frame = pd.concat(gross_legs, axis=1)
            rows_known = ~gross_leg_frame.isna().any(axis=1)
            symbol_exposures.loc[rows_known, (symbol, "net_market_value")] = signed_frame.loc[rows_known].sum(axis=1)
            symbol_exposures.loc[rows_known, (symbol, "gross_market_value")] = gross_leg_frame.loc[rows_known].sum(axis=1)
            symbol_exposures.loc[rows_known, (symbol, "gross_exposure_fraction")] = (
                symbol_exposures.loc[rows_known, (symbol, "gross_market_value")]
                / aggregate.loc[rows_known, "total_gross_exposure"].replace(0.0, np.nan)
            )
        gross_symbols = symbol_exposures.xs("gross_market_value", level="metric", axis=1)
        aggregate["largest_symbol_gross_exposure_fraction"] = (
            gross_symbols.max(axis=1)
            / aggregate["total_gross_exposure"].replace(0.0, np.nan)
        )
    shared_symbols = tuple(
        symbol for symbol, pair_ids in sorted(symbols_to_pairs.items()) if len(pair_ids) > 1
    )
    execution_rows = pd.Series(False, index=index, dtype=bool, name="execution_row")
    for source in sources:
        if allocation_by_id[source.pair_id].weight > 0.0:
            source_execution = source.execution_rows.fillna(False).astype(bool).copy()
            gross = pair_exposures[(source.pair_id, "gross_exposure")]
            positive_positions = np.flatnonzero(gross.fillna(0.0).to_numpy() > 0.0)
            if len(positive_positions):
                source_execution.iloc[int(positive_positions[0])] = True
            execution_rows |= source_execution
    return pair_exposures, aggregate, symbol_exposures, shared_symbols, execution_rows


def build_portfolio_schedule(
    portfolio_returns: pd.Series,
    portfolio_equity: pd.Series,
    cash_contribution: pd.Series,
    pair_inputs: Iterable[PairPortfolioInput],
    allocations: Iterable[PairAllocation],
    aggregate_exposures: pd.DataFrame,
    unavailable_rows: pd.DataFrame,
    *,
    max_total_gross_exposure_ratio: float | None = None,
    execution_rows: pd.Series | None = None,
) -> pd.DataFrame:
    """Compose row-level portfolio state and enforce execution-time limits."""
    sources = tuple(sorted(tuple(pair_inputs), key=lambda item: item.pair_id))
    allocation_tuple = tuple(allocations)
    allocation_by_id = {item.pair_id: item for item in allocation_tuple}
    if len(allocation_by_id) != len(allocation_tuple):
        raise ValueError("allocations must have unique pair IDs.")
    if set(allocation_by_id) != {source.pair_id for source in sources}:
        raise ValueError("allocations must match pair_inputs exactly.")
    index = portfolio_returns.index.copy()
    for name, pandas_object in (
        ("portfolio_equity", portfolio_equity),
        ("cash_contribution", cash_contribution),
        ("aggregate_exposures", aggregate_exposures),
        ("unavailable_rows", unavailable_rows),
    ):
        if not pandas_object.index.equals(index):
            raise ValueError(f"{name} must align exactly with portfolio_returns.")
    if any(not source.calendar_returns.index.equals(index) for source in sources):
        raise ValueError("pair inputs and portfolio returns must align exactly.")
    active_count = np.zeros(len(index), dtype=int)
    active_available = np.ones(len(index), dtype=bool)
    for source in sources:
        if allocation_by_id[source.pair_id].weight <= 0.0:
            continue
        states = source.active_state.astype("boolean")
        active_count += states.fillna(False).to_numpy(dtype=bool).astype(int)
        active_available &= states.notna().to_numpy(dtype=bool)
    if execution_rows is None:
        execution_rows = pd.Series(False, index=index, dtype=bool)
    if not execution_rows.index.equals(index):
        raise ValueError("execution_rows must align exactly with portfolio returns.")
    gross_ratio = aggregate_exposures["gross_exposure_ratio"].astype(float)
    if max_total_gross_exposure_ratio is None:
        limit = None
        breaches = pd.Series(False, index=index, dtype=bool)
    else:
        limit = _finite_positive(
            max_total_gross_exposure_ratio,
            "max_total_gross_exposure_ratio",
        )
        breaches = gross_ratio.gt(limit + _WEIGHT_TOLERANCE).fillna(False)
        execution_breaches = breaches & execution_rows.astype(bool)
        if bool(execution_breaches.any()):
            labels = list(index[execution_breaches])
            raise ValueError(
                "Static allocation exceeds max_total_gross_exposure_ratio on "
                f"an initial/execution row: {labels}."
            )
    schedule = pd.DataFrame(index=index)
    schedule["portfolio_return"] = portfolio_returns
    schedule["portfolio_equity"] = portfolio_equity
    schedule["cash_return_contribution"] = cash_contribution
    schedule["allocated_pair_count"] = sum(
        allocation.weight > 0.0 for allocation in allocation_tuple
    )
    schedule["active_pair_count"] = active_count
    schedule["active_pair_count_available"] = active_available
    schedule["unavailable_pair_ids"] = unavailable_rows["unavailable_pair_ids"]
    schedule["total_gross_exposure"] = aggregate_exposures["total_gross_exposure"]
    schedule["total_long_exposure"] = aggregate_exposures["total_long_exposure"]
    schedule["total_short_exposure"] = aggregate_exposures["total_short_exposure"]
    schedule["total_net_exposure"] = aggregate_exposures["total_net_exposure"]
    schedule["gross_exposure_ratio"] = gross_ratio
    schedule["gross_exposure_limit"] = limit
    schedule["gross_exposure_limit_breach"] = breaches
    schedule["execution_row"] = execution_rows.astype(bool)
    schedule["catastrophic_portfolio_return"] = portfolio_returns.le(-1.0).fillna(False)
    return schedule


def run_multi_pair_portfolio(
    pair_results: (
        Mapping[PairIdentifier, BacktestResult | WalkForwardResult | PairPortfolioInput]
        | Iterable[
            tuple[
                PairIdentifier,
                BacktestResult | WalkForwardResult | PairPortfolioInput,
            ]
        ]
    ),
    portfolio_initial_capital: Any,
    *,
    allocation_method: AllocationMethod | str = AllocationMethod.EQUAL_WEIGHT,
    fixed_weights: Mapping[PairIdentifier, Any] | None = None,
    normalize_weights: bool = False,
    source_capital_bases: Mapping[PairIdentifier, Any] | None = None,
    portfolio_index: pd.Index | None = None,
    max_total_gross_exposure_ratio: Any | None = None,
) -> PortfolioResult:
    """Run deterministic static multi-pair capital allocation and aggregation."""
    capital = _finite_positive(portfolio_initial_capital, "portfolio_initial_capital")
    limit = (
        None
        if max_total_gross_exposure_ratio is None
        else _finite_positive(
            max_total_gross_exposure_ratio,
            "max_total_gross_exposure_ratio",
        )
    )
    sources = validate_pair_results(
        pair_results,
        source_capital_bases=source_capital_bases,
        portfolio_index=portfolio_index,
    )
    method = _normalise_method(allocation_method)
    allocations = allocate_pair_capital(
        sources,
        capital,
        allocation_method=method,
        fixed_weights=fixed_weights,
        normalize_weights=normalize_weights,
    )
    weight_total = float(sum(item.weight for item in allocations))
    cash_weight = max(0.0, 1.0 - weight_total)
    policy = PortfolioAllocationPolicy(
        method=method,
        pair_weights=tuple((item.pair_id, item.weight) for item in allocations),
        weights_normalized=bool(normalize_weights),
        cash_weight=cash_weight,
        cash_return=0.0,
        max_total_gross_exposure_ratio=limit,
    )
    contributions, portfolio_returns, cash_contribution, unavailable = (
        calculate_portfolio_returns(sources, allocations)
    )
    portfolio_equity = _compound_equity(portfolio_returns, capital)
    (
        pair_exposures,
        aggregate_exposures,
        symbol_exposures,
        shared_symbols,
        execution_rows,
    ) = calculate_portfolio_exposure_metrics(
        sources,
        allocations,
        portfolio_equity,
    )
    schedule = build_portfolio_schedule(
        portfolio_returns,
        portfolio_equity,
        cash_contribution,
        sources,
        allocations,
        aggregate_exposures,
        unavailable,
        max_total_gross_exposure_ratio=limit,
        execution_rows=execution_rows,
    )
    finite_returns = portfolio_returns.dropna()
    catastrophic_count = int(finite_returns.le(-1.0).sum())
    missing_count = int(portfolio_returns.isna().sum())
    positive_allocations = tuple(item for item in allocations if item.weight > 0.0)
    exposure_available_count = sum(item.exposure_available for item in positive_allocations)
    exposure_unavailable_count = len(positive_allocations) - exposure_available_count
    exposure_unavailable_rows = int(
        (~aggregate_exposures["exposure_available"].fillna(False).astype(bool)).sum()
    )
    if catastrophic_count or finite_returns.empty:
        availability = PortfolioAvailability.UNAVAILABLE
    elif missing_count or exposure_unavailable_count or exposure_unavailable_rows:
        availability = PortfolioAvailability.PARTIALLY_AVAILABLE
    else:
        availability = PortfolioAvailability.AVAILABLE
    weights = np.asarray([item.weight for item in allocations], dtype=float)
    hhi = float(np.square(weights).sum())
    effective = float(1.0 / hhi) if hhi > 0.0 else float("nan")
    finite_gross_ratio = aggregate_exposures["gross_exposure_ratio"].dropna()
    finite_symbol_fraction = aggregate_exposures[
        "largest_symbol_gross_exposure_fraction"
    ].dropna()
    metrics = PortfolioMetrics(
        pair_count=len(allocations),
        allocated_pair_count=len(positive_allocations),
        largest_pair_weight=float(weights.max()),
        allocation_hhi=hhi,
        effective_number_of_pairs=effective,
        cash_weight=cash_weight,
        unavailable_return_row_count=missing_count,
        catastrophic_return_row_count=catastrophic_count,
        exposure_available_pair_count=exposure_available_count,
        exposure_unavailable_pair_count=exposure_unavailable_count,
        exposure_unavailable_row_count=exposure_unavailable_rows,
        maximum_gross_exposure_ratio=(
            float(finite_gross_ratio.max())
            if not finite_gross_ratio.empty else float("nan")
        ),
        gross_exposure_limit_breach_count=int(
            schedule["gross_exposure_limit_breach"].sum()
        ),
        shared_symbols=shared_symbols,
        largest_symbol_gross_exposure_fraction=(
            float(finite_symbol_fraction.max())
            if not finite_symbol_fraction.empty else float("nan")
        ),
    )
    pair_positions = tuple(
        PortfolioPosition(
            pair_id=source.pair_id,
            allocation_weight=next(
                item.weight for item in allocations if item.pair_id == source.pair_id
            ),
            allocated_capital=next(
                item.allocated_capital
                for item in allocations
                if item.pair_id == source.pair_id
            ),
            active_state_available=not bool(source.active_state.isna().any()),
            active_observations=int(source.active_state.fillna(False).sum()),
            exposure_available=next(
                item.exposure_available
                for item in allocations
                if item.pair_id == source.pair_id
            ),
        )
        for source in sources
    )
    capital_policies = tuple(sorted({source.capital_policy for source in sources}))
    provenance_warnings = tuple(
        dict.fromkeys(
            warning
            for source in sources
            for warning in (
                *source.provenance_warnings,
                f"{source.pair_id} universe provenance: {source.universe_provenance}.",
                f"{source.pair_id} cleaning provenance: {source.cleaning_provenance}.",
            )
        )
    )
    warnings = [_PORTFOLIO_WARNING]
    if shared_symbols:
        warnings.append(
            "Shared symbols are aggregated at symbol level; pair gross exposures "
            "must not be interpreted as independent."
        )
    if exposure_unavailable_count:
        warnings.append(
            "One or more positive-weight sleeves lack an explicit source capital "
            "basis or pair exposure schedule; return aggregation remains available "
            "but aggregate exposure is unavailable."
        )
    if any(source.capital_policy == "equal_capital_reset" for source in sources):
        warnings.append(
            "At least one input uses equal-capital-reset walk-forward semantics; "
            "raw source dollar P&L is not aggregated."
        )
    sleeve_returns = pd.DataFrame(
        {source.pair_id: source.calendar_returns for source in sources},
        index=portfolio_returns.index.copy(),
    )
    return PortfolioResult(
        pair_ids=tuple(source.pair_id for source in sources),
        allocation_policy=policy,
        pair_allocations=allocations,
        pair_positions=pair_positions,
        portfolio_schedule=schedule,
        portfolio_returns=portfolio_returns,
        portfolio_equity=portfolio_equity,
        pair_sleeve_returns=sleeve_returns,
        pair_return_contributions=contributions,
        pair_exposures=pair_exposures,
        aggregate_exposures=aggregate_exposures,
        symbol_exposures=symbol_exposures,
        unavailable_rows=unavailable,
        metrics=metrics,
        availability=availability,
        initial_capital=capital,
        cash_weight=cash_weight,
        cash_capital=capital * cash_weight,
        all_pair_universes_point_in_time_validated=all(
            source.point_in_time_universe_validated for source in sources
        ),
        pair_capital_policies=capital_policies,
        pair_indices_differ=False,
        provenance_warnings=provenance_warnings,
        warnings=tuple(warnings),
    )
