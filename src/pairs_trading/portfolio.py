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
    "LeverageStatus",
    "PortfolioAvailability",
    "SourceCapitalProvenance",
    "SourceReturnPathPolicy",
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
_ACCOUNTING_TOLERANCE = 1e-9
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
    "Static sleeve initial weights are predefined and never changed using OOS "
    "outcomes. Sleeve equities compound independently without cross-sleeve "
    "capital transfers or portfolio-level rebalancing. "
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


class LeverageStatus(str, Enum):
    """Row-level evaluability and outcome of the configured gross limit."""

    WITHIN_LIMIT = "WITHIN_LIMIT"
    BREACH = "BREACH"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class SourceCapitalProvenance(str, Enum):
    """How a pair source's standalone capital basis was established."""

    INFERRED_AND_VERIFIED = "inferred_and_verified"
    CALLER_SUPPLIED_UNVERIFIED = "caller_supplied_unverified"
    UNAVAILABLE = "unavailable"


class SourceReturnPathPolicy(str, Enum):
    """Economic interpretation of a supplied pair return path."""

    CONTINUOUS_BACKTEST = "continuous_backtest_capital"
    SYNTHETIC_EQUAL_CAPITAL_RESET = "synthetic_equal_capital_reset_research_index"
    CALLER_SUPPLIED = "caller_supplied_return_path"


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
    source_capital_provenance: str = (
        SourceCapitalProvenance.CALLER_SUPPLIED_UNVERIFIED.value
    )
    source_return_path_policy: str = SourceReturnPathPolicy.CALLER_SUPPLIED.value

    def __post_init__(self) -> None:
        if not isinstance(self.point_in_time_universe_validated, (bool, np.bool_)):
            raise TypeError(
                "point_in_time_universe_validated must be an actual Boolean."
            )
        if not isinstance(self.execution_rows, pd.Series):
            raise TypeError("execution_rows must be a pandas Series.")
        _validate_boolean_series(
            self.execution_rows,
            f"{self.pair_id} execution_rows",
            allow_missing=False,
        )
        if self.source_capital_basis is not None:
            _finite_positive(
                self.source_capital_basis,
                f"{self.pair_id} source_capital_basis",
            )
        for value, name in (
            (self.source_capital_provenance, "source_capital_provenance"),
            (self.source_return_path_policy, "source_return_path_policy"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string.")
        if self.source_capital_provenance not in {
            item.value for item in SourceCapitalProvenance
        }:
            raise ValueError("source_capital_provenance is not recognized.")
        if self.source_return_path_policy not in {
            item.value for item in SourceReturnPathPolicy
        }:
            raise ValueError("source_return_path_policy is not recognized.")
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
    source_capital_provenance: str
    source_return_path_policy: str

    @property
    def initial_weight(self) -> float:
        """Initial sleeve weight; it is never a rebalancing target."""
        return self.weight

    @property
    def initial_capital(self) -> float:
        """Capital contributed to the sleeve once at portfolio inception."""
        return self.allocated_capital


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
    final_sleeve_equity: float
    sleeve_insolvent: bool


@dataclass(frozen=True)
class PortfolioMetrics:
    """Concentration, availability, and exposure diagnostics."""

    pair_count: int
    allocated_pair_count: int
    largest_pair_weight: float
    pair_hhi: float
    effective_allocated_pair_count: float
    whole_portfolio_hhi: float
    cash_weight: float
    unavailable_return_row_count: int
    catastrophic_pair_row_count: int
    catastrophic_portfolio_row_count: int
    insolvent_pair_count: int
    exposure_available_pair_count: int
    exposure_unavailable_pair_count: int
    exposure_unavailable_row_count: int
    maximum_gross_exposure_ratio: float
    gross_exposure_limit_breach_count: int
    leverage_evaluated_row_count: int
    leverage_within_limit_row_count: int
    leverage_unevaluable_row_count: int
    shared_symbols: tuple[str, ...]
    largest_symbol_unnetted_sleeve_gross_fraction: float

    @property
    def allocation_hhi(self) -> float:
        """Compatibility alias for conditional pair HHI."""
        return self.pair_hhi

    @property
    def effective_number_of_pairs(self) -> float:
        """Compatibility alias for effective allocated pair count."""
        return self.effective_allocated_pair_count

    @property
    def catastrophic_return_row_count(self) -> int:
        """Compatibility alias for aggregate portfolio catastrophe rows."""
        return self.catastrophic_portfolio_row_count

    @property
    def largest_symbol_gross_exposure_fraction(self) -> float:
        """Compatibility alias for unnetted pair-sleeve symbol gross."""
        return self.largest_symbol_unnetted_sleeve_gross_fraction


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
    portfolio_pnl: pd.Series
    pair_sleeve_returns: pd.DataFrame
    pair_prior_sleeve_equity: pd.DataFrame
    pair_sleeve_equity: pd.DataFrame
    pair_pnl_contributions: pd.DataFrame
    pair_return_contributions: pd.DataFrame
    pair_current_equity_weights: pd.DataFrame
    pair_insolvency_state: pd.DataFrame
    catastrophic_pair_rows: pd.DataFrame
    pair_exposures: pd.DataFrame
    aggregate_exposures: pd.DataFrame
    symbol_exposures: pd.DataFrame
    leverage_status: pd.Series
    unavailable_rows: pd.DataFrame
    metrics: PortfolioMetrics
    availability: PortfolioAvailability
    initial_capital: float
    cash_weight: float
    cash_capital: float
    all_pair_universes_point_in_time_validated: bool
    pair_capital_policies: tuple[str, ...]
    source_path_provenance: tuple[tuple[str, str, str], ...]
    contains_synthetic_reset_sources: bool
    self_financing_interpretation: str
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
            "pair_prior_sleeve_equity",
            "pair_sleeve_equity",
            "pair_pnl_contributions",
            "pair_return_contributions",
            "pair_current_equity_weights",
            "pair_insolvency_state",
            "catastrophic_pair_rows",
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
        object.__setattr__(self, "portfolio_pnl", self.portfolio_pnl.copy(deep=True))
        object.__setattr__(
            self,
            "leverage_status",
            self.leverage_status.copy(deep=True),
        )
        object.__setattr__(
            self,
            "pair_capital_policies",
            tuple(self.pair_capital_policies),
        )
        object.__setattr__(
            self,
            "source_path_provenance",
            tuple(tuple(item) for item in self.source_path_provenance),
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


def _validate_boolean_series(
    series: pd.Series,
    name: str,
    *,
    allow_missing: bool,
) -> pd.Series:
    """Return an owned Boolean Series without truthiness coercion."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if not allow_missing and bool(series.isna().any()):
        raise ValueError(f"{name} must not contain missing values.")
    nonmissing = series.loc[series.notna()].tolist()
    if any(not isinstance(value, (bool, np.bool_)) for value in nonmissing):
        raise TypeError(f"{name} must contain only actual Boolean values.")
    dtype = "boolean" if allow_missing else bool
    return series.astype(dtype).copy(deep=True)


def _validate_exposure_frame(
    frame: pd.DataFrame,
    pair_id: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate complete exposure tuples and identify usable rows.

    All-missing or partially missing tuples are legitimate unavailable source
    observations.  A fully populated tuple is required to be finite,
    non-negative where appropriate, and internally reconcilable.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{pair_id} source_exposure must be a DataFrame.")
    if not set(_EXPOSURE_COLUMNS).issubset(frame.columns):
        raise ValueError(f"{pair_id} source_exposure lacks required columns.")
    result = frame.loc[:, _EXPOSURE_COLUMNS].copy(deep=True)
    for column in _EXPOSURE_COLUMNS:
        nonmissing = result[column].loc[result[column].notna()].tolist()
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in nonmissing
        ):
            raise TypeError(
                f"{pair_id} exposure column {column!r} must contain only "
                "non-Boolean real numbers or missing values."
            )
        result[column] = result[column].astype(float)
    complete = result.notna().all(axis=1)
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError(f"{pair_id} exposure data must not contain infinity.")
    if not bool(complete.any()):
        return result, complete.rename("exposure_available")

    usable = result.loc[complete]
    for column in ("gross_exposure", "long_exposure", "short_exposure"):
        if bool(usable[column].lt(-_ACCOUNTING_TOLERANCE).any()):
            raise ValueError(
                f"{pair_id} {column} must be non-negative on complete rows."
            )

    def require_close(left: pd.Series, right: pd.Series, identity: str) -> None:
        if not np.allclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            rtol=_ACCOUNTING_TOLERANCE,
            atol=_ACCOUNTING_TOLERANCE,
        ):
            raise ValueError(f"{pair_id} exposure identity failed: {identity}.")

    market_y = usable["market_value_y"]
    market_x = usable["market_value_x"]
    require_close(
        usable["gross_exposure"],
        market_y.abs() + market_x.abs(),
        "gross_exposure = abs(market_value_y) + abs(market_value_x)",
    )
    require_close(
        usable["gross_exposure"],
        usable["long_exposure"] + usable["short_exposure"],
        "gross_exposure = long_exposure + short_exposure",
    )
    require_close(
        usable["net_exposure"],
        market_y + market_x,
        "net_exposure = market_value_y + market_value_x",
    )
    require_close(
        usable["net_exposure"],
        usable["long_exposure"] - usable["short_exposure"],
        "net_exposure = long_exposure - short_exposure",
    )
    return result, complete.rename("exposure_available")


def _infer_backtest_initial_capital(
    accounting: pd.DataFrame,
    pair_id: str,
) -> float | None:
    """Infer and validate standalone initial capital from backtest accounting."""
    required = (
        "net_equity_after_carry",
        "cumulative_net_pnl_after_carry",
    )
    if not set(required).issubset(accounting.columns):
        return None
    values = accounting.loc[:, required].copy(deep=True)
    for column in required:
        nonmissing = values[column].loc[values[column].notna()].tolist()
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in nonmissing
        ):
            raise TypeError(
                f"{pair_id} accounting column {column!r} must contain real numbers."
            )
        values[column] = values[column].astype(float)
    finite = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    inferred = (
        values.loc[finite, "net_equity_after_carry"]
        - values.loc[finite, "cumulative_net_pnl_after_carry"]
    )
    if inferred.empty:
        return None
    basis = float(inferred.iloc[0])
    if not np.isfinite(basis) or basis <= 0.0:
        raise ValueError(f"{pair_id} inferred source capital must be positive.")
    if not np.allclose(
        inferred.to_numpy(dtype=float),
        basis,
        rtol=_ACCOUNTING_TOLERANCE,
        atol=_ACCOUNTING_TOLERANCE * max(1.0, abs(basis)),
    ):
        raise ValueError(
            f"{pair_id} source initial capital is inconsistent across accounting rows."
        )
    return basis


def _capital_bases_agree(left: float, right: float) -> bool:
    return bool(
        np.isclose(
            left,
            right,
            rtol=_ACCOUNTING_TOLERANCE,
            atol=_ACCOUNTING_TOLERANCE * max(1.0, abs(left), abs(right)),
        )
    )


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
        rebalance = _validate_boolean_series(
            positions["rebalance"],
            "BacktestResult rebalance",
            allow_missing=False,
        )
        execution |= rebalance
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
    inferred_capital = _infer_backtest_initial_capital(accounting, pair_id)
    if inferred_capital is not None:
        if capital_basis is not None and not _capital_bases_agree(
            capital_basis,
            inferred_capital,
        ):
            raise ValueError(
                f"{pair_id} supplied source capital basis conflicts with "
                "BacktestResult accounting."
            )
        effective_capital = inferred_capital
        capital_provenance = SourceCapitalProvenance.INFERRED_AND_VERIFIED.value
    else:
        effective_capital = capital_basis
        capital_provenance = (
            SourceCapitalProvenance.CALLER_SUPPLIED_UNVERIFIED.value
            if capital_basis is not None
            else SourceCapitalProvenance.UNAVAILABLE.value
        )
    if set(_EXPOSURE_COLUMNS).issubset(accounting.columns):
        exposure, _ = _validate_exposure_frame(accounting, pair_id)
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
        source_capital_basis=effective_capital,
        source_type="BacktestResult",
        capital_policy="standalone_pair_capital",
        point_in_time_universe_validated=False,
        universe_provenance="not_exposed_by_backtest_result",
        cleaning_provenance="not_exposed_by_backtest_result",
        provenance_warnings=(result.research_metadata.warning,),
        source_capital_provenance=capital_provenance,
        source_return_path_policy=SourceReturnPathPolicy.CONTINUOUS_BACKTEST.value,
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
        source_capital_provenance=(
            SourceCapitalProvenance.CALLER_SUPPLIED_UNVERIFIED.value
            if capital_basis is not None
            else SourceCapitalProvenance.UNAVAILABLE.value
        ),
        source_return_path_policy=(
            SourceReturnPathPolicy.SYNTHETIC_EQUAL_CAPITAL_RESET.value
            if result.capital_policy == "equal_capital_reset"
            else SourceReturnPathPolicy.CALLER_SUPPLIED.value
        ),
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
            if (
                capital_basis is not None
                and result.source_capital_basis is not None
                and not _capital_bases_agree(
                    capital_basis,
                    float(result.source_capital_basis),
                )
            ):
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
            validated_execution = _validate_boolean_series(
                result.execution_rows,
                f"{pair_id} execution_rows",
                allow_missing=False,
            )
            exposure, _ = _validate_exposure_frame(result.source_exposure, pair_id)
            capital_provenance = (
                SourceCapitalProvenance.CALLER_SUPPLIED_UNVERIFIED.value
                if effective_basis is not None
                else SourceCapitalProvenance.UNAVAILABLE.value
            )
            source = PairPortfolioInput(
                **{
                    **result.__dict__,
                    "calendar_returns": validated_returns,
                    "active_state": result.active_state.astype("boolean"),
                    "execution_rows": validated_execution,
                    "source_exposure": exposure,
                    "source_capital_basis": effective_basis,
                    "source_capital_provenance": capital_provenance,
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
    normalized_values = {
        pair_id: values[pair_id] / total for pair_id in normalized_ids
    }
    if normalized_ids:
        final_id = normalized_ids[-1]
        normalized_values[final_id] = 1.0 - sum(
            normalized_values[pair_id] for pair_id in normalized_ids[:-1]
        )
    return tuple((pair_id, normalized_values[pair_id]) for pair_id in normalized_ids)


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
        if abs(total - 1.0) <= _WEIGHT_TOLERANCE and total != 1.0:
            # Canonicalize representational noise instead of retaining a hidden
            # over-allocation and merely clipping the cash sleeve.
            ordered_ids = tuple(sorted(weights))
            weights = {
                pair_id: weight / total
                for pair_id, weight in weights.items()
            }
            final_id = ordered_ids[-1]
            weights[final_id] = 1.0 - sum(
                weights[pair_id] for pair_id in ordered_ids[:-1]
            )
    allocations: list[PairAllocation] = []
    for source in sources:
        weight = float(weights[source.pair_id])
        allocated = capital * weight
        scale = (
            allocated / source.source_capital_basis
            if source.source_capital_basis is not None
            else None
        )
        _, complete_exposure = _validate_exposure_frame(
            source.source_exposure,
            source.pair_id,
        )
        exposure_available = bool(
            weight == 0.0
            or (
                scale is not None
                and not source.source_exposure.empty
                and bool(complete_exposure.any())
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
                source_capital_provenance=source.source_capital_provenance,
                source_return_path_policy=source.source_return_path_policy,
            )
        )
    return tuple(allocations)


@dataclass(frozen=True)
class _StaticSleeveAccounting:
    sleeve_returns: pd.DataFrame
    prior_sleeve_equity: pd.DataFrame
    sleeve_equity: pd.DataFrame
    sleeve_pnl: pd.DataFrame
    return_contributions: pd.DataFrame
    current_equity_weights: pd.DataFrame
    insolvency_state: pd.DataFrame
    catastrophic_pair_rows: pd.DataFrame
    portfolio_pnl: pd.Series
    portfolio_returns: pd.Series
    portfolio_equity: pd.Series
    cash_return_contribution: pd.Series
    unavailable_rows: pd.DataFrame
    portfolio_catastrophic_state: pd.Series


def _derived_initial_capital(
    allocations: tuple[PairAllocation, ...],
) -> float:
    candidates = [
        item.allocated_capital / item.weight
        for item in allocations
        if item.weight > 0.0
    ]
    if not candidates:
        raise ValueError(
            "portfolio_initial_capital is required when every pair weight is zero."
        )
    capital = float(candidates[0])
    if not np.allclose(
        np.asarray(candidates, dtype=float),
        capital,
        rtol=_ACCOUNTING_TOLERANCE,
        atol=_ACCOUNTING_TOLERANCE * max(1.0, abs(capital)),
    ):
        raise ValueError("allocations imply inconsistent portfolio initial capital.")
    return capital


def _calculate_static_sleeve_accounting(
    pair_inputs: Iterable[PairPortfolioInput],
    allocations: Iterable[PairAllocation],
    portfolio_initial_capital: Any,
) -> _StaticSleeveAccounting:
    """Compound independently funded sleeves without capital recycling."""
    sources = tuple(sorted(tuple(pair_inputs), key=lambda item: item.pair_id))
    if not sources:
        raise ValueError("pair_inputs must not be empty.")
    capital = _finite_positive(portfolio_initial_capital, "portfolio_initial_capital")
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
    columns = sleeve_returns.columns
    prior_equity = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    sleeve_equity = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    sleeve_pnl = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    insolvency = pd.DataFrame(False, index=index, columns=columns, dtype=bool)
    catastrophic_rows = pd.DataFrame(False, index=index, columns=columns, dtype=bool)

    for source in sources:
        pair_id = source.pair_id
        allocation = allocation_by_id[pair_id]
        running_equity = float(allocation.allocated_capital)
        continuous = True
        insolvent = False
        for row_number, pair_return in enumerate(
            sleeve_returns[pair_id].to_numpy(dtype=float)
        ):
            if allocation.weight == 0.0:
                prior_equity.at[index[row_number], pair_id] = 0.0
                sleeve_pnl.at[index[row_number], pair_id] = 0.0
                sleeve_equity.at[index[row_number], pair_id] = 0.0
                continue
            catastrophic_rows.at[index[row_number], pair_id] = bool(
                np.isfinite(pair_return) and pair_return <= -1.0
            )
            if not continuous:
                insolvency.at[index[row_number], pair_id] = insolvent
                continue
            prior_equity.at[index[row_number], pair_id] = running_equity
            if insolvent:
                sleeve_pnl.at[index[row_number], pair_id] = 0.0
                sleeve_equity.at[index[row_number], pair_id] = 0.0
                insolvency.at[index[row_number], pair_id] = True
                continue
            if not np.isfinite(pair_return):
                continuous = False
                continue
            if pair_return < -1.0:
                # Negative sleeve wealth is outside the supported static-capital
                # policy.  Preserve the raw source return but invalidate wealth.
                continuous = False
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                pnl = running_equity * pair_return
                closing = running_equity + pnl
            if not np.isfinite(pnl) or not np.isfinite(closing):
                continuous = False
                continue
            if pair_return == -1.0:
                closing = 0.0
                insolvent = True
            sleeve_pnl.at[index[row_number], pair_id] = pnl
            sleeve_equity.at[index[row_number], pair_id] = closing
            insolvency.at[index[row_number], pair_id] = insolvent
            running_equity = closing

    cash_weight = 1.0 - float(sum(item.weight for item in allocation_tuple))
    if abs(cash_weight) <= _WEIGHT_TOLERANCE:
        cash_weight = 0.0
    if cash_weight < 0.0:
        raise ValueError("allocations exceed portfolio initial capital.")
    cash_equity = capital * cash_weight

    portfolio_pnl_values = np.full(len(index), np.nan, dtype=float)
    portfolio_return_values = np.full(len(index), np.nan, dtype=float)
    portfolio_equity_values = np.full(len(index), np.nan, dtype=float)
    portfolio_catastrophic = np.zeros(len(index), dtype=bool)
    return_contributions = pd.DataFrame(
        np.nan,
        index=index,
        columns=columns,
        dtype=float,
    )
    current_weights = pd.DataFrame(
        np.nan,
        index=index,
        columns=columns,
        dtype=float,
    )
    unavailable_ids: list[tuple[str, ...]] = []
    reasons: list[str | None] = []
    invalid_ids_by_row: list[tuple[str, ...]] = []
    insolvent_ids_by_row: list[tuple[str, ...]] = []
    prior_portfolio_equity = capital
    portfolio_continuous = True
    portfolio_insolvent = False
    for row_number in range(len(index)):
        positive_ids = [
            source.pair_id
            for source in sources
            if allocation_by_id[source.pair_id].weight > 0.0
        ]
        unavailable = tuple(
            pair_id
            for pair_id in positive_ids
            if not np.isfinite(sleeve_equity[pair_id].iat[row_number])
        )
        invalid = tuple(
            pair_id
            for pair_id in positive_ids
            if (
                np.isfinite(sleeve_returns[pair_id].iat[row_number])
                and sleeve_returns[pair_id].iat[row_number] < -1.0
            )
        )
        insolvent_ids = tuple(
            pair_id
            for pair_id in positive_ids
            if bool(insolvency[pair_id].iat[row_number])
        )
        unavailable_ids.append(tuple(sorted(unavailable)))
        invalid_ids_by_row.append(tuple(sorted(invalid)))
        insolvent_ids_by_row.append(tuple(sorted(insolvent_ids)))

        if portfolio_continuous and prior_portfolio_equity > 0.0:
            for pair_id in columns:
                pnl = sleeve_pnl[pair_id].iat[row_number]
                if np.isfinite(pnl):
                    return_contributions.at[index[row_number], pair_id] = (
                        pnl / prior_portfolio_equity
                    )

        if portfolio_continuous and not unavailable:
            row_pnl = float(sleeve_pnl.iloc[row_number].sum(skipna=False))
            closing_equity = cash_equity + float(
                sleeve_equity.iloc[row_number].sum(skipna=False)
            )
            if not np.isfinite(row_pnl) or not np.isfinite(closing_equity):
                portfolio_continuous = False
            else:
                portfolio_pnl_values[row_number] = row_pnl
                portfolio_equity_values[row_number] = closing_equity
                if prior_portfolio_equity > 0.0:
                    portfolio_return_values[row_number] = (
                        row_pnl / prior_portfolio_equity
                    )
                if closing_equity > 0.0:
                    for pair_id in columns:
                        equity_value = sleeve_equity[pair_id].iat[row_number]
                        if np.isfinite(equity_value):
                            current_weights.at[index[row_number], pair_id] = (
                                equity_value / closing_equity
                            )
                if closing_equity == 0.0:
                    portfolio_insolvent = True
                prior_portfolio_equity = closing_equity
        else:
            portfolio_continuous = False

        portfolio_catastrophic[row_number] = portfolio_insolvent
        if invalid:
            reasons.append("Pair return below -100% invalidated sleeve wealth continuity.")
        elif unavailable:
            reasons.append(
                "Positive-weight sleeve wealth is unavailable; capital was not "
                "renormalized or restored."
            )
        elif portfolio_insolvent:
            reasons.append("Aggregate portfolio capital is exhausted and not replenished.")
        else:
            reasons.append(None)

    portfolio_returns = pd.Series(
        portfolio_return_values, index=index, name="portfolio_return"
    )
    portfolio_equity = pd.Series(
        portfolio_equity_values, index=index, name="portfolio_equity"
    )
    portfolio_pnl = pd.Series(portfolio_pnl_values, index=index, name="portfolio_pnl")
    cash_contribution = pd.Series(0.0, index=index, name="cash_return_contribution")
    unavailable = pd.DataFrame(
        {
            "unavailable_pair_ids": unavailable_ids,
            "invalid_pair_ids": invalid_ids_by_row,
            "insolvent_pair_ids": insolvent_ids_by_row,
            "reason": reasons,
        },
        index=index,
    )
    return _StaticSleeveAccounting(
        sleeve_returns=sleeve_returns,
        prior_sleeve_equity=prior_equity,
        sleeve_equity=sleeve_equity,
        sleeve_pnl=sleeve_pnl,
        return_contributions=return_contributions,
        current_equity_weights=current_weights,
        insolvency_state=insolvency,
        catastrophic_pair_rows=catastrophic_rows,
        portfolio_pnl=portfolio_pnl,
        portfolio_returns=portfolio_returns,
        portfolio_equity=portfolio_equity,
        cash_return_contribution=cash_contribution,
        unavailable_rows=unavailable,
        portfolio_catastrophic_state=pd.Series(
            portfolio_catastrophic,
            index=index,
            name="portfolio_catastrophic",
        ),
    )


def calculate_portfolio_returns(
    pair_inputs: Iterable[PairPortfolioInput],
    allocations: Iterable[PairAllocation],
    *,
    portfolio_initial_capital: Any | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return causal static-sleeve contributions and portfolio returns.

    The four-value compatibility return is retained.  Contributions are now
    sleeve P&L divided by prior portfolio equity, rather than original static
    weights multiplied by every later pair return.
    """
    allocation_tuple = tuple(allocations)
    capital = (
        _derived_initial_capital(allocation_tuple)
        if portfolio_initial_capital is None
        else _finite_positive(portfolio_initial_capital, "portfolio_initial_capital")
    )
    result = _calculate_static_sleeve_accounting(
        pair_inputs,
        allocation_tuple,
        capital,
    )
    return (
        result.return_contributions.copy(deep=True),
        result.portfolio_returns.copy(deep=True),
        result.cash_return_contribution.copy(deep=True),
        result.unavailable_rows.copy(deep=True),
    )


def calculate_portfolio_exposure_metrics(
    pair_inputs: Iterable[PairPortfolioInput],
    allocations: Iterable[PairAllocation],
    portfolio_equity: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...], pd.Series]:
    """Scale complete source exposure tuples using initial sleeve funding.

    Portfolio-level sleeve equity drift never resizes a source strategy's
    holdings.  Every aggregate uses the same canonical complete-row mask.
    """
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
        (
            [item.pair_id for item in sources],
            (*_EXPOSURE_COLUMNS, "gross_exposure_fraction", "exposure_available"),
        ),
        names=("pair_id", "metric"),
    )
    pair_exposures = pd.DataFrame(np.nan, index=index, columns=pair_columns)
    symbols_to_pairs: dict[str, set[str]] = {}
    available_by_pair: dict[str, pd.Series] = {}
    for source in sources:
        symbols_to_pairs.setdefault(source.symbol_y, set()).add(source.pair_id)
        symbols_to_pairs.setdefault(source.symbol_x, set()).add(source.pair_id)
        allocation = allocation_by_id[source.pair_id]
        if allocation.weight == 0.0:
            scaled = pd.DataFrame(0.0, index=index, columns=_EXPOSURE_COLUMNS)
            available = pd.Series(True, index=index, dtype=bool)
        elif allocation.exposure_scaling_factor is None:
            available_by_pair[source.pair_id] = pd.Series(
                False, index=index, dtype=bool
            )
            pair_exposures[(source.pair_id, "exposure_available")] = False
            continue
        else:
            validated, available = _validate_exposure_frame(
                source.source_exposure,
                source.pair_id,
            )
            scaled = validated.astype(float)
            scaled = scaled * allocation.exposure_scaling_factor
            scaled.loc[~available, :] = np.nan
        available_by_pair[source.pair_id] = available.astype(bool)
        for column in _EXPOSURE_COLUMNS:
            pair_exposures[(source.pair_id, column)] = scaled[column]
        denominator = portfolio_equity.where(portfolio_equity > 0.0)
        pair_exposures[(source.pair_id, "gross_exposure_fraction")] = (
            scaled["gross_exposure"] / denominator
        )
        pair_exposures[(source.pair_id, "exposure_available")] = available.astype(bool)

    aggregate = pd.DataFrame(
        np.nan,
        index=index,
        columns=(
            "total_gross_exposure",
            "total_long_exposure",
            "total_short_exposure",
            "total_net_exposure",
            "gross_exposure_ratio",
            "largest_symbol_unnetted_sleeve_gross_fraction",
            "exposure_available",
        ),
    )
    positive_allocations = tuple(
        allocation for allocation in allocation_tuple if allocation.weight > 0.0
    )
    if positive_allocations:
        rows_known = pd.Series(True, index=index, dtype=bool)
        for allocation in positive_allocations:
            rows_known &= available_by_pair[allocation.pair_id]
    else:
        rows_known = pd.Series(True, index=index, dtype=bool)
    if bool(rows_known.any()):
        gross_frame = pair_exposures.xs("gross_exposure", level="metric", axis=1)
        long_frame = pair_exposures.xs("long_exposure", level="metric", axis=1)
        short_frame = pair_exposures.xs("short_exposure", level="metric", axis=1)
        net_frame = pair_exposures.xs("net_exposure", level="metric", axis=1)
        aggregate.loc[rows_known, "total_gross_exposure"] = gross_frame.loc[
            rows_known
        ].sum(axis=1, skipna=False)
        aggregate.loc[rows_known, "total_long_exposure"] = long_frame.loc[
            rows_known
        ].sum(axis=1, skipna=False)
        aggregate.loc[rows_known, "total_short_exposure"] = short_frame.loc[
            rows_known
        ].sum(axis=1, skipna=False)
        aggregate.loc[rows_known, "total_net_exposure"] = net_frame.loc[
            rows_known
        ].sum(axis=1, skipna=False)
        aggregate.loc[rows_known, "gross_exposure_ratio"] = (
            aggregate.loc[rows_known, "total_gross_exposure"]
            / portfolio_equity.loc[rows_known].where(portfolio_equity.loc[rows_known] > 0.0)
        )
    aggregate["exposure_available"] = rows_known.astype(bool)

    symbols = tuple(sorted(symbols_to_pairs))
    symbol_columns = pd.MultiIndex.from_product(
        (
            symbols,
            (
                "net_market_value",
                "unnetted_sleeve_gross_market_value",
                "consolidated_gross_market_value",
                "unnetted_sleeve_gross_fraction",
            ),
        ),
        names=("symbol", "metric"),
    )
    symbol_exposures = pd.DataFrame(np.nan, index=index, columns=symbol_columns)
    for symbol in symbols:
        signed_legs: list[pd.Series] = []
        for source in sources:
            allocation = allocation_by_id[source.pair_id]
            scale = (
                0.0
                if allocation.weight == 0.0
                else allocation.exposure_scaling_factor
            )
            if scale is None:
                continue
            available = available_by_pair[source.pair_id]
            if source.symbol_y == symbol:
                leg = source.source_exposure["market_value_y"].astype(float) * scale
                signed_legs.append(leg.where(available))
            if source.symbol_x == symbol:
                leg = source.source_exposure["market_value_x"].astype(float) * scale
                signed_legs.append(leg.where(available))
        if not signed_legs:
            continue
        signed_frame = pd.concat(signed_legs, axis=1)
        net = signed_frame.loc[rows_known].sum(axis=1, skipna=False)
        unnetted = signed_frame.loc[rows_known].abs().sum(axis=1, skipna=False)
        symbol_exposures.loc[rows_known, (symbol, "net_market_value")] = net
        symbol_exposures.loc[
            rows_known, (symbol, "unnetted_sleeve_gross_market_value")
        ] = unnetted
        symbol_exposures.loc[
            rows_known, (symbol, "consolidated_gross_market_value")
        ] = net.abs()
        symbol_exposures.loc[
            rows_known, (symbol, "unnetted_sleeve_gross_fraction")
        ] = unnetted / aggregate.loc[rows_known, "total_gross_exposure"].replace(
            0.0, np.nan
        )
    if symbols:
        gross_symbols = symbol_exposures.xs(
            "unnetted_sleeve_gross_market_value", level="metric", axis=1
        )
        aggregate["largest_symbol_unnetted_sleeve_gross_fraction"] = (
            gross_symbols.max(axis=1, skipna=False)
            / aggregate["total_gross_exposure"].replace(0.0, np.nan)
        )
    shared_symbols = tuple(
        symbol for symbol, pair_ids in sorted(symbols_to_pairs.items()) if len(pair_ids) > 1
    )
    execution_rows = pd.Series(False, index=index, dtype=bool, name="execution_row")
    for source in sources:
        if allocation_by_id[source.pair_id].weight > 0.0:
            source_execution = _validate_boolean_series(
                source.execution_rows,
                f"{source.pair_id} execution_rows",
                allow_missing=False,
            )
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
    portfolio_pnl: pd.Series | None = None,
    portfolio_catastrophic_state: pd.Series | None = None,
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
    execution_rows = _validate_boolean_series(
        execution_rows,
        "execution_rows",
        allow_missing=False,
    )
    if portfolio_pnl is None:
        portfolio_pnl = pd.Series(np.nan, index=index, name="portfolio_pnl")
    if not portfolio_pnl.index.equals(index):
        raise ValueError("portfolio_pnl must align exactly with portfolio returns.")
    if portfolio_catastrophic_state is None:
        portfolio_catastrophic_state = portfolio_returns.le(-1.0).cummax()
    if not portfolio_catastrophic_state.index.equals(index):
        raise ValueError(
            "portfolio_catastrophic_state must align exactly with portfolio returns."
        )
    portfolio_catastrophic_state = _validate_boolean_series(
        portfolio_catastrophic_state,
        "portfolio_catastrophic_state",
        allow_missing=False,
    )
    gross_ratio = aggregate_exposures["gross_exposure_ratio"].astype(float)
    if max_total_gross_exposure_ratio is None:
        limit = None
        leverage_status = pd.Series(
            [LeverageStatus.NOT_CONFIGURED] * len(index),
            index=index,
            name="leverage_status",
            dtype=object,
        )
        breaches = pd.Series(False, index=index, dtype="boolean")
    else:
        limit = _finite_positive(
            max_total_gross_exposure_ratio,
            "max_total_gross_exposure_ratio",
        )
        statuses = np.empty(len(index), dtype=object)
        statuses[:] = [LeverageStatus.UNEVALUABLE] * len(index)
        evaluable = (
            aggregate_exposures["exposure_available"].fillna(False).astype(bool)
            & gross_ratio.notna()
        )
        within = evaluable & gross_ratio.le(limit + _WEIGHT_TOLERANCE)
        breached = evaluable & gross_ratio.gt(limit + _WEIGHT_TOLERANCE)
        statuses[within.to_numpy(dtype=bool)] = LeverageStatus.WITHIN_LIMIT
        statuses[breached.to_numpy(dtype=bool)] = LeverageStatus.BREACH
        leverage_status = pd.Series(
            statuses,
            index=index,
            name="leverage_status",
            dtype=object,
        )
        breaches = pd.Series(pd.NA, index=index, dtype="boolean")
        breaches.loc[within] = False
        breaches.loc[breached] = True
        initial_or_execution = execution_rows.copy(deep=True)
        if len(initial_or_execution):
            initial_or_execution.iloc[0] = True
        unevaluable_enforcement = (
            leverage_status.eq(LeverageStatus.UNEVALUABLE) & initial_or_execution
        )
        if bool(unevaluable_enforcement.any()):
            labels = list(index[unevaluable_enforcement])
            raise ValueError(
                "Gross exposure is unevaluable on an initial/execution row while "
                f"a hard leverage limit is configured: {labels}."
            )
        execution_breaches = breached & execution_rows
        if bool(execution_breaches.any()):
            labels = list(index[execution_breaches])
            raise ValueError(
                "Static allocation exceeds max_total_gross_exposure_ratio on "
                f"an initial/execution row: {labels}."
            )
    schedule = pd.DataFrame(index=index)
    schedule["portfolio_return"] = portfolio_returns
    schedule["portfolio_pnl"] = portfolio_pnl
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
    schedule["leverage_status"] = leverage_status
    schedule["execution_row"] = execution_rows
    schedule["catastrophic_portfolio_return"] = portfolio_returns.le(-1.0).fillna(False)
    schedule["portfolio_catastrophic"] = portfolio_catastrophic_state
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
    if abs(weight_total - 1.0) <= _WEIGHT_TOLERANCE:
        weight_total = 1.0
    cash_weight = 1.0 - weight_total
    if cash_weight == 0.0:
        cash_weight = 0.0
    if cash_weight < 0.0:
        raise ValueError("Pair weights exceed total portfolio capital.")
    policy = PortfolioAllocationPolicy(
        method=method,
        pair_weights=tuple((item.pair_id, item.weight) for item in allocations),
        weights_normalized=bool(normalize_weights),
        cash_weight=cash_weight,
        cash_return=0.0,
        max_total_gross_exposure_ratio=limit,
    )
    accounting = _calculate_static_sleeve_accounting(
        sources,
        allocations,
        capital,
    )
    portfolio_returns = accounting.portfolio_returns
    portfolio_equity = accounting.portfolio_equity
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
        accounting.cash_return_contribution,
        sources,
        allocations,
        aggregate_exposures,
        accounting.unavailable_rows,
        portfolio_pnl=accounting.portfolio_pnl,
        portfolio_catastrophic_state=accounting.portfolio_catastrophic_state,
        max_total_gross_exposure_ratio=limit,
        execution_rows=execution_rows,
    )
    finite_returns = portfolio_returns.dropna()
    catastrophic_portfolio_count = int(finite_returns.le(-1.0).sum())
    catastrophic_pair_count = int(
        accounting.catastrophic_pair_rows.to_numpy(dtype=bool).sum()
    )
    insolvent_pair_ids = tuple(
        pair_id
        for pair_id in accounting.insolvency_state.columns
        if bool(accounting.insolvency_state[pair_id].any())
    )
    missing_count = int(portfolio_returns.isna().sum())
    positive_allocations = tuple(item for item in allocations if item.weight > 0.0)
    exposure_available_count = sum(item.exposure_available for item in positive_allocations)
    exposure_unavailable_count = len(positive_allocations) - exposure_available_count
    exposure_unavailable_rows = int(
        (~aggregate_exposures["exposure_available"].fillna(False).astype(bool)).sum()
    )
    invalid_pair_path = bool(
        accounting.unavailable_rows["invalid_pair_ids"].map(bool).any()
    )
    primary_wealth_broken = bool(missing_count) or invalid_pair_path
    if catastrophic_portfolio_count or finite_returns.empty or primary_wealth_broken:
        availability = PortfolioAvailability.UNAVAILABLE
    elif (
        insolvent_pair_ids
        or exposure_unavailable_count
        or exposure_unavailable_rows
        or bool(schedule["leverage_status"].eq(LeverageStatus.UNEVALUABLE).any())
    ):
        availability = PortfolioAvailability.PARTIALLY_AVAILABLE
    else:
        availability = PortfolioAvailability.AVAILABLE
    weights = np.asarray([item.weight for item in allocations], dtype=float)
    invested_weight = float(weights.sum())
    if invested_weight > 0.0:
        conditional_weights = weights[weights > 0.0] / invested_weight
        pair_hhi = float(np.square(conditional_weights).sum())
        effective = float(1.0 / pair_hhi)
    else:
        pair_hhi = float("nan")
        effective = 0.0
    whole_portfolio_hhi = float(np.square(weights).sum() + cash_weight**2)
    finite_gross_ratio = aggregate_exposures["gross_exposure_ratio"].dropna()
    finite_symbol_fraction = aggregate_exposures[
        "largest_symbol_unnetted_sleeve_gross_fraction"
    ].dropna()
    leverage_status = schedule["leverage_status"]
    leverage_evaluated = int(
        leverage_status.isin(
            (LeverageStatus.WITHIN_LIMIT, LeverageStatus.BREACH)
        ).sum()
    )
    metrics = PortfolioMetrics(
        pair_count=len(allocations),
        allocated_pair_count=len(positive_allocations),
        largest_pair_weight=float(weights.max()),
        pair_hhi=pair_hhi,
        effective_allocated_pair_count=effective,
        whole_portfolio_hhi=whole_portfolio_hhi,
        cash_weight=cash_weight,
        unavailable_return_row_count=missing_count,
        catastrophic_pair_row_count=catastrophic_pair_count,
        catastrophic_portfolio_row_count=catastrophic_portfolio_count,
        insolvent_pair_count=len(insolvent_pair_ids),
        exposure_available_pair_count=exposure_available_count,
        exposure_unavailable_pair_count=exposure_unavailable_count,
        exposure_unavailable_row_count=exposure_unavailable_rows,
        maximum_gross_exposure_ratio=(
            float(finite_gross_ratio.max())
            if not finite_gross_ratio.empty else float("nan")
        ),
        gross_exposure_limit_breach_count=int(
            leverage_status.eq(LeverageStatus.BREACH).sum()
        ),
        leverage_evaluated_row_count=leverage_evaluated,
        leverage_within_limit_row_count=int(
            leverage_status.eq(LeverageStatus.WITHIN_LIMIT).sum()
        ),
        leverage_unevaluable_row_count=int(
            leverage_status.eq(LeverageStatus.UNEVALUABLE).sum()
        ),
        shared_symbols=shared_symbols,
        largest_symbol_unnetted_sleeve_gross_fraction=(
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
            final_sleeve_equity=(
                float(accounting.sleeve_equity[source.pair_id].iloc[-1])
                if len(accounting.sleeve_equity)
                else float("nan")
            ),
            sleeve_insolvent=(
                bool(accounting.insolvency_state[source.pair_id].iloc[-1])
                if len(accounting.insolvency_state)
                else False
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
            "Shared symbols are reported with both signed/consolidated values and "
            "unnetted pair-sleeve gross exposure. Portfolio leverage uses the "
            "unnetted measure and does not imply broker margin or transaction-cost "
            "netting."
        )
    if exposure_unavailable_count:
        warnings.append(
            "One or more positive-weight sleeves lack an explicit source capital "
            "basis or pair exposure schedule; return aggregation remains available "
            "but aggregate exposure is unavailable."
        )
    if any(source.capital_policy == "equal_capital_reset" for source in sources):
        warnings.append(
            "At least one source is an equal-capital-reset synthetic research "
            "return index. Portfolio sleeve compounding does not reconstruct a "
            "historically continuous source dollar-capital path."
        )
    if insolvent_pair_ids:
        warnings.append(
            "One or more pair sleeves exhausted their allocated equity. They "
            "remain at zero and are never recapitalized."
        )
    contains_synthetic = any(
        source.source_return_path_policy
        == SourceReturnPathPolicy.SYNTHETIC_EQUAL_CAPITAL_RESET.value
        for source in sources
    )
    return PortfolioResult(
        pair_ids=tuple(source.pair_id for source in sources),
        allocation_policy=policy,
        pair_allocations=allocations,
        pair_positions=pair_positions,
        portfolio_schedule=schedule,
        portfolio_returns=portfolio_returns,
        portfolio_equity=portfolio_equity,
        portfolio_pnl=accounting.portfolio_pnl,
        pair_sleeve_returns=accounting.sleeve_returns,
        pair_prior_sleeve_equity=accounting.prior_sleeve_equity,
        pair_sleeve_equity=accounting.sleeve_equity,
        pair_pnl_contributions=accounting.sleeve_pnl,
        pair_return_contributions=accounting.return_contributions,
        pair_current_equity_weights=accounting.current_equity_weights,
        pair_insolvency_state=accounting.insolvency_state,
        catastrophic_pair_rows=accounting.catastrophic_pair_rows,
        pair_exposures=pair_exposures,
        aggregate_exposures=aggregate_exposures,
        symbol_exposures=symbol_exposures,
        leverage_status=leverage_status,
        unavailable_rows=accounting.unavailable_rows,
        metrics=metrics,
        availability=availability,
        initial_capital=capital,
        cash_weight=cash_weight,
        cash_capital=capital * cash_weight,
        all_pair_universes_point_in_time_validated=all(
            source.point_in_time_universe_validated for source in sources
        ),
        pair_capital_policies=capital_policies,
        source_path_provenance=tuple(
            (
                source.pair_id,
                source.source_capital_provenance,
                source.source_return_path_policy,
            )
            for source in sources
        ),
        contains_synthetic_reset_sources=contains_synthetic,
        self_financing_interpretation=(
            "synthetic_static_sleeve_composition_with_reset_based_sources"
            if contains_synthetic
            else (
                "static_unrebalanced_self_financing_pair_sleeves"
                if all(
                    source.source_return_path_policy
                    == SourceReturnPathPolicy.CONTINUOUS_BACKTEST.value
                    for source in sources
                )
                else "static_unrebalanced_composition_with_caller_supplied_source_paths"
            )
        ),
        pair_indices_differ=False,
        provenance_warnings=provenance_warnings,
        warnings=tuple(warnings),
    )
