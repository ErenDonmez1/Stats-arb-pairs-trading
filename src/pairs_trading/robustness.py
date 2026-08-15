"""Deterministic diagnostic sensitivity analysis for walk-forward research.

Milestone 8B evaluates a caller-defined parameter grid without choosing a
winner or feeding out-of-sample performance back into parameter selection.
Calendar-time returns from the hardened walk-forward engine are the primary
performance basis; conditional returns are retained as a secondary view.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from itertools import product
from math import prod
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .analytics import calculate_core_metrics, calculate_drawdown_metrics
from .stats import ADF_MIN_OBSERVATIONS
from .walkforward import (
    WalkForwardFold,
    WalkForwardAnalyticsStatus,
    WalkForwardResult,
    WalkForwardReturnReport,
    generate_walk_forward_folds,
    run_walk_forward_analysis,
)


__all__ = [
    "ParameterScenario",
    "ScenarioStatus",
    "MetricAvailabilityStatus",
    "PerformanceMetrics",
    "MetricDistribution",
    "MetricDistributions",
    "MetricFractionSummary",
    "ParameterRanges",
    "ParameterAxisMetadata",
    "LocalNeighbor",
    "ScenarioResult",
    "SensitivitySummary",
    "RobustnessResult",
    "generate_parameter_scenarios",
    "run_parameter_scenario",
    "run_sensitivity_analysis",
    "summarize_sensitivity",
    "sensitivity_table",
]


PURPOSE = "diagnostic_sensitivity_not_parameter_optimization"
NO_OPTIMIZATION_WARNING = (
    "Scenario comparisons are descriptive diagnostics. They must not be used "
    "to retrospectively promote the highest OOS Sharpe, return, or any other "
    "scenario to a production parameter choice."
)
DISTRIBUTION_POLICY = "equal_weight_per_tested_grid_point"
GRID_DENSITY_WARNING = (
    "Medians and quartiles describe equally weighted tested grid points; they "
    "are not confidence intervals, and denser sampling of one region changes "
    "the reported distribution."
)
TRANSACTION_COST_BASIS = "equal_capital_reset_fold_dollar_total"
HEADLINE_METRIC_BASIS = "structural_common_scheduled_horizon"

_PARAMETER_NAMES = (
    "entry_z",
    "exit_z",
    "stop_z",
    "zscore_lookback",
    "formation_window",
    "trading_window",
    "screening_min_observations",
    "commission_bps",
    "slippage_bps",
    "financing_rate",
    "borrow_rate_y",
    "borrow_rate_x",
)
_INTEGER_PARAMETERS = {
    "zscore_lookback",
    "formation_window",
    "trading_window",
    "screening_min_observations",
}
_SCENARIO_CONTROLLED_ARGUMENTS = set(_PARAMETER_NAMES) | {
    "step_size",
    "minimum_observations",
    "groups",
}
_ADDITIONAL_MATERIAL_DIMENSIONS = (
    "fdr_threshold",
    "half_life_threshold",
    "hurst_threshold",
    "execution_lag",
    "maximum_holding_period",
    "cooldown_period",
    "target_notional_or_leverage",
    "fixed_commission",
    "forced_liquidation_policy",
    "hedge_estimation_policy",
)
_MATERIAL_DIMENSIONS = _PARAMETER_NAMES + _ADDITIONAL_MATERIAL_DIMENSIONS


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer.")
    return int(value)


def _positive_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class ParameterScenario:
    """One immutable, predefined sensitivity configuration."""

    scenario_id: str
    entry_z: float
    exit_z: float
    stop_z: float
    zscore_lookback: int
    formation_window: int
    trading_window: int
    screening_min_observations: int
    commission_bps: float
    slippage_bps: float
    financing_rate: float
    borrow_rate_y: float = 0.0
    borrow_rate_x: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string.")
        for name in _PARAMETER_NAMES:
            value = getattr(self, name)
            normalized = (
                _integer(value, name)
                if name in _INTEGER_PARAMETERS
                else _finite_real(value, name)
            )
            object.__setattr__(self, name, normalized)

    def parameter_tuple(self) -> tuple[Any, ...]:
        """Return values in the canonical grid and neighborhood order."""
        return tuple(getattr(self, name) for name in _PARAMETER_NAMES)


class ScenarioStatus(str, Enum):
    """Explicit execution/analytics status for one sensitivity scenario."""

    COMPLETED = "COMPLETED"
    NO_VALID_FOLDS = "NO_VALID_FOLDS"
    ANALYTICS_UNAVAILABLE = "ANALYTICS_UNAVAILABLE"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    FAILED = "FAILED"


class MetricAvailabilityStatus(str, Enum):
    """Why a native or common-horizon performance metric is (un)available."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE_PARTIAL_CALENDAR = "UNAVAILABLE_PARTIAL_CALENDAR"
    UNAVAILABLE_NO_OBSERVATIONS = "UNAVAILABLE_NO_OBSERVATIONS"
    UNAVAILABLE_INSUFFICIENT_OBSERVATIONS = (
        "UNAVAILABLE_INSUFFICIENT_OBSERVATIONS"
    )
    UNAVAILABLE_INVALID_RETURNS = "UNAVAILABLE_INVALID_RETURNS"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class PerformanceMetrics:
    """Return and risk metrics for one explicitly identified return series."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    calmar_ratio: float
    observations: int


@dataclass(frozen=True)
class MetricDistribution:
    """Median and interquartile range over finite scenario values."""

    median: float
    lower_quartile: float
    upper_quartile: float
    observations: int


@dataclass(frozen=True)
class MetricDistributions:
    """Robust cross-scenario distributions for all reported metrics."""

    total_return: MetricDistribution
    annualized_return: MetricDistribution
    annualized_volatility: MetricDistribution
    sharpe_ratio: MetricDistribution
    sortino_ratio: MetricDistribution
    maximum_drawdown: MetricDistribution
    calmar_ratio: MetricDistribution


@dataclass(frozen=True)
class MetricFractionSummary:
    """Explicit numerator and denominator accounting for a headline fraction."""

    criterion: str
    eligible_scenarios: int
    defined_scenarios: int
    undefined_scenarios: int
    positive_scenarios: int
    invalid_scenarios: int
    failed_scenarios: int
    positive_fraction_defined: float
    positive_fraction_all_eligible: float


@dataclass(frozen=True)
class ParameterRanges:
    """Deterministic parameter values represented by scenario records."""

    entry_z: tuple[float, ...]
    exit_z: tuple[float, ...]
    stop_z: tuple[float, ...]
    zscore_lookback: tuple[int, ...]
    formation_window: tuple[int, ...]
    trading_window: tuple[int, ...]
    screening_min_observations: tuple[int, ...]
    commission_bps: tuple[float, ...]
    slippage_bps: tuple[float, ...]
    financing_rate: tuple[float, ...]
    borrow_rate_y: tuple[float, ...]
    borrow_rate_x: tuple[float, ...]


@dataclass(frozen=True)
class ParameterAxisMetadata:
    """Deterministic grid-density metadata for one scenario parameter."""

    parameter: str
    tested_values: tuple[int | float, ...]
    count: int
    numeric_spacing: tuple[float, ...]


@dataclass(frozen=True)
class LocalNeighbor:
    """An immediately adjacent one-axis baseline variation."""

    scenario_id: str
    changed_parameter: str
    baseline_value: int | float
    neighbor_value: int | float
    absolute_distance: float
    relative_distance: float


@dataclass(frozen=True)
class ScenarioResult:
    """Independently owned result for one predefined scenario.

    The wrapper is frozen. Stored pandas Series and the nested walk-forward
    result are defensive copies, but pandas objects remain caller-mutable.
    """

    scenario: ParameterScenario
    status: ScenarioStatus
    error: str | None
    walk_forward_result: WalkForwardResult | None
    calendar_metrics: PerformanceMetrics | None
    conditional_metrics: PerformanceMetrics | None
    common_horizon_metrics: PerformanceMetrics | None
    common_horizon_error: str | None
    calendar_oos_returns: pd.Series
    conditional_oos_returns: pd.Series
    common_horizon_returns: pd.Series
    evaluated_start_position: int | None
    evaluated_end_position: int | None
    scheduled_oos_observations: int
    available_oos_observations: int
    selected_oos_observations: int
    unavailable_oos_observations: int
    selection_coverage: float
    trade_count: int
    total_transaction_cost: float
    periods_per_year: int
    risk_free_rate: float
    available_observations_metrics: PerformanceMetrics | None = None
    calendar_metrics_status: MetricAvailabilityStatus = (
        MetricAvailabilityStatus.NOT_APPLICABLE
    )
    calendar_metrics_error: str | None = None
    common_horizon_structurally_available: bool = False
    common_horizon_fully_observed: bool = False
    common_horizon_analytics_status: MetricAvailabilityStatus = (
        MetricAvailabilityStatus.NOT_APPLICABLE
    )
    equal_capital_reset_fold_transaction_cost_total: float = 0.0
    transaction_cost_known_components: int = 0
    transaction_cost_unknown_components: int = 0
    transaction_cost_basis: str = TRANSACTION_COST_BASIS

    def __post_init__(self) -> None:
        for field_name, series_name in (
            ("calendar_oos_returns", "calendar_oos_return"),
            ("conditional_oos_returns", "conditional_oos_return"),
            ("common_horizon_returns", "common_horizon_return"),
        ):
            series = getattr(self, field_name)
            if not isinstance(series, pd.Series):
                raise TypeError(f"{field_name} must be a pandas Series.")
            owned = series.copy(deep=True)
            owned.name = series_name
            object.__setattr__(self, field_name, owned)
        if self.walk_forward_result is not None:
            object.__setattr__(
                self,
                "walk_forward_result",
                replace(self.walk_forward_result),
            )


@dataclass(frozen=True)
class SensitivitySummary:
    """Robust descriptive distributions without parameter selection."""

    scenario_count: int
    completed_scenarios: int
    unavailable_scenarios: int
    invalid_scenarios: int
    failed_scenarios: int
    baseline_scenario_id: str
    baseline_metrics: PerformanceMetrics | None
    metric_distributions: MetricDistributions
    fraction_positive_total_return: float
    fraction_positive_annualized_return: float
    fraction_positive_sharpe: float
    drawdown_severity_threshold: float
    fraction_drawdown_no_worse_than_threshold: float
    parameter_ranges: ParameterRanges
    baseline_neighbor_count: int
    neighbor_metric_distributions: MetricDistributions
    fraction_neighbors_same_total_return_sign: float
    fraction_neighbors_positive_sharpe: float
    headline_metric_basis: str = HEADLINE_METRIC_BASIS
    headline_metrics_available: bool = False
    baseline_native_metrics: PerformanceMetrics | None = None
    native_metric_distributions: MetricDistributions = field(
        default_factory=lambda: _distributions(())
    )
    total_return_fraction: MetricFractionSummary = field(
        default_factory=lambda: _empty_fraction("total_return > 0")
    )
    annualized_return_fraction: MetricFractionSummary = field(
        default_factory=lambda: _empty_fraction("annualized_return > 0")
    )
    sharpe_fraction: MetricFractionSummary = field(
        default_factory=lambda: _empty_fraction("sharpe_ratio > 0")
    )
    drawdown_fraction: MetricFractionSummary = field(
        default_factory=lambda: _empty_fraction(
            "maximum_drawdown >= drawdown_severity_threshold"
        )
    )
    one_parameter_variant_count: int = 0
    variant_metric_distributions: MetricDistributions = field(
        default_factory=lambda: _distributions(())
    )
    variant_sign_agreement: float = float("nan")
    variant_positive_sharpe_fraction: float = float("nan")
    local_neighbor_count: int = 0
    local_neighbors: tuple[LocalNeighbor, ...] = ()


@dataclass(frozen=True)
class RobustnessResult:
    """Canonical immutable sensitivity result with explicit research policy."""

    scenarios: tuple[ScenarioResult, ...]
    baseline_scenario_id: str
    baseline_scenario: ParameterScenario
    baseline_result: ScenarioResult
    summary: SensitivitySummary
    common_horizon_index: pd.Index
    common_horizon_observations: int
    common_horizon_scenario_count: int
    common_horizon_excluded_scenarios: int
    common_horizon_available: bool
    common_horizon_error: str | None
    purpose: str
    warning: str
    common_horizon_structurally_available: bool = False
    common_horizon_fully_observed: bool = False
    common_horizon_analytics_available: bool = False
    common_horizon_analytics_status: MetricAvailabilityStatus = (
        MetricAvailabilityStatus.NOT_APPLICABLE
    )
    common_horizon_analytics_error: str | None = None
    distribution_policy: str = DISTRIBUTION_POLICY
    grid_density_warning: str = GRID_DENSITY_WARNING
    axis_metadata: tuple[ParameterAxisMetadata, ...] = ()
    tested_dimensions: tuple[str, ...] = ()
    untested_material_dimensions: tuple[str, ...] = ()
    universe_provenance: tuple[str, ...] = ()
    cleaning_provenance: tuple[str, ...] = ()
    point_in_time_universe_validated: bool = False
    provenance_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        owned = tuple(replace(result) for result in self.scenarios)
        object.__setattr__(self, "scenarios", owned)
        baseline = next(
            result
            for result in owned
            if result.scenario.scenario_id == self.baseline_scenario_id
        )
        object.__setattr__(self, "baseline_result", baseline)
        object.__setattr__(self, "baseline_scenario", baseline.scenario)
        object.__setattr__(self, "common_horizon_index", self.common_horizon_index.copy())
        object.__setattr__(self, "axis_metadata", tuple(self.axis_metadata))
        object.__setattr__(self, "tested_dimensions", tuple(self.tested_dimensions))
        object.__setattr__(
            self,
            "untested_material_dimensions",
            tuple(self.untested_material_dimensions),
        )
        object.__setattr__(self, "universe_provenance", tuple(self.universe_provenance))
        object.__setattr__(self, "cleaning_provenance", tuple(self.cleaning_provenance))
        object.__setattr__(self, "provenance_warnings", tuple(self.provenance_warnings))


def _configuration_error(scenario: ParameterScenario) -> str | None:
    if scenario.entry_z <= 0.0:
        return "entry_z must be strictly positive."
    if scenario.exit_z < 0.0:
        return "exit_z must be non-negative."
    if scenario.entry_z <= scenario.exit_z:
        return "entry_z must be greater than exit_z."
    if scenario.stop_z <= scenario.entry_z:
        return "stop_z must be greater than entry_z."
    if scenario.zscore_lookback < 2:
        return "zscore_lookback must be at least 2."
    if scenario.formation_window <= scenario.zscore_lookback:
        return "formation_window must exceed zscore_lookback."
    if scenario.formation_window < scenario.screening_min_observations:
        return "formation_window must be at least screening_min_observations."
    if scenario.screening_min_observations < ADF_MIN_OBSERVATIONS:
        return (
            "screening_min_observations must be at least "
            f"{ADF_MIN_OBSERVATIONS}."
        )
    if scenario.trading_window <= 0:
        return "trading_window must be positive."
    if any(
        getattr(scenario, name) < 0.0
        for name in (
            "commission_bps",
            "slippage_bps",
            "financing_rate",
            "borrow_rate_y",
            "borrow_rate_x",
        )
    ):
        return "Execution and financing costs must be non-negative."
    return None


def _scenario_identifier(values: Mapping[str, Any]) -> str:
    payload = "|".join(f"{name}={values[name]!r}" for name in _PARAMETER_NAMES)
    return f"scenario-{sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _materialize_grid_values(
    values: Iterable[Any] | None,
    baseline_value: Any,
    name: str,
) -> tuple[Any, ...]:
    if values is None:
        materialized = (baseline_value,)
    else:
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise TypeError(f"{name}_values must be an iterable.")
        raw = tuple(values)
        if not raw:
            raise ValueError(f"{name}_values must not be empty.")
        materialized = tuple(
            _integer(value, name)
            if name in _INTEGER_PARAMETERS
            else _finite_real(value, name)
            for value in raw
        )
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{name}_values contains duplicate values.")
    ordered = tuple(sorted(materialized))
    if baseline_value not in ordered:
        raise ValueError(
            f"{name}_values must include the predefined baseline value "
            f"{baseline_value!r}."
        )
    return ordered


def generate_parameter_scenarios(
    baseline: ParameterScenario,
    *,
    entry_z_values: Iterable[Real] | None = None,
    exit_z_values: Iterable[Real] | None = None,
    stop_z_values: Iterable[Real] | None = None,
    zscore_lookback_values: Iterable[int] | None = None,
    formation_window_values: Iterable[int] | None = None,
    trading_window_values: Iterable[int] | None = None,
    screening_min_observations_values: Iterable[int] | None = None,
    commission_bps_values: Iterable[Real] | None = None,
    slippage_bps_values: Iterable[Real] | None = None,
    financing_rate_values: Iterable[Real] | None = None,
    borrow_rate_y_values: Iterable[Real] | None = None,
    borrow_rate_x_values: Iterable[Real] | None = None,
    max_scenarios: int = 256,
) -> tuple[ParameterScenario, ...]:
    """Build a deterministic Cartesian grid around an explicit baseline.

    Invalid cross-parameter combinations remain in the returned grid so the
    execution layer can mark them ``INVALID_CONFIGURATION`` transparently.
    The baseline itself must be valid and every supplied axis must contain its
    baseline value.
    """
    if not isinstance(baseline, ParameterScenario):
        raise TypeError("baseline must be a ParameterScenario.")
    baseline_error = _configuration_error(baseline)
    if baseline_error is not None:
        raise ValueError(f"The baseline scenario is invalid: {baseline_error}")
    limit = _positive_integer(max_scenarios, "max_scenarios")
    supplied = {
        "entry_z": entry_z_values,
        "exit_z": exit_z_values,
        "stop_z": stop_z_values,
        "zscore_lookback": zscore_lookback_values,
        "formation_window": formation_window_values,
        "trading_window": trading_window_values,
        "screening_min_observations": screening_min_observations_values,
        "commission_bps": commission_bps_values,
        "slippage_bps": slippage_bps_values,
        "financing_rate": financing_rate_values,
        "borrow_rate_y": borrow_rate_y_values,
        "borrow_rate_x": borrow_rate_x_values,
    }
    axes = {
        name: _materialize_grid_values(
            supplied[name],
            getattr(baseline, name),
            name,
        )
        for name in _PARAMETER_NAMES
    }
    scenario_count = prod(len(axes[name]) for name in _PARAMETER_NAMES)
    if scenario_count > limit:
        raise ValueError(
            f"The Cartesian grid contains {scenario_count} scenarios, exceeding "
            f"max_scenarios={limit}."
        )

    scenarios: list[ParameterScenario] = []
    for combination in product(*(axes[name] for name in _PARAMETER_NAMES)):
        values = dict(zip(_PARAMETER_NAMES, combination))
        scenario_id = (
            baseline.scenario_id
            if tuple(combination) == baseline.parameter_tuple()
            else _scenario_identifier(values)
        )
        scenarios.append(ParameterScenario(scenario_id=scenario_id, **values))
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise RuntimeError("Stable scenario ID collision detected.")
    return tuple(sorted(scenarios, key=lambda scenario: scenario.scenario_id))


def _raw_total_return(returns: pd.Series, *, require_complete: bool = True) -> float:
    """Compound returns without silently compressing a scheduled calendar."""
    if returns.empty or (require_complete and bool(returns.isna().any())):
        return float("nan")
    valid = returns.dropna()
    if valid.empty:
        return float("nan")
    values = valid.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return float("nan")
    return float(np.prod(1.0 + values) - 1.0)


def _metrics_from_report(
    report: WalkForwardReturnReport,
) -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return=report.core.total_return,
        annualized_return=report.core.annualized_return,
        annualized_volatility=report.core.annualized_volatility,
        sharpe_ratio=report.core.sharpe_ratio,
        sortino_ratio=report.core.sortino_ratio,
        maximum_drawdown=report.drawdown.maximum_drawdown,
        calmar_ratio=report.drawdown.calmar_ratio,
        observations=report.report_observations,
    )


def _metrics_from_returns(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
) -> tuple[PerformanceMetrics | None, MetricAvailabilityStatus, str | None]:
    """Calculate metrics only for a complete, compoundable return horizon."""
    if returns.empty:
        return (
            None,
            MetricAvailabilityStatus.UNAVAILABLE_NO_OBSERVATIONS,
            "No scheduled return observations are available.",
        )
    if bool(returns.isna().any()):
        return (
            None,
            MetricAvailabilityStatus.UNAVAILABLE_PARTIAL_CALENDAR,
            "Scheduled return observations contain unavailable values.",
        )
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return (
            None,
            MetricAvailabilityStatus.UNAVAILABLE_INVALID_RETURNS,
            "Scheduled returns must be finite.",
        )
    if len(returns) < 2:
        return (
            None,
            MetricAvailabilityStatus.UNAVAILABLE_INSUFFICIENT_OBSERVATIONS,
            "At least two scheduled return observations are required.",
        )
    if bool(returns.le(-1.0).any()):
        return (
            None,
            MetricAvailabilityStatus.UNAVAILABLE_INVALID_RETURNS,
            "Returns at or below -100% cannot be geometrically compounded.",
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compounded = np.cumprod(1.0 + values)
        annualized = np.power(compounded[-1], periods_per_year / len(values)) - 1.0
    if (
        not np.isfinite(compounded).all()
        or bool((compounded <= 0.0).any())
        or not np.isfinite(annualized)
    ):
        return (
            None,
            MetricAvailabilityStatus.UNAVAILABLE_INVALID_RETURNS,
            "Geometric return compounding is not finite and strictly positive.",
        )
    # Inputs now satisfy all documented analytics preconditions.  Any remaining
    # exception is unexpected and must remain visible to the caller.
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
    return (
        PerformanceMetrics(
            total_return=core.total_return,
            annualized_return=core.annualized_return,
            annualized_volatility=core.annualized_volatility,
            sharpe_ratio=core.sharpe_ratio,
            sortino_ratio=core.sortino_ratio,
            maximum_drawdown=drawdown.maximum_drawdown,
            calmar_ratio=drawdown.calmar_ratio,
            observations=core.observations,
        ),
        MetricAvailabilityStatus.AVAILABLE,
        None,
    )


def _available_observation_diagnostic(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
) -> PerformanceMetrics | None:
    """Secondary observed-only diagnostic; never used in headline summaries."""
    observed = returns.dropna().copy(deep=True)
    metrics, _, _ = _metrics_from_returns(
        observed,
        periods_per_year,
        risk_free_rate,
    )
    return metrics


def _empty_scenario_result(
    scenario: ParameterScenario,
    status: ScenarioStatus,
    error: str,
    *,
    periods_per_year: int,
    risk_free_rate: float,
) -> ScenarioResult:
    return ScenarioResult(
        scenario=scenario,
        status=status,
        error=error,
        walk_forward_result=None,
        calendar_metrics=None,
        conditional_metrics=None,
        common_horizon_metrics=None,
        common_horizon_error="Scenario has no walk-forward return series.",
        calendar_oos_returns=pd.Series(dtype=float),
        conditional_oos_returns=pd.Series(dtype=float),
        common_horizon_returns=pd.Series(dtype=float),
        evaluated_start_position=None,
        evaluated_end_position=None,
        scheduled_oos_observations=0,
        available_oos_observations=0,
        selected_oos_observations=0,
        unavailable_oos_observations=0,
        selection_coverage=float("nan"),
        trade_count=0,
        total_transaction_cost=0.0,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        calendar_metrics_status=MetricAvailabilityStatus.NOT_APPLICABLE,
        calendar_metrics_error=error,
        common_horizon_analytics_status=MetricAvailabilityStatus.NOT_APPLICABLE,
        equal_capital_reset_fold_transaction_cost_total=0.0,
    )


def _normalized_shared_arguments(
    values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("walk_forward_kwargs must be a mapping or None.")
    overlap = sorted(set(values).intersection(_SCENARIO_CONTROLLED_ARGUMENTS))
    if overlap:
        raise ValueError(
            "walk_forward_kwargs cannot override scenario-controlled arguments: "
            f"{overlap}."
        )
    return deepcopy(dict(values))


def _normalized_groups(
    groups: (
        Mapping[str, Iterable[str]]
        | Callable[[WalkForwardFold], Mapping[str, Iterable[str]] | None]
        | None
    ),
) -> Any:
    if groups is None or callable(groups):
        return groups
    if not isinstance(groups, Mapping):
        raise TypeError("groups must be a mapping, per-fold callable, or None.")
    normalized: dict[str, tuple[str, ...]] = {}
    for group, symbols in groups.items():
        if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Iterable):
            raise TypeError(f"Group {group!r} symbols must be an iterable.")
        normalized[group] = tuple(symbols)
    return normalized


def _require_unique_parameter_tuples(
    scenarios: Iterable[ParameterScenario],
) -> tuple[ParameterScenario, ...]:
    """Reject duplicate configurations even when their IDs differ."""
    materialized = tuple(scenarios)
    seen: dict[tuple[Any, ...], str] = {}
    for scenario in materialized:
        values = scenario.parameter_tuple()
        if values in seen:
            raise ValueError(
                "Scenario parameter tuples must be unique; "
                f"{seen[values]!r} and {scenario.scenario_id!r} are duplicates."
            )
        seen[values] = scenario.scenario_id
    return materialized


def _fold_snapshot_key(fold: WalkForwardFold) -> tuple[int, int, str, str]:
    """Identify the point-in-time universe by its formation boundary."""
    return (
        int(fold.formation_start_position),
        int(fold.formation_end_position),
        repr(fold.formation_start_label),
        repr(fold.formation_end_label),
    )


def _normalize_group_snapshot(
    groups: Mapping[str, Iterable[str]] | None,
) -> Mapping[str, tuple[str, ...]] | None:
    """Own and deterministically freeze one callable-provider response."""
    if groups is None:
        return None
    if not isinstance(groups, Mapping):
        raise TypeError("A group provider must return a mapping or None.")
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("Group names must be non-empty strings.")
    normalized: dict[str, tuple[str, ...]] = {}
    for group in sorted(groups):
        symbols = groups[group]
        if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Iterable):
            raise ValueError(f"Group {group!r} symbols must be an iterable.")
        materialized = tuple(symbols)
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in materialized):
            raise ValueError(f"Group {group!r} contains an invalid symbol.")
        normalized[group] = tuple(sorted(materialized))
    return MappingProxyType(normalized)


def _snapshotted_group_provider(
    prices: pd.DataFrame,
    scenarios: tuple[ParameterScenario, ...],
    provider: Callable[
        [WalkForwardFold], Mapping[str, Iterable[str]] | None
    ],
) -> Callable[[WalkForwardFold], Mapping[str, tuple[str, ...]] | None]:
    """Materialize fold-universe snapshots before any scenario execution."""
    representative_folds: dict[tuple[int, int, str, str], WalkForwardFold] = {}
    for scenario in sorted(
        scenarios,
        key=lambda item: (item.parameter_tuple(), item.scenario_id),
    ):
        if _configuration_error(scenario) is not None:
            continue
        folds = generate_walk_forward_folds(
            prices.index,
            scenario.formation_window,
            scenario.trading_window,
            step_size=scenario.trading_window,
            minimum_observations=scenario.screening_min_observations,
        )
        for fold in folds:
            representative_folds.setdefault(_fold_snapshot_key(fold), fold)

    snapshots = {
        key: _normalize_group_snapshot(provider(representative_folds[key]))
        for key in sorted(representative_folds)
    }

    def snapshot_for(
        fold: WalkForwardFold,
    ) -> Mapping[str, tuple[str, ...]] | None:
        key = _fold_snapshot_key(fold)
        if key not in snapshots:
            raise RuntimeError(
                "Walk-forward requested a group snapshot outside the "
                "pre-materialized deterministic fold set."
            )
        return snapshots[key]

    return snapshot_for


def run_parameter_scenario(
    prices: pd.DataFrame,
    scenario: ParameterScenario,
    *,
    groups: (
        Mapping[str, Iterable[str]]
        | Callable[[WalkForwardFold], Mapping[str, Iterable[str]] | None]
        | None
    ) = None,
    walk_forward_kwargs: Mapping[str, Any] | None = None,
) -> ScenarioResult:
    """Execute one predefined scenario independently through Milestone 8A."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if not isinstance(scenario, ParameterScenario):
        raise TypeError("scenario must be a ParameterScenario.")
    shared = _normalized_shared_arguments(walk_forward_kwargs)
    periods = _positive_integer(shared.get("periods_per_year", 252), "periods_per_year")
    risk_free = _finite_real(shared.get("risk_free_rate", 0.0), "risk_free_rate")
    configuration_error = _configuration_error(scenario)
    if configuration_error is not None:
        return _empty_scenario_result(
            scenario,
            ScenarioStatus.INVALID_CONFIGURATION,
            configuration_error,
            periods_per_year=periods,
            risk_free_rate=risk_free,
        )

    owned_prices = prices.copy(deep=True)
    owned_prices.attrs = deepcopy(prices.attrs)
    # Unexpected execution, accounting, causality, or programming errors are
    # deliberately allowed to propagate.  Only configuration combinations
    # rejected above become nonfatal scenario records.
    walk_forward = run_walk_forward_analysis(
        owned_prices,
        scenario.formation_window,
        scenario.trading_window,
        step_size=scenario.trading_window,
        minimum_observations=scenario.screening_min_observations,
        groups=groups,
        screening_min_observations=scenario.screening_min_observations,
        zscore_lookback=scenario.zscore_lookback,
        entry_z=scenario.entry_z,
        exit_z=scenario.exit_z,
        stop_z=scenario.stop_z,
        commission_bps=scenario.commission_bps,
        slippage_bps=scenario.slippage_bps,
        financing_rate=scenario.financing_rate,
        borrow_rate_y=scenario.borrow_rate_y,
        borrow_rate_x=scenario.borrow_rate_x,
        **shared,
    )

    calendar = walk_forward.calendar_oos_returns.copy(deep=True)
    conditional = walk_forward.conditional_oos_returns.copy(deep=True)
    available = int(calendar.notna().sum())
    calendar_is_complete = available == len(calendar) and len(calendar) > 0
    if not calendar_is_complete:
        calendar_metrics = None
        if available == 0:
            calendar_metrics_status = (
                MetricAvailabilityStatus.UNAVAILABLE_NO_OBSERVATIONS
            )
            calendar_metrics_error = (
                "No available calendar OOS observations were produced."
            )
        else:
            calendar_metrics_status = (
                MetricAvailabilityStatus.UNAVAILABLE_PARTIAL_CALENDAR
            )
            calendar_metrics_error = (
                "Primary calendar analytics require every scheduled OOS "
                "observation; at least one row is unavailable."
            )
    elif walk_forward.calendar_performance_report is not None:
        calendar_metrics = _metrics_from_report(
            walk_forward.calendar_performance_report
        )
        calendar_metrics_status = MetricAvailabilityStatus.AVAILABLE
        calendar_metrics_error = None
    else:
        calendar_metrics = None
        if len(calendar) < 2:
            calendar_metrics_status = (
                MetricAvailabilityStatus.UNAVAILABLE_INSUFFICIENT_OBSERVATIONS
            )
        elif bool(calendar.le(-1.0).any()):
            calendar_metrics_status = (
                MetricAvailabilityStatus.UNAVAILABLE_INVALID_RETURNS
            )
        elif (
            walk_forward.calendar_analytics_status
            is WalkForwardAnalyticsStatus.FAILED
        ):
            calendar_metrics_status = MetricAvailabilityStatus.FAILED
        else:
            calendar_metrics_status = (
                MetricAvailabilityStatus.UNAVAILABLE_INVALID_RETURNS
            )
        calendar_metrics_error = (
            walk_forward.calendar_analytics_error
            or "Calendar performance analytics are unavailable."
        )

    available_metrics = _available_observation_diagnostic(
        calendar,
        periods,
        risk_free,
    )
    if walk_forward.conditional_performance_report is not None:
        conditional_metrics = _metrics_from_report(
            walk_forward.conditional_performance_report
        )
    else:
        conditional_metrics = _available_observation_diagnostic(
            conditional,
            periods,
            risk_free,
        )
    if available == 0:
        status = ScenarioStatus.NO_VALID_FOLDS
        error = "No available calendar OOS observations were produced."
    elif calendar_metrics is None:
        status = ScenarioStatus.ANALYTICS_UNAVAILABLE
        error = calendar_metrics_error
    else:
        status = ScenarioStatus.COMPLETED
        error = None

    trade_count = sum(fold.trade_count for fold in walk_forward.folds)
    known_cost_components = 0
    unknown_cost_components = 0
    known_transaction_cost_total = 0.0
    for fold in walk_forward.folds:
        if fold.backtest is None:
            continue
        accounting = fold.backtest.accounting
        if "transaction_cost" not in accounting:
            unknown_cost_components += max(len(accounting), 1)
            continue
        costs = pd.to_numeric(accounting["transaction_cost"], errors="coerce")
        finite = pd.Series(
            np.isfinite(costs.to_numpy(dtype=float)),
            index=costs.index,
            dtype=bool,
        )
        known = costs.notna() & finite
        if bool(costs.loc[known].lt(0.0).any()):
            raise RuntimeError(
                "Walk-forward accounting produced a negative transaction cost."
            )
        known_cost_components += int(known.sum())
        unknown_cost_components += int((~known).sum())
        known_transaction_cost_total += float(costs.loc[known].sum())
    total_transaction_cost = (
        float("nan")
        if unknown_cost_components
        else known_transaction_cost_total
    )
    return ScenarioResult(
        scenario=scenario,
        status=status,
        error=error,
        walk_forward_result=walk_forward,
        calendar_metrics=calendar_metrics,
        conditional_metrics=conditional_metrics,
        common_horizon_metrics=None,
        common_horizon_error="Common horizon is assigned by run_sensitivity_analysis.",
        calendar_oos_returns=calendar,
        conditional_oos_returns=conditional,
        common_horizon_returns=pd.Series(dtype=float),
        evaluated_start_position=walk_forward.evaluated_start_position,
        evaluated_end_position=walk_forward.evaluated_end_position,
        scheduled_oos_observations=walk_forward.scheduled_oos_observations,
        available_oos_observations=available,
        selected_oos_observations=walk_forward.selected_oos_observations,
        unavailable_oos_observations=walk_forward.unavailable_oos_observations,
        selection_coverage=walk_forward.selection_coverage,
        trade_count=trade_count,
        total_transaction_cost=total_transaction_cost,
        periods_per_year=periods,
        risk_free_rate=risk_free,
        available_observations_metrics=available_metrics,
        calendar_metrics_status=calendar_metrics_status,
        calendar_metrics_error=calendar_metrics_error,
        common_horizon_analytics_status=MetricAvailabilityStatus.NOT_APPLICABLE,
        equal_capital_reset_fold_transaction_cost_total=total_transaction_cost,
        transaction_cost_known_components=known_cost_components,
        transaction_cost_unknown_components=unknown_cost_components,
    )


def _common_horizon(
    results: tuple[ScenarioResult, ...],
) -> tuple[
    tuple[ScenarioResult, ...],
    pd.Index,
    int,
    bool,
    bool,
    bool,
    str | None,
]:
    """Apply a structural scheduled-index intersection without dropna selection."""
    comparable = tuple(
        result
        for result in results
        if result.status
        not in {ScenarioStatus.INVALID_CONFIGURATION, ScenarioStatus.FAILED}
        and len(result.calendar_oos_returns) > 0
    )
    empty_index = pd.Index([], dtype=object)
    if not comparable:
        error = "No scenario has a scheduled calendar OOS horizon."
        return (
            tuple(
                replace(
                    result,
                    common_horizon_metrics=None,
                    common_horizon_error=error,
                    common_horizon_returns=pd.Series(dtype=float),
                    common_horizon_structurally_available=False,
                    common_horizon_fully_observed=False,
                    common_horizon_analytics_status=(
                        MetricAvailabilityStatus.NOT_APPLICABLE
                    ),
                )
                for result in results
            ),
            empty_index,
            0,
            False,
            False,
            False,
            error,
        )

    # Membership depends only on scheduled labels, never on realized returns.
    common = comparable[0].calendar_oos_returns.index.copy()
    for result in comparable[1:]:
        common = common[common.isin(result.calendar_oos_returns.index)]
    structurally_available = len(common) > 0

    updated: list[ScenarioResult] = []
    comparable_ids = {result.scenario.scenario_id for result in comparable}
    for result in results:
        if result.scenario.scenario_id not in comparable_ids:
            updated.append(
                replace(
                    result,
                    common_horizon_metrics=None,
                    common_horizon_error="Scenario is unavailable on the common horizon.",
                    common_horizon_returns=pd.Series(dtype=float),
                    common_horizon_structurally_available=False,
                    common_horizon_fully_observed=False,
                    common_horizon_analytics_status=(
                        MetricAvailabilityStatus.NOT_APPLICABLE
                    ),
                )
            )
            continue
        common_returns = result.calendar_oos_returns.reindex(common).copy(deep=True)
        fully_observed = bool(len(common_returns)) and not bool(
            common_returns.isna().any()
        )
        metrics, analytics_status, analytics_error = _metrics_from_returns(
            common_returns,
            result.periods_per_year,
            result.risk_free_rate,
        )
        updated.append(
            replace(
                result,
                common_horizon_metrics=metrics,
                common_horizon_error=analytics_error,
                common_horizon_returns=common_returns,
                common_horizon_structurally_available=structurally_available,
                common_horizon_fully_observed=fully_observed,
                common_horizon_analytics_status=analytics_status,
            )
        )

    comparable_updated = tuple(
        result
        for result in updated
        if result.scenario.scenario_id in comparable_ids
    )
    fully_observed = structurally_available and all(
        result.common_horizon_fully_observed for result in comparable_updated
    )
    analytics_available = (
        len(common) >= 2
        and fully_observed
        and all(
            result.common_horizon_analytics_status
            is MetricAvailabilityStatus.AVAILABLE
            for result in comparable_updated
        )
    )
    if not structurally_available:
        error = "Scenarios have no structurally common scheduled OOS observation."
    elif len(common) < 2:
        error = (
            "The structural common horizon has fewer than two observations; "
            "cross-scenario analytics are unavailable."
        )
    elif not fully_observed:
        error = (
            "At least one scenario is unavailable on the structural common "
            "scheduled horizon."
        )
    elif not analytics_available:
        reasons = sorted(
            {
                result.common_horizon_error
                for result in comparable_updated
                if result.common_horizon_error is not None
            }
        )
        error = (
            "Common-horizon analytics are unavailable: " + "; ".join(reasons)
            if reasons
            else "Common-horizon analytics are unavailable."
        )
    else:
        error = None
    return (
        tuple(updated),
        common,
        len(comparable),
        structurally_available,
        fully_observed,
        analytics_available,
        error,
    )


def _distribution(values: Iterable[float]) -> MetricDistribution:
    array = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=float,
    )
    if not len(array):
        return MetricDistribution(float("nan"), float("nan"), float("nan"), 0)
    return MetricDistribution(
        median=float(np.quantile(array, 0.50)),
        lower_quartile=float(np.quantile(array, 0.25)),
        upper_quartile=float(np.quantile(array, 0.75)),
        observations=len(array),
    )


def _distributions(metrics: Iterable[PerformanceMetrics]) -> MetricDistributions:
    values = tuple(metrics)
    return MetricDistributions(
        total_return=_distribution(metric.total_return for metric in values),
        annualized_return=_distribution(
            metric.annualized_return for metric in values
        ),
        annualized_volatility=_distribution(
            metric.annualized_volatility for metric in values
        ),
        sharpe_ratio=_distribution(metric.sharpe_ratio for metric in values),
        sortino_ratio=_distribution(metric.sortino_ratio for metric in values),
        maximum_drawdown=_distribution(
            metric.maximum_drawdown for metric in values
        ),
        calmar_ratio=_distribution(metric.calmar_ratio for metric in values),
    )


def _defined_fraction(values: Iterable[float], predicate: Callable[[float], bool]) -> float:
    defined = [float(value) for value in values if np.isfinite(float(value))]
    if not defined:
        return float("nan")
    return float(np.mean([predicate(value) for value in defined]))


def _empty_fraction(criterion: str) -> MetricFractionSummary:
    return MetricFractionSummary(
        criterion=criterion,
        eligible_scenarios=0,
        defined_scenarios=0,
        undefined_scenarios=0,
        positive_scenarios=0,
        invalid_scenarios=0,
        failed_scenarios=0,
        positive_fraction_defined=float("nan"),
        positive_fraction_all_eligible=float("nan"),
    )


def _fraction_summary(
    results: tuple[ScenarioResult, ...],
    metrics: tuple[PerformanceMetrics, ...],
    attribute: str,
    predicate: Callable[[float], bool],
    criterion: str,
) -> MetricFractionSummary:
    eligible = sum(
        result.status
        not in {ScenarioStatus.INVALID_CONFIGURATION, ScenarioStatus.FAILED}
        for result in results
    )
    defined_values = tuple(
        float(getattr(metric, attribute))
        for metric in metrics
        if np.isfinite(float(getattr(metric, attribute)))
    )
    positive = sum(predicate(value) for value in defined_values)
    invalid = sum(
        result.status is ScenarioStatus.INVALID_CONFIGURATION for result in results
    )
    failed = sum(result.status is ScenarioStatus.FAILED for result in results)
    return MetricFractionSummary(
        criterion=criterion,
        eligible_scenarios=eligible,
        defined_scenarios=len(defined_values),
        undefined_scenarios=eligible - len(defined_values),
        positive_scenarios=positive,
        invalid_scenarios=invalid,
        failed_scenarios=failed,
        positive_fraction_defined=(
            float(positive / len(defined_values))
            if defined_values
            else float("nan")
        ),
        positive_fraction_all_eligible=(
            float(positive / eligible) if eligible else float("nan")
        ),
    )


def _parameter_ranges(results: tuple[ScenarioResult, ...]) -> ParameterRanges:
    values: dict[str, tuple[Any, ...]] = {}
    for name in _PARAMETER_NAMES:
        values[name] = tuple(
            sorted({getattr(result.scenario, name) for result in results})
        )
    return ParameterRanges(**values)


def _changed_parameters(
    candidate: ParameterScenario,
    baseline: ParameterScenario,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in _PARAMETER_NAMES
        if getattr(candidate, name) != getattr(baseline, name)
    )


def _axis_metadata(
    ranges: ParameterRanges,
) -> tuple[ParameterAxisMetadata, ...]:
    axes: list[ParameterAxisMetadata] = []
    for name in _PARAMETER_NAMES:
        values = tuple(getattr(ranges, name))
        spacing = tuple(
            float(values[position + 1]) - float(values[position])
            for position in range(len(values) - 1)
        )
        axes.append(
            ParameterAxisMetadata(
                parameter=name,
                tested_values=values,
                count=len(values),
                numeric_spacing=spacing,
            )
        )
    return tuple(axes)


def _local_neighbors(
    variants: tuple[ScenarioResult, ...],
    baseline: ParameterScenario,
    ranges: ParameterRanges,
) -> tuple[LocalNeighbor, ...]:
    local: list[LocalNeighbor] = []
    for result in variants:
        changed_parameter = _changed_parameters(result.scenario, baseline)[0]
        values = tuple(getattr(ranges, changed_parameter))
        baseline_value = getattr(baseline, changed_parameter)
        baseline_position = values.index(baseline_value)
        adjacent: set[int | float] = set()
        if baseline_position > 0:
            adjacent.add(values[baseline_position - 1])
        if baseline_position + 1 < len(values):
            adjacent.add(values[baseline_position + 1])
        neighbor_value = getattr(result.scenario, changed_parameter)
        if neighbor_value not in adjacent:
            continue
        absolute = abs(float(neighbor_value) - float(baseline_value))
        relative = (
            absolute / abs(float(baseline_value))
            if float(baseline_value) != 0.0
            else float("nan")
        )
        local.append(
            LocalNeighbor(
                scenario_id=result.scenario.scenario_id,
                changed_parameter=changed_parameter,
                baseline_value=baseline_value,
                neighbor_value=neighbor_value,
                absolute_distance=absolute,
                relative_distance=relative,
            )
        )
    return tuple(sorted(local, key=lambda item: (item.changed_parameter, item.neighbor_value, item.scenario_id)))


def summarize_sensitivity(
    scenario_results: Iterable[ScenarioResult],
    baseline_scenario_id: str,
    *,
    drawdown_severity_threshold: float = -0.20,
) -> SensitivitySummary:
    """Summarize comparable common-horizon diagnostics without selection."""
    results = tuple(scenario_results)
    if not results:
        raise ValueError("scenario_results must not be empty.")
    if any(not isinstance(result, ScenarioResult) for result in results):
        raise TypeError("scenario_results must contain ScenarioResult values.")
    scenario_ids = tuple(result.scenario.scenario_id for result in results)
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Scenario result IDs must be unique.")
    _require_unique_parameter_tuples(result.scenario for result in results)
    matches = [
        result for result in results if result.scenario.scenario_id == baseline_scenario_id
    ]
    if len(matches) != 1:
        raise ValueError("baseline_scenario_id must identify exactly one result.")
    threshold = _finite_real(
        drawdown_severity_threshold,
        "drawdown_severity_threshold",
    )
    if not -1.0 <= threshold <= 0.0:
        raise ValueError("drawdown_severity_threshold must lie in [-1, 0].")

    baseline_result = matches[0]
    eligible_results = tuple(
        result
        for result in results
        if result.status
        not in {ScenarioStatus.INVALID_CONFIGURATION, ScenarioStatus.FAILED}
    )
    all_common_available = bool(eligible_results) and all(
        result.common_horizon_metrics is not None
        and result.common_horizon_analytics_status
        in {
            MetricAvailabilityStatus.AVAILABLE,
            # Compatibility for explicitly constructed ScenarioResult fixtures.
            MetricAvailabilityStatus.NOT_APPLICABLE,
        }
        for result in eligible_results
    )
    metrics = (
        tuple(
            result.common_horizon_metrics
            for result in eligible_results
            if result.common_horizon_metrics is not None
        )
        if all_common_available
        else ()
    )
    native_metrics = tuple(
        result.calendar_metrics
        for result in eligible_results
        if result.calendar_metrics is not None
    )
    variants = tuple(
        result
        for result in results
        if result.scenario.scenario_id != baseline_scenario_id
        and len(_changed_parameters(result.scenario, baseline_result.scenario)) == 1
    )
    variant_metrics = tuple(
        result.common_horizon_metrics
        for result in variants
        if all_common_available and result.common_horizon_metrics is not None
    )
    ranges = _parameter_ranges(results)
    local = _local_neighbors(variants, baseline_result.scenario, ranges)
    baseline_total = (
        float("nan")
        if not all_common_available
        or baseline_result.common_horizon_metrics is None
        else baseline_result.common_horizon_metrics.total_return
    )
    if np.isfinite(baseline_total):
        same_sign = _defined_fraction(
            (metric.total_return for metric in variant_metrics),
            lambda value: np.sign(value) == np.sign(baseline_total),
        )
    else:
        same_sign = float("nan")

    total_fraction = _fraction_summary(
        results,
        metrics,
        "total_return",
        lambda value: value > 0.0,
        "total_return > 0",
    )
    annual_fraction = _fraction_summary(
        results,
        metrics,
        "annualized_return",
        lambda value: value > 0.0,
        "annualized_return > 0",
    )
    sharpe_fraction = _fraction_summary(
        results,
        metrics,
        "sharpe_ratio",
        lambda value: value > 0.0,
        "sharpe_ratio > 0",
    )
    drawdown_fraction = _fraction_summary(
        results,
        metrics,
        "maximum_drawdown",
        lambda value: value >= threshold,
        "maximum_drawdown >= drawdown_severity_threshold",
    )

    return SensitivitySummary(
        scenario_count=len(results),
        completed_scenarios=sum(
            result.status is ScenarioStatus.COMPLETED for result in results
        ),
        unavailable_scenarios=sum(
            result.status
            in {ScenarioStatus.NO_VALID_FOLDS, ScenarioStatus.ANALYTICS_UNAVAILABLE}
            for result in results
        ),
        invalid_scenarios=sum(
            result.status is ScenarioStatus.INVALID_CONFIGURATION for result in results
        ),
        failed_scenarios=sum(
            result.status is ScenarioStatus.FAILED for result in results
        ),
        baseline_scenario_id=baseline_scenario_id,
        baseline_metrics=(
            baseline_result.common_horizon_metrics
            if all_common_available
            else None
        ),
        metric_distributions=_distributions(metrics),
        fraction_positive_total_return=total_fraction.positive_fraction_defined,
        fraction_positive_annualized_return=(
            annual_fraction.positive_fraction_defined
        ),
        fraction_positive_sharpe=sharpe_fraction.positive_fraction_defined,
        drawdown_severity_threshold=threshold,
        fraction_drawdown_no_worse_than_threshold=(
            drawdown_fraction.positive_fraction_defined
        ),
        parameter_ranges=ranges,
        baseline_neighbor_count=len(variants),
        neighbor_metric_distributions=_distributions(variant_metrics),
        fraction_neighbors_same_total_return_sign=same_sign,
        fraction_neighbors_positive_sharpe=_defined_fraction(
            (metric.sharpe_ratio for metric in variant_metrics),
            lambda value: value > 0.0,
        ),
        headline_metrics_available=all_common_available,
        baseline_native_metrics=baseline_result.calendar_metrics,
        native_metric_distributions=_distributions(native_metrics),
        total_return_fraction=total_fraction,
        annualized_return_fraction=annual_fraction,
        sharpe_fraction=sharpe_fraction,
        drawdown_fraction=drawdown_fraction,
        one_parameter_variant_count=len(variants),
        variant_metric_distributions=_distributions(variant_metrics),
        variant_sign_agreement=same_sign,
        variant_positive_sharpe_fraction=_defined_fraction(
            (metric.sharpe_ratio for metric in variant_metrics),
            lambda value: value > 0.0,
        ),
        local_neighbor_count=len(local),
        local_neighbors=local,
    )


def _provenance_summary(
    results: tuple[ScenarioResult, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], bool, tuple[str, ...]]:
    walk_forward_results = tuple(
        result.walk_forward_result
        for result in results
        if result.walk_forward_result is not None
    )
    universe = tuple(
        sorted({result.universe_provenance for result in walk_forward_results})
    )
    cleaning = tuple(
        sorted({result.cleaning_provenance for result in walk_forward_results})
    )
    validation_states = {
        bool(result.point_in_time_universe_validated)
        for result in walk_forward_results
    }
    validated = bool(walk_forward_results) and validation_states == {True}
    warnings = {
        warning
        for result in walk_forward_results
        for warning in result.provenance_warnings
    }
    if len(universe) > 1:
        warnings.add(
            "Scenarios report conflicting universe provenance states: "
            f"{universe}."
        )
    if len(cleaning) > 1:
        warnings.add(
            "Scenarios report conflicting cleaning provenance states: "
            f"{cleaning}."
        )
    if len(validation_states) > 1:
        warnings.add(
            "Scenarios conflict on point-in-time universe validation; the "
            "top-level result is conservatively not validated."
        )
    return universe, cleaning, validated, tuple(sorted(warnings))


def run_sensitivity_analysis(
    prices: pd.DataFrame,
    scenarios: Iterable[ParameterScenario],
    baseline_scenario_id: str,
    *,
    groups: (
        Mapping[str, Iterable[str]]
        | Callable[[WalkForwardFold], Mapping[str, Iterable[str]] | None]
        | None
    ) = None,
    walk_forward_kwargs: Mapping[str, Any] | None = None,
    drawdown_severity_threshold: float = -0.20,
) -> RobustnessResult:
    """Run a fixed scenario grid independently without OOS parameter selection."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    scenario_tuple = tuple(scenarios)
    if not scenario_tuple:
        raise ValueError("scenarios must not be empty.")
    if any(not isinstance(scenario, ParameterScenario) for scenario in scenario_tuple):
        raise TypeError("scenarios must contain ParameterScenario values.")
    ids = [scenario.scenario_id for scenario in scenario_tuple]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs must be unique.")
    if ids.count(baseline_scenario_id) != 1:
        raise ValueError("baseline_scenario_id must identify exactly one scenario.")
    _require_unique_parameter_tuples(scenario_tuple)
    baseline = scenario_tuple[ids.index(baseline_scenario_id)]
    baseline_error = _configuration_error(baseline)
    if baseline_error is not None:
        raise ValueError(f"The predefined baseline is invalid: {baseline_error}")

    normalized_groups = _normalized_groups(groups)
    shared = _normalized_shared_arguments(walk_forward_kwargs)
    owned_prices = prices.copy(deep=True)
    owned_prices.attrs = deepcopy(prices.attrs)
    ordered_scenarios = tuple(
        sorted(scenario_tuple, key=lambda scenario: scenario.scenario_id)
    )
    if callable(normalized_groups):
        normalized_groups = _snapshotted_group_provider(
            owned_prices,
            ordered_scenarios,
            normalized_groups,
        )
    raw_results = tuple(
        run_parameter_scenario(
            owned_prices,
            scenario,
            groups=normalized_groups,
            walk_forward_kwargs=shared,
        )
        for scenario in ordered_scenarios
    )
    (
        common_results,
        common_index,
        comparable_count,
        structurally_available,
        fully_observed,
        analytics_available,
        common_error,
    ) = _common_horizon(raw_results)
    summary = summarize_sensitivity(
        common_results,
        baseline_scenario_id,
        drawdown_severity_threshold=drawdown_severity_threshold,
    )
    baseline_result = next(
        result
        for result in common_results
        if result.scenario.scenario_id == baseline_scenario_id
    )
    ranges = _parameter_ranges(common_results)
    axes = _axis_metadata(ranges)
    tested_dimensions = tuple(
        axis.parameter for axis in axes if axis.count > 1
    )
    untested_dimensions = tuple(
        dimension
        for dimension in _MATERIAL_DIMENSIONS
        if dimension not in tested_dimensions
    )
    universe, cleaning, point_in_time_validated, provenance_warnings = (
        _provenance_summary(common_results)
    )
    if analytics_available:
        common_analytics_status = MetricAvailabilityStatus.AVAILABLE
    elif structurally_available and len(common_index) < 2:
        common_analytics_status = (
            MetricAvailabilityStatus.UNAVAILABLE_INSUFFICIENT_OBSERVATIONS
        )
    elif structurally_available and not fully_observed:
        common_analytics_status = (
            MetricAvailabilityStatus.UNAVAILABLE_PARTIAL_CALENDAR
        )
    elif structurally_available:
        unavailable_statuses = tuple(
            result.common_horizon_analytics_status
            for result in common_results
            if result.common_horizon_analytics_status
            is not MetricAvailabilityStatus.AVAILABLE
            and result.status
            not in {ScenarioStatus.INVALID_CONFIGURATION, ScenarioStatus.FAILED}
        )
        common_analytics_status = (
            unavailable_statuses[0]
            if unavailable_statuses
            else MetricAvailabilityStatus.NOT_APPLICABLE
        )
    else:
        common_analytics_status = MetricAvailabilityStatus.NOT_APPLICABLE
    return RobustnessResult(
        scenarios=common_results,
        baseline_scenario_id=baseline_scenario_id,
        baseline_scenario=baseline,
        baseline_result=baseline_result,
        summary=summary,
        common_horizon_index=common_index,
        common_horizon_observations=len(common_index),
        common_horizon_scenario_count=comparable_count,
        common_horizon_excluded_scenarios=len(common_results) - comparable_count,
        common_horizon_available=analytics_available,
        common_horizon_error=common_error,
        purpose=PURPOSE,
        warning=NO_OPTIMIZATION_WARNING,
        common_horizon_structurally_available=structurally_available,
        common_horizon_fully_observed=fully_observed,
        common_horizon_analytics_available=analytics_available,
        common_horizon_analytics_status=common_analytics_status,
        common_horizon_analytics_error=common_error,
        axis_metadata=axes,
        tested_dimensions=tested_dimensions,
        untested_material_dimensions=untested_dimensions,
        universe_provenance=universe,
        cleaning_provenance=cleaning,
        point_in_time_universe_validated=point_in_time_validated,
        provenance_warnings=provenance_warnings,
    )


def sensitivity_table(result: RobustnessResult) -> pd.DataFrame:
    """Return a scenario-ID-ordered diagnostic table without winner labels."""
    if not isinstance(result, RobustnessResult):
        raise TypeError("result must be a RobustnessResult.")
    rows: list[dict[str, Any]] = []
    for scenario_result in sorted(
        result.scenarios,
        key=lambda item: item.scenario.scenario_id,
    ):
        scenario = scenario_result.scenario
        metrics = scenario_result.common_horizon_metrics
        native_metrics = scenario_result.calendar_metrics
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "entry_z": scenario.entry_z,
                "exit_z": scenario.exit_z,
                "stop_z": scenario.stop_z,
                "zscore_lookback": scenario.zscore_lookback,
                "formation_window": scenario.formation_window,
                "trading_window": scenario.trading_window,
                "screening_min_observations": scenario.screening_min_observations,
                "commission_bps": scenario.commission_bps,
                "slippage_bps": scenario.slippage_bps,
                "financing_rate": scenario.financing_rate,
                "borrow_rate_y": scenario.borrow_rate_y,
                "borrow_rate_x": scenario.borrow_rate_x,
                "status": scenario_result.status.value,
                "calendar_metrics_status": scenario_result.calendar_metrics_status.value,
                "common_horizon_analytics_status": (
                    scenario_result.common_horizon_analytics_status.value
                ),
                "headline_common_total_return": (
                    np.nan if metrics is None else metrics.total_return
                ),
                "headline_common_annualized_return": (
                    np.nan if metrics is None else metrics.annualized_return
                ),
                "headline_common_annualized_volatility": (
                    np.nan if metrics is None else metrics.annualized_volatility
                ),
                "headline_common_sharpe_ratio": (
                    np.nan if metrics is None else metrics.sharpe_ratio
                ),
                "headline_common_sortino_ratio": (
                    np.nan if metrics is None else metrics.sortino_ratio
                ),
                "headline_common_maximum_drawdown": (
                    np.nan if metrics is None else metrics.maximum_drawdown
                ),
                "headline_common_calmar_ratio": (
                    np.nan if metrics is None else metrics.calmar_ratio
                ),
                "native_calendar_total_return": (
                    np.nan
                    if native_metrics is None
                    else native_metrics.total_return
                ),
                "trade_count": scenario_result.trade_count,
                "total_transaction_cost": scenario_result.total_transaction_cost,
                "equal_capital_reset_fold_transaction_cost_total": (
                    scenario_result.equal_capital_reset_fold_transaction_cost_total
                ),
                "transaction_cost_known_components": (
                    scenario_result.transaction_cost_known_components
                ),
                "transaction_cost_unknown_components": (
                    scenario_result.transaction_cost_unknown_components
                ),
                "transaction_cost_basis": scenario_result.transaction_cost_basis,
                "selection_coverage": scenario_result.selection_coverage,
                "scheduled_oos_observations": (
                    scenario_result.scheduled_oos_observations
                ),
                "available_oos_observations": (
                    scenario_result.available_oos_observations
                ),
                "unavailable_oos_observations": (
                    scenario_result.unavailable_oos_observations
                ),
            }
        )
    return pd.DataFrame(rows)
