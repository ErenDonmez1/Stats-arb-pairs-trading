"""Deterministic diagnostic sensitivity analysis for walk-forward research.

Milestone 8B evaluates a caller-defined parameter grid without choosing a
winner or feeding out-of-sample performance back into parameter selection.
Calendar-time returns from the hardened walk-forward engine are the primary
performance basis; conditional returns are retained as a secondary view.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from itertools import product
from math import prod
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from .analytics import calculate_core_metrics, calculate_drawdown_metrics
from .stats import ADF_MIN_OBSERVATIONS
from .walkforward import (
    WalkForwardFold,
    WalkForwardResult,
    WalkForwardReturnReport,
    run_walk_forward_analysis,
)


__all__ = [
    "ParameterScenario",
    "ScenarioStatus",
    "PerformanceMetrics",
    "MetricDistribution",
    "MetricDistributions",
    "ParameterRanges",
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


def _raw_total_return(returns: pd.Series) -> float:
    valid = returns.dropna()
    if valid.empty:
        return float("nan")
    values = valid.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return float("nan")
    return float(np.prod(1.0 + values) - 1.0)


def _metrics_from_report(
    returns: pd.Series,
    report: WalkForwardReturnReport | None,
) -> PerformanceMetrics:
    total = _raw_total_return(returns)
    if report is None:
        return PerformanceMetrics(
            total_return=total,
            annualized_return=float("nan"),
            annualized_volatility=float("nan"),
            sharpe_ratio=float("nan"),
            sortino_ratio=float("nan"),
            maximum_drawdown=float("nan"),
            calmar_ratio=float("nan"),
            observations=int(returns.notna().sum()),
        )
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
) -> PerformanceMetrics:
    total = _raw_total_return(returns)
    valid = returns.dropna()
    if len(valid) < 2 or bool(valid.le(-1.0).any()):
        return _metrics_from_report(returns, None)
    try:
        core = calculate_core_metrics(returns, periods_per_year, risk_free_rate)
        drawdown = calculate_drawdown_metrics(returns, periods_per_year)
    except (TypeError, ValueError, FloatingPointError, ArithmeticError):
        return _metrics_from_report(returns, None)
    return PerformanceMetrics(
        total_return=core.total_return,
        annualized_return=core.annualized_return,
        annualized_volatility=core.annualized_volatility,
        sharpe_ratio=core.sharpe_ratio,
        sortino_ratio=core.sortino_ratio,
        maximum_drawdown=drawdown.maximum_drawdown,
        calmar_ratio=drawdown.calmar_ratio,
        observations=core.observations,
    )


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
    try:
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
    except (TypeError, ValueError) as exc:
        return _empty_scenario_result(
            scenario,
            ScenarioStatus.FAILED,
            f"{type(exc).__name__}: {exc}",
            periods_per_year=periods,
            risk_free_rate=risk_free,
        )

    calendar = walk_forward.calendar_oos_returns.copy(deep=True)
    conditional = walk_forward.conditional_oos_returns.copy(deep=True)
    available = int(calendar.notna().sum())
    calendar_metrics = (
        None
        if available == 0
        else _metrics_from_report(calendar, walk_forward.calendar_performance_report)
    )
    conditional_metrics = (
        None
        if conditional.notna().sum() == 0
        else _metrics_from_report(
            conditional,
            walk_forward.conditional_performance_report,
        )
    )
    if available == 0:
        status = ScenarioStatus.NO_VALID_FOLDS
        error = "No available calendar OOS observations were produced."
    elif walk_forward.calendar_performance_report is None:
        status = ScenarioStatus.ANALYTICS_UNAVAILABLE
        error = walk_forward.calendar_analytics_error
    else:
        status = ScenarioStatus.COMPLETED
        error = None

    trade_count = sum(fold.trade_count for fold in walk_forward.folds)
    total_transaction_cost = 0.0
    for fold in walk_forward.folds:
        if fold.backtest is None or "transaction_cost" not in fold.backtest.accounting:
            continue
        total_transaction_cost += float(
            fold.backtest.accounting["transaction_cost"].sum(skipna=True)
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
    )


def _common_horizon(
    results: tuple[ScenarioResult, ...],
) -> tuple[tuple[ScenarioResult, ...], pd.Index, str | None, int]:
    comparable = tuple(
        result
        for result in results
        if result.status
        not in {ScenarioStatus.INVALID_CONFIGURATION, ScenarioStatus.FAILED}
        and bool(result.calendar_oos_returns.notna().any())
    )
    empty_index = pd.Index([], dtype=object)
    if not comparable:
        error = "No scenario has available calendar observations."
        return (
            tuple(
                replace(
                    result,
                    common_horizon_metrics=None,
                    common_horizon_error=error,
                    common_horizon_returns=pd.Series(dtype=float),
                )
                for result in results
            ),
            empty_index,
            error,
            0,
        )

    common = comparable[0].calendar_oos_returns.dropna().index.copy()
    for result in comparable[1:]:
        available_index = result.calendar_oos_returns.dropna().index
        common = common[common.isin(available_index)]
    if len(common) < 2:
        error = "Fewer than two calendar observations share a common horizon."
        return (
            tuple(
                replace(
                    result,
                    common_horizon_metrics=None,
                    common_horizon_error=error,
                    common_horizon_returns=pd.Series(dtype=float),
                )
                for result in results
            ),
            common,
            error,
            len(comparable),
        )

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
                )
            )
            continue
        common_returns = result.calendar_oos_returns.loc[common].copy(deep=True)
        updated.append(
            replace(
                result,
                common_horizon_metrics=_metrics_from_returns(
                    common_returns,
                    result.periods_per_year,
                    result.risk_free_rate,
                ),
                common_horizon_error=None,
                common_horizon_returns=common_returns,
            )
        )
    return tuple(updated), common, None, len(comparable)


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


def _parameter_ranges(results: tuple[ScenarioResult, ...]) -> ParameterRanges:
    values: dict[str, tuple[Any, ...]] = {}
    for name in _PARAMETER_NAMES:
        values[name] = tuple(
            sorted({getattr(result.scenario, name) for result in results})
        )
    return ParameterRanges(**values)


def _is_neighbor(candidate: ParameterScenario, baseline: ParameterScenario) -> bool:
    differences = sum(
        getattr(candidate, name) != getattr(baseline, name)
        for name in _PARAMETER_NAMES
    )
    return differences == 1


def summarize_sensitivity(
    scenario_results: Iterable[ScenarioResult],
    baseline_scenario_id: str,
    *,
    drawdown_severity_threshold: float = -0.20,
) -> SensitivitySummary:
    """Summarize medians, quartiles, coverage, and baseline neighbors."""
    results = tuple(scenario_results)
    if not results:
        raise ValueError("scenario_results must not be empty.")
    if any(not isinstance(result, ScenarioResult) for result in results):
        raise TypeError("scenario_results must contain ScenarioResult values.")
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
    metric_values = tuple(
        result.calendar_metrics
        for result in results
        if result.calendar_metrics is not None
        and result.status
        not in {ScenarioStatus.INVALID_CONFIGURATION, ScenarioStatus.FAILED}
    )
    metrics = tuple(metric for metric in metric_values if metric is not None)
    neighbors = tuple(
        result
        for result in results
        if result.scenario.scenario_id != baseline_scenario_id
        and _is_neighbor(result.scenario, baseline_result.scenario)
    )
    neighbor_metrics = tuple(
        result.calendar_metrics
        for result in neighbors
        if result.calendar_metrics is not None
        and result.status
        not in {ScenarioStatus.INVALID_CONFIGURATION, ScenarioStatus.FAILED}
    )
    defined_neighbor_metrics = tuple(
        metric for metric in neighbor_metrics if metric is not None
    )
    baseline_total = (
        float("nan")
        if baseline_result.calendar_metrics is None
        else baseline_result.calendar_metrics.total_return
    )
    if np.isfinite(baseline_total):
        same_sign = _defined_fraction(
            (metric.total_return for metric in defined_neighbor_metrics),
            lambda value: np.sign(value) == np.sign(baseline_total),
        )
    else:
        same_sign = float("nan")

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
        baseline_metrics=baseline_result.calendar_metrics,
        metric_distributions=_distributions(metrics),
        fraction_positive_total_return=_defined_fraction(
            (metric.total_return for metric in metrics),
            lambda value: value > 0.0,
        ),
        fraction_positive_annualized_return=_defined_fraction(
            (metric.annualized_return for metric in metrics),
            lambda value: value > 0.0,
        ),
        fraction_positive_sharpe=_defined_fraction(
            (metric.sharpe_ratio for metric in metrics),
            lambda value: value > 0.0,
        ),
        drawdown_severity_threshold=threshold,
        fraction_drawdown_no_worse_than_threshold=_defined_fraction(
            (metric.maximum_drawdown for metric in metrics),
            lambda value: value >= threshold,
        ),
        parameter_ranges=_parameter_ranges(results),
        baseline_neighbor_count=len(neighbors),
        neighbor_metric_distributions=_distributions(defined_neighbor_metrics),
        fraction_neighbors_same_total_return_sign=same_sign,
        fraction_neighbors_positive_sharpe=_defined_fraction(
            (metric.sharpe_ratio for metric in defined_neighbor_metrics),
            lambda value: value > 0.0,
        ),
    )


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
    baseline = scenario_tuple[ids.index(baseline_scenario_id)]
    baseline_error = _configuration_error(baseline)
    if baseline_error is not None:
        raise ValueError(f"The predefined baseline is invalid: {baseline_error}")

    normalized_groups = _normalized_groups(groups)
    shared = _normalized_shared_arguments(walk_forward_kwargs)
    owned_prices = prices.copy(deep=True)
    owned_prices.attrs = deepcopy(prices.attrs)
    raw_results = tuple(
        run_parameter_scenario(
            owned_prices,
            scenario,
            groups=normalized_groups,
            walk_forward_kwargs=shared,
        )
        for scenario in scenario_tuple
    )
    common_results, common_index, common_error, comparable_count = _common_horizon(
        raw_results
    )
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
        common_horizon_available=common_error is None,
        common_horizon_error=common_error,
        purpose=PURPOSE,
        warning=NO_OPTIMIZATION_WARNING,
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
        metrics = scenario_result.calendar_metrics
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
                "calendar_total_return": (
                    np.nan if metrics is None else metrics.total_return
                ),
                "annualized_return": (
                    np.nan if metrics is None else metrics.annualized_return
                ),
                "annualized_volatility": (
                    np.nan if metrics is None else metrics.annualized_volatility
                ),
                "sharpe_ratio": np.nan if metrics is None else metrics.sharpe_ratio,
                "sortino_ratio": np.nan if metrics is None else metrics.sortino_ratio,
                "maximum_drawdown": (
                    np.nan if metrics is None else metrics.maximum_drawdown
                ),
                "calmar_ratio": np.nan if metrics is None else metrics.calmar_ratio,
                "trade_count": scenario_result.trade_count,
                "total_transaction_cost": scenario_result.total_transaction_cost,
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
