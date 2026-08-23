"""Deterministic portfolio-risk monitoring for Milestone 9A results.

The module evaluates caller-declared scalar limits against an existing
``PortfolioResult``.  It never recomputes portfolio accounting, resizes pair
sleeves, selects risk limits, or claims that an action intent was executed.
Risk observations at row ``t`` may create an intent effective no earlier than
row ``t + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from .portfolio import (
    RISK_LIMIT_TOLERANCE,
    PortfolioResult,
    validate_portfolio_result_invariants,
)


__all__ = [
    "RiskControlStatus",
    "RiskAction",
    "RiskState",
    "PortfolioRiskLimits",
    "PortfolioRiskObservation",
    "PortfolioRiskBreach",
    "PortfolioRiskSummary",
    "PortfolioRiskResult",
    "validate_portfolio_risk_limits",
    "evaluate_portfolio_risk",
    "build_portfolio_risk_schedule",
    "summarize_portfolio_risk",
    "apply_portfolio_risk_policy",
    "run_portfolio_risk_controls",
]


_TOLERANCE = 1e-12
_SYMBOL_GROSS_METRIC = "unnetted_sleeve_gross_market_value"
_ACTION_EXECUTION_POLICY = "intent_only_no_position_or_accounting_replay"


class RiskControlStatus(str, Enum):
    """Evaluation state for one risk control on one row."""

    WITHIN_LIMIT = "WITHIN_LIMIT"
    BREACH = "BREACH"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class RiskAction(str, Enum):
    """Caller-selected action intent after a detected breach."""

    NONE = "NONE"
    HALT_NEW_ENTRIES = "HALT_NEW_ENTRIES"
    LIQUIDATE_ALL = "LIQUIDATE_ALL"


class RiskState(str, Enum):
    """Sticky causal state of the portfolio risk policy.

    ``NORMAL`` means that no breach action has activated.  It does not override
    an ``UNEVALUABLE`` control status or establish permission to trade.
    """

    NORMAL = "NORMAL"
    ENTRY_HALTED = "ENTRY_HALTED"
    LIQUIDATION_REQUIRED = "LIQUIDATION_REQUIRED"
    TERMINAL = "TERMINAL"


_CONTROL_SPECS = (
    ("gross_exposure", "gross_exposure_ratio", "max_gross_exposure_ratio", "maximum"),
    ("abs_net_exposure", "abs_net_exposure_ratio", "max_abs_net_exposure_ratio", "maximum"),
    ("pair_concentration", "largest_pair_equity_weight", "max_pair_equity_weight", "maximum"),
    (
        "symbol_concentration",
        "largest_symbol_unnetted_gross_exposure_ratio",
        "max_symbol_gross_exposure_ratio",
        "maximum",
    ),
    ("active_pairs", "active_pair_count", "max_active_pairs", "maximum"),
    ("drawdown", "drawdown_magnitude", "max_drawdown", "maximum"),
    (
        "minimum_equity",
        "portfolio_equity_ratio",
        "min_portfolio_equity_ratio",
        "minimum",
    ),
    ("long_exposure", "long_exposure_ratio", "max_long_exposure_ratio", "maximum"),
    ("short_exposure", "short_exposure_ratio", "max_short_exposure_ratio", "maximum"),
    (
        "insolvent_sleeves",
        "insolvent_sleeve_count",
        "max_insolvent_sleeves",
        "maximum",
    ),
)
_CONTROL_NAMES = tuple(spec[0] for spec in _CONTROL_SPECS)


def _optional_positive_real(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real scalar or None.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _optional_unit_interval(value: Any, name: str) -> float | None:
    result = _optional_positive_real(value, name)
    if result is not None and result > 1.0:
        raise ValueError(f"{name} must be at most one.")
    return result


def _optional_integer(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer or None.")
    result = int(value)
    if result < minimum:
        relation = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {relation}.")
    return result


def _coerce_action(value: Any) -> RiskAction:
    if isinstance(value, RiskAction):
        return value
    if not isinstance(value, str):
        raise TypeError("breach_action must be a RiskAction or string.")
    try:
        return RiskAction(value.strip().upper())
    except ValueError as exc:
        raise ValueError(f"Unknown breach_action {value!r}.") from exc


@dataclass(frozen=True)
class PortfolioRiskLimits:
    """Predefined scalar portfolio limits; ``None`` means unconfigured."""

    max_gross_exposure_ratio: float | None = None
    max_abs_net_exposure_ratio: float | None = None
    max_pair_equity_weight: float | None = None
    max_symbol_gross_exposure_ratio: float | None = None
    max_active_pairs: int | None = None
    max_drawdown: float | None = None
    min_portfolio_equity_ratio: float | None = None
    max_long_exposure_ratio: float | None = None
    max_short_exposure_ratio: float | None = None
    max_insolvent_sleeves: int | None = None
    breach_action: RiskAction = RiskAction.NONE

    def __post_init__(self) -> None:
        for name in (
            "max_gross_exposure_ratio",
            "max_abs_net_exposure_ratio",
            "max_symbol_gross_exposure_ratio",
            "max_long_exposure_ratio",
            "max_short_exposure_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_real(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "max_pair_equity_weight",
            _optional_unit_interval(
                self.max_pair_equity_weight,
                "max_pair_equity_weight",
            ),
        )
        object.__setattr__(
            self,
            "max_drawdown",
            _optional_unit_interval(self.max_drawdown, "max_drawdown"),
        )
        object.__setattr__(
            self,
            "min_portfolio_equity_ratio",
            _optional_unit_interval(
                self.min_portfolio_equity_ratio,
                "min_portfolio_equity_ratio",
            ),
        )
        object.__setattr__(
            self,
            "max_active_pairs",
            _optional_integer(self.max_active_pairs, "max_active_pairs", minimum=1),
        )
        object.__setattr__(
            self,
            "max_insolvent_sleeves",
            _optional_integer(
                self.max_insolvent_sleeves,
                "max_insolvent_sleeves",
                minimum=0,
            ),
        )
        object.__setattr__(self, "breach_action", _coerce_action(self.breach_action))


@dataclass(frozen=True)
class PortfolioRiskObservation:
    """Typed row-level portfolio risk observation."""

    row_position: int
    row_label: Any
    gross_exposure_ratio: float
    abs_net_exposure_ratio: float
    largest_pair_equity_weight: float
    largest_symbol_unnetted_gross_exposure_ratio: float
    active_pair_count: float
    drawdown: float
    drawdown_magnitude: float
    portfolio_equity_ratio: float
    long_exposure_ratio: float
    short_exposure_ratio: float
    insolvent_sleeve_count: float
    insolvent_pair_ids: tuple[str, ...]
    configured_control_count: int
    evaluated_control_count: int
    breach_control_count: int
    unevaluable_control_count: int
    overall_status: RiskControlStatus
    terminal: bool


@dataclass(frozen=True)
class PortfolioRiskBreach:
    """One deterministically ordered limit breach."""

    row_position: int
    row_label: Any
    control_name: str
    observed_value: float
    configured_limit: float
    status: RiskControlStatus
    message: str
    requested_action: RiskAction
    action_effective_position: int | None
    action_effective_label: Any | None
    action_executed: bool = False


@dataclass(frozen=True)
class PortfolioRiskSummary:
    """Immutable aggregate diagnostics over the complete supplied horizon."""

    observations: int
    configured_controls: tuple[str, ...]
    rows_with_any_breach: int
    rows_unevaluable: int
    first_breach_position: int | None
    first_breach_label: Any | None
    last_breach_position: int | None
    last_breach_label: Any | None
    total_breach_events: int
    breach_count_by_control: tuple[tuple[str, int], ...]
    unevaluable_count_by_control: tuple[tuple[str, int], ...]
    evaluated_count_by_control: tuple[tuple[str, int], ...]
    maximum_observed_gross_exposure_ratio: float
    maximum_observed_abs_net_exposure_ratio: float
    maximum_observed_pair_weight: float
    maximum_observed_symbol_concentration: float
    maximum_drawdown_magnitude: float
    minimum_equity_ratio: float
    maximum_active_pairs: float
    maximum_insolvent_sleeves: float
    terminal_state_reached: bool
    requested_action_count: int


@dataclass(frozen=True)
class PortfolioRiskResult:
    """Defensively owned risk observations and causal action intent."""

    limits: PortfolioRiskLimits
    risk_schedule: pd.DataFrame
    control_statuses: pd.DataFrame
    observed_metrics: pd.DataFrame
    breach_records: tuple[PortfolioRiskBreach, ...]
    observations: tuple[PortfolioRiskObservation, ...]
    policy_state: pd.Series
    requested_actions: pd.Series
    summary: PortfolioRiskSummary
    upstream_portfolio_availability: str
    upstream_self_financing_interpretation: str
    contains_synthetic_reset_sources: bool
    source_path_provenance: tuple[tuple[str, str, str], ...]
    provenance_warnings: tuple[str, ...]
    warnings: tuple[str, ...]
    action_execution_policy: str
    symbol_gross_measure: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "limits", replace(self.limits))
        for name in ("risk_schedule", "control_statuses", "observed_metrics"):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))
        object.__setattr__(self, "policy_state", self.policy_state.copy(deep=True))
        object.__setattr__(
            self,
            "requested_actions",
            self.requested_actions.copy(deep=True),
        )
        object.__setattr__(self, "breach_records", tuple(self.breach_records))
        object.__setattr__(self, "observations", tuple(self.observations))
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


def validate_portfolio_risk_limits(
    limits: PortfolioRiskLimits,
) -> PortfolioRiskLimits:
    """Validate and defensively reproduce immutable risk limits."""
    if not isinstance(limits, PortfolioRiskLimits):
        raise TypeError("limits must be a PortfolioRiskLimits instance.")
    return replace(limits)


def _numeric_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    nonmissing = series.loc[series.notna()].tolist()
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in nonmissing
    ):
        raise TypeError(f"{name} must contain real numbers or missing values.")
    result = series.astype(float).copy(deep=True)
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError(f"{name} must not contain infinity.")
    return result


def _boolean_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if bool(series.isna().any()):
        raise ValueError(f"{name} must not contain missing values.")
    if any(not isinstance(value, (bool, np.bool_)) for value in series.tolist()):
        raise TypeError(f"{name} must contain actual Boolean values.")
    return series.astype(bool).copy(deep=True)


def _validate_control_statuses(
    control_statuses: pd.DataFrame,
    *,
    expected_index: pd.Index | None = None,
) -> None:
    """Validate the complete, enum-valued control-status frame contract."""
    if not isinstance(control_statuses, pd.DataFrame):
        raise TypeError("control_statuses must be a pandas DataFrame.")
    if tuple(control_statuses.columns) != _CONTROL_NAMES:
        raise ValueError("control_statuses columns do not match the risk contract.")
    if not control_statuses.index.is_unique:
        raise ValueError("control_statuses index must be unique.")
    if expected_index is not None and not control_statuses.index.equals(
        expected_index
    ):
        raise ValueError("control_statuses must align exactly with the required index.")
    if bool(control_statuses.isna().to_numpy(dtype=bool).any()):
        raise ValueError("control_statuses must not contain missing values.")
    if any(
        not isinstance(value, RiskControlStatus)
        for value in control_statuses.to_numpy(dtype=object).ravel()
    ):
        raise TypeError(
            "control_statuses must contain only RiskControlStatus values."
        )


def _validate_portfolio_contract(portfolio: PortfolioResult) -> pd.Index:
    if not isinstance(portfolio, PortfolioResult):
        raise TypeError("portfolio must be a PortfolioResult.")
    validate_portfolio_result_invariants(portfolio)
    index = portfolio.portfolio_equity.index.copy()
    if not index.is_unique:
        raise ValueError("PortfolioResult index must be unique.")
    aligned = (
        ("portfolio_returns", portfolio.portfolio_returns),
        ("portfolio_pnl", portfolio.portfolio_pnl),
        ("portfolio_schedule", portfolio.portfolio_schedule),
        ("aggregate_exposures", portfolio.aggregate_exposures),
        ("symbol_exposures", portfolio.symbol_exposures),
        ("pair_current_equity_weights", portfolio.pair_current_equity_weights),
        ("pair_insolvency_state", portfolio.pair_insolvency_state),
    )
    for name, value in aligned:
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            raise TypeError(f"PortfolioResult.{name} must be a pandas object.")
        if not value.index.equals(index):
            raise ValueError(f"PortfolioResult.{name} must align exactly with equity.")
    if tuple(portfolio.pair_current_equity_weights.columns) != portfolio.pair_ids:
        raise ValueError("Current pair-weight columns must match pair_ids exactly.")
    if tuple(portfolio.pair_insolvency_state.columns) != portfolio.pair_ids:
        raise ValueError("Pair-insolvency columns must match pair_ids exactly.")
    required_aggregate = {
        "total_gross_exposure",
        "gross_exposure_ratio",
        "total_long_exposure",
        "total_short_exposure",
        "total_net_exposure",
        "exposure_available",
    }
    if not required_aggregate.issubset(portfolio.aggregate_exposures.columns):
        raise ValueError("PortfolioResult aggregate exposure contract is incomplete.")
    required_schedule = {
        "active_pair_count",
        "active_pair_count_available",
        "portfolio_catastrophic",
    }
    if not required_schedule.issubset(portfolio.portfolio_schedule.columns):
        raise ValueError("PortfolioResult schedule risk contract is incomplete.")
    if not isinstance(portfolio.symbol_exposures.columns, pd.MultiIndex):
        raise ValueError("PortfolioResult symbol exposures require MultiIndex columns.")
    metrics = set(portfolio.symbol_exposures.columns.get_level_values("metric"))
    if _SYMBOL_GROSS_METRIC not in metrics:
        raise ValueError(
            "PortfolioResult lacks unnetted pair-sleeve symbol gross exposure."
        )
    if not isinstance(portfolio.initial_capital, Real) or isinstance(
        portfolio.initial_capital,
        (bool, np.bool_),
    ):
        raise TypeError("PortfolioResult initial capital must be a real number.")
    if not np.isfinite(float(portfolio.initial_capital)) or portfolio.initial_capital <= 0:
        raise ValueError("PortfolioResult initial capital must be finite and positive.")
    if not isinstance(portfolio.contains_synthetic_reset_sources, (bool, np.bool_)):
        raise TypeError(
            "PortfolioResult contains_synthetic_reset_sources must be Boolean."
        )
    return index


def _calculate_observed_metrics(portfolio: PortfolioResult) -> pd.DataFrame:
    index = _validate_portfolio_contract(portfolio)
    equity = _numeric_series(portfolio.portfolio_equity, "portfolio equity")
    if bool(equity.dropna().lt(0.0).any()):
        raise ValueError("Portfolio equity must not be negative.")
    wealth_continuity = pd.Series(False, index=index, dtype=bool)
    continuous = True
    for row_position, value in enumerate(equity.to_numpy(dtype=float)):
        if not continuous or not np.isfinite(value):
            continuous = False
            continue
        wealth_continuity.iloc[row_position] = True
        if value == 0.0:
            continuous = False
    gross_value = _numeric_series(
        portfolio.aggregate_exposures["total_gross_exposure"],
        "gross exposure",
    )
    long_value = _numeric_series(
        portfolio.aggregate_exposures["total_long_exposure"],
        "long exposure",
    )
    short_value = _numeric_series(
        portfolio.aggregate_exposures["total_short_exposure"],
        "short exposure",
    )
    net_value = _numeric_series(
        portfolio.aggregate_exposures["total_net_exposure"],
        "net exposure",
    )
    exposure_available = _boolean_series(
        portfolio.aggregate_exposures["exposure_available"],
        "aggregate exposure availability",
    )
    positive_equity = equity.gt(0.0) & wealth_continuity
    if bool(gross_value.loc[gross_value.notna()].lt(0.0).any()):
        raise ValueError("Gross exposure must be non-negative.")
    if bool(long_value.loc[long_value.notna()].lt(0.0).any()):
        raise ValueError("Long exposure must be non-negative.")
    if bool(short_value.loc[short_value.notna()].lt(0.0).any()):
        raise ValueError("Short exposure must be non-negative.")

    observed = pd.DataFrame(index=index)
    observed["gross_exposure_ratio"] = (
        gross_value / equity.where(positive_equity)
    ).where(exposure_available)
    observed["abs_net_exposure_ratio"] = (
        net_value.abs() / equity.where(positive_equity)
    ).where(exposure_available)
    observed["long_exposure_ratio"] = (
        long_value / equity.where(positive_equity)
    ).where(exposure_available)
    observed["short_exposure_ratio"] = (
        short_value / equity.where(positive_equity)
    ).where(exposure_available)

    weights = portfolio.pair_sleeve_equity.div(
        equity.where(positive_equity),
        axis=0,
    )
    for column in weights:
        weights[column] = _numeric_series(
            weights[column],
            f"current pair weight {column}",
        )
    largest_pair = np.full(len(index), np.nan, dtype=float)
    for row_position in range(len(index)):
        if not bool(positive_equity.iloc[row_position]):
            continue
        row = weights.iloc[row_position]
        if bool(row.isna().any()):
            continue
        if bool(row.lt(-_TOLERANCE).any()):
            raise ValueError("Current pair equity weights must be non-negative.")
        if float(row.sum()) > 1.0 + _TOLERANCE:
            raise ValueError("Current pair equity weights exceed total equity.")
        positive = row.loc[row > 0.0]
        largest_pair[row_position] = float(positive.max()) if not positive.empty else 0.0
    observed["largest_pair_equity_weight"] = largest_pair

    symbol_gross = portfolio.symbol_exposures.xs(
        _SYMBOL_GROSS_METRIC,
        level="metric",
        axis=1,
    ).copy(deep=True)
    for column in symbol_gross:
        symbol_gross[column] = _numeric_series(
            symbol_gross[column],
            f"symbol gross exposure {column}",
        )
    largest_symbol = np.full(len(index), np.nan, dtype=float)
    for row_position in range(len(index)):
        if not bool(positive_equity.iloc[row_position]):
            continue
        row = symbol_gross.iloc[row_position]
        if bool(row.isna().any()):
            continue
        if bool(row.lt(-_TOLERANCE).any()):
            raise ValueError("Symbol unnetted gross values must be non-negative.")
        largest_symbol[row_position] = float(row.max() / equity.iloc[row_position])
    observed["largest_symbol_unnetted_gross_exposure_ratio"] = largest_symbol

    active_available = _boolean_series(
        portfolio.portfolio_schedule["active_pair_count_available"],
        "active pair count availability",
    )
    active = _numeric_series(
        portfolio.portfolio_schedule["active_pair_count"],
        "active pair count",
    )
    finite_active = active.dropna()
    if bool((finite_active < 0.0).any()) or bool(
        (finite_active % 1.0 != 0.0).any()
    ):
        raise ValueError("Active pair count must contain non-negative integers.")
    observed["active_pair_count"] = active.where(active_available)

    running_peak = np.full(len(index), np.nan, dtype=float)
    drawdown = np.full(len(index), np.nan, dtype=float)
    peak = float(portfolio.initial_capital)
    continuity = True
    for row_position, value in enumerate(equity.to_numpy(dtype=float)):
        if not continuity or not np.isfinite(value):
            continuity = False
            continue
        peak = max(peak, value)
        running_peak[row_position] = peak
        drawdown[row_position] = value / peak - 1.0
        if value == 0.0:
            continuity = False
    observed["running_equity_peak"] = running_peak
    observed["drawdown"] = drawdown
    observed["drawdown_magnitude"] = np.maximum(-drawdown, 0.0)
    observed["portfolio_equity_ratio"] = (
        equity.where(wealth_continuity) / float(portfolio.initial_capital)
    )

    insolvency = portfolio.pair_insolvency_state.copy(deep=True)
    for column in insolvency:
        insolvency[column] = _boolean_series(
            insolvency[column],
            f"pair insolvency {column}",
        )
    observed["insolvent_sleeve_count"] = insolvency.sum(axis=1).astype(float)
    observed["insolvent_pair_ids"] = [
        tuple(
            pair_id
            for pair_id in portfolio.pair_ids
            if bool(insolvency.at[label, pair_id])
        )
        for label in index
    ]
    terminal = _boolean_series(
        portfolio.portfolio_schedule["portfolio_catastrophic"],
        "portfolio catastrophic state",
    )
    if bool((terminal.astype(int).diff().fillna(0) < 0).any()):
        raise ValueError("Portfolio terminal state must not revert to non-terminal.")
    observed["terminal"] = terminal
    return observed


def _configured_controls(limits: PortfolioRiskLimits) -> tuple[str, ...]:
    return tuple(
        control
        for control, _, limit_name, _ in _CONTROL_SPECS
        if getattr(limits, limit_name) is not None
    )


def evaluate_portfolio_risk(
    portfolio: PortfolioResult,
    limits: PortfolioRiskLimits,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[PortfolioRiskObservation, ...]]:
    """Evaluate all predefined controls without applying action intent."""
    validated_limits = validate_portfolio_risk_limits(limits)
    observed = _calculate_observed_metrics(portfolio)
    statuses = pd.DataFrame(index=observed.index, columns=_CONTROL_NAMES, dtype=object)
    for control, metric, limit_name, boundary in _CONTROL_SPECS:
        limit = getattr(validated_limits, limit_name)
        if limit is None:
            statuses[control] = [RiskControlStatus.NOT_CONFIGURED] * len(observed)
            continue
        values = observed[metric]
        control_status: list[RiskControlStatus] = []
        for value in values.to_numpy(dtype=float):
            if not np.isfinite(value):
                control_status.append(RiskControlStatus.UNEVALUABLE)
            elif (
                boundary == "maximum"
                and control == "gross_exposure"
                and value > float(limit) + RISK_LIMIT_TOLERANCE
            ):
                control_status.append(RiskControlStatus.BREACH)
            elif (
                boundary == "maximum"
                and control != "gross_exposure"
                and value > float(limit)
            ):
                control_status.append(RiskControlStatus.BREACH)
            elif boundary == "minimum" and value < float(limit):
                control_status.append(RiskControlStatus.BREACH)
            else:
                control_status.append(RiskControlStatus.WITHIN_LIMIT)
        statuses[control] = control_status

    configured = _configured_controls(validated_limits)
    observations: list[PortfolioRiskObservation] = []
    for row_position, row_label in enumerate(observed.index):
        row_status = statuses.iloc[row_position]
        breach_count = int(row_status.eq(RiskControlStatus.BREACH).sum())
        unevaluable_count = int(row_status.eq(RiskControlStatus.UNEVALUABLE).sum())
        within_count = int(row_status.eq(RiskControlStatus.WITHIN_LIMIT).sum())
        if not configured:
            overall = RiskControlStatus.NOT_CONFIGURED
        elif breach_count:
            overall = RiskControlStatus.BREACH
        elif unevaluable_count:
            overall = RiskControlStatus.UNEVALUABLE
        else:
            overall = RiskControlStatus.WITHIN_LIMIT
        metric_row = observed.iloc[row_position]
        observations.append(
            PortfolioRiskObservation(
                row_position=row_position,
                row_label=row_label,
                gross_exposure_ratio=float(metric_row["gross_exposure_ratio"]),
                abs_net_exposure_ratio=float(metric_row["abs_net_exposure_ratio"]),
                largest_pair_equity_weight=float(
                    metric_row["largest_pair_equity_weight"]
                ),
                largest_symbol_unnetted_gross_exposure_ratio=float(
                    metric_row["largest_symbol_unnetted_gross_exposure_ratio"]
                ),
                active_pair_count=float(metric_row["active_pair_count"]),
                drawdown=float(metric_row["drawdown"]),
                drawdown_magnitude=float(metric_row["drawdown_magnitude"]),
                portfolio_equity_ratio=float(metric_row["portfolio_equity_ratio"]),
                long_exposure_ratio=float(metric_row["long_exposure_ratio"]),
                short_exposure_ratio=float(metric_row["short_exposure_ratio"]),
                insolvent_sleeve_count=float(metric_row["insolvent_sleeve_count"]),
                insolvent_pair_ids=tuple(metric_row["insolvent_pair_ids"]),
                configured_control_count=len(configured),
                evaluated_control_count=within_count + breach_count,
                breach_control_count=breach_count,
                unevaluable_control_count=unevaluable_count,
                overall_status=overall,
                terminal=bool(metric_row["terminal"]),
            )
        )
    return observed.copy(deep=True), statuses.copy(deep=True), tuple(observations)


def apply_portfolio_risk_policy(
    control_statuses: pd.DataFrame,
    terminal_state: pd.Series,
    breach_action: RiskAction | str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create sticky next-row action intent without executing or replaying it."""
    _validate_control_statuses(control_statuses)
    if not terminal_state.index.equals(control_statuses.index):
        raise ValueError("terminal_state must align exactly with control_statuses.")
    terminal = _boolean_series(terminal_state, "terminal_state")
    action = _coerce_action(breach_action)
    index = control_statuses.index.copy()
    states: list[RiskState] = []
    effective_actions: list[RiskAction] = [RiskAction.NONE] * len(index)
    detected_intents: list[RiskAction] = [RiskAction.NONE] * len(index)
    effective_positions: list[int | None] = [None] * len(index)
    effective_labels: list[Any | None] = [None] * len(index)
    current_state = RiskState.NORMAL
    pending_action = RiskAction.NONE
    for row_position in range(len(index)):
        if pending_action is not RiskAction.NONE:
            effective_actions[row_position] = pending_action
            if pending_action is RiskAction.HALT_NEW_ENTRIES:
                current_state = RiskState.ENTRY_HALTED
            elif pending_action is RiskAction.LIQUIDATE_ALL:
                current_state = RiskState.LIQUIDATION_REQUIRED
            pending_action = RiskAction.NONE
        if bool(terminal.iloc[row_position]):
            current_state = RiskState.TERMINAL
        states.append(current_state)

        any_breach = bool(
            control_statuses.iloc[row_position].eq(RiskControlStatus.BREACH).any()
        )
        can_emit_action = (
            any_breach
            and action is not RiskAction.NONE
            and current_state is RiskState.NORMAL
            and row_position + 1 < len(index)
        )
        if can_emit_action:
            detected_intents[row_position] = action
            pending_action = action
            effective_positions[row_position] = row_position + 1
            effective_labels[row_position] = index[row_position + 1]

    return (
        pd.Series(states, index=index, name="risk_state", dtype=object),
        pd.Series(
            effective_actions,
            index=index,
            name="requested_action",
            dtype=object,
        ),
        pd.Series(
            detected_intents,
            index=index,
            name="risk_action_intent",
            dtype=object,
        ),
        pd.Series(
            effective_positions,
            index=index,
            name="action_effective_position",
            dtype=object,
        ),
        pd.Series(
            effective_labels,
            index=index,
            name="action_effective_label",
            dtype=object,
        ),
    )


def _build_breach_records(
    observed: pd.DataFrame,
    statuses: pd.DataFrame,
    limits: PortfolioRiskLimits,
    detected_intents: pd.Series,
    action_effective_positions: pd.Series,
    action_effective_labels: pd.Series,
) -> tuple[PortfolioRiskBreach, ...]:
    if not detected_intents.index.equals(observed.index):
        raise ValueError("detected_intents must align exactly with observed metrics.")
    records: list[PortfolioRiskBreach] = []
    for row_position, row_label in enumerate(observed.index):
        for control, metric, limit_name, boundary in _CONTROL_SPECS:
            if statuses.at[row_label, control] is not RiskControlStatus.BREACH:
                continue
            observed_value = float(observed.at[row_label, metric])
            configured_limit = float(getattr(limits, limit_name))
            relation = "exceeds maximum" if boundary == "maximum" else "is below minimum"
            records.append(
                PortfolioRiskBreach(
                    row_position=row_position,
                    row_label=row_label,
                    control_name=control,
                    observed_value=observed_value,
                    configured_limit=configured_limit,
                    status=RiskControlStatus.BREACH,
                    message=(
                        f"{control} observed value {observed_value:.12g} {relation} "
                        f"{configured_limit:.12g}."
                    ),
                    requested_action=detected_intents.iloc[row_position],
                    action_effective_position=action_effective_positions.iloc[
                        row_position
                    ],
                    action_effective_label=action_effective_labels.iloc[row_position],
                    action_executed=False,
                )
            )
    return tuple(records)


def _configured_controls_from_statuses(
    control_statuses: pd.DataFrame,
) -> tuple[str, ...]:
    configured: list[str] = []
    for control in _CONTROL_NAMES:
        not_configured = control_statuses[control].eq(
            RiskControlStatus.NOT_CONFIGURED
        )
        if bool(not_configured.all()):
            continue
        if bool(not_configured.any()):
            raise ValueError(
                f"Control {control!r} mixes configured and NOT_CONFIGURED rows."
            )
        configured.append(control)
    return tuple(configured)


def _canonical_overall_status(
    row: pd.Series,
    configured_controls: tuple[str, ...],
) -> RiskControlStatus:
    if not configured_controls:
        return RiskControlStatus.NOT_CONFIGURED
    configured_row = row.loc[list(configured_controls)]
    if bool(configured_row.eq(RiskControlStatus.BREACH).any()):
        return RiskControlStatus.BREACH
    if bool(configured_row.eq(RiskControlStatus.UNEVALUABLE).any()):
        return RiskControlStatus.UNEVALUABLE
    return RiskControlStatus.WITHIN_LIMIT


def _same_scalar(left: Any, right: Any) -> bool:
    try:
        left_missing = bool(pd.isna(left))
        right_missing = bool(pd.isna(right))
    except (TypeError, ValueError):
        return left == right
    if left_missing or right_missing:
        return left_missing and right_missing
    if isinstance(left, Real) and isinstance(right, Real):
        return bool(np.isclose(float(left), float(right), rtol=_TOLERANCE, atol=_TOLERANCE))
    return bool(left == right)


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.ndim(missing) == 0 else False


def _validate_observations_against_statuses(
    observed_metrics: pd.DataFrame,
    control_statuses: pd.DataFrame,
    observations: tuple[PortfolioRiskObservation, ...],
) -> tuple[RiskControlStatus, ...]:
    if len(observations) != len(observed_metrics):
        raise ValueError("observations must contain one item per row.")
    configured = _configured_controls_from_statuses(control_statuses)
    metric_fields = (
        "gross_exposure_ratio",
        "abs_net_exposure_ratio",
        "largest_pair_equity_weight",
        "largest_symbol_unnetted_gross_exposure_ratio",
        "active_pair_count",
        "drawdown",
        "drawdown_magnitude",
        "portfolio_equity_ratio",
        "long_exposure_ratio",
        "short_exposure_ratio",
        "insolvent_sleeve_count",
    )
    canonical_overall: list[RiskControlStatus] = []
    for row_position, (row_label, status_row) in enumerate(
        control_statuses.iterrows()
    ):
        observation = observations[row_position]
        if not isinstance(observation, PortfolioRiskObservation):
            raise TypeError("observations must contain PortfolioRiskObservation values.")
        if observation.row_position != row_position or not _same_scalar(
            observation.row_label,
            row_label,
        ):
            raise ValueError("Observation row identity is inconsistent.")
        overall = _canonical_overall_status(status_row, configured)
        canonical_overall.append(overall)
        breach_count = int(status_row.eq(RiskControlStatus.BREACH).sum())
        unevaluable_count = int(status_row.eq(RiskControlStatus.UNEVALUABLE).sum())
        evaluated_count = int(
            status_row.isin(
                (RiskControlStatus.WITHIN_LIMIT, RiskControlStatus.BREACH)
            ).sum()
        )
        expected_counts = (
            len(configured),
            evaluated_count,
            breach_count,
            unevaluable_count,
        )
        actual_counts = (
            observation.configured_control_count,
            observation.evaluated_control_count,
            observation.breach_control_count,
            observation.unevaluable_control_count,
        )
        if actual_counts != expected_counts or observation.overall_status is not overall:
            raise ValueError("Observation status/count fields contradict control statuses.")
        metric_row = observed_metrics.loc[row_label]
        for field_name in metric_fields:
            if not _same_scalar(getattr(observation, field_name), metric_row[field_name]):
                raise ValueError(
                    f"Observation field {field_name!r} contradicts observed metrics."
                )
        if observation.insolvent_pair_ids != tuple(metric_row["insolvent_pair_ids"]):
            raise ValueError("Observation insolvent pair IDs contradict observed metrics.")
        if observation.terminal is not bool(metric_row["terminal"]):
            raise ValueError("Observation terminal state contradicts observed metrics.")
    return tuple(canonical_overall)


def _validate_enum_series(
    series: pd.Series,
    enum_type: type[Enum],
    name: str,
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if bool(series.isna().any()):
        raise ValueError(f"{name} must not contain missing values.")
    if any(not isinstance(value, enum_type) for value in series.tolist()):
        raise TypeError(f"{name} contains invalid enum values.")
    return series.copy(deep=True)


def _validate_action_schedule(
    observed_metrics: pd.DataFrame,
    control_statuses: pd.DataFrame,
    policy_state: pd.Series,
    requested_actions: pd.Series,
    detected_intents: pd.Series,
    action_effective_positions: pd.Series,
    action_effective_labels: pd.Series,
) -> None:
    index = observed_metrics.index
    states = _validate_enum_series(policy_state, RiskState, "policy_state")
    requested = _validate_enum_series(
        requested_actions,
        RiskAction,
        "requested_actions",
    )
    intents = _validate_enum_series(
        detected_intents,
        RiskAction,
        "detected_intents",
    )
    terminal = _boolean_series(observed_metrics["terminal"], "terminal")
    current_state = RiskState.NORMAL
    for row_position, row_label in enumerate(index):
        requested_action = requested.iloc[row_position]
        if requested_action is not RiskAction.NONE:
            if current_state is not RiskState.NORMAL:
                raise ValueError("Requested risk actions may activate only once.")
            current_state = (
                RiskState.ENTRY_HALTED
                if requested_action is RiskAction.HALT_NEW_ENTRIES
                else RiskState.LIQUIDATION_REQUIRED
            )
        if bool(terminal.iloc[row_position]):
            current_state = RiskState.TERMINAL
        if states.iloc[row_position] is not current_state:
            raise ValueError("policy_state contradicts requested actions or terminal state.")

        intent = intents.iloc[row_position]
        effective_position = action_effective_positions.iloc[row_position]
        effective_label = action_effective_labels.iloc[row_position]
        if intent is RiskAction.NONE:
            if not _is_missing_scalar(effective_position) or not _is_missing_scalar(
                effective_label
            ):
                raise ValueError("NONE action intent must not have an effective row.")
            continue
        if not bool(
            control_statuses.iloc[row_position].eq(RiskControlStatus.BREACH).any()
        ):
            raise ValueError("Risk action intent requires a same-row control breach.")
        if states.iloc[row_position] is not RiskState.NORMAL:
            raise ValueError("Risk action intent cannot be emitted after state activation.")
        if row_position + 1 >= len(index):
            raise ValueError("Final-row risk action intent has no causal effective row.")
        if isinstance(effective_position, (bool, np.bool_)) or not isinstance(
            effective_position,
            Integral,
        ) or int(effective_position) != row_position + 1:
            raise ValueError("Risk action effective position must be the next row.")
        if not _same_scalar(effective_label, index[row_position + 1]):
            raise ValueError("Risk action effective label must identify the next row.")
        if requested.iloc[row_position + 1] is not intent:
            raise ValueError("Requested action does not match its prior-row intent.")
    for row_position, action in enumerate(requested):
        if action is RiskAction.NONE:
            continue
        if row_position == 0 or intents.iloc[row_position - 1] is not action:
            raise ValueError("Requested action lacks a matching prior-row intent.")


def _validate_breach_records(
    observed_metrics: pd.DataFrame,
    control_statuses: pd.DataFrame,
    breach_records: tuple[PortfolioRiskBreach, ...],
    requested_actions: pd.Series,
    configured_controls: tuple[str, ...],
) -> None:
    canonical_configured = _configured_controls_from_statuses(control_statuses)
    if (
        len(configured_controls) != len(set(configured_controls))
        or any(control not in _CONTROL_NAMES for control in configured_controls)
        or tuple(configured_controls) != canonical_configured
    ):
        raise ValueError("configured_controls contradict the canonical status frame.")
    requested = _validate_enum_series(
        requested_actions,
        RiskAction,
        "requested_actions",
    )
    specs = {control: (metric, boundary) for control, metric, _, boundary in _CONTROL_SPECS}
    expected_keys = [
        (row_position, control)
        for row_position in range(len(control_statuses))
        for control in _CONTROL_NAMES
        if control_statuses.iloc[row_position][control] is RiskControlStatus.BREACH
    ]
    if len(breach_records) != len(expected_keys):
        raise ValueError("breach_records do not contain exactly one record per breach.")
    observed_limits: dict[str, float] = {}
    expected_requested = [RiskAction.NONE] * len(observed_metrics)
    for record, (expected_position, expected_control) in zip(
        breach_records,
        expected_keys,
    ):
        if not isinstance(record, PortfolioRiskBreach):
            raise TypeError("breach_records must contain PortfolioRiskBreach values.")
        expected_label = observed_metrics.index[expected_position]
        if (
            record.row_position != expected_position
            or record.control_name != expected_control
            or not _same_scalar(record.row_label, expected_label)
            or record.status is not RiskControlStatus.BREACH
        ):
            raise ValueError("Breach record row/control/status is inconsistent.")
        metric, boundary = specs[expected_control]
        expected_value = float(observed_metrics.iloc[expected_position][metric])
        if not np.isfinite(record.observed_value) or not np.isclose(
            record.observed_value,
            expected_value,
            rtol=_TOLERANCE,
            atol=_TOLERANCE,
        ):
            raise ValueError("Breach record observed value is inconsistent.")
        limit = float(record.configured_limit)
        if not np.isfinite(limit):
            raise ValueError("Breach record configured limit must be finite.")
        if expected_control in observed_limits and not np.isclose(
            observed_limits[expected_control],
            limit,
            rtol=_TOLERANCE,
            atol=_TOLERANCE,
        ):
            raise ValueError("Breach records use inconsistent limits for one control.")
        observed_limits[expected_control] = limit
        is_breach = (
            expected_value > limit + RISK_LIMIT_TOLERANCE
            if expected_control == "gross_exposure"
            else (
                expected_value > limit
                if boundary == "maximum"
                else expected_value < limit
            )
        )
        if not is_breach:
            raise ValueError("Breach record value does not breach its stated limit.")
        if type(record.action_executed) is not bool or record.action_executed:
            raise ValueError("Risk breach actions must remain unexecuted advisory intent.")
        if record.action_effective_position is None:
            if (
                record.requested_action is not RiskAction.NONE
                or record.action_effective_label is not None
            ):
                raise ValueError("Unemitted breach action metadata is inconsistent.")
        else:
            effective = record.action_effective_position
            if isinstance(effective, (bool, np.bool_)) or not isinstance(
                effective,
                Integral,
            ) or int(effective) != expected_position + 1:
                raise ValueError("Breach action must be effective on the next row.")
            if int(effective) >= len(observed_metrics):
                raise ValueError("Breach action effective row is outside the sample.")
            if record.requested_action is RiskAction.NONE:
                raise ValueError("Effective breach action must not be NONE.")
            if requested.iloc[int(effective)] is not record.requested_action:
                raise ValueError("Breach record action contradicts requested actions.")
            prior_expected = expected_requested[int(effective)]
            if (
                prior_expected is not RiskAction.NONE
                and prior_expected is not record.requested_action
            ):
                raise ValueError("Breach records request conflicting effective actions.")
            expected_requested[int(effective)] = record.requested_action
            if not _same_scalar(
                record.action_effective_label,
                observed_metrics.index[int(effective)],
            ):
                raise ValueError("Breach record effective label is inconsistent.")
    if any(
        actual is not expected
        for actual, expected in zip(requested.tolist(), expected_requested)
    ):
        raise ValueError("requested_actions are not derived from canonical breaches.")


def build_portfolio_risk_schedule(
    observed_metrics: pd.DataFrame,
    control_statuses: pd.DataFrame,
    observations: tuple[PortfolioRiskObservation, ...],
    policy_state: pd.Series,
    requested_actions: pd.Series,
    detected_intents: pd.Series,
    action_effective_positions: pd.Series,
    action_effective_labels: pd.Series,
) -> pd.DataFrame:
    """Combine observed metrics, statuses, counts, and action intent by row."""
    if not isinstance(observed_metrics, pd.DataFrame) or not isinstance(
        control_statuses,
        pd.DataFrame,
    ):
        raise TypeError("observed_metrics and control_statuses must be DataFrames.")
    index = observed_metrics.index.copy()
    _validate_control_statuses(control_statuses, expected_index=index)
    for name, value in (
        ("policy_state", policy_state),
        ("requested_actions", requested_actions),
        ("detected_intents", detected_intents),
        ("action_effective_positions", action_effective_positions),
        ("action_effective_labels", action_effective_labels),
    ):
        if not value.index.equals(index):
            raise ValueError(f"{name} must align exactly with observed_metrics.")
    canonical_overall = _validate_observations_against_statuses(
        observed_metrics,
        control_statuses,
        observations,
    )
    _validate_action_schedule(
        observed_metrics,
        control_statuses,
        policy_state,
        requested_actions,
        detected_intents,
        action_effective_positions,
        action_effective_labels,
    )
    configured_controls = _configured_controls_from_statuses(control_statuses)
    schedule = observed_metrics.copy(deep=True)
    schedule["overall_status"] = canonical_overall
    schedule["configured_control_count"] = len(configured_controls)
    schedule["evaluated_control_count"] = control_statuses.isin(
        (RiskControlStatus.WITHIN_LIMIT, RiskControlStatus.BREACH)
    ).sum(axis=1)
    schedule["breach_control_count"] = control_statuses.eq(
        RiskControlStatus.BREACH
    ).sum(axis=1)
    schedule["unevaluable_control_count"] = control_statuses.eq(
        RiskControlStatus.UNEVALUABLE
    ).sum(axis=1)
    schedule["risk_action_intent"] = detected_intents
    schedule["action_effective_position"] = action_effective_positions
    schedule["action_effective_label"] = action_effective_labels
    schedule["requested_action"] = requested_actions
    schedule["risk_state"] = policy_state
    schedule["action_executed"] = False
    return schedule


def _finite_max(series: pd.Series) -> float:
    finite = series.loc[np.isfinite(series.to_numpy(dtype=float))]
    return float(finite.max()) if not finite.empty else float("nan")


def _finite_min(series: pd.Series) -> float:
    finite = series.loc[np.isfinite(series.to_numpy(dtype=float))]
    return float(finite.min()) if not finite.empty else float("nan")


def summarize_portfolio_risk(
    observed_metrics: pd.DataFrame,
    control_statuses: pd.DataFrame,
    breach_records: tuple[PortfolioRiskBreach, ...],
    requested_actions: pd.Series,
    configured_controls: tuple[str, ...],
) -> PortfolioRiskSummary:
    """Summarize the complete, uncompressed risk horizon."""
    index = observed_metrics.index
    _validate_control_statuses(control_statuses, expected_index=index)
    if not requested_actions.index.equals(index):
        raise ValueError("Risk summary inputs must share the exact index.")
    _validate_breach_records(
        observed_metrics,
        control_statuses,
        breach_records,
        requested_actions,
        configured_controls,
    )
    breach_rows = control_statuses.eq(RiskControlStatus.BREACH).any(axis=1)
    unevaluable_rows = control_statuses.eq(RiskControlStatus.UNEVALUABLE).any(axis=1)
    breach_positions = np.flatnonzero(breach_rows.to_numpy(dtype=bool))
    first_position = int(breach_positions[0]) if len(breach_positions) else None
    last_position = int(breach_positions[-1]) if len(breach_positions) else None
    return PortfolioRiskSummary(
        observations=len(index),
        configured_controls=tuple(configured_controls),
        rows_with_any_breach=int(breach_rows.sum()),
        rows_unevaluable=int(unevaluable_rows.sum()),
        first_breach_position=first_position,
        first_breach_label=(index[first_position] if first_position is not None else None),
        last_breach_position=last_position,
        last_breach_label=(index[last_position] if last_position is not None else None),
        total_breach_events=int(
            control_statuses.eq(RiskControlStatus.BREACH).sum().sum()
        ),
        breach_count_by_control=tuple(
            (
                control,
                int(control_statuses[control].eq(RiskControlStatus.BREACH).sum()),
            )
            for control in configured_controls
        ),
        unevaluable_count_by_control=tuple(
            (
                control,
                int(
                    control_statuses[control]
                    .eq(RiskControlStatus.UNEVALUABLE)
                    .sum()
                ),
            )
            for control in configured_controls
        ),
        evaluated_count_by_control=tuple(
            (
                control,
                int(
                    control_statuses[control]
                    .isin(
                        (
                            RiskControlStatus.WITHIN_LIMIT,
                            RiskControlStatus.BREACH,
                        )
                    )
                    .sum()
                ),
            )
            for control in configured_controls
        ),
        maximum_observed_gross_exposure_ratio=_finite_max(
            observed_metrics["gross_exposure_ratio"]
        ),
        maximum_observed_abs_net_exposure_ratio=_finite_max(
            observed_metrics["abs_net_exposure_ratio"]
        ),
        maximum_observed_pair_weight=_finite_max(
            observed_metrics["largest_pair_equity_weight"]
        ),
        maximum_observed_symbol_concentration=_finite_max(
            observed_metrics["largest_symbol_unnetted_gross_exposure_ratio"]
        ),
        maximum_drawdown_magnitude=_finite_max(
            observed_metrics["drawdown_magnitude"]
        ),
        minimum_equity_ratio=_finite_min(
            observed_metrics["portfolio_equity_ratio"]
        ),
        maximum_active_pairs=_finite_max(observed_metrics["active_pair_count"]),
        maximum_insolvent_sleeves=_finite_max(
            observed_metrics["insolvent_sleeve_count"]
        ),
        terminal_state_reached=bool(observed_metrics["terminal"].any()),
        requested_action_count=int(
            requested_actions.ne(RiskAction.NONE).sum()
        ),
    )


def run_portfolio_risk_controls(
    portfolio: PortfolioResult,
    limits: PortfolioRiskLimits | None = None,
) -> PortfolioRiskResult:
    """Evaluate static risk limits and return causal, unexecuted action intent."""
    resolved_limits = validate_portfolio_risk_limits(
        PortfolioRiskLimits() if limits is None else limits
    )
    observed, statuses, observations = evaluate_portfolio_risk(
        portfolio,
        resolved_limits,
    )
    (
        policy_state,
        requested_actions,
        detected_intents,
        effective_positions,
        effective_labels,
    ) = apply_portfolio_risk_policy(
        statuses,
        observed["terminal"],
        resolved_limits.breach_action,
    )
    breach_records = _build_breach_records(
        observed,
        statuses,
        resolved_limits,
        detected_intents,
        effective_positions,
        effective_labels,
    )
    schedule = build_portfolio_risk_schedule(
        observed,
        statuses,
        observations,
        policy_state,
        requested_actions,
        detected_intents,
        effective_positions,
        effective_labels,
    )
    summary = summarize_portfolio_risk(
        observed,
        statuses,
        breach_records,
        requested_actions,
        _configured_controls(resolved_limits),
    )
    warnings = (
        "Risk controls are post-accounting monitoring. Action values are causal "
        "intent only and never rewrite, resize, liquidate, or rebalance Milestone "
        "9A portfolio accounting.",
        "Gross leverage and hard symbol concentration use unnetted pair-sleeve "
        "gross exposure, not broker-netted margin exposure.",
        "Gross leverage breaches only above its limit plus the shared 1e-12 "
        "numerical tolerance. Other maximum controls breach only above their "
        "exact limits; the minimum-equity control breaches only below its limit.",
        "RiskState.NORMAL means no breach action has activated; it does not "
        "override an UNEVALUABLE overall status or establish permission to trade.",
    )
    provenance_warnings = tuple(
        dict.fromkeys(
            (
                *portfolio.provenance_warnings,
                *portfolio.warnings,
                *warnings,
            )
        )
    )
    return PortfolioRiskResult(
        limits=resolved_limits,
        risk_schedule=schedule,
        control_statuses=statuses,
        observed_metrics=observed,
        breach_records=breach_records,
        observations=observations,
        policy_state=policy_state,
        requested_actions=requested_actions,
        summary=summary,
        upstream_portfolio_availability=portfolio.availability.value,
        upstream_self_financing_interpretation=(
            portfolio.self_financing_interpretation
        ),
        contains_synthetic_reset_sources=portfolio.contains_synthetic_reset_sources,
        source_path_provenance=portfolio.source_path_provenance,
        provenance_warnings=provenance_warnings,
        warnings=warnings,
        action_execution_policy=_ACTION_EXECUTION_POLICY,
        symbol_gross_measure=_SYMBOL_GROSS_METRIC,
    )
