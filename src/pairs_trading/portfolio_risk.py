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

from .portfolio import PortfolioResult


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
            "max_pair_equity_weight",
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
    gross = _numeric_series(
        portfolio.aggregate_exposures["gross_exposure_ratio"],
        "gross exposure ratio",
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
    if bool(gross.loc[gross.notna()].lt(0.0).any()):
        raise ValueError("Gross exposure ratio must be non-negative.")
    if bool(long_value.loc[long_value.notna()].lt(0.0).any()):
        raise ValueError("Long exposure must be non-negative.")
    if bool(short_value.loc[short_value.notna()].lt(0.0).any()):
        raise ValueError("Short exposure must be non-negative.")

    observed = pd.DataFrame(index=index)
    observed["gross_exposure_ratio"] = gross.where(
        exposure_available & positive_equity
    )
    observed["abs_net_exposure_ratio"] = (
        net_value.abs() / equity.where(positive_equity)
    ).where(exposure_available)
    observed["long_exposure_ratio"] = (
        long_value / equity.where(positive_equity)
    ).where(exposure_available)
    observed["short_exposure_ratio"] = (
        short_value / equity.where(positive_equity)
    ).where(exposure_available)

    weights = portfolio.pair_current_equity_weights.copy(deep=True)
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
            elif boundary == "maximum" and value > float(limit):
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
    if len(observations) != len(index):
        raise ValueError("observations must contain one item per row.")
    schedule = observed_metrics.copy(deep=True)
    schedule["overall_status"] = [item.overall_status for item in observations]
    schedule["configured_control_count"] = [
        item.configured_control_count for item in observations
    ]
    schedule["evaluated_control_count"] = [
        item.evaluated_control_count for item in observations
    ]
    schedule["breach_control_count"] = [
        item.breach_control_count for item in observations
    ]
    schedule["unevaluable_control_count"] = [
        item.unevaluable_control_count for item in observations
    ]
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
        total_breach_events=len(breach_records),
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
        "Maximum controls breach only above the limit; the minimum-equity control "
        "breaches only below its limit. Exact equality remains within limit.",
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
