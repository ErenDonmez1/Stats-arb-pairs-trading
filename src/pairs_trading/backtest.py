"""Causal pair execution, accounting, and completed-trade attribution.

The module converts state-changing decisions into lagged executions, sizes the
two legs on the actual execution row, and accounts for exposure, P&L,
execution costs, and simple carry costs.  It also supports an explicit final
liquidation and attributes existing accounting rows to completed trades.
Performance metrics remain outside this milestone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
from enum import Enum
from numbers import Integral, Real
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd

from .data import OBSERVED_PRICE_MASK_ATTR
from .signals import (
    ExitReason,
    PositionState,
    TradeEvent,
    _TRADE_SIGNAL_COLUMNS,
    _TradeSignalPolicy,
    _coerce_trade_signal_dtypes,
    _entry_decision,
    _exit_decision,
    _trade_signal_record,
    _validated_trade_signal_inputs,
)


__all__ = [
    "PairUnits",
    "lag_trade_decisions",
    "calculate_pair_units",
    "build_position_schedule",
    "calculate_position_pnl",
    "calculate_strategy_returns",
    "build_pnl_schedule",
    "calculate_transaction_costs",
    "apply_execution_costs",
    "build_net_pnl_schedule",
    "calculate_borrow_costs",
    "calculate_financing_costs",
    "calculate_rebalancing_costs",
    "build_financed_pnl_schedule",
    "TradeExitReason",
    "TradeRecord",
    "LedgerReconciliation",
    "BacktestResearchMetadata",
    "BacktestResult",
    "force_liquidate_open_position",
    "build_trade_ledger",
    "reconcile_trade_ledger",
    "validate_backtest_invariants",
    "run_pair_backtest",
]


_OUTPUT_COLUMNS = (
    "decision_state",
    "executed_state",
    "decision_event",
    "execution_event",
    "execution_decision_row",
    "execution_due_row",
    "hedge_ratio",
    "price_y",
    "price_x",
    "observed_y",
    "observed_x",
    "units_y",
    "units_x",
    "notional_y",
    "notional_x",
    "gross_exposure",
    "net_exposure",
)

_FORCED_EXIT_EVENT = "FORCED_EXIT"

_EXIT_EVENTS = {
    TradeEvent.EXIT_MEAN_REVERSION.value,
    TradeEvent.EXIT_STOP.value,
    TradeEvent.EXIT_TIME.value,
    _FORCED_EXIT_EVENT,
}

_PNL_OUTPUT_COLUMNS = (
    "price_y",
    "price_x",
    "units_y",
    "units_x",
    "market_value_y",
    "market_value_x",
    "gross_exposure",
    "net_exposure",
    "long_exposure",
    "short_exposure",
    "pnl_y",
    "pnl_x",
    "gross_pnl",
    "realised_pnl",
    "unrealised_pnl",
    "cumulative_realised_pnl",
    "cumulative_gross_pnl",
    "portfolio_equity",
    "strategy_return",
)

_NET_PNL_OUTPUT_COLUMNS = (
    "price_y",
    "price_x",
    "units_y",
    "units_x",
    "delta_units_y",
    "delta_units_x",
    "traded_notional_y",
    "traded_notional_x",
    "commission_y",
    "commission_x",
    "fixed_commission_y",
    "fixed_commission_x",
    "commission_cost",
    "slippage_y",
    "slippage_x",
    "slippage_cost",
    "transaction_cost",
    "cumulative_transaction_cost",
    "gross_pnl",
    "net_pnl",
    "cumulative_gross_pnl",
    "cumulative_net_pnl",
    "portfolio_equity",
    "net_portfolio_equity",
    "strategy_return",
    "net_strategy_return",
)

_FINANCED_OUTPUT_COLUMNS = _NET_PNL_OUTPUT_COLUMNS + (
    "borrow_cost_y",
    "borrow_cost_x",
    "borrow_cost",
    "financing_cost",
    "carry_cost",
    "cumulative_borrow_cost",
    "cumulative_financing_cost",
    "cumulative_carry_cost",
    "rebalance",
    "rebalance_beta",
    "rebalance_delta_units_y",
    "rebalance_delta_units_x",
    "rebalance_decision_row",
    "net_pnl_after_carry",
    "cumulative_net_pnl_after_carry",
    "net_equity_after_carry",
    "net_return_after_carry",
)

_TRADE_LEDGER_COLUMNS = (
    "trade_id",
    "side",
    "entry_row",
    "exit_row",
    "entry_index",
    "exit_index",
    "entry_event",
    "exit_event",
    "exit_reason",
    "entry_price_y",
    "entry_price_x",
    "exit_price_y",
    "exit_price_x",
    "entry_units_y",
    "entry_units_x",
    "exit_units_y",
    "exit_units_x",
    "entry_hedge_ratio",
    "exit_hedge_ratio",
    "holding_period_rows",
    "entry_gross_notional",
    "gross_pnl",
    "commission_cost",
    "slippage_cost",
    "transaction_cost",
    "borrow_cost",
    "financing_cost",
    "carry_cost",
    "total_cost",
    "net_pnl",
    "return_on_entry_gross_notional",
    "forced_exit",
)

_TRADE_LEDGER_FLOAT_COLUMNS = frozenset(
    {
        "entry_price_y",
        "entry_price_x",
        "exit_price_y",
        "exit_price_x",
        "entry_units_y",
        "entry_units_x",
        "exit_units_y",
        "exit_units_x",
        "entry_hedge_ratio",
        "exit_hedge_ratio",
        "entry_gross_notional",
        "gross_pnl",
        "commission_cost",
        "slippage_cost",
        "transaction_cost",
        "borrow_cost",
        "financing_cost",
        "carry_cost",
        "total_cost",
        "net_pnl",
        "return_on_entry_gross_notional",
    }
)


class TradeExitReason(str, Enum):
    """Canonical reasons recorded when an executed trade closes."""

    MEAN_REVERSION = "MEAN_REVERSION"
    STOP = "STOP"
    TIME = "TIME"
    END_OF_BACKTEST = "END_OF_BACKTEST"


@dataclass(frozen=True)
class TradeRecord:
    """Immutable attribution record for one completed executed trade."""

    trade_id: int
    side: str
    entry_row: int
    exit_row: int
    entry_index: Any
    exit_index: Any
    entry_event: str
    exit_event: str
    exit_reason: str
    entry_price_y: float
    entry_price_x: float
    exit_price_y: float
    exit_price_x: float
    entry_units_y: float
    entry_units_x: float
    exit_units_y: float
    exit_units_x: float
    entry_hedge_ratio: float
    exit_hedge_ratio: float
    holding_period_rows: int
    entry_gross_notional: float
    gross_pnl: float
    commission_cost: float
    slippage_cost: float
    transaction_cost: float
    borrow_cost: float
    financing_cost: float
    carry_cost: float
    total_cost: float
    net_pnl: float
    return_on_entry_gross_notional: float
    forced_exit: bool


@dataclass(frozen=True)
class LedgerReconciliation:
    """Immutable comparison of ledger totals with their accounting rows."""

    status: str
    completed_trade_count: int
    has_open_trade: bool
    fully_reconcilable: bool
    completed_totals_match: bool
    final_accounting_match: bool | None
    gross_pnl_match: bool | None
    transaction_cost_match: bool | None
    carry_cost_match: bool | None
    net_pnl_match: bool | None
    ledger_gross_pnl: float
    schedule_completed_gross_pnl: float
    ledger_transaction_cost: float
    schedule_completed_transaction_cost: float
    ledger_carry_cost: float
    schedule_completed_carry_cost: float
    ledger_net_pnl: float
    schedule_completed_net_pnl: float
    open_trade_gross_pnl: float
    open_trade_transaction_cost: float
    open_trade_carry_cost: float
    open_trade_net_pnl: float
    final_cumulative_net_pnl_after_carry: float


@dataclass(frozen=True)
class BacktestResearchMetadata:
    """Explicit limitations and timing assumptions for research consumers."""

    upstream_inputs_assumed_causal: bool
    upstream_provenance_validated: bool
    warning: str
    price_policy: str
    hedge_ratio_policy: str
    sizing_policy: str
    dollar_neutrality_note: str


@dataclass(frozen=True)
class BacktestResult:
    """Independently owned outputs from one integrated pair backtest.

    The frozen wrapper prevents field replacement.  Its DataFrames are fresh
    defensive copies with no shared storage with caller inputs or other runs;
    callers may therefore mutate their returned frames without affecting the
    pipeline or a separately produced result.
    """

    signals: pd.DataFrame
    positions: pd.DataFrame
    accounting: pd.DataFrame
    ledger: pd.DataFrame
    reconciliation: LedgerReconciliation
    research_metadata: BacktestResearchMetadata
    forced_liquidation_requested: bool
    forced_liquidation_applied: bool
    execution_lag: int


class PairUnits(NamedTuple):
    """Signed fractional units of the dependent and explanatory symbols."""

    units_y: float
    units_x: float


@dataclass(frozen=True)
class _PendingOrder:
    """One matured state target with immutable decision and due provenance."""

    target_state: str
    event: str
    decision_position: int
    decision_label: Any
    due_position: int
    due_label: Any


@dataclass(frozen=True)
class _ExecutionDecision:
    """One current-row target that will mature after the configured lag."""

    target_state: str
    event: str


@dataclass(frozen=True)
class _ExecutionFeedback:
    """Actual state transition produced before the current decision is made."""

    previous_state: str
    executed_state: str
    execution_event: str


@dataclass(frozen=True)
class _ExecutionMarketInputs:
    """Defensively validated arrays shared by both scheduling pathways."""

    price_y: pd.Series
    price_x: pd.Series
    observed_y: pd.Series
    observed_x: pd.Series
    hedge_ratio: pd.Series
    target_gross_notional: float
    execution_lag: int


class _ExecutionAwareStrategy:
    """Causal z-score policy whose clocks follow acknowledged executions."""

    def __init__(self, zscore: pd.Series, policy: _TradeSignalPolicy) -> None:
        self._values = zscore.to_numpy(dtype=float)
        self._policy = policy
        self._desired_state = PositionState.FLAT
        self._holding_period = 0
        self._cooldown_remaining = 0
        self._rows: list[tuple[Any, ...]] = []

    def decide(
        self,
        row_number: int,
        feedback: _ExecutionFeedback,
    ) -> _ExecutionDecision:
        """Create the current decision after applying this row's actual fill."""
        current_zscore = float(self._values[row_number])
        previous_actual = PositionState[feedback.previous_state]
        current_actual = PositionState[feedback.executed_state]
        entered = feedback.execution_event in {
            TradeEvent.ENTER_LONG.value,
            TradeEvent.ENTER_SHORT.value,
        }
        exited = feedback.execution_event in _EXIT_EVENTS

        row_holding_period = 0
        if previous_actual is not PositionState.FLAT:
            self._holding_period += 1
            row_holding_period = self._holding_period
        if entered:
            self._holding_period = 0
            row_holding_period = 0

        if exited:
            self._cooldown_remaining = self._policy.cooldown_period
        row_cooldown_remaining = self._cooldown_remaining

        event = TradeEvent.NONE
        exit_reason = ExitReason.NONE
        if exited:
            if self._desired_state is not PositionState.FLAT:
                raise RuntimeError(
                    "An executed close requires an existing desired FLAT target."
                )
        elif current_actual is not PositionState.FLAT:
            if self._desired_state is current_actual:
                event, exit_reason = _exit_decision(
                    current_actual,
                    current_zscore,
                    self._holding_period,
                    self._policy,
                )
                if event is not TradeEvent.NONE:
                    self._desired_state = PositionState.FLAT
            elif self._desired_state is not PositionState.FLAT:
                raise RuntimeError(
                    "Desired and actual open states cannot point in opposite directions."
                )
        elif self._desired_state is not PositionState.FLAT:
            # A pending entry is not exposure, so stop and time-exit rules do
            # not apply.  Mean reversion can still invalidate the unfilled
            # entry intent through a normally lagged FLAT target; it does not
            # start cooldown because no actual exit has filled.
            pending_entry_reverted = bool(
                not np.isnan(current_zscore)
                and (
                    (
                        self._desired_state is PositionState.LONG_SPREAD
                        and current_zscore >= -self._policy.exit_z
                    )
                    or (
                        self._desired_state is PositionState.SHORT_SPREAD
                        and current_zscore <= self._policy.exit_z
                    )
                )
            )
            if pending_entry_reverted:
                self._desired_state = PositionState.FLAT
                event = TradeEvent.EXIT_MEAN_REVERSION
                exit_reason = ExitReason.MEAN_REVERSION
        elif self._desired_state is PositionState.FLAT:
            # The exit-fill row itself never admits a new entry, including
            # when cooldown is configured as zero.
            if not exited and self._cooldown_remaining == 0:
                self._desired_state, event = _entry_decision(
                    current_zscore,
                    self._policy,
                )

        self._rows.append(
            _trade_signal_record(
                current_zscore,
                self._desired_state,
                event,
                exit_reason,
                row_holding_period,
                row_cooldown_remaining,
            )
        )

        if exited:
            self._holding_period = 0
        elif (
            current_actual is PositionState.FLAT
            and self._desired_state is PositionState.FLAT
            and self._cooldown_remaining > 0
        ):
            # Report the pre-decrement value that blocked this whole row.
            self._cooldown_remaining -= 1

        return _ExecutionDecision(
            target_state=self._desired_state.name,
            event=event.value,
        )

    def to_frame(self, index: pd.Index) -> pd.DataFrame:
        """Return the execution-aware decisions with their actual clocks."""
        result = pd.DataFrame.from_records(
            self._rows,
            columns=list(_TRADE_SIGNAL_COLUMNS),
        )
        result.index = index
        return _coerce_trade_signal_dtypes(result)


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive, non-Boolean integer within the int64 range."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer.")
    normalised = int(value)
    if normalised <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    if normalised > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the supported int64 range.")
    return normalised


def _finite_positive_scalar(value: Any, name: str) -> float:
    """Return a finite, strictly positive, non-Boolean real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    try:
        normalised = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as a float.") from exc
    if not np.isfinite(normalised):
        raise ValueError(f"{name} must be finite.")
    if normalised <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return normalised


def _finite_nonnegative_scalar(value: Any, name: str) -> float:
    """Return a finite, non-negative, non-Boolean real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    try:
        normalised = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as a float.") from exc
    if not np.isfinite(normalised):
        raise ValueError(f"{name} must be finite.")
    if normalised < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return normalised


def _require_causal_index(index: pd.Index, name: str) -> None:
    """Reject duplicate or nonchronological dated rows without sorting."""
    if not index.is_unique:
        raise ValueError(f"{name} must have a unique index.")
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        raise ValueError(
            f"{name} DatetimeIndex must be monotonically increasing."
        )


def _normalise_state(value: Any, *, name: str = "state") -> str:
    """Return a canonical :class:`PositionState` member name."""
    if isinstance(value, PositionState):
        return value.name
    if isinstance(value, str):
        if value in PositionState.__members__:
            return value
        raise ValueError(f"{name} contains an unknown position state: {value!r}.")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not contain Boolean values.")
    if isinstance(value, Integral):
        try:
            return PositionState(int(value)).name
        except ValueError as exc:
            raise ValueError(
                f"{name} contains an unknown position encoding: {value!r}."
            ) from exc
    raise TypeError(
        f"{name} values must be PositionState members, names, or integer encodings."
    )


def _normalise_event(value: Any) -> str:
    """Return a canonical :class:`TradeEvent` value."""
    if isinstance(value, TradeEvent):
        return value.value
    if isinstance(value, str):
        if value == _FORCED_EXIT_EVENT:
            return value
        try:
            return TradeEvent(value).value
        except ValueError as exc:
            raise ValueError(f"event contains an unknown trade event: {value!r}.") from exc
    raise TypeError("event values must be TradeEvent members or canonical strings.")


def _validated_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Copy and validate the state/event contract emitted by signal generation."""
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals must be a pandas DataFrame.")
    _require_causal_index(signals.index, "signals")

    required = {"state", "event"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals is missing required columns: {sorted(missing)}.")

    states = [_normalise_state(value, name="signals['state']") for value in signals["state"]]
    events = [_normalise_event(value) for value in signals["event"]]

    previous_state = PositionState.FLAT.name
    for row_number, (state, event) in enumerate(zip(states, events, strict=True)):
        if event == TradeEvent.NONE.value:
            if state != previous_state:
                raise ValueError(
                    "signals contains a state change without a corresponding event "
                    f"at row position {row_number}."
                )
        elif event == TradeEvent.ENTER_LONG.value:
            if previous_state != PositionState.FLAT.name or state != PositionState.LONG_SPREAD.name:
                raise ValueError(
                    f"signals contains an invalid long-entry transition at row position {row_number}."
                )
        elif event == TradeEvent.ENTER_SHORT.value:
            if previous_state != PositionState.FLAT.name or state != PositionState.SHORT_SPREAD.name:
                raise ValueError(
                    f"signals contains an invalid short-entry transition at row position {row_number}."
                )
        elif event in _EXIT_EVENTS:
            if previous_state == PositionState.FLAT.name or state != PositionState.FLAT.name:
                raise ValueError(
                    f"signals contains an invalid exit transition at row position {row_number}."
                )
        previous_state = state

    result = pd.DataFrame(
        {
            "decision_state": pd.Series(states, index=signals.index, dtype=object),
            "decision_event": pd.Series(events, index=signals.index, dtype=object),
        }
    )
    result.index = signals.index
    return result


def _validated_market_series(
    series: pd.Series,
    name: str,
    *,
    strictly_positive: bool,
) -> pd.Series:
    """Copy a numeric Series, allowing missing rows solely for deferral."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    _require_causal_index(series.index, name)

    copied = series.copy(deep=True)
    non_missing = copied.loc[copied.notna()]
    numeric = non_missing.map(
        lambda value: isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if not bool(numeric.all()):
        raise ValueError(f"Non-missing {name} observations must be real numeric values.")

    try:
        array = copied.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Non-missing {name} observations must be representable as floats."
        ) from exc

    values = pd.Series(array, index=series.index, name=name, dtype=float)
    present = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(present).all():
        raise ValueError(f"Non-missing {name} observations must be finite.")
    if strictly_positive and bool((present <= 0.0).any()):
        raise ValueError(f"Non-missing {name} observations must be strictly positive.")
    return values


def _require_matching_index(reference: pd.Index, candidate: pd.Index, name: str) -> None:
    """Require the same labels in the same order without sorting or aligning."""
    if not candidate.equals(reference):
        raise ValueError(f"{name} index must exactly match the price index and order.")


def _validated_observed_mask(
    price: pd.Series,
    supplied: pd.Series | None,
    name: str,
    schedule_mask: pd.Series | None = None,
) -> pd.Series:
    """Return the conservative intersection of all price-provenance masks.

    Explicit masks, metadata attached by :class:`MarketDataLoader`, and mask
    columns carried by a position schedule are independent evidence.  Every
    available source is validated and combined with logical AND, so no True
    source can override another source's False value.  Raw Series with no
    provenance information treat non-missing values as observed for backward
    compatibility.
    """
    sources: list[tuple[str, pd.Series]] = []
    if supplied is not None:
        sources.append(("explicit mask", supplied))

    if OBSERVED_PRICE_MASK_ATTR in price.attrs:
        provenance = price.attrs[OBSERVED_PRICE_MASK_ATTR]
        if isinstance(provenance, pd.DataFrame):
            if price.name not in provenance.columns:
                raise ValueError(
                    f"{name} price metadata has no column for {price.name!r}."
                )
            sources.append(("price metadata", provenance[price.name]))
        elif isinstance(provenance, pd.Series):
            sources.append(("price metadata", provenance))
        else:
            raise TypeError(
                f"{name} price metadata must be a pandas Series or DataFrame."
            )

    if schedule_mask is not None:
        sources.append(("schedule mask", schedule_mask))

    if not sources:
        return price.notna().astype(bool).rename(name)

    result = pd.Series(True, index=price.index, name=name, dtype=bool)
    for source_label, source in sources:
        if not isinstance(source, pd.Series):
            raise TypeError(f"{name} {source_label} must be a pandas Series.")
        if not source.index.is_unique:
            raise ValueError(f"{name} {source_label} must have a unique index.")
        _require_matching_index(
            price.index,
            source.index,
            f"{name} {source_label}",
        )
        if source.isna().any():
            raise ValueError(f"{name} {source_label} must not contain missing values.")
        valid_values = source.map(
            lambda value: isinstance(value, (bool, np.bool_))
        )
        if not bool(valid_values.all()):
            raise TypeError(
                f"{name} {source_label} must contain only Boolean values."
            )
        result &= source.astype(bool)

    result = result.copy(deep=True)
    result.name = name
    return result


def _available_hedge_ratios(
    hedge_ratio: float | pd.Series,
    index: pd.Index,
    availability_lag: int,
) -> pd.Series:
    """Return beta values known before each execution row.

    Scalar hedge ratios are treated as pre-frozen parameters.  A Series is
    treated as close-derived posterior estimates and shifted by the explicit
    row lag, so ``beta_t`` cannot be used for a same-close execution.
    """
    lag = _positive_integer(availability_lag, "hedge_ratio_lag")
    if isinstance(hedge_ratio, pd.Series):
        values = _validated_market_series(
            hedge_ratio,
            "hedge_ratio",
            strictly_positive=True,
        )
        _require_matching_index(index, hedge_ratio.index, "hedge_ratio")
        if lag:
            values = values.shift(lag)
        values.name = "hedge_ratio"
        return values
    beta = _finite_positive_scalar(hedge_ratio, "hedge_ratio")
    return pd.Series(beta, index=index, name="hedge_ratio", dtype=float)


def _validated_execution_market_inputs(
    price_y: pd.Series,
    price_x: pd.Series,
    hedge_ratio: float | pd.Series,
    target_gross_notional: float,
    execution_lag: int,
    observed_y: pd.Series | None,
    observed_x: pd.Series | None,
    hedge_ratio_lag: int,
) -> _ExecutionMarketInputs:
    """Validate and own every market input used by the execution scheduler."""
    lag = _positive_integer(execution_lag, "execution_lag")
    gross_target = _finite_positive_scalar(
        target_gross_notional,
        "target_gross_notional",
    )
    y_values = _validated_market_series(
        price_y,
        "price_y",
        strictly_positive=True,
    )
    x_values = _validated_market_series(
        price_x,
        "price_x",
        strictly_positive=True,
    )
    _require_matching_index(price_y.index, price_x.index, "price_x")
    observed_y_values = _validated_observed_mask(
        price_y,
        observed_y,
        "observed_y",
    )
    observed_x_values = _validated_observed_mask(
        price_x,
        observed_x,
        "observed_x",
    )
    beta_values = _available_hedge_ratios(
        hedge_ratio,
        price_y.index,
        hedge_ratio_lag,
    )
    return _ExecutionMarketInputs(
        price_y=y_values,
        price_x=x_values,
        observed_y=observed_y_values,
        observed_x=observed_x_values,
        hedge_ratio=beta_values,
        target_gross_notional=gross_target,
        execution_lag=lag,
    )


def lag_trade_decisions(
    signals: pd.DataFrame,
    execution_lag: int = 1,
) -> pd.DataFrame:
    """Return decisions and the orders that become due after a positional lag.

    Lagging is by row position, never by timestamp arithmetic.  ``due_event``
    describes an order whose minimum delay has elapsed; it is not proof of an
    execution because market inputs may be unavailable.  The final
    ``execution_lag`` decisions have no in-sample due row and remain
    unexecuted.
    """
    lag = _positive_integer(execution_lag, "execution_lag")
    decisions = _validated_signals(signals)
    row_count = len(decisions)

    due_states: list[str | None] = [None] * row_count
    due_events = [TradeEvent.NONE.value] * row_count
    if lag < row_count:
        due_states[lag:] = decisions["decision_state"].iloc[:-lag].tolist()
        due_events[lag:] = decisions["decision_event"].iloc[:-lag].tolist()

    result = decisions.copy(deep=True)
    result["due_state"] = pd.Series(due_states, index=signals.index, dtype=object)
    result["due_event"] = pd.Series(due_events, index=signals.index, dtype=object)
    result.index = signals.index
    return result.loc[
        :, ["decision_state", "decision_event", "due_state", "due_event"]
    ]


def calculate_pair_units(
    state: PositionState | str | int,
    price_y: float,
    price_x: float,
    hedge_ratio: float,
    target_gross_notional: float,
) -> PairUnits:
    """Calculate signed pair units from positive-beta gross-notional weights.

    For ``beta > 0``, the absolute notional weights are
    ``1 / (1 + beta)`` for symbol y and ``beta / (1 + beta)`` for symbol x.
    LONG_SPREAD buys y and shorts x; SHORT_SPREAD reverses both signs.  The
    weights sum to one, so gross exposure at sizing equals the requested
    target.  Net dollar exposure is exactly zero only when ``beta == 1``.

    Zero and negative hedge ratios are rejected.  Supporting them would either
    remove the explanatory leg or require same-direction legs, contrary to the
    pair-direction contract for this milestone.
    """
    normalised_state = _normalise_state(state)
    normalised_price_y = _finite_positive_scalar(price_y, "price_y")
    normalised_price_x = _finite_positive_scalar(price_x, "price_x")
    beta = _finite_positive_scalar(hedge_ratio, "hedge_ratio")
    gross_target = _finite_positive_scalar(
        target_gross_notional,
        "target_gross_notional",
    )

    direction = int(PositionState[normalised_state])
    if direction == 0:
        return PairUnits(0.0, 0.0)

    # Form the small weight through reciprocal beta so the denominator remains
    # well scaled even for hedge ratios close to the float64 maximum.
    if beta >= 1.0:
        inverse_beta = 1.0 / beta
        weight_y = inverse_beta / (1.0 + inverse_beta)
        weight_x = 1.0 - weight_y
    else:
        weight_x = beta / (1.0 + beta)
        weight_y = 1.0 - weight_x

    notional_y = direction * gross_target * weight_y
    notional_x = -direction * gross_target * weight_x
    units_y = notional_y / normalised_price_y
    units_x = notional_x / normalised_price_x
    if not np.isfinite([units_y, units_x]).all():
        raise ValueError("Calculated pair units must be finite.")
    if direction > 0 and not (units_y > 0.0 and units_x < 0.0):
        raise ValueError(
            "Calculated long-spread units must contain two representable, "
            "nonzero legs."
        )
    if direction < 0 and not (units_y < 0.0 and units_x > 0.0):
        raise ValueError(
            "Calculated short-spread units must contain two representable, "
            "nonzero legs."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        represented_gross = (
            abs(units_y * normalised_price_y)
            + abs(units_x * normalised_price_x)
        )
    if (
        not np.isfinite(represented_gross)
        or represented_gross <= 0.0
        or not np.isclose(
            represented_gross,
            gross_target,
            rtol=1e-12,
            atol=0.0,
        )
    ):
        raise ValueError(
            "Calculated pair units cannot represent the target gross notional."
        )
    return PairUnits(float(units_y), float(units_x))


def _mark_exposure(
    state: str,
    units_y: float,
    units_x: float,
    price_y: float,
    price_x: float,
) -> tuple[float, float, float, float]:
    """Return signed leg and aggregate current notionals."""
    if state == PositionState.FLAT.name:
        return 0.0, 0.0, 0.0, 0.0

    with np.errstate(over="ignore", invalid="ignore"):
        notional_y = units_y * price_y if np.isfinite(price_y) else np.nan
        notional_x = units_x * price_x if np.isfinite(price_x) else np.nan
    if np.isnan(notional_y) or np.isnan(notional_x):
        return float(notional_y), float(notional_x), np.nan, np.nan
    if not np.isfinite([notional_y, notional_x]).all():
        raise ValueError("Marked leg notionals must remain finite.")
    with np.errstate(over="ignore", invalid="ignore"):
        gross = abs(notional_y) + abs(notional_x)
        net = notional_y + notional_x
    if not np.isfinite([gross, net]).all():
        raise ValueError("Marked gross and net exposures must remain finite.")
    return float(notional_y), float(notional_x), float(gross), float(net)


def _queue_matured_order(
    pending_orders: deque[_PendingOrder],
    order: _PendingOrder,
    executed_state: str,
) -> deque[_PendingOrder]:
    """Apply one matured target while removing transitions it invalidates."""
    if order.target_state != PositionState.FLAT.name:
        pending_orders.append(order)
        return pending_orders

    if executed_state == PositionState.FLAT.name:
        # The matured FLAT target invalidates every unfilled entry.  Any queued
        # FLAT target is redundant because the real position is already flat.
        return deque()

    # The real position still needs one close.  Preserve the earliest matured
    # close and its event provenance, while cancelling later entries and the
    # redundant closes that existed only for those cancelled entries.
    required_close = next(
        (
            candidate
            for candidate in pending_orders
            if candidate.target_state == PositionState.FLAT.name
        ),
        order,
    )
    return deque([required_close])


def _build_position_schedule_core(
    index: pd.Index,
    inputs: _ExecutionMarketInputs,
    decision_provider: Callable[
        [int, _ExecutionFeedback],
        _ExecutionDecision,
    ],
    *,
    suppress_terminal_entries: bool,
) -> pd.DataFrame:
    """Execute one causal decision provider through the hardened order queue."""
    pending_orders: deque[_PendingOrder] = deque()
    decision_history: list[_ExecutionDecision] = []
    executed_state = PositionState.FLAT.name
    units_y = 0.0
    units_x = 0.0
    rows: list[tuple[Any, ...]] = []

    y_array = inputs.price_y.to_numpy(dtype=float)
    x_array = inputs.price_x.to_numpy(dtype=float)
    beta_array = inputs.hedge_ratio.to_numpy(dtype=float)
    observed_y_array = inputs.observed_y.to_numpy(dtype=bool)
    observed_x_array = inputs.observed_x.to_numpy(dtype=bool)
    lag = inputs.execution_lag

    for row_number in range(len(index)):
        previous_executed_state = executed_state
        due_decision = (
            decision_history[row_number - lag]
            if row_number >= lag
            else None
        )
        due_event = (
            TradeEvent.NONE.value
            if due_decision is None
            else due_decision.event
        )
        if due_event != TradeEvent.NONE.value:
            if due_decision is None:  # Defensive: branch condition excludes this.
                raise RuntimeError("A due event has no associated decision.")
            decision_position = row_number - lag
            matured_order = _PendingOrder(
                target_state=due_decision.target_state,
                event=due_event,
                decision_position=decision_position,
                decision_label=index[decision_position],
                due_position=row_number,
                due_label=index[row_number],
            )
            # Only this matured target may change the pending path.  The
            # current-row decision remains causally immature until its own due
            # position.
            pending_orders = _queue_matured_order(
                pending_orders,
                matured_order,
                executed_state,
            )

        # Discard a redundant target without presenting it as an execution.
        while pending_orders and pending_orders[0].target_state == executed_state:
            pending_orders.popleft()

        current_y = y_array[row_number]
        current_x = x_array[row_number]
        current_beta = beta_array[row_number]
        execution_event = TradeEvent.NONE.value
        executed_order: _PendingOrder | None = None
        prices_available = bool(
            np.isfinite(current_y)
            and np.isfinite(current_x)
            and observed_y_array[row_number]
            and observed_x_array[row_number]
        )

        if pending_orders:
            candidate = pending_orders[0]
            beta_required = candidate.target_state != PositionState.FLAT.name
            execution_inputs_available = bool(
                prices_available
                and (not beta_required or np.isfinite(current_beta))
            )
        else:
            candidate = None
            execution_inputs_available = False

        if candidate is not None and execution_inputs_available:
            target_state = candidate.target_state
            source_event = candidate.event
            terminal_entry_suppressed = bool(
                suppress_terminal_entries
                and row_number == len(index) - 1
                and executed_state == PositionState.FLAT.name
                and target_state != PositionState.FLAT.name
            )
            if terminal_entry_suppressed:
                pending_orders.popleft()
                target_state = executed_state
                source_event = TradeEvent.NONE.value
            else:
                executed_order = pending_orders.popleft()
            if (
                executed_state != PositionState.FLAT.name
                and target_state != PositionState.FLAT.name
            ):
                raise RuntimeError(
                    "A direct executed-position reversal would violate the signal contract."
                )

            if target_state == PositionState.FLAT.name:
                units_y = 0.0
                units_x = 0.0
            else:
                sizing = calculate_pair_units(
                    target_state,
                    current_y,
                    current_x,
                    current_beta,
                    inputs.target_gross_notional,
                )
                units_y, units_x = sizing
            executed_state = target_state
            execution_event = source_event

        current_decision = decision_provider(
            row_number,
            _ExecutionFeedback(
                previous_state=previous_executed_state,
                executed_state=executed_state,
                execution_event=execution_event,
            ),
        )
        decision_state = _normalise_state(
            current_decision.target_state,
            name="decision_provider state",
        )
        decision_event = _normalise_event(current_decision.event)
        decision_history.append(
            _ExecutionDecision(
                target_state=decision_state,
                event=decision_event,
            )
        )

        notional_y, notional_x, gross_exposure, net_exposure = _mark_exposure(
            executed_state,
            units_y,
            units_x,
            current_y,
            current_x,
        )
        rows.append(
            (
                decision_state,
                executed_state,
                decision_event,
                execution_event,
                (
                    np.nan
                    if executed_order is None
                    else float(executed_order.decision_position)
                ),
                (
                    np.nan
                    if executed_order is None
                    else float(executed_order.due_position)
                ),
                current_beta,
                current_y,
                current_x,
                bool(observed_y_array[row_number]),
                bool(observed_x_array[row_number]),
                units_y,
                units_x,
                notional_y,
                notional_x,
                gross_exposure,
                net_exposure,
            )
        )

    result = pd.DataFrame.from_records(rows, columns=list(_OUTPUT_COLUMNS))
    result.index = index
    for column in (
        "execution_decision_row",
        "execution_due_row",
        "hedge_ratio",
        "price_y",
        "price_x",
        "units_y",
        "units_x",
        "notional_y",
        "notional_x",
        "gross_exposure",
        "net_exposure",
    ):
        result[column] = result[column].astype("float64")
    for column in ("observed_y", "observed_x"):
        result[column] = result[column].astype(bool)
    for column in (
        "decision_state",
        "executed_state",
        "decision_event",
        "execution_event",
    ):
        result[column] = result[column].astype(object)
    return result.loc[:, list(_OUTPUT_COLUMNS)]


def build_position_schedule(
    price_y: pd.Series,
    price_x: pd.Series,
    signals: pd.DataFrame,
    hedge_ratio: float | pd.Series,
    target_gross_notional: float,
    execution_lag: int = 1,
    *,
    observed_y: pd.Series | None = None,
    observed_x: pd.Series | None = None,
    hedge_ratio_lag: int = 1,
    suppress_terminal_entries: bool = False,
) -> pd.DataFrame:
    """Build a causal schedule of executed states, units, and raw exposures.

    State-changing orders become eligible after ``execution_lag`` rows.  If
    either execution-row price is imputed or missing, the order remains
    pending.  Entries additionally require an execution-available hedge ratio;
    closing known units does not.  A matured FLAT target cancels later unfilled
    entries while preserving the earliest close still required by the actual
    executed position.  At most one state-changing order executes on a valid
    row, preventing same-row reversal.

    Every execution, including a close, requires genuine observations for both
    current prices.  Forward-filled prices remain valid valuation marks only.
    A dynamic beta Series is treated as a close-derived posterior and shifted
    by ``hedge_ratio_lag``; scalars are treated as pre-frozen parameters.
    Reported notionals may still use imputed valuation marks.  When terminal
    entry suppression is enabled, no new flat-to-open execution occurs on the
    final row.

    This standalone scheduler executes caller-supplied decisions literally; it
    does not reinterpret their holding or cooldown clocks.  Z-score-driven
    integrated backtests use a separate execution-aware decision provider over
    the same core scheduler.
    """
    if type(suppress_terminal_entries) is not bool:
        raise TypeError("suppress_terminal_entries must be a bool.")
    inputs = _validated_execution_market_inputs(
        price_y,
        price_x,
        hedge_ratio,
        target_gross_notional,
        execution_lag,
        observed_y,
        observed_x,
        hedge_ratio_lag,
    )
    decisions = _validated_signals(signals)
    _require_matching_index(price_y.index, signals.index, "signals")

    def supplied_decision(
        row_number: int,
        _: _ExecutionFeedback,
    ) -> _ExecutionDecision:
        return _ExecutionDecision(
            target_state=decisions["decision_state"].iat[row_number],
            event=decisions["decision_event"].iat[row_number],
        )

    return _build_position_schedule_core(
        price_y.index,
        inputs,
        supplied_decision,
        suppress_terminal_entries=suppress_terminal_entries,
    )


def _build_execution_aware_position_schedule(
    price_y: pd.Series,
    price_x: pd.Series,
    zscore: pd.Series,
    hedge_ratio: float | pd.Series,
    target_gross_notional: float,
    *,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_holding_period: int | None,
    cooldown_period: int | None,
    missing_policy: str,
    execution_lag: int,
    observed_y: pd.Series | None,
    observed_x: pd.Series | None,
    hedge_ratio_lag: int,
    suppress_terminal_entries: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interleave z-score decisions with actual fills and execution clocks."""
    if type(suppress_terminal_entries) is not bool:
        raise TypeError("suppress_terminal_entries must be a bool.")
    values, policy = _validated_trade_signal_inputs(
        zscore,
        entry_z,
        exit_z,
        stop_z,
        max_holding_period=max_holding_period,
        cooldown_period=cooldown_period,
        missing_policy=missing_policy,
    )
    _require_matching_index(price_y.index, zscore.index, "zscore")
    inputs = _validated_execution_market_inputs(
        price_y,
        price_x,
        hedge_ratio,
        target_gross_notional,
        execution_lag,
        observed_y,
        observed_x,
        hedge_ratio_lag,
    )
    strategy = _ExecutionAwareStrategy(values, policy)
    positions = _build_position_schedule_core(
        price_y.index,
        inputs,
        strategy.decide,
        suppress_terminal_entries=suppress_terminal_entries,
    )
    signals = strategy.to_frame(zscore.index)
    _validated_signals(signals)
    return signals, positions


def _validated_position_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Copy and validate the state and unit contract from Milestone 6A."""
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("position_schedule must be a pandas DataFrame.")
    _require_causal_index(schedule.index, "position_schedule")

    required = {"executed_state", "execution_event", "units_y", "units_x"}
    missing = required.difference(schedule.columns)
    if missing:
        raise ValueError(
            "position_schedule is missing required columns: "
            f"{sorted(missing)}."
        )

    states = [
        _normalise_state(value, name="position_schedule['executed_state']")
        for value in schedule["executed_state"]
    ]
    events = [_normalise_event(value) for value in schedule["execution_event"]]
    units_y = _validated_market_series(
        schedule["units_y"],
        "position_schedule['units_y']",
        strictly_positive=False,
    )
    units_x = _validated_market_series(
        schedule["units_x"],
        "position_schedule['units_x']",
        strictly_positive=False,
    )
    if units_y.isna().any() or units_x.isna().any():
        raise ValueError("Position units must not contain missing values.")
    if "rebalance" in schedule.columns:
        rebalance_flags: list[bool] = []
        for value in schedule["rebalance"]:
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError("position_schedule['rebalance'] must contain bool values.")
            rebalance_flags.append(bool(value))
    else:
        rebalance_flags = [False] * len(schedule)

    previous_state = PositionState.FLAT.name
    previous_y = 0.0
    previous_x = 0.0
    for row_number, (state, event, current_y, current_x, is_rebalance) in enumerate(
        zip(states, events, units_y, units_x, rebalance_flags, strict=True)
    ):
        if row_number == 0 and (
            state != PositionState.FLAT.name
            or event != TradeEvent.NONE.value
            or current_y != 0.0
            or current_x != 0.0
            or is_rebalance
        ):
            raise ValueError(
                "The first position-schedule row must be an unexecuted flat "
                "position."
            )

        if state == PositionState.FLAT.name:
            if current_y != 0.0 or current_x != 0.0:
                raise ValueError(
                    f"Flat position has nonzero units at row position {row_number}."
                )
        elif state == PositionState.LONG_SPREAD.name:
            if current_y <= 0.0 or current_x >= 0.0:
                raise ValueError(
                    f"Long-spread units have invalid signs at row position {row_number}."
                )
        elif current_y >= 0.0 or current_x <= 0.0:
            raise ValueError(
                f"Short-spread units have invalid signs at row position {row_number}."
            )

        if is_rebalance:
            if (
                event != TradeEvent.NONE.value
                or previous_state == PositionState.FLAT.name
                or state != previous_state
                or (current_y == previous_y and current_x == previous_x)
            ):
                raise ValueError(
                    "Invalid hedge-rebalancing unit change at row position "
                    f"{row_number}."
                )
        elif event == TradeEvent.NONE.value:
            if (
                state != previous_state
                or current_y != previous_y
                or current_x != previous_x
            ):
                raise ValueError(
                    "Position state or units changed without an execution event "
                    f"at row position {row_number}."
                )
        elif event == TradeEvent.ENTER_LONG.value:
            if (
                previous_state != PositionState.FLAT.name
                or state != PositionState.LONG_SPREAD.name
            ):
                raise ValueError(
                    f"Invalid executed long entry at row position {row_number}."
                )
        elif event == TradeEvent.ENTER_SHORT.value:
            if (
                previous_state != PositionState.FLAT.name
                or state != PositionState.SHORT_SPREAD.name
            ):
                raise ValueError(
                    f"Invalid executed short entry at row position {row_number}."
                )
        elif event in _EXIT_EVENTS:
            if (
                previous_state == PositionState.FLAT.name
                or state != PositionState.FLAT.name
            ):
                raise ValueError(
                    f"Invalid executed exit at row position {row_number}."
                )

        previous_state = state
        previous_y = float(current_y)
        previous_x = float(current_x)

    result = pd.DataFrame(
        {
            "executed_state": pd.Series(states, index=schedule.index, dtype=object),
            "execution_event": pd.Series(events, index=schedule.index, dtype=object),
            "units_y": units_y,
            "units_x": units_x,
            "rebalance": pd.Series(
                rebalance_flags,
                index=schedule.index,
                dtype=bool,
            ),
        }
    )
    result.index = schedule.index
    return result


def _causal_cumulative(values: np.ndarray, name: str) -> np.ndarray:
    """Cumulatively sum until an unknown observation makes the total unknown."""
    cumulative = np.empty(len(values), dtype=float)
    running_total = 0.0
    total_is_known = True
    for row_number, value in enumerate(values):
        if np.isnan(value):
            total_is_known = False
            cumulative[row_number] = np.nan
            continue
        if not total_is_known:
            cumulative[row_number] = np.nan
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            running_total += value
        if not np.isfinite(running_total):
            raise ValueError(f"{name} overflowed at row position {row_number}.")
        cumulative[row_number] = running_total
    return cumulative


def _calculate_exposures(
    schedule: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
) -> pd.DataFrame:
    """Mark current post-execution units to current prices."""
    states = schedule["executed_state"].to_numpy(dtype=object)
    units_y = schedule["units_y"].to_numpy(dtype=float)
    units_x = schedule["units_x"].to_numpy(dtype=float)
    y_values = price_y.to_numpy(dtype=float)
    x_values = price_x.to_numpy(dtype=float)
    rows: list[tuple[float, ...]] = []

    for row_number, (state, unit_y, unit_x, mark_y, mark_x) in enumerate(
        zip(states, units_y, units_x, y_values, x_values, strict=True)
    ):
        if state == PositionState.FLAT.name:
            rows.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            market_value_y = unit_y * mark_y if np.isfinite(mark_y) else np.nan
            market_value_x = unit_x * mark_x if np.isfinite(mark_x) else np.nan
        finite_values = np.array([market_value_y, market_value_x], dtype=float)
        if np.isinf(finite_values).any():
            raise ValueError(
                "Market value overflowed at row position "
                f"{row_number}."
            )
        if np.isnan(finite_values).any():
            rows.append(
                (
                    float(market_value_y),
                    float(market_value_x),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            )
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            gross_exposure = abs(market_value_y) + abs(market_value_x)
            net_exposure = market_value_y + market_value_x
            long_exposure = max(market_value_y, 0.0) + max(market_value_x, 0.0)
            short_exposure = abs(min(market_value_y, 0.0)) + abs(
                min(market_value_x, 0.0)
            )
        aggregate = np.array(
            [gross_exposure, net_exposure, long_exposure, short_exposure],
            dtype=float,
        )
        if not np.isfinite(aggregate).all():
            raise ValueError(
                "Exposure accounting overflowed at row position "
                f"{row_number}."
            )
        rows.append(
            (
                float(market_value_y),
                float(market_value_x),
                float(gross_exposure),
                float(net_exposure),
                float(long_exposure),
                float(short_exposure),
            )
        )

    result = pd.DataFrame.from_records(
        rows,
        columns=[
            "market_value_y",
            "market_value_x",
            "gross_exposure",
            "net_exposure",
            "long_exposure",
            "short_exposure",
        ],
    )
    result.index = schedule.index
    return result.astype("float64")


def _interval_leg_pnl(
    prior_units: float,
    prior_price: float,
    current_price: float,
    *,
    leg: str,
    row_number: int,
) -> float:
    """Calculate one leg's interval P&L under the explicit missing-mark policy."""
    if prior_units == 0.0:
        return 0.0
    if np.isnan(prior_price) or np.isnan(current_price):
        return np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        result = prior_units * (current_price - prior_price)
    if not np.isfinite(result):
        raise ValueError(f"{leg} P&L overflowed at row position {row_number}.")
    return float(result)


def _decompose_realised_pnl(
    states: np.ndarray,
    gross_pnl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split gross P&L into open-trade unrealised and close-row realised P&L."""
    realised = np.zeros(len(gross_pnl), dtype=float)
    unrealised = np.zeros(len(gross_pnl), dtype=float)
    cumulative_realised = np.zeros(len(gross_pnl), dtype=float)
    previous_state = PositionState.FLAT.name
    open_trade_pnl = 0.0
    running_realised = 0.0
    realised_total_is_known = True

    for row_number, (state, interval_pnl) in enumerate(
        zip(states, gross_pnl, strict=True)
    ):
        if previous_state == PositionState.FLAT.name:
            if state != PositionState.FLAT.name:
                open_trade_pnl = 0.0
        elif state == previous_state:
            if np.isnan(open_trade_pnl) or np.isnan(interval_pnl):
                open_trade_pnl = np.nan
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    open_trade_pnl += interval_pnl
                if not np.isfinite(open_trade_pnl):
                    raise ValueError(
                        "Unrealised P&L overflowed at row position "
                        f"{row_number}."
                    )
        elif state == PositionState.FLAT.name:
            if np.isnan(open_trade_pnl) or np.isnan(interval_pnl):
                terminal_trade_pnl = np.nan
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    terminal_trade_pnl = open_trade_pnl + interval_pnl
                if not np.isfinite(terminal_trade_pnl):
                    raise ValueError(
                        "Realised P&L overflowed at row position "
                        f"{row_number}."
                    )
            realised[row_number] = terminal_trade_pnl
            if np.isnan(terminal_trade_pnl):
                realised_total_is_known = False
            elif realised_total_is_known:
                with np.errstate(over="ignore", invalid="ignore"):
                    running_realised += terminal_trade_pnl
                if not np.isfinite(running_realised):
                    raise ValueError(
                        "Cumulative realised P&L overflowed at row position "
                        f"{row_number}."
                    )
            open_trade_pnl = 0.0
        else:  # The validated schedule cannot reverse directly.
            raise RuntimeError("Position direction changed without a flat row.")

        unrealised[row_number] = (
            open_trade_pnl if state != PositionState.FLAT.name else 0.0
        )
        cumulative_realised[row_number] = (
            running_realised if realised_total_is_known else np.nan
        )
        previous_state = state

    return realised, unrealised, cumulative_realised


def calculate_position_pnl(
    position_schedule: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
) -> pd.DataFrame:
    """Calculate causal per-leg, gross, realised, and unrealised pair P&L.

    Row ``t`` uses units effective at ``t-1`` against the price change from
    ``t-1`` to ``t``.  Consequently an entry earns no pre-fill interval P&L,
    while an exit row includes the final interval earned by the position held
    before that exit.  The first row is zero because it has no prior interval.

    If a nonzero prior holding lacks either required mark, that leg and gross
    interval P&L are missing.  Row-level P&L resumes when both endpoint marks
    are again available, but cumulative P&L remains unknown after any missing
    held interval because the omitted wealth change cannot be reconstructed.
    """
    schedule = _validated_position_schedule(position_schedule)
    y_values = _validated_market_series(price_y, "price_y", strictly_positive=True)
    x_values = _validated_market_series(price_x, "price_x", strictly_positive=True)
    _require_matching_index(schedule.index, price_y.index, "price_y")
    _require_matching_index(schedule.index, price_x.index, "price_x")

    y_array = y_values.to_numpy(dtype=float)
    x_array = x_values.to_numpy(dtype=float)
    units_y = schedule["units_y"].to_numpy(dtype=float)
    units_x = schedule["units_x"].to_numpy(dtype=float)
    pnl_y = np.zeros(len(schedule), dtype=float)
    pnl_x = np.zeros(len(schedule), dtype=float)
    gross_pnl = np.zeros(len(schedule), dtype=float)

    for row_number in range(1, len(schedule)):
        pnl_y[row_number] = _interval_leg_pnl(
            units_y[row_number - 1],
            y_array[row_number - 1],
            y_array[row_number],
            leg="symbol_y",
            row_number=row_number,
        )
        pnl_x[row_number] = _interval_leg_pnl(
            units_x[row_number - 1],
            x_array[row_number - 1],
            x_array[row_number],
            leg="symbol_x",
            row_number=row_number,
        )
        if np.isnan(pnl_y[row_number]) or np.isnan(pnl_x[row_number]):
            gross_pnl[row_number] = np.nan
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                gross_pnl[row_number] = pnl_y[row_number] + pnl_x[row_number]
            if not np.isfinite(gross_pnl[row_number]):
                raise ValueError(
                    f"Gross P&L overflowed at row position {row_number}."
                )

    states = schedule["executed_state"].to_numpy(dtype=object)
    realised, unrealised, cumulative_realised = _decompose_realised_pnl(
        states,
        gross_pnl,
    )
    cumulative_gross = _causal_cumulative(gross_pnl, "Cumulative gross P&L")
    result = pd.DataFrame(
        {
            "pnl_y": pnl_y,
            "pnl_x": pnl_x,
            "gross_pnl": gross_pnl,
            "realised_pnl": realised,
            "unrealised_pnl": unrealised,
            "cumulative_realised_pnl": cumulative_realised,
            "cumulative_gross_pnl": cumulative_gross,
        },
        index=schedule.index,
    )
    result.index = position_schedule.index
    return result.astype("float64")


def calculate_strategy_returns(
    gross_pnl: pd.Series,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Calculate cumulative gross P&L, equity, and prior-equity returns.

    The first gross P&L observation must be zero and its return is defined as
    zero.  Later returns divide by prior-row equity.  A finite non-positive
    prior equity raises an error rather than permitting division by zero or a
    misleading return.  Missing interval P&L makes cumulative equity and all
    later returns unknown.
    """
    capital = _finite_positive_scalar(initial_capital, "initial_capital")
    values = _validated_market_series(
        gross_pnl,
        "gross_pnl",
        strictly_positive=False,
    )
    if len(values) and (np.isnan(values.iat[0]) or values.iat[0] != 0.0):
        raise ValueError("The first gross_pnl observation must be zero.")

    pnl_array = values.to_numpy(dtype=float)
    cumulative = _causal_cumulative(pnl_array, "Cumulative gross P&L")
    equity = np.full(len(values), np.nan, dtype=float)
    returns = np.full(len(values), np.nan, dtype=float)

    for row_number, cumulative_pnl in enumerate(cumulative):
        if np.isnan(cumulative_pnl):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            equity[row_number] = capital + cumulative_pnl
        if not np.isfinite(equity[row_number]):
            raise ValueError(
                f"Portfolio equity overflowed at row position {row_number}."
            )

    if len(values):
        returns[0] = 0.0
    for row_number in range(1, len(values)):
        prior_equity = equity[row_number - 1]
        if np.isnan(prior_equity):
            returns[row_number] = np.nan
            continue
        if prior_equity <= 0.0:
            raise ValueError(
                "Prior portfolio equity must be positive to calculate a return; "
                f"row position {row_number} has prior equity {prior_equity}."
            )
        if np.isnan(pnl_array[row_number]):
            returns[row_number] = np.nan
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            returns[row_number] = pnl_array[row_number] / prior_equity
        if not np.isfinite(returns[row_number]):
            raise ValueError(
                f"Strategy return is non-finite at row position {row_number}."
            )

    result = pd.DataFrame(
        {
            "cumulative_gross_pnl": cumulative,
            "portfolio_equity": equity,
            "strategy_return": returns,
        },
        index=values.index,
    )
    result.index = gross_pnl.index
    return result.astype("float64")


def build_pnl_schedule(
    position_schedule: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Build the complete gross marked-to-market accounting schedule.

    Exposures use current post-execution units and marks.  Interval P&L uses
    prior-row units, so executions never earn a price move that occurred before
    the fill.  No transaction, borrow, financing, or other costs are applied.
    """
    schedule = _validated_position_schedule(position_schedule)
    y_values = _validated_market_series(price_y, "price_y", strictly_positive=True)
    x_values = _validated_market_series(price_x, "price_x", strictly_positive=True)
    _require_matching_index(schedule.index, price_y.index, "price_y")
    _require_matching_index(schedule.index, price_x.index, "price_x")
    capital = _finite_positive_scalar(initial_capital, "initial_capital")

    exposures = _calculate_exposures(schedule, y_values, x_values)
    pnl = calculate_position_pnl(schedule, y_values, x_values)
    returns = calculate_strategy_returns(pnl["gross_pnl"], capital)
    result = pd.concat(
        [
            y_values.rename("price_y"),
            x_values.rename("price_x"),
            schedule[["units_y", "units_x"]],
            exposures,
            pnl[
                [
                    "pnl_y",
                    "pnl_x",
                    "gross_pnl",
                    "realised_pnl",
                    "unrealised_pnl",
                    "cumulative_realised_pnl",
                ]
            ],
            returns,
        ],
        axis=1,
        copy=False,
    )
    result.index = position_schedule.index
    return result.loc[:, list(_PNL_OUTPUT_COLUMNS)]


def _traded_leg_costs(
    delta_units: float,
    execution_price: float,
    commission_rate: float,
    slippage_rate: float,
    fixed_commission: float,
    *,
    leg: str,
    row_number: int,
) -> tuple[float, float, float, float]:
    """Return traded notional, variable commission, fixed fee, and slippage."""
    if delta_units == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    if not np.isfinite(execution_price):
        raise ValueError(
            f"{leg} execution price is missing on a traded row at position "
            f"{row_number}."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        traded_notional = abs(delta_units) * execution_price
        variable_commission = traded_notional * commission_rate
        slippage = traded_notional * slippage_rate
    calculated = np.array(
        [traded_notional, variable_commission, slippage],
        dtype=float,
    )
    if not np.isfinite(calculated).all():
        raise ValueError(
            f"{leg} transaction-cost calculation overflowed at row position "
            f"{row_number}."
        )
    return (
        float(traded_notional),
        float(variable_commission),
        fixed_commission,
        float(slippage),
    )


def calculate_transaction_costs(
    position_schedule: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fixed_commission_per_leg: float = 0.0,
) -> pd.DataFrame:
    """Calculate deterministic execution costs from actual unit changes.

    The first row assumes zero prior units.  Every later row compares current
    post-execution units with the previous row.  A leg with no unit change has
    exactly zero traded notional, variable commission, fixed commission, and
    slippage, irrespective of signal events.  Changed legs use current-row
    market prices; a missing execution price is rejected defensively.

    Basis-point commission and slippage are charged on absolute traded
    notional.  Slippage is represented as an adverse cost deduction rather
    than modifying stored market prices.
    """
    schedule = _validated_position_schedule(position_schedule)
    y_values = _validated_market_series(price_y, "price_y", strictly_positive=True)
    x_values = _validated_market_series(price_x, "price_x", strictly_positive=True)
    _require_matching_index(schedule.index, price_y.index, "price_y")
    _require_matching_index(schedule.index, price_x.index, "price_x")

    commission = _finite_nonnegative_scalar(commission_bps, "commission_bps")
    slippage = _finite_nonnegative_scalar(slippage_bps, "slippage_bps")
    fixed = _finite_nonnegative_scalar(
        fixed_commission_per_leg,
        "fixed_commission_per_leg",
    )
    commission_rate = commission / 10_000.0
    slippage_rate = slippage / 10_000.0

    units_y = schedule["units_y"].to_numpy(dtype=float)
    units_x = schedule["units_x"].to_numpy(dtype=float)
    y_array = y_values.to_numpy(dtype=float)
    x_array = x_values.to_numpy(dtype=float)
    rows: list[tuple[float, ...]] = []
    previous_y = 0.0
    previous_x = 0.0

    for row_number, (current_y, current_x, mark_y, mark_x) in enumerate(
        zip(units_y, units_x, y_array, x_array, strict=True)
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            delta_y = current_y - previous_y
            delta_x = current_x - previous_x
        if not np.isfinite([delta_y, delta_x]).all():
            raise ValueError(
                f"Unit delta overflowed at row position {row_number}."
            )
        delta_y = 0.0 if delta_y == 0.0 else float(delta_y)
        delta_x = 0.0 if delta_x == 0.0 else float(delta_x)
        traded_y, commission_y, fixed_y, slippage_y = _traded_leg_costs(
            delta_y,
            mark_y,
            commission_rate,
            slippage_rate,
            fixed,
            leg="symbol_y",
            row_number=row_number,
        )
        traded_x, commission_x, fixed_x, slippage_x = _traded_leg_costs(
            delta_x,
            mark_x,
            commission_rate,
            slippage_rate,
            fixed,
            leg="symbol_x",
            row_number=row_number,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            commission_cost = commission_y + commission_x + fixed_y + fixed_x
            slippage_cost = slippage_y + slippage_x
            transaction_cost = commission_cost + slippage_cost
        totals = np.array(
            [commission_cost, slippage_cost, transaction_cost],
            dtype=float,
        )
        if not np.isfinite(totals).all():
            raise ValueError(
                "Aggregate transaction cost overflowed at row position "
                f"{row_number}."
            )
        if bool((totals < 0.0).any()):  # Defensive against future cost models.
            raise RuntimeError("Transaction costs must be non-negative.")
        rows.append(
            (
                delta_y,
                delta_x,
                traded_y,
                traded_x,
                commission_y,
                commission_x,
                fixed_y,
                fixed_x,
                float(commission_cost),
                slippage_y,
                slippage_x,
                float(slippage_cost),
                float(transaction_cost),
            )
        )
        previous_y = current_y
        previous_x = current_x

    result = pd.DataFrame.from_records(
        rows,
        columns=[
            "delta_units_y",
            "delta_units_x",
            "traded_notional_y",
            "traded_notional_x",
            "commission_y",
            "commission_x",
            "fixed_commission_y",
            "fixed_commission_x",
            "commission_cost",
            "slippage_y",
            "slippage_x",
            "slippage_cost",
            "transaction_cost",
        ],
    )
    result.index = position_schedule.index
    transaction_array = result["transaction_cost"].to_numpy(dtype=float)
    result["cumulative_transaction_cost"] = _causal_cumulative(
        transaction_array,
        "Cumulative transaction cost",
    )
    return result.astype("float64")


def _validated_transaction_costs(
    transaction_costs: pd.DataFrame | pd.Series,
) -> pd.Series:
    """Return an independent non-negative transaction-cost Series."""
    if isinstance(transaction_costs, pd.DataFrame):
        if not transaction_costs.index.is_unique:
            raise ValueError("transaction_costs must have a unique index.")
        if "transaction_cost" not in transaction_costs.columns:
            raise ValueError(
                "transaction_costs is missing required column: transaction_cost."
            )
        source = transaction_costs["transaction_cost"]
    elif isinstance(transaction_costs, pd.Series):
        source = transaction_costs
    else:
        raise TypeError("transaction_costs must be a pandas DataFrame or Series.")

    values = _validated_market_series(
        source,
        "transaction_cost",
        strictly_positive=False,
    )
    if values.isna().any():
        raise ValueError("transaction_cost must not contain missing values.")
    if bool((values < 0.0).any()):
        raise ValueError("transaction_cost must be non-negative.")
    values.name = "transaction_cost"
    return values


def apply_execution_costs(
    pnl_schedule: pd.DataFrame,
    transaction_costs: pd.DataFrame | pd.Series,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Deduct execution costs and calculate cumulative net wealth and returns.

    If gross P&L is missing, net P&L remains missing even when the row's cost is
    known.  Cumulative net P&L and equity therefore remain unknown afterward,
    matching the gross accounting policy.  The first-row net return uses
    initial capital as its denominator if a nonzero first-row cost is ever
    supplied; normal Milestone 6A schedules start flat and therefore return
    zero on the first row.
    """
    if not isinstance(pnl_schedule, pd.DataFrame):
        raise TypeError("pnl_schedule must be a pandas DataFrame.")
    if not pnl_schedule.index.is_unique:
        raise ValueError("pnl_schedule must have a unique index.")
    if "gross_pnl" not in pnl_schedule.columns:
        raise ValueError("pnl_schedule is missing required column: gross_pnl.")

    capital = _finite_positive_scalar(initial_capital, "initial_capital")
    gross = _validated_market_series(
        pnl_schedule["gross_pnl"],
        "gross_pnl",
        strictly_positive=False,
    )
    if len(gross) and (np.isnan(gross.iat[0]) or gross.iat[0] != 0.0):
        raise ValueError("The first gross_pnl observation must be zero.")
    costs = _validated_transaction_costs(transaction_costs)
    _require_matching_index(pnl_schedule.index, costs.index, "transaction_costs")

    gross_array = gross.to_numpy(dtype=float)
    cost_array = costs.to_numpy(dtype=float)
    net_pnl = np.full(len(gross), np.nan, dtype=float)
    for row_number, (gross_value, cost_value) in enumerate(
        zip(gross_array, cost_array, strict=True)
    ):
        if np.isnan(gross_value):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            net_pnl[row_number] = gross_value - cost_value
        if not np.isfinite(net_pnl[row_number]):
            raise ValueError(f"Net P&L overflowed at row position {row_number}.")

    cumulative_cost = _causal_cumulative(
        cost_array,
        "Cumulative transaction cost",
    )
    cumulative_net = _causal_cumulative(net_pnl, "Cumulative net P&L")
    gross_returns = calculate_strategy_returns(gross, capital)
    net_equity = np.full(len(gross), np.nan, dtype=float)
    net_returns = np.full(len(gross), np.nan, dtype=float)

    for row_number, cumulative_value in enumerate(cumulative_net):
        if np.isnan(cumulative_value):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            net_equity[row_number] = capital + cumulative_value
        if not np.isfinite(net_equity[row_number]):
            raise ValueError(
                f"Net portfolio equity overflowed at row position {row_number}."
            )

    if len(gross) and not np.isnan(net_pnl[0]):
        net_returns[0] = net_pnl[0] / capital
    for row_number in range(1, len(gross)):
        prior_equity = net_equity[row_number - 1]
        if np.isnan(prior_equity):
            continue
        if prior_equity <= 0.0:
            raise ValueError(
                "Prior net portfolio equity must be positive to calculate a "
                f"return; row position {row_number} has prior equity "
                f"{prior_equity}."
            )
        if np.isnan(net_pnl[row_number]):
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            net_returns[row_number] = net_pnl[row_number] / prior_equity
        if not np.isfinite(net_returns[row_number]):
            raise ValueError(
                f"Net strategy return is non-finite at row position {row_number}."
            )

    result = pd.DataFrame(
        {
            "gross_pnl": gross_array,
            "transaction_cost": cost_array,
            "cumulative_transaction_cost": cumulative_cost,
            "net_pnl": net_pnl,
            "cumulative_gross_pnl": gross_returns["cumulative_gross_pnl"],
            "cumulative_net_pnl": cumulative_net,
            "portfolio_equity": gross_returns["portfolio_equity"],
            "net_portfolio_equity": net_equity,
            "strategy_return": gross_returns["strategy_return"],
            "net_strategy_return": net_returns,
        },
        index=pnl_schedule.index,
    )
    result.index = pnl_schedule.index
    return result.astype("float64")


def build_net_pnl_schedule(
    position_schedule: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
    initial_capital: float = 1_000_000.0,
    *,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fixed_commission_per_leg: float = 0.0,
) -> pd.DataFrame:
    """Build gross and net accounting with deterministic execution costs."""
    gross = build_pnl_schedule(
        position_schedule,
        price_y,
        price_x,
        initial_capital,
    )
    costs = calculate_transaction_costs(
        position_schedule,
        price_y,
        price_x,
        commission_bps,
        slippage_bps,
        fixed_commission_per_leg,
    )
    net = apply_execution_costs(gross, costs, initial_capital)
    result = pd.concat(
        [
            gross[["price_y", "price_x", "units_y", "units_x"]],
            costs,
            net.drop(
                columns=["transaction_cost", "cumulative_transaction_cost"]
            ),
        ],
        axis=1,
        copy=False,
    )
    result.index = position_schedule.index
    return result.loc[:, list(_NET_PNL_OUTPUT_COLUMNS)]


def _aligned_nonnegative_rate(
    rate: float | pd.Series,
    index: pd.Index,
    name: str,
) -> pd.Series:
    """Return a finite non-negative scalar or exactly aligned rate Series."""
    if isinstance(rate, pd.Series):
        values = _validated_market_series(rate, name, strictly_positive=False)
        _require_matching_index(index, rate.index, name)
        if values.isna().any():
            raise ValueError(f"{name} must not contain missing values.")
        if bool((values < 0.0).any()):
            raise ValueError(f"{name} must be non-negative.")
        return values
    normalised = _finite_nonnegative_scalar(rate, name)
    return pd.Series(
        np.full(len(index), normalised, dtype=float),
        index=index,
        name=name,
    )


def calculate_borrow_costs(
    pnl_schedule: pd.DataFrame,
    borrow_rate_y: float | pd.Series = 0.0,
    borrow_rate_x: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Accrue annualised short-borrow fees from prior-row short exposure.

    A dynamic rate observed at ``t-1`` applies to interval ``t-1`` to ``t``.
    The first row is zero.  Long and flat legs have zero borrow cost.  A short
    leg with an unavailable prior market value has unknown borrow cost rather
    than an assumed zero charge.
    """
    if not isinstance(pnl_schedule, pd.DataFrame):
        raise TypeError("pnl_schedule must be a pandas DataFrame.")
    if not pnl_schedule.index.is_unique:
        raise ValueError("pnl_schedule must have a unique index.")
    required = {"units_y", "units_x", "market_value_y", "market_value_x"}
    missing = required.difference(pnl_schedule.columns)
    if missing:
        raise ValueError(
            f"pnl_schedule is missing required columns: {sorted(missing)}."
        )

    periods = _positive_integer(periods_per_year, "periods_per_year")
    units_y = _validated_market_series(
        pnl_schedule["units_y"],
        "units_y",
        strictly_positive=False,
    )
    units_x = _validated_market_series(
        pnl_schedule["units_x"],
        "units_x",
        strictly_positive=False,
    )
    values_y = _validated_market_series(
        pnl_schedule["market_value_y"],
        "market_value_y",
        strictly_positive=False,
    )
    values_x = _validated_market_series(
        pnl_schedule["market_value_x"],
        "market_value_x",
        strictly_positive=False,
    )
    if units_y.isna().any() or units_x.isna().any():
        raise ValueError("Position units must not contain missing values.")
    rates_y = _aligned_nonnegative_rate(
        borrow_rate_y,
        pnl_schedule.index,
        "borrow_rate_y",
    )
    rates_x = _aligned_nonnegative_rate(
        borrow_rate_x,
        pnl_schedule.index,
        "borrow_rate_x",
    )

    unit_y_array = units_y.to_numpy(dtype=float)
    unit_x_array = units_x.to_numpy(dtype=float)
    value_y_array = values_y.to_numpy(dtype=float)
    value_x_array = values_x.to_numpy(dtype=float)
    rate_y_array = rates_y.to_numpy(dtype=float)
    rate_x_array = rates_x.to_numpy(dtype=float)
    cost_y = np.zeros(len(pnl_schedule), dtype=float)
    cost_x = np.zeros(len(pnl_schedule), dtype=float)
    total = np.zeros(len(pnl_schedule), dtype=float)

    for row_number in range(1, len(pnl_schedule)):
        if unit_y_array[row_number - 1] < 0.0:
            prior_value_y = value_y_array[row_number - 1]
            if np.isnan(prior_value_y):
                cost_y[row_number] = np.nan
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    cost_y[row_number] = (
                        max(-prior_value_y, 0.0)
                        * rate_y_array[row_number - 1]
                        / periods
                    )
        if unit_x_array[row_number - 1] < 0.0:
            prior_value_x = value_x_array[row_number - 1]
            if np.isnan(prior_value_x):
                cost_x[row_number] = np.nan
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    cost_x[row_number] = (
                        max(-prior_value_x, 0.0)
                        * rate_x_array[row_number - 1]
                        / periods
                    )
        if np.isinf([cost_y[row_number], cost_x[row_number]]).any():
            raise ValueError(
                f"Borrow cost overflowed at row position {row_number}."
            )
        if np.isnan(cost_y[row_number]) or np.isnan(cost_x[row_number]):
            total[row_number] = np.nan
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                total[row_number] = cost_y[row_number] + cost_x[row_number]
            if not np.isfinite(total[row_number]):
                raise ValueError(
                    f"Total borrow cost overflowed at row position {row_number}."
                )

    result = pd.DataFrame(
        {
            "borrow_cost_y": cost_y,
            "borrow_cost_x": cost_x,
            "borrow_cost": total,
            "cumulative_borrow_cost": _causal_cumulative(
                total,
                "Cumulative borrow cost",
            ),
        },
        index=pnl_schedule.index,
    )
    result.index = pnl_schedule.index
    return result.astype("float64")


def calculate_financing_costs(
    pnl_schedule: pd.DataFrame,
    financing_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Accrue a simple funding charge on prior-row gross exposure.

    This is a deliberately simplified gross-exposure charge, not a
    prime-broker-specific cash, margin, or collateral financing model.
    """
    if not isinstance(pnl_schedule, pd.DataFrame):
        raise TypeError("pnl_schedule must be a pandas DataFrame.")
    if not pnl_schedule.index.is_unique:
        raise ValueError("pnl_schedule must have a unique index.")
    if "gross_exposure" not in pnl_schedule.columns:
        raise ValueError("pnl_schedule is missing required column: gross_exposure.")

    rate = _finite_nonnegative_scalar(financing_rate, "financing_rate")
    periods = _positive_integer(periods_per_year, "periods_per_year")
    exposure = _validated_market_series(
        pnl_schedule["gross_exposure"],
        "gross_exposure",
        strictly_positive=False,
    )
    present = exposure.dropna()
    if bool((present < 0.0).any()):
        raise ValueError("gross_exposure must be non-negative.")

    exposure_array = exposure.to_numpy(dtype=float)
    financing = np.zeros(len(exposure), dtype=float)
    for row_number in range(1, len(exposure)):
        prior_exposure = exposure_array[row_number - 1]
        if np.isnan(prior_exposure):
            financing[row_number] = np.nan
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            financing[row_number] = prior_exposure * rate / periods
        if not np.isfinite(financing[row_number]):
            raise ValueError(
                f"Financing cost overflowed at row position {row_number}."
            )

    result = pd.DataFrame(
        {
            "financing_cost": financing,
            "cumulative_financing_cost": _causal_cumulative(
                financing,
                "Cumulative financing cost",
            ),
        },
        index=pnl_schedule.index,
    )
    result.index = pnl_schedule.index
    return result.astype("float64")


def calculate_rebalancing_costs(
    position_schedule: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
    hedge_ratio: float | pd.Series | None = None,
    target_gross_notional: float | None = None,
    *,
    rebalance: bool = False,
    rebalance_threshold: float = 0.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fixed_commission_per_leg: float = 0.0,
    observed_y: pd.Series | None = None,
    observed_x: pd.Series | None = None,
    hedge_ratio_lag: int = 1,
) -> pd.DataFrame:
    """Overlay causal hedge rebalancing and reuse normal execution costs.

    The relative trigger compares the latest execution-available beta with the
    beta used at the last entry or rebalance.  A dynamic posterior beta Series
    is shifted by ``hedge_ratio_lag`` so its observation and execution rows are
    distinct by default.  Missing or imputed execution prices suppress the
    rebalance; each later valid row is reconsidered causally.  State and
    execution-event values remain unchanged.
    """
    if type(rebalance) is not bool:  # Actual bool, not np.bool_, is required.
        raise TypeError("rebalance must be a bool.")
    threshold = _finite_nonnegative_scalar(
        rebalance_threshold,
        "rebalance_threshold",
    )
    base = _validated_position_schedule(position_schedule)
    y_values = _validated_market_series(price_y, "price_y", strictly_positive=True)
    x_values = _validated_market_series(price_x, "price_x", strictly_positive=True)
    _require_matching_index(base.index, price_y.index, "price_y")
    _require_matching_index(base.index, price_x.index, "price_x")
    observed_y_values = _validated_observed_mask(
        price_y,
        observed_y,
        "observed_y",
        position_schedule.get("observed_y"),
    )
    observed_x_values = _validated_observed_mask(
        price_x,
        observed_x,
        "observed_x",
        position_schedule.get("observed_x"),
    )
    beta_lag = _positive_integer(hedge_ratio_lag, "hedge_ratio_lag")

    beta_source = hedge_ratio
    beta_is_execution_available = False
    if beta_source is None and "hedge_ratio" in position_schedule.columns:
        beta_source = position_schedule["hedge_ratio"]
        beta_is_execution_available = True
    if isinstance(beta_source, pd.Series):
        if beta_is_execution_available:
            beta_values = _validated_market_series(
                beta_source,
                "hedge_ratio",
                strictly_positive=True,
            )
            _require_matching_index(base.index, beta_source.index, "hedge_ratio")
        else:
            beta_values = _available_hedge_ratios(
                beta_source,
                base.index,
                beta_lag,
            )
    elif beta_source is not None:
        beta = _finite_positive_scalar(beta_source, "hedge_ratio")
        beta_values = pd.Series(
            np.full(len(base), beta, dtype=float),
            index=base.index,
            name="hedge_ratio",
        )
    else:
        beta_values = pd.Series(
            np.full(len(base), np.nan, dtype=float),
            index=base.index,
            name="hedge_ratio",
        )
    if rebalance and beta_source is None:
        raise ValueError("hedge_ratio is required when rebalance=True.")
    if target_gross_notional is None:
        if rebalance:
            raise ValueError(
                "target_gross_notional is required when rebalance=True."
            )
        target = None
    else:
        target = _finite_positive_scalar(
            target_gross_notional,
            "target_gross_notional",
        )

    states = base["executed_state"].to_numpy(dtype=object)
    events = base["execution_event"].to_numpy(dtype=object)
    base_y = base["units_y"].to_numpy(dtype=float)
    base_x = base["units_x"].to_numpy(dtype=float)
    marks_y = y_values.to_numpy(dtype=float)
    marks_x = x_values.to_numpy(dtype=float)
    betas = beta_values.to_numpy(dtype=float)
    observed_y_array = observed_y_values.to_numpy(dtype=bool)
    observed_x_array = observed_x_values.to_numpy(dtype=bool)
    adjusted_y = np.zeros(len(base), dtype=float)
    adjusted_x = np.zeros(len(base), dtype=float)
    rebalance_flags = np.zeros(len(base), dtype=bool)
    rebalance_betas = np.full(len(base), np.nan, dtype=float)
    rebalance_delta_y = np.zeros(len(base), dtype=float)
    rebalance_delta_x = np.zeros(len(base), dtype=float)
    rebalance_decision_rows = np.full(len(base), np.nan, dtype=float)
    last_sizing_beta: float | None = None

    for row_number, (state, event) in enumerate(zip(states, events, strict=True)):
        if not rebalance:
            adjusted_y[row_number] = base_y[row_number]
            adjusted_x[row_number] = base_x[row_number]
            continue
        current_beta = betas[row_number]
        if event in {
            TradeEvent.ENTER_LONG.value,
            TradeEvent.ENTER_SHORT.value,
        }:
            if not np.isfinite(current_beta):
                raise ValueError(
                    "An executed entry requires a current hedge ratio when "
                    "rebalance=True."
                )
            adjusted_y[row_number] = base_y[row_number]
            adjusted_x[row_number] = base_x[row_number]
            last_sizing_beta = float(current_beta)
            continue
        if event in _EXIT_EVENTS or state == PositionState.FLAT.name:
            adjusted_y[row_number] = 0.0
            adjusted_x[row_number] = 0.0
            last_sizing_beta = None
            continue

        adjusted_y[row_number] = adjusted_y[row_number - 1]
        adjusted_x[row_number] = adjusted_x[row_number - 1]
        current_inputs_available = bool(
            np.isfinite(current_beta)
            and np.isfinite(marks_y[row_number])
            and np.isfinite(marks_x[row_number])
            and observed_y_array[row_number]
            and observed_x_array[row_number]
        )
        if not current_inputs_available or last_sizing_beta is None:
            continue

        beta_changed = current_beta != last_sizing_beta
        relative_change = abs(current_beta - last_sizing_beta) / abs(
            last_sizing_beta
        )
        if not beta_changed or relative_change < threshold:
            continue
        if target is None:  # Defensive: rebalance=True validates this above.
            raise RuntimeError("Missing rebalancing target gross notional.")
        desired = calculate_pair_units(
            state,
            marks_y[row_number],
            marks_x[row_number],
            current_beta,
            target,
        )
        desired_matches_current = bool(
            np.isclose(
                desired.units_y,
                adjusted_y[row_number],
                rtol=1e-12,
                atol=1e-12,
            )
            and np.isclose(
                desired.units_x,
                adjusted_x[row_number],
                rtol=1e-12,
                atol=1e-12,
            )
        )
        last_sizing_beta = float(current_beta)
        if desired_matches_current:
            continue

        rebalance_flags[row_number] = True
        rebalance_betas[row_number] = current_beta
        rebalance_decision_rows[row_number] = row_number - beta_lag
        rebalance_delta_y[row_number] = (
            desired.units_y - adjusted_y[row_number]
        )
        rebalance_delta_x[row_number] = (
            desired.units_x - adjusted_x[row_number]
        )
        adjusted_y[row_number] = desired.units_y
        adjusted_x[row_number] = desired.units_x

    adjusted = pd.DataFrame(
        {
            "executed_state": base["executed_state"],
            "execution_event": base["execution_event"],
            "units_y": adjusted_y,
            "units_x": adjusted_x,
            "rebalance": rebalance_flags,
        },
        index=base.index,
    )
    costs = calculate_transaction_costs(
        adjusted,
        y_values,
        x_values,
        commission_bps,
        slippage_bps,
        fixed_commission_per_leg,
    )
    result = pd.concat(
        [
            adjusted,
            pd.Series(
                rebalance_betas,
                index=base.index,
                name="rebalance_beta",
            ),
            pd.Series(
                rebalance_delta_y,
                index=base.index,
                name="rebalance_delta_units_y",
            ),
            pd.Series(
                rebalance_delta_x,
                index=base.index,
                name="rebalance_delta_units_x",
            ),
            pd.Series(
                rebalance_decision_rows,
                index=base.index,
                name="rebalance_decision_row",
            ),
            costs,
        ],
        axis=1,
        copy=False,
    )
    result.index = position_schedule.index
    return result


def _equity_and_returns_after_carry(
    net_pnl: np.ndarray,
    initial_capital: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cumulative P&L, equity, and prior-equity returns after carry."""
    cumulative = _causal_cumulative(net_pnl, "Cumulative net P&L after carry")
    equity = np.full(len(net_pnl), np.nan, dtype=float)
    returns = np.full(len(net_pnl), np.nan, dtype=float)
    for row_number, cumulative_value in enumerate(cumulative):
        if np.isnan(cumulative_value):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            equity[row_number] = initial_capital + cumulative_value
        if not np.isfinite(equity[row_number]):
            raise ValueError(
                "Net equity after carry overflowed at row position "
                f"{row_number}."
            )
    if len(net_pnl) and not np.isnan(net_pnl[0]):
        returns[0] = net_pnl[0] / initial_capital
    for row_number in range(1, len(net_pnl)):
        prior_equity = equity[row_number - 1]
        if np.isnan(prior_equity):
            continue
        if prior_equity <= 0.0:
            raise ValueError(
                "Prior net equity after carry must be positive to calculate a "
                f"return at row position {row_number}."
            )
        if np.isnan(net_pnl[row_number]):
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            returns[row_number] = net_pnl[row_number] / prior_equity
        if not np.isfinite(returns[row_number]):
            raise ValueError(
                "Net return after carry is non-finite at row position "
                f"{row_number}."
            )
    return cumulative, equity, returns


def build_financed_pnl_schedule(
    position_schedule: pd.DataFrame,
    price_y: pd.Series,
    price_x: pd.Series,
    initial_capital: float = 1_000_000.0,
    *,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    fixed_commission_per_leg: float = 0.0,
    borrow_rate_y: float | pd.Series = 0.0,
    borrow_rate_x: float | pd.Series = 0.0,
    financing_rate: float = 0.0,
    periods_per_year: int = 252,
    rebalance: bool = False,
    hedge_ratio: float | pd.Series | None = None,
    target_gross_notional: float | None = None,
    rebalance_threshold: float = 0.0,
    observed_y: pd.Series | None = None,
    observed_x: pd.Series | None = None,
    hedge_ratio_lag: int = 1,
) -> pd.DataFrame:
    """Build execution, gross, transaction, carry, and financed net accounting."""
    capital = _finite_positive_scalar(initial_capital, "initial_capital")
    periods = _positive_integer(periods_per_year, "periods_per_year")
    rebalanced = calculate_rebalancing_costs(
        position_schedule,
        price_y,
        price_x,
        hedge_ratio,
        target_gross_notional,
        rebalance=rebalance,
        rebalance_threshold=rebalance_threshold,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        fixed_commission_per_leg=fixed_commission_per_leg,
        observed_y=observed_y,
        observed_x=observed_x,
        hedge_ratio_lag=hedge_ratio_lag,
    )
    adjusted_schedule = rebalanced[
        [
            "executed_state",
            "execution_event",
            "units_y",
            "units_x",
            "rebalance",
        ]
    ].copy()
    gross = build_pnl_schedule(
        adjusted_schedule,
        price_y,
        price_x,
        capital,
    )
    cost_columns = [
        "delta_units_y",
        "delta_units_x",
        "traded_notional_y",
        "traded_notional_x",
        "commission_y",
        "commission_x",
        "fixed_commission_y",
        "fixed_commission_x",
        "commission_cost",
        "slippage_y",
        "slippage_x",
        "slippage_cost",
        "transaction_cost",
        "cumulative_transaction_cost",
    ]
    costs = rebalanced[cost_columns]
    net = apply_execution_costs(gross, costs, capital)
    base_net = pd.concat(
        [
            gross[["price_y", "price_x", "units_y", "units_x"]],
            costs,
            net.drop(
                columns=["transaction_cost", "cumulative_transaction_cost"]
            ),
        ],
        axis=1,
        copy=False,
    ).loc[:, list(_NET_PNL_OUTPUT_COLUMNS)]

    borrow = calculate_borrow_costs(
        gross,
        borrow_rate_y,
        borrow_rate_x,
        periods,
    )
    financing = calculate_financing_costs(gross, financing_rate, periods)
    borrow_array = borrow["borrow_cost"].to_numpy(dtype=float)
    financing_array = financing["financing_cost"].to_numpy(dtype=float)
    gross_array = gross["gross_pnl"].to_numpy(dtype=float)
    transaction_array = costs["transaction_cost"].to_numpy(dtype=float)
    carry = np.full(len(gross), np.nan, dtype=float)
    financed_net = np.full(len(gross), np.nan, dtype=float)

    for row_number, (borrow_value, financing_value) in enumerate(
        zip(borrow_array, financing_array, strict=True)
    ):
        if np.isnan(borrow_value) or np.isnan(financing_value):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            carry[row_number] = borrow_value + financing_value
        if not np.isfinite(carry[row_number]):
            raise ValueError(
                f"Carry cost overflowed at row position {row_number}."
            )
        if np.isnan(gross_array[row_number]):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            financed_net[row_number] = (
                gross_array[row_number]
                - transaction_array[row_number]
                - carry[row_number]
            )
        if not np.isfinite(financed_net[row_number]):
            raise ValueError(
                "Net P&L after carry overflowed at row position "
                f"{row_number}."
            )

    cumulative_carry = _causal_cumulative(carry, "Cumulative carry cost")
    cumulative_financed, financed_equity, financed_returns = (
        _equity_and_returns_after_carry(financed_net, capital)
    )
    additions = pd.DataFrame(
        {
            "borrow_cost_y": borrow["borrow_cost_y"],
            "borrow_cost_x": borrow["borrow_cost_x"],
            "borrow_cost": borrow["borrow_cost"],
            "financing_cost": financing["financing_cost"],
            "carry_cost": carry,
            "cumulative_borrow_cost": borrow["cumulative_borrow_cost"],
            "cumulative_financing_cost": financing[
                "cumulative_financing_cost"
            ],
            "cumulative_carry_cost": cumulative_carry,
            "rebalance": rebalanced["rebalance"].astype(bool),
            "rebalance_beta": rebalanced["rebalance_beta"],
            "rebalance_delta_units_y": rebalanced[
                "rebalance_delta_units_y"
            ],
            "rebalance_delta_units_x": rebalanced[
                "rebalance_delta_units_x"
            ],
            "rebalance_decision_row": rebalanced[
                "rebalance_decision_row"
            ],
            "net_pnl_after_carry": financed_net,
            "cumulative_net_pnl_after_carry": cumulative_financed,
            "net_equity_after_carry": financed_equity,
            "net_return_after_carry": financed_returns,
        },
        index=gross.index,
    )
    result = pd.concat([base_net, additions], axis=1, copy=False)
    result.index = position_schedule.index
    return result.loc[:, list(_FINANCED_OUTPUT_COLUMNS)]


def _liquidation_series(
    supplied: pd.Series | None,
    schedule: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Resolve and validate a current-data Series used for final liquidation."""
    if supplied is None:
        if column not in schedule.columns:
            raise ValueError(
                f"{column} is required to force-liquidate an open position."
            )
        source = schedule[column]
    else:
        source = supplied
    values = _validated_market_series(source, column, strictly_positive=True)
    _require_matching_index(schedule.index, source.index, column)
    return values


def force_liquidate_open_position(
    position_schedule: pd.DataFrame,
    price_y: pd.Series | None = None,
    price_x: pd.Series | None = None,
    hedge_ratio: float | pd.Series | None = None,
    *,
    force_liquidation: bool = True,
    observed_y: pd.Series | None = None,
    observed_x: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a copied schedule with an optional final-row forced close.

    Only the final row is changed.  Its prior-row holdings still drive final
    interval P&L when the returned schedule is passed through the existing
    accounting functions.  Normal 6C transaction costs then arise from the
    final unit deltas; no penalty fee is introduced.

    Final prices are taken from explicit aligned inputs when supplied,
    otherwise from matching columns in ``position_schedule``.  Both final
    prices must be genuine observations rather than forward-filled valuation
    marks.  Known units can be set to zero without a new hedge ratio.  The
    ``hedge_ratio`` argument is retained for API compatibility but never
    overwrites the schedule's causal, execution-available beta metadata.
    """
    if type(force_liquidation) is not bool:
        raise TypeError("force_liquidation must be a bool.")
    validated = _validated_position_schedule(position_schedule)
    result = position_schedule.copy(deep=True)
    if not force_liquidation or result.empty:
        return result

    final_row = len(result) - 1
    if validated["executed_state"].iat[final_row] == PositionState.FLAT.name:
        return result

    previous_state = (
        PositionState.FLAT.name
        if final_row == 0
        else validated["executed_state"].iat[final_row - 1]
    )
    final_event = validated["execution_event"].iat[final_row]
    terminal_entry = bool(
        previous_state == PositionState.FLAT.name
        and final_event
        in {
            TradeEvent.ENTER_LONG.value,
            TradeEvent.ENTER_SHORT.value,
        }
    )
    if terminal_entry:
        # A position opened only on the terminal row has no executable holding
        # interval.  Suppress the entry rather than fabricate an entry and
        # forced exit at the same timestamp.
        result.iat[
            final_row, result.columns.get_loc("executed_state")
        ] = PositionState.FLAT.name
        result.iat[
            final_row, result.columns.get_loc("execution_event")
        ] = TradeEvent.NONE.value
        for column in (
            "units_y",
            "units_x",
            "notional_y",
            "notional_x",
            "gross_exposure",
            "net_exposure",
            "rebalance_delta_units_y",
            "rebalance_delta_units_x",
        ):
            if column in result.columns:
                result.iat[final_row, result.columns.get_loc(column)] = 0.0
        if "rebalance" in result.columns:
            result.iat[final_row, result.columns.get_loc("rebalance")] = False
        for column in ("execution_decision_row", "execution_due_row"):
            if column in result.columns:
                result.iat[final_row, result.columns.get_loc(column)] = np.nan
        _validated_position_schedule(result)
        result.index = position_schedule.index
        return result

    y_values = _liquidation_series(price_y, result, "price_y")
    x_values = _liquidation_series(price_x, result, "price_x")
    y_source = price_y if price_y is not None else result["price_y"]
    x_source = price_x if price_x is not None else result["price_x"]
    y_observed = _validated_observed_mask(
        y_source,
        observed_y,
        "observed_y",
        result.get("observed_y"),
    )
    x_observed = _validated_observed_mask(
        x_source,
        observed_x,
        "observed_x",
        result.get("observed_x"),
    )
    final_y = y_values.iat[final_row]
    final_x = x_values.iat[final_row]
    if (
        not np.isfinite(final_y)
        or not np.isfinite(final_x)
        or not y_observed.iat[final_row]
        or not x_observed.iat[final_row]
    ):
        raise ValueError(
            "Final prices must be genuine observed values to force-liquidate "
            "an open position."
        )

    # If the schedule stores market inputs, keep its final row consistent with
    # the explicit current observations used to authorize the close.
    for column, current_value in (
        ("price_y", float(final_y)),
        ("price_x", float(final_x)),
    ):
        if column in result.columns:
            result.iat[final_row, result.columns.get_loc(column)] = current_value

    result.iat[
        final_row, result.columns.get_loc("executed_state")
    ] = PositionState.FLAT.name
    result.iat[
        final_row, result.columns.get_loc("execution_event")
    ] = _FORCED_EXIT_EVENT
    result.iat[final_row, result.columns.get_loc("units_y")] = 0.0
    result.iat[final_row, result.columns.get_loc("units_x")] = 0.0
    for column in (
        "notional_y",
        "notional_x",
        "gross_exposure",
        "net_exposure",
    ):
        if column in result.columns:
            result.iat[final_row, result.columns.get_loc(column)] = 0.0
    if "rebalance" in result.columns:
        result.iat[final_row, result.columns.get_loc("rebalance")] = False
    for column in ("execution_decision_row", "execution_due_row"):
        if column in result.columns:
            result.iat[final_row, result.columns.get_loc(column)] = np.nan
    if "observed_y" in result.columns:
        result.iat[final_row, result.columns.get_loc("observed_y")] = True
    if "observed_x" in result.columns:
        result.iat[final_row, result.columns.get_loc("observed_x")] = True

    _validated_position_schedule(result)
    result.index = position_schedule.index
    return result


def _validated_ledger_inputs(
    accounting_schedule: pd.DataFrame,
    position_schedule: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Validate and copy the financed and executed-state ledger inputs."""
    if not isinstance(accounting_schedule, pd.DataFrame):
        raise TypeError("accounting_schedule must be a pandas DataFrame.")
    if not accounting_schedule.index.is_unique:
        raise ValueError("accounting_schedule must have a unique index.")

    required_accounting = {
        "price_y",
        "price_x",
        "units_y",
        "units_x",
        "gross_pnl",
        "commission_cost",
        "slippage_cost",
        "transaction_cost",
        "borrow_cost",
        "financing_cost",
        "carry_cost",
        "net_pnl_after_carry",
        "cumulative_net_pnl_after_carry",
        "rebalance",
    }
    missing = required_accounting.difference(accounting_schedule.columns)
    if missing:
        raise ValueError(
            "accounting_schedule is missing required financed columns: "
            f"{sorted(missing)}."
        )

    if position_schedule is None:
        metadata = accounting_schedule
    else:
        if not isinstance(position_schedule, pd.DataFrame):
            raise TypeError("position_schedule must be a pandas DataFrame.")
        if not position_schedule.index.is_unique:
            raise ValueError("position_schedule must have a unique index.")
        _require_matching_index(
            accounting_schedule.index,
            position_schedule.index,
            "position_schedule",
        )
        metadata = position_schedule
    required_metadata = {"executed_state", "execution_event", "hedge_ratio"}
    missing_metadata = required_metadata.difference(metadata.columns)
    if missing_metadata:
        raise ValueError(
            "position_schedule is missing required ledger metadata: "
            f"{sorted(missing_metadata)}."
        )

    accounting = accounting_schedule.copy(deep=True)
    validation_schedule = pd.DataFrame(
        {
            "executed_state": metadata["executed_state"],
            "execution_event": metadata["execution_event"],
            "units_y": accounting["units_y"],
            "units_x": accounting["units_x"],
            "rebalance": accounting["rebalance"],
        },
        index=accounting.index,
    )
    states = _validated_position_schedule(validation_schedule)
    hedge = _validated_market_series(
        metadata["hedge_ratio"],
        "position_schedule['hedge_ratio']",
        strictly_positive=True,
    )
    _require_matching_index(accounting.index, hedge.index, "hedge_ratio")

    numeric_columns = {
        "price_y": True,
        "price_x": True,
        "gross_pnl": False,
        "commission_cost": False,
        "slippage_cost": False,
        "transaction_cost": False,
        "borrow_cost": False,
        "financing_cost": False,
        "carry_cost": False,
        "net_pnl_after_carry": False,
        "cumulative_net_pnl_after_carry": False,
    }
    validated_numeric: dict[str, pd.Series] = {}
    for column, strictly_positive in numeric_columns.items():
        validated_numeric[column] = _validated_market_series(
            accounting[column],
            f"accounting_schedule['{column}']",
            strictly_positive=strictly_positive,
        )
        accounting[column] = validated_numeric[column]

    for column in (
        "commission_cost",
        "slippage_cost",
        "transaction_cost",
        "borrow_cost",
        "financing_cost",
        "carry_cost",
    ):
        present = accounting[column].dropna()
        if bool((present < 0.0).any()):
            raise ValueError(f"accounting_schedule['{column}'] must be non-negative.")

    commission = accounting["commission_cost"].to_numpy(dtype=float)
    slippage = accounting["slippage_cost"].to_numpy(dtype=float)
    transaction = accounting["transaction_cost"].to_numpy(dtype=float)
    borrow = accounting["borrow_cost"].to_numpy(dtype=float)
    financing = accounting["financing_cost"].to_numpy(dtype=float)
    carry = accounting["carry_cost"].to_numpy(dtype=float)
    gross = accounting["gross_pnl"].to_numpy(dtype=float)
    net = accounting["net_pnl_after_carry"].to_numpy(dtype=float)
    for row_number in range(len(accounting)):
        if np.isfinite([commission[row_number], slippage[row_number]]).all():
            expected_transaction = commission[row_number] + slippage[row_number]
            if not np.isclose(
                transaction[row_number], expected_transaction, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    "transaction_cost is inconsistent at row position "
                    f"{row_number}."
                )
        elif not np.isnan(transaction[row_number]):
            raise ValueError(
                "transaction_cost must remain unknown when a required cost "
                f"component is unknown at row position {row_number}."
            )
        if np.isfinite([borrow[row_number], financing[row_number]]).all():
            expected_carry = borrow[row_number] + financing[row_number]
            if not np.isclose(
                carry[row_number], expected_carry, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f"carry_cost is inconsistent at row position {row_number}."
                )
        elif not np.isnan(carry[row_number]):
            raise ValueError(
                "carry_cost must remain unknown when a required carry "
                f"component is unknown at row position {row_number}."
            )
        if np.isfinite(
            [gross[row_number], transaction[row_number], carry[row_number]]
        ).all():
            expected_net = (
                gross[row_number]
                - transaction[row_number]
                - carry[row_number]
            )
            if not np.isclose(
                net[row_number], expected_net, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    "net_pnl_after_carry is inconsistent at row position "
                    f"{row_number}."
                )
        elif not np.isnan(net[row_number]):
            raise ValueError(
                "net_pnl_after_carry must remain unknown when gross P&L or a "
                f"required cost is unknown at row position {row_number}."
            )

    accounting.index = accounting_schedule.index
    states.index = accounting_schedule.index
    hedge.index = accounting_schedule.index
    return accounting, states, hedge


def _trade_windows(
    states: pd.DataFrame,
) -> tuple[list[tuple[int, int]], int | None]:
    """Return completed inclusive row windows and any final open entry row."""
    completed: list[tuple[int, int]] = []
    open_entry: int | None = None
    previous_state = PositionState.FLAT.name
    for row_number, current_state in enumerate(states["executed_state"]):
        if previous_state == PositionState.FLAT.name and current_state != previous_state:
            if open_entry is not None:
                raise ValueError("A new trade began while another trade was open.")
            open_entry = row_number
        elif previous_state != PositionState.FLAT.name and current_state == PositionState.FLAT.name:
            if open_entry is None:
                raise ValueError("A trade closed when no trade was open.")
            completed.append((open_entry, row_number))
            open_entry = None
        previous_state = current_state
    return completed, open_entry


def _sum_preserving_unknown(values: pd.Series | np.ndarray) -> float:
    """Sum known values, returning NaN when any attributed value is unknown."""
    array = np.asarray(values, dtype=float)
    if np.isnan(array).any():
        return np.nan
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(array.sum(dtype=float))
    if not np.isfinite(total):
        raise ValueError("Attributed accounting total overflowed.")
    return total


def _exit_reason(event: str) -> TradeExitReason:
    """Map an executed close event to its canonical ledger reason."""
    reasons = {
        TradeEvent.EXIT_MEAN_REVERSION.value: TradeExitReason.MEAN_REVERSION,
        TradeEvent.EXIT_STOP.value: TradeExitReason.STOP,
        TradeEvent.EXIT_TIME.value: TradeExitReason.TIME,
        _FORCED_EXIT_EVENT: TradeExitReason.END_OF_BACKTEST,
    }
    try:
        return reasons[event]
    except KeyError as exc:
        raise ValueError(f"Unsupported executed exit event: {event!r}.") from exc


def _finite_trade_endpoint(value: Any, name: str) -> float:
    """Validate an observable positive trade endpoint price or hedge ratio."""
    return _finite_positive_scalar(value, name)


def _optional_trade_endpoint(value: Any, name: str) -> float:
    """Return a positive endpoint value or NaN when causal metadata is absent."""
    if pd.isna(value):
        return np.nan
    return _finite_trade_endpoint(value, name)


def build_trade_ledger(
    accounting_schedule: pd.DataFrame,
    position_schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attribute existing financed accounting rows to completed trades.

    ``accounting_schedule`` supplies valued holdings, P&L, and costs, while
    ``position_schedule`` supplies executed states, events, and hedge ratios.
    The latter may be omitted only when those metadata columns were explicitly
    joined to the accounting frame by the caller.  An open final trade is not
    fabricated as completed; callers may first use
    :func:`force_liquidate_open_position` and rebuild accounting.
    """
    accounting, states, hedge = _validated_ledger_inputs(
        accounting_schedule,
        position_schedule,
    )
    completed, _ = _trade_windows(states)
    records: list[TradeRecord] = []

    for trade_number, (entry_row, exit_row) in enumerate(completed, start=1):
        side = states["executed_state"].iat[entry_row]
        entry_event = states["execution_event"].iat[entry_row]
        exit_event = states["execution_event"].iat[exit_row]
        entry_price_y = _finite_trade_endpoint(
            accounting["price_y"].iat[entry_row],
            "entry_price_y",
        )
        entry_price_x = _finite_trade_endpoint(
            accounting["price_x"].iat[entry_row],
            "entry_price_x",
        )
        exit_price_y = _finite_trade_endpoint(
            accounting["price_y"].iat[exit_row],
            "exit_price_y",
        )
        exit_price_x = _finite_trade_endpoint(
            accounting["price_x"].iat[exit_row],
            "exit_price_x",
        )
        entry_beta = _finite_trade_endpoint(
            hedge.iat[entry_row],
            "entry_hedge_ratio",
        )
        exit_beta = _optional_trade_endpoint(
            hedge.iat[exit_row],
            "exit_hedge_ratio",
        )
        entry_units_y = float(accounting["units_y"].iat[entry_row])
        entry_units_x = float(accounting["units_x"].iat[entry_row])
        with np.errstate(over="ignore", invalid="ignore"):
            entry_gross = (
                abs(entry_units_y * entry_price_y)
                + abs(entry_units_x * entry_price_x)
            )
        entry_gross = _finite_trade_endpoint(
            entry_gross,
            "entry_gross_notional",
        )
        attributed = accounting.iloc[entry_row : exit_row + 1]
        gross_pnl = _sum_preserving_unknown(attributed["gross_pnl"])
        commission_cost = _sum_preserving_unknown(
            attributed["commission_cost"]
        )
        slippage_cost = _sum_preserving_unknown(attributed["slippage_cost"])
        transaction_cost = _sum_preserving_unknown(
            attributed["transaction_cost"]
        )
        borrow_cost = _sum_preserving_unknown(attributed["borrow_cost"])
        financing_cost = _sum_preserving_unknown(
            attributed["financing_cost"]
        )
        carry_cost = _sum_preserving_unknown(attributed["carry_cost"])
        if np.isnan(carry_cost) or np.isnan(transaction_cost):
            total_cost = np.nan
        else:
            total_cost = transaction_cost + carry_cost
        if np.isnan(gross_pnl) or np.isnan(total_cost):
            net_pnl = np.nan
            trade_return = np.nan
        else:
            net_pnl = gross_pnl - total_cost
            trade_return = net_pnl / entry_gross
        if np.isinf([total_cost, net_pnl, trade_return]).any():
            raise ValueError(f"Trade {trade_number} attribution overflowed.")

        records.append(
            TradeRecord(
                trade_id=trade_number,
                side=side,
                entry_row=entry_row,
                exit_row=exit_row,
                entry_index=accounting.index[entry_row],
                exit_index=accounting.index[exit_row],
                entry_event=entry_event,
                exit_event=exit_event,
                exit_reason=_exit_reason(exit_event).value,
                entry_price_y=entry_price_y,
                entry_price_x=entry_price_x,
                exit_price_y=exit_price_y,
                exit_price_x=exit_price_x,
                entry_units_y=entry_units_y,
                entry_units_x=entry_units_x,
                exit_units_y=float(accounting["units_y"].iat[exit_row - 1]),
                exit_units_x=float(accounting["units_x"].iat[exit_row - 1]),
                entry_hedge_ratio=entry_beta,
                exit_hedge_ratio=exit_beta,
                holding_period_rows=exit_row - entry_row,
                entry_gross_notional=entry_gross,
                gross_pnl=gross_pnl,
                commission_cost=commission_cost,
                slippage_cost=slippage_cost,
                transaction_cost=transaction_cost,
                borrow_cost=borrow_cost,
                financing_cost=financing_cost,
                carry_cost=carry_cost,
                total_cost=total_cost,
                net_pnl=net_pnl,
                return_on_entry_gross_notional=trade_return,
                forced_exit=exit_event == _FORCED_EXIT_EVENT,
            )
        )

    result = pd.DataFrame.from_records(
        [asdict(record) for record in records],
        columns=list(_TRADE_LEDGER_COLUMNS),
    )
    if not result.empty:
        result["trade_id"] = result["trade_id"].astype("int64")
        result["entry_row"] = result["entry_row"].astype("int64")
        result["exit_row"] = result["exit_row"].astype("int64")
        result["holding_period_rows"] = result["holding_period_rows"].astype(
            "int64"
        )
        result["forced_exit"] = result["forced_exit"].astype(bool)
    return result.loc[:, list(_TRADE_LEDGER_COLUMNS)]


def _validated_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Validate the public completed-trade ledger contract."""
    if not isinstance(ledger, pd.DataFrame):
        raise TypeError("ledger must be a pandas DataFrame.")
    missing = set(_TRADE_LEDGER_COLUMNS).difference(ledger.columns)
    if missing:
        raise ValueError(f"ledger is missing required columns: {sorted(missing)}.")
    result = ledger.loc[:, list(_TRADE_LEDGER_COLUMNS)].copy(deep=True)
    expected_ids = list(range(1, len(result) + 1))
    ids: list[int] = []
    for value in result["trade_id"]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("ledger trade_id values must be integers.")
        ids.append(int(value))
    if ids != expected_ids:
        raise ValueError("ledger trade_id values must be unique and consecutive from 1.")
    for value in result["forced_exit"]:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError("ledger forced_exit values must be Boolean.")

    numeric = (
        "gross_pnl",
        "transaction_cost",
        "carry_cost",
        "net_pnl",
    )
    for column in numeric:
        result[column] = _validated_market_series(
            result[column],
            f"ledger['{column}']",
            strictly_positive=False,
        )
    return result


def _assert_canonical_ledger(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Require the supplied ledger to equal the authoritative rebuilt ledger."""
    if not actual.index.equals(expected.index):
        raise ValueError("Backtest invariant failed: ledger index is not canonical.")
    if len(actual) != len(expected):
        raise ValueError("Backtest invariant failed: ledger trade count is not canonical.")

    for column in _TRADE_LEDGER_COLUMNS:
        if column in _TRADE_LEDGER_FLOAT_COLUMNS:
            try:
                actual_values = actual[column].to_numpy(dtype=float)
                expected_values = expected[column].to_numpy(dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Backtest invariant failed: ledger {column} is not numeric."
                ) from exc
            if not np.array_equal(
                np.isnan(actual_values),
                np.isnan(expected_values),
            ):
                raise ValueError(
                    f"Backtest invariant failed: ledger {column} has inconsistent NaNs."
                )
            known = ~np.isnan(expected_values)
            if (
                not np.isfinite(actual_values[known]).all()
                or not np.allclose(
                    actual_values[known],
                    expected_values[known],
                    rtol=rtol,
                    atol=atol,
                )
            ):
                raise ValueError(
                    f"Backtest invariant failed: ledger {column} is not canonical."
                )
        elif not actual[column].equals(expected[column]):
            raise ValueError(
                f"Backtest invariant failed: ledger {column} is not canonical."
            )


def _assert_reconciliation_matches(
    stored: LedgerReconciliation,
    recomputed: LedgerReconciliation,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Require a stored reconciliation to match current result components."""
    if not isinstance(stored, LedgerReconciliation):
        raise TypeError("reconciliation must be a LedgerReconciliation.")
    for field_info in fields(LedgerReconciliation):
        stored_value = getattr(stored, field_info.name)
        recomputed_value = getattr(recomputed, field_info.name)
        if isinstance(recomputed_value, Real) and not isinstance(
            recomputed_value,
            (bool, np.bool_),
        ):
            stored_number = float(stored_value)
            recomputed_number = float(recomputed_value)
            matches = bool(
                (np.isnan(stored_number) and np.isnan(recomputed_number))
                or (
                    np.isfinite(stored_number)
                    and np.isfinite(recomputed_number)
                    and np.isclose(
                        stored_number,
                        recomputed_number,
                        rtol=rtol,
                        atol=atol,
                    )
                )
            )
        else:
            matches = stored_value == recomputed_value
        if not matches:
            raise ValueError(
                "Backtest invariant failed: stored reconciliation is stale "
                f"for field {field_info.name}."
            )


def _rows_total(
    accounting: pd.DataFrame,
    windows: list[tuple[int, int]],
    column: str,
) -> float:
    """Aggregate one accounting column across disjoint inclusive windows."""
    if not windows:
        return 0.0
    values = np.concatenate(
        [
            accounting[column].iloc[entry : exit + 1].to_numpy(dtype=float)
            for entry, exit in windows
        ]
    )
    return _sum_preserving_unknown(values)


def _optional_close(
    left: float,
    right: float,
    *,
    rtol: float,
    atol: float,
) -> bool | None:
    """Compare known totals, using None when both sides are unknown."""
    left_unknown = bool(np.isnan(left))
    right_unknown = bool(np.isnan(right))
    if left_unknown and right_unknown:
        return None
    if left_unknown or right_unknown:
        return False
    return bool(np.isclose(left, right, rtol=rtol, atol=atol))


def reconcile_trade_ledger(
    ledger: pd.DataFrame,
    accounting_schedule: pd.DataFrame,
    position_schedule: pd.DataFrame | None = None,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> LedgerReconciliation:
    """Reconcile completed ledger totals with their causal accounting rows.

    A final open trade is reported separately and deliberately prevents a
    final cumulative comparison.  Unknown completed-trade accounting produces
    ``UNKNOWN_ACCOUNTING`` rather than a false successful reconciliation.
    """
    relative_tolerance = _finite_nonnegative_scalar(rtol, "rtol")
    absolute_tolerance = _finite_nonnegative_scalar(atol, "atol")
    accounting, states, _ = _validated_ledger_inputs(
        accounting_schedule,
        position_schedule,
    )
    completed, open_entry = _trade_windows(states)
    checked_ledger = _validated_ledger(ledger)

    ledger_gross = _sum_preserving_unknown(checked_ledger["gross_pnl"])
    ledger_transaction = _sum_preserving_unknown(
        checked_ledger["transaction_cost"]
    )
    ledger_carry = _sum_preserving_unknown(checked_ledger["carry_cost"])
    ledger_net = _sum_preserving_unknown(checked_ledger["net_pnl"])
    schedule_gross = _rows_total(accounting, completed, "gross_pnl")
    schedule_transaction = _rows_total(
        accounting,
        completed,
        "transaction_cost",
    )
    schedule_carry = _rows_total(accounting, completed, "carry_cost")
    schedule_net = _rows_total(
        accounting,
        completed,
        "net_pnl_after_carry",
    )

    gross_match = _optional_close(
        ledger_gross,
        schedule_gross,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    transaction_match = _optional_close(
        ledger_transaction,
        schedule_transaction,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    carry_match = _optional_close(
        ledger_carry,
        schedule_carry,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    net_match = _optional_close(
        ledger_net,
        schedule_net,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    component_matches = (
        gross_match,
        transaction_match,
        carry_match,
        net_match,
    )
    count_matches = len(checked_ledger) == len(completed)
    completed_match = count_matches and all(value is True for value in component_matches)
    fully_reconcilable = all(value is not None for value in component_matches)

    if accounting.empty:
        final_cumulative = 0.0
    else:
        final_cumulative = float(
            accounting["cumulative_net_pnl_after_carry"].iat[-1]
        )
    if open_entry is None:
        final_match = _optional_close(
            ledger_net,
            final_cumulative,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        open_gross = 0.0
        open_transaction = 0.0
        open_carry = 0.0
        open_net = 0.0
    else:
        final_match = None
        open_gross = _sum_preserving_unknown(
            accounting["gross_pnl"].iloc[open_entry:]
        )
        open_transaction = _sum_preserving_unknown(
            accounting["transaction_cost"].iloc[open_entry:]
        )
        open_carry = _sum_preserving_unknown(
            accounting["carry_cost"].iloc[open_entry:]
        )
        open_net = _sum_preserving_unknown(
            accounting["net_pnl_after_carry"].iloc[open_entry:]
        )

    explicit_mismatch = (
        not count_matches
        or any(value is False for value in component_matches)
        or final_match is False
    )
    if explicit_mismatch:
        status = "MISMATCH"
    elif not fully_reconcilable or final_match is None and open_entry is None:
        status = "UNKNOWN_ACCOUNTING"
    elif open_entry is not None:
        status = "OPEN_TRADE"
    else:
        status = "RECONCILED"

    return LedgerReconciliation(
        status=status,
        completed_trade_count=len(completed),
        has_open_trade=open_entry is not None,
        fully_reconcilable=fully_reconcilable,
        completed_totals_match=completed_match,
        final_accounting_match=final_match,
        gross_pnl_match=gross_match,
        transaction_cost_match=transaction_match,
        carry_cost_match=carry_match,
        net_pnl_match=net_match,
        ledger_gross_pnl=ledger_gross,
        schedule_completed_gross_pnl=schedule_gross,
        ledger_transaction_cost=ledger_transaction,
        schedule_completed_transaction_cost=schedule_transaction,
        ledger_carry_cost=ledger_carry,
        schedule_completed_carry_cost=schedule_carry,
        ledger_net_pnl=ledger_net,
        schedule_completed_net_pnl=schedule_net,
        open_trade_gross_pnl=open_gross,
        open_trade_transaction_cost=open_transaction,
        open_trade_carry_cost=open_carry,
        open_trade_net_pnl=open_net,
        final_cumulative_net_pnl_after_carry=final_cumulative,
    )


def _assert_accounting_identity(
    actual: pd.Series,
    expected: pd.Series,
    name: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Raise when an accounting identity differs, including its NaN mask."""
    actual_values = actual.to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    if not np.array_equal(np.isnan(actual_values), np.isnan(expected_values)):
        raise ValueError(f"Backtest invariant failed: {name} has inconsistent NaNs.")
    known = np.isfinite(actual_values) & np.isfinite(expected_values)
    if not np.allclose(
        actual_values[known],
        expected_values[known],
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError(f"Backtest invariant failed: {name}.")


def _assert_cumulative_cost(
    values: pd.Series,
    name: str,
    *,
    atol: float,
) -> None:
    """Require a non-decreasing cumulative cost with causal NaN propagation."""
    array = values.to_numpy(dtype=float)
    missing = np.flatnonzero(np.isnan(array))
    known_stop = int(missing[0]) if len(missing) else len(array)
    if len(missing) and not np.isnan(array[known_stop:]).all():
        raise ValueError(
            f"Backtest invariant failed: {name} resumed after becoming unknown."
        )
    if known_stop > 1 and bool((np.diff(array[:known_stop]) < -atol).any()):
        raise ValueError(f"Backtest invariant failed: {name} decreased.")


def validate_backtest_invariants(
    result: BacktestResult,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Validate cross-stage accounting, state, timing, and ledger invariants.

    The function reports the first violated contract and never repairs output.
    Unknown accounting is accepted only when the corresponding identity also
    remains unknown under the conservative missing-data policy.
    """
    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult.")
    relative_tolerance = _finite_nonnegative_scalar(rtol, "rtol")
    absolute_tolerance = _finite_nonnegative_scalar(atol, "atol")
    lag = _positive_integer(result.execution_lag, "execution_lag")
    if type(result.forced_liquidation_requested) is not bool:
        raise TypeError("forced_liquidation_requested must be a bool.")
    if type(result.forced_liquidation_applied) is not bool:
        raise TypeError("forced_liquidation_applied must be a bool.")
    if not isinstance(result.research_metadata, BacktestResearchMetadata):
        raise TypeError("research_metadata must be BacktestResearchMetadata.")
    if result.research_metadata.upstream_provenance_validated:
        raise ValueError(
            "Backtest invariant failed: upstream causal provenance cannot be "
            "claimed by the execution runner."
        )

    signals = result.signals
    positions = result.positions
    accounting = result.accounting
    ledger = result.ledger
    for frame, name in (
        (signals, "signals"),
        (positions, "positions"),
        (accounting, "accounting"),
        (ledger, "ledger"),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"result.{name} must be a pandas DataFrame.")
        if not frame.index.is_unique:
            raise ValueError(f"result.{name} must have a unique index.")
    _require_causal_index(signals.index, "result.signals")
    _require_causal_index(positions.index, "result.positions")
    _require_causal_index(accounting.index, "result.accounting")
    _require_matching_index(positions.index, signals.index, "signals")
    _require_matching_index(positions.index, accounting.index, "accounting")
    _validated_signals(signals)

    required_accounting = {
        "pnl_y",
        "pnl_x",
        "gross_pnl",
        "commission_cost",
        "slippage_cost",
        "transaction_cost",
        "cumulative_transaction_cost",
        "borrow_cost",
        "financing_cost",
        "carry_cost",
        "cumulative_carry_cost",
        "net_pnl_after_carry",
        "gross_exposure",
        "long_exposure",
        "short_exposure",
    }
    missing = required_accounting.difference(accounting.columns)
    if missing:
        raise ValueError(
            f"result.accounting is missing invariant columns: {sorted(missing)}."
        )
    required_positions = {
        "executed_state",
        "execution_event",
        "decision_event",
        "units_y",
        "units_x",
        "rebalance",
        "rebalance_decision_row",
        "observed_y",
        "observed_x",
        "hedge_ratio",
    }
    missing_positions = required_positions.difference(positions.columns)
    if missing_positions:
        raise ValueError(
            f"result.positions is missing invariant columns: {sorted(missing_positions)}."
        )
    if "event" not in signals.columns:
        raise ValueError("result.signals is missing required column: event.")

    _assert_accounting_identity(
        accounting["gross_pnl"],
        accounting["pnl_y"] + accounting["pnl_x"],
        "gross_pnl = pnl_y + pnl_x",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    _assert_accounting_identity(
        accounting["transaction_cost"],
        accounting["commission_cost"] + accounting["slippage_cost"],
        "transaction_cost = commission_cost + slippage_cost",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    _assert_accounting_identity(
        accounting["carry_cost"],
        accounting["borrow_cost"] + accounting["financing_cost"],
        "carry_cost = borrow_cost + financing_cost",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    _assert_accounting_identity(
        accounting["net_pnl_after_carry"],
        (
            accounting["gross_pnl"]
            - accounting["transaction_cost"]
            - accounting["carry_cost"]
        ),
        "net_pnl_after_carry = gross_pnl - transaction_cost - carry_cost",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    _assert_cumulative_cost(
        accounting["cumulative_transaction_cost"],
        "cumulative_transaction_cost",
        atol=absolute_tolerance,
    )
    _assert_cumulative_cost(
        accounting["cumulative_carry_cost"],
        "cumulative_carry_cost",
        atol=absolute_tolerance,
    )

    actual_schedule = _validated_position_schedule(
        positions[
            [
                "executed_state",
                "execution_event",
                "units_y",
                "units_x",
                "rebalance",
            ]
        ]
    )
    trade_or_rebalance = (
        positions["execution_event"].ne(TradeEvent.NONE.value)
        | positions["rebalance"].astype(bool)
    )
    if not bool(
        positions.loc[
            trade_or_rebalance,
            ["observed_y", "observed_x"],
        ].astype(bool).all().all()
    ):
        raise ValueError(
            "Backtest invariant failed: execution used an imputed price."
        )
    for row_number in np.flatnonzero(positions["rebalance"].to_numpy(dtype=bool)):
        decision_row = positions["rebalance_decision_row"].iat[row_number]
        if not np.isfinite(decision_row) or int(decision_row) >= row_number:
            raise ValueError(
                "Backtest invariant failed: rebalance decision and execution "
                "rows are not distinct."
            )
    flat = actual_schedule["executed_state"].eq(PositionState.FLAT.name)
    if not bool(
        actual_schedule.loc[flat, ["units_y", "units_x"]].eq(0.0).all().all()
    ):
        raise ValueError("Backtest invariant failed: flat positions have nonzero units.")
    for column in ("gross_exposure", "long_exposure", "short_exposure"):
        if bool((accounting[column].dropna() < -absolute_tolerance).any()):
            raise ValueError(
                f"Backtest invariant failed: {column} contains negative values."
            )

    if result.forced_liquidation_requested and not positions.empty:
        final_units = positions[["units_y", "units_x"]].iloc[-1].to_numpy(
            dtype=float
        )
        if not np.allclose(final_units, 0.0, rtol=0.0, atol=absolute_tolerance):
            raise ValueError(
                "Backtest invariant failed: forced liquidation left open units."
            )

    checked_ledger = _validated_ledger(ledger)
    canonical_ledger = build_trade_ledger(accounting, positions)
    _assert_canonical_ledger(
        checked_ledger,
        canonical_ledger,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    recomputed_reconciliation = reconcile_trade_ledger(
        checked_ledger,
        accounting,
        positions,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    _assert_reconciliation_matches(
        result.reconciliation,
        recomputed_reconciliation,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    for row in checked_ledger.itertuples(index=False):
        known = np.isfinite(
            [row.gross_pnl, row.total_cost, row.net_pnl]
        ).all()
        if known and not np.isclose(
            row.net_pnl,
            row.gross_pnl - row.total_cost,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise ValueError(
                f"Backtest invariant failed: ledger trade {row.trade_id} net P&L."
            )

    final_is_flat = bool(
        positions.empty
        or positions["executed_state"].iat[-1] == PositionState.FLAT.name
    )
    if final_is_flat:
        expected_status = (
            "RECONCILED"
            if result.reconciliation.fully_reconcilable
            else "UNKNOWN_ACCOUNTING"
        )
        if result.reconciliation.status != expected_status:
            raise ValueError(
                "Backtest invariant failed: closed-ledger reconciliation status."
            )
    elif result.reconciliation.status != "OPEN_TRADE":
        raise ValueError(
            "Backtest invariant failed: open trade was not reported explicitly."
        )

    signal_events = signals["event"].astype(object).to_numpy()
    execution_events = positions["execution_event"].astype(object).to_numpy()
    for event in (
        TradeEvent.ENTER_LONG.value,
        TradeEvent.ENTER_SHORT.value,
        TradeEvent.EXIT_MEAN_REVERSION.value,
        TradeEvent.EXIT_STOP.value,
        TradeEvent.EXIT_TIME.value,
    ):
        executed_count = 0
        for row_number, executed_event in enumerate(execution_events):
            if executed_event != event:
                continue
            executed_count += 1
            if row_number < lag:
                raise ValueError(
                    "Backtest invariant failed: trade executed before its lag."
                )
            due_decisions = int(
                np.count_nonzero(signal_events[: row_number - lag + 1] == event)
            )
            if executed_count > due_decisions:
                raise ValueError(
                    "Backtest invariant failed: execution has no lagged decision."
                )

    previous_state = PositionState.FLAT.name
    for row_number, state in enumerate(actual_schedule["executed_state"]):
        if (
            previous_state != PositionState.FLAT.name
            and state != PositionState.FLAT.name
            and state != previous_state
        ):
            raise ValueError(
                "Backtest invariant failed: same-row position reversal at row "
                f"{row_number}."
            )
        previous_state = state


def run_pair_backtest(
    price_y: pd.Series,
    price_x: pd.Series,
    hedge_ratio: float | pd.Series,
    target_gross_notional: float,
    *,
    zscore: pd.Series | None = None,
    signals: pd.DataFrame | None = None,
    initial_capital: float = 1_000_000.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_period: int | None = None,
    cooldown_period: int | None = None,
    missing_policy: str = "hold",
    execution_lag: int = 1,
    commission_bps: float = 0.0,
    fixed_commission_per_leg: float = 0.0,
    slippage_bps: float = 0.0,
    borrow_rate_y: float | pd.Series = 0.0,
    borrow_rate_x: float | pd.Series = 0.0,
    financing_rate: float = 0.0,
    periods_per_year: int = 252,
    rebalance: bool = False,
    rebalance_threshold: float = 0.0,
    force_liquidation: bool = False,
    observed_y: pd.Series | None = None,
    observed_x: pd.Series | None = None,
    hedge_ratio_lag: int = 1,
) -> BacktestResult:
    """Run the complete causal one-pair workflow through ledger reconciliation.

    Exactly one of ``zscore`` and ``signals`` is required.  Z-score-driven runs
    interleave actual fills and current decisions: holding starts at the entry
    fill, cooldown starts at the exit fill, and the current decision is made
    only after processing an older order due on the current row.  Caller-
    supplied signal frames remain explicit external decisions and are executed
    literally; their upstream clock semantics cannot be reconstructed here.

    Supplied z-scores, signals, and hedge ratios are assumed causal; this
    accounting runner cannot prove how upstream arrays were estimated.  A
    dynamic hedge-ratio Series is treated as close-derived and becomes
    execution-available only after ``hedge_ratio_lag`` rows.  Beta-weighted
    gross-notional sizing is retained and is not dollar-neutral unless beta
    equals one.  Forward-filled prices may mark holdings but cannot execute
    when their observed masks are supplied or attached by market-data cleaning.
    """
    if (zscore is None) == (signals is None):
        raise ValueError("Exactly one of zscore or signals must be supplied.")
    lag = _positive_integer(execution_lag, "execution_lag")
    if zscore is not None:
        signal_frame, base_positions = _build_execution_aware_position_schedule(
            price_y,
            price_x,
            zscore,
            hedge_ratio,
            target_gross_notional,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_holding_period=max_holding_period,
            cooldown_period=cooldown_period,
            missing_policy=missing_policy,
            execution_lag=lag,
            observed_y=observed_y,
            observed_x=observed_x,
            hedge_ratio_lag=hedge_ratio_lag,
            suppress_terminal_entries=force_liquidation,
        )
    else:
        if not isinstance(signals, pd.DataFrame):
            raise TypeError("signals must be a pandas DataFrame.")
        signal_frame = signals.copy(deep=True)
        base_positions = build_position_schedule(
            price_y,
            price_x,
            signal_frame,
            hedge_ratio,
            target_gross_notional,
            execution_lag=lag,
            observed_y=observed_y,
            observed_x=observed_x,
            hedge_ratio_lag=hedge_ratio_lag,
            suppress_terminal_entries=force_liquidation,
        )

    accounting_kwargs = {
        "initial_capital": initial_capital,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "fixed_commission_per_leg": fixed_commission_per_leg,
        "borrow_rate_y": borrow_rate_y,
        "borrow_rate_x": borrow_rate_x,
        "financing_rate": financing_rate,
        "periods_per_year": periods_per_year,
        "rebalance": rebalance,
        "hedge_ratio": hedge_ratio,
        "target_gross_notional": target_gross_notional,
        "rebalance_threshold": rebalance_threshold,
        "observed_y": observed_y,
        "observed_x": observed_x,
        "hedge_ratio_lag": hedge_ratio_lag,
    }
    accounting = build_financed_pnl_schedule(
        base_positions,
        price_y,
        price_x,
        **accounting_kwargs,
    )

    final_positions = force_liquidate_open_position(
        base_positions,
        price_y,
        price_x,
        hedge_ratio,
        force_liquidation=force_liquidation,
        observed_y=observed_y,
        observed_x=observed_x,
    )
    forced_applied = bool(
        not final_positions.empty
        and final_positions["execution_event"].iat[-1] == _FORCED_EXIT_EVENT
        and base_positions["execution_event"].iat[-1] != _FORCED_EXIT_EVENT
    )
    if forced_applied:
        accounting = build_financed_pnl_schedule(
            final_positions,
            price_y,
            price_x,
            **accounting_kwargs,
        )

    rebalanced = calculate_rebalancing_costs(
        final_positions,
        price_y,
        price_x,
        hedge_ratio,
        target_gross_notional,
        rebalance=rebalance,
        rebalance_threshold=rebalance_threshold,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        fixed_commission_per_leg=fixed_commission_per_leg,
        observed_y=observed_y,
        observed_x=observed_x,
        hedge_ratio_lag=hedge_ratio_lag,
    )
    actual_schedule = rebalanced[
        [
            "executed_state",
            "execution_event",
            "units_y",
            "units_x",
            "rebalance",
        ]
    ].copy()
    gross_detail = build_pnl_schedule(
        actual_schedule,
        price_y,
        price_x,
        initial_capital,
    )

    integrated_positions = final_positions.copy(deep=True)
    integrated_positions["units_y"] = rebalanced["units_y"]
    integrated_positions["units_x"] = rebalanced["units_x"]
    integrated_positions["notional_y"] = gross_detail["market_value_y"]
    integrated_positions["notional_x"] = gross_detail["market_value_x"]
    integrated_positions["gross_exposure"] = gross_detail["gross_exposure"]
    integrated_positions["net_exposure"] = gross_detail["net_exposure"]
    for column in (
        "rebalance",
        "rebalance_beta",
        "rebalance_delta_units_y",
        "rebalance_delta_units_x",
        "rebalance_decision_row",
    ):
        integrated_positions[column] = rebalanced[column]
    integrated_positions.index = price_y.index

    detail_columns = [
        "market_value_y",
        "market_value_x",
        "gross_exposure",
        "net_exposure",
        "long_exposure",
        "short_exposure",
        "pnl_y",
        "pnl_x",
        "realised_pnl",
        "unrealised_pnl",
        "cumulative_realised_pnl",
    ]
    integrated_accounting = pd.concat(
        [accounting.copy(deep=True), gross_detail[detail_columns]],
        axis=1,
        copy=False,
    )
    integrated_accounting.index = price_y.index

    ledger = build_trade_ledger(integrated_accounting, integrated_positions)
    reconciliation = reconcile_trade_ledger(
        ledger,
        integrated_accounting,
        integrated_positions,
    )
    result = BacktestResult(
        signals=signal_frame.copy(deep=True),
        positions=integrated_positions.copy(deep=True),
        accounting=integrated_accounting.copy(deep=True),
        ledger=ledger.copy(deep=True),
        reconciliation=reconciliation,
        research_metadata=BacktestResearchMetadata(
            upstream_inputs_assumed_causal=True,
            upstream_provenance_validated=False,
            warning=(
                "Supplied z-scores, signals, and hedge ratios are assumed causal; "
                "run_pair_backtest does not validate their upstream estimation "
                "or selection provenance. Z-score-driven holding and cooldown "
                "clocks follow actual fills; caller-supplied signal frames remain "
                "external decisions whose clock semantics are not reinterpreted."
            ),
            price_policy=(
                "Forward-filled prices may value holdings but genuine observed "
                "prices on both legs are required for execution."
            ),
            hedge_ratio_policy=(
                "Dynamic close-derived posterior beta values become available "
                f"after {hedge_ratio_lag} row(s)."
            ),
            sizing_policy="beta_weighted_gross_notional",
            dollar_neutrality_note=(
                "Beta-weighted sizing is not dollar-neutral unless beta == 1."
            ),
        ),
        forced_liquidation_requested=force_liquidation,
        forced_liquidation_applied=forced_applied,
        execution_lag=lag,
    )
    validate_backtest_invariants(result)
    return result
