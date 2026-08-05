"""Causal execution scheduling and gross-normalized pair unit sizing.

This milestone deliberately stops before portfolio accounting.  It converts
state-changing signal decisions into lagged executions, sizes the two legs on
the actual execution row, and carries raw units forward without calculating
P&L, returns, costs, or a trade ledger.
"""

from __future__ import annotations

from collections import deque
from numbers import Integral, Real
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .signals import PositionState, TradeEvent


__all__ = [
    "PairUnits",
    "lag_trade_decisions",
    "calculate_pair_units",
    "build_position_schedule",
]


_OUTPUT_COLUMNS = (
    "decision_state",
    "executed_state",
    "decision_event",
    "execution_event",
    "hedge_ratio",
    "price_y",
    "price_x",
    "units_y",
    "units_x",
    "notional_y",
    "notional_x",
    "gross_exposure",
    "net_exposure",
)

_EXIT_EVENTS = {
    TradeEvent.EXIT_MEAN_REVERSION.value,
    TradeEvent.EXIT_STOP.value,
    TradeEvent.EXIT_TIME.value,
}


class PairUnits(NamedTuple):
    """Signed fractional units of the dependent and explanatory symbols."""

    units_y: float
    units_x: float


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
        try:
            return TradeEvent(value).value
        except ValueError as exc:
            raise ValueError(f"event contains an unknown trade event: {value!r}.") from exc
    raise TypeError("event values must be TradeEvent members or canonical strings.")


def _validated_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Copy and validate the state/event contract emitted by signal generation."""
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals must be a pandas DataFrame.")
    if not signals.index.is_unique:
        raise ValueError("signals must have a unique index.")

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
    if not series.index.is_unique:
        raise ValueError(f"{name} must have a unique index.")

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


def build_position_schedule(
    price_y: pd.Series,
    price_x: pd.Series,
    signals: pd.DataFrame,
    hedge_ratio: float | pd.Series,
    target_gross_notional: float,
    execution_lag: int = 1,
) -> pd.DataFrame:
    """Build a causal schedule of executed states, units, and raw exposures.

    State-changing orders become eligible after ``execution_lag`` rows.  If
    either execution-row price or the execution-row hedge ratio is missing,
    the order joins a FIFO queue and remains pending without expiry.  At most
    one state-changing order executes on a valid row; this prevents the
    scheduler from fabricating a same-row reversal.  Pending orders left after
    the final row remain unexecuted.

    Every execution, including a close, requires both current prices and the
    current positive hedge ratio.  Open-position units are sized once using
    those execution-row inputs and then remain unchanged until another
    execution.  Reported notionals mark those carried units at current prices,
    so gross exposure generally drifts away from its entry target.  Missing
    marks remain missing rather than being backfilled.
    """
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

    decisions = _validated_signals(signals)
    _require_matching_index(price_y.index, signals.index, "signals")
    lagged = lag_trade_decisions(signals, execution_lag=lag)

    if isinstance(hedge_ratio, pd.Series):
        beta_values = _validated_market_series(
            hedge_ratio,
            "hedge_ratio",
            strictly_positive=True,
        )
        _require_matching_index(price_y.index, hedge_ratio.index, "hedge_ratio")
    else:
        beta = _finite_positive_scalar(hedge_ratio, "hedge_ratio")
        beta_values = pd.Series(
            np.full(len(price_y), beta, dtype=float),
            index=price_y.index,
            name="hedge_ratio",
        )

    pending_orders: deque[tuple[str, str]] = deque()
    executed_state = PositionState.FLAT.name
    units_y = 0.0
    units_x = 0.0
    rows: list[tuple[Any, ...]] = []

    y_array = y_values.to_numpy(dtype=float)
    x_array = x_values.to_numpy(dtype=float)
    beta_array = beta_values.to_numpy(dtype=float)

    for row_number in range(len(price_y)):
        due_event = lagged["due_event"].iat[row_number]
        if due_event != TradeEvent.NONE.value:
            due_state = lagged["due_state"].iat[row_number]
            if due_state is None:  # Defensive: validation/lagging make this unreachable.
                raise RuntimeError("A due event has no associated target state.")
            pending_orders.append((due_state, due_event))

        # Discard a redundant target without presenting it as an execution.
        while pending_orders and pending_orders[0][0] == executed_state:
            pending_orders.popleft()

        current_y = y_array[row_number]
        current_x = x_array[row_number]
        current_beta = beta_array[row_number]
        execution_event = TradeEvent.NONE.value
        execution_inputs_available = bool(
            np.isfinite(current_y)
            and np.isfinite(current_x)
            and np.isfinite(current_beta)
        )

        if pending_orders and execution_inputs_available:
            target_state, source_event = pending_orders.popleft()
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
                    gross_target,
                )
                units_y, units_x = sizing
            executed_state = target_state
            execution_event = source_event

        notional_y, notional_x, gross_exposure, net_exposure = _mark_exposure(
            executed_state,
            units_y,
            units_x,
            current_y,
            current_x,
        )
        rows.append(
            (
                decisions["decision_state"].iat[row_number],
                executed_state,
                decisions["decision_event"].iat[row_number],
                execution_event,
                current_beta,
                current_y,
                current_x,
                units_y,
                units_x,
                notional_y,
                notional_x,
                gross_exposure,
                net_exposure,
            )
        )

    result = pd.DataFrame.from_records(rows, columns=list(_OUTPUT_COLUMNS))
    result.index = price_y.index
    for column in (
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
    for column in (
        "decision_state",
        "executed_state",
        "decision_event",
        "execution_event",
    ):
        result[column] = result[column].astype(object)
    return result.loc[:, list(_OUTPUT_COLUMNS)]
