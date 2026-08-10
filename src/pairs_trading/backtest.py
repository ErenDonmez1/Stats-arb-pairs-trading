"""Causal pair execution, unit sizing, and marked-to-market accounting.

The module converts state-changing decisions into lagged executions, sizes the
two legs on the actual execution row, and accounts for exposure and gross P&L.
Transaction costs, financing, trade ledgers, and performance metrics remain
outside this milestone.
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
    "calculate_position_pnl",
    "calculate_strategy_returns",
    "build_pnl_schedule",
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


def _validated_position_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Copy and validate the state and unit contract from Milestone 6A."""
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("position_schedule must be a pandas DataFrame.")
    if not schedule.index.is_unique:
        raise ValueError("position_schedule must have a unique index.")

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

    previous_state = PositionState.FLAT.name
    previous_y = 0.0
    previous_x = 0.0
    for row_number, (state, event, current_y, current_x) in enumerate(
        zip(states, events, units_y, units_x, strict=True)
    ):
        if row_number == 0 and (
            state != PositionState.FLAT.name
            or event != TradeEvent.NONE.value
            or current_y != 0.0
            or current_x != 0.0
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

        if event == TradeEvent.NONE.value:
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
