"""Causal spread standardisation and deterministic trading-signal state.

Missing rows are preserved in place.  A z-score is defined only when the
current spread is present and every observation in its fixed-size prior window
is present.  Values are never filled, interpolated, dropped, or reordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, unique
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


# A prior standard deviation at or below this absolute number of spread units
# is treated as numerically indistinguishable from zero.  The raw standard
# deviation remains visible in standardise_spread(); only division is masked.
_NEAR_ZERO_STANDARD_DEVIATION = 1e-12

__all__ = [
    "PositionState",
    "TradeEvent",
    "ExitReason",
    "rolling_zscore",
    "standardise_spread",
    "generate_trade_signals",
]


@unique
class PositionState(IntEnum):
    """Immutable desired pair-trading state and its position encoding."""

    FLAT = 0
    LONG_SPREAD = 1
    SHORT_SPREAD = -1


@unique
class TradeEvent(str, Enum):
    """Deterministic events emitted by the signal state machine."""

    NONE = "NONE"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_MEAN_REVERSION = "EXIT_MEAN_REVERSION"
    EXIT_STOP = "EXIT_STOP"
    EXIT_TIME = "EXIT_TIME"


@unique
class ExitReason(str, Enum):
    """Deterministic exit classifications for downstream audit trails."""

    NONE = "NONE"
    MEAN_REVERSION = "MEAN_REVERSION"
    STOP = "STOP"
    TIME = "TIME"


_TRADE_SIGNAL_COLUMNS = (
    "zscore",
    "state",
    "position",
    "entry_long",
    "entry_short",
    "exit",
    "stop",
    "time_exit",
    "event",
    "exit_reason",
    "holding_period",
    "cooldown_remaining",
)


@dataclass(frozen=True)
class _TradeSignalPolicy:
    """Validated thresholds and row-count limits shared by signal engines."""

    entry_z: float
    exit_z: float
    stop_z: float
    max_holding_period: int | None
    cooldown_period: int


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive non-Boolean integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer.")
    normalised = int(value)
    if normalised <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalised


def _valid_ddof(value: Any, lookback: int) -> int:
    """Validate degrees of freedom for a fixed window of ``lookback`` rows."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("ddof must be a non-Boolean integer.")
    normalised = int(value)
    if normalised < 0 or normalised >= lookback:
        raise ValueError("ddof must satisfy 0 <= ddof < lookback.")
    return normalised


def _require_causal_index(index: pd.Index, name: str) -> None:
    """Reject duplicate or nonchronological dated observations without sorting."""
    if not index.is_unique:
        raise ValueError(f"{name} must have a unique index.")
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        raise ValueError(
            f"{name} DatetimeIndex must be monotonically increasing."
        )


def _validated_spread(
    spread: pd.Series,
    lookback: Any,
    ddof: Any,
) -> tuple[pd.Series, int, int]:
    """Return an independent float spread after strict structural validation."""
    if not isinstance(spread, pd.Series):
        raise TypeError("spread must be a pandas Series.")
    _require_causal_index(spread.index, "spread")

    normalised_lookback = _positive_integer(lookback, "lookback")
    normalised_ddof = _valid_ddof(ddof, normalised_lookback)
    if len(spread) <= normalised_lookback:
        raise ValueError(
            "spread must contain at least lookback + 1 rows to evaluate one "
            "post-warm-up observation."
        )

    copied = spread.copy(deep=True)
    non_missing = copied.loc[copied.notna()]
    numeric = non_missing.map(
        lambda value: isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if not bool(numeric.all()):
        raise ValueError("Non-missing spread observations must be real numeric values.")

    try:
        array = copied.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Non-missing spread observations must be representable as floats."
        ) from exc
    values = pd.Series(
        array,
        index=spread.index,
        name="spread",
        dtype=float,
    )
    finite_values = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_values).all():
        raise ValueError("Non-missing spread observations must be finite.")
    return values, normalised_lookback, normalised_ddof


def standardise_spread(
    spread: pd.Series,
    lookback: int = 60,
    *,
    method: str = "rolling",
    ddof: int = 1,
) -> pd.DataFrame:
    """Return causal rolling distribution statistics and spread z-scores.

    At row ``t``, ``rolling_mean`` and ``rolling_std`` use exactly ``lookback``
    rows ending at ``t-1``.  The current spread is excluded from both.  The
    default ``ddof=1`` produces the sample standard deviation; valid values
    satisfy ``0 <= ddof < lookback``.

    The first ``lookback`` rows are warm-up rows and remain missing.  Any
    missing value in a prior fixed-size window makes that row's mean, standard
    deviation, and z-score missing.  A missing current value makes its z-score
    missing without changing its prior-window statistics.

    Standard deviations less than or equal to ``1e-12`` spread units are
    considered near zero.  Their raw values remain in ``rolling_std`` for
    auditability, while the corresponding z-score is missing rather than zero
    or infinity.

    Only ``method="rolling"`` is supported in Milestone 5A.  The explicit
    selector keeps the public interface extensible without silently applying
    an unrequested standardisation policy.
    """
    if not isinstance(method, str) or method != "rolling":
        raise ValueError("method must be 'rolling'.")

    values, normalised_lookback, normalised_ddof = _validated_spread(
        spread,
        lookback,
        ddof,
    )
    prior = values.shift(1)
    rolling = prior.rolling(
        window=normalised_lookback,
        min_periods=normalised_lookback,
    )
    rolling_mean = rolling.mean().rename("rolling_mean")
    rolling_std = rolling.std(ddof=normalised_ddof).rename("rolling_std")
    usable_std = rolling_std.where(
        np.isfinite(rolling_std)
        & rolling_std.gt(_NEAR_ZERO_STANDARD_DEVIATION)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        zscore = ((values - rolling_mean) / usable_std).rename("zscore")
    zscore = zscore.where(np.isfinite(zscore))

    result = pd.concat(
        [values, rolling_mean, rolling_std, zscore],
        axis=1,
        copy=False,
    )
    # Reuse the caller's index object so names, timezone, frequency, and custom
    # ordering metadata remain intact.
    result.index = spread.index
    return result.loc[:, ["spread", "rolling_mean", "rolling_std", "zscore"]]


def rolling_zscore(
    spread: pd.Series,
    lookback: int = 60,
    *,
    ddof: int = 1,
) -> pd.Series:
    """Return a causal rolling z-score named ``zscore``.

    This is the compact Series interface to :func:`standardise_spread`; both
    functions therefore share exactly the same validation and missing-value
    policies.
    """
    result = standardise_spread(
        spread,
        lookback,
        method="rolling",
        ddof=ddof,
    )["zscore"].copy()
    result.name = "zscore"
    return result


def _finite_real_parameter(value: Any, name: str) -> float:
    """Return a finite real-valued strategy threshold."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    try:
        normalised = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as a float.") from exc
    if not np.isfinite(normalised):
        raise ValueError(f"{name} must be finite.")
    return normalised


def _optional_positive_integer(value: Any, name: str) -> int | None:
    """Return ``None`` or a positive non-Boolean integer."""
    if value is None:
        return None
    normalised = _positive_integer(value, name)
    if normalised > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the supported int64 range.")
    return normalised


def _optional_nonnegative_integer(value: Any, name: str) -> int:
    """Return zero for ``None`` or a non-negative non-Boolean integer."""
    if value is None:
        return 0
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer.")
    normalised = int(value)
    if normalised < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    if normalised > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the supported int64 range.")
    return normalised


def _validated_zscore(zscore: pd.Series) -> pd.Series:
    """Return an independent float z-score Series without changing row order."""
    if not isinstance(zscore, pd.Series):
        raise TypeError("zscore must be a pandas Series.")
    _require_causal_index(zscore.index, "zscore")

    copied = zscore.copy(deep=True)
    non_missing = copied.loc[copied.notna()]
    numeric = non_missing.map(
        lambda value: isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
    )
    if not bool(numeric.all()):
        raise ValueError("Non-missing zscore observations must be real numeric values.")
    try:
        array = copied.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Non-missing zscore observations must be representable as floats."
        ) from exc

    values = pd.Series(array, index=zscore.index, name="zscore", dtype=float)
    if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
        raise ValueError("Non-missing zscore observations must be finite.")
    return values


def _coerce_trade_signal_dtypes(result: pd.DataFrame) -> pd.DataFrame:
    """Apply stable output dtypes, including for a legitimate empty input."""
    result["zscore"] = result["zscore"].astype("float64")
    for column in ("entry_long", "entry_short", "exit", "stop", "time_exit"):
        result[column] = result[column].astype(bool)
    result["position"] = result["position"].astype("int8")
    result["holding_period"] = result["holding_period"].astype("int64")
    result["cooldown_remaining"] = result["cooldown_remaining"].astype("int64")
    for column in ("state", "event", "exit_reason"):
        result[column] = result[column].astype(object)
    return result.loc[:, list(_TRADE_SIGNAL_COLUMNS)]


def _validated_trade_signal_inputs(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    *,
    max_holding_period: int | None,
    cooldown_period: int | None,
    missing_policy: str,
) -> tuple[pd.Series, _TradeSignalPolicy]:
    """Validate common strategy inputs for diagnostic and integrated engines."""
    values = _validated_zscore(zscore)
    normalised_entry = _finite_real_parameter(entry_z, "entry_z")
    normalised_exit = _finite_real_parameter(exit_z, "exit_z")
    normalised_stop = _finite_real_parameter(stop_z, "stop_z")
    if normalised_exit < 0:
        raise ValueError("exit_z must be non-negative.")
    if normalised_entry <= normalised_exit:
        raise ValueError("entry_z must be greater than exit_z.")
    if normalised_stop <= normalised_entry:
        raise ValueError("stop_z must be greater than entry_z.")
    maximum_holding = _optional_positive_integer(
        max_holding_period,
        "max_holding_period",
    )
    configured_cooldown = _optional_nonnegative_integer(
        cooldown_period,
        "cooldown_period",
    )
    if not isinstance(missing_policy, str) or missing_policy != "hold":
        raise ValueError("missing_policy must be 'hold'.")
    return values, _TradeSignalPolicy(
        entry_z=normalised_entry,
        exit_z=normalised_exit,
        stop_z=normalised_stop,
        max_holding_period=maximum_holding,
        cooldown_period=configured_cooldown,
    )


def _entry_decision(
    current_zscore: float,
    policy: _TradeSignalPolicy,
) -> tuple[PositionState, TradeEvent]:
    """Return the inclusive entry decision for one eligible flat row."""
    if np.isnan(current_zscore):
        return PositionState.FLAT, TradeEvent.NONE
    if current_zscore <= -policy.entry_z:
        return PositionState.LONG_SPREAD, TradeEvent.ENTER_LONG
    if current_zscore >= policy.entry_z:
        return PositionState.SHORT_SPREAD, TradeEvent.ENTER_SHORT
    return PositionState.FLAT, TradeEvent.NONE


def _exit_decision(
    state: PositionState,
    current_zscore: float,
    holding_period: int,
    policy: _TradeSignalPolicy,
) -> tuple[TradeEvent, ExitReason]:
    """Return an actual/open-state exit decision using canonical precedence."""
    stop_triggered = False
    mean_reversion_triggered = False
    if not np.isnan(current_zscore):
        if state is PositionState.LONG_SPREAD:
            stop_triggered = current_zscore <= -policy.stop_z
            mean_reversion_triggered = current_zscore >= -policy.exit_z
        elif state is PositionState.SHORT_SPREAD:
            stop_triggered = current_zscore >= policy.stop_z
            mean_reversion_triggered = current_zscore <= policy.exit_z
        else:
            raise ValueError("Exit decisions require an open position state.")
    time_triggered = bool(
        policy.max_holding_period is not None
        and holding_period >= policy.max_holding_period
    )
    if stop_triggered:
        return TradeEvent.EXIT_STOP, ExitReason.STOP
    if time_triggered:
        return TradeEvent.EXIT_TIME, ExitReason.TIME
    if mean_reversion_triggered:
        return TradeEvent.EXIT_MEAN_REVERSION, ExitReason.MEAN_REVERSION
    return TradeEvent.NONE, ExitReason.NONE


def _trade_signal_record(
    current_zscore: float,
    state: PositionState,
    event: TradeEvent,
    exit_reason: ExitReason,
    holding_period: int,
    cooldown_remaining: int,
) -> tuple[Any, ...]:
    """Build one canonical primitive signal row from a strategy decision."""
    is_exit = event in {
        TradeEvent.EXIT_MEAN_REVERSION,
        TradeEvent.EXIT_STOP,
        TradeEvent.EXIT_TIME,
    }
    return (
        current_zscore,
        state.name,
        int(state),
        event is TradeEvent.ENTER_LONG,
        event is TradeEvent.ENTER_SHORT,
        is_exit,
        event is TradeEvent.EXIT_STOP,
        event is TradeEvent.EXIT_TIME,
        event.value,
        exit_reason.value,
        holding_period,
        cooldown_remaining,
    )


def generate_trade_signals(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    *,
    max_holding_period: int | None = None,
    cooldown_period: int | None = None,
    missing_policy: str = "hold",
) -> pd.DataFrame:
    """Generate desired pair-trading states from an existing causal z-score.

    The output is a sequence of order decisions, not executed portfolio
    holdings.  A decision at row ``t`` may use ``zscore_t``; a later backtester
    is responsible for applying execution lag and fills.

    This standalone function has no execution feedback.  Its holding and
    cooldown columns therefore describe the desired-state diagnostic path,
    not actual filled exposure.  :func:`pairs_trading.backtest.run_pair_backtest`
    uses an interleaved execution-aware pathway for z-score-driven runs.

    The entry-decision row has ``holding_period=0``.  Every later desired-open
    row advances the counter, including a missing-z-score row.  Reaching
    ``max_holding_period`` exits on that row.  Exit rows report the completed
    desired-state holding period even though their state is flat; only the
    internal counter used by subsequent rows resets to zero.  Exit precedence
    is stop, time limit, then mean reversion.

    The exit row does not consume cooldown.  An exit reports the configured
    cooldown, and each of the next ``cooldown_period`` rows is entry-ineligible
    while reporting the pre-decrement value applying during that row.  A zero
    value therefore means the current row is eligible for entry.

    ``missing_policy="hold"`` preserves missing values and suppresses threshold
    entries and exits.  Open states and flat states are retained, holding time
    and cooldown still advance, and a time exit may occur on a missing row.
    No values are filled, interpolated, dropped, or reordered.

    Empty Series are accepted and return an empty, fully typed frame.  State,
    event, and exit-reason values are emitted as stable primitive strings.
    """
    values, policy = _validated_trade_signal_inputs(
        zscore,
        entry_z,
        exit_z,
        stop_z,
        max_holding_period=max_holding_period,
        cooldown_period=cooldown_period,
        missing_policy=missing_policy,
    )

    state = PositionState.FLAT
    holding_period = 0
    cooldown_remaining = 0
    rows: list[tuple[Any, ...]] = []

    for current_zscore in values.to_numpy(dtype=float):
        event = TradeEvent.NONE
        exit_reason = ExitReason.NONE
        is_missing = bool(np.isnan(current_zscore))
        row_holding_period = 0
        row_cooldown_remaining = cooldown_remaining

        if state is PositionState.FLAT:
            holding_period = 0
            if cooldown_remaining > 0:
                # Consume this whole row before another entry becomes eligible.
                cooldown_remaining -= 1
            elif not is_missing:
                state, event = _entry_decision(current_zscore, policy)
        else:
            holding_period += 1
            row_holding_period = holding_period
            event, exit_reason = _exit_decision(
                state,
                current_zscore,
                holding_period,
                policy,
            )
            if event is not TradeEvent.NONE:
                state = PositionState.FLAT
                holding_period = 0
                cooldown_remaining = policy.cooldown_period
                row_cooldown_remaining = cooldown_remaining

        rows.append(
            _trade_signal_record(
                current_zscore,
                state,
                event,
                exit_reason,
                row_holding_period,
                row_cooldown_remaining,
            )
        )

    result = pd.DataFrame.from_records(
        rows,
        columns=list(_TRADE_SIGNAL_COLUMNS),
    )
    result.index = zscore.index
    return _coerce_trade_signal_dtypes(result)
