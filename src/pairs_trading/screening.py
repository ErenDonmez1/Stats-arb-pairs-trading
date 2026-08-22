"""Deterministic cointegration screening for candidate security pairs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import combinations
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InfeasibleTestError, MissingDataError
from statsmodels.tsa.stattools import coint

from .stats import (
    ADF_MIN_OBSERVATIONS,
    SpreadValidationError,
    diagnose_spread,
    ols_spread,
)


@dataclass(frozen=True, order=True)
class PairCandidate:
    """One canonical, alphabetically oriented candidate pair."""

    symbol_y: str
    symbol_x: str
    group: str | None = None


@dataclass(frozen=True)
class PairScreeningResult:
    """Immutable statistical screening result for one candidate pair."""

    symbol_y: str
    symbol_x: str
    group: str | None
    observations: int
    alpha: float | None
    beta: float | None
    spread_standard_deviation: float | None
    cointegration_statistic: float | None
    cointegration_pvalue: float | None
    corrected_pvalue: float | None
    cointegration_critical_values: Mapping[str, float]
    adf_statistic: float | None
    adf_pvalue: float | None
    half_life: float | None
    hurst: float | None
    selected: bool
    rank: int | None
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cointegration_critical_values",
            MappingProxyType(dict(self.cointegration_critical_values)),
        )
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))


class _PairEvaluationError(ValueError):
    """Expected candidate-specific validation failure."""

    def __init__(self, reason: str, observations: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.observations = observations


class _PairHypothesisError(RuntimeError):
    """Expected unavailable Engle-Granger result after family eligibility."""

    def __init__(self, reason: str, observations: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.observations = observations


def _validate_price_frame(prices: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.shape[1] < 2:
        raise ValueError("prices must contain at least two symbol columns.")
    if prices.columns.duplicated().any():
        raise ValueError("prices must have unique symbol columns.")
    if any(not isinstance(symbol, str) or not symbol.strip() for symbol in prices.columns):
        raise ValueError("price column names must be non-empty strings.")
    if not prices.index.is_unique:
        raise ValueError("prices must have a unique index.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be monotonically increasing.")


def generate_candidate_pairs(
    prices: pd.DataFrame,
    groups: Mapping[str, Iterable[str]] | None = None,
) -> tuple[PairCandidate, ...]:
    """Generate unique unordered pairs in deterministic alphabetical order."""
    _validate_price_frame(prices)
    available = set(prices.columns)

    if groups is None:
        symbols = sorted(available)
        return tuple(PairCandidate(y, x, None) for y, x in combinations(symbols, 2))
    if not isinstance(groups, Mapping):
        raise TypeError("groups must be a mapping from group names to symbols.")

    invalid_groups = [
        group for group in groups if not isinstance(group, str) or not group.strip()
    ]
    if invalid_groups:
        raise ValueError("group names must be non-empty strings.")

    assigned: dict[str, str] = {}
    candidates: list[PairCandidate] = []
    for group in sorted(groups):
        configured_symbols = groups[group]
        if isinstance(configured_symbols, (str, bytes)) or not isinstance(
            configured_symbols, Iterable
        ):
            raise ValueError(f"Group {group!r} symbols must be an iterable.")
        symbols = list(configured_symbols)
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
            raise ValueError(f"Group {group!r} contains an invalid symbol.")
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"Group {group!r} contains duplicate symbols.")

        unknown = sorted(set(symbols) - available)
        if unknown:
            raise ValueError(f"Group {group!r} contains unknown symbols: {unknown}.")
        for symbol in symbols:
            if symbol in assigned:
                raise ValueError(
                    f"Symbol {symbol!r} belongs to both {assigned[symbol]!r} "
                    f"and {group!r}."
                )
            assigned[symbol] = group

        for symbol_y, symbol_x in combinations(sorted(symbols), 2):
            candidates.append(PairCandidate(symbol_y, symbol_x, group))

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.symbol_y,
                candidate.symbol_x,
                candidate.group or "",
            ),
        )
    )


def benjamini_hochberg(pvalues: Iterable[Real]) -> np.ndarray:
    """Return monotonic Benjamini-Hochberg adjusted p-values in input order."""
    if isinstance(pvalues, (str, bytes, set, frozenset)):
        raise TypeError("pvalues must be an ordered one-dimensional collection.")
    try:
        raw = np.asarray(list(pvalues), dtype=object)
    except TypeError as exc:
        raise TypeError("pvalues must be an ordered one-dimensional collection.") from exc
    if raw.ndim != 1:
        raise ValueError("pvalues must be one-dimensional.")

    values = np.empty(len(raw), dtype=float)
    for position, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("pvalues must contain only numeric non-boolean values.")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("pvalues must be finite.")
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("pvalues must lie in [0, 1].")
        values[position] = numeric

    count = len(values)
    if count == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    scaled = sorted_values * count / np.arange(1, count + 1)
    monotonic = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(monotonic, 0.0, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _formation_slice(
    prices: pd.DataFrame,
    formation_start: Any | None,
    formation_end: Any | None,
) -> pd.DataFrame:
    start = formation_start
    end = formation_end
    if isinstance(prices.index, pd.DatetimeIndex):
        try:
            start = pd.Timestamp(start) if start is not None else None
            end = pd.Timestamp(end) if end is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError("Formation boundaries must be valid timestamps.") from exc
    try:
        if start is not None and end is not None and start > end:
            raise ValueError("formation_start must not be after formation_end.")
        mask = np.ones(len(prices), dtype=bool)
        if start is not None:
            mask &= np.asarray(prices.index >= start)
        if end is not None:
            mask &= np.asarray(prices.index <= end)
    except TypeError as exc:
        raise ValueError("Formation boundaries are incompatible with the price index.") from exc

    formation = prices.loc[mask].copy(deep=True)
    if formation.empty:
        raise ValueError("The requested formation window is empty.")
    return formation


def _positive_finite_threshold(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return numeric


def _probability_threshold(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric.")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return numeric


def _validated_pair_prices(
    formation: pd.DataFrame,
    candidate: PairCandidate,
    min_observations: int,
) -> pd.DataFrame:
    aligned = formation.loc[:, [candidate.symbol_y, candidate.symbol_x]].dropna(
        how="any"
    )
    observations = len(aligned)
    if observations < min_observations:
        raise _PairEvaluationError("insufficient_observations", observations)

    for symbol in (candidate.symbol_y, candidate.symbol_x):
        values = aligned[symbol]
        numeric = values.map(
            lambda value: isinstance(value, Real) and not isinstance(value, bool)
        )
        if not bool(numeric.all()):
            raise _PairEvaluationError("non_numeric_prices", observations)
    aligned = aligned.astype(float)
    if not np.isfinite(aligned.to_numpy()).all():
        raise _PairEvaluationError("non_finite_prices", observations)
    if aligned.le(0).any().any():
        raise _PairEvaluationError("non_positive_prices", observations)
    for symbol in (candidate.symbol_y, candidate.symbol_x):
        if aligned[symbol].nunique(dropna=False) < 2:
            raise _PairEvaluationError("degenerate_prices", observations)
    return aligned


def _failed_result(
    candidate: PairCandidate,
    observations: int,
    reason: str,
) -> PairScreeningResult:
    return PairScreeningResult(
        symbol_y=candidate.symbol_y,
        symbol_x=candidate.symbol_x,
        group=candidate.group,
        observations=observations,
        alpha=None,
        beta=None,
        spread_standard_deviation=None,
        cointegration_statistic=None,
        cointegration_pvalue=None,
        corrected_pvalue=None,
        cointegration_critical_values={},
        adf_statistic=None,
        adf_pvalue=None,
        half_life=None,
        hurst=None,
        selected=False,
        rank=None,
        rejection_reasons=(reason,),
    )


def _evaluate_candidate(
    formation: pd.DataFrame,
    candidate: PairCandidate,
    min_observations: int,
) -> PairScreeningResult:
    pair = _validated_pair_prices(formation, candidate, min_observations)
    symbol_y, symbol_x = candidate.symbol_y, candidate.symbol_x
    log_y = np.log(pair[symbol_y])
    log_x = np.log(pair[symbol_x])
    try:
        statistic, pvalue, critical_values = coint(
            log_y,
            log_x,
            trend="c",
            autolag="aic",
        )
    except (
        FloatingPointError,
        InfeasibleTestError,
        MissingDataError,
        np.linalg.LinAlgError,
    ) as exc:
        raise _PairHypothesisError(
            f"statistical_estimation_failed:{type(exc).__name__}:{exc}",
            len(pair),
        ) from exc
    if not np.isfinite(statistic) or not np.isfinite(pvalue):
        raise _PairHypothesisError(
            "statistical_estimation_failed:non_finite_cointegration_output",
            len(pair),
        )
    critical_array = np.asarray(critical_values, dtype=float)
    if critical_array.shape != (3,) or not np.isfinite(critical_array).all():
        raise _PairHypothesisError(
            "statistical_estimation_failed:invalid_cointegration_critical_values",
            len(pair),
        )

    critical_mapping = {
        "1%": float(critical_array[0]),
        "5%": float(critical_array[1]),
        "10%": float(critical_array[2]),
    }
    try:
        spread, alpha, beta = ols_spread(pair[symbol_y], pair[symbol_x])
        diagnostics = diagnose_spread(spread)
        spread_standard_deviation = float(spread.std(ddof=1))
        if (
            not np.isfinite(spread_standard_deviation)
            or spread_standard_deviation <= 0
        ):
            raise SpreadValidationError(
                "Spread standard deviation is non-positive or non-finite."
            )
    except (SpreadValidationError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return PairScreeningResult(
            symbol_y=symbol_y,
            symbol_x=symbol_x,
            group=candidate.group,
            observations=len(pair),
            alpha=None,
            beta=None,
            spread_standard_deviation=None,
            cointegration_statistic=float(statistic),
            cointegration_pvalue=float(pvalue),
            corrected_pvalue=None,
            cointegration_critical_values=critical_mapping,
            adf_statistic=None,
            adf_pvalue=None,
            half_life=None,
            hurst=None,
            selected=False,
            rank=None,
            rejection_reasons=(
                f"spread_diagnostics_failed:{type(exc).__name__}:{exc}",
            ),
        )

    return PairScreeningResult(
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        group=candidate.group,
        observations=len(pair),
        alpha=alpha,
        beta=beta,
        spread_standard_deviation=spread_standard_deviation,
        cointegration_statistic=float(statistic),
        cointegration_pvalue=float(pvalue),
        corrected_pvalue=None,
        cointegration_critical_values=critical_mapping,
        adf_statistic=diagnostics.adf_statistic,
        adf_pvalue=diagnostics.adf_pvalue,
        half_life=diagnostics.half_life,
        hurst=diagnostics.hurst,
        selected=False,
        rank=None,
        rejection_reasons=(),
    )


def screen_pairs(
    prices: pd.DataFrame,
    groups: Mapping[str, Iterable[str]] | None = None,
    *,
    formation_start: Any | None = None,
    formation_end: Any | None = None,
    min_observations: int = 100,
    fdr_threshold: float = 0.05,
    max_half_life: float = 60.0,
    hurst_threshold: float = 0.5,
) -> tuple[PairScreeningResult, ...]:
    """Evaluate, FDR-correct, select, and deterministically rank pairs.

    Deterministic price eligibility is fixed before Engle-Granger testing.
    Every eligible candidate remains in the BH family: an unavailable expected
    test result contributes a conservative p-value of one. Unexpected
    programming or invariant errors propagate instead of becoming rejections.
    Screening FDR is separate from later strategy-scenario multiplicity.
    """
    _validate_price_frame(prices)
    if type(min_observations) is not int or min_observations < ADF_MIN_OBSERVATIONS:
        raise ValueError(
            f"min_observations must be an integer of at least "
            f"{ADF_MIN_OBSERVATIONS}."
        )
    fdr_limit = _probability_threshold("fdr_threshold", fdr_threshold)
    half_life_limit = _positive_finite_threshold(
        "max_half_life", max_half_life
    )
    if isinstance(hurst_threshold, bool) or not isinstance(hurst_threshold, Real):
        raise ValueError("hurst_threshold must be numeric.")
    hurst_limit = float(hurst_threshold)
    if not np.isfinite(hurst_limit):
        raise ValueError("hurst_threshold must be finite.")

    candidates = generate_candidate_pairs(prices, groups)
    if not candidates:
        raise ValueError("No candidate pairs were generated.")
    formation = _formation_slice(prices, formation_start, formation_end)

    # Determine family eligibility for every candidate before observing any
    # cointegration outcome. This prevents an early test result or failure from
    # influencing which later hypotheses enter the multiplicity correction.
    staged_results: list[PairScreeningResult | None] = []
    eligible_candidates: list[tuple[int, PairCandidate]] = []
    for candidate in candidates:
        try:
            _validated_pair_prices(formation, candidate, min_observations)
        except _PairEvaluationError as exc:
            staged_results.append(
                _failed_result(candidate, exc.observations, exc.reason)
            )
        else:
            eligible_candidates.append((len(staged_results), candidate))
            staged_results.append(None)

    family_positions = [position for position, _ in eligible_candidates]
    family_pvalues: list[float] = []
    for position, candidate in eligible_candidates:
        try:
            result = _evaluate_candidate(formation, candidate, min_observations)
        except _PairHypothesisError as exc:
            result = _failed_result(candidate, exc.observations, exc.reason)
            family_pvalues.append(1.0)
        else:
            if result.cointegration_pvalue is None:
                raise RuntimeError(
                    "An evaluated cointegration hypothesis has no p-value."
                )
            family_pvalues.append(result.cointegration_pvalue)
        staged_results[position] = result

    if any(result is None for result in staged_results):
        raise RuntimeError("Screening left an eligible hypothesis unevaluated.")
    results = [result for result in staged_results if result is not None]

    adjusted = benjamini_hochberg(family_pvalues)
    for position, corrected in zip(family_positions, adjusted):
        result = results[position]
        reasons = list(result.rejection_reasons)
        if corrected > fdr_limit:
            reasons.append("corrected_cointegration_pvalue_above_threshold")
        if (
            result.beta is None
            or not np.isfinite(result.beta)
            or result.beta <= 0
        ):
            reasons.append("beta_not_finite_positive")
        if (
            result.half_life is None
            or not np.isfinite(result.half_life)
            or result.half_life <= 0
        ):
            reasons.append("half_life_not_finite_positive")
        elif result.half_life > half_life_limit:
            reasons.append("half_life_above_maximum")
        if (
            result.hurst is None
            or not np.isfinite(result.hurst)
            or result.hurst >= hurst_limit
        ):
            reasons.append("hurst_not_below_threshold")
        results[position] = replace(
            result,
            corrected_pvalue=float(corrected),
            selected=not reasons,
            rejection_reasons=tuple(reasons),
        )

    ordered = sorted(
        results,
        key=lambda result: (
            not result.selected,
            result.corrected_pvalue
            if result.corrected_pvalue is not None
            and np.isfinite(result.corrected_pvalue)
            else float("inf"),
            result.half_life
            if result.half_life is not None and np.isfinite(result.half_life)
            else float("inf"),
            result.symbol_y,
            result.symbol_x,
        ),
    )
    next_rank = 1
    ranked: list[PairScreeningResult] = []
    for result in ordered:
        if result.selected:
            ranked.append(replace(result, rank=next_rank))
            next_rank += 1
        else:
            ranked.append(replace(result, rank=None))
    return tuple(ranked)
