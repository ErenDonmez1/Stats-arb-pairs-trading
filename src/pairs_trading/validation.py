"""Statistical diagnostics for finite out-of-sample pairs-trading evidence.

The module quantifies uncertainty, temporal concentration, regime dependence,
and scenario multiplicity.  It never selects parameters or represents
historical statistical evidence as proof of future profitability.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import Enum
from math import ceil
from numbers import Integral, Real
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .analytics import (
    NEAR_ZERO_TOLERANCE,
    annualized_return,
    annualized_volatility,
    maximum_drawdown,
    sharpe_ratio,
    total_return,
)
from .robustness import MetricAvailabilityStatus, RobustnessResult, ScenarioStatus
from .walkforward import WalkForwardResult, WalkForwardStatus


__all__ = [
    "ValidationAvailability",
    "ConfidenceInterval",
    "BootstrapMetricResult",
    "BootstrapPerformanceResult",
    "ProbabilisticSharpeResult",
    "MinimumTrackRecordResult",
    "FoldPerformance",
    "FoldConsistencyMetrics",
    "RegimePerformance",
    "RegimeConsistencyMetrics",
    "ScenarioPValue",
    "ParameterAxisCount",
    "MultipleTestingDiagnostics",
    "StatisticalValidationResult",
    "moving_block_bootstrap",
    "bootstrap_performance_metrics",
    "probabilistic_sharpe_ratio",
    "minimum_track_record_length",
    "analyze_fold_consistency",
    "analyze_regime_consistency",
    "multiple_testing_diagnostics",
    "build_statistical_validation_report",
]


PURPOSE = "statistical_diagnostics_not_proof_of_future_profitability"
PSR_CONVENTION = (
    "PSR uses per-period arithmetic Sharpe ratios; annualized input benchmarks "
    "are divided by sqrt(periods_per_year), ordinary sample kurtosis is used, "
    "and the result is not a probability of making money."
)
BOOTSTRAP_INTERPRETATION = (
    "Percentile intervals are empirical moving-block resampling uncertainty "
    "intervals conditional on the supplied OOS sample and block assumptions. "
    "Validity assumes the observed OOS history is informative about local "
    "dependence, the caller-declared block length adequately represents that "
    "dependence, and approximate stationarity/mixing conditions are reasonable. "
    "Row adjacency, not elapsed calendar time, defines a block. These empirical "
    "diagnostics are not guarantees of future performance."
)
REGIME_PROVENANCE_WARNING = (
    "Regime labels are caller-supplied and their causal provenance is not "
    "verified; conclusions are valid only if labels were available point-in-time."
)
MULTIPLE_TESTING_WARNING = (
    "Scenario-grid tests are highly dependent. Bonferroni and Benjamini-Hochberg "
    "adjustments are diagnostic, not definitive proof, and do not replace the "
    "separate FDR procedure used during pair screening."
)
VALIDATION_WARNINGS = (
    "OOS data remain a finite historical sample.",
    "Moving-block inference assumes observed OOS history is informative about future local dependence, approximate stationarity/mixing is reasonable, and the caller-declared block length adequately represents row-adjacent dependence.",
    "Bootstrap intervals are empirical resampling diagnostics and do not guarantee future performance.",
    "Parameter-grid scenarios are dependent, so multiplicity adjustments remain diagnostic.",
    "Repeated research iterations create researcher degrees of freedom not fully captured by the formal test count.",
    "Point-in-time universe and cleaning provenance remain caller-supplied under current project limitations.",
    "Statistical significance does not imply economic significance.",
    "Economic significance does not imply future profitability.",
    "Parameter sensitivity and temporal fold consistency are separate diagnostics and are not pooled into one variance estimate.",
)


class ValidationAvailability(str, Enum):
    """Availability of one statistical diagnostic."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class ConfidenceInterval:
    """Empirical lower and upper percentile bounds."""

    lower: float
    upper: float
    confidence_level: float


@dataclass(frozen=True)
class BootstrapMetricResult:
    """Point estimate and resampling uncertainty for one metric."""

    metric: str
    point_estimate: float
    bootstrap_median: float
    interval: ConfidenceInterval | None
    status: ValidationAvailability
    total_replicates: int
    valid_replicates: int
    undefined_replicates: int
    valid_fraction: float
    error: str | None
    conditional_bootstrap_median: float | None
    conditional_interval: ConfidenceInterval | None
    conditional_status: ValidationAvailability
    primary_interval_is_unconditional: bool


@dataclass(frozen=True)
class BootstrapPerformanceResult:
    """Moving-block samples and per-metric empirical uncertainty."""

    status: ValidationAvailability
    error: str | None
    observations: int
    block_length: int
    n_bootstrap: int
    random_seed: int
    confidence_level: float
    minimum_valid_fraction: float
    possible_block_starts: int
    effective_unique_replicates: int
    inference_design_status: ValidationAvailability
    inference_design_error: str | None
    total_return: BootstrapMetricResult
    annualized_return: BootstrapMetricResult
    annualized_volatility: BootstrapMetricResult
    sharpe_ratio: BootstrapMetricResult
    maximum_drawdown: BootstrapMetricResult
    original_returns: pd.Series
    bootstrap_samples: pd.DataFrame
    replicate_metrics: pd.DataFrame
    interpretation: str = BOOTSTRAP_INTERPRETATION

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_returns", self.original_returns.copy(deep=True))
        object.__setattr__(self, "bootstrap_samples", self.bootstrap_samples.copy(deep=True))
        object.__setattr__(self, "replicate_metrics", self.replicate_metrics.copy(deep=True))


@dataclass(frozen=True)
class ProbabilisticSharpeResult:
    """Finite-sample PSR diagnostic using a per-period Sharpe convention."""

    status: ValidationAvailability
    probability: float
    observations: int
    observed_sharpe_per_period: float
    observed_sharpe_annualized: float
    benchmark_sharpe_per_period: float
    benchmark_sharpe_annualized: float
    skewness: float
    ordinary_kurtosis: float
    variance_term: float
    statistic: float
    one_sided_pvalue: float
    error: str | None
    convention: str = PSR_CONVENTION


@dataclass(frozen=True)
class MinimumTrackRecordResult:
    """Approximate sample length required by the same PSR framework."""

    status: ValidationAvailability
    current_observations: int
    estimated_required_observations: int | None
    sufficient_track_record: bool | None
    confidence_level: float
    observed_sharpe_annualized: float
    benchmark_sharpe_annualized: float
    error: str | None
    convention: str = PSR_CONVENTION


@dataclass(frozen=True)
class FoldPerformance:
    """One walk-forward fold retained in temporal consistency analysis."""

    fold_id: int
    selection_status: str
    availability: ValidationAvailability
    observations: int
    total_return: float
    mean_return: float
    annualized_volatility: float
    sharpe_ratio: float
    trade_count: int
    catastrophic: bool
    invalid_return: bool
    error: str | None


@dataclass(frozen=True)
class FoldConsistencyMetrics:
    """Temporal fold dispersion and positive-contribution concentration.

    ``pre_execution_insufficient_fold_count`` counts walk-forward folds that
    never produced usable return evidence.  ``risk_analytics_insufficient_fold_count``
    counts folds with a finite total return but insufficient fold-level risk
    analytics.  The legacy ``insufficient_data_fold_count`` retains its prior
    exact meaning and aliases the latter count.
    """

    status: ValidationAvailability
    folds: tuple[FoldPerformance, ...]
    fold_count: int
    observable_fold_count: int
    positive_return_fold_count: int
    negative_return_fold_count: int
    zero_return_no_selection_fold_count: int
    unavailable_fold_count: int
    analytically_available_fold_count: int
    insufficient_data_fold_count: int
    pre_execution_insufficient_fold_count: int
    risk_analytics_insufficient_fold_count: int
    catastrophic_fold_count: int
    catastrophic_fold_ids: tuple[int, ...]
    invalid_return_fold_count: int
    summaries_complete: bool
    fraction_observable_folds_positive: float
    median_fold_total_return: float
    lower_quartile_fold_total_return: float
    upper_quartile_fold_total_return: float
    worst_fold_return: float
    strongest_fold_return: float
    strongest_single_positive_fold_concentration: float
    strongest_twenty_percent_positive_folds_concentration: float
    strongest_twenty_percent_positive_fold_count: int
    contribution_basis: str
    warning: str


@dataclass(frozen=True)
class RegimePerformance:
    """Descriptive metrics for one caller-supplied regime label."""

    regime_label: Any
    availability: ValidationAvailability
    observations: int
    total_return: float
    mean_return: float
    annualized_volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    fraction_positive_observations: float
    arithmetic_return_contribution: float
    error: str | None


@dataclass(frozen=True)
class RegimeConsistencyMetrics:
    """Cross-regime diagnostics without inferring or selecting regimes."""

    status: ValidationAvailability
    regimes: tuple[RegimePerformance, ...]
    regime_count: int
    observations: int
    positive_regime_count: int
    negative_regime_count: int
    zero_regime_count: int
    worst_regime: Any | None
    strongest_regime_positive_contribution_concentration: float
    contribution_basis: str
    contribution_warning: str
    returns: pd.Series
    regime_labels: pd.Series
    point_in_time_regime_labels_validated: bool
    provenance_warning: str
    error: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "returns", self.returns.copy(deep=True))
        object.__setattr__(self, "regime_labels", self.regime_labels.copy(deep=True))
        object.__setattr__(self, "regimes", tuple(self.regimes))


@dataclass(frozen=True)
class ScenarioPValue:
    """One scenario's PSR-style p-value and multiplicity adjustments."""

    scenario_id: str
    scenario_status: str
    psr_status: ValidationAvailability
    raw_pvalue: float
    bonferroni_adjusted_pvalue: float
    benjamini_hochberg_adjusted_pvalue: float
    error: str | None


@dataclass(frozen=True)
class ParameterAxisCount:
    """Tested value count for one predefined parameter axis."""

    parameter: str
    count: int


@dataclass(frozen=True)
class MultipleTestingDiagnostics:
    """Scenario-level multiplicity accounting without scenario selection."""

    scenario_pvalues: tuple[ScenarioPValue, ...]
    total_tested_configurations: int
    valid_comparable_configurations: int
    invalid_configurations: int
    analytically_unavailable_configurations: int
    failed_configurations: int
    valid_pvalue_count: int
    unavailable_pvalue_count: int
    eligible_hypothesis_count: int
    finite_pvalue_count: int
    unavailable_eligible_hypothesis_count: int
    family_size: int
    tested_dimension_count: int
    parameter_axis_counts: tuple[ParameterAxisCount, ...]
    significance_level: float
    bonferroni_valid_test_count: int
    raw_threshold_exceedance_count: int
    bonferroni_threshold_exceedance_count: int
    benjamini_hochberg_discovery_count: int
    baseline_scenario_id: str
    warning: str = MULTIPLE_TESTING_WARNING


@dataclass(frozen=True)
class StatisticalValidationResult:
    """Composed statistical evidence-quality report with no decision output.

    ``primary_inference_availability`` composes bootstrap, probabilistic Sharpe,
    and minimum-track-record evidence.  ``overall_availability`` additionally
    includes fold consistency and caller-requested regime consistency.  The
    legacy ``availability`` field is retained as a backward-compatible alias of
    ``primary_inference_availability`` and must not be interpreted as a status
    for every requested diagnostic.
    """

    bootstrap: BootstrapPerformanceResult
    probabilistic_sharpe: ProbabilisticSharpeResult
    minimum_track_record: MinimumTrackRecordResult
    fold_consistency: FoldConsistencyMetrics
    multiple_testing: MultipleTestingDiagnostics
    regime_consistency: RegimeConsistencyMetrics | None
    primary_oos_returns: pd.Series
    observations: int
    primary_inference_availability: ValidationAvailability
    overall_availability: ValidationAvailability
    availability: ValidationAvailability
    validation_warnings: tuple[str, ...]
    provenance_warnings: tuple[str, ...]
    purpose: str = PURPOSE

    def __post_init__(self) -> None:
        if self.availability is not self.primary_inference_availability:
            raise ValueError(
                "Legacy availability must equal primary_inference_availability."
            )
        object.__setattr__(self, "primary_oos_returns", self.primary_oos_returns.copy(deep=True))
        object.__setattr__(self, "validation_warnings", tuple(self.validation_warnings))
        object.__setattr__(self, "provenance_warnings", tuple(self.provenance_warnings))
        object.__setattr__(self, "bootstrap", replace(self.bootstrap))
        if self.regime_consistency is not None:
            object.__setattr__(self, "regime_consistency", replace(self.regime_consistency))


_METRIC_NAMES = (
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "maximum_drawdown",
)


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-Boolean integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _confidence_level(value: Any, name: str = "confidence_level") -> float:
    result = _finite_real(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")
    return result


def _validated_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if not returns.index.is_unique:
        raise ValueError("returns must have a unique index.")
    nonmissing = returns.loc[returns.notna()]
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in nonmissing.tolist()
    ):
        raise ValueError(
            "Non-missing returns must already be non-Boolean real numeric values."
        )
    numeric = returns.astype(float)
    finite = numeric.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("Non-missing returns must be finite.")
    numeric.index = returns.index
    numeric.name = returns.name
    return numeric.copy(deep=True)


def _require_bootstrap_chronology(returns: pd.Series) -> None:
    """Require deterministic chronological row order without sorting input."""
    if not returns.index.is_monotonic_increasing:
        raise ValueError(
            "Bootstrap returns must have a monotonically increasing index; "
            "input is not silently sorted."
        )


def _unavailable_metric(
    metric: str,
    total_replicates: int,
    error: str,
) -> BootstrapMetricResult:
    return BootstrapMetricResult(
        metric=metric,
        point_estimate=float("nan"),
        bootstrap_median=float("nan"),
        interval=None,
        status=ValidationAvailability.UNAVAILABLE,
        total_replicates=total_replicates,
        valid_replicates=0,
        undefined_replicates=total_replicates,
        valid_fraction=0.0,
        error=error,
        conditional_bootstrap_median=None,
        conditional_interval=None,
        conditional_status=ValidationAvailability.UNAVAILABLE,
        primary_interval_is_unconditional=False,
    )


def moving_block_bootstrap(
    returns: pd.Series,
    block_length: int,
    n_bootstrap: int,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Return deterministic moving-block resamples using a local NumPy RNG."""
    values = _validated_returns(returns)
    _require_bootstrap_chronology(values)
    if bool(values.isna().any()):
        raise ValueError("returns must not contain missing values for bootstrap inference.")
    if values.empty:
        raise ValueError("returns must contain at least one observation.")
    block = _integer(block_length, "block_length")
    replicates = _integer(n_bootstrap, "n_bootstrap")
    seed = _integer(random_seed, "random_seed", minimum=0)
    if block > len(values):
        raise ValueError("block_length must not exceed the number of observations.")

    source = values.to_numpy(dtype=float)
    block_count = ceil(len(source) / block)
    maximum_start = len(source) - block
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, len(source)), dtype=float)
    for replicate in range(replicates):
        starts = rng.integers(0, maximum_start + 1, size=block_count)
        joined = np.concatenate(
            [source[start : start + block] for start in starts]
        )
        samples[replicate] = joined[: len(source)]
    return pd.DataFrame(
        samples,
        index=pd.RangeIndex(replicates, name="bootstrap_replicate"),
        columns=pd.RangeIndex(len(source), name="observation"),
    )


def _metric_values(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
) -> dict[str, float]:
    """Calculate metrics after handling only documented mathematical gaps."""
    values = _validated_returns(returns)
    if bool(values.isna().any()):
        raise ValueError("Metric inputs must not contain missing observations.")
    results = {name: float("nan") for name in _METRIC_NAMES}
    if values.empty:
        return results

    compoundable = not bool(values.le(-1.0).any())
    if compoundable:
        results["total_return"] = float(
            total_return(values, missing_policy="drop")
        )
        results["annualized_return"] = float(
            annualized_return(
                values,
                periods_per_year,
                missing_policy="drop",
            )
        )
        results["maximum_drawdown"] = float(
            maximum_drawdown(values, missing_policy="drop")
        )
    if len(values) >= 2:
        results["annualized_volatility"] = float(
            annualized_volatility(
                values,
                periods_per_year,
                missing_policy="drop",
            )
        )
        results["sharpe_ratio"] = float(
            sharpe_ratio(
                values,
                periods_per_year,
                risk_free_rate,
                missing_policy="drop",
            )
        )
    return {
        name: value if np.isfinite(value) else float("nan")
        for name, value in results.items()
    }


def bootstrap_performance_metrics(
    returns: pd.Series,
    block_length: int,
    n_bootstrap: int,
    *,
    random_seed: int = 0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    minimum_valid_fraction: float = 0.80,
) -> BootstrapPerformanceResult:
    """Estimate empirical metric uncertainty with moving contiguous blocks."""
    values = _validated_returns(returns)
    _require_bootstrap_chronology(values)
    block = _integer(block_length, "block_length")
    replicates = _integer(n_bootstrap, "n_bootstrap")
    seed = _integer(random_seed, "random_seed", minimum=0)
    periods = _integer(periods_per_year, "periods_per_year")
    risk_free = _finite_real(risk_free_rate, "risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")
    confidence = _confidence_level(confidence_level)
    valid_minimum = _finite_real(minimum_valid_fraction, "minimum_valid_fraction")
    if not 0.0 < valid_minimum <= 1.0:
        raise ValueError("minimum_valid_fraction must lie in (0, 1].")
    if values.empty or bool(values.isna().any()):
        if values.empty:
            error = "Primary bootstrap inference requires at least one scheduled OOS observation."
            overall_status = ValidationAvailability.INSUFFICIENT_DATA
        else:
            error = (
                "Primary bootstrap inference is unavailable because scheduled OOS "
                "returns contain missing observations; no rows were dropped."
            )
            overall_status = ValidationAvailability.UNAVAILABLE
        empty_samples = pd.DataFrame(
            index=pd.RangeIndex(0, name="bootstrap_replicate"),
            columns=pd.RangeIndex(len(values), name="observation"),
            dtype=float,
        )
        empty_metrics = pd.DataFrame(columns=_METRIC_NAMES, dtype=float)
        unavailable = {
            name: _unavailable_metric(name, replicates, error)
            for name in _METRIC_NAMES
        }
        return BootstrapPerformanceResult(
            status=overall_status,
            error=error,
            observations=len(values),
            block_length=block,
            n_bootstrap=replicates,
            random_seed=seed,
            confidence_level=confidence,
            minimum_valid_fraction=valid_minimum,
            possible_block_starts=max(len(values) - block + 1, 0),
            effective_unique_replicates=0,
            inference_design_status=overall_status,
            inference_design_error=error,
            original_returns=values,
            bootstrap_samples=empty_samples,
            replicate_metrics=empty_metrics,
            **unavailable,
        )
    if block > len(values):
        raise ValueError("block_length must not exceed the number of observations.")

    samples = moving_block_bootstrap(values, block, replicates, seed)
    possible_block_starts = len(values) - block + 1
    effective_unique_replicates = int(
        np.unique(samples.to_numpy(dtype=float), axis=0).shape[0]
    )
    if len(values) < 2:
        design_status = ValidationAvailability.INSUFFICIENT_DATA
        design_error = (
            "At least two observations are required for bootstrap interval inference."
        )
    elif replicates < 2:
        design_status = ValidationAvailability.UNAVAILABLE
        design_error = (
            "At least two bootstrap replicates are required for interval inference."
        )
    elif possible_block_starts < 2:
        design_status = ValidationAvailability.UNAVAILABLE
        design_error = (
            "The block design has only one possible source block and no "
            "inferential resampling diversity."
        )
    elif effective_unique_replicates < 2:
        design_status = ValidationAvailability.UNAVAILABLE
        design_error = (
            "Generated replicates contain fewer than two distinct return paths."
        )
    else:
        design_status = ValidationAvailability.AVAILABLE
        design_error = None
    point = _metric_values(values, periods, risk_free)
    replicate_rows = []
    for row in samples.to_numpy(dtype=float):
        sample = pd.Series(row, dtype=float)
        replicate_rows.append(_metric_values(sample, periods, risk_free))
    replicate_metrics = pd.DataFrame(replicate_rows, columns=_METRIC_NAMES)
    alpha = (1.0 - confidence) / 2.0
    minimum_valid_count = ceil(replicates * valid_minimum)
    metric_results: dict[str, BootstrapMetricResult] = {}
    for name in _METRIC_NAMES:
        distribution = replicate_metrics[name]
        valid = distribution.loc[np.isfinite(distribution.to_numpy(dtype=float))]
        valid_count = len(valid)
        undefined_count = replicates - valid_count
        point_estimate = point[name]
        enough_finite = valid_count >= minimum_valid_count
        point_available = bool(np.isfinite(point_estimate))
        primary_available = (
            design_status is ValidationAvailability.AVAILABLE
            and enough_finite
            and point_available
            and undefined_count == 0
        )
        conditional_available = (
            design_status is ValidationAvailability.AVAILABLE
            and enough_finite
            and point_available
            and undefined_count > 0
        )
        if primary_available:
            interval = ConfidenceInterval(
                lower=float(valid.quantile(alpha)),
                upper=float(valid.quantile(1.0 - alpha)),
                confidence_level=confidence,
            )
            median = float(valid.median())
            error = None
            status = ValidationAvailability.AVAILABLE
        else:
            interval = None
            median = float("nan")
            status = (
                ValidationAvailability.INSUFFICIENT_DATA
                if design_status is ValidationAvailability.INSUFFICIENT_DATA
                else ValidationAvailability.UNAVAILABLE
            )
            if not point_available:
                error = "The point estimate is undefined."
            elif design_status is not ValidationAvailability.AVAILABLE:
                error = design_error
            elif undefined_count:
                error = (
                    "Primary inference is unavailable because one or more "
                    "replicates have undefined metric values."
                )
            else:
                error = (
                    "Too few valid bootstrap replicates for the requested "
                    "minimum fraction."
                )
        if conditional_available:
            conditional_interval = ConfidenceInterval(
                lower=float(valid.quantile(alpha)),
                upper=float(valid.quantile(1.0 - alpha)),
                confidence_level=confidence,
            )
            conditional_median = float(valid.median())
            conditional_status = ValidationAvailability.AVAILABLE
        else:
            conditional_interval = None
            conditional_median = None
            conditional_status = ValidationAvailability.UNAVAILABLE
        metric_results[name] = BootstrapMetricResult(
            metric=name,
            point_estimate=point_estimate,
            bootstrap_median=median,
            interval=interval,
            status=status,
            total_replicates=replicates,
            valid_replicates=valid_count,
            undefined_replicates=undefined_count,
            valid_fraction=float(valid_count / replicates),
            error=error,
            conditional_bootstrap_median=conditional_median,
            conditional_interval=conditional_interval,
            conditional_status=conditional_status,
            primary_interval_is_unconditional=primary_available,
        )
    if all(
        metric_results[name].status is ValidationAvailability.AVAILABLE
        for name in _METRIC_NAMES
    ):
        overall = ValidationAvailability.AVAILABLE
    elif design_status is ValidationAvailability.INSUFFICIENT_DATA:
        overall = ValidationAvailability.INSUFFICIENT_DATA
    else:
        overall = ValidationAvailability.UNAVAILABLE
    return BootstrapPerformanceResult(
        status=overall,
        error=(None if overall is ValidationAvailability.AVAILABLE else "One or more bootstrap metrics are unavailable."),
        observations=len(values),
        block_length=block,
        n_bootstrap=replicates,
        random_seed=seed,
        confidence_level=confidence,
        minimum_valid_fraction=valid_minimum,
        possible_block_starts=possible_block_starts,
        effective_unique_replicates=effective_unique_replicates,
        inference_design_status=design_status,
        inference_design_error=design_error,
        original_returns=values,
        bootstrap_samples=samples,
        replicate_metrics=replicate_metrics,
        **metric_results,
    )


def _unavailable_psr(
    observations: int,
    benchmark_sharpe: float,
    error: str,
    *,
    status: ValidationAvailability = ValidationAvailability.UNAVAILABLE,
) -> ProbabilisticSharpeResult:
    return ProbabilisticSharpeResult(
        status=status,
        probability=float("nan"),
        observations=observations,
        observed_sharpe_per_period=float("nan"),
        observed_sharpe_annualized=float("nan"),
        benchmark_sharpe_per_period=float("nan"),
        benchmark_sharpe_annualized=benchmark_sharpe,
        skewness=float("nan"),
        ordinary_kurtosis=float("nan"),
        variance_term=float("nan"),
        statistic=float("nan"),
        one_sided_pvalue=float("nan"),
        error=error,
    )


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> ProbabilisticSharpeResult:
    """Calculate the documented finite-sample PSR on a per-period scale."""
    values = _validated_returns(returns)
    benchmark = _finite_real(benchmark_sharpe, "benchmark_sharpe")
    periods = _integer(periods_per_year, "periods_per_year")
    risk_free = _finite_real(risk_free_rate, "risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")
    if bool(values.le(-1.0).any()):
        return _unavailable_psr(
            len(values),
            benchmark,
            "PSR inference is unavailable because at least one return is at "
            "or below -100%; catastrophic observations were retained.",
        )
    if bool(values.isna().any()):
        return _unavailable_psr(
            len(values), benchmark,
            "PSR is unavailable because scheduled OOS returns contain missing observations.",
        )
    if len(values) < 4:
        return _unavailable_psr(
            len(values), benchmark,
            "At least four observations are required for skewness and kurtosis estimation.",
            status=ValidationAvailability.INSUFFICIENT_DATA,
        )
    array = values.to_numpy(dtype=float)
    period_risk_free = float(np.expm1(np.log1p(risk_free) / periods))
    excess = array - period_risk_free
    volatility = float(np.std(excess, ddof=1))
    scale = max(1.0, float(np.max(np.abs(excess))))
    if (
        not np.isfinite(volatility)
        or abs(volatility) <= NEAR_ZERO_TOLERANCE * scale
    ):
        return _unavailable_psr(len(values), benchmark, "Observed Sharpe is undefined because return volatility is zero.")
    observed_period = float(np.mean(excess) / volatility)
    benchmark_period = float(benchmark / np.sqrt(periods))
    skewness = float(skew(array, bias=False))
    ordinary_kurtosis = float(kurtosis(array, fisher=False, bias=False))
    variance_term = float(
        1.0
        - skewness * observed_period
        + ((ordinary_kurtosis - 1.0) / 4.0) * observed_period**2
    )
    if (
        not np.isfinite(skewness)
        or not np.isfinite(ordinary_kurtosis)
        or not np.isfinite(variance_term)
        or variance_term <= 0.0
    ):
        return ProbabilisticSharpeResult(
            status=ValidationAvailability.UNAVAILABLE,
            probability=float("nan"),
            observations=len(values),
            observed_sharpe_per_period=observed_period,
            observed_sharpe_annualized=observed_period * float(np.sqrt(periods)),
            benchmark_sharpe_per_period=benchmark_period,
            benchmark_sharpe_annualized=benchmark,
            skewness=skewness,
            ordinary_kurtosis=ordinary_kurtosis,
            variance_term=variance_term,
            statistic=float("nan"),
            one_sided_pvalue=float("nan"),
            error="The PSR variance term must be finite and strictly positive.",
        )
    statistic = (
        (observed_period - benchmark_period)
        * float(np.sqrt(len(values) - 1))
        / float(np.sqrt(variance_term))
    )
    if not np.isfinite(statistic):
        return ProbabilisticSharpeResult(
            status=ValidationAvailability.UNAVAILABLE,
            probability=float("nan"),
            observations=len(values),
            observed_sharpe_per_period=observed_period,
            observed_sharpe_annualized=observed_period * float(np.sqrt(periods)),
            benchmark_sharpe_per_period=benchmark_period,
            benchmark_sharpe_annualized=benchmark,
            skewness=skewness,
            ordinary_kurtosis=ordinary_kurtosis,
            variance_term=variance_term,
            statistic=float("nan"),
            one_sided_pvalue=float("nan"),
            error="The PSR statistic is not finite.",
        )
    probability = float(norm.cdf(statistic))
    one_sided_pvalue = float(norm.sf(statistic))
    return ProbabilisticSharpeResult(
        status=ValidationAvailability.AVAILABLE,
        probability=probability,
        observations=len(values),
        observed_sharpe_per_period=observed_period,
        observed_sharpe_annualized=observed_period * float(np.sqrt(periods)),
        benchmark_sharpe_per_period=benchmark_period,
        benchmark_sharpe_annualized=benchmark,
        skewness=skewness,
        ordinary_kurtosis=ordinary_kurtosis,
        variance_term=variance_term,
        statistic=float(statistic),
        one_sided_pvalue=one_sided_pvalue,
        error=None,
    )


def minimum_track_record_length(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    *,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> MinimumTrackRecordResult:
    """Estimate required observations under the same finite-sample PSR approximation."""
    confidence = _confidence_level(confidence_level)
    if confidence <= 0.5:
        raise ValueError("confidence_level must be greater than 0.5 for track-record evidence.")
    benchmark = _finite_real(benchmark_sharpe, "benchmark_sharpe")
    psr = probabilistic_sharpe_ratio(
        returns,
        benchmark,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    if psr.status is not ValidationAvailability.AVAILABLE:
        return MinimumTrackRecordResult(
            status=psr.status,
            current_observations=psr.observations,
            estimated_required_observations=None,
            sufficient_track_record=None,
            confidence_level=confidence,
            observed_sharpe_annualized=psr.observed_sharpe_annualized,
            benchmark_sharpe_annualized=benchmark,
            error=psr.error,
        )
    difference = psr.observed_sharpe_per_period - psr.benchmark_sharpe_per_period
    if difference <= 0.0:
        return MinimumTrackRecordResult(
            status=ValidationAvailability.UNAVAILABLE,
            current_observations=psr.observations,
            estimated_required_observations=None,
            sufficient_track_record=None,
            confidence_level=confidence,
            observed_sharpe_annualized=psr.observed_sharpe_annualized,
            benchmark_sharpe_annualized=benchmark,
            error="Observed Sharpe does not exceed the benchmark Sharpe.",
        )
    quantile = float(norm.ppf(confidence))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = float(np.divide(quantile, difference))
        theoretical_required = float(
            1.0 + psr.variance_term * np.square(ratio)
        )
    if (
        not np.isfinite(ratio)
        or not np.isfinite(theoretical_required)
        or theoretical_required > sys.maxsize
    ):
        return MinimumTrackRecordResult(
            status=ValidationAvailability.UNAVAILABLE,
            current_observations=psr.observations,
            estimated_required_observations=None,
            sufficient_track_record=None,
            confidence_level=confidence,
            observed_sharpe_annualized=psr.observed_sharpe_annualized,
            benchmark_sharpe_annualized=benchmark,
            error=(
                "The theoretical minimum track record is numerically unbounded "
                "or exceeds the safely representable integer range."
            ),
        )
    required = max(4, int(ceil(theoretical_required)))
    return MinimumTrackRecordResult(
        status=ValidationAvailability.AVAILABLE,
        current_observations=psr.observations,
        estimated_required_observations=required,
        sufficient_track_record=psr.observations >= required,
        confidence_level=confidence,
        observed_sharpe_annualized=psr.observed_sharpe_annualized,
        benchmark_sharpe_annualized=benchmark,
        error=None,
    )


def _safe_fold_metrics(
    returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
) -> tuple[
    ValidationAvailability,
    float,
    float,
    float,
    float,
    bool,
    bool,
    str | None,
]:
    values = _validated_returns(returns)
    if values.empty or bool(values.isna().any()):
        return (
            ValidationAvailability.UNAVAILABLE,
            *(float("nan"),) * 4,
            False,
            False,
            "Fold returns are empty or unavailable.",
        )
    catastrophic = bool(values.eq(-1.0).any())
    invalid_return = bool(values.lt(-1.0).any())
    metrics = _metric_values(values, periods_per_year, risk_free_rate)
    total = metrics["total_return"]
    mean = float(values.mean())
    volatility = metrics["annualized_volatility"]
    sharpe = metrics["sharpe_ratio"]
    if invalid_return:
        return (
            ValidationAvailability.UNAVAILABLE,
            float("nan"),
            mean,
            volatility,
            sharpe,
            False,
            True,
            "A fold return below -100% is invalid under portfolio-return semantics.",
        )
    if catastrophic:
        return (
            ValidationAvailability.UNAVAILABLE,
            float("nan"),
            mean,
            volatility,
            sharpe,
            True,
            False,
            "The fold contains an exactly -100% catastrophic return.",
        )
    if (
        len(values) < 2
        or not np.isfinite(volatility)
        or not np.isfinite(sharpe)
    ):
        return (
            ValidationAvailability.INSUFFICIENT_DATA,
            total,
            mean,
            volatility,
            sharpe,
            False,
            False,
            "Fold risk analytics require at least two observations and non-zero volatility.",
        )
    if not np.isfinite(total):
        return (
            ValidationAvailability.UNAVAILABLE,
            float("nan"),
            mean,
            volatility,
            sharpe,
            False,
            False,
            "Fold total return is undefined.",
        )
    return (
        ValidationAvailability.AVAILABLE,
        total,
        mean,
        volatility,
        sharpe,
        False,
        False,
        None,
    )


def analyze_fold_consistency(
    walk_forward_result: WalkForwardResult,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> FoldConsistencyMetrics:
    """Describe dispersion and positive contribution across all scheduled folds."""
    if not isinstance(walk_forward_result, WalkForwardResult):
        raise TypeError("walk_forward_result must be a WalkForwardResult.")
    periods = _integer(periods_per_year, "periods_per_year")
    risk_free = _finite_real(risk_free_rate, "risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")
    records: list[FoldPerformance] = []
    for fold_result in walk_forward_result.folds:
        if fold_result.status is WalkForwardStatus.NO_SELECTION:
            records.append(FoldPerformance(
                fold_id=fold_result.fold.fold_id,
                selection_status=fold_result.status.value,
                availability=ValidationAvailability.AVAILABLE,
                observations=fold_result.trading_observations,
                total_return=0.0,
                mean_return=0.0,
                annualized_volatility=0.0,
                sharpe_ratio=float("nan"),
                trade_count=0,
                catastrophic=False,
                invalid_return=False,
                error=None,
            ))
            continue
        if fold_result.status is WalkForwardStatus.INSUFFICIENT_DATA:
            records.append(FoldPerformance(
                fold_id=fold_result.fold.fold_id,
                selection_status=fold_result.status.value,
                availability=ValidationAvailability.UNAVAILABLE,
                observations=fold_result.trading_observations,
                total_return=float("nan"), mean_return=float("nan"),
                annualized_volatility=float("nan"), sharpe_ratio=float("nan"),
                trade_count=0, catastrophic=False, invalid_return=False,
                error=fold_result.message,
            ))
            continue
        (
            availability,
            total,
            mean,
            volatility,
            sharpe,
            catastrophic,
            invalid_return,
            error,
        ) = _safe_fold_metrics(
            fold_result.oos_returns, periods, risk_free
        )
        records.append(FoldPerformance(
            fold_id=fold_result.fold.fold_id,
            selection_status=fold_result.status.value,
            availability=availability,
            observations=len(fold_result.oos_returns),
            total_return=total, mean_return=mean,
            annualized_volatility=volatility, sharpe_ratio=sharpe,
            trade_count=fold_result.trade_count,
            catastrophic=catastrophic,
            invalid_return=invalid_return,
            error=error,
        ))
    observable = tuple(
        record for record in records if np.isfinite(record.total_return)
    )
    analytically_available = tuple(
        record
        for record in records
        if record.availability is ValidationAvailability.AVAILABLE
    )
    catastrophic_records = tuple(record for record in records if record.catastrophic)
    invalid_records = tuple(record for record in records if record.invalid_return)
    totals = np.asarray([record.total_return for record in observable], dtype=float)
    positive = totals[totals > 0.0]
    positive_sum = float(positive.sum())
    summaries_complete = (
        bool(records)
        and len(observable) == len(records)
        and not catastrophic_records
        and not invalid_records
    )
    if len(positive) and summaries_complete:
        top_count = int(ceil(0.20 * len(positive)))
        ordered_positive = np.sort(positive)[::-1]
        strongest_one = float(ordered_positive[0] / positive_sum)
        strongest_twenty = float(ordered_positive[:top_count].sum() / positive_sum)
    else:
        top_count = 0
        strongest_one = float("nan")
        strongest_twenty = float("nan")
    quantiles = (
        np.quantile(totals, [0.25, 0.50, 0.75])
        if len(totals) and summaries_complete
        else np.full(3, np.nan)
    )
    unavailable_count = sum(
        record.availability is not ValidationAvailability.AVAILABLE
        for record in records
    )
    insufficient_count = sum(
        record.availability is ValidationAvailability.INSUFFICIENT_DATA
        for record in records
    )
    pre_execution_insufficient_count = sum(
        record.selection_status == WalkForwardStatus.INSUFFICIENT_DATA.value
        for record in records
    )
    if any(
        record.availability is ValidationAvailability.UNAVAILABLE
        for record in records
    ):
        overall_status = ValidationAvailability.UNAVAILABLE
    elif insufficient_count:
        overall_status = ValidationAvailability.INSUFFICIENT_DATA
    else:
        overall_status = ValidationAvailability.AVAILABLE
    adverse_special_count = len(catastrophic_records) + len(invalid_records)
    evidence_count = len(totals) + adverse_special_count
    return FoldConsistencyMetrics(
        status=overall_status,
        folds=tuple(records), fold_count=len(records), observable_fold_count=len(observable),
        positive_return_fold_count=int((totals > 0.0).sum()),
        negative_return_fold_count=int((totals < 0.0).sum()) + adverse_special_count,
        zero_return_no_selection_fold_count=sum(record.selection_status == WalkForwardStatus.NO_SELECTION.value for record in records),
        unavailable_fold_count=unavailable_count,
        analytically_available_fold_count=len(analytically_available),
        insufficient_data_fold_count=insufficient_count,
        pre_execution_insufficient_fold_count=pre_execution_insufficient_count,
        risk_analytics_insufficient_fold_count=insufficient_count,
        catastrophic_fold_count=len(catastrophic_records),
        catastrophic_fold_ids=tuple(record.fold_id for record in catastrophic_records),
        invalid_return_fold_count=len(invalid_records),
        summaries_complete=summaries_complete,
        fraction_observable_folds_positive=(
            float((totals > 0.0).sum() / evidence_count)
            if evidence_count else float("nan")
        ),
        median_fold_total_return=float(quantiles[1]),
        lower_quartile_fold_total_return=float(quantiles[0]),
        upper_quartile_fold_total_return=float(quantiles[2]),
        worst_fold_return=(
            float(totals.min())
            if len(totals) and summaries_complete else float("nan")
        ),
        strongest_fold_return=(
            float(totals.max())
            if len(totals) and summaries_complete else float("nan")
        ),
        strongest_single_positive_fold_concentration=strongest_one,
        strongest_twenty_percent_positive_folds_concentration=strongest_twenty,
        strongest_twenty_percent_positive_fold_count=top_count,
        contribution_basis="equal_capital_reset_fold_total_return_positive_contributions",
        warning=(
            "Fold contributions describe independently reset folds and are not "
            "additive self-financing portfolio P&L. Aggregate return summaries "
            "are incomplete when any scheduled fold lacks finite return evidence "
            "or contains a catastrophic or invalid return. Conditional observable-"
            "fold diagnostics retain their explicitly labelled denominator."
        ),
    )


def analyze_regime_consistency(
    returns: pd.Series,
    regime_labels: pd.Series,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> RegimeConsistencyMetrics:
    """Describe performance across exactly aligned caller-supplied regimes."""
    values = _validated_returns(returns)
    if not isinstance(regime_labels, pd.Series):
        raise TypeError("regime_labels must be a pandas Series.")
    if not regime_labels.index.is_unique:
        raise ValueError("regime_labels must have a unique index.")
    if not values.index.equals(regime_labels.index):
        raise ValueError("returns and regime_labels must have exactly aligned indices.")
    labels = regime_labels.copy(deep=True)
    periods = _integer(periods_per_year, "periods_per_year")
    risk_free = _finite_real(risk_free_rate, "risk_free_rate")
    if risk_free <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")
    normalized_labels: list[Any] = []
    for label in labels.tolist():
        if label is None or label is pd.NA or label is pd.NaT:
            raise ValueError("regime_labels must not be missing on evaluated rows.")
        if isinstance(label, (bool, np.bool_)):
            raise ValueError(
                "regime labels must be stable immutable scalar values."
            )
        if isinstance(label, Real):
            if not np.isfinite(float(label)):
                raise ValueError(
                    "Numeric regime labels must be finite and non-missing."
                )
            normalized_labels.append(label)
            continue
        if isinstance(label, np.datetime64):
            if np.isnat(label):
                raise ValueError(
                    "regime_labels must not be missing on evaluated rows."
                )
            normalized_labels.append(label)
            continue
        if isinstance(label, (str, Enum, date, datetime, pd.Timestamp)):
            normalized_labels.append(label)
            continue
        raise ValueError(
            "regime labels must be immutable scalar strings, finite numerics, "
            "enums, or timestamp-like values; nested mutable objects are rejected."
        )
    labels = pd.Series(
        normalized_labels,
        index=labels.index.copy(),
        name=labels.name,
    )
    unique_labels = tuple(dict.fromkeys(normalized_labels))
    if values.empty:
        return RegimeConsistencyMetrics(
            status=ValidationAvailability.INSUFFICIENT_DATA,
            regimes=(),
            regime_count=0,
            observations=0,
            positive_regime_count=0,
            negative_regime_count=0,
            zero_regime_count=0,
            worst_regime=None,
            strongest_regime_positive_contribution_concentration=float("nan"),
            contribution_basis="arithmetic_return_sum_diagnostic",
            contribution_warning=(
                "Regime contributions are arithmetic return-sum diagnostics, "
                "not compounded portfolio wealth attribution."
            ),
            returns=values,
            regime_labels=labels,
            point_in_time_regime_labels_validated=False,
            provenance_warning=REGIME_PROVENANCE_WARNING,
            error="Regime analysis requires at least one aligned observation.",
        )
    if bool(values.isna().any()):
        return RegimeConsistencyMetrics(
            status=ValidationAvailability.UNAVAILABLE, regimes=(), regime_count=len(unique_labels),
            observations=len(values), positive_regime_count=0, negative_regime_count=0,
            zero_regime_count=0, worst_regime=None,
            strongest_regime_positive_contribution_concentration=float("nan"),
            contribution_basis="arithmetic_return_sum_diagnostic",
            contribution_warning=(
                "Regime contributions are arithmetic return-sum diagnostics, "
                "not compounded portfolio wealth attribution."
            ),
            returns=values, regime_labels=labels,
            point_in_time_regime_labels_validated=False,
            provenance_warning=REGIME_PROVENANCE_WARNING,
            error="Regime inference is unavailable because scheduled OOS returns contain missing observations.",
        )
    regimes: list[RegimePerformance] = []
    for label in unique_labels:
        subset = values.loc[labels == label]
        metrics = _metric_values(subset, periods, risk_free)
        total = metrics["total_return"]
        if bool(subset.le(-1.0).any()) or not np.isfinite(total):
            availability = ValidationAvailability.UNAVAILABLE
            error = "Geometric regime analytics are unavailable."
        elif (
            len(subset) < 2
            or not np.isfinite(metrics["annualized_volatility"])
            or not np.isfinite(metrics["sharpe_ratio"])
        ):
            availability = ValidationAvailability.INSUFFICIENT_DATA
            error = (
                "Regime risk analytics require at least two observations and "
                "non-zero volatility."
            )
        else:
            availability = ValidationAvailability.AVAILABLE
            error = None
        regimes.append(RegimePerformance(
            regime_label=label, availability=availability, observations=len(subset),
            total_return=total, mean_return=float(subset.mean()),
            annualized_volatility=metrics["annualized_volatility"],
            sharpe_ratio=metrics["sharpe_ratio"], maximum_drawdown=metrics["maximum_drawdown"],
            fraction_positive_observations=float(subset.gt(0.0).mean()),
            arithmetic_return_contribution=float(subset.sum()),
            error=error,
        ))
    contributions = np.asarray([item.arithmetic_return_contribution for item in regimes], dtype=float)
    positive_contributions = contributions[contributions > 0.0]
    concentration = (
        float(positive_contributions.max() / positive_contributions.sum())
        if len(positive_contributions) else float("nan")
    )
    worst_position = int(np.argmin(contributions))
    if any(
        item.availability is ValidationAvailability.UNAVAILABLE
        for item in regimes
    ):
        overall_status = ValidationAvailability.UNAVAILABLE
    elif any(
        item.availability is ValidationAvailability.INSUFFICIENT_DATA
        for item in regimes
    ):
        overall_status = ValidationAvailability.INSUFFICIENT_DATA
    else:
        overall_status = ValidationAvailability.AVAILABLE
    return RegimeConsistencyMetrics(
        status=overall_status,
        regimes=tuple(regimes), regime_count=len(regimes), observations=len(values),
        positive_regime_count=int((contributions > 0.0).sum()),
        negative_regime_count=int((contributions < 0.0).sum()),
        zero_regime_count=int((contributions == 0.0).sum()),
        worst_regime=regimes[worst_position].regime_label,
        strongest_regime_positive_contribution_concentration=concentration,
        contribution_basis="arithmetic_return_sum_diagnostic",
        contribution_warning=(
            "Regime contributions are arithmetic return-sum diagnostics, not "
            "compounded portfolio wealth attribution."
        ),
        returns=values, regime_labels=labels,
        point_in_time_regime_labels_validated=False,
        provenance_warning=REGIME_PROVENANCE_WARNING, error=None,
    )


def _benjamini_hochberg(
    pvalues: dict[str, float],
    family_size: int | None = None,
) -> dict[str, float]:
    """Adjust finite p-values using the full predefined eligible family size."""
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    finite_count = len(ordered)
    count = finite_count if family_size is None else family_size
    if count < finite_count:
        raise ValueError("family_size must not be smaller than finite p-value count.")
    adjusted = [0.0] * finite_count
    running = 1.0
    for position in range(finite_count - 1, -1, -1):
        rank = position + 1
        candidate = min(ordered[position][1] * count / rank, 1.0)
        running = min(running, candidate)
        adjusted[position] = running
    return {scenario_id: float(value) for (scenario_id, _), value in zip(ordered, adjusted)}


def multiple_testing_diagnostics(
    robustness_result: RobustnessResult,
    *,
    benchmark_sharpe: float = 0.0,
    significance_level: float = 0.05,
) -> MultipleTestingDiagnostics:
    """Apply diagnostic scenario-level Bonferroni and BH corrections."""
    if not isinstance(robustness_result, RobustnessResult):
        raise TypeError("robustness_result must be a RobustnessResult.")
    scenario_ids = [
        result.scenario.scenario_id for result in robustness_result.scenarios
    ]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(
            "Scenario IDs must be unique before multiple-testing diagnostics."
        )
    benchmark = _finite_real(benchmark_sharpe, "benchmark_sharpe")
    alpha = _confidence_level(significance_level, "significance_level")
    raw: dict[str, float] = {}
    psr_results: dict[str, ProbabilisticSharpeResult] = {}
    invalid = sum(
        result.status is ScenarioStatus.INVALID_CONFIGURATION
        for result in robustness_result.scenarios
    )
    failed = sum(
        result.status is ScenarioStatus.FAILED
        for result in robustness_result.scenarios
    )
    eligible_results = tuple(
        result
        for result in robustness_result.scenarios
        if result.status
        not in {ScenarioStatus.INVALID_CONFIGURATION, ScenarioStatus.FAILED}
    )
    eligible_count = len(eligible_results)
    for result in robustness_result.scenarios:
        structurally_eligible = result.status not in {
            ScenarioStatus.INVALID_CONFIGURATION,
            ScenarioStatus.FAILED,
        }
        inferentially_available = (
            structurally_eligible
            and robustness_result.common_horizon_structurally_available
            and result.common_horizon_structurally_available
            and result.common_horizon_fully_observed
            and result.common_horizon_analytics_status
            is MetricAvailabilityStatus.AVAILABLE
            and len(result.common_horizon_returns) > 0
        )
        if (
            inferentially_available
        ):
            psr = probabilistic_sharpe_ratio(
                result.common_horizon_returns,
                benchmark,
                periods_per_year=result.periods_per_year,
                risk_free_rate=result.risk_free_rate,
            )
        else:
            if structurally_eligible:
                reason = (
                    "Scenario is structurally eligible but lacks fully observed, "
                    "available common-horizon analytics."
                )
            else:
                reason = (
                    "Scenario is invalid or failed and is not an eligible hypothesis."
                )
            psr = _unavailable_psr(
                len(result.common_horizon_returns), benchmark,
                reason,
            )
        psr_results[result.scenario.scenario_id] = psr
        if (
            structurally_eligible
            and psr.status is ValidationAvailability.AVAILABLE
            and np.isfinite(psr.one_sided_pvalue)
            and 0.0 <= psr.one_sided_pvalue <= 1.0
        ):
            raw[result.scenario.scenario_id] = float(psr.one_sided_pvalue)
    finite_count = len(raw)
    unavailable_eligible = eligible_count - finite_count
    bonferroni = {
        scenario_id: min(pvalue * eligible_count, 1.0)
        for scenario_id, pvalue in raw.items()
    }
    bh = _benjamini_hochberg(raw, eligible_count)
    scenario_rows = tuple(
        ScenarioPValue(
            scenario_id=result.scenario.scenario_id,
            scenario_status=result.status.value,
            psr_status=psr_results[result.scenario.scenario_id].status,
            raw_pvalue=raw.get(result.scenario.scenario_id, float("nan")),
            bonferroni_adjusted_pvalue=bonferroni.get(result.scenario.scenario_id, float("nan")),
            benjamini_hochberg_adjusted_pvalue=bh.get(result.scenario.scenario_id, float("nan")),
            error=psr_results[result.scenario.scenario_id].error,
        )
        for result in sorted(robustness_result.scenarios, key=lambda item: item.scenario.scenario_id)
    )
    axes = tuple(ParameterAxisCount(axis.parameter, axis.count) for axis in robustness_result.axis_metadata)
    return MultipleTestingDiagnostics(
        scenario_pvalues=scenario_rows,
        total_tested_configurations=len(robustness_result.scenarios),
        valid_comparable_configurations=eligible_count,
        invalid_configurations=invalid,
        analytically_unavailable_configurations=unavailable_eligible,
        failed_configurations=failed,
        valid_pvalue_count=finite_count,
        unavailable_pvalue_count=(
            len(robustness_result.scenarios) - finite_count
        ),
        eligible_hypothesis_count=eligible_count,
        finite_pvalue_count=finite_count,
        unavailable_eligible_hypothesis_count=unavailable_eligible,
        family_size=eligible_count,
        tested_dimension_count=len(robustness_result.tested_dimensions),
        parameter_axis_counts=axes,
        significance_level=alpha,
        bonferroni_valid_test_count=eligible_count,
        raw_threshold_exceedance_count=sum(value <= alpha for value in raw.values()),
        bonferroni_threshold_exceedance_count=sum(value <= alpha for value in bonferroni.values()),
        benjamini_hochberg_discovery_count=sum(value <= alpha for value in bh.values()),
        baseline_scenario_id=robustness_result.baseline_scenario_id,
    )


def _empty_fold_consistency() -> FoldConsistencyMetrics:
    return FoldConsistencyMetrics(
        status=ValidationAvailability.INSUFFICIENT_DATA,
        folds=(), fold_count=0, observable_fold_count=0,
        positive_return_fold_count=0, negative_return_fold_count=0,
        zero_return_no_selection_fold_count=0, unavailable_fold_count=0,
        analytically_available_fold_count=0,
        insufficient_data_fold_count=0,
        pre_execution_insufficient_fold_count=0,
        risk_analytics_insufficient_fold_count=0,
        catastrophic_fold_count=0,
        catastrophic_fold_ids=(),
        invalid_return_fold_count=0,
        summaries_complete=False,
        fraction_observable_folds_positive=float("nan"),
        median_fold_total_return=float("nan"), lower_quartile_fold_total_return=float("nan"),
        upper_quartile_fold_total_return=float("nan"), worst_fold_return=float("nan"),
        strongest_fold_return=float("nan"), strongest_single_positive_fold_concentration=float("nan"),
        strongest_twenty_percent_positive_folds_concentration=float("nan"),
        strongest_twenty_percent_positive_fold_count=0,
        contribution_basis="equal_capital_reset_fold_total_return_positive_contributions",
        warning="No baseline walk-forward folds are available.",
    )


def build_statistical_validation_report(
    robustness_result: RobustnessResult,
    *,
    block_length: int,
    n_bootstrap: int,
    random_seed: int = 0,
    benchmark_sharpe: float = 0.0,
    confidence_level: float = 0.95,
    minimum_valid_fraction: float = 0.80,
    significance_level: float = 0.05,
    regime_labels: pd.Series | None = None,
) -> StatisticalValidationResult:
    """Compose standalone diagnostics for the predefined baseline scenario."""
    if not isinstance(robustness_result, RobustnessResult):
        raise TypeError("robustness_result must be a RobustnessResult.")
    baseline = robustness_result.baseline_result
    primary = baseline.common_horizon_returns.copy(deep=True)
    periods = baseline.periods_per_year
    risk_free = baseline.risk_free_rate
    bootstrap = bootstrap_performance_metrics(
        primary, block_length, n_bootstrap, random_seed=random_seed,
        periods_per_year=periods, risk_free_rate=risk_free,
        confidence_level=confidence_level,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    psr = probabilistic_sharpe_ratio(
        primary, benchmark_sharpe,
        periods_per_year=periods, risk_free_rate=risk_free,
    )
    track_record = minimum_track_record_length(
        primary, benchmark_sharpe,
        confidence_level=confidence_level,
        periods_per_year=periods, risk_free_rate=risk_free,
    )
    fold_consistency = (
        analyze_fold_consistency(
            baseline.walk_forward_result,
            periods_per_year=periods,
            risk_free_rate=risk_free,
        )
        if baseline.walk_forward_result is not None
        else _empty_fold_consistency()
    )
    multiple = multiple_testing_diagnostics(
        robustness_result,
        benchmark_sharpe=benchmark_sharpe,
        significance_level=significance_level,
    )
    regime = (
        analyze_regime_consistency(
            primary, regime_labels,
            periods_per_year=periods,
            risk_free_rate=risk_free,
        )
        if regime_labels is not None
        else None
    )
    primary_inference_availability = (
        ValidationAvailability.AVAILABLE
        if bootstrap.status is ValidationAvailability.AVAILABLE
        and psr.status is ValidationAvailability.AVAILABLE
        and track_record.status is ValidationAvailability.AVAILABLE
        else ValidationAvailability.UNAVAILABLE
    )
    requested_availability = [
        primary_inference_availability,
        fold_consistency.status,
    ]
    if regime is not None:
        requested_availability.append(regime.status)
    if ValidationAvailability.UNAVAILABLE in requested_availability:
        overall_availability = ValidationAvailability.UNAVAILABLE
    elif ValidationAvailability.INSUFFICIENT_DATA in requested_availability:
        overall_availability = ValidationAvailability.INSUFFICIENT_DATA
    else:
        overall_availability = ValidationAvailability.AVAILABLE
    provenance = tuple(
        dict.fromkeys(
            (*robustness_result.provenance_warnings,
             *((REGIME_PROVENANCE_WARNING,) if regime is not None else ()))
        )
    )
    return StatisticalValidationResult(
        bootstrap=bootstrap,
        probabilistic_sharpe=psr,
        minimum_track_record=track_record,
        fold_consistency=fold_consistency,
        multiple_testing=multiple,
        regime_consistency=regime,
        primary_oos_returns=primary,
        observations=len(primary),
        primary_inference_availability=primary_inference_availability,
        overall_availability=overall_availability,
        availability=primary_inference_availability,
        validation_warnings=VALIDATION_WARNINGS,
        provenance_warnings=provenance,
    )
