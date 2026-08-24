"""Typed orchestration for reproducible pairs-trading research experiments.

This module composes the package's existing research APIs.  It does not place
orders, select parameters from out-of-sample performance, or certify caller
data provenance.  The selected-pair backtest is an explicitly in-sample
diagnostic; walk-forward calendar returns remain the package's OOS evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .analytics import StrategyPerformanceReport, build_performance_report
from .backtest import BacktestResult, run_pair_backtest, validate_backtest_invariants
from .config import (
    CostConfig,
    DataConfig,
    ResearchConfig,
    ScreeningConfig,
    StrategyConfig,
    WalkForwardConfig,
    screening_kwargs_from_config,
)
from .data import OBSERVED_PRICE_MASK_ATTR
from .robustness import (
    DISTRIBUTION_POLICY,
    HEADLINE_METRIC_BASIS,
    MetricAvailabilityStatus,
    NO_OPTIMIZATION_WARNING,
    ParameterScenario,
    PURPOSE as ROBUSTNESS_PURPOSE,
    RobustnessResult,
    ScenarioStatus,
    run_sensitivity_analysis,
    summarize_sensitivity,
)
from .screening import PairScreeningResult, screen_pairs
from .signals import rolling_zscore
from .stats import rolling_ols_spread
from .validation import (
    PURPOSE as STATISTICAL_VALIDATION_PURPOSE,
    StatisticalValidationResult,
    ValidationAvailability,
    build_statistical_validation_report,
)
from .walkforward import (
    WalkForwardAnalyticsStatus,
    WalkForwardResult,
    WalkForwardStatus,
    run_walk_forward_analysis,
)


__all__ = [
    "RESEARCH_PIPELINE_VERSION",
    "CONFIGURATION_SNAPSHOT_VERSION",
    "EXPERIMENT_SCHEMA_VERSION",
    "PipelineStageStatus",
    "ResearchPipelineStatus",
    "DiagnosticExecutionCoverage",
    "StatisticalValidationSettings",
    "ResearchExperimentRequest",
    "ResearchExperimentResult",
    "build_configuration_snapshot",
    "configuration_digest",
    "price_content_digest",
    "research_content_digest",
    "validate_research_experiment_result",
    "run_research_pipeline",
]


RESEARCH_PIPELINE_VERSION = "10A.1"
CONFIGURATION_SNAPSHOT_VERSION = 1
EXPERIMENT_SCHEMA_VERSION = 1


class ResearchPipelineStatus(str, Enum):
    """Terminal status of one research orchestration run."""

    COMPLETED = "COMPLETED"
    NO_PAIR_SELECTED = "NO_PAIR_SELECTED"


class PipelineStageStatus(str, Enum):
    """Whether one pipeline stage ran, was omitted, or lacked a prerequisite."""

    COMPLETED = "COMPLETED"
    NOT_REQUESTED = "NOT_REQUESTED"
    UNAVAILABLE = "UNAVAILABLE"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive non-Boolean integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _finite_real(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-Boolean real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _json_value(value: Any, path: str = "value") -> Any:
    """Return deterministic JSON primitives or reject unsupported metadata."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return result
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be non-empty strings.")
            normalized[key] = _json_value(nested, f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{path}[]") for item in value]
    raise TypeError(f"{path} contains a value that is not JSON serialisable.")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class StatisticalValidationSettings:
    """Explicit settings for the optional bootstrap/statistical stage."""

    block_length: int = 5
    n_bootstrap: int = 1_000
    random_seed: int | None = None
    benchmark_sharpe: float = 0.0
    confidence_level: float = 0.95
    minimum_valid_fraction: float = 0.80
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_length", _positive_integer(self.block_length, "block_length")
        )
        object.__setattr__(
            self, "n_bootstrap", _positive_integer(self.n_bootstrap, "n_bootstrap")
        )
        if self.random_seed is not None:
            if isinstance(self.random_seed, (bool, np.bool_)) or not isinstance(
                self.random_seed, Integral
            ):
                raise TypeError("random_seed must be a non-Boolean integer or None.")
            object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(
            self,
            "benchmark_sharpe",
            _finite_real(self.benchmark_sharpe, "benchmark_sharpe"),
        )
        for name in ("confidence_level", "minimum_valid_fraction", "significance_level"):
            value = _finite_real(getattr(self, name), name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between zero and one.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ResearchExperimentRequest:
    """Owned inputs for one deterministic research experiment.

    Robustness scenarios are caller-defined before any OOS results are
    observed.  When robustness is requested without a supplied grid, the
    pipeline runs one predefined baseline scenario only; it does not optimize.
    """

    experiment_name: str
    prices: pd.DataFrame
    config: ResearchConfig
    initial_capital: float = 1_000_000.0
    target_gross_notional: float = 100_000.0
    periods_per_year: int = 252
    risk_free_rate: float = 0.0
    run_robustness: bool = False
    run_statistical_validation: bool = False
    robustness_scenarios: tuple[ParameterScenario, ...] = ()
    robustness_baseline_scenario_id: str | None = None
    validation_settings: StatisticalValidationSettings = field(
        default_factory=StatisticalValidationSettings
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string.")
        object.__setattr__(self, "experiment_name", self.experiment_name.strip())
        if not isinstance(self.prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        owned_prices = self.prices.copy(deep=True)
        owned_prices.attrs = deepcopy(self.prices.attrs)
        object.__setattr__(self, "prices", owned_prices)
        if not isinstance(self.config, ResearchConfig):
            raise TypeError("config must be a ResearchConfig instance.")
        expected_sections = {
            "data": DataConfig,
            "screening": ScreeningConfig,
            "strategy": StrategyConfig,
            "costs": CostConfig,
            "walk_forward": WalkForwardConfig,
        }
        for section_name, section_type in expected_sections.items():
            if not isinstance(getattr(self.config, section_name), section_type):
                raise TypeError(
                    f"config.{section_name} must be a {section_type.__name__} instance."
                )
        if not isinstance(self.config.universe, Mapping):
            raise TypeError("config.universe must be a mapping.")
        for group, symbols in self.config.universe.items():
            if not isinstance(group, str) or not group.strip():
                raise ValueError("config.universe group names must be non-empty strings.")
            if isinstance(symbols, (str, bytes)):
                raise TypeError("config.universe symbol groups must be sequences.")
        owned_universe = MappingProxyType(
            {
                group: tuple(symbols)
                for group, symbols in self.config.universe.items()
            }
        )
        object.__setattr__(
            self,
            "config",
            ResearchConfig(
                data=self.config.data,
                universe=owned_universe,
                screening=self.config.screening,
                strategy=self.config.strategy,
                costs=self.config.costs,
                walk_forward=self.config.walk_forward,
                random_seed=self.config.random_seed,
            ),
        )
        object.__setattr__(
            self,
            "initial_capital",
            _finite_real(self.initial_capital, "initial_capital", positive=True),
        )
        object.__setattr__(
            self,
            "target_gross_notional",
            _finite_real(
                self.target_gross_notional,
                "target_gross_notional",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "periods_per_year",
            _positive_integer(self.periods_per_year, "periods_per_year"),
        )
        object.__setattr__(
            self,
            "risk_free_rate",
            _finite_real(self.risk_free_rate, "risk_free_rate"),
        )
        if type(self.run_robustness) is not bool:
            raise TypeError("run_robustness must be Boolean.")
        if type(self.run_statistical_validation) is not bool:
            raise TypeError("run_statistical_validation must be Boolean.")
        if self.run_statistical_validation and not self.run_robustness:
            raise ValueError(
                "Statistical validation requires run_robustness=True because its "
                "canonical input is a RobustnessResult."
            )
        scenarios = tuple(self.robustness_scenarios)
        if any(not isinstance(item, ParameterScenario) for item in scenarios):
            raise TypeError("robustness_scenarios must contain ParameterScenario values.")
        identifiers = [scenario.scenario_id for scenario in scenarios]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("robustness scenario IDs must be unique.")
        object.__setattr__(self, "robustness_scenarios", scenarios)
        baseline_id = self.robustness_baseline_scenario_id
        if baseline_id is not None:
            if not isinstance(baseline_id, str) or not baseline_id.strip():
                raise ValueError(
                    "robustness_baseline_scenario_id must be a non-empty string or None."
                )
            baseline_id = baseline_id.strip()
            object.__setattr__(self, "robustness_baseline_scenario_id", baseline_id)
        if scenarios and baseline_id is None:
            raise ValueError(
                "robustness_baseline_scenario_id is required with supplied scenarios."
            )
        if baseline_id is not None and baseline_id not in identifiers:
            raise ValueError(
                "robustness_baseline_scenario_id must identify one supplied scenario."
            )
        if not self.run_robustness and (scenarios or baseline_id is not None):
            raise ValueError(
                "Robustness scenarios and baseline IDs require run_robustness=True."
            )
        if not isinstance(self.validation_settings, StatisticalValidationSettings):
            raise TypeError(
                "validation_settings must be StatisticalValidationSettings."
            )
        metadata = _json_value(self.metadata, "metadata")
        object.__setattr__(self, "metadata", _freeze_json(metadata))


@dataclass(frozen=True)
class DiagnosticExecutionCoverage:
    """Explain signal availability separately from positive-beta execution."""

    symbol_y: str
    symbol_x: str
    total_rows: int
    finite_beta_rows: int
    positive_execution_beta_rows: int
    non_positive_execution_beta_rows: int
    finite_signal_rows: int
    entry_execution_unavailable_rows_due_to_beta: int
    signal_observation_coverage: float
    beta_execution_policy: str = (
        "beta_gt_zero_required_for_entry_and_rebalance;close_uses_existing_units"
    )


@dataclass(frozen=True)
class ResearchExperimentResult:
    """Composed result of one research-only orchestration run."""

    run_id: str
    research_content_digest: str
    price_content_digest: str
    research_pipeline_version: str
    configuration_snapshot_version: int
    experiment_schema_version: int
    configuration_digest: str
    experiment_name: str
    created_at: datetime
    status: ResearchPipelineStatus
    configuration_snapshot: Mapping[str, Any]
    metadata: Mapping[str, Any]
    screening_results: tuple[PairScreeningResult, ...]
    selected_pair: PairScreeningResult | None
    hedge_estimates: pd.DataFrame | None
    zscore: pd.Series | None
    backtest_result: BacktestResult | None
    performance_report: StrategyPerformanceReport | None
    diagnostic_execution_coverage: DiagnosticExecutionCoverage | None
    walk_forward_result: WalkForwardResult | None
    robustness_result: RobustnessResult | None
    statistical_validation_result: StatisticalValidationResult | None
    analytics_stage: PipelineStageStatus
    walk_forward_stage: PipelineStageStatus
    robustness_stage: PipelineStageStatus
    validation_stage: PipelineStageStatus
    provenance: Mapping[str, Any]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration_snapshot",
            _freeze_json(_json_value(self.configuration_snapshot, "configuration_snapshot")),
        )
        object.__setattr__(
            self, "metadata", _freeze_json(_json_value(self.metadata, "metadata"))
        )
        object.__setattr__(
            self, "provenance", _freeze_json(_json_value(self.provenance, "provenance"))
        )
        object.__setattr__(self, "screening_results", tuple(self.screening_results))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.hedge_estimates is not None:
            object.__setattr__(
                self, "hedge_estimates", self.hedge_estimates.copy(deep=True)
            )
        if self.zscore is not None:
            object.__setattr__(self, "zscore", self.zscore.copy(deep=True))

    @property
    def experiment_id(self) -> str:
        """Backward-compatible read alias for the unique persistence run ID."""
        return self.run_id


def _canonical_label(value: Any, path: str) -> Any:
    """Canonicalize an index/name label or reject unstable arbitrary objects."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{path} must be finite.")
        return number
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError(f"{path} must not be NaT.")
        return timestamp.isoformat()
    raise TypeError(
        f"{path} has no supported stable canonical representation; arbitrary "
        "object repr() values are rejected."
    )


def _canonical_price_payload(prices: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if isinstance(prices.columns, pd.MultiIndex):
        raise TypeError("prices must use a one-dimensional string column index.")
    if not prices.columns.is_unique:
        raise ValueError("prices must have unique columns.")
    if not prices.index.is_unique:
        raise ValueError("prices must have a unique index.")
    if any(not isinstance(column, str) or not column for column in prices.columns):
        raise TypeError("price column labels must be non-empty strings.")
    ordered_columns = sorted(prices.columns)
    ordered = prices.loc[:, ordered_columns]
    rows: list[list[int | float]] = []
    for row_number, row in enumerate(ordered.itertuples(index=False, name=None)):
        normalized_row: list[int | float] = []
        for column, value in zip(ordered_columns, row, strict=True):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(
                    f"prices[{row_number}, {column!r}] must be non-Boolean numeric."
                )
            number = float(value)
            if not np.isfinite(number):
                raise ValueError("price content must contain only finite values.")
            normalized_row.append(number)
        rows.append(normalized_row)
    index_values = [
        _canonical_label(value, f"prices.index[{position}]")
        for position, value in enumerate(prices.index)
    ]
    column_name = _canonical_label(prices.columns.name, "prices.columns.name")
    index_name = _canonical_label(prices.index.name, "prices.index.name")
    valuation_policy = prices.attrs.get("valuation_policy")
    if valuation_policy is not None and not isinstance(valuation_policy, str):
        raise TypeError("prices.attrs['valuation_policy'] must be a string or None.")
    observed = prices.attrs.get(OBSERVED_PRICE_MASK_ATTR)
    observed_values: list[list[bool]] | None = None
    if observed is not None:
        if not isinstance(observed, pd.DataFrame):
            raise TypeError("observed price provenance must be a pandas DataFrame.")
        if not observed.index.equals(prices.index) or not observed.columns.equals(
            prices.columns
        ):
            raise ValueError("observed price provenance must align exactly with prices.")
        if any(dtype != bool for dtype in observed.dtypes):
            raise TypeError("observed price provenance must contain Boolean values.")
        observed_values = (
            observed.loc[:, ordered_columns].to_numpy(dtype=bool).tolist()
        )
    return {
        "columns": ordered_columns,
        "column_name": column_name,
        "index_name": index_name,
        "index_type": type(prices.index).__name__,
        "index_timezone": (
            str(prices.index.tz)
            if isinstance(prices.index, pd.DatetimeIndex)
            and prices.index.tz is not None
            else None
        ),
        "index_values": index_values,
        "dtypes": [str(ordered[column].dtype) for column in ordered_columns],
        "values": rows,
        "observed_mask": observed_values,
        "valuation_policy": valuation_policy,
    }


def price_content_digest(prices: pd.DataFrame) -> str:
    """Return a cross-process stable SHA-256 digest of canonical price content."""
    payload = _canonical_price_payload(prices)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sample_label(value: Any, path: str) -> Any:
    return _canonical_label(value, path)


def build_configuration_snapshot(request: ResearchExperimentRequest) -> dict[str, Any]:
    """Return actual effective settings and supplied-sample identity."""
    if not isinstance(request, ResearchExperimentRequest):
        raise TypeError("request must be a ResearchExperimentRequest.")
    effective_scenarios = request.robustness_scenarios
    effective_baseline_id = request.robustness_baseline_scenario_id
    if request.run_robustness and not effective_scenarios:
        effective_scenarios = (_baseline_scenario(request),)
        effective_baseline_id = "baseline"
    config = request.config.to_dict()
    config["universe"] = {
        group: sorted(symbols)
        for group, symbols in sorted(config["universe"].items())
    }
    digest = price_content_digest(request.prices)
    snapshot = {
        "versions": {
            "research_pipeline": RESEARCH_PIPELINE_VERSION,
            "configuration_snapshot": CONFIGURATION_SNAPSHOT_VERSION,
            "experiment_schema": EXPERIMENT_SCHEMA_VERSION,
        },
        "research_config": config,
        "data_sample": {
            "start_label": (
                None
                if request.prices.empty
                else _sample_label(request.prices.index[0], "data_sample.start_label")
            ),
            "end_label": (
                None
                if request.prices.empty
                else _sample_label(request.prices.index[-1], "data_sample.end_label")
            ),
            "rows": len(request.prices),
            "columns": len(request.prices.columns),
            "symbols": sorted(request.prices.columns),
            "price_content_digest": digest,
            "observed_price_mask_present": (
                OBSERVED_PRICE_MASK_ATTR in request.prices.attrs
            ),
            "valuation_policy": request.prices.attrs.get("valuation_policy"),
        },
        "effective_behavior": {
            "signal": {
                "entry_z": request.config.strategy.entry_z,
                "exit_z": request.config.strategy.exit_z,
                "stop_z": request.config.strategy.stop_z,
                "zscore_lookback": request.config.strategy.zscore_lookback,
                "zscore_ddof": 1,
                "missing_policy": "hold",
                "max_holding_period": request.config.strategy.max_holding_days,
                "cooldown_period": None,
                "holding_and_cooldown_clock": "execution_aware_actual_fill_rows",
            },
            "diagnostic_execution": {
                "scope": "full_sample_in_sample_diagnostic",
                "hedge_lookback": request.config.strategy.hedge_lookback,
                "execution_lag": 1,
                "hedge_ratio_lag": 1,
                "non_positive_beta_policy": (
                    "spread_and_zscore_preserved;entry_and_rebalance_unavailable;"
                    "close_uses_existing_units"
                ),
                "target_gross_notional": request.target_gross_notional,
                "initial_capital": request.initial_capital,
                "commission_bps": request.config.costs.commission_bps,
                "slippage_bps": request.config.costs.slippage_bps,
                "annual_borrow_rate_y": request.config.costs.annual_borrow_rate,
                "annual_borrow_rate_x": request.config.costs.annual_borrow_rate,
                "financing_rate": 0.0,
                "fixed_commission_per_leg": 0.0,
                "rebalance": False,
                "rebalance_threshold": 0.0,
                "force_liquidation": True,
                "sizing_policy": "positive_beta_weighted_gross_notional",
            },
            "walk_forward": {
                "formation_window": request.config.screening.formation_days,
                "trading_window": request.config.walk_forward.trading_days,
                "step_size": request.config.walk_forward.trading_days,
                "expanding": False,
                "minimum_observations": None,
                "independent_of_full_sample_selection": True,
                "capital_policy": "equal_capital_reset",
                "aggregate_return_policy": "time_weighted_equal_capital_reset",
                "inactive_capital_policy": "zero_return_cash_for_no_selection_rows",
                "unavailable_row_policy": "scheduled_rows_remain_nan",
                "aggregate_dollar_pnl_available": False,
            },
            "robustness": {
                "requested": request.run_robustness,
                "baseline_scenario_id": effective_baseline_id,
                "scenarios": [asdict(item) for item in effective_scenarios],
                "drawdown_severity_threshold": -0.20,
                "headline_metric_basis": HEADLINE_METRIC_BASIS,
                "common_horizon_policy": "structural_common_scheduled_horizon",
                "distribution_policy": DISTRIBUTION_POLICY,
            },
            "validation": {
                **asdict(request.validation_settings),
                "requested": request.run_statistical_validation,
                "effective_random_seed": (
                    request.config.random_seed
                    if request.validation_settings.random_seed is None
                    else request.validation_settings.random_seed
                ),
                "multiplicity_policy": "bonferroni_and_benjamini_hochberg_diagnostic",
            },
            "analytics": {
                "periods_per_year": request.periods_per_year,
                "risk_free_rate": request.risk_free_rate,
            },
        },
        "configured_but_not_enforced": [
            "data (prices are supplied directly)",
            "strategy.target_annual_vol",
            "strategy.max_gross_leverage",
            "walk_forward.min_selected_pairs",
        ],
    }
    return _json_value(snapshot, "configuration_snapshot")


def configuration_digest(snapshot: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for a configuration snapshot."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping.")
    return hashlib.sha256(_canonical_json(snapshot).encode("utf-8")).hexdigest()


def research_content_digest(snapshot: Mapping[str, Any]) -> str:
    """Hash material research content; display metadata is intentionally absent."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping.")
    identity = {
        "configuration_snapshot": _thaw_json(snapshot),
        "research_pipeline_version": RESEARCH_PIPELINE_VERSION,
        "configuration_snapshot_version": CONFIGURATION_SNAPSHOT_VERSION,
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _groups(config: ResearchConfig) -> Mapping[str, tuple[str, ...]] | None:
    if not config.universe:
        return None
    return MappingProxyType(
        {group: tuple(symbols) for group, symbols in config.universe.items()}
    )


def _selected_rank_one(
    results: tuple[PairScreeningResult, ...],
) -> PairScreeningResult | None:
    selected = [result for result in results if result.selected]
    if not selected:
        return None
    rank_one = [result for result in selected if result.rank == 1]
    if len(rank_one) != 1:
        raise RuntimeError("Canonical screening did not return exactly one rank-1 pair.")
    return rank_one[0]


def _observed_pair_masks(
    prices: pd.DataFrame,
    index: pd.Index,
    symbol_y: str,
    symbol_x: str,
) -> tuple[pd.Series | None, pd.Series | None]:
    observed = prices.attrs.get(OBSERVED_PRICE_MASK_ATTR)
    if observed is None:
        return None, None
    if not isinstance(observed, pd.DataFrame):
        raise TypeError("observed price provenance must be a pandas DataFrame.")
    if not observed.index.equals(prices.index) or not observed.columns.equals(
        prices.columns
    ):
        raise ValueError("observed price provenance must align exactly with prices.")
    return (
        observed.loc[index, symbol_y].copy(deep=True),
        observed.loc[index, symbol_x].copy(deep=True),
    )


def _baseline_scenario(request: ResearchExperimentRequest) -> ParameterScenario:
    config = request.config
    return ParameterScenario(
        scenario_id="baseline",
        entry_z=config.strategy.entry_z,
        exit_z=config.strategy.exit_z,
        stop_z=config.strategy.stop_z,
        zscore_lookback=config.strategy.zscore_lookback,
        formation_window=config.screening.formation_days,
        trading_window=config.walk_forward.trading_days,
        screening_min_observations=config.screening.min_observations,
        commission_bps=config.costs.commission_bps,
        slippage_bps=config.costs.slippage_bps,
        financing_rate=0.0,
        borrow_rate_y=config.costs.annual_borrow_rate,
        borrow_rate_x=config.costs.annual_borrow_rate,
    )


def _walk_forward_kwargs(request: ResearchExperimentRequest) -> dict[str, Any]:
    config = request.config
    return {
        "fdr_threshold": config.screening.fdr_threshold,
        "max_half_life": config.screening.max_half_life,
        "hurst_threshold": config.screening.hurst_threshold,
        "max_holding_period": config.strategy.max_holding_days,
        "target_gross_notional": request.target_gross_notional,
        "initial_capital": request.initial_capital,
        "execution_lag": 1,
        "periods_per_year": request.periods_per_year,
        "risk_free_rate": request.risk_free_rate,
        "force_liquidation": True,
    }


def _diagnostic_coverage(
    symbol_y: str,
    symbol_x: str,
    estimates: pd.DataFrame,
    zscore: pd.Series,
) -> DiagnosticExecutionCoverage:
    beta = pd.to_numeric(estimates["beta"], errors="coerce")
    signal = pd.to_numeric(zscore, errors="coerce")
    finite_beta = pd.Series(np.isfinite(beta), index=beta.index)
    finite_signal = pd.Series(np.isfinite(signal), index=signal.index)
    non_positive = finite_beta & beta.le(0.0)
    total = len(estimates)
    return DiagnosticExecutionCoverage(
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        total_rows=total,
        finite_beta_rows=int(finite_beta.sum()),
        positive_execution_beta_rows=int((finite_beta & beta.gt(0.0)).sum()),
        non_positive_execution_beta_rows=int(non_positive.sum()),
        finite_signal_rows=int(finite_signal.sum()),
        entry_execution_unavailable_rows_due_to_beta=int(
            (finite_signal & non_positive).sum()
        ),
        signal_observation_coverage=(
            float(finite_signal.sum() / total) if total else 0.0
        ),
    )


def _require_enum(value: Any, enum_type: type[Enum], path: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{path} must be a declared {enum_type.__name__} member.")


def _semantically_equal(left: Any, right: Any) -> bool:
    if is_dataclass(left) and is_dataclass(right):
        if type(left) is not type(right):
            return False
        return all(
            _semantically_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
        )
    if isinstance(left, Enum) or isinstance(right, Enum):
        return type(left) is type(right) and left is right
    if isinstance(left, Real) and isinstance(right, Real):
        if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
            return bool(left) is bool(right)
        return bool(
            np.isclose(
                float(left),
                float(right),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _semantically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _semantically_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _validate_walk_forward_result(result: WalkForwardResult) -> None:
    if not isinstance(result, WalkForwardResult):
        raise TypeError("walk_forward_result must be a WalkForwardResult.")
    _require_enum(
        result.conditional_analytics_status,
        WalkForwardAnalyticsStatus,
        "walk_forward_result.conditional_analytics_status",
    )
    _require_enum(
        result.calendar_analytics_status,
        WalkForwardAnalyticsStatus,
        "walk_forward_result.calendar_analytics_status",
    )
    if result.fold_count != len(result.folds):
        raise ValueError("walk-forward fold_count does not match folds.")
    status_counts = {status: 0 for status in WalkForwardStatus}
    selected_rows = no_selection_rows = unavailable_rows = 0
    for fold in result.folds:
        _require_enum(fold.status, WalkForwardStatus, "walk_forward fold status")
        _require_enum(
            fold.analytics_status,
            WalkForwardAnalyticsStatus,
            "walk_forward fold analytics_status",
        )
        status_counts[fold.status] += 1
        if fold.status is WalkForwardStatus.COMPLETED:
            selected_rows += fold.trading_observations
        elif fold.status is WalkForwardStatus.NO_SELECTION:
            no_selection_rows += fold.trading_observations
        else:
            unavailable_rows += fold.trading_observations
    declared = (
        result.completed_fold_count,
        result.no_selection_fold_count,
        result.insufficient_data_fold_count,
    )
    actual = (
        status_counts[WalkForwardStatus.COMPLETED],
        status_counts[WalkForwardStatus.NO_SELECTION],
        status_counts[WalkForwardStatus.INSUFFICIENT_DATA],
    )
    if declared != actual or sum(declared) != result.fold_count:
        raise ValueError("walk-forward fold status counts do not reconcile.")
    if (
        result.selected_oos_observations != selected_rows
        or result.no_selection_oos_observations != no_selection_rows
        or result.unavailable_oos_observations != unavailable_rows
    ):
        raise ValueError("walk-forward status observation counts do not reconcile.")
    if result.scheduled_eligible_oos_observations != (
        result.selected_oos_observations + result.no_selection_oos_observations
    ):
        raise ValueError("walk-forward eligible observation count is stale.")
    if result.scheduled_oos_observations != (
        result.scheduled_eligible_oos_observations
        + result.unavailable_oos_observations
    ):
        raise ValueError("walk-forward scheduled observation count is stale.")
    if len(result.calendar_oos_returns) != result.scheduled_oos_observations:
        raise ValueError("walk-forward calendar return length is stale.")
    if len(result.conditional_oos_returns) != result.selected_oos_observations:
        raise ValueError("walk-forward conditional return length is stale.")
    if not result.calendar_oos_returns.index.is_unique:
        raise ValueError("walk-forward calendar OOS index must be unique.")
    if not result.conditional_oos_returns.index.is_unique:
        raise ValueError("walk-forward conditional OOS index must be unique.")
    if result.scheduled_oos_observations:
        if (
            result.calendar_oos_returns.index[0] != result.evaluated_start_label
            or result.calendar_oos_returns.index[-1] != result.evaluated_end_label
        ):
            raise ValueError("walk-forward evaluation labels do not match calendar OOS.")
        if result.evaluated_end_position < result.evaluated_start_position:
            raise ValueError("walk-forward evaluation positions are reversed.")
        if (
            result.evaluated_end_position
            - result.evaluated_start_position
            + 1
            != result.scheduled_oos_observations
        ):
            raise ValueError(
                "walk-forward evaluation range does not match scheduled calendar rows."
            )
    denominator = result.scheduled_eligible_oos_observations
    expected_coverage = (
        float(result.selected_oos_observations / denominator)
        if denominator
        else float("nan")
    )
    if not _semantically_equal(result.selection_coverage, expected_coverage):
        raise ValueError("walk-forward selection_coverage is stale.")
    if result.capital_policy != "equal_capital_reset":
        raise ValueError("walk-forward capital_policy must be equal_capital_reset.")
    if result.aggregate_return_policy != "time_weighted_equal_capital_reset":
        raise ValueError("walk-forward aggregate_return_policy is inconsistent.")
    if result.inactive_capital_policy != "zero_return_cash_for_no_selection_rows":
        raise ValueError("walk-forward inactive_capital_policy is inconsistent.")
    if result.selection_coverage_denominator != "selected_plus_no_selection_scheduled_rows":
        raise ValueError("walk-forward selection coverage denominator is inconsistent.")
    if result.aggregate_dollar_pnl_available or result.aggregate_trade_dollar_metrics_available:
        raise ValueError("equal-capital-reset folds cannot expose aggregate dollar P&L.")
    report = result.calendar_performance_report
    if result.calendar_analytics_status is WalkForwardAnalyticsStatus.AVAILABLE:
        if report is None:
            raise ValueError("available calendar analytics require a report.")
        if bool(result.calendar_oos_returns.isna().any()):
            raise ValueError("available calendar analytics cannot omit scheduled rows.")
        if report.report_observations != len(result.calendar_oos_returns):
            raise ValueError("calendar performance report observation count is stale.")
        if report.core.observations != report.report_observations:
            raise ValueError("calendar performance core observation count is stale.")
    elif report is not None:
        raise ValueError("unavailable calendar analytics must not contain a report.")


def _validate_robustness_result(result: RobustnessResult) -> None:
    if not isinstance(result, RobustnessResult):
        raise TypeError("robustness_result must be a RobustnessResult.")
    scenario_ids = tuple(item.scenario.scenario_id for item in result.scenarios)
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("robustness scenario IDs must be unique.")
    if scenario_ids.count(result.baseline_scenario_id) != 1:
        raise ValueError("robustness baseline must identify exactly one scenario.")
    if result.summary.baseline_scenario_id != result.baseline_scenario_id:
        raise ValueError("robustness summary baseline ID is stale.")
    if result.baseline_result.scenario.scenario_id != result.baseline_scenario_id:
        raise ValueError("robustness baseline_result is inconsistent.")
    if result.purpose != ROBUSTNESS_PURPOSE or result.warning != NO_OPTIMIZATION_WARNING:
        raise ValueError("robustness purpose/no-optimization safeguards are stale.")
    counts = {status: 0 for status in ScenarioStatus}
    for scenario in result.scenarios:
        _require_enum(scenario.status, ScenarioStatus, "robustness scenario status")
        _require_enum(
            scenario.calendar_metrics_status,
            MetricAvailabilityStatus,
            "robustness calendar metric status",
        )
        _require_enum(
            scenario.common_horizon_analytics_status,
            MetricAvailabilityStatus,
            "robustness common-horizon metric status",
        )
        counts[scenario.status] += 1
    summary_counts = (
        result.summary.scenario_count,
        result.summary.completed_scenarios,
        result.summary.unavailable_scenarios,
        result.summary.invalid_scenarios,
        result.summary.failed_scenarios,
    )
    actual_counts = (
        len(result.scenarios),
        counts[ScenarioStatus.COMPLETED],
        counts[ScenarioStatus.NO_VALID_FOLDS]
        + counts[ScenarioStatus.ANALYTICS_UNAVAILABLE],
        counts[ScenarioStatus.INVALID_CONFIGURATION],
        counts[ScenarioStatus.FAILED],
    )
    if summary_counts != actual_counts:
        raise ValueError("robustness scenario status counts do not reconcile.")
    if result.common_horizon_observations != len(result.common_horizon_index):
        raise ValueError("robustness common-horizon observation count is stale.")
    if (
        result.common_horizon_scenario_count
        + result.common_horizon_excluded_scenarios
        != len(result.scenarios)
    ):
        raise ValueError("robustness common-horizon scenario counts do not reconcile.")
    if result.common_horizon_available != result.common_horizon_analytics_available:
        raise ValueError("robustness common-horizon availability aliases disagree.")
    if result.common_horizon_analytics_available != (
        result.common_horizon_analytics_status is MetricAvailabilityStatus.AVAILABLE
    ):
        raise ValueError("robustness common-horizon analytics status is contradictory.")
    if result.common_horizon_fully_observed and not result.common_horizon_structurally_available:
        raise ValueError("a fully observed common horizon must be structurally available.")
    if result.summary.headline_metric_basis != HEADLINE_METRIC_BASIS:
        raise ValueError("robustness headline metric basis is inconsistent.")
    if result.distribution_policy != DISTRIBUTION_POLICY:
        raise ValueError("robustness distribution policy is inconsistent.")
    if set(result.tested_dimensions).intersection(result.untested_material_dimensions):
        raise ValueError("robustness tested and untested dimensions overlap.")
    canonical_summary = summarize_sensitivity(
        result.scenarios,
        result.baseline_scenario_id,
        drawdown_severity_threshold=result.summary.drawdown_severity_threshold,
    )
    if not _semantically_equal(result.summary, canonical_summary):
        raise ValueError("robustness sensitivity summary is inconsistent with scenarios.")


def _validate_statistical_validation_result(
    result: StatisticalValidationResult,
    robustness: RobustnessResult,
) -> None:
    if not isinstance(result, StatisticalValidationResult):
        raise TypeError("statistical_validation_result has an invalid type.")
    for path, value in (
        ("primary_inference_availability", result.primary_inference_availability),
        ("overall_availability", result.overall_availability),
        ("availability", result.availability),
        ("bootstrap.status", result.bootstrap.status),
        ("probabilistic_sharpe.status", result.probabilistic_sharpe.status),
        ("minimum_track_record.status", result.minimum_track_record.status),
        ("fold_consistency.status", result.fold_consistency.status),
    ):
        _require_enum(value, ValidationAvailability, f"validation.{path}")
    if result.availability is not result.primary_inference_availability:
        raise ValueError("validation legacy and primary availability disagree.")
    if result.purpose != STATISTICAL_VALIDATION_PURPOSE:
        raise ValueError("validation purpose safeguard is stale.")
    if any(not isinstance(item, str) or not item for item in result.validation_warnings):
        raise ValueError("validation warnings must be non-empty strings.")
    if any(not isinstance(item, str) or not item for item in result.provenance_warnings):
        raise ValueError("validation provenance warnings must be non-empty strings.")
    if result.observations != len(result.primary_oos_returns):
        raise ValueError("validation primary OOS observation count is stale.")
    baseline_returns = robustness.baseline_result.common_horizon_returns
    if not result.primary_oos_returns.index.equals(baseline_returns.index) or not np.allclose(
        result.primary_oos_returns.to_numpy(dtype=float),
        baseline_returns.to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError("validation primary OOS returns differ from the baseline scenario.")
    expected_primary = (
        ValidationAvailability.AVAILABLE
        if all(
            value is ValidationAvailability.AVAILABLE
            for value in (
                result.bootstrap.status,
                result.probabilistic_sharpe.status,
                result.minimum_track_record.status,
            )
        )
        else ValidationAvailability.UNAVAILABLE
    )
    if result.primary_inference_availability is not expected_primary:
        raise ValueError("validation primary availability is contradictory.")
    requested = [result.primary_inference_availability, result.fold_consistency.status]
    if result.regime_consistency is not None:
        _require_enum(
            result.regime_consistency.status,
            ValidationAvailability,
            "validation.regime_consistency.status",
        )
        requested.append(result.regime_consistency.status)
    expected_overall = (
        ValidationAvailability.UNAVAILABLE
        if ValidationAvailability.UNAVAILABLE in requested
        else (
            ValidationAvailability.INSUFFICIENT_DATA
            if ValidationAvailability.INSUFFICIENT_DATA in requested
            else ValidationAvailability.AVAILABLE
        )
    )
    if result.overall_availability is not expected_overall:
        raise ValueError("validation overall availability is contradictory.")
    multiple = result.multiple_testing
    scenario_ids = tuple(item.scenario.scenario_id for item in robustness.scenarios)
    pvalue_ids = tuple(item.scenario_id for item in multiple.scenario_pvalues)
    if len(pvalue_ids) != len(set(pvalue_ids)) or set(pvalue_ids) != set(scenario_ids):
        raise ValueError("validation scenario-to-p-value mapping is inconsistent.")
    invalid = sum(
        item.status is ScenarioStatus.INVALID_CONFIGURATION
        for item in robustness.scenarios
    )
    failed = sum(item.status is ScenarioStatus.FAILED for item in robustness.scenarios)
    eligible = len(robustness.scenarios) - invalid - failed
    if (
        multiple.total_tested_configurations != len(robustness.scenarios)
        or multiple.invalid_configurations != invalid
        or multiple.failed_configurations != failed
        or multiple.valid_comparable_configurations != eligible
        or multiple.eligible_hypothesis_count != eligible
        or multiple.family_size != eligible
        or multiple.valid_pvalue_count + multiple.unavailable_pvalue_count
        != multiple.total_tested_configurations
        or multiple.finite_pvalue_count != multiple.valid_pvalue_count
        or multiple.unavailable_eligible_hypothesis_count
        != eligible - multiple.valid_pvalue_count
        or multiple.analytically_unavailable_configurations
        != multiple.unavailable_eligible_hypothesis_count
        or multiple.baseline_scenario_id != robustness.baseline_scenario_id
    ):
        raise ValueError("validation multiple-testing counts do not reconcile.")
    if result.fold_consistency.fold_count != len(result.fold_consistency.folds):
        raise ValueError("validation fold-consistency count is stale.")
    fold_consistency = result.fold_consistency
    for fold in fold_consistency.folds:
        _require_enum(
            fold.availability,
            ValidationAvailability,
            "validation fold availability",
        )
        try:
            WalkForwardStatus(fold.selection_status)
        except (TypeError, ValueError) as exc:
            raise ValueError("validation fold selection_status is unknown.") from exc
    observable = sum(
        np.isfinite(fold.total_return) for fold in fold_consistency.folds
    )
    available = sum(
        fold.availability is ValidationAvailability.AVAILABLE
        for fold in fold_consistency.folds
    )
    unavailable = sum(
        fold.availability is not ValidationAvailability.AVAILABLE
        for fold in fold_consistency.folds
    )
    insufficient = sum(
        fold.availability is ValidationAvailability.INSUFFICIENT_DATA
        for fold in fold_consistency.folds
    )
    pre_execution_insufficient = sum(
        fold.selection_status == WalkForwardStatus.INSUFFICIENT_DATA.value
        for fold in fold_consistency.folds
    )
    if (
        fold_consistency.observable_fold_count != observable
        or fold_consistency.analytically_available_fold_count != available
        or fold_consistency.unavailable_fold_count != unavailable
        or fold_consistency.insufficient_data_fold_count != insufficient
        or fold_consistency.risk_analytics_insufficient_fold_count != insufficient
        or fold_consistency.pre_execution_insufficient_fold_count
        != pre_execution_insufficient
    ):
        raise ValueError("validation fold-consistency counts do not reconcile.")
    psr = result.probabilistic_sharpe
    if psr.status is ValidationAvailability.AVAILABLE:
        if not np.isfinite(psr.probability) or not 0.0 <= psr.probability <= 1.0:
            raise ValueError("available PSR requires a finite probability.")
    elif np.isfinite(psr.probability):
        raise ValueError("unavailable PSR must not publish a finite probability.")
    track_record = result.minimum_track_record
    if track_record.status is ValidationAvailability.AVAILABLE:
        if (
            track_record.estimated_required_observations is None
            or track_record.sufficient_track_record is None
        ):
            raise ValueError("available MTRL requires a finite track-record estimate.")
    elif (
        track_record.estimated_required_observations is not None
        or track_record.sufficient_track_record is not None
    ):
        raise ValueError("unavailable MTRL must not publish a track-record estimate.")


def validate_research_experiment_result(result: ResearchExperimentResult) -> None:
    """Reject contradictory or mutated experiment results before persistence."""
    if not isinstance(result, ResearchExperimentResult):
        raise TypeError("result must be a ResearchExperimentResult.")
    if not isinstance(result.run_id, str) or not result.run_id.startswith("run_"):
        raise ValueError("run_id must use the canonical run_<uuid-hex> format.")
    run_hex = result.run_id[4:]
    if len(run_hex) != 32 or any(char not in "0123456789abcdef" for char in run_hex):
        raise ValueError("run_id must use the canonical run_<uuid-hex> format.")
    for name, digest in (
        ("configuration_digest", result.configuration_digest),
        ("research_content_digest", result.research_content_digest),
        ("price_content_digest", result.price_content_digest),
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    if result.research_pipeline_version != RESEARCH_PIPELINE_VERSION:
        raise ValueError("research_pipeline_version is unsupported.")
    if result.configuration_snapshot_version != CONFIGURATION_SNAPSHOT_VERSION:
        raise ValueError("configuration_snapshot_version is unsupported.")
    if result.experiment_schema_version != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("experiment_schema_version is unsupported.")
    snapshot = result.configuration_snapshot
    if configuration_digest(snapshot) != result.configuration_digest:
        raise ValueError("configuration_digest does not match configuration_snapshot.")
    if research_content_digest(snapshot) != result.research_content_digest:
        raise ValueError("research_content_digest does not match material research content.")
    try:
        snapshot_price_digest = snapshot["data_sample"]["price_content_digest"]
        snapshot_versions = snapshot["versions"]
    except (KeyError, TypeError) as exc:
        raise ValueError("configuration_snapshot is missing identity fields.") from exc
    if snapshot_price_digest != result.price_content_digest:
        raise ValueError("price_content_digest does not match configuration_snapshot.")
    expected_versions = {
        "research_pipeline": RESEARCH_PIPELINE_VERSION,
        "configuration_snapshot": CONFIGURATION_SNAPSHOT_VERSION,
        "experiment_schema": EXPERIMENT_SCHEMA_VERSION,
    }
    if not _semantically_equal(snapshot_versions, expected_versions):
        raise ValueError("configuration_snapshot version fields are inconsistent.")
    if not isinstance(result.created_at, datetime) or result.created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware.")
    _require_enum(result.status, ResearchPipelineStatus, "status")
    if result.status is not ResearchPipelineStatus.COMPLETED:
        raise ValueError(
            "NO_PAIR_SELECTED is a legacy terminal status; hardened experiments "
            "use COMPLETED plus diagnostic stage availability."
        )
    for name in (
        "analytics_stage",
        "walk_forward_stage",
        "robustness_stage",
        "validation_stage",
    ):
        _require_enum(getattr(result, name), PipelineStageStatus, name)
    if any(not isinstance(item, PairScreeningResult) for item in result.screening_results):
        raise TypeError("screening_results must contain PairScreeningResult values.")
    selected_results = tuple(item for item in result.screening_results if item.selected)
    ranks = sorted(item.rank for item in selected_results)
    if ranks != list(range(1, len(selected_results) + 1)):
        raise ValueError("screening selected ranks must be consecutive from one.")
    if any(item.rank is not None for item in result.screening_results if not item.selected):
        raise ValueError("rejected screening rows must not have ranks.")
    diagnostic_objects = (
        result.hedge_estimates,
        result.zscore,
        result.backtest_result,
        result.performance_report,
        result.diagnostic_execution_coverage,
    )
    if result.selected_pair is None:
        if selected_results or any(item is not None for item in diagnostic_objects):
            raise ValueError("no diagnostic pair may not contain diagnostic results.")
        if result.analytics_stage is not PipelineStageStatus.UNAVAILABLE:
            raise ValueError("no diagnostic pair requires analytics_stage UNAVAILABLE.")
    else:
        if result.selected_pair not in selected_results or result.selected_pair.rank != 1:
            raise ValueError("selected_pair must be the rank-1 screening result.")
        if any(item is None for item in diagnostic_objects):
            raise ValueError("a selected diagnostic pair requires all diagnostic results.")
        if result.analytics_stage is not PipelineStageStatus.COMPLETED:
            raise ValueError("completed diagnostic results require analytics_stage COMPLETED.")
        assert result.hedge_estimates is not None
        assert result.zscore is not None
        assert result.backtest_result is not None
        assert result.performance_report is not None
        assert result.diagnostic_execution_coverage is not None
        required_columns = {"alpha", "beta", "spread"}
        if not required_columns.issubset(result.hedge_estimates.columns):
            raise ValueError("hedge_estimates is missing canonical columns.")
        if not result.hedge_estimates.index.equals(result.zscore.index):
            raise ValueError("diagnostic hedge estimates and z-score must align.")
        if not result.backtest_result.signals.index.equals(result.zscore.index):
            raise ValueError("diagnostic backtest and z-score indices must align.")
        backtest_zscore = result.backtest_result.signals["zscore"].to_numpy(
            dtype=float
        )
        result_zscore = result.zscore.to_numpy(dtype=float)
        if not np.allclose(
            backtest_zscore,
            result_zscore,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise ValueError("diagnostic backtest does not contain the supplied z-score.")
        expected_coverage = _diagnostic_coverage(
            result.selected_pair.symbol_y,
            result.selected_pair.symbol_x,
            result.hedge_estimates,
            result.zscore,
        )
        if not _semantically_equal(result.diagnostic_execution_coverage, expected_coverage):
            raise ValueError("diagnostic execution coverage is stale or mismatched.")
        validate_backtest_invariants(result.backtest_result)
        analytics_settings = snapshot["effective_behavior"]["analytics"]
        execution_settings = snapshot["effective_behavior"]["diagnostic_execution"]
        canonical_report = build_performance_report(
            result.backtest_result.accounting["net_return_after_carry"],
            result.backtest_result.ledger,
            periods_per_year=analytics_settings["periods_per_year"],
            risk_free_rate=analytics_settings["risk_free_rate"],
            accounting=result.backtest_result.accounting,
            initial_capital=execution_settings["initial_capital"],
        )
        if not _semantically_equal(result.performance_report, canonical_report):
            raise ValueError("diagnostic performance report is inconsistent with backtest.")
    if result.walk_forward_stage is PipelineStageStatus.COMPLETED:
        if result.walk_forward_result is None:
            raise ValueError("completed walk-forward stage requires a result.")
        _validate_walk_forward_result(result.walk_forward_result)
    elif result.walk_forward_result is not None:
        raise ValueError("non-completed walk-forward stage must not contain a result.")
    robustness_requested = bool(
        snapshot["effective_behavior"]["robustness"]["requested"]
    )
    validation_requested = bool(
        snapshot["effective_behavior"]["validation"]["requested"]
    )
    for stage_name, stage, nested, requested in (
        ("robustness", result.robustness_stage, result.robustness_result, robustness_requested),
        ("validation", result.validation_stage, result.statistical_validation_result, validation_requested),
    ):
        if stage is PipelineStageStatus.COMPLETED and nested is None:
            raise ValueError(f"completed {stage_name} stage requires a result.")
        if stage is not PipelineStageStatus.COMPLETED and nested is not None:
            raise ValueError(f"non-completed {stage_name} stage must not contain a result.")
        if stage is PipelineStageStatus.NOT_REQUESTED and requested:
            raise ValueError(f"requested {stage_name} cannot be NOT_REQUESTED.")
        if stage is not PipelineStageStatus.NOT_REQUESTED and not requested:
            raise ValueError(f"unrequested {stage_name} must be NOT_REQUESTED.")
    if result.robustness_result is not None:
        _validate_robustness_result(result.robustness_result)
    if result.statistical_validation_result is not None:
        if result.robustness_result is None:
            raise ValueError("statistical validation requires a robustness result.")
        _validate_statistical_validation_result(
            result.statistical_validation_result,
            result.robustness_result,
        )


def run_research_pipeline(
    request: ResearchExperimentRequest,
) -> ResearchExperimentResult:
    """Run one research experiment through canonical package APIs.

    Full-sample rank-1 selection controls only the optional in-sample diagnostic.
    Walk-forward performs its own formation-only selection independently.  A
    missing current diagnostic pair therefore does not erase historical OOS
    evidence.  Robustness uses a fixed scenario tuple, and validation consumes
    that immutable result without promoting any scenario.
    """
    if not isinstance(request, ResearchExperimentRequest):
        raise TypeError("request must be a ResearchExperimentRequest.")
    prices = request.prices.copy(deep=True)
    prices.attrs = deepcopy(request.prices.attrs)
    groups = _groups(request.config)
    screening_kwargs = screening_kwargs_from_config(request.config.screening)
    snapshot = build_configuration_snapshot(request)
    run_id = f"run_{uuid4().hex}"
    config_hash = configuration_digest(snapshot)
    content_hash = research_content_digest(snapshot)
    price_hash = str(snapshot["data_sample"]["price_content_digest"])
    created_at = datetime.now(timezone.utc)

    screening_results = tuple(
        screen_pairs(prices, groups=groups, **screening_kwargs)
    )
    selected = _selected_rank_one(screening_results)
    estimates: pd.DataFrame | None = None
    zscore: pd.Series | None = None
    backtest: BacktestResult | None = None
    performance: StrategyPerformanceReport | None = None
    diagnostic_coverage: DiagnosticExecutionCoverage | None = None
    analytics_stage = PipelineStageStatus.UNAVAILABLE
    if selected is not None:
        symbol_y, symbol_x = selected.symbol_y, selected.symbol_x
        estimates = rolling_ols_spread(
            prices[symbol_y],
            prices[symbol_x],
            lookback=request.config.strategy.hedge_lookback,
        )
        # Research state remains visible through beta sign changes.  Only the
        # separate execution input is made unavailable when beta is non-positive.
        zscore = rolling_zscore(
            estimates["spread"],
            lookback=request.config.strategy.zscore_lookback,
            ddof=1,
        )
        execution_beta = estimates["beta"].where(estimates["beta"].gt(0.0))
        diagnostic_coverage = _diagnostic_coverage(
            symbol_y,
            symbol_x,
            estimates,
            zscore,
        )
        research_y = prices.loc[estimates.index, symbol_y].copy(deep=True)
        research_x = prices.loc[estimates.index, symbol_x].copy(deep=True)
        observed_y, observed_x = _observed_pair_masks(
            prices, estimates.index, symbol_y, symbol_x
        )
        backtest = run_pair_backtest(
            research_y,
            research_x,
            execution_beta,
            request.target_gross_notional,
            zscore=zscore,
            initial_capital=request.initial_capital,
            entry_z=request.config.strategy.entry_z,
            exit_z=request.config.strategy.exit_z,
            stop_z=request.config.strategy.stop_z,
            max_holding_period=request.config.strategy.max_holding_days,
            missing_policy="hold",
            execution_lag=1,
            commission_bps=request.config.costs.commission_bps,
            fixed_commission_per_leg=0.0,
            slippage_bps=request.config.costs.slippage_bps,
            borrow_rate_y=request.config.costs.annual_borrow_rate,
            borrow_rate_x=request.config.costs.annual_borrow_rate,
            financing_rate=0.0,
            periods_per_year=request.periods_per_year,
            rebalance=False,
            rebalance_threshold=0.0,
            force_liquidation=True,
            observed_y=observed_y,
            observed_x=observed_x,
            hedge_ratio_lag=1,
        )
        performance = build_performance_report(
            backtest.accounting["net_return_after_carry"],
            backtest.ledger,
            periods_per_year=request.periods_per_year,
            risk_free_rate=request.risk_free_rate,
            accounting=backtest.accounting,
            initial_capital=request.initial_capital,
        )
        analytics_stage = PipelineStageStatus.COMPLETED

    # This formation-only process is deliberately not gated by `selected`.
    walk_forward = run_walk_forward_analysis(
        prices,
        formation_window=request.config.screening.formation_days,
        trading_window=request.config.walk_forward.trading_days,
        groups=groups,
        screening_min_observations=request.config.screening.min_observations,
        fdr_threshold=request.config.screening.fdr_threshold,
        max_half_life=request.config.screening.max_half_life,
        hurst_threshold=request.config.screening.hurst_threshold,
        zscore_lookback=request.config.strategy.zscore_lookback,
        entry_z=request.config.strategy.entry_z,
        exit_z=request.config.strategy.exit_z,
        stop_z=request.config.strategy.stop_z,
        max_holding_period=request.config.strategy.max_holding_days,
        cooldown_period=None,
        target_gross_notional=request.target_gross_notional,
        initial_capital=request.initial_capital,
        execution_lag=1,
        commission_bps=request.config.costs.commission_bps,
        fixed_commission_per_leg=0.0,
        slippage_bps=request.config.costs.slippage_bps,
        borrow_rate_y=request.config.costs.annual_borrow_rate,
        borrow_rate_x=request.config.costs.annual_borrow_rate,
        financing_rate=0.0,
        periods_per_year=request.periods_per_year,
        risk_free_rate=request.risk_free_rate,
        force_liquidation=True,
    )

    robustness: RobustnessResult | None = None
    validation: StatisticalValidationResult | None = None
    robustness_stage = PipelineStageStatus.NOT_REQUESTED
    validation_stage = PipelineStageStatus.NOT_REQUESTED
    if request.run_robustness:
        scenarios = request.robustness_scenarios or (_baseline_scenario(request),)
        baseline_id = request.robustness_baseline_scenario_id or "baseline"
        robustness = run_sensitivity_analysis(
            prices,
            scenarios,
            baseline_id,
            groups=groups,
            walk_forward_kwargs=_walk_forward_kwargs(request),
        )
        robustness_stage = PipelineStageStatus.COMPLETED
        if request.run_statistical_validation:
            settings = request.validation_settings
            validation = build_statistical_validation_report(
                robustness,
                block_length=settings.block_length,
                n_bootstrap=settings.n_bootstrap,
                random_seed=(
                    request.config.random_seed
                    if settings.random_seed is None
                    else settings.random_seed
                ),
                benchmark_sharpe=settings.benchmark_sharpe,
                confidence_level=settings.confidence_level,
                minimum_valid_fraction=settings.minimum_valid_fraction,
                significance_level=settings.significance_level,
            )
            validation_stage = PipelineStageStatus.COMPLETED

    provenance = {
        "run_id_policy": "unique_uuid4_audit_identifier",
        "research_content_digest_policy": (
            "sha256_canonical_material_inputs_excluding_display_name_and_metadata"
        ),
        "research_content_digest": content_hash,
        "price_content_digest": price_hash,
        "input_price_provenance": "caller_supplied",
        "observed_price_mask_present": OBSERVED_PRICE_MASK_ATTR in prices.attrs,
        "cleaning_policy": prices.attrs.get(
            "valuation_policy", "caller_supplied_without_cleaning_metadata"
        ),
        "universe_provenance": "caller_supplied_static_research_config",
        "point_in_time_universe_validated": False,
        "selected_pair_backtest_scope": "full_sample_in_sample_diagnostic",
        "full_sample_diagnostic_pair_available": selected is not None,
        "walk_forward_independent_of_full_sample_selection": True,
        "non_positive_beta_execution_policy": (
            "spread_and_zscore_preserved;entry_and_rebalance_unavailable;"
            "close_uses_existing_units"
        ),
        "backtest_upstream_provenance_validated": (
            None
            if backtest is None
            else backtest.research_metadata.upstream_provenance_validated
        ),
        "walk_forward_universe_provenance": walk_forward.universe_provenance,
        "walk_forward_cleaning_provenance": walk_forward.cleaning_provenance,
        "walk_forward_point_in_time_universe_validated": (
            walk_forward.point_in_time_universe_validated
        ),
        "walk_forward_capital_policy": walk_forward.capital_policy,
    }
    warnings = (
        *(
            (
                "No full-sample diagnostic pair passed canonical screening; "
                "walk-forward evidence was evaluated independently.",
            )
            if selected is None
            else (
                "The selected-pair full-sample backtest is an in-sample diagnostic and must not be interpreted as OOS performance.",
            )
        ),
        "Non-positive rolling beta preserves spread and z-score research state but makes entry/rebalance execution unavailable; negative-beta sizing is not supported.",
        "Walk-forward folds reset to the canonical equal capital base; aggregate dollar P&L is unavailable under that policy.",
        "The pipeline preserves caller provenance and does not certify point-in-time universe construction or upstream causality.",
        "Supplied prices bypass DataConfig download/cleaning settings; target annual volatility, maximum gross leverage, and minimum selected-pair count are recorded but not implemented by the current canonical single-pair/walk-forward APIs.",
        "Observed-price masks constrain execution, but canonical screening/statistics consume the supplied numeric marks and do not independently exclude imputed valuations.",
        *walk_forward.provenance_warnings,
    )
    result = ResearchExperimentResult(
        run_id=run_id,
        research_content_digest=content_hash,
        price_content_digest=price_hash,
        research_pipeline_version=RESEARCH_PIPELINE_VERSION,
        configuration_snapshot_version=CONFIGURATION_SNAPSHOT_VERSION,
        experiment_schema_version=EXPERIMENT_SCHEMA_VERSION,
        configuration_digest=config_hash,
        experiment_name=request.experiment_name,
        created_at=created_at,
        status=ResearchPipelineStatus.COMPLETED,
        configuration_snapshot=snapshot,
        metadata=request.metadata,
        screening_results=screening_results,
        selected_pair=selected,
        hedge_estimates=estimates,
        zscore=zscore,
        backtest_result=backtest,
        performance_report=performance,
        diagnostic_execution_coverage=diagnostic_coverage,
        walk_forward_result=walk_forward,
        robustness_result=robustness,
        statistical_validation_result=validation,
        analytics_stage=analytics_stage,
        walk_forward_stage=PipelineStageStatus.COMPLETED,
        robustness_stage=robustness_stage,
        validation_stage=validation_stage,
        provenance=provenance,
        warnings=tuple(dict.fromkeys(warnings)),
    )
    validate_research_experiment_result(result)
    return result
