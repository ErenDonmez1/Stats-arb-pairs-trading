"""Typed, immutable configuration for reproducible research runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import date
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .stats import ADF_MIN_OBSERVATIONS


@dataclass(frozen=True)
class DataConfig:
    """Market-data request and cleaning settings."""

    start: str = "2018-01-01"
    end: str | None = None
    interval: str = "1d"
    min_coverage: float = 0.95
    max_forward_fill: int = 3
    cache_dir: str = "data/raw"


@dataclass(frozen=True)
class ScreeningConfig:
    """Formation horizon and active production-screening settings."""

    formation_days: int = 504
    min_observations: int = 100
    fdr_threshold: float = 0.05
    max_half_life: float = 60.0
    hurst_threshold: float = 0.5


@dataclass(frozen=True)
class StrategyConfig:
    """Dynamic hedge, signal, holding-period, and risk settings."""

    hedge_lookback: int = 120
    zscore_lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_holding_days: int = 60
    target_annual_vol: float = 0.10
    max_gross_leverage: float = 1.5


@dataclass(frozen=True)
class CostConfig:
    """Transaction and financing cost assumptions."""

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    annual_borrow_rate: float = 0.02


@dataclass(frozen=True)
class WalkForwardConfig:
    """Out-of-sample fold and portfolio-selection settings."""

    trading_days: int = 63
    min_selected_pairs: int = 1


@dataclass(frozen=True)
class ResearchConfig:
    """Complete immutable research configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    universe: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable, serialisable representation of the configuration."""
        return {
            "data": asdict(self.data),
            "universe": {
                group: list(symbols) for group, symbols in self.universe.items()
            },
            "screening": asdict(self.screening),
            "strategy": asdict(self.strategy),
            "costs": asdict(self.costs),
            "walk_forward": asdict(self.walk_forward),
            "random_seed": self.random_seed,
        }


_TOP_LEVEL_KEYS = frozenset(field_.name for field_ in fields(ResearchConfig))


def _as_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Require a YAML mapping and provide a useful location in any error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a YAML mapping.")
    return value


def _section(
    raw: Mapping[str, Any],
    name: str,
    config_type: type,
) -> dict[str, Any]:
    """Return a section after rejecting unknown keys."""
    values = _as_mapping(raw.get(name, {}), name)
    known = {field_.name for field_ in fields(config_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return dict(values)


def _immutable_universe(raw: Any) -> Mapping[str, tuple[str, ...]]:
    """Validate economic groups and freeze both mapping and symbol sequences."""
    groups = _as_mapping(raw, "universe")
    frozen: dict[str, tuple[str, ...]] = {}
    normalised_groups: set[str] = set()
    assigned_symbols: dict[str, str] = {}
    for group, symbols in groups.items():
        if not isinstance(group, str) or not group.strip():
            raise ValueError("universe group names must be non-empty strings.")
        if group != group.strip():
            raise ValueError("universe group names must not contain outer whitespace.")
        normalised_group = group.casefold()
        if normalised_group in normalised_groups:
            raise ValueError("universe group names must be unique ignoring case.")
        normalised_groups.add(normalised_group)
        if not isinstance(symbols, (list, tuple)):
            raise ValueError(f"universe.{group} must be a YAML sequence.")
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
            raise ValueError(f"universe.{group} symbols must be non-empty strings.")
        if any(symbol != symbol.strip() for symbol in symbols):
            raise ValueError(
                f"universe.{group} symbols must not contain outer whitespace."
            )
        normalised_symbols = [symbol.casefold() for symbol in symbols]
        if len(normalised_symbols) != len(set(normalised_symbols)):
            raise ValueError(
                f"universe.{group} contains duplicate symbols ignoring case."
            )
        for symbol, normalised_symbol in zip(symbols, normalised_symbols):
            if normalised_symbol in assigned_symbols:
                raise ValueError(
                    f"universe symbol {symbol!r} belongs to both "
                    f"{assigned_symbols[normalised_symbol]!r} and {group!r}."
                )
            assigned_symbols[normalised_symbol] = group
        frozen[group] = tuple(symbols)
    return MappingProxyType(frozen)


def screening_kwargs_from_config(config: ScreeningConfig) -> dict[str, int | float]:
    """Translate validated active screening fields to ``screen_pairs`` kwargs.

    Formation length belongs to fold construction and is therefore not passed
    to ``screen_pairs``. Historical correlation, raw-p-value, ADF-gate,
    minimum-half-life, and per-sector-cap fields were removed from the typed
    configuration because production screening does not implement them.
    """
    if not isinstance(config, ScreeningConfig):
        raise TypeError("config must be a ScreeningConfig instance.")
    _validate_screening_config(config)
    return {
        "min_observations": config.min_observations,
        "fdr_threshold": config.fdr_threshold,
        "max_half_life": config.max_half_life,
        "hurst_threshold": config.hurst_threshold,
    }


def load_config(path: str | Path = "config.yaml") -> ResearchConfig:
    """Safely load, type, freeze, and validate a YAML research configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    # PyYAML returns None for both an empty document and an explicit YAML null.
    # Other falsy values are still user-provided roots and must be rejected.
    raw = {} if loaded is None else _as_mapping(loaded, "configuration root")
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")

    config = ResearchConfig(
        data=DataConfig(**_section(raw, "data", DataConfig)),
        universe=_immutable_universe(raw.get("universe", {})),
        screening=ScreeningConfig(
            **_section(raw, "screening", ScreeningConfig)
        ),
        strategy=StrategyConfig(**_section(raw, "strategy", StrategyConfig)),
        costs=CostConfig(**_section(raw, "costs", CostConfig)),
        walk_forward=WalkForwardConfig(
            **_section(raw, "walk_forward", WalkForwardConfig)
        ),
        random_seed=raw.get("random_seed", 42),
    )
    _validate(config)
    return config


def _validate(config: ResearchConfig) -> None:
    """Validate cross-field rules and assumptions used by later milestones."""
    _validate_types(config)

    start = _iso_date(config.data.start, "data.start")
    end = None if config.data.end is None else _iso_date(config.data.end, "data.end")
    if end is not None and start >= end:
        raise ValueError("data.start must be earlier than data.end.")
    _non_empty_string(config.data.interval, "data.interval")
    _non_empty_string(config.data.cache_dir, "data.cache_dir")

    if not 0 < config.data.min_coverage <= 1:
        raise ValueError("data.min_coverage must be in (0, 1].")
    if config.data.max_forward_fill < 0:
        raise ValueError("data.max_forward_fill must be non-negative.")

    _validate_screening_config(config.screening)

    positive_strategy = {
        "strategy.hedge_lookback": config.strategy.hedge_lookback,
        "strategy.zscore_lookback": config.strategy.zscore_lookback,
        "strategy.max_holding_days": config.strategy.max_holding_days,
        "strategy.target_annual_vol": config.strategy.target_annual_vol,
        "strategy.max_gross_leverage": config.strategy.max_gross_leverage,
    }
    _require_positive(positive_strategy)
    if config.strategy.exit_z < 0:
        raise ValueError("strategy.exit_z must be non-negative.")
    if config.strategy.entry_z <= 0:
        raise ValueError("strategy.entry_z must be positive.")
    if config.strategy.stop_z <= 0:
        raise ValueError("strategy.stop_z must be positive.")
    if config.strategy.entry_z <= config.strategy.exit_z:
        raise ValueError("strategy.entry_z must be greater than strategy.exit_z.")
    if config.strategy.stop_z <= config.strategy.entry_z:
        raise ValueError("strategy.stop_z must be greater than strategy.entry_z.")
    if config.screening.formation_days <= config.strategy.hedge_lookback:
        raise ValueError("formation_days must exceed hedge_lookback.")
    if config.screening.formation_days <= config.strategy.zscore_lookback:
        raise ValueError("formation_days must exceed zscore_lookback.")
    if config.screening.formation_days < config.screening.min_observations:
        raise ValueError(
            "screening.formation_days must be at least "
            "screening.min_observations."
        )

    non_negative_costs = {
        "costs.commission_bps": config.costs.commission_bps,
        "costs.slippage_bps": config.costs.slippage_bps,
        "costs.annual_borrow_rate": config.costs.annual_borrow_rate,
    }
    for name, value in non_negative_costs.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    if config.walk_forward.trading_days <= 0:
        raise ValueError("walk_forward.trading_days must be positive.")
    if config.walk_forward.min_selected_pairs < 1:
        raise ValueError("walk_forward.min_selected_pairs must be at least 1.")


def _validate_screening_config(config: ScreeningConfig) -> None:
    """Validate constraints intrinsic to one screening configuration."""
    if not isinstance(config, ScreeningConfig):
        raise TypeError("config must be a ScreeningConfig instance.")

    integer_values = {
        "screening.formation_days": config.formation_days,
        "screening.min_observations": config.min_observations,
    }
    for name, value in integer_values.items():
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer.")

    numeric_values = {
        "screening.fdr_threshold": config.fdr_threshold,
        "screening.max_half_life": config.max_half_life,
        "screening.hurst_threshold": config.hurst_threshold,
    }
    for name, value in numeric_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if config.formation_days <= 0:
        raise ValueError("screening.formation_days must be positive.")
    if config.min_observations < ADF_MIN_OBSERVATIONS:
        raise ValueError(
            "screening.min_observations must be an integer of at least "
            f"{ADF_MIN_OBSERVATIONS}."
        )
    if not 0 <= config.fdr_threshold <= 1:
        raise ValueError("screening.fdr_threshold must be in [0, 1].")
    if config.max_half_life <= 0:
        raise ValueError("screening.max_half_life must be positive.")


def _validate_types(config: ResearchConfig) -> None:
    """Reject YAML coercions that violate the declared numeric field types."""
    integer_values = {
        "data.max_forward_fill": config.data.max_forward_fill,
        "screening.formation_days": config.screening.formation_days,
        "screening.min_observations": config.screening.min_observations,
        "strategy.hedge_lookback": config.strategy.hedge_lookback,
        "strategy.zscore_lookback": config.strategy.zscore_lookback,
        "strategy.max_holding_days": config.strategy.max_holding_days,
        "walk_forward.trading_days": config.walk_forward.trading_days,
        "walk_forward.min_selected_pairs": config.walk_forward.min_selected_pairs,
        "random_seed": config.random_seed,
    }
    for name, value in integer_values.items():
        # Exact type checking deliberately rejects bool, which subclasses int.
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer.")

    numeric_values = {
        "data.min_coverage": config.data.min_coverage,
        "screening.fdr_threshold": config.screening.fdr_threshold,
        "screening.max_half_life": config.screening.max_half_life,
        "screening.hurst_threshold": config.screening.hurst_threshold,
        "strategy.entry_z": config.strategy.entry_z,
        "strategy.exit_z": config.strategy.exit_z,
        "strategy.stop_z": config.strategy.stop_z,
        "strategy.target_annual_vol": config.strategy.target_annual_vol,
        "strategy.max_gross_leverage": config.strategy.max_gross_leverage,
        "costs.commission_bps": config.costs.commission_bps,
        "costs.slippage_bps": config.costs.slippage_bps,
        "costs.annual_borrow_rate": config.costs.annual_borrow_rate,
    }
    for name, value in numeric_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")


def _require_positive(values: Mapping[str, int | float]) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty.")
    if value != value.strip():
        raise ValueError(f"{name} must not contain outer whitespace.")
    return value


def _iso_date(value: Any, name: str) -> date:
    text = _non_empty_string(value, name)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 calendar date.") from exc
