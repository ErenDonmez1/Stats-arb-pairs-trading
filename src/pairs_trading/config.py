"""Typed, immutable configuration for reproducible research runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


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
    """Candidate-pair formation and statistical screening settings."""

    formation_days: int = 504
    min_correlation: float = 0.60
    coint_pvalue: float = 0.05
    adf_pvalue: float = 0.05
    min_half_life: float = 2.0
    max_half_life: float = 90.0
    max_pairs_per_sector: int = 3


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
    for group, symbols in groups.items():
        if not isinstance(group, str) or not group.strip():
            raise ValueError("universe group names must be non-empty strings.")
        if not isinstance(symbols, (list, tuple)):
            raise ValueError(f"universe.{group} must be a YAML sequence.")
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
            raise ValueError(f"universe.{group} symbols must be non-empty strings.")
        frozen[group] = tuple(symbols)
    return MappingProxyType(frozen)


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

    if not 0 < config.data.min_coverage <= 1:
        raise ValueError("data.min_coverage must be in (0, 1].")
    if config.data.max_forward_fill < 0:
        raise ValueError("data.max_forward_fill must be non-negative.")

    positive_screening = {
        "screening.formation_days": config.screening.formation_days,
        "screening.min_half_life": config.screening.min_half_life,
        "screening.max_half_life": config.screening.max_half_life,
        "screening.max_pairs_per_sector": config.screening.max_pairs_per_sector,
    }
    _require_positive(positive_screening)
    if config.screening.min_half_life >= config.screening.max_half_life:
        raise ValueError("min_half_life must be smaller than max_half_life.")
    if not -1 <= config.screening.min_correlation <= 1:
        raise ValueError("screening.min_correlation must be in [-1, 1].")
    if not 0 < config.screening.coint_pvalue <= 1:
        raise ValueError("screening.coint_pvalue must be in (0, 1].")
    if not 0 < config.screening.adf_pvalue <= 1:
        raise ValueError("screening.adf_pvalue must be in (0, 1].")

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


def _validate_types(config: ResearchConfig) -> None:
    """Reject YAML coercions that violate the declared numeric field types."""
    integer_values = {
        "data.max_forward_fill": config.data.max_forward_fill,
        "screening.formation_days": config.screening.formation_days,
        "screening.max_pairs_per_sector": config.screening.max_pairs_per_sector,
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
        "screening.min_correlation": config.screening.min_correlation,
        "screening.coint_pvalue": config.screening.coint_pvalue,
        "screening.adf_pvalue": config.screening.adf_pvalue,
        "screening.min_half_life": config.screening.min_half_life,
        "screening.max_half_life": config.screening.max_half_life,
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
