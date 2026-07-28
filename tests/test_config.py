"""Tests for safe, strict, and immutable research configuration."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from pairs_trading.config import (
    CostConfig,
    DataConfig,
    ResearchConfig,
    ScreeningConfig,
    StrategyConfig,
    WalkForwardConfig,
    load_config,
)


def write_yaml(tmp_path: Path, values: Any) -> Path:
    """Write a temporary configuration file used by one test."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def write_raw_yaml(tmp_path: Path, text: str) -> Path:
    """Write exact YAML text when its document shape is under test."""
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_loading(tmp_path: Path) -> None:
    config = load_config(write_yaml(tmp_path, {}))

    assert config.data == DataConfig()
    assert config.screening == ScreeningConfig()
    assert config.strategy == StrategyConfig()
    assert config.costs == CostConfig()
    assert config.walk_forward == WalkForwardConfig()
    assert config.random_seed == 42
    assert dict(config.universe) == {}


def test_empty_yaml_file_loads_defaults(tmp_path: Path) -> None:
    assert load_config(write_raw_yaml(tmp_path, "")) == ResearchConfig()


def test_yaml_null_loads_defaults(tmp_path: Path) -> None:
    assert load_config(write_raw_yaml(tmp_path, "null\n")) == ResearchConfig()


@pytest.mark.parametrize("root", [[], False, 0, ""])
def test_falsy_non_mapping_roots_are_rejected(tmp_path: Path, root: Any) -> None:
    with pytest.raises(ValueError, match="configuration root must be a YAML mapping"):
        load_config(write_yaml(tmp_path, root))


@pytest.mark.parametrize(
    "values, expected",
    [
        ({"unexpected": True}, "Unknown top-level config keys"),
        ({"data": {"min_coverge": 0.9}}, "Unknown data keys"),
        ({"strategy": {"entry": 2.0}}, "Unknown strategy keys"),
    ],
)
def test_unknown_keys_are_rejected(
    tmp_path: Path, values: dict[str, Any], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        load_config(write_yaml(tmp_path, values))


@pytest.mark.parametrize(
    "strategy",
    [
        {"entry_z": 0.5, "exit_z": 0.5},
        {"entry_z": 0.4, "exit_z": 0.5},
        {"entry_z": 2.0, "stop_z": 2.0},
        {"entry_z": 2.0, "stop_z": 1.9},
    ],
)
def test_invalid_thresholds_are_rejected(
    tmp_path: Path, strategy: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        load_config(write_yaml(tmp_path, {"strategy": strategy}))


@pytest.mark.parametrize(
    "values",
    [
        {"screening": {"formation_days": 120}},
        {
            "screening": {"formation_days": 60},
            "strategy": {"hedge_lookback": 20, "zscore_lookback": 60},
        },
    ],
)
def test_invalid_formation_windows_are_rejected(
    tmp_path: Path, values: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="formation_days must exceed"):
        load_config(write_yaml(tmp_path, values))


@pytest.mark.parametrize("coverage", [0, -0.1, 1.01])
def test_invalid_coverage_is_rejected(tmp_path: Path, coverage: float) -> None:
    with pytest.raises(ValueError, match="min_coverage"):
        load_config(write_yaml(tmp_path, {"data": {"min_coverage": coverage}}))


@pytest.mark.parametrize(
    "field",
    ["commission_bps", "slippage_bps", "annual_borrow_rate"],
)
def test_negative_costs_are_rejected(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        load_config(write_yaml(tmp_path, {"costs": {field: -0.01}}))


@pytest.mark.parametrize(
    "strategy",
    [
        {"hedge_lookback": 0},
        {"zscore_lookback": 0},
        {"target_annual_vol": 0},
        {"max_gross_leverage": 0},
    ],
)
def test_non_positive_strategy_settings_are_rejected(
    tmp_path: Path, strategy: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        load_config(write_yaml(tmp_path, {"strategy": strategy}))


def test_minimum_selected_pairs_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_selected_pairs must be at least 1"):
        load_config(
            write_yaml(tmp_path, {"walk_forward": {"min_selected_pairs": 0}})
        )


def test_configuration_is_immutable(tmp_path: Path) -> None:
    config = load_config(
        write_yaml(tmp_path, {"universe": {"Technology": ["AAA", "BBB"]}})
    )

    with pytest.raises(FrozenInstanceError):
        config.strategy.entry_z = 3.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.universe["Technology"] = ("CCC",)  # type: ignore[index]
    assert config.universe["Technology"] == ("AAA", "BBB")


def test_non_mapping_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strategy must be a YAML mapping"):
        load_config(write_yaml(tmp_path, {"strategy": ["not", "a", "mapping"]}))


@pytest.mark.parametrize(
    "values, field",
    [
        ({"data": {"min_coverage": "0.95"}}, "data.min_coverage"),
        ({"strategy": {"entry_z": "2.0"}}, "strategy.entry_z"),
        ({"screening": {"formation_days": "504"}}, "screening.formation_days"),
        ({"random_seed": "42"}, "random_seed"),
    ],
)
def test_string_numeric_values_are_rejected(
    tmp_path: Path, values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        load_config(write_yaml(tmp_path, values))


@pytest.mark.parametrize(
    "values, field",
    [
        ({"data": {"min_coverage": True}}, "data.min_coverage"),
        ({"costs": {"commission_bps": False}}, "costs.commission_bps"),
        ({"strategy": {"hedge_lookback": True}}, "strategy.hedge_lookback"),
        ({"random_seed": False}, "random_seed"),
    ],
)
def test_boolean_numeric_values_are_rejected(
    tmp_path: Path, values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        load_config(write_yaml(tmp_path, values))


@pytest.mark.parametrize(
    "values, field",
    [
        ({"data": {"max_forward_fill": 3.5}}, "data.max_forward_fill"),
        ({"screening": {"formation_days": 504.5}}, "screening.formation_days"),
        ({"walk_forward": {"trading_days": 63.5}}, "walk_forward.trading_days"),
    ],
)
def test_fractional_integer_fields_are_rejected(
    tmp_path: Path, values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        load_config(write_yaml(tmp_path, values))


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_numeric_values_are_rejected(
    tmp_path: Path, value: float
) -> None:
    with pytest.raises(ValueError, match="strategy.target_annual_vol must be finite"):
        load_config(
            write_yaml(tmp_path, {"strategy": {"target_annual_vol": value}})
        )


@pytest.mark.parametrize(
    "strategy, field",
    [
        ({"exit_z": -0.1}, "strategy.exit_z"),
        ({"entry_z": -1.0}, "strategy.entry_z"),
        ({"stop_z": -3.0}, "strategy.stop_z"),
    ],
)
def test_negative_zscore_thresholds_are_rejected(
    tmp_path: Path, strategy: dict[str, float], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        load_config(write_yaml(tmp_path, {"strategy": strategy}))


def test_to_dict_returns_serialisable_mutable_values(tmp_path: Path) -> None:
    config = load_config(
        write_yaml(tmp_path, {"universe": {"Technology": ["AAA", "BBB"]}})
    )

    result = config.to_dict()

    assert isinstance(result["universe"], dict)
    assert result["universe"]["Technology"] == ["AAA", "BBB"]
    assert yaml.safe_load(yaml.safe_dump(result)) == result
