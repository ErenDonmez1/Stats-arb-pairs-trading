"""Tests for safe, strict, and immutable research configuration."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
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
    screening_kwargs_from_config,
)
from pairs_trading.data import make_synthetic_universe
from pairs_trading.screening import screen_pairs
from pairs_trading.stats import ADF_MIN_OBSERVATIONS


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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("start", False),
        ("start", "2024/01/01"),
        ("end", []),
        ("end", "not-a-date"),
        ("interval", 1),
        ("interval", "   "),
        ("cache_dir", False),
        ("cache_dir", ""),
    ),
)
def test_data_string_and_date_fields_are_validated(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError, match=f"data.{field}"):
        load_config(write_yaml(tmp_path, {"data": {field: value}}))


def test_data_start_must_precede_explicit_end(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start must be earlier"):
        load_config(
            write_yaml(
                tmp_path,
                {"data": {"start": "2024-01-02", "end": "2024-01-02"}},
            )
        )


@pytest.mark.parametrize(
    "universe",
    (
        {"Technology": ["AAA", " AAA"]},
        {"Technology": ["AAA", "aaa"]},
        {"Technology": ["AAA"], "Banks": ["aaa"]},
        {" Technology": ["AAA"]},
    ),
)
def test_universe_normalisation_collisions_fail_early(
    tmp_path: Path,
    universe: dict[str, list[str]],
) -> None:
    with pytest.raises(ValueError, match="universe|duplicate|belongs"):
        load_config(write_yaml(tmp_path, {"universe": universe}))


@pytest.mark.parametrize(
    "obsolete_field",
    (
        "min_correlation",
        "coint_pvalue",
        "adf_pvalue",
        "min_half_life",
        "max_pairs_per_sector",
    ),
)
def test_unsupported_legacy_screening_fields_are_rejected(
    tmp_path: Path,
    obsolete_field: str,
) -> None:
    with pytest.raises(ValueError, match="Unknown screening keys"):
        load_config(
            write_yaml(tmp_path, {"screening": {obsolete_field: 0.1}})
        )


def test_adapter_accepts_screening_from_valid_custom_strategy_config(
    tmp_path: Path,
) -> None:
    config = load_config(
        write_yaml(
            tmp_path,
            {
                "screening": {
                    "formation_days": 100,
                    "min_observations": ADF_MIN_OBSERVATIONS,
                },
                "strategy": {
                    "hedge_lookback": 60,
                    "zscore_lookback": 40,
                },
            },
        )
    )

    assert config.screening.formation_days <= StrategyConfig().hedge_lookback
    assert screening_kwargs_from_config(config.screening)["min_observations"] == (
        ADF_MIN_OBSERVATIONS
    )


def test_screening_min_observations_below_adf_floor_is_rejected(
    tmp_path: Path,
) -> None:
    invalid_minimum = ADF_MIN_OBSERVATIONS - 1

    with pytest.raises(ValueError, match=f"at least {ADF_MIN_OBSERVATIONS}"):
        load_config(
            write_yaml(
                tmp_path,
                {"screening": {"min_observations": invalid_minimum}},
            )
        )
    with pytest.raises(ValueError, match=f"at least {ADF_MIN_OBSERVATIONS}"):
        screening_kwargs_from_config(
            ScreeningConfig(min_observations=invalid_minimum)
        )


def test_screening_min_observations_at_adf_floor_is_valid(
    tmp_path: Path,
) -> None:
    config = load_config(
        write_yaml(
            tmp_path,
            {"screening": {"min_observations": ADF_MIN_OBSERVATIONS}},
        )
    )

    assert config.screening.min_observations == ADF_MIN_OBSERVATIONS
    assert screening_kwargs_from_config(config.screening)["min_observations"] == (
        ADF_MIN_OBSERVATIONS
    )


def test_formation_days_must_cover_screening_minimum_observations(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="formation_days must be at least screening.min_observations",
    ):
        load_config(
            write_yaml(
                tmp_path,
                {
                    "screening": {
                        "formation_days": ADF_MIN_OBSERVATIONS - 1,
                        "min_observations": ADF_MIN_OBSERVATIONS,
                    },
                    "strategy": {
                        "hedge_lookback": 20,
                        "zscore_lookback": 20,
                    },
                },
            )
        )


def test_formation_days_equal_to_min_observations_is_valid(
    tmp_path: Path,
) -> None:
    config = load_config(
        write_yaml(
            tmp_path,
            {
                "screening": {
                    "formation_days": ADF_MIN_OBSERVATIONS,
                    "min_observations": ADF_MIN_OBSERVATIONS,
                },
                "strategy": {
                    "hedge_lookback": 20,
                    "zscore_lookback": 20,
                },
            },
        )
    )

    assert config.screening.formation_days == config.screening.min_observations


def test_adapter_boundary_kwargs_are_accepted_by_screen_pairs() -> None:
    prices, groups = make_synthetic_universe(n_days=300, seed=42)
    kwargs = screening_kwargs_from_config(
        ScreeningConfig(
            formation_days=ADF_MIN_OBSERVATIONS,
            min_observations=ADF_MIN_OBSERVATIONS,
        )
    )

    results = screen_pairs(prices, groups, **kwargs)

    assert results


def test_screening_adapter_matches_active_production_parameters() -> None:
    config = ScreeningConfig(
        min_observations=150,
        fdr_threshold=0.10,
        max_half_life=45.0,
        hurst_threshold=0.40,
    )

    assert screening_kwargs_from_config(config) == {
        "min_observations": 150,
        "fdr_threshold": 0.10,
        "max_half_life": 45.0,
        "hurst_threshold": 0.40,
    }


def test_screening_adapter_drives_the_scalar_screening_api() -> None:
    prices, groups = make_synthetic_universe(n_days=500, seed=42)
    baseline = screen_pairs(
        prices,
        groups,
        **screening_kwargs_from_config(ScreeningConfig(min_observations=300)),
    )
    strict = screen_pairs(
        prices,
        groups,
        **screening_kwargs_from_config(
            ScreeningConfig(
                min_observations=300,
                fdr_threshold=0.0,
                max_half_life=1.0,
                hurst_threshold=-1.0,
            )
        ),
    )

    assert any(result.selected for result in baseline)
    assert not any(result.selected for result in strict)
    assert all(
        result.corrected_pvalue is None
        or np.isfinite(result.corrected_pvalue)
        for result in strict
    )
