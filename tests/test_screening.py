"""Tests for deterministic cointegration screening and FDR correction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pandas as pd
import pytest

import pairs_trading.screening as screening_module
from pairs_trading.data import make_synthetic_universe
from pairs_trading.screening import (
    PairCandidate,
    PairScreeningResult,
    benjamini_hochberg,
    generate_candidate_pairs,
    screen_pairs,
)


def compact_synthetic(
    n_days: int = 700,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    return make_synthetic_universe(n_days=n_days, seed=seed)


def test_all_universe_candidates_are_unique_and_alphabetical() -> None:
    prices = pd.DataFrame(columns=["CCC", "AAA", "BBB"])

    candidates = generate_candidate_pairs(prices)

    assert candidates == (
        PairCandidate("AAA", "BBB", None),
        PairCandidate("AAA", "CCC", None),
        PairCandidate("BBB", "CCC", None),
    )


def test_candidates_are_generated_only_within_groups() -> None:
    prices = pd.DataFrame(columns=["A", "B", "C", "D"])
    groups = {"Second": ["D", "C"], "First": ["B", "A"]}

    candidates = generate_candidate_pairs(prices, groups)

    assert candidates == (
        PairCandidate("A", "B", "First"),
        PairCandidate("C", "D", "Second"),
    )


@pytest.mark.parametrize(
    "groups, message",
    [
        ({"Group": ["A", "A"]}, "duplicate"),
        ({"One": ["A", "B"], "Two": ["B", "C"]}, "belongs to both"),
        ({"Group": ["A", "UNKNOWN"]}, "unknown"),
    ],
)
def test_invalid_group_assignments_are_rejected(
    groups: dict[str, list[str]],
    message: str,
) -> None:
    prices = pd.DataFrame(columns=["A", "B", "C"])

    with pytest.raises(ValueError, match=message):
        generate_candidate_pairs(prices, groups)


def test_benjamini_hochberg_matches_hand_calculation() -> None:
    pvalues = [0.01, 0.04, 0.03, 0.002]

    adjusted = benjamini_hochberg(pvalues)

    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])


def test_benjamini_hochberg_preserves_original_order() -> None:
    pvalues = [0.2, 0.001, 0.04, 0.8, 0.02]

    adjusted = benjamini_hochberg(pvalues)
    reordered = benjamini_hochberg(list(reversed(pvalues)))

    np.testing.assert_allclose(adjusted, reordered[::-1])


def test_benjamini_hochberg_is_monotonic_and_capped() -> None:
    pvalues = np.array([1.0, 0.9, 0.001, 0.7, 0.02])

    adjusted = benjamini_hochberg(pvalues)
    order = np.argsort(pvalues, kind="mergesort")

    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((0 <= adjusted) & (adjusted <= 1))
    assert adjusted.max() == 1.0


@pytest.mark.parametrize(
    "invalid",
    [
        [True, 0.2],
        ["0.1", 0.2],
        [np.nan, 0.2],
        [np.inf, 0.2],
        [-0.1, 0.2],
        [1.1, 0.2],
        [[0.1, 0.2]],
        {0.1, 0.2},
    ],
)
def test_benjamini_hochberg_rejects_invalid_pvalues(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        benjamini_hochberg(invalid)  # type: ignore[arg-type]


def test_known_cointegrated_pairs_rank_ahead_of_unrelated_pairs() -> None:
    prices, groups = compact_synthetic()

    results = screen_pairs(prices, groups, min_observations=300)
    positions = {
        (result.symbol_y, result.symbol_x): position
        for position, result in enumerate(results)
    }

    assert positions[("TECH_A", "TECH_B")] < positions[("TECH_A", "TECH_C")]
    assert positions[("BANK_A", "BANK_B")] < positions[("BANK_A", "BANK_C")]
    assert next(
        result for result in results
        if (result.symbol_y, result.symbol_x) == ("TECH_A", "TECH_B")
    ).corrected_pvalue < 0.05


def test_screening_result_contains_all_required_fields() -> None:
    required = {
        "symbol_y",
        "symbol_x",
        "group",
        "observations",
        "alpha",
        "beta",
        "spread_standard_deviation",
        "cointegration_statistic",
        "cointegration_pvalue",
        "corrected_pvalue",
        "cointegration_critical_values",
        "adf_statistic",
        "adf_pvalue",
        "half_life",
        "hurst",
        "selected",
        "rank",
        "rejection_reasons",
    }

    assert required.issubset(field.name for field in fields(PairScreeningResult))


def test_selected_ranks_are_consecutive_and_deterministic() -> None:
    prices, groups = compact_synthetic()

    first = screen_pairs(prices, groups, min_observations=300)
    second = screen_pairs(prices, groups, min_observations=300)
    selected = [result for result in first if result.selected]

    assert [result.rank for result in selected] == list(range(1, len(selected) + 1))
    assert all(result.rank is None for result in first if not result.selected)
    assert first == second


def test_reversed_duplicate_candidates_are_never_generated() -> None:
    prices = pd.DataFrame(columns=["D", "C", "B", "A"])

    candidates = generate_candidate_pairs(prices)
    pairs = {(candidate.symbol_y, candidate.symbol_x) for candidate in candidates}

    assert len(candidates) == 6
    assert len(pairs) == 6
    assert all(symbol_y < symbol_x for symbol_y, symbol_x in pairs)
    assert all((symbol_x, symbol_y) not in pairs for symbol_y, symbol_x in pairs)


def test_screening_does_not_mutate_input_prices() -> None:
    prices, groups = compact_synthetic()
    before = prices.copy(deep=True)

    screen_pairs(prices, groups, min_observations=300)

    pd.testing.assert_frame_equal(prices, before)


def test_formation_end_prevents_future_data_leakage() -> None:
    prices, groups = compact_synthetic(n_days=900)
    formation_end = prices.index[599]
    changed = prices.copy()
    changed.loc[changed.index > formation_end, :] *= np.linspace(
        1.0, 5.0, len(changed.loc[changed.index > formation_end])
    )[:, None]

    original = screen_pairs(
        prices,
        groups,
        formation_end=formation_end,
        min_observations=300,
    )
    altered = screen_pairs(
        changed,
        groups,
        formation_end=formation_end,
        min_observations=300,
    )

    assert original == altered


def test_missing_observations_are_aligned_pair_by_pair() -> None:
    prices, _ = compact_synthetic(n_days=500)
    prices.loc[prices.index[:7], "TECH_A"] = np.nan
    prices.loc[prices.index[10:15], "TECH_B"] = np.nan
    groups = {"Technology": ["TECH_A", "TECH_B"]}

    result = screen_pairs(prices, groups, min_observations=300)[0]

    assert result.observations == 488


def test_invalid_non_positive_and_insufficient_pairs_are_rejected() -> None:
    periods = 160
    index = pd.bdate_range("2020-01-01", periods=periods)
    rng = np.random.default_rng(91)
    common = 4.0 + np.cumsum(rng.normal(0.0, 0.01, periods))
    prices = pd.DataFrame(
        {
            "GOOD_A": np.exp(common + rng.normal(0.0, 0.01, periods)),
            "GOOD_B": np.exp(0.1 + 1.05 * common + rng.normal(0.0, 0.01, periods)),
            "BAD_A": np.exp(common),
            "BAD_B": np.r_[0.0, np.exp(common[1:])],
            "TEXT_A": np.exp(common),
            "TEXT_B": np.array(
                ["bad", *np.exp(common[1:])],
                dtype=object,
            ),
            "INF_A": np.exp(common),
            "INF_B": np.r_[np.inf, np.exp(common[1:])],
            "SHORT_A": np.r_[np.exp(common[:40]), np.full(periods - 40, np.nan)],
            "SHORT_B": np.r_[np.exp(common[:40]), np.full(periods - 40, np.nan)],
        },
        index=index,
    )
    groups = {
        "Good": ["GOOD_A", "GOOD_B"],
        "Bad": ["BAD_A", "BAD_B"],
        "Text": ["TEXT_A", "TEXT_B"],
        "Infinite": ["INF_A", "INF_B"],
        "Short": ["SHORT_A", "SHORT_B"],
    }

    results = screen_pairs(prices, groups, min_observations=100)
    by_group = {result.group: result for result in results}

    assert by_group["Bad"].selected is False
    assert by_group["Bad"].rejection_reasons == ("non_positive_prices",)
    assert by_group["Text"].rejection_reasons == ("non_numeric_prices",)
    assert by_group["Infinite"].rejection_reasons == ("non_finite_prices",)
    assert by_group["Short"].selected is False
    assert by_group["Short"].rejection_reasons == ("insufficient_observations",)
    assert by_group["Good"].observations == periods


def test_valid_cointegration_pvalues_remain_in_fdr_after_diagnostic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.bdate_range("2021-01-01", periods=100)
    prices = pd.DataFrame(
        {
            "A": np.exp(np.linspace(3.0, 4.0, len(index))),
            "B": np.exp(np.linspace(3.2, 4.1, len(index))),
            "C": np.exp(np.linspace(3.4, 4.4, len(index))),
        },
        index=index,
    )
    raw_pvalues = iter([0.01, 0.04, 0.20])

    def fake_coint(*args, **kwargs):
        return -3.0, next(raw_pvalues), np.array([-4.0, -3.5, -3.0])

    def fail_diagnostics(*args, **kwargs):
        raise ValueError("deliberate diagnostic failure")

    monkeypatch.setattr(screening_module, "coint", fake_coint)
    monkeypatch.setattr(screening_module, "diagnose_spread", fail_diagnostics)

    results = screen_pairs(prices, min_observations=50)
    by_pair = {
        (result.symbol_y, result.symbol_x): result for result in results
    }

    assert by_pair[("A", "B")].corrected_pvalue == pytest.approx(0.03)
    assert by_pair[("A", "C")].corrected_pvalue == pytest.approx(0.06)
    assert by_pair[("B", "C")].corrected_pvalue == pytest.approx(0.20)
    assert all(
        any(reason.startswith("spread_diagnostics_failed") for reason in result.rejection_reasons)
        for result in results
    )


def test_screening_results_are_deeply_immutable() -> None:
    prices, groups = compact_synthetic()
    result = screen_pairs(prices, groups, min_observations=300)[0]

    with pytest.raises(FrozenInstanceError):
        result.selected = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.cointegration_critical_values["5%"] = -999.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.rejection_reasons[0] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "start, end",
    [
        ("2030-01-01", "2030-12-31"),
        ("2022-01-01", "2021-01-01"),
    ],
)
def test_invalid_or_empty_formation_windows_are_rejected(
    start: str,
    end: str,
) -> None:
    prices, groups = compact_synthetic(n_days=300)

    with pytest.raises(ValueError):
        screen_pairs(
            prices,
            groups,
            formation_start=start,
            formation_end=end,
        )
