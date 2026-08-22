"""Tests for market-data ingestion, cleaning, caching, and synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data import (
    DataQualityError,
    MarketDataLoader,
    OBSERVED_PRICE_MASK_ATTR,
    make_synthetic_universe,
)


def provider_frame(
    symbols: list[str],
    periods: int = 5,
    reversed_levels: bool = False,
) -> pd.DataFrame:
    """Build a realistic yfinance Close MultiIndex response."""
    dates = pd.bdate_range("2024-01-02", periods=periods)
    if reversed_levels:
        columns = pd.MultiIndex.from_product(
            [symbols, ["Close"]], names=["Ticker", "Price"]
        )
    else:
        columns = pd.MultiIndex.from_product(
            [["Close"], symbols], names=["Price", "Ticker"]
        )
    values = np.arange(periods * len(symbols), dtype=float).reshape(periods, -1) + 100
    return pd.DataFrame(values, index=dates, columns=columns)


def install_mock_yfinance(
    monkeypatch: pytest.MonkeyPatch,
    download,
) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "yfinance",
        SimpleNamespace(download=download),
    )


def test_duplicate_and_blank_tickers_are_normalized() -> None:
    symbols = MarketDataLoader.normalize_tickers(
        [" msft ", "", "AAPL", "MSFT", "  aapl  "]
    )

    assert symbols == ("AAPL", "MSFT")


@pytest.mark.parametrize("tickers", [[], [""], ["AAPL"], ["AAPL", " aapl "]])
def test_fewer_than_two_unique_tickers_are_rejected(
    tmp_path: Path, tickers: list[str]
) -> None:
    loader = MarketDataLoader(tmp_path / "cache")

    with pytest.raises(ValueError, match="At least two unique"):
        loader.download(tickers, start="2024-01-01")

    assert not loader.cache_dir.exists()


def test_synthetic_output_is_deterministic() -> None:
    first_prices, first_universe = make_synthetic_universe(seed=7)
    second_prices, second_universe = make_synthetic_universe(seed=7)

    pd.testing.assert_frame_equal(first_prices, second_prices)
    assert first_universe == second_universe


def test_synthetic_prices_are_strictly_positive() -> None:
    prices, _ = make_synthetic_universe(n_days=300)

    assert np.isfinite(prices.to_numpy()).all()
    assert prices.gt(0).all().all()


def test_synthetic_universe_contains_known_cointegrated_groups() -> None:
    prices, universe = make_synthetic_universe(n_days=300)

    assert {"TECH_A", "TECH_B"}.issubset(universe["Technology"])
    assert {"BANK_A", "BANK_B"}.issubset(universe["Financials"])
    assert {"TECH_C", "BANK_C"}.issubset(prices.columns)
    assert set(symbol for group in universe.values() for symbol in group) == set(
        prices.columns
    )


def test_different_seeds_produce_different_prices() -> None:
    first, _ = make_synthetic_universe(n_days=300, seed=1)
    second, _ = make_synthetic_universe(n_days=300, seed=2)

    assert not first.equals(second)


def test_clean_removes_duplicate_dates_deterministically() -> None:
    dates = pd.to_datetime(
        ["2024-01-03 16:00Z", "2024-01-02 16:00Z", "2024-01-02 16:00Z"]
    )
    original = pd.DataFrame(
        {"AAA": [103, 100, 101], "BBB": [203, 200, 201]}, index=dates
    )

    clean, _ = MarketDataLoader.clean(
        original, min_coverage=1.0, max_forward_fill=0, min_observations=2
    )

    assert isinstance(clean.index, pd.DatetimeIndex)
    assert clean.index.tz is None
    assert clean.index.is_monotonic_increasing
    assert clean.index.is_unique
    assert clean.iloc[0].to_dict() == {"AAA": 101.0, "BBB": 201.0}


def test_invalid_prices_are_reported_and_limited_forward_filled() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "AAA": [100, 0, -1, "bad", 104, 105],
            "BBB": [200, 201, 202, 203, 204, 205],
        },
        index=dates,
    )

    clean, report = MarketDataLoader.clean(
        prices, min_coverage=0.5, max_forward_fill=2, min_observations=5
    )

    assert report.loc["AAA", "non_positive"] == 2
    assert report.loc["AAA", "missing_or_invalid"] == 3
    assert report.loc["AAA", "forward_filled"] == 2
    assert clean.loc[dates[1], "AAA"] == 100
    assert clean.loc[dates[2], "AAA"] == 100
    assert dates[3] not in clean.index


def test_nonfinite_prices_are_invalid_before_coverage_fill_and_provenance() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "AAA": [100.0, np.inf, -np.inf, 103.0, 104.0],
            "BBB": [200.0, 201.0, 202.0, 203.0, 204.0],
        },
        index=dates,
    )
    before = prices.copy(deep=True)

    clean, report = MarketDataLoader.clean(
        prices,
        min_coverage=0.6,
        max_forward_fill=1,
        min_observations=4,
    )

    assert report.loc["AAA", "valid_observations"] == 3
    assert report.loc["AAA", "missing_or_invalid"] == 2
    assert report.loc["AAA", "coverage"] == pytest.approx(0.6)
    assert report.loc["AAA", "forward_filled"] == 1
    assert clean.loc[dates[1], "AAA"] == 100.0
    assert dates[2] not in clean.index
    assert np.isfinite(clean.to_numpy()).all()
    observed = clean.attrs[OBSERVED_PRICE_MASK_ATTR]
    assert not bool(observed.loc[dates[1], "AAA"])
    pd.testing.assert_frame_equal(prices, before)


def test_clean_preserves_observed_mask_for_forward_filled_prices() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "AAA": [100.0, np.nan, 102.0, 103.0, 104.0],
            "BBB": [200.0, 201.0, 202.0, 203.0, 204.0],
        },
        index=dates,
    )

    clean, _ = MarketDataLoader.clean(
        prices,
        min_coverage=0.8,
        max_forward_fill=1,
        min_observations=5,
    )

    observed = clean.attrs[OBSERVED_PRICE_MASK_ATTR]
    pd.testing.assert_index_equal(observed.index, clean.index)
    pd.testing.assert_index_equal(observed.columns, clean.columns)
    assert observed.dtypes.eq(bool).all()
    assert clean.loc[dates[1], "AAA"] == 100.0
    assert not bool(observed.loc[dates[1], "AAA"])
    assert bool(observed.loc[dates[1], "BBB"])
    assert bool(observed.loc[dates[2], "AAA"])
    assert "valuation_only" in clean.attrs["valuation_policy"]


def test_limited_forward_fill_never_backfills_earlier_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "AAA": [np.nan, 100, np.nan, np.nan, np.nan, 104],
            "BBB": [200, 201, 202, 203, 204, 205],
        },
        index=dates,
    )

    clean, _ = MarketDataLoader.clean(
        prices, min_coverage=0.3, max_forward_fill=2, min_observations=4
    )

    assert dates[0] not in clean.index
    assert clean.index.min() == dates[1]
    assert clean.loc[dates[2], "AAA"] == 100
    assert clean.loc[dates[3], "AAA"] == 100
    assert dates[4] not in clean.index


def test_low_coverage_symbol_is_removed_but_remains_in_report() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "AAA": [10, 11, 12, 13, 14],
            "BBB": [20, 21, 22, 23, 24],
            "LOW": [30, np.nan, np.nan, np.nan, np.nan],
        },
        index=dates,
    )

    clean, report = MarketDataLoader.clean(
        prices, min_coverage=0.8, max_forward_fill=1, min_observations=5
    )

    assert list(clean.columns) == ["AAA", "BBB"]
    assert bool(report.loc["LOW", "retained"]) is False
    assert report.loc["LOW", "coverage"] == pytest.approx(0.2)


def test_coverage_scope_is_the_complete_frame_supplied_by_the_caller() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    full = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 12.0, np.nan, np.nan, np.nan],
            "BBB": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            "CCC": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        },
        index=dates,
    )

    formation_clean, formation_report = MarketDataLoader.clean(
        full.iloc[:3], min_coverage=1.0, max_forward_fill=0, min_observations=3
    )
    _, full_report = MarketDataLoader.clean(
        full,
        min_coverage=0.75,
        max_forward_fill=0,
        min_observations=6,
    )

    assert "AAA" in formation_clean.columns
    assert bool(formation_report.loc["AAA", "retained"])
    assert not bool(full_report.loc["AAA", "retained"])


def test_observed_mask_aligns_after_symbol_and_complete_row_filtering() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "AAA": [100.0, np.nan, np.nan, 103.0, 104.0, 105.0],
            "BBB": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
            "LOW": [300.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=dates,
    )

    clean, report = MarketDataLoader.clean(
        prices,
        min_coverage=0.6,
        max_forward_fill=1,
        min_observations=5,
    )

    observed = clean.attrs[OBSERVED_PRICE_MASK_ATTR]
    assert list(clean.columns) == ["AAA", "BBB"]
    assert not bool(report.loc["LOW", "retained"])
    assert dates[2] not in clean.index
    pd.testing.assert_index_equal(observed.index, clean.index, exact=True)
    pd.testing.assert_index_equal(observed.columns, clean.columns, exact=True)
    assert observed.shape == clean.shape
    assert not bool(observed.loc[dates[1], "AAA"])
    assert bool(observed.loc[dates[1], "BBB"])


def test_clean_does_not_mutate_callers_dataframe() -> None:
    dates = pd.date_range("2024-01-01", periods=4, tz="Europe/London")
    prices = pd.DataFrame(
        {"AAA": ["10", "11", "12", "13"], "BBB": [20, 21, 22, 23]},
        index=dates,
    )
    before = prices.copy(deep=True)

    MarketDataLoader.clean(
        prices, min_coverage=1.0, max_forward_fill=0, min_observations=4
    )

    pd.testing.assert_frame_equal(prices, before)


def test_empty_data_is_rejected() -> None:
    with pytest.raises(DataQualityError, match="empty"):
        MarketDataLoader.clean(pd.DataFrame())


def test_insufficient_clean_data_is_rejected() -> None:
    prices = pd.DataFrame(
        {"AAA": [10, 11, 12], "BBB": [20, 21, 22]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )

    with pytest.raises(DataQualityError, match="require 4"):
        MarketDataLoader.clean(prices, min_observations=4)


@pytest.mark.parametrize(
    "response",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame(
            [[1.0, 2.0]],
            columns=pd.MultiIndex.from_product([["Open"], ["AAPL", "MSFT"]]),
        ),
    ],
)
def test_malformed_provider_output_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
) -> None:
    install_mock_yfinance(monkeypatch, lambda **_: response)
    loader = MarketDataLoader(tmp_path / "cache")

    with pytest.raises(DataQualityError):
        loader.download(["AAPL", "MSFT"], start="2024-01-01")

    assert not loader.cache_dir.exists()


def test_missing_requested_ticker_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_mock_yfinance(
        monkeypatch, lambda **_: provider_frame(["AAPL"], periods=3)
    )

    with pytest.raises(DataQualityError, match="MSFT"):
        MarketDataLoader(tmp_path / "cache").download(
            ["AAPL", "MSFT"], start="2024-01-01"
        )


def test_cache_round_trip_avoids_second_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        return provider_frame(kwargs["tickers"], periods=3, reversed_levels=True)

    install_mock_yfinance(monkeypatch, download)
    loader = MarketDataLoader(tmp_path / "cache")
    assert not loader.cache_dir.exists()

    first = loader.download(
        ["MSFT", "AAPL"],
        start="2024-01-01",
        end="2024-02-01",
        interval="1d",
    )
    second = loader.download(
        [" aapl ", "msft"],
        start="2024-01-01",
        end="2024-02-01",
        interval="1d",
    )

    assert len(calls) == 1
    assert calls[0] == {
        "tickers": ["AAPL", "MSFT"],
        "start": "2024-01-01",
        "end": "2024-02-01",
        "interval": "1d",
        "auto_adjust": True,
        "actions": False,
        "progress": False,
        "group_by": "column",
        "threads": True,
    }
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert len(list(loader.cache_dir.glob("*.csv"))) == 1


def test_cache_is_separate_for_different_ticker_universes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def download(**kwargs: Any) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return provider_frame(kwargs["tickers"], periods=3)

    install_mock_yfinance(monkeypatch, download)
    loader = MarketDataLoader(tmp_path / "cache")

    first = loader.download(["AAPL", "MSFT"], start="2024-01-01")
    second = loader.download(["AAPL", "GOOG"], start="2024-01-01")

    assert calls == 2
    assert list(first.columns) == ["AAPL", "MSFT"]
    assert list(second.columns) == ["AAPL", "GOOG"]
    assert len(list(loader.cache_dir.glob("*.csv"))) == 2


def test_cache_is_separate_for_different_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def download(**kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs["interval"])
        return provider_frame(kwargs["tickers"], periods=3)

    install_mock_yfinance(monkeypatch, download)
    loader = MarketDataLoader(tmp_path / "cache")

    loader.download(
        ["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-02-01",
        interval="1d",
    )
    loader.download(
        ["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-02-01",
        interval="1h",
    )

    assert calls == ["1d", "1h"]
    assert len(list(loader.cache_dir.glob("*.csv"))) == 2


def test_open_ended_cache_identity_advances_with_utc_as_of_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def download(**kwargs: Any) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return provider_frame(kwargs["tickers"], periods=3)

    install_mock_yfinance(monkeypatch, download)
    loader = MarketDataLoader(tmp_path / "cache")
    monkeypatch.setattr(loader, "_cache_as_of_date", lambda: "2024-02-01")
    loader.download(["AAPL", "MSFT"], start="2024-01-01")
    loader.download(["AAPL", "MSFT"], start="2024-01-01")
    monkeypatch.setattr(loader, "_cache_as_of_date", lambda: "2024-02-02")
    loader.download(["AAPL", "MSFT"], start="2024-01-01")

    assert calls == 2
    assert len(list(loader.cache_dir.glob("*.csv"))) == 2


def test_open_and_explicit_same_date_cache_requests_have_distinct_stable_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_ends: list[str | None] = []

    def download(**kwargs: Any) -> pd.DataFrame:
        requested_ends.append(kwargs["end"])
        return provider_frame(kwargs["tickers"], periods=3)

    install_mock_yfinance(monkeypatch, download)
    loader = MarketDataLoader(tmp_path / "cache")
    monkeypatch.setattr(loader, "_cache_as_of_date", lambda: "2024-02-01")

    loader.download(["AAPL", "MSFT"], start="2024-01-01")
    loader.download(["MSFT", "AAPL"], start="2024-01-01")
    loader.download(
        ["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-02-01",
    )
    loader.download(
        ["MSFT", "AAPL"],
        start="2024-01-01",
        end="2024-02-01",
    )

    assert requested_ends == [None, "2024-02-01"]
    assert len(list(loader.cache_dir.glob("*.csv"))) == 2


def test_cache_metadata_is_exact_and_malformed_metadata_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def download(**kwargs: Any) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return provider_frame(kwargs["tickers"], periods=3)

    install_mock_yfinance(monkeypatch, download)
    loader = MarketDataLoader(tmp_path / "cache")
    loader.download(
        ["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-02-01",
        interval="1h",
    )
    metadata_path = next(loader.cache_dir.glob("*.csv.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["symbols"] == ["AAPL", "MSFT"]
    assert metadata["start"] == "2024-01-01"
    assert metadata["resolved_end"] == "2024-02-01"
    assert metadata["interval"] == "1h"
    assert metadata["source"] == "yahoo_finance"
    assert metadata["adjustment_policy"] == "auto_adjust_true_close"
    assert metadata["format_version"] == 1
    assert metadata["retrieved_at_utc"].endswith("+00:00")

    metadata_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(DataQualityError, match="cache metadata"):
        loader.download(
            ["AAPL", "MSFT"],
            start="2024-01-01",
            end="2024-02-01",
            interval="1h",
        )
    assert calls == 1


@pytest.mark.parametrize("metadata_root", [[], False, 0, ""])
def test_non_mapping_cache_metadata_is_rejected_as_data_quality_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_root: Any,
) -> None:
    calls = 0

    def download(**kwargs: Any) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return provider_frame(kwargs["tickers"], periods=3)

    install_mock_yfinance(monkeypatch, download)
    loader = MarketDataLoader(tmp_path / "cache")
    loader.download(
        ["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-02-01",
    )
    metadata_path = next(loader.cache_dir.glob("*.csv.json"))
    metadata_path.write_text(json.dumps(metadata_root), encoding="utf-8")

    with pytest.raises(DataQualityError, match="JSON object"):
        loader.download(
            ["AAPL", "MSFT"],
            start="2024-01-01",
            end="2024-02-01",
        )
    assert calls == 1


def test_flattened_provider_columns_are_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = pd.DataFrame(
        {"msft": [200, 201], " aapl ": [100, 101]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    install_mock_yfinance(monkeypatch, lambda **_: raw)

    prices = MarketDataLoader(tmp_path / "cache").download(
        ["AAPL", "MSFT"], start="2024-01-01"
    )

    assert list(prices.columns) == ["AAPL", "MSFT"]


def test_mismatched_cache_is_rejected_without_provider_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def download(**kwargs: Any) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return provider_frame(kwargs["tickers"], periods=3)

    install_mock_yfinance(monkeypatch, download)
    loader = MarketDataLoader(tmp_path / "cache")
    loader.download(["AAPL", "MSFT"], start="2024-01-01")
    cache_file = next(loader.cache_dir.glob("*.csv"))
    cached = pd.read_csv(cache_file, index_col=0)
    cached["EXTRA"] = 1.0
    cached.to_csv(cache_file)

    with pytest.raises(DataQualityError, match="does not exactly match"):
        loader.download(["AAPL", "MSFT"], start="2024-01-01")

    assert calls == 1
