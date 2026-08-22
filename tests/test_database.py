"""Offline tests for DuckDB research persistence and SQL analysis."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from importlib import resources
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

import pairs_trading.database as database_module
from pairs_trading.backtest import build_position_schedule
from pairs_trading.database import (
    QUALITY_COLUMNS,
    SCREENING_RESULT_COLUMNS,
    SCREENING_SUMMARY_COLUMNS,
    connect_database,
    initialise_database,
    load_data_quality_report,
    load_pair_screening_results,
    load_prices,
    load_selected_pairs,
    store_data_quality_report,
    store_pair_screening_results,
    store_prices,
    summarise_screening_runs,
)
from pairs_trading.data import (
    OBSERVED_PRICE_MASK_ATTR,
    MarketDataLoader,
    make_synthetic_universe,
)
from pairs_trading.screening import PairScreeningResult, screen_pairs


PRICE_COLUMNS = ["ZZZ", "AAA"]


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ZZZ": [103.0, 101.0, 102.0],
            "AAA": [53.0, 51.0, 52.0],
        },
        index=pd.DatetimeIndex(
            ["2024-01-03", "2024-01-01", "2024-01-02"], name="Date"
        ),
    )


def _quality_report() -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "total_observations": [4, 4, 4],
            "valid_observations": [3, 4, 0],
            "missing_or_invalid": [1, 0, 4],
            "non_positive": [0, 0, 1],
            "coverage": [0.75, 1.0, 0.0],
            "stale_fraction": [0.0, 0.25, np.nan],
            "first_valid": pd.to_datetime(["2024-01-01", "2024-01-01", None]),
            "last_valid": pd.to_datetime(["2024-01-03", "2024-01-04", None]),
            "forward_filled": [1, 0, 0],
            "retained": [True, True, False],
        },
        index=pd.Index(["BBB", "AAA", "CCC"], name="symbol"),
    )
    return report.astype(
        {
            "total_observations": "int64",
            "valid_observations": "int64",
            "missing_or_invalid": "int64",
            "non_positive": "int64",
            "coverage": "float64",
            "stale_fraction": "float64",
            "forward_filled": "int64",
            "retained": "bool",
        }
    )


def _expected_prices(prices: pd.DataFrame) -> pd.DataFrame:
    expected = prices.copy(deep=True)
    expected.index = pd.DatetimeIndex(expected.index).normalize()
    expected.index.name = "date"
    expected = expected.sort_index().reindex(sorted(expected.columns), axis=1)
    expected = expected.astype(float)
    expected.columns.name = None
    expected.attrs[OBSERVED_PRICE_MASK_ATTR] = pd.DataFrame(
        True,
        index=expected.index.copy(),
        columns=expected.columns.copy(),
        dtype=bool,
    )
    expected.attrs["valuation_policy"] = (
        "database_observed_provenance; execution_requires_observed"
    )
    return expected


def _assert_connection_closed(connection: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def _screening_result(
    symbol_y: str = "AAA",
    symbol_x: str = "BBB",
    *,
    selected: bool = True,
    rank: int | None = 1,
    corrected_pvalue: float | None = 0.01,
    half_life: float | None = 10.0,
    hurst: float | None = 0.25,
    rejection_reasons: tuple[str, ...] = (),
    group: str | None = "Technology",
) -> PairScreeningResult:
    return PairScreeningResult(
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        group=group,
        observations=250,
        alpha=0.2,
        beta=1.1,
        spread_standard_deviation=0.03,
        cointegration_statistic=-4.2,
        cointegration_pvalue=0.004,
        corrected_pvalue=corrected_pvalue,
        cointegration_critical_values={"1%": -3.9, "5%": -3.3, "10%": -3.0},
        adf_statistic=-4.0,
        adf_pvalue=0.006,
        half_life=half_life,
        hurst=hurst,
        selected=selected,
        rank=rank,
        rejection_reasons=rejection_reasons,
    )


def _screening_batch() -> tuple[PairScreeningResult, ...]:
    return (
        _screening_result(
            "AAA", "CCC", rank=2, corrected_pvalue=0.03, half_life=20.0, hurst=0.35
        ),
        _screening_result(
            "AAA", "BBB", rank=1, corrected_pvalue=0.01, half_life=10.0, hurst=0.25
        ),
        _screening_result(
            "BBB",
            "DDD",
            selected=False,
            rank=None,
            corrected_pvalue=0.20,
            half_life=35.0,
            hurst=0.55,
            rejection_reasons=(
                "corrected_cointegration_pvalue_above_threshold",
                "hurst_not_below_threshold",
            ),
        ),
        _screening_result(
            "CCC",
            "DDD",
            selected=False,
            rank=None,
            corrected_pvalue=None,
            half_life=float("inf"),
            hurst=None,
            rejection_reasons=("half_life_not_finite_positive",),
            group=None,
        ),
    )


def _replace_screening_table_with_observation_limit(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    connection.execute("DROP TABLE pair_screening_results")
    connection.execute(
        """
        CREATE TABLE pair_screening_results (
            run_id VARCHAR NOT NULL,
            formation_start DATE NOT NULL,
            formation_end DATE NOT NULL,
            symbol_y VARCHAR NOT NULL,
            symbol_x VARCHAR NOT NULL,
            group_name VARCHAR,
            observations BIGINT NOT NULL CHECK (observations < 500),
            alpha DOUBLE,
            beta DOUBLE,
            spread_standard_deviation DOUBLE,
            cointegration_statistic DOUBLE,
            cointegration_pvalue DOUBLE,
            corrected_pvalue DOUBLE,
            cointegration_critical_values VARCHAR NOT NULL,
            adf_statistic DOUBLE,
            adf_pvalue DOUBLE,
            half_life DOUBLE,
            half_life_was_infinite BOOLEAN NOT NULL DEFAULT FALSE,
            hurst DOUBLE,
            selected BOOLEAN NOT NULL,
            rank BIGINT,
            rejection_reasons VARCHAR,
            loaded_at TIMESTAMP NOT NULL,
            UNIQUE (run_id, symbol_y, symbol_x)
        )
        """
    )


def test_schema_creation_is_idempotent_and_preserves_existing_data() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        connection.execute(
            """
            INSERT INTO prices VALUES
                (DATE '2024-01-01', 'AAA', 100.0, TRUE, 'test',
                 TIMESTAMP '2024-02-01')
            """
        )

        initialise_database(connection)

        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone() == (1,)
    finally:
        connection.close()


def test_schema_has_required_tables_columns_and_unique_keys() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        assert {
            row[0] for row in connection.execute("SHOW TABLES").fetchall()
        } == {"prices", "data_quality_reports", "pair_screening_results"}

        price_info = connection.execute("PRAGMA table_info('prices')").fetchall()
        quality_info = connection.execute(
            "PRAGMA table_info('data_quality_reports')"
        ).fetchall()
        assert [(row[1], row[2], bool(row[3])) for row in price_info] == [
            ("date", "DATE", True),
            ("symbol", "VARCHAR", True),
            ("adjusted_close", "DOUBLE", True),
            ("observed", "BOOLEAN", True),
            ("source", "VARCHAR", True),
            ("loaded_at", "TIMESTAMP", True),
        ]
        assert [(row[1], row[2], bool(row[3])) for row in quality_info] == [
            ("run_id", "VARCHAR", True),
            ("symbol", "VARCHAR", True),
            ("total_observations", "BIGINT", True),
            ("valid_observations", "BIGINT", True),
            ("missing_or_invalid", "BIGINT", True),
            ("non_positive", "BIGINT", True),
            ("coverage", "DOUBLE", True),
            ("stale_fraction", "DOUBLE", False),
            ("first_valid", "TIMESTAMP", False),
            ("last_valid", "TIMESTAMP", False),
            ("forward_filled", "BIGINT", True),
            ("retained", "BOOLEAN", True),
            ("loaded_at", "TIMESTAMP", True),
        ]

        duplicate_price = """
            INSERT INTO prices VALUES
            (DATE '2024-01-01', 'AAA', 1.0, TRUE, 'test',
             TIMESTAMP '2024-01-01'),
            (DATE '2024-01-01', 'AAA', 2.0, FALSE, 'test',
             TIMESTAMP '2024-01-02')
        """
        with pytest.raises(duckdb.ConstraintException):
            connection.execute(duplicate_price)

        duplicate_quality = """
            INSERT INTO data_quality_reports VALUES
            ('run', 'AAA', 1, 1, 0, 0, 1.0, NULL, NULL, NULL, 0, TRUE,
             TIMESTAMP '2024-01-01'),
            ('run', 'AAA', 2, 2, 0, 0, 1.0, NULL, NULL, NULL, 0, TRUE,
             TIMESTAMP '2024-01-02')
        """
        with pytest.raises(duckdb.ConstraintException):
            connection.execute(duplicate_quality)
    finally:
        connection.close()


def test_connect_database_creates_parent_directories_and_returns_open_connection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "research.duckdb"

    connection = connect_database(path)
    try:
        assert path.parent.is_dir()
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_connect_database_reuses_caller_connection() -> None:
    connection = duckdb.connect(":memory:")
    try:
        assert connect_database(connection) is connection
    finally:
        connection.close()


def test_caller_supplied_connection_remains_open_after_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        store_prices(connection, _prices(), source="test")
        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone() == (6,)

        monkeypatch.setattr(
            database_module,
            "_validate_schema",
            lambda connection: (_ for _ in ()).throw(
                RuntimeError("deliberate schema validation failure")
            ),
        )
        with pytest.raises(RuntimeError, match="deliberate"):
            initialise_database(connection)
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_read_only_caller_connection_can_load_existing_schema(tmp_path: Path) -> None:
    path = tmp_path / "read-only.duckdb"
    store_prices(path, _prices(), "test")
    connection = duckdb.connect(str(path), read_only=True)
    try:
        loaded = load_prices(connection, "test")
        assert not loaded.empty
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_module_owned_connections_close_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_connect = database_module.duckdb.connect
    opened: list[duckdb.DuckDBPyConnection] = []

    def tracking_connect(*args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        connection = actual_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(database_module.duckdb, "connect", tracking_connect)
    initialise_database(tmp_path / "owned.duckdb")

    assert len(opened) == 1
    _assert_connection_closed(opened[0])


def test_module_owned_connections_close_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_connect = database_module.duckdb.connect
    opened: list[duckdb.DuckDBPyConnection] = []

    def tracking_connect(*args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        connection = actual_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(database_module.duckdb, "connect", tracking_connect)
    monkeypatch.setattr(database_module, "_load_schema", lambda: "INVALID SQL")

    with pytest.raises(duckdb.ParserException):
        initialise_database(tmp_path / "failed.duckdb")

    assert len(opened) == 1
    _assert_connection_closed(opened[0])


def test_prices_round_trip_and_are_stored_as_long_rows(tmp_path: Path) -> None:
    path = tmp_path / "prices.duckdb"
    prices = _prices()

    assert store_prices(
        path,
        prices,
        source="synthetic",
        loaded_at="2024-02-01 12:00:00",
    ) == prices.size

    pd.testing.assert_frame_equal(
        load_prices(path, source="synthetic"), _expected_prices(prices)
    )
    connection = duckdb.connect(str(path), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT date, symbol, adjusted_close, observed, source, loaded_at
            FROM prices ORDER BY date, symbol
            """
        ).fetchall()
        assert len(rows) == prices.size
        assert rows[0] == (
            datetime(2024, 1, 1).date(),
            "AAA",
            51.0,
            True,
            "synthetic",
            datetime(2024, 2, 1, 12),
        )
    finally:
        connection.close()


def test_loaded_prices_are_sorted_by_date_and_symbol(tmp_path: Path) -> None:
    path = tmp_path / "sorted.duckdb"
    store_prices(path, _prices(), source="test")

    loaded = load_prices(path, source="test")

    assert loaded.index.is_monotonic_increasing
    assert loaded.columns.tolist() == ["AAA", "ZZZ"]


def test_price_filters_are_source_scoped_and_date_bounds_are_inclusive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "filters.duckdb"
    prices = _prices()
    store_prices(path, prices, source="source-a")
    store_prices(path, prices * 10, source="source-b")

    loaded = load_prices(
        path,
        source="source-a",
        symbols=["zzz"],
        start="2024-01-02",
        end=pd.Timestamp("2024-01-03"),
    )

    expected = _expected_prices(prices).loc["2024-01-02":"2024-01-03", ["ZZZ"]]
    pd.testing.assert_frame_equal(loaded, expected)
    assert load_prices(path, source="source-b").loc["2024-01-01", "AAA"] == 510.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", ""),
        ("source", "   "),
        ("source", 7),
        ("symbols", []),
        ("symbols", [""]),
        ("symbols", [7]),
        ("symbols", 7),
        ("start", True),
        ("start", 0),
        ("start", "not-a-date"),
        ("end", False),
        ("end", "not-a-date"),
    ],
)
def test_invalid_price_filters_are_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    arguments: dict[str, Any] = {"source": "test"}
    arguments[field] = value

    with pytest.raises((TypeError, ValueError)):
        load_prices(tmp_path / "filters.duckdb", **arguments)

    with pytest.raises(ValueError, match="start"):
        load_prices(
            tmp_path / "filters.duckdb",
            source="test",
            start="2024-01-03",
            end="2024-01-01",
        )


def test_repeated_price_write_updates_value_and_timestamp_without_duplicate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "upsert.duckdb"
    original = _prices().iloc[[0], [0]]
    replacement = original * 2
    original.attrs[OBSERVED_PRICE_MASK_ATTR] = pd.DataFrame(
        False, index=original.index, columns=original.columns
    )
    replacement.attrs[OBSERVED_PRICE_MASK_ATTR] = pd.DataFrame(
        True, index=replacement.index, columns=replacement.columns
    )
    store_prices(path, original, source="test", loaded_at="2024-02-01")
    store_prices(path, replacement, source="test", loaded_at="2024-02-02")

    connection = duckdb.connect(str(path), read_only=True)
    try:
        assert connection.execute(
            "SELECT COUNT(*), MIN(adjusted_close), MIN(observed), "
            "MIN(loaded_at) FROM prices"
        ).fetchone() == (1, 206.0, True, datetime(2024, 2, 2))
    finally:
        connection.close()


def test_same_price_key_from_different_sources_remains_separate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.duckdb"
    price = _prices().iloc[[0], [0]]
    store_prices(path, price, source="source-a")
    store_prices(path, price * 2, source="source-b")

    assert load_prices(path, source="source-a").iloc[0, 0] == 103.0
    assert load_prices(path, source="source-b").iloc[0, 0] == 206.0
    connection = duckdb.connect(str(path), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone() == (2,)
    finally:
        connection.close()


def test_store_prices_does_not_mutate_input(tmp_path: Path) -> None:
    prices = _prices()
    before = prices.copy(deep=True)

    store_prices(tmp_path / "immutable.duckdb", prices, source="test")

    pd.testing.assert_frame_equal(prices, before)


def test_cleaned_observation_provenance_survives_database_round_trip(
    tmp_path: Path,
) -> None:
    raw = pd.DataFrame(
        {
            "AAA": [100.0, np.nan, 102.0, 103.0],
            "BBB": [50.0, 51.0, 52.0, 53.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    clean, _ = MarketDataLoader.clean(
        raw,
        min_coverage=0.75,
        max_forward_fill=1,
        min_observations=4,
    )
    path = tmp_path / "observed-roundtrip.duckdb"

    store_prices(path, clean, source="cleaned")
    loaded = load_prices(path, source="cleaned", symbols=["AAA", "BBB"])

    pd.testing.assert_frame_equal(
        loaded,
        clean.rename_axis("date").astype(float),
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        loaded.attrs[OBSERVED_PRICE_MASK_ATTR],
        clean.attrs[OBSERVED_PRICE_MASK_ATTR].rename_axis("date"),
        check_freq=False,
    )
    assert bool(loaded.attrs[OBSERVED_PRICE_MASK_ATTR].loc["2024-01-02", "AAA"]) is False
    assert bool(loaded.attrs[OBSERVED_PRICE_MASK_ATTR].loc["2024-01-03", "AAA"]) is True
    assert "execution_requires_observed" in loaded.attrs["valuation_policy"]


def test_persisted_imputed_price_cannot_authorize_backtest_execution(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    raw = pd.DataFrame(
        {
            "AAA": [100.0, np.nan, 102.0],
            "BBB": [50.0, 51.0, 52.0],
        },
        index=dates,
    )
    clean, _ = MarketDataLoader.clean(
        raw,
        min_coverage=2 / 3,
        max_forward_fill=1,
        min_observations=3,
    )
    path = tmp_path / "provenance-execution.duckdb"
    store_prices(path, clean, source="cleaned")
    loaded = load_prices(
        path,
        source="cleaned",
        symbols=["AAA", "BBB"],
    )
    signals = pd.DataFrame(
        {
            "state": ["LONG_SPREAD", "LONG_SPREAD", "LONG_SPREAD"],
            "event": ["ENTER_LONG", "NONE", "NONE"],
        },
        index=loaded.index,
    )

    schedule = build_position_schedule(
        loaded["AAA"],
        loaded["BBB"],
        signals,
        hedge_ratio=1.0,
        target_gross_notional=2_000.0,
    )

    imputed_date = loaded.index[1]
    next_observed_date = loaded.index[2]
    assert np.isfinite(loaded.loc[imputed_date, "AAA"])
    assert not bool(
        loaded.attrs[OBSERVED_PRICE_MASK_ATTR].loc[imputed_date, "AAA"]
    )
    assert not bool(schedule.loc[imputed_date, "observed_y"])
    assert schedule.loc[imputed_date, "executed_state"] == "FLAT"
    assert schedule.loc[imputed_date, "execution_event"] == "NONE"
    assert schedule.loc[next_observed_date, "execution_event"] == "ENTER_LONG"


def test_price_provenance_slicing_preserves_exact_alignment(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "AAA": [100.0, np.nan, 102.0, 103.0],
            "BBB": [50.0, 51.0, 52.0, 53.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    clean, _ = MarketDataLoader.clean(
        raw,
        min_coverage=0.75,
        max_forward_fill=1,
        min_observations=4,
    )
    path = tmp_path / "observed-slicing.duckdb"
    store_prices(path, clean, source="cleaned")

    loaded = load_prices(
        path,
        source="cleaned",
        symbols=["BBB", "AAA"],
        start="2024-01-02",
        end="2024-01-03",
    )
    mask = loaded.attrs[OBSERVED_PRICE_MASK_ATTR]

    assert loaded.columns.tolist() == ["BBB", "AAA"]
    assert mask.index.equals(loaded.index)
    assert mask.columns.equals(loaded.columns)
    assert mask.dtypes.eq(bool).all()
    assert bool(mask.loc["2024-01-02", "AAA"]) is False


@pytest.mark.parametrize(
    "case",
    ["non-frame", "non-boolean", "missing", "index", "columns"],
)
def test_malformed_observed_price_masks_are_rejected(
    tmp_path: Path,
    case: str,
) -> None:
    prices = _prices().iloc[:2].copy(deep=True)
    mask: Any = pd.DataFrame(
        True, index=prices.index.copy(), columns=prices.columns.copy()
    )
    if case == "non-frame":
        mask = np.ones(prices.shape, dtype=bool)
    elif case == "non-boolean":
        mask = mask.astype(object)
        mask.iloc[0, 0] = 1
    elif case == "missing":
        mask = mask.astype(object)
        mask.iloc[0, 0] = pd.NA
    elif case == "index":
        mask.index = pd.date_range("2030-01-01", periods=len(mask))
    elif case == "columns":
        mask = mask.iloc[:, ::-1]
    prices.attrs[OBSERVED_PRICE_MASK_ATTR] = mask

    with pytest.raises((TypeError, ValueError), match="observed_price_mask"):
        store_prices(tmp_path / "bad-mask.duckdb", prices, source="test")


def test_cleaned_claim_without_provenance_mask_is_rejected(tmp_path: Path) -> None:
    prices = _prices()
    prices.attrs["valuation_policy"] = "cleaned valuation data"

    with pytest.raises(ValueError, match="observed_price_mask"):
        store_prices(tmp_path / "missing-mask.duckdb", prices, source="test")


def test_timezone_aware_price_dates_are_converted_to_utc_naive_dates(
    tmp_path: Path,
) -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0]},
        index=pd.DatetimeIndex(["2024-01-01 23:30"], tz="America/New_York"),
    )
    path = tmp_path / "timezone.duckdb"

    store_prices(path, prices, source="test")

    loaded = load_prices(path, source="test")
    assert loaded.index.equals(pd.DatetimeIndex(["2024-01-02"], name="date"))
    assert loaded.index.tz is None


@pytest.mark.parametrize(
    "case",
    [
        "not-frame",
        "empty",
        "no-symbols",
        "non-datetime-index",
        "nat-index",
        "duplicate-timestamp",
        "duplicate-database-date",
        "duplicate-symbol",
        "duplicate-normalised-symbol",
        "blank-symbol",
        "non-string-symbol",
        "numeric-string",
        "boolean",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "zero",
        "negative",
        "complex",
        "overflow",
        "out-of-ns-date",
    ],
)
def test_invalid_price_frames_are_rejected(tmp_path: Path, case: str) -> None:
    candidate: Any = _prices().iloc[:2, :1].copy(deep=True)
    if case == "not-frame":
        candidate = [[1.0]]
    elif case == "empty":
        candidate = pd.DataFrame()
    elif case == "no-symbols":
        candidate = pd.DataFrame(index=pd.DatetimeIndex(["2024-01-01"]))
    elif case == "non-datetime-index":
        candidate.index = ["one", "two"]
    elif case == "nat-index":
        candidate.index = pd.DatetimeIndex(["2024-01-01", pd.NaT])
    elif case == "duplicate-timestamp":
        candidate.index = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    elif case == "duplicate-database-date":
        candidate.index = pd.DatetimeIndex(
            ["2024-01-01 01:00", "2024-01-01 12:00"]
        )
    elif case == "duplicate-symbol":
        candidate = pd.concat([candidate, candidate], axis=1)
    elif case == "duplicate-normalised-symbol":
        candidate = pd.concat([candidate, candidate], axis=1)
        candidate.columns = ["AAA", " aaa "]
    elif case == "blank-symbol":
        candidate.columns = [" "]
    elif case == "non-string-symbol":
        candidate.columns = [7]
    elif case == "out-of-ns-date":
        candidate.index = pd.DatetimeIndex(
            np.array(["2500-01-01", "2500-01-02"], dtype="datetime64[s]")
        )
    else:
        invalid_values = {
            "numeric-string": "100.0",
            "boolean": True,
            "nan": np.nan,
            "positive-infinity": np.inf,
            "negative-infinity": -np.inf,
            "zero": 0.0,
            "negative": -1.0,
            "complex": 1 + 2j,
            "overflow": 10**400,
        }
        candidate = candidate.astype(object)
        candidate.iloc[0, 0] = invalid_values[case]

    with pytest.raises((TypeError, ValueError)):
        store_prices(tmp_path / "invalid.duckdb", candidate, source="test")


@pytest.mark.parametrize("source", [None, "", "   ", 4, False])
def test_invalid_source_names_are_rejected(tmp_path: Path, source: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        store_prices(tmp_path / "invalid-source.duckdb", _prices(), source=source)


def test_legitimate_empty_price_query_returns_typed_empty_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty-query.duckdb"
    initialise_database(path)

    result = load_prices(
        path,
        source="unknown",
        symbols=["BBB", "AAA"],
        strict=False,
    )

    assert result.empty
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.dtype == "datetime64[ns]"
    assert result.columns.tolist() == ["BBB", "AAA"]
    assert result.attrs[OBSERVED_PRICE_MASK_ATTR].columns.tolist() == ["BBB", "AAA"]


def test_explicit_price_load_is_strict_by_default_and_non_strict_is_complete(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial-universe.duckdb"
    store_prices(path, _prices(), source="test")

    with pytest.raises(ValueError, match="missing requested symbols.*CCC"):
        load_prices(path, source="test", symbols=["ZZZ", "CCC", "AAA"])

    loaded = load_prices(
        path,
        source="test",
        symbols=["ZZZ", "CCC", "AAA"],
        strict=False,
    )
    mask = loaded.attrs[OBSERVED_PRICE_MASK_ATTR]
    assert loaded.columns.tolist() == ["ZZZ", "CCC", "AAA"]
    assert loaded["CCC"].isna().all()
    assert mask["CCC"].eq(False).all()
    assert mask.columns.equals(loaded.columns)


def test_date_filtered_symbol_absence_is_not_silently_dropped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "date-filtered-universe.duckdb"
    prices = pd.DataFrame(
        {"AAA": [100.0, 101.0], "BBB": [np.nan, 50.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    store_prices(path, prices[["AAA"]], source="test")
    store_prices(path, prices.loc[[prices.index[1]], ["BBB"]], source="test")

    with pytest.raises(ValueError, match="missing requested symbols.*BBB"):
        load_prices(
            path,
            source="test",
            symbols=["AAA", "BBB"],
            start="2024-01-01",
            end="2024-01-01",
        )

    loaded = load_prices(
        path,
        source="test",
        symbols=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-01-01",
        strict=False,
    )
    assert loaded.columns.tolist() == ["AAA", "BBB"]
    assert pd.isna(loaded.loc["2024-01-01", "BBB"])


def test_duplicate_requested_symbols_are_rejected_before_query(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicates.duckdb"
    initialise_database(path)

    with pytest.raises(ValueError, match="duplicates after normalisation"):
        load_prices(path, source="test", symbols=["AAA", " aaa "])


def test_strict_empty_price_load_raises_for_complete_requested_universe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-empty.duckdb"
    initialise_database(path)

    with pytest.raises(ValueError, match="none of the requested symbols"):
        load_prices(path, source="test", symbols=["AAA", "BBB"])


def test_missing_database_reads_do_not_create_typo_files(tmp_path: Path) -> None:
    paths = [tmp_path / f"missing-{position}.duckdb" for position in range(4)]

    with pytest.raises(FileNotFoundError):
        load_prices(paths[0], source="test")
    with pytest.raises(FileNotFoundError):
        load_data_quality_report(paths[1], run_id="run")
    with pytest.raises(FileNotFoundError):
        load_pair_screening_results(paths[2], run_id="run")
    with pytest.raises(FileNotFoundError):
        summarise_screening_runs(paths[3])

    assert all(not path.exists() for path in paths)


def test_legacy_price_schema_without_provenance_is_rejected() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        connection.execute("DROP TABLE prices")
        connection.execute(
            """
            CREATE TABLE prices (
                date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                adjusted_close DOUBLE NOT NULL,
                source VARCHAR NOT NULL,
                loaded_at TIMESTAMP NOT NULL,
                UNIQUE (date, symbol, source)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO prices VALUES (
                DATE '2024-01-01', 'AAA', 100.0, 'legacy',
                TIMESTAMP '2024-02-01'
            )
            """
        )

        with pytest.raises(RuntimeError, match="missing columns.*observed"):
            load_prices(connection, source="legacy")

        assert connection.execute(
            "SELECT adjusted_close FROM prices"
        ).fetchall() == [(100.0,)]
    finally:
        connection.close()


def test_failed_price_batch_rolls_back_without_partial_updates() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        connection.execute("DROP TABLE prices")
        connection.execute(
            """
            CREATE TABLE prices (
                date DATE NOT NULL,
                symbol VARCHAR NOT NULL,
                adjusted_close DOUBLE NOT NULL CHECK (adjusted_close < 150),
                observed BOOLEAN NOT NULL,
                source VARCHAR NOT NULL,
                loaded_at TIMESTAMP NOT NULL,
                UNIQUE (date, symbol, source)
            )
            """
        )
        baseline = pd.DataFrame(
            {"AAA": [100.0]}, index=pd.DatetimeIndex(["2024-01-01"])
        )
        baseline.attrs[OBSERVED_PRICE_MASK_ATTR] = pd.DataFrame(
            False, index=baseline.index, columns=baseline.columns
        )
        store_prices(connection, baseline, source="test", loaded_at="2024-02-01")
        failing_batch = pd.DataFrame(
            {"AAA": [120.0], "BBB": [200.0]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )

        with pytest.raises(duckdb.ConstraintException):
            store_prices(
                connection,
                failing_batch,
                source="test",
                loaded_at="2024-02-02",
            )

        assert connection.execute(
            "SELECT symbol, adjusted_close, observed, loaded_at FROM prices"
        ).fetchall() == [("AAA", 100.0, False, datetime(2024, 2, 1))]
    finally:
        connection.close()


def test_quality_report_round_trip_preserves_values_types_and_symbol_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quality.duckdb"
    report = _quality_report()

    assert store_data_quality_report(
        path, report, run_id="run-001", loaded_at="2024-02-01"
    ) == len(report)
    loaded = load_data_quality_report(path, run_id="run-001")

    expected = report.sort_index().loc[:, list(QUALITY_COLUMNS)]
    pd.testing.assert_frame_equal(loaded, expected)
    assert all(pd.api.types.is_integer_dtype(loaded[column]) for column in [
        "total_observations",
        "valid_observations",
        "missing_or_invalid",
        "non_positive",
        "forward_filled",
    ])
    assert pd.api.types.is_bool_dtype(loaded["retained"])
    assert pd.api.types.is_datetime64_ns_dtype(loaded["first_valid"])
    assert pd.api.types.is_datetime64_ns_dtype(loaded["last_valid"])


def test_market_data_loader_quality_report_is_persistable(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {"AAA": [10.0, 0.0, 12.0], "BBB": [20.0, 20.0, 21.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    _, report = MarketDataLoader.clean(
        raw, min_coverage=0.5, max_forward_fill=1, min_observations=2
    )
    path = tmp_path / "cleaner-contract.duckdb"

    store_data_quality_report(path, report, "clean-run")
    loaded = load_data_quality_report(path, "clean-run")

    pd.testing.assert_frame_equal(loaded, report.sort_index())


def test_repeated_quality_write_updates_fields_and_timestamp_without_duplicate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quality-upsert.duckdb"
    report = _quality_report().iloc[[0]].copy(deep=True)
    store_data_quality_report(
        path, report, run_id="run", loaded_at="2024-02-01"
    )
    replacement = report.copy(deep=True)
    replacement.loc[:, "valid_observations"] = 2
    replacement.loc[:, "missing_or_invalid"] = 2
    replacement.loc[:, "coverage"] = 0.5
    replacement.loc[:, "retained"] = False
    store_data_quality_report(
        path, replacement, run_id="run", loaded_at="2024-02-02"
    )

    connection = duckdb.connect(str(path), read_only=True)
    try:
        assert connection.execute(
            "SELECT COUNT(*), MIN(coverage), BOOL_AND(NOT retained), MIN(loaded_at) "
            "FROM data_quality_reports"
        ).fetchone() == (1, 0.5, True, datetime(2024, 2, 2))
    finally:
        connection.close()


def test_unknown_run_id_returns_typed_empty_quality_report(tmp_path: Path) -> None:
    path = tmp_path / "unknown-run.duckdb"
    initialise_database(path)

    loaded = load_data_quality_report(path, run_id="not-present")

    assert loaded.empty
    assert loaded.index.name == "symbol"
    assert loaded.columns.tolist() == list(QUALITY_COLUMNS)
    assert pd.api.types.is_bool_dtype(loaded["retained"])


def test_store_quality_report_does_not_mutate_input(tmp_path: Path) -> None:
    report = _quality_report()
    before = report.copy(deep=True)

    store_data_quality_report(tmp_path / "immutable.duckdb", report, run_id="run")

    pd.testing.assert_frame_equal(report, before)


def test_missing_required_quality_report_columns_are_rejected(
    tmp_path: Path,
) -> None:
    report = _quality_report().drop(columns=["coverage"])

    with pytest.raises(ValueError, match="coverage"):
        store_data_quality_report(tmp_path / "missing.duckdb", report, run_id="run")


@pytest.mark.parametrize(
    ("case", "value"),
    [
        ("negative-count", -1),
        ("fractional-count", 1.5),
        ("string-count", "1"),
        ("boolean-count", True),
        ("coverage-negative", -0.1),
        ("coverage-high", 1.1),
        ("coverage-nan", np.nan),
        ("coverage-infinity", np.inf),
        ("coverage-string", "0.5"),
        ("coverage-boolean", True),
        ("coverage-overflow", 10**400),
        ("stale-negative", -0.1),
        ("stale-high", 1.1),
        ("stale-infinity", np.inf),
        ("stale-string", "0.5"),
        ("retained-integer", 1),
        ("retained-string", "true"),
        ("first-valid-malformed", "not-a-date"),
        ("first-valid-numeric", 1),
        ("first-after-last", pd.Timestamp("2025-01-01")),
    ],
)
def test_invalid_quality_report_values_are_rejected(
    tmp_path: Path, case: str, value: Any
) -> None:
    report = _quality_report().iloc[[0]].copy(deep=True)
    if "count" in case:
        report["total_observations"] = pd.Series(
            [value], index=report.index, dtype=object
        )
    elif case.startswith("coverage"):
        report["coverage"] = pd.Series([value], index=report.index, dtype=object)
    elif case.startswith("stale"):
        report["stale_fraction"] = pd.Series(
            [value], index=report.index, dtype=object
        )
    elif case.startswith("retained"):
        report["retained"] = pd.Series([value], index=report.index, dtype=object)
    else:
        report["first_valid"] = pd.Series(
            [value], index=report.index, dtype=object
        )

    with pytest.raises((TypeError, ValueError)):
        store_data_quality_report(
            tmp_path / "invalid-quality.duckdb", report, run_id="run"
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"valid_observations": 2, "missing_or_invalid": 1},
            "must equal total_observations",
        ),
        ({"non_positive": 2}, "must not exceed missing_or_invalid"),
        ({"forward_filled": 2}, "must not exceed missing_or_invalid"),
        ({"coverage": 0.5}, "must equal valid_observations"),
        (
            {
                "total_observations": 0,
                "valid_observations": 0,
                "missing_or_invalid": 0,
                "coverage": 0.0,
            },
            "must be positive",
        ),
    ],
)
def test_quality_report_cross_field_identities_are_validated(
    tmp_path: Path,
    updates: dict[str, Any],
    message: str,
) -> None:
    report = _quality_report().loc[["BBB"]].copy(deep=True)
    for field, value in updates.items():
        report.loc["BBB", field] = value

    with pytest.raises(ValueError, match=message):
        store_data_quality_report(
            tmp_path / "invalid-quality-identity.duckdb",
            report,
            run_id="run",
        )


@pytest.mark.parametrize(
    "case",
    [
        "not-frame",
        "empty",
        "duplicate-symbol",
        "duplicate-normalised-symbol",
        "blank-symbol",
        "non-string-symbol",
        "multi-index",
    ],
)
def test_malformed_quality_report_frames_are_rejected(
    tmp_path: Path, case: str
) -> None:
    candidate: Any = _quality_report()
    if case == "not-frame":
        candidate = {"coverage": [1.0]}
    elif case == "empty":
        candidate = candidate.iloc[0:0]
    elif case == "duplicate-symbol":
        candidate.index = ["AAA", "AAA", "CCC"]
    elif case == "duplicate-normalised-symbol":
        candidate.index = ["AAA", " aaa ", "CCC"]
    elif case == "blank-symbol":
        candidate.index = ["AAA", " ", "CCC"]
    elif case == "non-string-symbol":
        candidate.index = ["AAA", 2, "CCC"]
    elif case == "multi-index":
        candidate.index = pd.MultiIndex.from_tuples(
            [("A", 1), ("B", 1), ("C", 1)]
        )

    with pytest.raises((TypeError, ValueError)):
        store_data_quality_report(
            tmp_path / "malformed-quality.duckdb", candidate, run_id="run"
        )


@pytest.mark.parametrize("run_id", [None, "", "   ", 4, False])
def test_invalid_run_ids_are_rejected(tmp_path: Path, run_id: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        store_data_quality_report(
            tmp_path / "invalid-run.duckdb", _quality_report(), run_id=run_id
        )
    with pytest.raises((TypeError, ValueError)):
        load_data_quality_report(tmp_path / "invalid-run.duckdb", run_id=run_id)


@pytest.mark.parametrize(
    "loaded_at",
    [
        True,
        0,
        np.nan,
        "",
        "not-a-date",
        "2500-01-01",
        pd.Timestamp("2024-01-01 00:00:00.000000001"),
    ],
)
def test_invalid_loaded_at_values_are_rejected(
    tmp_path: Path, loaded_at: Any
) -> None:
    with pytest.raises((TypeError, ValueError)):
        store_prices(
            tmp_path / "invalid-time.duckdb",
            _prices(),
            source="test",
            loaded_at=loaded_at,
        )
    with pytest.raises((TypeError, ValueError)):
        store_data_quality_report(
            tmp_path / "invalid-quality-time.duckdb",
            _quality_report(),
            run_id="run",
            loaded_at=loaded_at,
        )


def test_failed_quality_batch_rolls_back_without_partial_updates() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        connection.execute("DROP TABLE data_quality_reports")
        connection.execute(
            """
            CREATE TABLE data_quality_reports (
                run_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                total_observations BIGINT NOT NULL,
                valid_observations BIGINT NOT NULL,
                missing_or_invalid BIGINT NOT NULL,
                non_positive BIGINT NOT NULL,
                coverage DOUBLE NOT NULL CHECK (coverage < 0.9),
                stale_fraction DOUBLE,
                first_valid TIMESTAMP,
                last_valid TIMESTAMP,
                forward_filled BIGINT NOT NULL,
                retained BOOLEAN NOT NULL,
                loaded_at TIMESTAMP NOT NULL,
                UNIQUE (run_id, symbol)
            )
            """
        )
        baseline = _quality_report().loc[["BBB"]].copy(deep=True)
        store_data_quality_report(
            connection, baseline, run_id="run", loaded_at="2024-02-01"
        )
        failing = _quality_report().loc[["BBB", "AAA"]].copy(deep=True)
        failing.loc["AAA", "coverage"] = 1.0

        with pytest.raises(duckdb.ConstraintException):
            store_data_quality_report(
                connection, failing, run_id="run", loaded_at="2024-02-02"
            )

        assert connection.execute(
            "SELECT symbol, coverage, loaded_at FROM data_quality_reports"
        ).fetchall() == [("BBB", 0.75, datetime(2024, 2, 1))]
    finally:
        connection.close()


def test_sql_values_are_parameterised(tmp_path: Path) -> None:
    path = tmp_path / "parameters.duckdb"
    source = "provider'; DROP TABLE prices; --"
    run_id = "run'; DROP TABLE data_quality_reports; --"

    store_prices(path, _prices().iloc[[0], [0]], source=source)
    store_data_quality_report(
        path, _quality_report().iloc[[0]], run_id=run_id
    )

    assert not load_prices(path, source=source).empty
    assert not load_data_quality_report(path, run_id=run_id).empty
    connection = duckdb.connect(str(path), read_only=True)
    try:
        assert {
            row[0] for row in connection.execute("SHOW TABLES").fetchall()
        } == {"prices", "data_quality_reports", "pair_screening_results"}
    finally:
        connection.close()


def test_screening_schema_is_idempotent_and_preserves_4ea_data() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        store_prices(
            connection,
            _prices().iloc[[0], [0]],
            "test",
            loaded_at="2024-02-01",
        )
        store_data_quality_report(
            connection,
            _quality_report().iloc[[0]],
            "quality-run",
            loaded_at="2024-02-01",
        )
        store_pair_screening_results(
            connection,
            [_screening_result()],
            "screen-run",
            "2023-01-01",
            "2023-12-31",
            loaded_at="2024-02-01",
        )

        initialise_database(connection)

        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM data_quality_reports"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM pair_screening_results"
        ).fetchone() == (1,)
    finally:
        connection.close()


def test_screening_schema_has_required_columns_and_unique_key() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        table_info = connection.execute(
            "PRAGMA table_info('pair_screening_results')"
        ).fetchall()
        assert [(row[1], row[2], bool(row[3])) for row in table_info] == [
            ("run_id", "VARCHAR", True),
            ("formation_start", "DATE", True),
            ("formation_end", "DATE", True),
            ("symbol_y", "VARCHAR", True),
            ("symbol_x", "VARCHAR", True),
            ("group_name", "VARCHAR", False),
            ("observations", "BIGINT", True),
            ("alpha", "DOUBLE", False),
            ("beta", "DOUBLE", False),
            ("spread_standard_deviation", "DOUBLE", False),
            ("cointegration_statistic", "DOUBLE", False),
            ("cointegration_pvalue", "DOUBLE", False),
            ("corrected_pvalue", "DOUBLE", False),
            ("cointegration_critical_values", "VARCHAR", True),
            ("adf_statistic", "DOUBLE", False),
            ("adf_pvalue", "DOUBLE", False),
            ("half_life", "DOUBLE", False),
            ("hurst", "DOUBLE", False),
            ("selected", "BOOLEAN", True),
            ("rank", "BIGINT", False),
            ("rejection_reasons", "VARCHAR", False),
            ("loaded_at", "TIMESTAMP", True),
            ("half_life_was_infinite", "BOOLEAN", True),
        ]
        store_pair_screening_results(
            connection,
            [_screening_result()],
            "run",
            "2023-01-01",
            "2023-12-31",
        )

        with pytest.raises(duckdb.ConstraintException):
            connection.execute(
                """
                INSERT INTO pair_screening_results
                SELECT * FROM pair_screening_results WHERE run_id = 'run'
                """
            )
    finally:
        connection.close()


def test_pair_screening_results_round_trip_with_selected_and_rejected_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "screening-roundtrip.duckdb"
    results = _screening_batch()

    assert store_pair_screening_results(
        path,
        results,
        "run-001",
        "2023-01-01",
        "2023-12-31",
        loaded_at="2024-01-15 12:00:00",
    ) == len(results)
    loaded = load_pair_screening_results(path, "run-001")

    assert loaded.columns.tolist() == list(SCREENING_RESULT_COLUMNS)
    assert loaded[["symbol_y", "symbol_x"]].apply(tuple, axis=1).tolist() == [
        ("AAA", "BBB"),
        ("AAA", "CCC"),
        ("BBB", "DDD"),
        ("CCC", "DDD"),
    ]
    selected = loaded.iloc[0]
    assert selected["run_id"] == "run-001"
    assert selected["group_name"] == "Technology"
    assert selected["observations"] == 250
    assert selected["alpha"] == pytest.approx(0.2)
    assert selected["beta"] == pytest.approx(1.1)
    assert selected["spread_standard_deviation"] == pytest.approx(0.03)
    assert selected["cointegration_statistic"] == pytest.approx(-4.2)
    assert selected["cointegration_pvalue"] == pytest.approx(0.004)
    assert selected["cointegration_critical_values"] == {
        "1%": -3.9,
        "5%": -3.3,
        "10%": -3.0,
    }
    assert selected["adf_statistic"] == pytest.approx(-4.0)
    assert selected["adf_pvalue"] == pytest.approx(0.006)
    assert bool(selected["selected"]) is True
    assert selected["rank"] == 1
    assert selected["rejection_reasons"] == ()

    rejected = loaded.loc[
        (loaded["symbol_y"] == "BBB") & (loaded["symbol_x"] == "DDD")
    ].iloc[0]
    assert bool(rejected["selected"]) is False
    assert pd.isna(rejected["rank"])
    assert rejected["rejection_reasons"] == (
        "corrected_cointegration_pvalue_above_threshold",
        "hurst_not_below_threshold",
    )
    assert loaded["formation_start"].eq(pd.Timestamp("2023-01-01")).all()
    assert loaded["formation_end"].eq(pd.Timestamp("2023-12-31")).all()
    assert loaded["loaded_at"].eq(pd.Timestamp("2024-01-15 12:00:00")).all()


def test_rejection_reasons_use_compact_ordered_json_and_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reasons.duckdb"
    reasons = ('reason with "quotes"', "second:reason", "Unicode £")
    result = _screening_result(
        selected=False,
        rank=None,
        rejection_reasons=reasons,
    )
    store_pair_screening_results(
        path, [result], "run", "2023-01-01", "2023-12-31"
    )

    connection = duckdb.connect(str(path), read_only=True)
    try:
        encoded = connection.execute(
            "SELECT rejection_reasons FROM pair_screening_results"
        ).fetchone()[0]
    finally:
        connection.close()

    assert encoded == json.dumps(list(reasons), ensure_ascii=False, separators=(",", ":"))
    assert load_pair_screening_results(path, "run").iloc[0][
        "rejection_reasons"
    ] == reasons


def test_infinite_and_unavailable_half_life_have_distinct_null_policies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "half-life-policy.duckdb"
    infinite = _screening_result(
        "AAA",
        "BBB",
        selected=False,
        rank=None,
        half_life=float("inf"),
        rejection_reasons=("half_life_not_finite_positive",),
    )
    unavailable = replace(
        infinite,
        symbol_y="CCC",
        symbol_x="DDD",
        half_life=None,
        rejection_reasons=("insufficient_observations",),
    )
    store_pair_screening_results(
        path, [infinite, unavailable], "run", "2023-01-01", "2023-12-31"
    )

    connection = duckdb.connect(str(path), read_only=True)
    try:
        stored = connection.execute(
            """
            SELECT symbol_y, half_life, half_life_was_infinite
            FROM pair_screening_results ORDER BY symbol_y
            """
        ).fetchall()
    finally:
        connection.close()
    assert stored == [("AAA", None, True), ("CCC", None, False)]

    loaded = load_pair_screening_results(path, "run").set_index("symbol_y")
    assert np.isposinf(loaded.loc["AAA", "half_life"])
    assert pd.isna(loaded.loc["CCC", "half_life"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", np.nan),
        ("alpha", np.inf),
        ("beta", -np.inf),
        ("cointegration_statistic", np.inf),
        ("cointegration_pvalue", np.nan),
        ("corrected_pvalue", -0.1),
        ("adf_pvalue", 1.1),
        ("half_life", np.nan),
        ("half_life", -np.inf),
        ("hurst", np.inf),
    ],
)
def test_invalid_screening_numbers_and_infinities_are_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    result = replace(_screening_result(), **{field: value})

    with pytest.raises((TypeError, ValueError)):
        store_pair_screening_results(
            tmp_path / "invalid-numeric.duckdb",
            [result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


@pytest.mark.parametrize("observations", [True, 1.5, "10", -1])
def test_invalid_screening_observation_counts_are_rejected(
    tmp_path: Path, observations: Any
) -> None:
    result = replace(_screening_result(), observations=observations)

    with pytest.raises((TypeError, ValueError)):
        store_pair_screening_results(
            tmp_path / "invalid-observations.duckdb",
            [result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf])
def test_invalid_cointegration_critical_values_are_rejected(
    tmp_path: Path, invalid: Any
) -> None:
    result = replace(
        _screening_result(), cointegration_critical_values={"5%": invalid}
    )

    with pytest.raises((TypeError, ValueError), match="critical"):
        store_pair_screening_results(
            tmp_path / "invalid-critical.duckdb",
            [result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


@pytest.mark.parametrize(
    "field",
    [
        "alpha",
        "beta",
        "spread_standard_deviation",
        "cointegration_statistic",
        "cointegration_pvalue",
        "corrected_pvalue",
        "adf_statistic",
        "adf_pvalue",
        "half_life",
        "hurst",
    ],
)
def test_selected_screening_results_require_complete_diagnostics(
    tmp_path: Path,
    field: str,
) -> None:
    result = replace(_screening_result(), **{field: None})

    with pytest.raises(ValueError, match="Selected pair"):
        store_pair_screening_results(
            tmp_path / "missing-selected-diagnostic.duckdb",
            [result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"observations": 0},
        {"beta": 0.0},
        {"beta": -1.0},
        {"spread_standard_deviation": 0.0},
        {"spread_standard_deviation": -0.01},
        {"half_life": float("inf")},
        {"rejection_reasons": ("impossible",)},
        {"cointegration_critical_values": {}},
        {"cointegration_critical_values": {"5%": -3.3}},
    ],
)
def test_semantically_impossible_selected_screening_rows_are_rejected(
    tmp_path: Path,
    updates: dict[str, Any],
) -> None:
    result = replace(_screening_result(), **updates)

    with pytest.raises((TypeError, ValueError)):
        store_pair_screening_results(
            tmp_path / "impossible-selected.duckdb",
            [result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


def test_rejected_screening_result_requires_rejection_reason(tmp_path: Path) -> None:
    result = replace(_screening_result(), selected=False, rank=None)

    with pytest.raises(ValueError, match="at least one rejection reason"):
        store_pair_screening_results(
            tmp_path / "reasonless-rejected.duckdb",
            [result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


def test_real_screening_output_round_trips_under_database_invariants(
    tmp_path: Path,
) -> None:
    prices, groups = make_synthetic_universe(n_days=500, seed=42)
    results = screen_pairs(prices, groups, min_observations=300)
    path = tmp_path / "real-screening-output.duckdb"

    stored = store_pair_screening_results(
        path,
        results,
        "real-run",
        prices.index[0],
        prices.index[-1],
    )
    loaded = load_pair_screening_results(path, "real-run")

    assert stored == len(results)
    assert len(loaded) == len(results)
    assert loaded["selected"].any()
    assert loaded.loc[loaded["selected"], "beta"].gt(0).all()
    assert loaded.loc[loaded["selected"], "rank"].tolist() == list(
        range(1, int(loaded["selected"].sum()) + 1)
    )
    for result in results:
        row = loaded.loc[
            (loaded["symbol_y"] == result.symbol_y)
            & (loaded["symbol_x"] == result.symbol_x)
        ].iloc[0]
        assert row["cointegration_critical_values"] == dict(
            result.cointegration_critical_values
        )


def test_screening_persistence_preserves_existing_symbol_and_group_text(
    tmp_path: Path,
) -> None:
    result = replace(
        _screening_result(),
        symbol_y=" AAA ",
        symbol_x="BBB",
        group=" Technology ",
    )
    path = tmp_path / "preserved-text.duckdb"

    store_pair_screening_results(
        path, [result], "run", "2023-01-01", "2023-12-31"
    )
    loaded = load_pair_screening_results(path, "run")

    assert loaded.iloc[0]["symbol_y"] == " AAA "
    assert loaded.iloc[0]["symbol_x"] == "BBB"
    assert loaded.iloc[0]["group_name"] == " Technology "


def test_repeated_screening_writes_update_pair_fields_without_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "screening-upsert.duckdb"
    original = _screening_result()
    store_pair_screening_results(
        path,
        [original],
        "run",
        "2023-01-01",
        "2023-12-31",
        loaded_at="2024-01-01",
    )
    replacement = replace(
        original,
        group="Updated",
        observations=300,
        alpha=0.8,
        corrected_pvalue=0.40,
        half_life=float("inf"),
        hurst=0.70,
        selected=False,
        rank=None,
        rejection_reasons=("updated_reason",),
    )
    store_pair_screening_results(
        path,
        [replacement],
        "run",
        "2023-01-01",
        "2023-12-31",
        loaded_at="2024-07-01",
    )

    loaded = load_pair_screening_results(path, "run")
    assert len(loaded) == 1
    row = loaded.iloc[0]
    assert row["formation_start"] == pd.Timestamp("2023-01-01")
    assert row["formation_end"] == pd.Timestamp("2023-12-31")
    assert row["group_name"] == "Updated"
    assert row["observations"] == 300
    assert row["alpha"] == pytest.approx(0.8)
    assert row["corrected_pvalue"] == pytest.approx(0.40)
    assert np.isposinf(row["half_life"])
    assert row["hurst"] == pytest.approx(0.70)
    assert bool(row["selected"]) is False
    assert pd.isna(row["rank"])
    assert row["rejection_reasons"] == ("updated_reason",)
    assert row["loaded_at"] == pd.Timestamp("2024-07-01")


def test_different_screening_run_ids_remain_independent(tmp_path: Path) -> None:
    path = tmp_path / "independent-runs.duckdb"
    result = _screening_result()
    store_pair_screening_results(
        path, [result], "run-a", "2023-01-01", "2023-12-31"
    )
    store_pair_screening_results(
        path,
        [replace(result, alpha=0.9)],
        "run-b",
        "2024-01-01",
        "2024-12-31",
    )

    assert load_pair_screening_results(path, "run-a").iloc[0]["alpha"] == 0.2
    assert load_pair_screening_results(path, "run-b").iloc[0]["alpha"] == 0.9


def test_duplicate_pairs_in_one_screening_batch_are_rejected(tmp_path: Path) -> None:
    result = _screening_result()

    with pytest.raises(ValueError, match="Duplicate"):
        store_pair_screening_results(
            tmp_path / "duplicate.duckdb",
            [result, result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


def test_reversed_duplicate_pairs_in_one_batch_are_rejected(tmp_path: Path) -> None:
    result = _screening_result()
    reversed_result = replace(result, symbol_y="BBB", symbol_x="AAA")

    with pytest.raises(ValueError, match="Reversed"):
        store_pair_screening_results(
            tmp_path / "reversed.duckdb",
            [result, reversed_result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


def test_noncanonical_and_identical_screening_pairs_are_rejected(
    tmp_path: Path,
) -> None:
    noncanonical = replace(_screening_result(), symbol_y="BBB", symbol_x="AAA")
    identical = replace(_screening_result(), symbol_y="AAA", symbol_x="AAA")

    with pytest.raises(ValueError, match="canonical"):
        store_pair_screening_results(
            tmp_path / "noncanonical.duckdb",
            [noncanonical],
            "run",
            "2023-01-01",
            "2023-12-31",
        )
    with pytest.raises(ValueError, match="different"):
        store_pair_screening_results(
            tmp_path / "identical.duckdb",
            [identical],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


@pytest.mark.parametrize("run_id", [None, "", "   ", 4, False])
def test_invalid_screening_run_ids_are_rejected(tmp_path: Path, run_id: Any) -> None:
    path = tmp_path / "invalid-screening-run.duckdb"
    with pytest.raises((TypeError, ValueError)):
        store_pair_screening_results(
            path,
            [_screening_result()],
            run_id,
            "2023-01-01",
            "2023-12-31",
        )
    with pytest.raises((TypeError, ValueError)):
        load_pair_screening_results(path, run_id)
    with pytest.raises((TypeError, ValueError)):
        load_selected_pairs(path, run_id)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-02", "2024-01-01"),
        ("not-a-date", "2024-01-01"),
        ("2024-01-01", "not-a-date"),
        (True, "2024-01-01"),
        ("2024-01-01", False),
        (None, "2024-01-01"),
    ],
)
def test_invalid_screening_formation_windows_are_rejected(
    tmp_path: Path, start: Any, end: Any
) -> None:
    with pytest.raises((TypeError, ValueError)):
        store_pair_screening_results(
            tmp_path / "invalid-window.duckdb",
            [_screening_result()],
            "run",
            start,
            end,
        )


def test_conflicting_formation_windows_within_new_run_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conflicting-appended-window.duckdb"
    store_pair_screening_results(
        path,
        [_screening_result()],
        "run",
        "2023-01-01",
        "2023-12-31",
        loaded_at="2024-01-01",
    )
    before = load_pair_screening_results(path, "run").copy(deep=True)
    additional_pair = _screening_result("AAA", "CCC", rank=2)

    with pytest.raises(ValueError, match="formation window"):
        store_pair_screening_results(
            path,
            [additional_pair],
            "run",
            "2024-01-01",
            "2024-12-31",
            loaded_at="2025-01-01",
        )

    pd.testing.assert_frame_equal(load_pair_screening_results(path, "run"), before)


def test_upsert_with_different_existing_formation_window_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conflicting-upsert-window.duckdb"
    baseline = _screening_result()
    store_pair_screening_results(
        path,
        [baseline],
        "run",
        "2023-01-01",
        "2023-12-31",
        loaded_at="2024-01-01",
    )
    before = load_pair_screening_results(path, "run").copy(deep=True)

    with pytest.raises(ValueError, match="formation window"):
        store_pair_screening_results(
            path,
            [replace(baseline, alpha=0.9)],
            "run",
            "2024-01-01",
            "2024-12-31",
            loaded_at="2025-01-01",
        )

    pd.testing.assert_frame_equal(load_pair_screening_results(path, "run"), before)


@pytest.mark.parametrize("rank", [None, 0, -1, 1.5, True, "1"])
def test_selected_screening_results_require_positive_integer_rank(
    tmp_path: Path, rank: Any
) -> None:
    result = replace(_screening_result(), rank=rank)

    with pytest.raises((TypeError, ValueError)):
        store_pair_screening_results(
            tmp_path / "invalid-selected-rank.duckdb",
            [result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


def test_duplicate_selected_ranks_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-selected-ranks.duckdb"
    initialise_database(path)
    results = (
        _screening_result("AAA", "BBB", rank=1),
        _screening_result("AAA", "CCC", rank=1),
    )

    with pytest.raises(ValueError, match="unique consecutive"):
        store_pair_screening_results(
            path, results, "run", "2023-01-01", "2023-12-31"
        )

    assert load_pair_screening_results(path, "run").empty


def test_gapped_selected_ranks_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gapped-selected-ranks.duckdb"
    initialise_database(path)
    results = (
        _screening_result("AAA", "BBB", rank=1),
        _screening_result("AAA", "CCC", rank=3),
    )

    with pytest.raises(ValueError, match="consecutive integers beginning at 1"):
        store_pair_screening_results(
            path, results, "run", "2023-01-01", "2023-12-31"
        )

    assert load_pair_screening_results(path, "run").empty


def test_valid_consecutive_selected_rank_sequence_passes(tmp_path: Path) -> None:
    path = tmp_path / "valid-selected-ranks.duckdb"
    results = (
        _screening_result("AAA", "CCC", rank=2),
        _screening_result("AAA", "BBB", rank=1),
    )

    assert (
        store_pair_screening_results(
            path, results, "run", "2023-01-01", "2023-12-31"
        )
        == 2
    )

    assert load_selected_pairs(path, "run")["rank"].tolist() == [1, 2]


def test_upsert_that_creates_selected_rank_collision_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "upsert-rank-collision.duckdb"
    baseline = (
        _screening_result("AAA", "BBB", rank=1),
        _screening_result("AAA", "CCC", rank=2),
    )
    store_pair_screening_results(
        path,
        baseline,
        "run",
        "2023-01-01",
        "2023-12-31",
        loaded_at="2024-01-01",
    )
    before = load_pair_screening_results(path, "run").copy(deep=True)

    with pytest.raises(ValueError, match="unique consecutive"):
        store_pair_screening_results(
            path,
            [replace(baseline[1], rank=1, alpha=0.9)],
            "run",
            "2023-01-01",
            "2023-12-31",
            loaded_at="2025-01-01",
        )

    pd.testing.assert_frame_equal(load_pair_screening_results(path, "run"), before)


def test_upsert_that_leaves_selected_rank_gap_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "upsert-rank-gap.duckdb"
    baseline = (
        _screening_result("AAA", "BBB", rank=1),
        _screening_result("AAA", "CCC", rank=2),
    )
    store_pair_screening_results(
        path,
        baseline,
        "run",
        "2023-01-01",
        "2023-12-31",
        loaded_at="2024-01-01",
    )
    before = load_pair_screening_results(path, "run").copy(deep=True)
    rejected = replace(
        baseline[0],
        selected=False,
        rank=None,
        rejection_reasons=("no_longer_selected",),
    )

    with pytest.raises(ValueError, match="consecutive integers beginning at 1"):
        store_pair_screening_results(
            path,
            [rejected],
            "run",
            "2023-01-01",
            "2023-12-31",
            loaded_at="2025-01-01",
        )

    pd.testing.assert_frame_equal(load_pair_screening_results(path, "run"), before)


def test_rejected_screening_results_require_rank_none(tmp_path: Path) -> None:
    result = _screening_result(selected=False, rank=1, rejection_reasons=("reason",))

    with pytest.raises(ValueError, match="rank None"):
        store_pair_screening_results(
            tmp_path / "invalid-rejected-rank.duckdb",
            [result],
            "run",
            "2023-01-01",
            "2023-12-31",
        )


def test_screening_inputs_and_generator_contents_are_not_mutated(
    tmp_path: Path,
) -> None:
    results = list(_screening_batch())
    before = tuple(results)

    store_pair_screening_results(
        tmp_path / "immutable-screening.duckdb",
        (result for result in results),
        "run",
        "2023-01-01",
        "2023-12-31",
    )

    assert tuple(results) == before


def test_failed_screening_batch_rolls_back_without_partial_updates() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        _replace_screening_table_with_observation_limit(connection)
        baseline = _screening_result()
        store_pair_screening_results(
            connection,
            [baseline],
            "run",
            "2023-01-01",
            "2023-12-31",
            loaded_at="2024-01-01",
        )
        updated = replace(baseline, observations=300, alpha=0.9)
        invalid_at_sql = replace(
            baseline,
            symbol_y="CCC",
            symbol_x="DDD",
            observations=600,
            selected=False,
            rank=None,
            rejection_reasons=("reason",),
        )

        with pytest.raises(duckdb.ConstraintException):
            store_pair_screening_results(
                connection,
                [updated, invalid_at_sql],
                "run",
                "2023-01-01",
                "2023-12-31",
                loaded_at="2025-01-01",
            )

        assert connection.execute(
            """
            SELECT symbol_y, symbol_x, observations, alpha, formation_start
            FROM pair_screening_results
            """
        ).fetchall() == [
            ("AAA", "BBB", 250, 0.2, datetime(2023, 1, 1).date())
        ]
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_unknown_screening_run_returns_typed_empty_frame(tmp_path: Path) -> None:
    path = tmp_path / "unknown-screening-run.duckdb"
    initialise_database(path)

    loaded = load_pair_screening_results(path, "unknown")

    assert loaded.empty
    assert loaded.columns.tolist() == list(SCREENING_RESULT_COLUMNS)
    assert loaded["observations"].dtype == "int64"
    assert loaded["rank"].dtype == "Int64"
    assert loaded["selected"].dtype == "bool"
    assert loaded["formation_start"].dtype == "datetime64[ns]"


def test_screening_load_order_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "screening-order.duckdb"
    results = (
        _screening_result(
            "CCC",
            "DDD",
            selected=False,
            rank=None,
            corrected_pvalue=None,
            rejection_reasons=("reason",),
        ),
        _screening_result("AAA", "CCC", rank=2, corrected_pvalue=0.001),
        _screening_result(
            "BBB",
            "CCC",
            selected=False,
            rank=None,
            corrected_pvalue=0.30,
            rejection_reasons=("reason",),
        ),
        _screening_result("AAA", "BBB", rank=1, corrected_pvalue=0.05),
        _screening_result(
            "AAA",
            "DDD",
            selected=False,
            rank=None,
            corrected_pvalue=0.10,
            rejection_reasons=("reason",),
        ),
    )
    store_pair_screening_results(
        path, results, "run", "2023-01-01", "2023-12-31"
    )

    loaded = load_pair_screening_results(path, "run")

    assert loaded[["symbol_y", "symbol_x"]].apply(tuple, axis=1).tolist() == [
        ("AAA", "BBB"),
        ("AAA", "CCC"),
        ("AAA", "DDD"),
        ("BBB", "CCC"),
        ("CCC", "DDD"),
    ]


def test_load_selected_pairs_supports_limit_and_maximum_rank(tmp_path: Path) -> None:
    path = tmp_path / "selected-filters.duckdb"
    store_pair_screening_results(
        path, _screening_batch(), "run", "2023-01-01", "2023-12-31"
    )

    selected = load_selected_pairs(path, "run")
    limited = load_selected_pairs(path, "run", limit=1)
    maximum_rank = load_selected_pairs(path, "run", max_rank=1)

    assert selected["rank"].tolist() == [1, 2]
    assert selected["selected"].all()
    assert limited["rank"].tolist() == [1]
    assert maximum_rank["rank"].tolist() == [1]


@pytest.mark.parametrize("field", ["limit", "max_rank"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1", np.nan])
def test_invalid_selected_pair_limits_are_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    arguments = {field: value}

    with pytest.raises((TypeError, ValueError)):
        load_selected_pairs(
            tmp_path / "invalid-limits.duckdb", "run", **arguments
        )


def test_screening_run_summaries_use_all_counts_and_selected_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "screening-summary.duckdb"
    store_pair_screening_results(
        path,
        _screening_batch(),
        "older-run",
        "2022-01-01",
        "2022-12-31",
        loaded_at="2023-01-01",
    )
    newer = (
        _screening_result(
            "AAA", "BBB", rank=1, corrected_pvalue=0.04, half_life=8.0, hurst=0.2
        ),
        _screening_result(
            "AAA",
            "CCC",
            rank=2,
            corrected_pvalue=0.06,
            half_life=12.0,
            hurst=0.4,
        ),
        _screening_result(
            "BBB",
            "CCC",
            selected=False,
            rank=None,
            corrected_pvalue=0.001,
            half_life=1.0,
            hurst=0.01,
            rejection_reasons=("reason",),
        ),
    )
    store_pair_screening_results(
        path,
        newer,
        "newer-run",
        "2023-01-01",
        "2023-12-31",
        loaded_at="2024-01-02",
    )

    summary = summarise_screening_runs(path)

    assert summary.columns.tolist() == list(SCREENING_SUMMARY_COLUMNS)
    assert summary["run_id"].tolist() == ["newer-run", "older-run"]
    newer_summary = summary.iloc[0]
    assert newer_summary["formation_start"] == pd.Timestamp("2023-01-01")
    assert newer_summary["formation_end"] == pd.Timestamp("2023-12-31")
    assert newer_summary["total_pairs"] == 3
    assert newer_summary["selected_pairs"] == 2
    assert newer_summary["selection_rate"] == pytest.approx(2 / 3)
    assert newer_summary["mean_corrected_pvalue"] == pytest.approx(0.05)
    assert newer_summary["median_half_life"] == pytest.approx(10.0)
    assert newer_summary["mean_hurst"] == pytest.approx(0.30)
    assert newer_summary["latest_loaded_at"] == pd.Timestamp("2024-01-02")

    older_summary = summary.iloc[1]
    assert older_summary["formation_start"] == pd.Timestamp("2022-01-01")
    assert older_summary["formation_end"] == pd.Timestamp("2022-12-31")
    assert older_summary["total_pairs"] == 4
    assert older_summary["selected_pairs"] == 2
    assert older_summary["selection_rate"] == pytest.approx(0.5)
    assert older_summary["mean_corrected_pvalue"] == pytest.approx(0.02)
    assert older_summary["median_half_life"] == pytest.approx(15.0)
    assert older_summary["mean_hurst"] == pytest.approx(0.30)


def test_empty_screening_summary_returns_typed_frame(tmp_path: Path) -> None:
    path = tmp_path / "empty-summary.duckdb"
    initialise_database(path)

    summary = summarise_screening_runs(path)

    assert summary.empty
    assert summary.columns.tolist() == list(SCREENING_SUMMARY_COLUMNS)
    assert summary["total_pairs"].dtype == "int64"
    assert summary["selection_rate"].dtype == "float64"
    assert summary["latest_loaded_at"].dtype == "datetime64[ns]"


def test_every_screening_analysis_query_executes_successfully() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        store_pair_screening_results(
            connection,
            _screening_batch(),
            "analysis-run",
            "2023-01-01",
            "2023-12-31",
        )
        second_run_result = replace(
            _screening_result(),
            cointegration_pvalue=0.008,
            corrected_pvalue=0.03,
            adf_pvalue=0.010,
            half_life=20.0,
            hurst=0.35,
        )
        store_pair_screening_results(
            connection,
            [second_run_result],
            "analysis-run-2",
            "2024-01-01",
            "2024-12-31",
        )
        sql = (
            resources.files("pairs_trading")
            .joinpath("sql", "screening_analysis.sql")
            .read_text(encoding="utf-8")
        )
        queries = [query.strip() for query in sql.split(";") if query.strip()]

        assert len(queries) == 5
        outputs = []
        for query in queries:
            if "$run_id" in query:
                outputs.append(
                    connection.execute(
                        query, {"run_id": "analysis-run"}
                    ).fetchdf()
                )
            else:
                outputs.append(connection.execute(query).fetchdf())

        assert not outputs[0].empty
        assert not outputs[1].empty
        assert not outputs[2].empty
        assert not outputs[3].empty
        repeated_pair = outputs[3].loc[
            (outputs[3]["symbol_y"] == "AAA")
            & (outputs[3]["symbol_x"] == "BBB")
        ].iloc[0]
        assert repeated_pair["selected_run_count"] == 2
        assert repeated_pair["mean_corrected_pvalue"] == pytest.approx(0.02)
        assert repeated_pair["mean_half_life"] == pytest.approx(15.0)
        assert set(outputs[4]["rejection_reason"]) == {
            "corrected_cointegration_pvalue_above_threshold",
            "hurst_not_below_threshold",
            "half_life_not_finite_positive",
        }
    finally:
        connection.close()


def test_screening_caller_connection_remains_open_after_success_and_failure() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        store_pair_screening_results(
            connection,
            [_screening_result()],
            "run",
            "2023-01-01",
            "2023-12-31",
        )
        assert not load_pair_screening_results(connection, "run").empty

        _replace_screening_table_with_observation_limit(connection)
        failing = replace(_screening_result(), observations=600)
        with pytest.raises(duckdb.ConstraintException):
            store_pair_screening_results(
                connection,
                [failing],
                "run",
                "2023-01-01",
                "2023-12-31",
            )
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_screening_module_owned_connection_closes_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_connect = database_module.duckdb.connect
    opened: list[duckdb.DuckDBPyConnection] = []

    def tracking_connect(*args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        connection = actual_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(database_module.duckdb, "connect", tracking_connect)
    store_pair_screening_results(
        tmp_path / "owned-screening.duckdb",
        [_screening_result()],
        "run",
        "2023-01-01",
        "2023-12-31",
    )

    assert len(opened) == 1
    _assert_connection_closed(opened[0])


def test_screening_module_owned_connection_closes_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "owned-screening-failure.duckdb"
    setup_connection = duckdb.connect(str(path))
    try:
        initialise_database(setup_connection)
        _replace_screening_table_with_observation_limit(setup_connection)
    finally:
        setup_connection.close()

    actual_connect = database_module.duckdb.connect
    opened: list[duckdb.DuckDBPyConnection] = []

    def tracking_connect(*args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        connection = actual_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(database_module.duckdb, "connect", tracking_connect)
    with pytest.raises(duckdb.ConstraintException):
        store_pair_screening_results(
            path,
            [replace(_screening_result(), observations=600)],
            "run",
            "2023-01-01",
            "2023-12-31",
        )

    assert len(opened) == 1
    _assert_connection_closed(opened[0])
