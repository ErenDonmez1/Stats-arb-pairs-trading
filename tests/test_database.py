"""Offline tests for the Milestone 4E-A DuckDB persistence boundary."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

import pairs_trading.database as database_module
from pairs_trading.database import (
    QUALITY_COLUMNS,
    connect_database,
    initialise_database,
    load_data_quality_report,
    load_prices,
    store_data_quality_report,
    store_prices,
)
from pairs_trading.data import MarketDataLoader


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
    return expected


def _assert_connection_closed(connection: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_schema_creation_is_idempotent_and_preserves_existing_data() -> None:
    connection = duckdb.connect(":memory:")
    try:
        initialise_database(connection)
        connection.execute(
            """
            INSERT INTO prices VALUES
                (DATE '2024-01-01', 'AAA', 100.0, 'test', TIMESTAMP '2024-02-01')
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
        } == {"prices", "data_quality_reports"}

        price_info = connection.execute("PRAGMA table_info('prices')").fetchall()
        quality_info = connection.execute(
            "PRAGMA table_info('data_quality_reports')"
        ).fetchall()
        assert [(row[1], row[2], bool(row[3])) for row in price_info] == [
            ("date", "DATE", True),
            ("symbol", "VARCHAR", True),
            ("adjusted_close", "DOUBLE", True),
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
            (DATE '2024-01-01', 'AAA', 1.0, 'test', TIMESTAMP '2024-01-01'),
            (DATE '2024-01-01', 'AAA', 2.0, 'test', TIMESTAMP '2024-01-02')
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

        monkeypatch.setattr(database_module, "_load_schema", lambda: "INVALID SQL")
        with pytest.raises(duckdb.ParserException):
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
            SELECT date, symbol, adjusted_close, source, loaded_at
            FROM prices ORDER BY date, symbol
            """
        ).fetchall()
        assert len(rows) == prices.size
        assert rows[0] == (
            datetime(2024, 1, 1).date(),
            "AAA",
            51.0,
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
    store_prices(path, original, source="test", loaded_at="2024-02-01")
    store_prices(path, replacement, source="test", loaded_at="2024-02-02")

    connection = duckdb.connect(str(path), read_only=True)
    try:
        assert connection.execute(
            "SELECT COUNT(*), MIN(adjusted_close), MIN(loaded_at) FROM prices"
        ).fetchone() == (1, 206.0, datetime(2024, 2, 2))
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

    result = load_prices(path, source="unknown", symbols=["BBB", "AAA"])

    assert result.empty
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.dtype == "datetime64[ns]"
    assert result.columns.tolist() == ["AAA", "BBB"]


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
                source VARCHAR NOT NULL,
                loaded_at TIMESTAMP NOT NULL,
                UNIQUE (date, symbol, source)
            )
            """
        )
        baseline = pd.DataFrame(
            {"AAA": [100.0]}, index=pd.DatetimeIndex(["2024-01-01"])
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
            "SELECT symbol, adjusted_close, loaded_at FROM prices"
        ).fetchall() == [("AAA", 100.0, datetime(2024, 2, 1))]
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
        baseline.loc[:, "coverage"] = 0.5
        store_data_quality_report(
            connection, baseline, run_id="run", loaded_at="2024-02-01"
        )
        failing = _quality_report().loc[["BBB", "AAA"]].copy(deep=True)
        failing.loc["BBB", "coverage"] = 0.7
        failing.loc["AAA", "coverage"] = 1.0

        with pytest.raises(duckdb.ConstraintException):
            store_data_quality_report(
                connection, failing, run_id="run", loaded_at="2024-02-02"
            )

        assert connection.execute(
            "SELECT symbol, coverage, loaded_at FROM data_quality_reports"
        ).fetchall() == [("BBB", 0.5, datetime(2024, 2, 1))]
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
        } == {"prices", "data_quality_reports"}
    finally:
        connection.close()
