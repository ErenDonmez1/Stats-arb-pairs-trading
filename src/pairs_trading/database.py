"""DuckDB persistence for market data, quality, and pair-screening research.

The public write functions validate complete batches before opening a
transaction.  Filesystem targets are opened and closed within each operation;
caller-supplied DuckDB connections are always left open.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from importlib import resources
import json
import os
from pathlib import Path
from typing import Any, TypeAlias

import duckdb
import numpy as np
import pandas as pd

from .data import OBSERVED_PRICE_MASK_ATTR
from .screening import PairScreeningResult


DatabasePath: TypeAlias = str | os.PathLike[str]
DatabaseTarget: TypeAlias = DatabasePath | duckdb.DuckDBPyConnection

QUALITY_COLUMNS = (
    "total_observations",
    "valid_observations",
    "missing_or_invalid",
    "non_positive",
    "coverage",
    "stale_fraction",
    "first_valid",
    "last_valid",
    "forward_filled",
    "retained",
)
_COUNT_COLUMNS = (
    "total_observations",
    "valid_observations",
    "missing_or_invalid",
    "non_positive",
    "forward_filled",
)
SCREENING_RESULT_COLUMNS = (
    "run_id",
    "formation_start",
    "formation_end",
    "symbol_y",
    "symbol_x",
    "group_name",
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
    "loaded_at",
)
SCREENING_SUMMARY_COLUMNS = (
    "run_id",
    "formation_start",
    "formation_end",
    "total_pairs",
    "selected_pairs",
    "selection_rate",
    "mean_corrected_pvalue",
    "median_half_life",
    "mean_hurst",
    "latest_loaded_at",
)
_SCREENING_FLOAT_COLUMNS = (
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
)
_REQUIRED_SCHEMA_COLUMNS = {
    "prices": {
        "date": "DATE",
        "symbol": "VARCHAR",
        "adjusted_close": "DOUBLE",
        "observed": "BOOLEAN",
        "source": "VARCHAR",
        "loaded_at": "TIMESTAMP",
    },
    "data_quality_reports": {
        "run_id": "VARCHAR",
        "symbol": "VARCHAR",
        "total_observations": "BIGINT",
        "valid_observations": "BIGINT",
        "missing_or_invalid": "BIGINT",
        "non_positive": "BIGINT",
        "coverage": "DOUBLE",
        "stale_fraction": "DOUBLE",
        "first_valid": "TIMESTAMP",
        "last_valid": "TIMESTAMP",
        "forward_filled": "BIGINT",
        "retained": "BOOLEAN",
        "loaded_at": "TIMESTAMP",
    },
    "pair_screening_results": {
        "run_id": "VARCHAR",
        "formation_start": "DATE",
        "formation_end": "DATE",
        "symbol_y": "VARCHAR",
        "symbol_x": "VARCHAR",
        "group_name": "VARCHAR",
        "observations": "BIGINT",
        "alpha": "DOUBLE",
        "beta": "DOUBLE",
        "spread_standard_deviation": "DOUBLE",
        "cointegration_statistic": "DOUBLE",
        "cointegration_pvalue": "DOUBLE",
        "corrected_pvalue": "DOUBLE",
        "cointegration_critical_values": "VARCHAR",
        "adf_statistic": "DOUBLE",
        "adf_pvalue": "DOUBLE",
        "half_life": "DOUBLE",
        "hurst": "DOUBLE",
        "selected": "BOOLEAN",
        "rank": "BIGINT",
        "rejection_reasons": "VARCHAR",
        "loaded_at": "TIMESTAMP",
        "half_life_was_infinite": "BOOLEAN",
    },
}

__all__ = [
    "connect_database",
    "initialise_database",
    "store_prices",
    "load_prices",
    "store_data_quality_report",
    "load_data_quality_report",
    "store_pair_screening_results",
    "load_pair_screening_results",
    "load_selected_pairs",
    "summarise_screening_runs",
]


def connect_database(
    database: DatabaseTarget = ":memory:",
) -> duckdb.DuckDBPyConnection:
    """Return an open DuckDB connection for ``database``.

    Passing an existing connection returns it unchanged.  A connection returned
    directly by this function is caller-owned.  Higher-level functions close a
    connection only when they opened it from a path themselves.
    """
    if isinstance(database, duckdb.DuckDBPyConnection):
        return database
    if isinstance(database, bool) or not isinstance(database, (str, os.PathLike)):
        raise TypeError("database must be a filesystem path or DuckDB connection.")

    raw_path = os.fspath(database)
    if not isinstance(raw_path, str):
        raise TypeError("database path must resolve to a string path.")
    if not raw_path.strip():
        raise ValueError("database path must not be empty.")

    if raw_path != ":memory:":
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = str(path)
    return duckdb.connect(database=raw_path)


@contextmanager
def _connection_scope(database: DatabaseTarget) -> Iterator[duckdb.DuckDBPyConnection]:
    caller_owned = isinstance(database, duckdb.DuckDBPyConnection)
    connection = connect_database(database)
    try:
        yield connection
    finally:
        if not caller_owned:
            connection.close()


@contextmanager
def _read_connection_scope(
    database: DatabaseTarget,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open an existing database for reading without creating paths or schema."""
    if isinstance(database, duckdb.DuckDBPyConnection):
        yield database
        return
    if isinstance(database, bool) or not isinstance(database, (str, os.PathLike)):
        raise TypeError("database must be a filesystem path or DuckDB connection.")
    raw_path = os.fspath(database)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("database path must not be empty.")
    if raw_path == ":memory:":
        raise ValueError(
            "Read operations require an existing connection for an in-memory database."
        )
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Database file does not exist: {path}")
    connection = duckdb.connect(database=str(path), read_only=True)
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def _transaction(connection: duckdb.DuckDBPyConnection) -> Iterator[None]:
    """Commit one unit of work or roll it back without hiding its failure."""
    transaction_started = False
    try:
        connection.execute("BEGIN TRANSACTION")
        transaction_started = True
        yield
        connection.execute("COMMIT")
    except Exception:
        if transaction_started:
            try:
                connection.execute("ROLLBACK")
            except duckdb.Error as rollback_error:
                raise RuntimeError(
                    "Database operation failed and its transaction could not be "
                    "rolled back."
                ) from rollback_error
        raise


def _load_schema() -> str:
    """Load the packaged schema without relying on the working directory."""
    schema_resource = resources.files("pairs_trading").joinpath("sql", "schema.sql")
    try:
        return schema_resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError("Packaged DuckDB schema.sql could not be loaded.") from exc


def _ensure_schema(connection: duckdb.DuckDBPyConnection) -> None:
    existing_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_name IN (
              'prices', 'data_quality_reports', 'pair_screening_results'
          )
        """
    ).fetchone()[0]
    if existing_count == 0:
        with _transaction(connection):
            connection.execute(_load_schema())
    elif existing_count != len(_REQUIRED_SCHEMA_COLUMNS):
        raise RuntimeError(
            "Database contains a partial research schema; explicit migration or "
            "recreation is required."
        )
    _validate_schema(connection)


def _validate_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Reject missing or legacy schema shapes without inventing provenance."""
    existing_tables = {
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        ).fetchall()
    }
    missing_tables = sorted(set(_REQUIRED_SCHEMA_COLUMNS) - existing_tables)
    if missing_tables:
        raise RuntimeError(
            f"Database schema is missing required tables: {missing_tables}."
        )
    for table, required in _REQUIRED_SCHEMA_COLUMNS.items():
        actual = {
            row[1]: str(row[2]).upper()
            for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        }
        missing_columns = sorted(set(required) - set(actual))
        if missing_columns:
            raise RuntimeError(
                f"Database table {table!r} is incompatible; missing columns "
                f"{missing_columns}. Recreate or explicitly migrate the database."
            )
        incompatible = {
            column: (actual[column], expected)
            for column, expected in required.items()
            if actual[column] != expected
        }
        if incompatible:
            raise RuntimeError(
                f"Database table {table!r} has incompatible column types: "
                f"{incompatible}."
            )


def initialise_database(database: DatabaseTarget) -> None:
    """Create the current research schema; repeated calls are harmless."""
    with _connection_scope(database) as connection:
        _ensure_schema(connection)


def _normalise_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    normalised = value.strip()
    if not normalised:
        raise ValueError(f"{field} must not be empty.")
    return normalised


def _normalise_symbol(value: Any, field: str = "symbol") -> str:
    return _normalise_non_empty_string(value, field).upper()


def _is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NaT or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _normalise_timestamp(
    value: Any,
    field: str,
    *,
    allow_missing: bool = False,
) -> datetime | None:
    if _is_missing_scalar(value):
        if allow_missing:
            return None
        raise ValueError(f"{field} must not be missing.")
    if isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)):
        raise TypeError(f"{field} must be a datetime-like value.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field} must not be empty.")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a valid datetime-like value.") from exc
    if pd.isna(timestamp):
        if allow_missing:
            return None
        raise ValueError(f"{field} must not be missing.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    try:
        timestamp = timestamp.as_unit("ns")
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must fit pandas nanosecond datetime range."
        ) from exc
    if timestamp.nanosecond:
        raise ValueError(
            f"{field} must not contain sub-microsecond precision."
        )
    return timestamp.to_pydatetime(warn=False)


def _normalise_loaded_at(value: Any | None) -> datetime:
    if value is None:
        return (
            pd.Timestamp.now(tz="UTC")
            .tz_localize(None)
            .floor("us")
            .to_pydatetime(warn=False)
        )
    normalised = _normalise_timestamp(value, "loaded_at")
    assert normalised is not None
    return normalised


def _normalise_date_filter(value: Any, field: str) -> date:
    timestamp = _normalise_timestamp(value, field)
    assert timestamp is not None
    return timestamp.date()


def _strict_real(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must contain real numeric values.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must fit a finite DuckDB DOUBLE.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field} must contain finite values.")
    return number


def _strict_non_negative_integer(value: Any, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{field} must contain non-boolean integers.")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{field} must contain non-negative integers.")
    if integer > np.iinfo(np.int64).max:
        raise ValueError(f"{field} exceeds DuckDB BIGINT range.")
    return integer


def _normalise_price_frame(
    prices: pd.DataFrame,
    source: Any,
    loaded_at: Any | None,
) -> list[tuple[date, str, float, bool, str, datetime]]:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("prices must contain at least one observation and symbol.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex.")
    if prices.index.hasnans:
        raise ValueError("prices index must not contain NaT.")
    if not prices.index.is_unique:
        raise ValueError("prices index must contain unique timestamps.")
    if prices.columns.duplicated().any():
        raise ValueError("prices must contain unique symbol columns.")

    mask_value = prices.attrs.get(OBSERVED_PRICE_MASK_ATTR)
    if mask_value is None:
        if "valuation_policy" in prices.attrs:
            raise ValueError(
                "A price frame declaring a valuation policy must include an "
                "observed_price_mask."
            )
        observed_mask = pd.DataFrame(
            True,
            index=prices.index.copy(),
            columns=prices.columns.copy(),
            dtype=bool,
        )
    else:
        if not isinstance(mask_value, pd.DataFrame):
            raise TypeError("observed_price_mask must be a pandas DataFrame.")
        if not mask_value.index.equals(prices.index):
            raise ValueError("observed_price_mask index must align exactly with prices.")
        if not mask_value.columns.equals(prices.columns):
            raise ValueError(
                "observed_price_mask columns must align exactly with prices."
            )
        if bool(mask_value.isna().to_numpy(dtype=bool).any()):
            raise ValueError("observed_price_mask must not contain missing values.")
        if any(
            not isinstance(value, (bool, np.bool_))
            for value in mask_value.to_numpy(dtype=object).ravel()
        ):
            raise TypeError("observed_price_mask must contain only Boolean values.")
        observed_mask = mask_value.astype(bool).copy(deep=True)

    symbols = [
        _normalise_symbol(column, f"prices column {position}")
        for position, column in enumerate(prices.columns)
    ]
    if len(set(symbols)) != len(symbols):
        raise ValueError("prices symbols must remain unique after normalisation.")

    index = prices.index.copy()
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    try:
        index = index.as_unit("ns")
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            "prices index must fit pandas nanosecond datetime range."
        ) from exc
    database_dates = index.normalize()
    if not database_dates.is_unique:
        raise ValueError(
            "prices timestamps must map to unique UTC-normalised database dates."
        )

    values = np.empty(prices.shape, dtype=float)
    for column_position, symbol in enumerate(symbols):
        for row_position, value in enumerate(prices.iloc[:, column_position].array):
            number = _strict_real(value, f"price for {symbol}")
            if number <= 0:
                raise ValueError(f"price for {symbol} must be strictly positive.")
            values[row_position, column_position] = number

    normalised_source = _normalise_non_empty_string(source, "source")
    batch_loaded_at = _normalise_loaded_at(loaded_at)
    records: list[tuple[date, str, float, bool, str, datetime]] = []
    for row_position, timestamp in enumerate(database_dates):
        for column_position, symbol in enumerate(symbols):
            records.append(
                (
                    timestamp.date(),
                    symbol,
                    values[row_position, column_position],
                    bool(observed_mask.iloc[row_position, column_position]),
                    normalised_source,
                    batch_loaded_at,
                )
            )
    return records


def store_prices(
    database: DatabaseTarget,
    prices: pd.DataFrame,
    source: str,
    *,
    loaded_at: Any | None = None,
) -> int:
    """Upsert a complete wide price frame and return its relational row count.

    A repeated ``(date, symbol, source)`` key replaces the price, observation
    provenance, and load timestamp atomically. Frames without cleaning metadata
    are explicitly treated as raw, fully observed inputs.
    """
    records = _normalise_price_frame(prices, source, loaded_at)
    statement = """
        INSERT INTO prices (
            date, symbol, adjusted_close, observed, source, loaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (date, symbol, source) DO UPDATE SET
            adjusted_close = EXCLUDED.adjusted_close,
            observed = EXCLUDED.observed,
            loaded_at = EXCLUDED.loaded_at
    """
    with _connection_scope(database) as connection:
        _ensure_schema(connection)
        with _transaction(connection):
            connection.executemany(statement, records)
    return len(records)


def _normalise_symbol_filter(symbols: Iterable[str] | str | None) -> tuple[str, ...] | None:
    if symbols is None:
        return None
    if isinstance(symbols, str):
        raw_symbols: Iterable[Any] = (symbols,)
    else:
        if isinstance(symbols, (bytes, bytearray)):
            raise TypeError("symbols must contain strings.")
        try:
            raw_symbols = tuple(symbols)
        except TypeError as exc:
            raise TypeError("symbols must be an iterable of strings.") from exc
    normalised_values = tuple(
        _normalise_symbol(symbol, "symbols entry") for symbol in raw_symbols
    )
    if len(normalised_values) != len(set(normalised_values)):
        raise ValueError("symbols must not contain duplicates after normalisation.")
    normalised = normalised_values
    if not normalised:
        raise ValueError("symbols must not be empty when supplied.")
    return normalised


def _empty_prices(symbols: tuple[str, ...] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        index=pd.DatetimeIndex([], dtype="datetime64[ns]", name="date"),
        columns=list(symbols or ()),
        dtype=float,
    )
    frame.columns.name = None
    frame.attrs[OBSERVED_PRICE_MASK_ATTR] = pd.DataFrame(
        False,
        index=frame.index.copy(),
        columns=frame.columns.copy(),
        dtype=bool,
    )
    frame.attrs["valuation_policy"] = (
        "database_observed_provenance; execution_requires_observed"
    )
    return frame


def load_prices(
    database: DatabaseTarget,
    source: str,
    *,
    symbols: Iterable[str] | str | None = None,
    start: Any | None = None,
    end: Any | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Load one source into a date-by-symbol adjusted-close frame.

    Date filters are inclusive.  Source scoping is mandatory because combining
    providers could produce more than one value for a wide-frame cell.
    """
    normalised_source = _normalise_non_empty_string(source, "source")
    normalised_symbols = _normalise_symbol_filter(symbols)
    if not isinstance(strict, (bool, np.bool_)):
        raise TypeError("strict must be Boolean.")
    start_date = None if start is None else _normalise_date_filter(start, "start")
    end_date = None if end is None else _normalise_date_filter(end, "end")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start must be on or before end.")

    clauses = ["source = ?"]
    parameters: list[Any] = [normalised_source]
    if normalised_symbols is not None:
        placeholders = ", ".join("?" for _ in normalised_symbols)
        clauses.append(f"symbol IN ({placeholders})")
        parameters.extend(normalised_symbols)
    if start_date is not None:
        clauses.append("date >= ?")
        parameters.append(start_date)
    if end_date is not None:
        clauses.append("date <= ?")
        parameters.append(end_date)

    query = f"""
        SELECT date, symbol, adjusted_close, observed
        FROM prices
        WHERE {' AND '.join(clauses)}
        ORDER BY date ASC, symbol ASC
    """
    with _read_connection_scope(database) as connection:
        _validate_schema(connection)
        result = connection.execute(query, parameters).fetchdf()

    if result.empty:
        if normalised_symbols is not None and strict:
            raise ValueError(
                "Database contains none of the requested symbols for the query: "
                f"{list(normalised_symbols)}."
            )
        return _empty_prices(normalised_symbols)
    if normalised_symbols is not None:
        returned_symbols = set(result["symbol"].tolist())
        missing_symbols = [
            symbol for symbol in normalised_symbols if symbol not in returned_symbols
        ]
        if missing_symbols and strict:
            raise ValueError(
                f"Database query is missing requested symbols: {missing_symbols}."
            )
    result["date"] = pd.to_datetime(result["date"]).astype("datetime64[ns]")
    wide = result.pivot(index="date", columns="symbol", values="adjusted_close")
    observed = result.pivot(index="date", columns="symbol", values="observed")
    output_columns = (
        list(normalised_symbols)
        if normalised_symbols is not None
        else sorted(wide.columns)
    )
    wide = wide.sort_index().reindex(columns=output_columns).astype(float)
    observed = (
        observed.sort_index()
        .reindex(index=wide.index, columns=output_columns)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    wide.index = pd.DatetimeIndex(wide.index, name="date")
    wide.columns.name = None
    observed.index = wide.index.copy()
    observed.columns = wide.columns.copy()
    observed.columns.name = None
    wide.attrs[OBSERVED_PRICE_MASK_ATTR] = observed
    wide.attrs["valuation_policy"] = (
        "database_observed_provenance; execution_requires_observed"
    )
    return wide


def _normalise_quality_report(
    report: pd.DataFrame,
    run_id: Any,
    loaded_at: Any | None,
) -> list[tuple[Any, ...]]:
    if not isinstance(report, pd.DataFrame):
        raise TypeError("report must be a pandas DataFrame.")
    if report.empty:
        raise ValueError("report must contain at least one symbol.")
    if report.columns.duplicated().any():
        raise ValueError("report must contain unique columns.")
    missing_columns = [column for column in QUALITY_COLUMNS if column not in report]
    if missing_columns:
        raise ValueError(f"report is missing required columns: {missing_columns}.")
    if isinstance(report.index, pd.MultiIndex):
        raise TypeError("report must use a one-dimensional symbol index.")
    if not report.index.is_unique:
        raise ValueError("report symbol index must be unique.")

    symbols = [
        _normalise_symbol(symbol, f"report symbol at position {position}")
        for position, symbol in enumerate(report.index)
    ]
    if len(set(symbols)) != len(symbols):
        raise ValueError("report symbols must remain unique after normalisation.")

    normalised_run_id = _normalise_non_empty_string(run_id, "run_id")
    batch_loaded_at = _normalise_loaded_at(loaded_at)
    records: list[tuple[Any, ...]] = []
    for row_position, symbol in enumerate(symbols):
        counts = {
            column: _strict_non_negative_integer(
                report.iloc[row_position][column], f"{column} for {symbol}"
            )
            for column in _COUNT_COLUMNS
        }
        coverage = _strict_real(
            report.iloc[row_position]["coverage"], f"coverage for {symbol}"
        )
        if not 0 <= coverage <= 1:
            raise ValueError(f"coverage for {symbol} must be in [0, 1].")
        total = counts["total_observations"]
        valid = counts["valid_observations"]
        invalid = counts["missing_or_invalid"]
        if total == 0:
            raise ValueError(f"total_observations for {symbol} must be positive.")
        if valid + invalid != total:
            raise ValueError(
                f"valid_observations plus missing_or_invalid for {symbol} "
                "must equal total_observations."
            )
        if counts["non_positive"] > invalid:
            raise ValueError(
                f"non_positive for {symbol} must not exceed missing_or_invalid."
            )
        if counts["forward_filled"] > invalid:
            raise ValueError(
                f"forward_filled for {symbol} must not exceed missing_or_invalid."
            )
        expected_coverage = valid / total
        if not np.isclose(coverage, expected_coverage, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"coverage for {symbol} must equal valid_observations / "
                "total_observations."
            )

        stale_value = report.iloc[row_position]["stale_fraction"]
        if _is_missing_scalar(stale_value):
            stale_fraction = None
        else:
            stale_fraction = _strict_real(
                stale_value, f"stale_fraction for {symbol}"
            )
            if not 0 <= stale_fraction <= 1:
                raise ValueError(
                    f"stale_fraction for {symbol} must be in [0, 1]."
                )

        first_valid = _normalise_timestamp(
            report.iloc[row_position]["first_valid"],
            f"first_valid for {symbol}",
            allow_missing=True,
        )
        last_valid = _normalise_timestamp(
            report.iloc[row_position]["last_valid"],
            f"last_valid for {symbol}",
            allow_missing=True,
        )
        if first_valid is not None and last_valid is not None and first_valid > last_valid:
            raise ValueError(f"first_valid for {symbol} must not exceed last_valid.")

        retained_value = report.iloc[row_position]["retained"]
        if not isinstance(retained_value, (bool, np.bool_)):
            raise TypeError(f"retained for {symbol} must be Boolean.")

        records.append(
            (
                normalised_run_id,
                symbol,
                counts["total_observations"],
                counts["valid_observations"],
                counts["missing_or_invalid"],
                counts["non_positive"],
                coverage,
                stale_fraction,
                first_valid,
                last_valid,
                counts["forward_filled"],
                bool(retained_value),
                batch_loaded_at,
            )
        )
    return records


def store_data_quality_report(
    database: DatabaseTarget,
    report: pd.DataFrame,
    run_id: str,
    *,
    loaded_at: Any | None = None,
) -> int:
    """Upsert a quality report and return the number of submitted symbols.

    Repeated ``(run_id, symbol)`` keys replace every non-key report field and
    ``loaded_at``.
    """
    records = _normalise_quality_report(report, run_id, loaded_at)
    statement = """
        INSERT INTO data_quality_reports (
            run_id, symbol, total_observations, valid_observations,
            missing_or_invalid, non_positive, coverage, stale_fraction,
            first_valid, last_valid, forward_filled, retained, loaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, symbol) DO UPDATE SET
            total_observations = EXCLUDED.total_observations,
            valid_observations = EXCLUDED.valid_observations,
            missing_or_invalid = EXCLUDED.missing_or_invalid,
            non_positive = EXCLUDED.non_positive,
            coverage = EXCLUDED.coverage,
            stale_fraction = EXCLUDED.stale_fraction,
            first_valid = EXCLUDED.first_valid,
            last_valid = EXCLUDED.last_valid,
            forward_filled = EXCLUDED.forward_filled,
            retained = EXCLUDED.retained,
            loaded_at = EXCLUDED.loaded_at
    """
    with _connection_scope(database) as connection:
        _ensure_schema(connection)
        with _transaction(connection):
            connection.executemany(statement, records)
    return len(records)


def _empty_quality_report() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "total_observations": pd.Series(dtype="int64"),
            "valid_observations": pd.Series(dtype="int64"),
            "missing_or_invalid": pd.Series(dtype="int64"),
            "non_positive": pd.Series(dtype="int64"),
            "coverage": pd.Series(dtype="float64"),
            "stale_fraction": pd.Series(dtype="float64"),
            "first_valid": pd.Series(dtype="datetime64[ns]"),
            "last_valid": pd.Series(dtype="datetime64[ns]"),
            "forward_filled": pd.Series(dtype="int64"),
            "retained": pd.Series(dtype="bool"),
        },
        index=pd.Index([], dtype="object", name="symbol"),
    )
    return frame.loc[:, list(QUALITY_COLUMNS)]


def load_data_quality_report(
    database: DatabaseTarget,
    run_id: str,
) -> pd.DataFrame:
    """Return one run's quality report, alphabetically indexed by symbol."""
    normalised_run_id = _normalise_non_empty_string(run_id, "run_id")
    query = """
        SELECT
            symbol, total_observations, valid_observations, missing_or_invalid,
            non_positive, coverage, stale_fraction, first_valid, last_valid,
            forward_filled, retained
        FROM data_quality_reports
        WHERE run_id = ?
        ORDER BY symbol ASC
    """
    with _read_connection_scope(database) as connection:
        _validate_schema(connection)
        result = connection.execute(query, [normalised_run_id]).fetchdf()

    if result.empty:
        return _empty_quality_report()
    result = result.set_index("symbol")
    result.index.name = "symbol"
    for column in _COUNT_COLUMNS:
        result[column] = result[column].astype("int64")
    result["coverage"] = result["coverage"].astype(float)
    result["stale_fraction"] = result["stale_fraction"].astype(float)
    for column in ("first_valid", "last_valid"):
        result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    result["retained"] = result["retained"].astype(bool)
    return result.loc[:, list(QUALITY_COLUMNS)].sort_index()


def _screening_symbol(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    if not value.strip():
        raise ValueError(f"{field} must not be empty.")
    return value


def _optional_screening_real(
    value: Any,
    field: str,
    *,
    allow_positive_infinity: bool = False,
) -> tuple[float | None, bool]:
    """Return a nullable finite value and whether +inf was mapped to NULL."""
    if value is None:
        return None, False
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be a real number or None.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must fit a DuckDB DOUBLE.") from exc
    if np.isnan(number):
        raise ValueError(f"{field} must not be NaN.")
    if np.isinf(number):
        if allow_positive_infinity and number > 0:
            return None, True
        raise ValueError(f"{field} contains an unsupported infinity.")
    return number, False


def _strict_positive_integer(value: Any, field: str) -> int:
    integer = _strict_non_negative_integer(value, field)
    if integer == 0:
        raise ValueError(f"{field} must be a positive integer.")
    return integer


def _normalise_critical_values(
    result: PairScreeningResult,
    *,
    selected: bool,
) -> str:
    critical_values = result.cointegration_critical_values
    if not isinstance(critical_values, Mapping):
        raise TypeError("cointegration_critical_values must be a mapping.")
    normalised: dict[str, float] = {}
    for label, value in critical_values.items():
        normalised_label = _normalise_non_empty_string(
            label, "cointegration critical-value label"
        )
        critical_value, _ = _optional_screening_real(
            value, f"cointegration critical value {label!r}"
        )
        if critical_value is None:
            raise TypeError("Cointegration critical values must not be nullable.")
        normalised[normalised_label] = critical_value
    if selected and set(normalised) != {"1%", "5%", "10%"}:
        raise ValueError(
            "Selected pairs require 1%, 5%, and 10% cointegration critical values."
        )
    return json.dumps(normalised, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_critical_values(encoded: Any) -> dict[str, float]:
    if not isinstance(encoded, str):
        raise ValueError("Stored cointegration_critical_values must contain JSON text.")
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Stored cointegration_critical_values contains invalid JSON."
        ) from exc
    if not isinstance(decoded, dict):
        raise ValueError("Stored cointegration_critical_values must be a JSON object.")
    result: dict[str, float] = {}
    for label, value in decoded.items():
        if not isinstance(label, str) or isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise ValueError("Stored cointegration critical values are malformed.")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Stored cointegration critical values must be finite.")
        result[label] = number
    return result


def _encode_rejection_reasons(reasons: Any) -> str:
    if not isinstance(reasons, tuple):
        raise TypeError("rejection_reasons must be a tuple of strings.")
    for position, reason in enumerate(reasons):
        if not isinstance(reason, str):
            raise TypeError(
                f"rejection reason at position {position} must be a string."
            )
        if not reason.strip():
            raise ValueError(
                f"rejection reason at position {position} must not be empty."
            )
    return json.dumps(list(reasons), ensure_ascii=False, separators=(",", ":"))


def _decode_rejection_reasons(encoded: Any) -> tuple[str, ...]:
    if not isinstance(encoded, str):
        raise ValueError("Stored rejection_reasons must contain JSON text.")
    try:
        decoded = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("Stored rejection_reasons contains invalid JSON.") from exc
    if not isinstance(decoded, list) or any(
        not isinstance(reason, str) for reason in decoded
    ):
        raise ValueError("Stored rejection_reasons must be a JSON string array.")
    return tuple(decoded)


def _normalise_pair_screening_results(
    results: Iterable[PairScreeningResult],
    run_id: Any,
    formation_start: Any,
    formation_end: Any,
    loaded_at: Any | None,
) -> list[tuple[Any, ...]]:
    if isinstance(results, (str, bytes, bytearray)):
        raise TypeError("results must be an iterable of PairScreeningResult objects.")
    try:
        batch = tuple(results)
    except TypeError as exc:
        raise TypeError(
            "results must be an iterable of PairScreeningResult objects."
        ) from exc
    if not batch:
        raise ValueError("results must contain at least one screening result.")

    normalised_run_id = _normalise_non_empty_string(run_id, "run_id")
    start_date = _normalise_date_filter(formation_start, "formation_start")
    end_date = _normalise_date_filter(formation_end, "formation_end")
    if start_date > end_date:
        raise ValueError("formation_start must be on or before formation_end.")
    batch_loaded_at = _normalise_loaded_at(loaded_at)

    records: list[tuple[Any, ...]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for position, result in enumerate(batch):
        if not isinstance(result, PairScreeningResult):
            raise TypeError(
                f"result at position {position} must be a PairScreeningResult."
            )
        symbol_y = _screening_symbol(result.symbol_y, "symbol_y")
        symbol_x = _screening_symbol(result.symbol_x, "symbol_x")
        pair = (symbol_y, symbol_x)
        reverse_pair = (symbol_x, symbol_y)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate pair in screening batch: {pair}.")
        if reverse_pair in seen_pairs:
            raise ValueError(f"Reversed duplicate pair in screening batch: {pair}.")
        if symbol_y == symbol_x:
            raise ValueError("symbol_y and symbol_x must identify different symbols.")
        if symbol_y > symbol_x:
            raise ValueError(
                "Pair orientation must be canonical: symbol_y must precede symbol_x."
            )
        seen_pairs.add(pair)

        if result.group is None:
            group_name = None
        else:
            if not isinstance(result.group, str):
                raise TypeError("group must be a string or None.")
            if not result.group.strip():
                raise ValueError("group must not be empty.")
            group_name = result.group

        observations = _strict_non_negative_integer(
            result.observations, f"observations for {symbol_y}/{symbol_x}"
        )
        selected_value = result.selected
        if not isinstance(selected_value, (bool, np.bool_)):
            raise TypeError(f"selected for {symbol_y}/{symbol_x} must be Boolean.")
        selected = bool(selected_value)
        if selected:
            rank = _strict_positive_integer(
                result.rank, f"rank for selected pair {symbol_y}/{symbol_x}"
            )
        else:
            if result.rank is not None:
                raise ValueError(
                    f"Rejected pair {symbol_y}/{symbol_x} must have rank None."
                )
            rank = None

        numeric_values: dict[str, float | None] = {}
        for field in (
            "alpha",
            "beta",
            "spread_standard_deviation",
            "cointegration_statistic",
            "cointegration_pvalue",
            "corrected_pvalue",
            "adf_statistic",
            "adf_pvalue",
            "hurst",
        ):
            numeric_values[field], _ = _optional_screening_real(
                getattr(result, field), f"{field} for {symbol_y}/{symbol_x}"
            )
        spread_standard_deviation = numeric_values["spread_standard_deviation"]
        if spread_standard_deviation is not None and spread_standard_deviation < 0:
            raise ValueError("spread_standard_deviation must be non-negative.")
        for field in ("cointegration_pvalue", "corrected_pvalue", "adf_pvalue"):
            probability = numeric_values[field]
            if probability is not None and not 0 <= probability <= 1:
                raise ValueError(f"{field} must be in [0, 1].")

        half_life, half_life_was_infinite = _optional_screening_real(
            result.half_life,
            f"half_life for {symbol_y}/{symbol_x}",
            allow_positive_infinity=True,
        )
        if half_life is not None and half_life <= 0:
            raise ValueError("Finite half_life must be positive.")
        encoded_reasons = _encode_rejection_reasons(result.rejection_reasons)
        encoded_critical_values = _normalise_critical_values(
            result,
            selected=selected,
        )
        if selected:
            required_selected = (
                "alpha",
                "beta",
                "spread_standard_deviation",
                "cointegration_statistic",
                "cointegration_pvalue",
                "corrected_pvalue",
                "adf_statistic",
                "adf_pvalue",
                "hurst",
            )
            missing_selected = [
                field for field in required_selected if numeric_values[field] is None
            ]
            if missing_selected:
                raise ValueError(
                    f"Selected pair {symbol_y}/{symbol_x} is missing required "
                    f"diagnostics: {missing_selected}."
                )
            if observations <= 0:
                raise ValueError("Selected pairs require positive observations.")
            if numeric_values["beta"] <= 0:  # type: ignore[operator]
                raise ValueError("Selected pairs require finite positive beta.")
            if spread_standard_deviation is None or spread_standard_deviation <= 0:
                raise ValueError(
                    "Selected pairs require positive spread_standard_deviation."
                )
            if half_life is None or half_life_was_infinite:
                raise ValueError("Selected pairs require finite positive half_life.")
            if result.rejection_reasons:
                raise ValueError("Selected pairs must not contain rejection reasons.")
        elif not result.rejection_reasons:
            raise ValueError("Rejected pairs must contain at least one rejection reason.")

        records.append(
            (
                normalised_run_id,
                start_date,
                end_date,
                symbol_y,
                symbol_x,
                group_name,
                observations,
                numeric_values["alpha"],
                numeric_values["beta"],
                spread_standard_deviation,
                numeric_values["cointegration_statistic"],
                numeric_values["cointegration_pvalue"],
                numeric_values["corrected_pvalue"],
                encoded_critical_values,
                numeric_values["adf_statistic"],
                numeric_values["adf_pvalue"],
                half_life,
                half_life_was_infinite,
                numeric_values["hurst"],
                selected,
                rank,
                encoded_reasons,
                batch_loaded_at,
            )
        )
    return records


def _validate_screening_upsert_consistency(
    connection: duckdb.DuckDBPyConnection,
    records: list[tuple[Any, ...]],
) -> None:
    """Validate run-level invariants against the complete post-upsert state.

    ``records`` is non-empty and already fully normalised.  Existing rows are
    overlaid in memory so a consistency error is detected before ``executemany``
    can change any persisted row.
    """
    run_id, formation_start, formation_end = records[0][:3]
    existing_rows = connection.execute(
        """
        SELECT
            formation_start,
            formation_end,
            symbol_y,
            symbol_x,
            selected,
            rank
        FROM pair_screening_results
        WHERE run_id = ?
        """,
        [run_id],
    ).fetchall()

    existing_windows = {(row[0], row[1]) for row in existing_rows}
    if len(existing_windows) > 1:
        raise ValueError(
            f"Stored screening run {run_id!r} has conflicting formation windows."
        )
    requested_window = (formation_start, formation_end)
    if existing_windows and existing_windows != {requested_window}:
        raise ValueError(
            f"Existing screening run {run_id!r} has formation window "
            f"{next(iter(existing_windows))!r}; incoming records must use it."
        )

    resulting_pairs: dict[tuple[str, str], tuple[bool, Any]] = {
        (symbol_y, symbol_x): (bool(selected), rank)
        for _, _, symbol_y, symbol_x, selected, rank in existing_rows
    }
    for record in records:
        resulting_pairs[(record[3], record[4])] = (record[19], record[20])

    selected_ranks: list[int] = []
    for pair, (selected, rank) in resulting_pairs.items():
        if selected:
            selected_ranks.append(
                _strict_positive_integer(
                    rank,
                    f"stored rank for selected pair {pair[0]}/{pair[1]}",
                )
            )
        elif rank is not None:
            raise ValueError(
                f"Stored rejected pair {pair[0]}/{pair[1]} must have rank None."
            )

    expected_ranks = list(range(1, len(selected_ranks) + 1))
    if sorted(selected_ranks) != expected_ranks:
        raise ValueError(
            f"Selected ranks for screening run {run_id!r} must be unique "
            "consecutive integers beginning at 1."
        )


def store_pair_screening_results(
    database: DatabaseTarget,
    results: Iterable[PairScreeningResult],
    run_id: str,
    formation_start: Any,
    formation_end: Any,
    *,
    loaded_at: Any | None = None,
) -> int:
    """Transactionally upsert one screening batch and return its row count.

    Positive-infinite half-life represents no estimated mean reversion.  It is
    stored as SQL NULL plus an internal provenance flag so that loaders restore
    only those values to Python positive infinity; ordinary nullable half-life
    values remain missing.
    """
    records = _normalise_pair_screening_results(
        results, run_id, formation_start, formation_end, loaded_at
    )
    statement = """
        INSERT INTO pair_screening_results (
            run_id, formation_start, formation_end, symbol_y, symbol_x,
            group_name, observations, alpha, beta,
            spread_standard_deviation, cointegration_statistic,
            cointegration_pvalue, corrected_pvalue,
            cointegration_critical_values, adf_statistic, adf_pvalue,
            half_life, half_life_was_infinite, hurst, selected, rank,
            rejection_reasons, loaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, symbol_y, symbol_x) DO UPDATE SET
            formation_start = EXCLUDED.formation_start,
            formation_end = EXCLUDED.formation_end,
            group_name = EXCLUDED.group_name,
            observations = EXCLUDED.observations,
            alpha = EXCLUDED.alpha,
            beta = EXCLUDED.beta,
            spread_standard_deviation = EXCLUDED.spread_standard_deviation,
            cointegration_statistic = EXCLUDED.cointegration_statistic,
            cointegration_pvalue = EXCLUDED.cointegration_pvalue,
            corrected_pvalue = EXCLUDED.corrected_pvalue,
            cointegration_critical_values = EXCLUDED.cointegration_critical_values,
            adf_statistic = EXCLUDED.adf_statistic,
            adf_pvalue = EXCLUDED.adf_pvalue,
            half_life = EXCLUDED.half_life,
            half_life_was_infinite = EXCLUDED.half_life_was_infinite,
            hurst = EXCLUDED.hurst,
            selected = EXCLUDED.selected,
            rank = EXCLUDED.rank,
            rejection_reasons = EXCLUDED.rejection_reasons,
            loaded_at = EXCLUDED.loaded_at
    """
    with _connection_scope(database) as connection:
        _ensure_schema(connection)
        with _transaction(connection):
            _validate_screening_upsert_consistency(connection, records)
            connection.executemany(statement, records)
    return len(records)


def _empty_screening_results() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "run_id": pd.Series(dtype="object"),
            "formation_start": pd.Series(dtype="datetime64[ns]"),
            "formation_end": pd.Series(dtype="datetime64[ns]"),
            "symbol_y": pd.Series(dtype="object"),
            "symbol_x": pd.Series(dtype="object"),
            "group_name": pd.Series(dtype="object"),
            "observations": pd.Series(dtype="int64"),
            "alpha": pd.Series(dtype="float64"),
            "beta": pd.Series(dtype="float64"),
            "spread_standard_deviation": pd.Series(dtype="float64"),
            "cointegration_statistic": pd.Series(dtype="float64"),
            "cointegration_pvalue": pd.Series(dtype="float64"),
            "corrected_pvalue": pd.Series(dtype="float64"),
            "cointegration_critical_values": pd.Series(dtype="object"),
            "adf_statistic": pd.Series(dtype="float64"),
            "adf_pvalue": pd.Series(dtype="float64"),
            "half_life": pd.Series(dtype="float64"),
            "hurst": pd.Series(dtype="float64"),
            "selected": pd.Series(dtype="bool"),
            "rank": pd.Series(dtype="Int64"),
            "rejection_reasons": pd.Series(dtype="object"),
            "loaded_at": pd.Series(dtype="datetime64[ns]"),
        }
    )
    return frame.loc[:, list(SCREENING_RESULT_COLUMNS)]


def _coerce_screening_results(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return _empty_screening_results()
    infinite_flags = result.pop("half_life_was_infinite").astype(bool)
    result["half_life"] = result["half_life"].astype(float)
    for row_position, was_infinite in enumerate(infinite_flags):
        half_life = result.iloc[row_position]["half_life"]
        if was_infinite:
            if not pd.isna(half_life):
                raise ValueError(
                    "Stored half-life infinity marker conflicts with a finite value."
                )
            result.iat[
                row_position, result.columns.get_loc("half_life")
            ] = float("inf")

    result["rejection_reasons"] = result["rejection_reasons"].map(
        _decode_rejection_reasons
    )
    result["cointegration_critical_values"] = result[
        "cointegration_critical_values"
    ].map(_decode_critical_values)
    for column in ("formation_start", "formation_end", "loaded_at"):
        result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    result["observations"] = result["observations"].astype("int64")
    result["selected"] = result["selected"].astype(bool)
    result["rank"] = result["rank"].astype("Int64")
    for column in _SCREENING_FLOAT_COLUMNS:
        result[column] = result[column].astype(float)
    return result.loc[:, list(SCREENING_RESULT_COLUMNS)]


def _load_screening_results(
    database: DatabaseTarget,
    run_id: Any,
    *,
    selected_only: bool,
    limit: int | None = None,
    max_rank: int | None = None,
) -> pd.DataFrame:
    normalised_run_id = _normalise_non_empty_string(run_id, "run_id")
    clauses = ["run_id = ?"]
    parameters: list[Any] = [normalised_run_id]
    if selected_only:
        clauses.append("selected")
    if max_rank is not None:
        clauses.append("rank <= ?")
        parameters.append(max_rank)
    query = f"""
        SELECT
            run_id, formation_start, formation_end, symbol_y, symbol_x,
            group_name, observations, alpha, beta,
            spread_standard_deviation, cointegration_statistic,
            cointegration_pvalue, corrected_pvalue,
            cointegration_critical_values, adf_statistic, adf_pvalue,
            half_life, half_life_was_infinite, hurst, selected, rank,
            rejection_reasons, loaded_at
        FROM pair_screening_results
        WHERE {' AND '.join(clauses)}
        ORDER BY
            selected DESC,
            CASE WHEN selected THEN rank END ASC NULLS LAST,
            corrected_pvalue ASC NULLS LAST,
            symbol_y ASC,
            symbol_x ASC
    """
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    with _read_connection_scope(database) as connection:
        _validate_schema(connection)
        result = connection.execute(query, parameters).fetchdf()
    return _coerce_screening_results(result)


def load_pair_screening_results(
    database: DatabaseTarget,
    run_id: str,
) -> pd.DataFrame:
    """Load selected and rejected results for one screening run."""
    return _load_screening_results(database, run_id, selected_only=False)


def load_selected_pairs(
    database: DatabaseTarget,
    run_id: str,
    *,
    limit: int | None = None,
    max_rank: int | None = None,
) -> pd.DataFrame:
    """Load selected pairs in stored rank order without renumbering them."""
    if limit is not None:
        limit = _strict_positive_integer(limit, "limit")
    if max_rank is not None:
        max_rank = _strict_positive_integer(max_rank, "max_rank")
    return _load_screening_results(
        database,
        run_id,
        selected_only=True,
        limit=limit,
        max_rank=max_rank,
    )


def _empty_screening_summary() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "run_id": pd.Series(dtype="object"),
            "formation_start": pd.Series(dtype="datetime64[ns]"),
            "formation_end": pd.Series(dtype="datetime64[ns]"),
            "total_pairs": pd.Series(dtype="int64"),
            "selected_pairs": pd.Series(dtype="int64"),
            "selection_rate": pd.Series(dtype="float64"),
            "mean_corrected_pvalue": pd.Series(dtype="float64"),
            "median_half_life": pd.Series(dtype="float64"),
            "mean_hurst": pd.Series(dtype="float64"),
            "latest_loaded_at": pd.Series(dtype="datetime64[ns]"),
        }
    )
    return frame.loc[:, list(SCREENING_SUMMARY_COLUMNS)]


def summarise_screening_runs(database: DatabaseTarget) -> pd.DataFrame:
    """Aggregate screening counts and selected-only diagnostics by run."""
    query = """
        SELECT
            run_id,
            formation_start,
            formation_end,
            COUNT(*) AS total_pairs,
            COUNT(*) FILTER (WHERE selected) AS selected_pairs,
            CAST(COUNT(*) FILTER (WHERE selected) AS DOUBLE)
                / COUNT(*) AS selection_rate,
            AVG(corrected_pvalue) FILTER (WHERE selected)
                AS mean_corrected_pvalue,
            MEDIAN(half_life) FILTER (WHERE selected) AS median_half_life,
            AVG(hurst) FILTER (WHERE selected) AS mean_hurst,
            MAX(loaded_at) AS latest_loaded_at
        FROM pair_screening_results
        GROUP BY run_id, formation_start, formation_end
        ORDER BY
            formation_end DESC,
            formation_start DESC,
            run_id ASC
    """
    with _read_connection_scope(database) as connection:
        _validate_schema(connection)
        result = connection.execute(query).fetchdf()
    if result.empty:
        return _empty_screening_summary()
    conflicting_runs = result.loc[
        result["run_id"].duplicated(keep=False), "run_id"
    ].drop_duplicates()
    if not conflicting_runs.empty:
        run_ids = ", ".join(repr(value) for value in conflicting_runs.tolist())
        raise ValueError(
            "Stored screening runs have conflicting formation windows: "
            f"{run_ids}."
        )
    for column in ("formation_start", "formation_end", "latest_loaded_at"):
        result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    for column in ("total_pairs", "selected_pairs"):
        result[column] = result[column].astype("int64")
    for column in (
        "selection_rate",
        "mean_corrected_pvalue",
        "median_half_life",
        "mean_hurst",
    ):
        result[column] = result[column].astype(float)
    return result.loc[:, list(SCREENING_SUMMARY_COLUMNS)]
