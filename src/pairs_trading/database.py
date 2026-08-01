"""DuckDB persistence for validated market prices and data-quality reports.

The public write functions validate complete batches before opening a
transaction.  Filesystem targets are opened and closed within each operation;
caller-supplied DuckDB connections are always left open.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime
from importlib import resources
import os
from pathlib import Path
from typing import Any, TypeAlias

import duckdb
import numpy as np
import pandas as pd


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

__all__ = [
    "connect_database",
    "initialise_database",
    "store_prices",
    "load_prices",
    "store_data_quality_report",
    "load_data_quality_report",
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
          AND table_name IN ('prices', 'data_quality_reports')
        """
    ).fetchone()[0]
    if existing_count == 2:
        return
    with _transaction(connection):
        connection.execute(_load_schema())


def initialise_database(database: DatabaseTarget) -> None:
    """Create the Milestone 4E-A schema; repeated calls are harmless."""
    with _connection_scope(database) as connection:
        with _transaction(connection):
            connection.execute(_load_schema())


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
) -> list[tuple[date, str, float, str, datetime]]:
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
    records: list[tuple[date, str, float, str, datetime]] = []
    for row_position, timestamp in enumerate(database_dates):
        for column_position, symbol in enumerate(symbols):
            records.append(
                (
                    timestamp.date(),
                    symbol,
                    values[row_position, column_position],
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

    A repeated ``(date, symbol, source)`` key replaces ``adjusted_close`` and
    ``loaded_at``.  Other sources remain independent.
    """
    records = _normalise_price_frame(prices, source, loaded_at)
    statement = """
        INSERT INTO prices (date, symbol, adjusted_close, source, loaded_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (date, symbol, source) DO UPDATE SET
            adjusted_close = EXCLUDED.adjusted_close,
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
    normalised = tuple(
        sorted({_normalise_symbol(symbol, "symbols entry") for symbol in raw_symbols})
    )
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
    return frame


def load_prices(
    database: DatabaseTarget,
    source: str,
    *,
    symbols: Iterable[str] | str | None = None,
    start: Any | None = None,
    end: Any | None = None,
) -> pd.DataFrame:
    """Load one source into a date-by-symbol adjusted-close frame.

    Date filters are inclusive.  Source scoping is mandatory because combining
    providers could produce more than one value for a wide-frame cell.
    """
    normalised_source = _normalise_non_empty_string(source, "source")
    normalised_symbols = _normalise_symbol_filter(symbols)
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
        SELECT date, symbol, adjusted_close
        FROM prices
        WHERE {' AND '.join(clauses)}
        ORDER BY date ASC, symbol ASC
    """
    with _connection_scope(database) as connection:
        _ensure_schema(connection)
        result = connection.execute(query, parameters).fetchdf()

    if result.empty:
        return _empty_prices(normalised_symbols)
    result["date"] = pd.to_datetime(result["date"]).astype("datetime64[ns]")
    wide = result.pivot(index="date", columns="symbol", values="adjusted_close")
    wide = wide.sort_index().reindex(sorted(wide.columns), axis=1).astype(float)
    wide.index = pd.DatetimeIndex(wide.index, name="date")
    wide.columns.name = None
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
    with _connection_scope(database) as connection:
        _ensure_schema(connection)
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
