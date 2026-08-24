"""DuckDB persistence for market data, quality, and pair-screening research.

The public write functions validate complete batches before opening a
transaction.  Filesystem targets are opened and closed within each operation;
caller-supplied DuckDB connections are always left open.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib import resources
import json
import os
from pathlib import Path
from types import MappingProxyType
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
EXPERIMENT_LIST_COLUMNS = (
    "run_id",
    "research_content_digest",
    "experiment_name",
    "created_at",
    "pipeline_status",
    "selected_pair_id",
    "diagnostic_in_sample_total_return",
    "diagnostic_in_sample_sharpe_ratio",
    "diagnostic_in_sample_maximum_drawdown",
    "diagnostic_in_sample_trade_count",
    "walk_forward_calendar_oos_total_return",
    "walk_forward_calendar_oos_sharpe_ratio",
    "walk_forward_calendar_oos_maximum_drawdown",
    "walk_forward_stage",
    "robustness_stage",
    "validation_stage",
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
    "research_experiments": {
        "run_id": "VARCHAR",
        "research_content_digest": "VARCHAR",
        "price_content_digest": "VARCHAR",
        "configuration_digest": "VARCHAR",
        "research_pipeline_version": "VARCHAR",
        "configuration_snapshot_version": "BIGINT",
        "experiment_schema_version": "BIGINT",
        "experiment_name": "VARCHAR",
        "created_at": "TIMESTAMP",
        "pipeline_status": "VARCHAR",
        "configuration_snapshot": "VARCHAR",
        "metadata": "VARCHAR",
        "provenance": "VARCHAR",
        "warnings": "VARCHAR",
    },
    "research_experiment_summaries": {
        "run_id": "VARCHAR",
        "selected_pair_id": "VARCHAR",
        "symbol_y": "VARCHAR",
        "symbol_x": "VARCHAR",
        "selected_rank": "BIGINT",
        "screening_candidate_count": "BIGINT",
        "screening_selected_count": "BIGINT",
        "screening_selection_scope": "VARCHAR",
        "screening_ranking_policy": "VARCHAR",
        "alpha": "DOUBLE",
        "beta": "DOUBLE",
        "cointegration_statistic": "DOUBLE",
        "cointegration_pvalue": "DOUBLE",
        "corrected_pvalue": "DOUBLE",
        "half_life": "DOUBLE",
        "hurst": "DOUBLE",
        "analytics_stage": "VARCHAR",
        "diagnostic_in_sample_total_return": "DOUBLE",
        "diagnostic_in_sample_annualized_return": "DOUBLE",
        "diagnostic_in_sample_annualized_volatility": "DOUBLE",
        "diagnostic_in_sample_sharpe_ratio": "DOUBLE",
        "diagnostic_in_sample_sortino_ratio": "DOUBLE",
        "diagnostic_in_sample_maximum_drawdown": "DOUBLE",
        "diagnostic_in_sample_calmar_ratio": "DOUBLE",
        "diagnostic_in_sample_trade_count": "BIGINT",
        "diagnostic_total_rows": "BIGINT",
        "diagnostic_finite_beta_rows": "BIGINT",
        "diagnostic_positive_execution_beta_rows": "BIGINT",
        "diagnostic_non_positive_execution_beta_rows": "BIGINT",
        "diagnostic_finite_signal_rows": "BIGINT",
        "diagnostic_entry_execution_unavailable_rows_due_to_beta": "BIGINT",
        "diagnostic_signal_observation_coverage": "DOUBLE",
        "diagnostic_beta_execution_policy": "VARCHAR",
        "walk_forward_stage": "VARCHAR",
        "walk_forward_calendar_analytics_status": "VARCHAR",
        "walk_forward_fold_count": "BIGINT",
        "walk_forward_completed_fold_count": "BIGINT",
        "walk_forward_no_selection_fold_count": "BIGINT",
        "walk_forward_insufficient_data_fold_count": "BIGINT",
        "walk_forward_scheduled_oos_observations": "BIGINT",
        "walk_forward_scheduled_eligible_oos_observations": "BIGINT",
        "walk_forward_selected_oos_observations": "BIGINT",
        "walk_forward_no_selection_oos_observations": "BIGINT",
        "walk_forward_unavailable_oos_observations": "BIGINT",
        "walk_forward_selection_coverage": "DOUBLE",
        "walk_forward_evaluated_start_label": "VARCHAR",
        "walk_forward_evaluated_end_label": "VARCHAR",
        "walk_forward_capital_policy": "VARCHAR",
        "walk_forward_aggregate_return_policy": "VARCHAR",
        "walk_forward_inactive_capital_policy": "VARCHAR",
        "walk_forward_selection_coverage_denominator": "VARCHAR",
        "walk_forward_aggregate_dollar_pnl_available": "BOOLEAN",
        "walk_forward_aggregate_trade_dollar_metrics_available": "BOOLEAN",
        "walk_forward_calendar_oos_total_return": "DOUBLE",
        "walk_forward_calendar_oos_annualized_return": "DOUBLE",
        "walk_forward_calendar_oos_annualized_volatility": "DOUBLE",
        "walk_forward_calendar_oos_sharpe_ratio": "DOUBLE",
        "walk_forward_calendar_oos_sortino_ratio": "DOUBLE",
        "walk_forward_calendar_oos_maximum_drawdown": "DOUBLE",
        "walk_forward_calendar_oos_calmar_ratio": "DOUBLE",
        "walk_forward_calendar_oos_report_observations": "BIGINT",
        "robustness_stage": "VARCHAR",
        "robustness_baseline_scenario_id": "VARCHAR",
        "robustness_scenario_count": "BIGINT",
        "robustness_completed_scenarios": "BIGINT",
        "robustness_analytically_unavailable_scenarios": "BIGINT",
        "robustness_invalid_scenarios": "BIGINT",
        "robustness_failed_scenarios": "BIGINT",
        "robustness_common_horizon_structurally_available": "BOOLEAN",
        "robustness_common_horizon_fully_observed": "BOOLEAN",
        "robustness_common_horizon_analytics_available": "BOOLEAN",
        "robustness_common_horizon_analytics_status": "VARCHAR",
        "robustness_common_horizon_observations": "BIGINT",
        "robustness_common_horizon_eligible_scenario_count": "BIGINT",
        "robustness_headline_metric_basis": "VARCHAR",
        "robustness_distribution_policy": "VARCHAR",
        "robustness_tested_dimensions": "VARCHAR",
        "robustness_untested_material_dimensions": "VARCHAR",
        "robustness_provenance_warning_summary": "VARCHAR",
        "validation_stage": "VARCHAR",
        "validation_primary_availability": "VARCHAR",
        "validation_overall_availability": "VARCHAR",
        "probabilistic_sharpe_probability": "DOUBLE",
        "minimum_track_record_observations": "BIGINT",
        "fold_consistency_availability": "VARCHAR",
        "multiple_testing_total_configurations": "BIGINT",
        "multiple_testing_valid_comparable_configurations": "BIGINT",
        "multiple_testing_valid_pvalue_count": "BIGINT",
        "multiple_testing_unavailable_pvalue_count": "BIGINT",
        "multiple_testing_eligible_hypothesis_count": "BIGINT",
        "validation_purpose": "VARCHAR",
        "validation_provenance_warning_summary": "VARCHAR",
    },
}

_BASE_SCHEMA_TABLES = frozenset(
    {"prices", "data_quality_reports", "pair_screening_results"}
)
_EXPERIMENT_NOT_NULL_COLUMNS = {
    "research_experiments": frozenset(_REQUIRED_SCHEMA_COLUMNS["research_experiments"]),
    "research_experiment_summaries": frozenset(
        {
            "run_id",
            "screening_candidate_count",
            "screening_selected_count",
            "screening_selection_scope",
            "screening_ranking_policy",
            "analytics_stage",
            "walk_forward_stage",
            "robustness_stage",
            "validation_stage",
        }
    ),
}
_EXPERIMENT_PRIMARY_KEYS = {
    "research_experiments": ("run_id",),
    "research_experiment_summaries": ("run_id",),
}
_EXPERIMENT_CHECK_TOKEN_SETS = {
    "research_experiments": (
        frozenset({"research_content_digest", "price_content_digest", "length"}),
        frozenset({"configuration_snapshot_version", "experiment_schema_version"}),
        frozenset({"pipeline_status", "completed"}),
    ),
    "research_experiment_summaries": (
        frozenset({"selected_pair_id", "selected_rank"}),
        frozenset({"screening_candidate_count", "screening_selected_count"}),
        frozenset({"analytics_stage", "walk_forward_stage", "robustness_stage"}),
        frozenset(
            {
                "walk_forward_fold_count",
                "walk_forward_completed_fold_count",
                "walk_forward_scheduled_oos_observations",
            }
        ),
        frozenset({"robustness_scenario_count", "robustness_completed_scenarios"}),
        frozenset(
            {
                "multiple_testing_total_configurations",
                "multiple_testing_valid_pvalue_count",
            }
        ),
    ),
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
    "ResearchExperimentSummary",
    "EXPERIMENT_LIST_COLUMNS",
    "store_research_experiment",
    "list_research_experiments",
    "load_research_experiment_summary",
]


def _freeze_summary_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_summary_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_summary_value(item) for item in value)
    return value


def _summary_api_value(value: Any, field: str = "summary") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, datetime):
        moment = value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{field} contains NaN or infinity.")
        return number
    if isinstance(value, Mapping):
        return {
            str(key): _summary_api_value(item, f"{field}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_summary_api_value(item, f"{field}[]") for item in value]
    raise TypeError(f"{field} contains a non-JSON-compatible value.")


@dataclass(frozen=True)
class ResearchExperimentSummary:
    """Immutable persisted experiment summary for API/UI retrieval."""

    run_id: str
    research_content_digest: str
    price_content_digest: str
    research_pipeline_version: str
    configuration_snapshot_version: int
    experiment_schema_version: int
    experiment_name: str
    created_at: datetime
    configuration_digest: str
    pipeline_status: str
    selected_pair: Mapping[str, Any] | None
    screening: Mapping[str, Any]
    diagnostic: Mapping[str, Any]
    walk_forward: Mapping[str, Any]
    robustness: Mapping[str, Any]
    validation: Mapping[str, Any]
    configuration_snapshot: Mapping[str, Any]
    metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "screening",
            "diagnostic",
            "walk_forward",
            "robustness",
            "validation",
            "configuration_snapshot",
            "metadata",
            "provenance",
        ):
            object.__setattr__(
                self, name, _freeze_summary_value(getattr(self, name))
            )
        if self.selected_pair is not None:
            object.__setattr__(
                self, "selected_pair", _freeze_summary_value(self.selected_pair)
            )
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def experiment_id(self) -> str:
        """Backward-compatible read alias for ``run_id``."""
        return self.run_id

    @property
    def performance(self) -> Mapping[str, Any]:
        """Backward-compatible read alias for explicit diagnostic metrics."""
        return self.diagnostic

    def to_dict(self) -> dict[str, Any]:
        """Return a normal, strict JSON-compatible API representation."""
        return _summary_api_value(
            {
                "run_id": self.run_id,
                "research_content_digest": self.research_content_digest,
                "price_content_digest": self.price_content_digest,
                "research_pipeline_version": self.research_pipeline_version,
                "configuration_snapshot_version": self.configuration_snapshot_version,
                "experiment_schema_version": self.experiment_schema_version,
                "experiment_name": self.experiment_name,
                "created_at": self.created_at,
                "configuration_digest": self.configuration_digest,
                "pipeline_status": self.pipeline_status,
                "selected_pair": self.selected_pair,
                "screening": self.screening,
                "diagnostic": self.diagnostic,
                "walk_forward": self.walk_forward,
                "robustness": self.robustness,
                "validation": self.validation,
                "configuration_snapshot": self.configuration_snapshot,
                "metadata": self.metadata,
                "provenance": self.provenance,
                "warnings": self.warnings,
            }
        )


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
    required_tables = frozenset(_REQUIRED_SCHEMA_COLUMNS)
    existing_tables = {
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        ).fetchall()
        if row[0] in required_tables
    }
    if not existing_tables:
        with _transaction(connection):
            connection.execute(_load_schema())
    elif existing_tables == _BASE_SCHEMA_TABLES:
        # Milestone 10A is an additive migration from the established market,
        # quality, and screening schema.  CREATE TABLE IF NOT EXISTS preserves
        # all earlier data while adding only experiment-history tables.
        _validate_schema_tables(connection, _BASE_SCHEMA_TABLES)
        with _transaction(connection):
            connection.execute(_load_schema())
    elif existing_tables != required_tables:
        raise RuntimeError(
            "Database contains a partial research schema; explicit migration or "
            "recreation is required."
        )
    _validate_schema(connection)


def _validate_schema(connection: duckdb.DuckDBPyConnection) -> None:
    """Reject missing or legacy schema shapes without inventing provenance."""
    _validate_schema_tables(connection, frozenset(_REQUIRED_SCHEMA_COLUMNS))


def _validate_schema_tables(
    connection: duckdb.DuckDBPyConnection,
    required_tables: frozenset[str],
) -> None:
    """Validate an explicit trusted subset before any additive migration."""
    existing_tables = {
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        ).fetchall()
    }
    missing_tables = sorted(set(required_tables) - existing_tables)
    if missing_tables:
        raise RuntimeError(
            f"Database schema is missing required tables: {missing_tables}."
        )
    for table in sorted(required_tables):
        required = _REQUIRED_SCHEMA_COLUMNS[table]
        table_info = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        actual = {row[1]: str(row[2]).upper() for row in table_info}
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
        if table in _EXPERIMENT_NOT_NULL_COLUMNS:
            not_null = {row[1] for row in table_info if bool(row[3])}
            missing_not_null = sorted(
                _EXPERIMENT_NOT_NULL_COLUMNS[table] - not_null
            )
            if missing_not_null:
                raise RuntimeError(
                    f"Database table {table!r} is missing required NOT NULL "
                    f"constraints on {missing_not_null}."
                )

    experiment_tables = required_tables.intersection(_EXPERIMENT_PRIMARY_KEYS)
    if not experiment_tables:
        return
    rows = connection.execute(
        """
        SELECT table_name, constraint_type, constraint_column_names,
               referenced_table, referenced_column_names, expression
        FROM duckdb_constraints()
        WHERE schema_name = current_schema()
        """
    ).fetchall()
    by_table: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        by_table.setdefault(str(row[0]), []).append(row)
    for table in sorted(experiment_tables):
        constraints = by_table.get(table, [])
        primary_keys = {
            tuple(row[2]) for row in constraints if row[1] == "PRIMARY KEY"
        }
        if _EXPERIMENT_PRIMARY_KEYS[table] not in primary_keys:
            raise RuntimeError(
                f"Database table {table!r} is missing its required primary key."
            )
        expressions = [
            str(row[5]).casefold()
            for row in constraints
            if row[1] == "CHECK" and row[5] is not None
        ]
        for required_tokens in _EXPERIMENT_CHECK_TOKEN_SETS[table]:
            if not any(all(token in expression for token in required_tokens) for expression in expressions):
                raise RuntimeError(
                    f"Database table {table!r} is missing a material CHECK constraint."
                )
    summary_constraints = by_table.get("research_experiment_summaries", [])
    required_fk = any(
        row[1] == "FOREIGN KEY"
        and tuple(row[2]) == ("run_id",)
        and row[3] == "research_experiments"
        and tuple(row[4]) == ("run_id",)
        for row in summary_constraints
    )
    if "research_experiment_summaries" in experiment_tables and not required_fk:
        raise RuntimeError(
            "Database research_experiment_summaries is missing the required "
            "run_id foreign key."
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


def _plain_json_value(value: Any, field: str) -> Any:
    """Return safe JSON primitives for experiment metadata persistence."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{field} must not contain NaN or infinity.")
        return number
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{field} keys must be non-empty strings.")
            result[key] = _plain_json_value(nested, f"{field}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _plain_json_value(nested, f"{field}[]") for nested in value
        ]
    raise TypeError(f"{field} contains an unsupported JSON value.")


def _encode_json_value(value: Any, field: str) -> str:
    return json.dumps(
        _plain_json_value(value, field),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _decode_json_mapping(value: Any, field: str) -> dict[str, Any]:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"Non-finite JSON constant {constant!r} is forbidden.")

    try:
        decoded = json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stored {field} is not valid JSON.") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Stored {field} must be a JSON object.")
    return _plain_json_value(decoded, field)


def _decode_json_strings(value: Any, field: str) -> tuple[str, ...]:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"Non-finite JSON constant {constant!r} is forbidden.")

    try:
        decoded = json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stored {field} is not valid JSON.") from exc
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise RuntimeError(f"Stored {field} must be a JSON string array.")
    return tuple(decoded)


def _optional_summary_float(value: Any, field: str) -> float | None:
    """Persist undefined/non-finite canonical metrics as SQL NULL."""
    if value is None or _is_missing_scalar(value):
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be numeric or missing.")
    number = float(value)
    return number if np.isfinite(number) else None


def _optional_summary_integer(value: Any, field: str) -> int | None:
    if value is None or _is_missing_scalar(value):
        return None
    return _strict_non_negative_integer(value, field)


def _enum_value(value: Any, enum_type: type, field: str) -> str:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field} must be a declared {enum_type.__name__} member.")
    return str(value.value)


def _strict_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be Boolean.")
    return value


def _persisted_label(value: Any, field: str) -> str:
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError(f"{field} must not be missing.")
        return timestamp.isoformat()
    normalized = _plain_json_value(value, field)
    if isinstance(normalized, (dict, list)):
        raise TypeError(f"{field} must be a scalar label.")
    return json.dumps(normalized, ensure_ascii=True, allow_nan=False)


def _normalise_experiment_rows(result: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Validate a pipeline result and materialize both normalized table rows."""
    from .pipeline import (
        DiagnosticExecutionCoverage,
        PipelineStageStatus,
        ResearchExperimentResult,
        ResearchPipelineStatus,
        validate_research_experiment_result,
    )
    from .robustness import MetricAvailabilityStatus
    from .validation import ValidationAvailability
    from .walkforward import WalkForwardAnalyticsStatus

    if not isinstance(result, ResearchExperimentResult):
        raise TypeError("result must be a ResearchExperimentResult.")
    validate_research_experiment_result(result)
    run_id = _normalise_non_empty_string(result.run_id, "run_id")
    experiment_name = _normalise_non_empty_string(
        result.experiment_name, "experiment_name"
    )
    created_at = _normalise_timestamp(result.created_at, "created_at")
    assert created_at is not None

    configuration_json = _encode_json_value(
        result.configuration_snapshot, "configuration_snapshot"
    )
    metadata_json = _encode_json_value(result.metadata, "metadata")
    provenance_json = _encode_json_value(result.provenance, "provenance")
    warnings_json = _encode_json_value(result.warnings, "warnings")
    experiment_row = (
        run_id,
        result.research_content_digest,
        result.price_content_digest,
        result.configuration_digest,
        result.research_pipeline_version,
        result.configuration_snapshot_version,
        result.experiment_schema_version,
        experiment_name,
        created_at,
        _enum_value(result.status, ResearchPipelineStatus, "pipeline_status"),
        configuration_json,
        metadata_json,
        provenance_json,
        warnings_json,
    )

    selected = result.selected_pair
    selected_pair_id = None
    symbol_y = None
    symbol_x = None
    selected_rank = None
    alpha = beta = cointegration_statistic = cointegration_pvalue = None
    corrected_pvalue = half_life = hurst = None
    if selected is not None:
        if not selected.selected or selected.rank is None:
            raise ValueError("selected_pair must be a canonically selected ranked result.")
        symbol_y = _normalise_symbol(selected.symbol_y, "selected_pair.symbol_y")
        symbol_x = _normalise_symbol(selected.symbol_x, "selected_pair.symbol_x")
        selected_pair_id = f"{symbol_y}|{symbol_x}"
        selected_rank = _strict_non_negative_integer(
            selected.rank, "selected_pair.rank"
        )
        if selected_rank < 1:
            raise ValueError("selected_pair.rank must be positive.")
        alpha = _optional_summary_float(selected.alpha, "selected_pair.alpha")
        beta = _optional_summary_float(selected.beta, "selected_pair.beta")
        cointegration_statistic = _optional_summary_float(
            selected.cointegration_statistic,
            "selected_pair.cointegration_statistic",
        )
        cointegration_pvalue = _optional_summary_float(
            selected.cointegration_pvalue, "selected_pair.cointegration_pvalue"
        )
        corrected_pvalue = _optional_summary_float(
            selected.corrected_pvalue, "selected_pair.corrected_pvalue"
        )
        half_life = _optional_summary_float(
            selected.half_life, "selected_pair.half_life"
        )
        hurst = _optional_summary_float(selected.hurst, "selected_pair.hurst")

    performance = result.performance_report
    diagnostic_performance_values: tuple[Any, ...]
    if performance is None:
        diagnostic_performance_values = (None,) * 8
    else:
        diagnostic_performance_values = (
            _optional_summary_float(
                performance.core.total_return,
                "diagnostic_in_sample_total_return",
            ),
            _optional_summary_float(
                performance.core.annualized_return,
                "diagnostic_in_sample_annualized_return",
            ),
            _optional_summary_float(
                performance.core.annualized_volatility,
                "diagnostic_in_sample_annualized_volatility",
            ),
            _optional_summary_float(
                performance.core.sharpe_ratio,
                "diagnostic_in_sample_sharpe_ratio",
            ),
            _optional_summary_float(
                performance.core.sortino_ratio,
                "diagnostic_in_sample_sortino_ratio",
            ),
            _optional_summary_float(
                performance.drawdown.maximum_drawdown,
                "diagnostic_in_sample_maximum_drawdown",
            ),
            _optional_summary_float(
                performance.drawdown.calmar_ratio,
                "diagnostic_in_sample_calmar_ratio",
            ),
            _strict_non_negative_integer(
                performance.trades.trades,
                "diagnostic_in_sample_trade_count",
            ),
        )

    coverage = result.diagnostic_execution_coverage
    diagnostic_coverage_values: tuple[Any, ...]
    if coverage is None:
        diagnostic_coverage_values = (None,) * 8
    else:
        if not isinstance(coverage, DiagnosticExecutionCoverage):
            raise TypeError("diagnostic_execution_coverage has an invalid type.")
        diagnostic_coverage_values = (
            coverage.total_rows,
            coverage.finite_beta_rows,
            coverage.positive_execution_beta_rows,
            coverage.non_positive_execution_beta_rows,
            coverage.finite_signal_rows,
            coverage.entry_execution_unavailable_rows_due_to_beta,
            coverage.signal_observation_coverage,
            coverage.beta_execution_policy,
        )

    walk_forward = result.walk_forward_result
    if walk_forward is None:
        walk_forward_values: tuple[Any, ...] = (None,) * 27
    else:
        calendar_report = walk_forward.calendar_performance_report
        if calendar_report is None:
            calendar_metrics: tuple[Any, ...] = (None,) * 8
        else:
            calendar_metrics = (
                _optional_summary_float(
                    calendar_report.core.total_return,
                    "walk_forward_calendar_oos_total_return",
                ),
                _optional_summary_float(
                    calendar_report.core.annualized_return,
                    "walk_forward_calendar_oos_annualized_return",
                ),
                _optional_summary_float(
                    calendar_report.core.annualized_volatility,
                    "walk_forward_calendar_oos_annualized_volatility",
                ),
                _optional_summary_float(
                    calendar_report.core.sharpe_ratio,
                    "walk_forward_calendar_oos_sharpe_ratio",
                ),
                _optional_summary_float(
                    calendar_report.core.sortino_ratio,
                    "walk_forward_calendar_oos_sortino_ratio",
                ),
                _optional_summary_float(
                    calendar_report.drawdown.maximum_drawdown,
                    "walk_forward_calendar_oos_maximum_drawdown",
                ),
                _optional_summary_float(
                    calendar_report.drawdown.calmar_ratio,
                    "walk_forward_calendar_oos_calmar_ratio",
                ),
                _strict_non_negative_integer(
                    calendar_report.report_observations,
                    "walk_forward_calendar_oos_report_observations",
                ),
            )
        walk_forward_values = (
            _enum_value(
                walk_forward.calendar_analytics_status,
                WalkForwardAnalyticsStatus,
                "walk_forward_analytics_status",
            ),
            _strict_non_negative_integer(walk_forward.fold_count, "walk_forward.fold_count"),
            _strict_non_negative_integer(
                walk_forward.completed_fold_count,
                "walk_forward.completed_fold_count",
            ),
            _strict_non_negative_integer(
                walk_forward.no_selection_fold_count,
                "walk_forward.no_selection_fold_count",
            ),
            _strict_non_negative_integer(
                walk_forward.insufficient_data_fold_count,
                "walk_forward.insufficient_data_fold_count",
            ),
            _strict_non_negative_integer(
                walk_forward.scheduled_oos_observations,
                "walk_forward.scheduled_oos_observations",
            ),
            _strict_non_negative_integer(
                walk_forward.scheduled_eligible_oos_observations,
                "walk_forward.scheduled_eligible_oos_observations",
            ),
            _strict_non_negative_integer(
                walk_forward.selected_oos_observations,
                "walk_forward.selected_oos_observations",
            ),
            _strict_non_negative_integer(
                walk_forward.no_selection_oos_observations,
                "walk_forward.no_selection_oos_observations",
            ),
            _strict_non_negative_integer(
                walk_forward.unavailable_oos_observations,
                "walk_forward.unavailable_oos_observations",
            ),
            _optional_summary_float(
                walk_forward.selection_coverage,
                "walk_forward.selection_coverage",
            ),
            _persisted_label(
                walk_forward.evaluated_start_label,
                "walk_forward.evaluated_start_label",
            ),
            _persisted_label(
                walk_forward.evaluated_end_label,
                "walk_forward.evaluated_end_label",
            ),
            walk_forward.capital_policy,
            walk_forward.aggregate_return_policy,
            walk_forward.inactive_capital_policy,
            walk_forward.selection_coverage_denominator,
            _strict_bool(
                walk_forward.aggregate_dollar_pnl_available,
                "walk_forward.aggregate_dollar_pnl_available",
            ),
            _strict_bool(
                walk_forward.aggregate_trade_dollar_metrics_available,
                "walk_forward.aggregate_trade_dollar_metrics_available",
            ),
            *calendar_metrics,
        )

    robustness = result.robustness_result
    if robustness is None:
        robustness_values: tuple[Any, ...] = (None,) * 17
    else:
        summary = robustness.summary
        robustness_values = (
            _normalise_non_empty_string(
                robustness.baseline_scenario_id, "robustness.baseline_scenario_id"
            ),
            _strict_non_negative_integer(summary.scenario_count, "robustness.scenario_count"),
            _strict_non_negative_integer(
                summary.completed_scenarios, "robustness.completed_scenarios"
            ),
            _strict_non_negative_integer(
                summary.unavailable_scenarios, "robustness.unavailable_scenarios"
            ),
            _strict_non_negative_integer(
                summary.invalid_scenarios, "robustness.invalid_scenarios"
            ),
            _strict_non_negative_integer(
                summary.failed_scenarios, "robustness.failed_scenarios"
            ),
            _strict_bool(
                robustness.common_horizon_structurally_available,
                "robustness.common_horizon_structurally_available",
            ),
            _strict_bool(
                robustness.common_horizon_fully_observed,
                "robustness.common_horizon_fully_observed",
            ),
            _strict_bool(
                robustness.common_horizon_analytics_available,
                "robustness.common_horizon_analytics_available",
            ),
            _enum_value(
                robustness.common_horizon_analytics_status,
                MetricAvailabilityStatus,
                "robustness.common_horizon_analytics_status",
            ),
            _strict_non_negative_integer(
                robustness.common_horizon_observations,
                "robustness.common_horizon_observations",
            ),
            _strict_non_negative_integer(
                robustness.common_horizon_scenario_count,
                "robustness.common_horizon_scenario_count",
            ),
            _normalise_non_empty_string(
                summary.headline_metric_basis,
                "robustness.headline_metric_basis",
            ),
            _normalise_non_empty_string(
                robustness.distribution_policy,
                "robustness.distribution_policy",
            ),
            _encode_json_value(
                robustness.tested_dimensions,
                "robustness.tested_dimensions",
            ),
            _encode_json_value(
                robustness.untested_material_dimensions,
                "robustness.untested_material_dimensions",
            ),
            _encode_json_value(
                robustness.provenance_warnings,
                "robustness.provenance_warnings",
            ),
        )

    validation = result.statistical_validation_result
    if validation is None:
        validation_values: tuple[Any, ...] = (None,) * 12
    else:
        validation_values = (
            _enum_value(
                validation.primary_inference_availability,
                ValidationAvailability,
                "validation.primary_inference_availability",
            ),
            _enum_value(
                validation.overall_availability,
                ValidationAvailability,
                "validation.overall_availability",
            ),
            _optional_summary_float(
                validation.probabilistic_sharpe.probability,
                "validation.probabilistic_sharpe_probability",
            ),
            _optional_summary_integer(
                validation.minimum_track_record.estimated_required_observations,
                "validation.minimum_track_record_observations",
            ),
            _enum_value(
                validation.fold_consistency.status,
                ValidationAvailability,
                "validation.fold_consistency_availability",
            ),
            _strict_non_negative_integer(
                validation.multiple_testing.total_tested_configurations,
                "validation.multiple_testing.total_tested_configurations",
            ),
            _strict_non_negative_integer(
                validation.multiple_testing.valid_comparable_configurations,
                "validation.multiple_testing.valid_comparable_configurations",
            ),
            _strict_non_negative_integer(
                validation.multiple_testing.valid_pvalue_count,
                "validation.multiple_testing.valid_pvalue_count",
            ),
            _strict_non_negative_integer(
                validation.multiple_testing.unavailable_pvalue_count,
                "validation.multiple_testing.unavailable_pvalue_count",
            ),
            _strict_non_negative_integer(
                validation.multiple_testing.eligible_hypothesis_count,
                "validation.multiple_testing.eligible_hypothesis_count",
            ),
            _normalise_non_empty_string(validation.purpose, "validation.purpose"),
            _encode_json_value(
                validation.provenance_warnings,
                "validation.provenance_warnings",
            ),
        )

    summary_row = (
        run_id,
        selected_pair_id,
        symbol_y,
        symbol_x,
        selected_rank,
        len(result.screening_results),
        sum(item.selected for item in result.screening_results),
        "full_sample_in_sample_diagnostic",
        "canonical_screen_pairs_selected_rank_ascending",
        alpha,
        beta,
        cointegration_statistic,
        cointegration_pvalue,
        corrected_pvalue,
        half_life,
        hurst,
        _enum_value(result.analytics_stage, PipelineStageStatus, "analytics_stage"),
        *diagnostic_performance_values,
        *diagnostic_coverage_values,
        _enum_value(
            result.walk_forward_stage,
            PipelineStageStatus,
            "walk_forward_stage",
        ),
        *walk_forward_values,
        _enum_value(
            result.robustness_stage,
            PipelineStageStatus,
            "robustness_stage",
        ),
        *robustness_values,
        _enum_value(
            result.validation_stage,
            PipelineStageStatus,
            "validation_stage",
        ),
        *validation_values,
    )
    return experiment_row, summary_row


_EXPERIMENT_INSERT_SQL = """
    INSERT INTO research_experiments (
        run_id, research_content_digest, price_content_digest,
        configuration_digest, research_pipeline_version,
        configuration_snapshot_version, experiment_schema_version,
        experiment_name, created_at, pipeline_status, configuration_snapshot,
        metadata, provenance, warnings
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_EXPERIMENT_SUMMARY_COLUMNS = tuple(
    _REQUIRED_SCHEMA_COLUMNS["research_experiment_summaries"]
)
_EXPERIMENT_SUMMARY_INSERT_SQL = (
    "INSERT INTO research_experiment_summaries ("
    + ", ".join(_EXPERIMENT_SUMMARY_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in _EXPERIMENT_SUMMARY_COLUMNS)
    + ")"
)


def _insert_research_experiment_summary(
    connection: duckdb.DuckDBPyConnection,
    summary_row: tuple[Any, ...],
) -> None:
    """Internal second-table insert kept separate for transactional testing."""
    connection.execute(_EXPERIMENT_SUMMARY_INSERT_SQL, summary_row)


def store_research_experiment(database: DatabaseTarget, result: Any) -> None:
    """Atomically persist one immutable experiment summary.

    Duplicate run identifiers are rejected explicitly.  Undefined or
    non-finite canonical metrics are stored as SQL NULL; they are never changed
    to zero.  Full paths, ledgers, and bootstrap samples remain in memory.
    """
    experiment_row, summary_row = _normalise_experiment_rows(result)
    run_id = experiment_row[0]
    with _connection_scope(database) as connection:
        _ensure_schema(connection)
        duplicate = connection.execute(
            "SELECT 1 FROM research_experiments WHERE run_id = ?",
            [run_id],
        ).fetchone()
        if duplicate is not None:
            raise ValueError(f"Experiment run {run_id!r} already exists.")
        with _transaction(connection):
            connection.execute(_EXPERIMENT_INSERT_SQL, experiment_row)
            _insert_research_experiment_summary(connection, summary_row)


def _empty_experiment_list() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": pd.Series(dtype="object"),
            "research_content_digest": pd.Series(dtype="object"),
            "experiment_name": pd.Series(dtype="object"),
            "created_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "pipeline_status": pd.Series(dtype="object"),
            "selected_pair_id": pd.Series(dtype="object"),
            "diagnostic_in_sample_total_return": pd.Series(dtype="float64"),
            "diagnostic_in_sample_sharpe_ratio": pd.Series(dtype="float64"),
            "diagnostic_in_sample_maximum_drawdown": pd.Series(dtype="float64"),
            "diagnostic_in_sample_trade_count": pd.Series(dtype="Int64"),
            "walk_forward_calendar_oos_total_return": pd.Series(dtype="float64"),
            "walk_forward_calendar_oos_sharpe_ratio": pd.Series(dtype="float64"),
            "walk_forward_calendar_oos_maximum_drawdown": pd.Series(dtype="float64"),
            "walk_forward_stage": pd.Series(dtype="object"),
            "robustness_stage": pd.Series(dtype="object"),
            "validation_stage": pd.Series(dtype="object"),
        }
    ).loc[:, list(EXPERIMENT_LIST_COLUMNS)]


def list_research_experiments(database: DatabaseTarget) -> pd.DataFrame:
    """List persisted experiments newest-first with deterministic ID tie-breaks."""
    query = """
        SELECT
            run_id, research_content_digest, experiment_name, created_at,
            pipeline_status, selected_pair_id,
            diagnostic_in_sample_total_return,
            diagnostic_in_sample_sharpe_ratio,
            diagnostic_in_sample_maximum_drawdown,
            diagnostic_in_sample_trade_count,
            walk_forward_calendar_oos_total_return,
            walk_forward_calendar_oos_sharpe_ratio,
            walk_forward_calendar_oos_maximum_drawdown,
            walk_forward_stage, robustness_stage, validation_stage
        FROM research_experiments
        JOIN research_experiment_summaries USING (run_id)
        ORDER BY created_at DESC, run_id ASC
    """
    with _read_connection_scope(database) as connection:
        _validate_schema(connection)
        frame = connection.execute(query).fetchdf()
    if frame.empty:
        return _empty_experiment_list()
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    frame["diagnostic_in_sample_trade_count"] = frame[
        "diagnostic_in_sample_trade_count"
    ].astype("Int64")
    for column in (
        "diagnostic_in_sample_total_return",
        "diagnostic_in_sample_sharpe_ratio",
        "diagnostic_in_sample_maximum_drawdown",
        "walk_forward_calendar_oos_total_return",
        "walk_forward_calendar_oos_sharpe_ratio",
        "walk_forward_calendar_oos_maximum_drawdown",
    ):
        frame[column] = frame[column].astype(float)
    return frame.loc[:, list(EXPERIMENT_LIST_COLUMNS)]


def _loaded_optional(value: Any) -> Any | None:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_research_experiment_summary(
    database: DatabaseTarget,
    run_id: str,
) -> ResearchExperimentSummary:
    """Load one persisted experiment without creating a missing database path."""
    from .pipeline import PipelineStageStatus, ResearchPipelineStatus
    from .robustness import MetricAvailabilityStatus
    from .validation import ValidationAvailability
    from .walkforward import WalkForwardAnalyticsStatus

    def loaded_enum(value: Any, enum_type: type, field: str) -> str | None:
        raw = _loaded_optional(value)
        if raw is None:
            return None
        try:
            return str(enum_type(str(raw)).value)
        except ValueError as exc:
            raise RuntimeError(f"Stored {field} has an unknown enum value.") from exc

    normalized_id = _normalise_non_empty_string(run_id, "run_id")
    query = """
        SELECT e.*, s.* EXCLUDE (run_id)
        FROM research_experiments e
        JOIN research_experiment_summaries s USING (run_id)
        WHERE run_id = ?
    """
    with _read_connection_scope(database) as connection:
        _validate_schema(connection)
        frame = connection.execute(query, [normalized_id]).fetchdf()
    if frame.empty:
        raise KeyError(f"Experiment {normalized_id!r} was not found.")
    if len(frame) != 1:
        raise RuntimeError("Experiment persistence contains duplicate summary rows.")
    row = frame.iloc[0]
    selected_pair = None
    if not _is_missing_scalar(row["selected_pair_id"]):
        selected_pair = {
            "pair_id": row["selected_pair_id"],
            "symbol_y": row["symbol_y"],
            "symbol_x": row["symbol_x"],
            "rank": int(row["selected_rank"]),
            "alpha": _loaded_optional(row["alpha"]),
            "beta": _loaded_optional(row["beta"]),
            "cointegration_statistic": _loaded_optional(
                row["cointegration_statistic"]
            ),
            "cointegration_pvalue": _loaded_optional(row["cointegration_pvalue"]),
            "corrected_pvalue": _loaded_optional(row["corrected_pvalue"]),
            "half_life": _loaded_optional(row["half_life"]),
            "hurst": _loaded_optional(row["hurst"]),
        }
    screening = {
        "candidate_count": int(row["screening_candidate_count"]),
        "selected_count": int(row["screening_selected_count"]),
        "selection_scope": str(row["screening_selection_scope"]),
        "ranking_policy": str(row["screening_ranking_policy"]),
    }
    diagnostic = {
        "stage": loaded_enum(
            row["analytics_stage"], PipelineStageStatus, "analytics_stage"
        ),
        "scope": "full_sample_in_sample_diagnostic",
        "total_return": _loaded_optional(
            row["diagnostic_in_sample_total_return"]
        ),
        "annualized_return": _loaded_optional(
            row["diagnostic_in_sample_annualized_return"]
        ),
        "annualized_volatility": _loaded_optional(
            row["diagnostic_in_sample_annualized_volatility"]
        ),
        "sharpe_ratio": _loaded_optional(
            row["diagnostic_in_sample_sharpe_ratio"]
        ),
        "sortino_ratio": _loaded_optional(
            row["diagnostic_in_sample_sortino_ratio"]
        ),
        "maximum_drawdown": _loaded_optional(
            row["diagnostic_in_sample_maximum_drawdown"]
        ),
        "calmar_ratio": _loaded_optional(
            row["diagnostic_in_sample_calmar_ratio"]
        ),
        "trade_count": (
            None
            if _is_missing_scalar(row["diagnostic_in_sample_trade_count"])
            else int(row["diagnostic_in_sample_trade_count"])
        ),
        "total_rows": _loaded_optional(row["diagnostic_total_rows"]),
        "finite_beta_rows": _loaded_optional(row["diagnostic_finite_beta_rows"]),
        "positive_execution_beta_rows": _loaded_optional(
            row["diagnostic_positive_execution_beta_rows"]
        ),
        "non_positive_execution_beta_rows": _loaded_optional(
            row["diagnostic_non_positive_execution_beta_rows"]
        ),
        "finite_signal_rows": _loaded_optional(
            row["diagnostic_finite_signal_rows"]
        ),
        "entry_execution_unavailable_rows_due_to_beta": _loaded_optional(
            row["diagnostic_entry_execution_unavailable_rows_due_to_beta"]
        ),
        "signal_observation_coverage": _loaded_optional(
            row["diagnostic_signal_observation_coverage"]
        ),
        "beta_execution_policy": _loaded_optional(
            row["diagnostic_beta_execution_policy"]
        ),
    }
    for key in (
        "total_rows",
        "finite_beta_rows",
        "positive_execution_beta_rows",
        "non_positive_execution_beta_rows",
        "finite_signal_rows",
        "entry_execution_unavailable_rows_due_to_beta",
    ):
        if diagnostic[key] is not None:
            diagnostic[key] = int(diagnostic[key])
    walk_forward = {
        "stage": loaded_enum(
            row["walk_forward_stage"], PipelineStageStatus, "walk_forward_stage"
        ),
        "calendar_analytics_status": loaded_enum(
            row["walk_forward_calendar_analytics_status"],
            WalkForwardAnalyticsStatus,
            "walk_forward_calendar_analytics_status",
        ),
        "fold_count": _loaded_optional(row["walk_forward_fold_count"]),
        "completed_fold_count": _loaded_optional(
            row["walk_forward_completed_fold_count"]
        ),
        "no_selection_fold_count": _loaded_optional(
            row["walk_forward_no_selection_fold_count"]
        ),
        "insufficient_data_fold_count": _loaded_optional(
            row["walk_forward_insufficient_data_fold_count"]
        ),
        "scheduled_oos_observations": _loaded_optional(
            row["walk_forward_scheduled_oos_observations"]
        ),
        "scheduled_eligible_oos_observations": _loaded_optional(
            row["walk_forward_scheduled_eligible_oos_observations"]
        ),
        "selected_oos_observations": _loaded_optional(
            row["walk_forward_selected_oos_observations"]
        ),
        "no_selection_oos_observations": _loaded_optional(
            row["walk_forward_no_selection_oos_observations"]
        ),
        "unavailable_oos_observations": _loaded_optional(
            row["walk_forward_unavailable_oos_observations"]
        ),
        "selection_coverage": _loaded_optional(
            row["walk_forward_selection_coverage"]
        ),
        "evaluated_start_label": _loaded_optional(
            row["walk_forward_evaluated_start_label"]
        ),
        "evaluated_end_label": _loaded_optional(
            row["walk_forward_evaluated_end_label"]
        ),
        "capital_policy": _loaded_optional(row["walk_forward_capital_policy"]),
        "aggregate_return_policy": _loaded_optional(
            row["walk_forward_aggregate_return_policy"]
        ),
        "inactive_capital_policy": _loaded_optional(
            row["walk_forward_inactive_capital_policy"]
        ),
        "selection_coverage_denominator": _loaded_optional(
            row["walk_forward_selection_coverage_denominator"]
        ),
        "aggregate_dollar_pnl_available": _loaded_optional(
            row["walk_forward_aggregate_dollar_pnl_available"]
        ),
        "aggregate_trade_dollar_metrics_available": _loaded_optional(
            row["walk_forward_aggregate_trade_dollar_metrics_available"]
        ),
        "calendar_oos_total_return": _loaded_optional(
            row["walk_forward_calendar_oos_total_return"]
        ),
        "calendar_oos_annualized_return": _loaded_optional(
            row["walk_forward_calendar_oos_annualized_return"]
        ),
        "calendar_oos_annualized_volatility": _loaded_optional(
            row["walk_forward_calendar_oos_annualized_volatility"]
        ),
        "calendar_oos_sharpe_ratio": _loaded_optional(
            row["walk_forward_calendar_oos_sharpe_ratio"]
        ),
        "calendar_oos_sortino_ratio": _loaded_optional(
            row["walk_forward_calendar_oos_sortino_ratio"]
        ),
        "calendar_oos_maximum_drawdown": _loaded_optional(
            row["walk_forward_calendar_oos_maximum_drawdown"]
        ),
        "calendar_oos_calmar_ratio": _loaded_optional(
            row["walk_forward_calendar_oos_calmar_ratio"]
        ),
        "calendar_oos_report_observations": _loaded_optional(
            row["walk_forward_calendar_oos_report_observations"]
        ),
    }
    for key in tuple(walk_forward):
        if key.endswith("count") or key.endswith("observations"):
            if walk_forward[key] is not None:
                walk_forward[key] = int(walk_forward[key])
    robustness = {
        "stage": loaded_enum(
            row["robustness_stage"], PipelineStageStatus, "robustness_stage"
        ),
        "baseline_scenario_id": _loaded_optional(
            row["robustness_baseline_scenario_id"]
        ),
        "scenario_count": _loaded_optional(row["robustness_scenario_count"]),
        "completed_scenarios": _loaded_optional(
            row["robustness_completed_scenarios"]
        ),
        "analytically_unavailable_scenarios": _loaded_optional(
            row["robustness_analytically_unavailable_scenarios"]
        ),
        "invalid_scenarios": _loaded_optional(row["robustness_invalid_scenarios"]),
        "failed_scenarios": _loaded_optional(row["robustness_failed_scenarios"]),
        "common_horizon_structurally_available": _loaded_optional(
            row["robustness_common_horizon_structurally_available"]
        ),
        "common_horizon_fully_observed": _loaded_optional(
            row["robustness_common_horizon_fully_observed"]
        ),
        "common_horizon_analytics_available": _loaded_optional(
            row["robustness_common_horizon_analytics_available"]
        ),
        "common_horizon_analytics_status": loaded_enum(
            row["robustness_common_horizon_analytics_status"],
            MetricAvailabilityStatus,
            "robustness_common_horizon_analytics_status",
        ),
        "common_horizon_observations": _loaded_optional(
            row["robustness_common_horizon_observations"]
        ),
        "common_horizon_eligible_scenario_count": _loaded_optional(
            row["robustness_common_horizon_eligible_scenario_count"]
        ),
        "headline_metric_basis": _loaded_optional(
            row["robustness_headline_metric_basis"]
        ),
        "distribution_policy": _loaded_optional(
            row["robustness_distribution_policy"]
        ),
        "tested_dimensions": (
            ()
            if _is_missing_scalar(row["robustness_tested_dimensions"])
            else _decode_json_strings(
                row["robustness_tested_dimensions"],
                "robustness_tested_dimensions",
            )
        ),
        "untested_material_dimensions": (
            ()
            if _is_missing_scalar(row["robustness_untested_material_dimensions"])
            else _decode_json_strings(
                row["robustness_untested_material_dimensions"],
                "robustness_untested_material_dimensions",
            )
        ),
        "provenance_warnings": (
            ()
            if _is_missing_scalar(row["robustness_provenance_warning_summary"])
            else _decode_json_strings(
                row["robustness_provenance_warning_summary"],
                "robustness_provenance_warning_summary",
            )
        ),
    }
    for key in (
        "scenario_count",
        "completed_scenarios",
        "analytically_unavailable_scenarios",
        "invalid_scenarios",
        "failed_scenarios",
        "common_horizon_observations",
        "common_horizon_eligible_scenario_count",
    ):
        if robustness[key] is not None:
            robustness[key] = int(robustness[key])
    validation = {
        "stage": loaded_enum(
            row["validation_stage"], PipelineStageStatus, "validation_stage"
        ),
        "primary_availability": loaded_enum(
            row["validation_primary_availability"],
            ValidationAvailability,
            "validation_primary_availability",
        ),
        "overall_availability": loaded_enum(
            row["validation_overall_availability"],
            ValidationAvailability,
            "validation_overall_availability",
        ),
        "probabilistic_sharpe_probability": _loaded_optional(
            row["probabilistic_sharpe_probability"]
        ),
        "minimum_track_record_observations": _loaded_optional(
            row["minimum_track_record_observations"]
        ),
        "fold_consistency_availability": loaded_enum(
            row["fold_consistency_availability"],
            ValidationAvailability,
            "fold_consistency_availability",
        ),
        "multiple_testing_total_configurations": _loaded_optional(
            row["multiple_testing_total_configurations"]
        ),
        "multiple_testing_valid_comparable_configurations": _loaded_optional(
            row["multiple_testing_valid_comparable_configurations"]
        ),
        "multiple_testing_valid_pvalue_count": _loaded_optional(
            row["multiple_testing_valid_pvalue_count"]
        ),
        "multiple_testing_unavailable_pvalue_count": _loaded_optional(
            row["multiple_testing_unavailable_pvalue_count"]
        ),
        "multiple_testing_eligible_hypothesis_count": _loaded_optional(
            row["multiple_testing_eligible_hypothesis_count"]
        ),
        "purpose": _loaded_optional(row["validation_purpose"]),
        "provenance_warnings": (
            ()
            if _is_missing_scalar(row["validation_provenance_warning_summary"])
            else _decode_json_strings(
                row["validation_provenance_warning_summary"],
                "validation_provenance_warning_summary",
            )
        ),
    }
    for key in (
        "minimum_track_record_observations",
        "multiple_testing_total_configurations",
        "multiple_testing_valid_comparable_configurations",
        "multiple_testing_valid_pvalue_count",
        "multiple_testing_unavailable_pvalue_count",
        "multiple_testing_eligible_hypothesis_count",
    ):
        if validation[key] is not None:
            validation[key] = int(validation[key])
    created_at = pd.Timestamp(row["created_at"])
    if created_at.tzinfo is None:
        created_at = created_at.tz_localize("UTC")
    else:
        created_at = created_at.tz_convert("UTC")
    return ResearchExperimentSummary(
        run_id=str(row["run_id"]),
        research_content_digest=str(row["research_content_digest"]),
        price_content_digest=str(row["price_content_digest"]),
        research_pipeline_version=str(row["research_pipeline_version"]),
        configuration_snapshot_version=int(row["configuration_snapshot_version"]),
        experiment_schema_version=int(row["experiment_schema_version"]),
        experiment_name=str(row["experiment_name"]),
        created_at=created_at.to_pydatetime(),
        configuration_digest=str(row["configuration_digest"]),
        pipeline_status=loaded_enum(
            row["pipeline_status"], ResearchPipelineStatus, "pipeline_status"
        ),
        selected_pair=selected_pair,
        screening=screening,
        diagnostic=diagnostic,
        walk_forward=walk_forward,
        robustness=robustness,
        validation=validation,
        configuration_snapshot=_decode_json_mapping(
            row["configuration_snapshot"], "configuration_snapshot"
        ),
        metadata=_decode_json_mapping(row["metadata"], "metadata"),
        provenance=_decode_json_mapping(row["provenance"], "provenance"),
        warnings=_decode_json_strings(row["warnings"], "warnings"),
    )
