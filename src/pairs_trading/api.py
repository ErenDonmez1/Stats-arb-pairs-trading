"""Read-only HTTP presentation layer for persisted research experiments."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import os
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .database import list_research_experiments, load_research_experiment_summary
from .pipeline import (
    CONFIGURATION_SNAPSHOT_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    RESEARCH_PIPELINE_VERSION,
)


DATABASE_PATH_ENV = "PAIRS_TRADING_DB_PATH"
CORS_ORIGINS_ENV = "PAIRS_TRADING_CORS_ORIGINS"
DEFAULT_DATABASE_PATH = Path("data/research.duckdb")
DEFAULT_CORS_ORIGINS = ("http://localhost:5173",)
DEFAULT_PAGE_LIMIT = 25
MAX_PAGE_LIMIT = 100

__all__ = [
    "CORS_ORIGINS_ENV",
    "DATABASE_PATH_ENV",
    "DEFAULT_CORS_ORIGINS",
    "DEFAULT_DATABASE_PATH",
    "DEFAULT_PAGE_LIMIT",
    "MAX_PAGE_LIMIT",
    "app",
    "create_app",
]


class HealthResponse(BaseModel):
    """Liveness response that does not touch persistence."""

    status: Literal["ok"]


class MetaResponse(BaseModel):
    """Version metadata imported from the canonical research pipeline."""

    research_pipeline_version: str
    configuration_snapshot_version: int
    experiment_schema_version: int


class ExperimentPage(BaseModel):
    """Deterministically ordered page of persisted experiment list records."""

    items: list[dict[str, Any]]
    limit: int
    offset: int
    count: int
    total: int


def _database_path(value: str | os.PathLike[str] | None) -> Path:
    raw_value: str | os.PathLike[str]
    if value is None:
        raw_value = os.environ.get(DATABASE_PATH_ENV, str(DEFAULT_DATABASE_PATH))
    else:
        raw_value = value
    if isinstance(raw_value, bool) or not isinstance(raw_value, (str, os.PathLike)):
        raise TypeError("database_path must be a filesystem path.")
    raw_path = os.fspath(raw_value)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("database_path must not be empty.")
    if raw_path == ":memory:":
        raise ValueError("The API requires a persistent DuckDB filesystem path.")
    return Path(raw_path).expanduser()


def _normalise_origin(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("CORS origins must be non-empty strings.")
    origin = value.strip().rstrip("/")
    if origin == "*":
        raise ValueError("Wildcard CORS origins are not supported.")
    parsed = urlsplit(origin)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Invalid CORS origin: {value!r}.")
    return origin


def _allowed_origins(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        configured = os.environ.get(CORS_ORIGINS_ENV)
        raw_values: Iterable[str] = (
            DEFAULT_CORS_ORIGINS if configured is None else configured.split(",")
        )
    elif isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = values
    origins = tuple(dict.fromkeys(_normalise_origin(item) for item in raw_values))
    if not origins:
        raise ValueError("At least one CORS origin is required.")
    return origins


def _list_value(value: Any, field: str) -> Any:
    """Convert one database-list scalar without hiding non-finite corruption."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        moment = value
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")
        else:
            moment = moment.tz_convert("UTC")
        return moment.isoformat().replace("+00:00", "Z")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return None
        if not np.isfinite(number):
            raise ValueError(f"Persisted experiment field {field!r} is non-finite.")
        return number
    if isinstance(value, str):
        return value
    raise TypeError(f"Persisted experiment field {field!r} is not JSON-compatible.")


def _page_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _list_value(value, column) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def create_app(
    *,
    database_path: str | os.PathLike[str] | None = None,
    allowed_origins: Iterable[str] | None = None,
) -> FastAPI:
    """Create a read-only API bound to one server-controlled DuckDB path.

    The path is injected directly for tests or read from
    ``PAIRS_TRADING_DB_PATH``. CORS origins are injected or read as a
    comma-separated ``PAIRS_TRADING_CORS_ORIGINS`` value.
    """
    database = _database_path(database_path)
    origins = _allowed_origins(allowed_origins)
    application = FastAPI(
        title="Pairs-Trading Research API",
        version=RESEARCH_PIPELINE_VERSION,
        description="Read-only access to canonical persisted research summaries.",
    )
    application.state.database_path = database
    application.state.allowed_origins = origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    router = APIRouter(prefix="/api/v1", tags=["research experiments"])

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["service"],
        summary="Check API liveness",
    )
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @router.get(
        "/meta",
        response_model=MetaResponse,
        summary="Get canonical research schema versions",
    )
    def metadata() -> MetaResponse:
        return MetaResponse(
            research_pipeline_version=RESEARCH_PIPELINE_VERSION,
            configuration_snapshot_version=CONFIGURATION_SNAPSHOT_VERSION,
            experiment_schema_version=EXPERIMENT_SCHEMA_VERSION,
        )

    @router.get(
        "/experiments",
        response_model=ExperimentPage,
        summary="List persisted research experiments",
    )
    def experiments(
        limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
        offset: int = Query(0, ge=0),
    ) -> ExperimentPage:
        listed = list_research_experiments(database)
        total = len(listed)
        page = listed.iloc[offset : offset + limit]
        items = _page_records(page)
        return ExperimentPage(
            items=items,
            limit=limit,
            offset=offset,
            count=len(items),
            total=total,
        )

    @router.get(
        "/experiments/{run_id}",
        response_model=dict[str, Any],
        summary="Get one canonical research experiment summary",
    )
    def experiment(run_id: str) -> dict[str, Any]:
        try:
            summary = load_research_experiment_summary(database, run_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Research experiment was not found.",
            ) from exc
        return summary.to_dict()

    application.include_router(router)
    return application


app = create_app()
