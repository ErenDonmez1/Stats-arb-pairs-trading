"""HTTP contract tests for the read-only research-summary API."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import MappingProxyType

from fastapi.testclient import TestClient
import pytest

from pairs_trading.api import (
    CORS_ORIGINS_ENV,
    DATABASE_PATH_ENV,
    DEFAULT_CORS_ORIGINS,
    MAX_PAGE_LIMIT,
    create_app,
)
from pairs_trading.config import (
    ResearchConfig,
    ScreeningConfig,
    StrategyConfig,
    WalkForwardConfig,
)
from pairs_trading.data import make_synthetic_universe
from pairs_trading.database import (
    initialise_database,
    load_research_experiment_summary,
    store_research_experiment,
)
from pairs_trading.pipeline import (
    CONFIGURATION_SNAPSHOT_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    RESEARCH_PIPELINE_VERSION,
    ResearchExperimentRequest,
    ResearchExperimentResult,
    run_research_pipeline,
)


def _request() -> ResearchExperimentRequest:
    prices, groups = make_synthetic_universe(n_days=330, seed=103)
    config = ResearchConfig(
        universe=MappingProxyType(
            {group: tuple(symbols) for group, symbols in groups.items()}
        ),
        screening=ScreeningConfig(
            formation_days=150,
            min_observations=60,
            fdr_threshold=0.20,
            max_half_life=120.0,
            hurst_threshold=0.80,
        ),
        strategy=StrategyConfig(
            hedge_lookback=50,
            zscore_lookback=25,
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
            max_holding_days=30,
            target_annual_vol=0.10,
            max_gross_leverage=1.5,
        ),
        walk_forward=WalkForwardConfig(trading_days=45, min_selected_pairs=1),
        random_seed=29,
    )
    return ResearchExperimentRequest(
        "api-test",
        prices,
        config,
        initial_capital=100_000.0,
        target_gross_notional=20_000.0,
        run_robustness=False,
        run_statistical_validation=False,
        metadata={"owner": "api-test", "tags": ["offline"]},
    )


@pytest.fixture(scope="module")
def experiment_result() -> ResearchExperimentResult:
    return run_research_pipeline(_request())


@pytest.fixture
def empty_database(tmp_path: Path) -> Path:
    path = tmp_path / "research.duckdb"
    initialise_database(path)
    return path


@pytest.fixture
def populated_database(
    empty_database: Path,
    experiment_result: ResearchExperimentResult,
) -> tuple[Path, ResearchExperimentResult, ResearchExperimentResult]:
    older = replace(
        experiment_result,
        run_id="run_11111111111111111111111111111111",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    newer = replace(
        experiment_result,
        run_id="run_22222222222222222222222222222222",
        created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    store_research_experiment(empty_database, older)
    store_research_experiment(empty_database, newer)
    return empty_database, newer, older


def _client(database: Path, **kwargs: object) -> TestClient:
    kwargs.setdefault("allowed_origins", DEFAULT_CORS_ORIGINS)
    return TestClient(create_app(database_path=database, **kwargs))


def test_health_returns_200_without_touching_database(tmp_path: Path) -> None:
    missing = tmp_path / "not-created.duckdb"
    response = _client(missing).get("/health")

    assert response.status_code == 200
    assert not missing.exists()


def test_health_response_is_json(empty_database: Path) -> None:
    response = _client(empty_database).get("/health")

    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"status": "ok"}


def test_meta_returns_canonical_version_constants(empty_database: Path) -> None:
    response = _client(empty_database).get("/api/v1/meta")

    assert response.status_code == 200
    assert response.json() == {
        "research_pipeline_version": RESEARCH_PIPELINE_VERSION,
        "configuration_snapshot_version": CONFIGURATION_SNAPSHOT_VERSION,
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
    }


def test_empty_experiment_list_returns_empty_page(empty_database: Path) -> None:
    response = _client(empty_database).get("/api/v1/experiments")

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "limit": 25,
        "offset": 0,
        "count": 0,
        "total": 0,
    }


def test_persisted_experiments_appear_newest_first(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, older = populated_database
    response = _client(database).get("/api/v1/experiments")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert [item["run_id"] for item in payload["items"]] == [
        newer.run_id,
        older.run_id,
    ]


def test_pagination_limit_is_applied(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, _ = populated_database
    payload = _client(database).get("/api/v1/experiments?limit=1").json()

    assert payload["limit"] == 1
    assert payload["count"] == 1
    assert payload["total"] == 2
    assert payload["items"][0]["run_id"] == newer.run_id


def test_pagination_offset_is_applied(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, _, older = populated_database
    payload = _client(database).get(
        "/api/v1/experiments?limit=1&offset=1"
    ).json()

    assert payload["offset"] == 1
    assert payload["count"] == 1
    assert payload["items"][0]["run_id"] == older.run_id


@pytest.mark.parametrize("limit", [0, -1, "invalid"])
def test_invalid_limit_is_rejected(empty_database: Path, limit: object) -> None:
    response = _client(empty_database).get(
        "/api/v1/experiments", params={"limit": limit}
    )

    assert response.status_code == 422


def test_excessive_limit_is_rejected(empty_database: Path) -> None:
    response = _client(empty_database).get(
        "/api/v1/experiments", params={"limit": MAX_PAGE_LIMIT + 1}
    )

    assert response.status_code == 422


def test_known_run_id_returns_canonical_summary(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, _ = populated_database
    response = _client(database).get(f"/api/v1/experiments/{newer.run_id}")

    assert response.status_code == 200
    assert response.json() == load_research_experiment_summary(
        database, newer.run_id
    ).to_dict()


def test_unknown_run_id_returns_404(empty_database: Path) -> None:
    response = _client(empty_database).get(
        "/api/v1/experiments/run_ffffffffffffffffffffffffffffffff"
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Research experiment was not found."}


def test_returned_run_id_matches_requested_run(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, _, older = populated_database
    payload = _client(database).get(
        f"/api/v1/experiments/{older.run_id}"
    ).json()

    assert payload["run_id"] == older.run_id


def test_diagnostic_and_calendar_oos_names_remain_distinct(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, _ = populated_database
    client = _client(database)
    detail = client.get(f"/api/v1/experiments/{newer.run_id}").json()
    listed = client.get("/api/v1/experiments").json()["items"][0]

    assert detail["diagnostic"]["scope"] == "full_sample_in_sample_diagnostic"
    assert "total_return" in detail["diagnostic"]
    assert "calendar_oos_total_return" in detail["walk_forward"]
    assert "diagnostic_in_sample_total_return" in listed
    assert "walk_forward_calendar_oos_total_return" in listed


def test_unavailable_values_are_null_not_zero(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, _ = populated_database
    payload = _client(database).get(
        f"/api/v1/experiments/{newer.run_id}"
    ).json()

    assert payload["robustness"]["stage"] == newer.robustness_stage.value
    assert payload["robustness"]["baseline_scenario_id"] is None
    assert payload["validation"]["stage"] == newer.validation_stage.value
    assert payload["validation"]["probabilistic_sharpe_probability"] is None


def test_enum_statuses_serialize_as_strings(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, _ = populated_database
    payload = _client(database).get(
        f"/api/v1/experiments/{newer.run_id}"
    ).json()

    assert payload["pipeline_status"] == "COMPLETED"
    assert isinstance(payload["diagnostic"]["stage"], str)
    assert isinstance(payload["walk_forward"]["stage"], str)
    assert isinstance(payload["robustness"]["stage"], str)
    assert isinstance(payload["validation"]["stage"], str)


def test_api_json_contains_no_nan_or_infinity(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, _ = populated_database
    client = _client(database)
    payloads = (
        client.get("/api/v1/experiments").json(),
        client.get(f"/api/v1/experiments/{newer.run_id}").json(),
    )

    for payload in payloads:
        json.dumps(payload, allow_nan=False)


def test_get_does_not_mutate_persisted_experiment(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, _ = populated_database
    before = load_research_experiment_summary(database, newer.run_id).to_dict()

    response = _client(database).get(f"/api/v1/experiments/{newer.run_id}")

    after = load_research_experiment_summary(database, newer.run_id).to_dict()
    assert response.status_code == 200
    assert after == before


def test_repeated_get_responses_are_deterministic(
    populated_database: tuple[Path, ResearchExperimentResult, ResearchExperimentResult],
) -> None:
    database, newer, _ = populated_database
    client = _client(database)

    assert client.get("/api/v1/experiments").content == client.get(
        "/api/v1/experiments"
    ).content
    assert client.get(f"/api/v1/experiments/{newer.run_id}").content == client.get(
        f"/api/v1/experiments/{newer.run_id}"
    ).content


def test_cors_allows_documented_development_origin(empty_database: Path) -> None:
    client = _client(empty_database)
    allowed = client.get("/health", headers={"Origin": DEFAULT_CORS_ORIGINS[0]})
    denied = client.get("/health", headers={"Origin": "https://example.invalid"})

    assert allowed.headers["access-control-allow-origin"] == DEFAULT_CORS_ORIGINS[0]
    assert "access-control-allow-credentials" not in allowed.headers
    assert "access-control-allow-origin" not in denied.headers


def test_cors_origin_is_configurable(empty_database: Path) -> None:
    origin = "https://dashboard.example.test"
    response = _client(empty_database, allowed_origins=[origin]).get(
        "/health", headers={"Origin": origin}
    )

    assert response.headers["access-control-allow-origin"] == origin


def test_database_path_can_be_configured_from_environment(
    empty_database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DATABASE_PATH_ENV, str(empty_database))
    application = create_app(allowed_origins=DEFAULT_CORS_ORIGINS)

    response = TestClient(application).get("/api/v1/experiments")

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert application.state.database_path == empty_database


def test_cors_origins_can_be_configured_from_environment(
    empty_database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = "https://one.example.test"
    second = "https://two.example.test"
    monkeypatch.setenv(CORS_ORIGINS_ENV, f"{first}, {second}")
    client = TestClient(create_app(database_path=empty_database))

    assert client.get(
        "/health", headers={"Origin": second}
    ).headers["access-control-allow-origin"] == second


def test_api_exposes_no_arbitrary_sql_endpoint(empty_database: Path) -> None:
    client = _client(empty_database)

    assert client.get("/api/v1/sql").status_code == 404
    assert client.post("/api/v1/sql", json={"query": "SELECT 1"}).status_code == 404


def test_api_exposes_no_research_execution_post(empty_database: Path) -> None:
    response = _client(empty_database).post("/research/run", json={})

    assert response.status_code == 404


def test_openapi_and_docs_remain_available(empty_database: Path) -> None:
    client = _client(empty_database)

    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
