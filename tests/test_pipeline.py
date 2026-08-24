"""Tests for typed research orchestration without live-trading behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import subprocess
import sys
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pairs_trading.pipeline as pipeline_module
from pairs_trading.analytics import StrategyPerformanceReport
from pairs_trading.backtest import (
    BacktestResult,
    run_pair_backtest,
    validate_backtest_invariants,
)
from pairs_trading.config import (
    ResearchConfig,
    ScreeningConfig,
    StrategyConfig,
    WalkForwardConfig,
    screening_kwargs_from_config,
)
from pairs_trading.data import OBSERVED_PRICE_MASK_ATTR, make_synthetic_universe
from pairs_trading.pipeline import (
    CONFIGURATION_SNAPSHOT_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    RESEARCH_PIPELINE_VERSION,
    DiagnosticExecutionCoverage,
    PipelineStageStatus,
    ResearchExperimentRequest,
    ResearchExperimentResult,
    ResearchPipelineStatus,
    StatisticalValidationSettings,
    build_configuration_snapshot,
    configuration_digest,
    price_content_digest,
    research_content_digest,
    run_research_pipeline,
    validate_research_experiment_result,
)
from pairs_trading.robustness import RobustnessResult
from pairs_trading.screening import PairScreeningResult, screen_pairs
from pairs_trading.validation import StatisticalValidationResult
from pairs_trading.walkforward import WalkForwardResult


def _config(groups: dict[str, list[str]]) -> ResearchConfig:
    return ResearchConfig(
        universe=MappingProxyType(
            {group: tuple(symbols) for group, symbols in groups.items()}
        ),
        screening=ScreeningConfig(
            formation_days=180,
            min_observations=60,
            fdr_threshold=0.20,
            max_half_life=120.0,
            hurst_threshold=0.80,
        ),
        strategy=StrategyConfig(
            hedge_lookback=60,
            zscore_lookback=30,
            entry_z=2.0,
            exit_z=0.5,
            stop_z=4.0,
            max_holding_days=30,
            target_annual_vol=0.10,
            max_gross_leverage=1.5,
        ),
        walk_forward=WalkForwardConfig(trading_days=45, min_selected_pairs=1),
        random_seed=17,
    )


def _request(
    *,
    run_robustness: bool = False,
    run_validation: bool = False,
) -> ResearchExperimentRequest:
    prices, groups = make_synthetic_universe(n_days=360, seed=42)
    return ResearchExperimentRequest(
        "synthetic-research",
        prices,
        _config(groups),
        initial_capital=100_000.0,
        target_gross_notional=20_000.0,
        run_robustness=run_robustness,
        run_statistical_validation=run_validation,
        validation_settings=StatisticalValidationSettings(
            block_length=5,
            n_bootstrap=20,
            random_seed=91,
        ),
        metadata={"owner": "research", "tags": ["synthetic", "offline"]},
    )


@pytest.fixture(scope="module")
def real_pipeline_case() -> tuple[ResearchExperimentRequest, ResearchExperimentResult]:
    request = _request()
    return request, run_research_pipeline(request)


@pytest.fixture(scope="module")
def optional_pipeline_result() -> ResearchExperimentResult:
    return run_research_pipeline(_request(run_robustness=True, run_validation=True))


def _rejected_result() -> PairScreeningResult:
    return PairScreeningResult(
        symbol_y="AAA",
        symbol_x="BBB",
        group="group",
        observations=300,
        alpha=0.0,
        beta=1.0,
        spread_standard_deviation=0.1,
        cointegration_statistic=-1.0,
        cointegration_pvalue=0.5,
        corrected_pvalue=0.5,
        cointegration_critical_values={"1%": -3.9, "5%": -3.3, "10%": -3.0},
        adf_statistic=-1.0,
        adf_pvalue=0.5,
        half_life=100.0,
        hurst=0.7,
        selected=False,
        rank=None,
        rejection_reasons=("corrected_cointegration_pvalue_above_threshold",),
    )


def _no_pair_request(
    *, run_robustness: bool = False, run_validation: bool = False
) -> ResearchExperimentRequest:
    index = pd.bdate_range("2020-01-01", periods=300, name="Date")
    prices = pd.DataFrame(
        {"AAA": range(100, 400), "BBB": range(200, 500)}, index=index
    ).astype(float)
    config = _config({"group": ["AAA", "BBB"]})
    return ResearchExperimentRequest(
        "no-selection",
        prices,
        config,
        run_robustness=run_robustness,
        run_statistical_validation=run_validation,
    )


def test_real_end_to_end_pipeline_matches_canonical_screening_and_result_types(
    real_pipeline_case: tuple[ResearchExperimentRequest, ResearchExperimentResult],
) -> None:
    request, result = real_pipeline_case
    direct_screening = screen_pairs(
        request.prices,
        groups=request.config.universe,
        **screening_kwargs_from_config(request.config.screening),
    )

    assert result.screening_results == direct_screening
    assert result.selected_pair == next(
        item for item in direct_screening if item.rank == 1
    )
    assert result.status is ResearchPipelineStatus.COMPLETED
    assert isinstance(result.backtest_result, BacktestResult)
    assert isinstance(result.performance_report, StrategyPerformanceReport)
    assert isinstance(result.walk_forward_result, WalkForwardResult)
    validate_backtest_invariants(result.backtest_result)


def test_no_full_sample_pair_keeps_independent_walk_forward_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"screening": 0}

    def no_selection(*args: Any, **kwargs: Any) -> tuple[PairScreeningResult, ...]:
        calls["screening"] += 1
        return (_rejected_result(),)

    monkeypatch.setattr(pipeline_module, "screen_pairs", no_selection)
    request = _no_pair_request()
    first = run_research_pipeline(request)
    second = run_research_pipeline(request)

    assert calls["screening"] == 2
    assert first.status is ResearchPipelineStatus.COMPLETED
    assert first.run_id != second.run_id
    assert first.research_content_digest == second.research_content_digest
    assert first.configuration_digest == second.configuration_digest
    assert first.selected_pair is None
    assert first.backtest_result is None
    assert first.performance_report is None
    assert isinstance(first.walk_forward_result, WalkForwardResult)
    assert first.analytics_stage is PipelineStageStatus.UNAVAILABLE
    assert first.walk_forward_stage is PipelineStageStatus.COMPLETED
    assert first.robustness_stage is PipelineStageStatus.NOT_REQUESTED
    assert first.validation_stage is PipelineStageStatus.NOT_REQUESTED


def test_optional_oos_stages_are_independent_of_full_sample_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_module, "screen_pairs", lambda *args, **kwargs: (_rejected_result(),)
    )
    result = run_research_pipeline(
        _no_pair_request(run_robustness=True, run_validation=True)
    )

    assert isinstance(result.robustness_result, RobustnessResult)
    assert isinstance(result.statistical_validation_result, StatisticalValidationResult)
    assert result.robustness_stage is PipelineStageStatus.COMPLETED
    assert result.validation_stage is PipelineStageStatus.COMPLETED


def test_rank_one_selection_precedes_and_ignores_downstream_results(
    real_pipeline_case: tuple[ResearchExperimentRequest, ResearchExperimentResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, canonical = real_pipeline_case
    rank_one = canonical.selected_pair
    assert rank_one is not None
    alternative = next(
        item
        for item in canonical.screening_results
        if (item.symbol_y, item.symbol_x)
        != (rank_one.symbol_y, rank_one.symbol_x)
    )
    rank_two = replace(
        alternative,
        selected=True,
        rank=2,
        rejection_reasons=(),
    )
    downstream_calls: list[str] = []
    monkeypatch.setattr(
        pipeline_module,
        "screen_pairs",
        lambda *args, **kwargs: (rank_two, rank_one),
    )

    def estimates(y: pd.Series, x: pd.Series, lookback: int) -> pd.DataFrame:
        assert (y.name, x.name) == (rank_one.symbol_y, rank_one.symbol_x)
        downstream_calls.append("statistics")
        assert canonical.hedge_estimates is not None
        return canonical.hedge_estimates.copy(deep=True)

    def zscore(*args: Any, **kwargs: Any) -> pd.Series:
        downstream_calls.append("zscore")
        assert canonical.zscore is not None
        return canonical.zscore.copy(deep=True)

    monkeypatch.setattr(pipeline_module, "rolling_ols_spread", estimates)
    monkeypatch.setattr(pipeline_module, "rolling_zscore", zscore)
    monkeypatch.setattr(
        pipeline_module,
        "run_pair_backtest",
        lambda *args, **kwargs: (
            downstream_calls.append("backtest") or canonical.backtest_result
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_performance_report",
        lambda *args, **kwargs: (
            downstream_calls.append("analytics") or canonical.performance_report
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_walk_forward_analysis",
        lambda *args, **kwargs: (
            downstream_calls.append("walk_forward") or canonical.walk_forward_result
        ),
    )

    result = run_research_pipeline(request)

    assert result.selected_pair == rank_one
    assert downstream_calls == [
        "statistics",
        "zscore",
        "backtest",
        "analytics",
        "walk_forward",
        "analytics",
    ]


def test_unexpected_stage_invariant_error_propagates(
    real_pipeline_case: tuple[ResearchExperimentRequest, ResearchExperimentResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, canonical = real_pipeline_case
    monkeypatch.setattr(
        pipeline_module,
        "screen_pairs",
        lambda *args, **kwargs: canonical.screening_results,
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_pair_backtest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("canonical accounting invariant failed")
        ),
    )

    with pytest.raises(RuntimeError, match="canonical accounting invariant"):
        run_research_pipeline(request)


def test_optional_expensive_stages_execute_only_when_requested(
    real_pipeline_case: tuple[ResearchExperimentRequest, ResearchExperimentResult],
    optional_pipeline_result: ResearchExperimentResult,
) -> None:
    _, baseline = real_pipeline_case
    assert baseline.robustness_stage is PipelineStageStatus.NOT_REQUESTED
    assert baseline.validation_stage is PipelineStageStatus.NOT_REQUESTED
    assert baseline.robustness_result is None
    assert baseline.statistical_validation_result is None

    assert optional_pipeline_result.robustness_stage is PipelineStageStatus.COMPLETED
    assert optional_pipeline_result.validation_stage is PipelineStageStatus.COMPLETED
    assert isinstance(optional_pipeline_result.robustness_result, RobustnessResult)
    assert isinstance(
        optional_pipeline_result.statistical_validation_result,
        StatisticalValidationResult,
    )
    assert optional_pipeline_result.robustness_result.baseline_scenario_id == "baseline"


def test_validation_cannot_be_requested_without_robustness() -> None:
    with pytest.raises(ValueError, match="requires run_robustness"):
        _request(run_validation=True)


def test_pipeline_preserves_caller_prices_groups_config_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices, groups = make_synthetic_universe(n_days=300, seed=11)
    config = _config(groups)
    prices_before = prices.copy(deep=True)
    groups_before = {key: list(value) for key, value in groups.items()}
    config_before = config.to_dict()
    metadata = {"tags": ["one", "two"]}
    request = ResearchExperimentRequest("ownership", prices, config, metadata=metadata)

    prices.iloc[0, 0] = -999.0
    groups[next(iter(groups))].append("MUTATED")
    metadata["tags"].append("three")
    monkeypatch.setattr(
        pipeline_module, "screen_pairs", lambda *args, **kwargs: (_rejected_result(),)
    )
    run_research_pipeline(request)

    pd.testing.assert_frame_equal(request.prices, prices_before)
    assert request.config.to_dict() == config_before
    assert request.metadata["tags"] == ("one", "two")
    assert groups_before != groups


def test_configuration_snapshot_and_content_digest_are_stable_but_run_ids_are_unique() -> None:
    first = _request()
    second = _request()
    first_snapshot = build_configuration_snapshot(first)
    second_snapshot = build_configuration_snapshot(second)

    assert first_snapshot == second_snapshot
    assert configuration_digest(first_snapshot) == configuration_digest(second_snapshot)
    first_result = run_research_pipeline(first)
    second_result = run_research_pipeline(second)
    assert first_result.run_id != second_result.run_id
    assert first_result.research_content_digest == second_result.research_content_digest
    assert first_result.configuration_digest == second_result.configuration_digest
    assert first_result.screening_results == second_result.screening_results
    assert first_result.selected_pair == second_result.selected_pair
    assert first_result.backtest_result is not None
    assert second_result.backtest_result is not None
    pd.testing.assert_frame_equal(
        first_result.backtest_result.accounting,
        second_result.backtest_result.accounting,
    )
    assert first_result.walk_forward_result is not None
    assert second_result.walk_forward_result is not None
    pd.testing.assert_series_equal(
        first_result.walk_forward_result.calendar_oos_returns,
        second_result.walk_forward_result.calendar_oos_returns,
    )


def test_result_is_frozen_and_owns_research_arrays(
    real_pipeline_case: tuple[ResearchExperimentRequest, ResearchExperimentResult],
) -> None:
    request, result = real_pipeline_case
    with pytest.raises(FrozenInstanceError):
        result.experiment_name = "changed"  # type: ignore[misc]
    assert result.hedge_estimates is not None
    before = request.prices.copy(deep=True)
    result.hedge_estimates.iloc[-1, 0] = -123.0
    pd.testing.assert_frame_equal(request.prices, before)


def test_content_identity_changes_with_prices_not_descriptive_metadata() -> None:
    baseline = _request()
    changed_prices = _request()
    changed_prices.prices.iloc[-1, 0] *= 1.01
    changed_metadata = replace(baseline, metadata={"owner": "different"})

    baseline_result = run_research_pipeline(baseline)
    price_result = run_research_pipeline(changed_prices)
    metadata_result = run_research_pipeline(changed_metadata)

    assert baseline_result.research_content_digest != price_result.research_content_digest
    assert (
        baseline_result.research_content_digest
        == metadata_result.research_content_digest
    )
    assert len({baseline_result.run_id, price_result.run_id, metadata_result.run_id}) == 3


def test_snapshot_records_effective_policies_versions_and_sample_identity() -> None:
    request = _request(run_robustness=True, run_validation=True)
    snapshot = build_configuration_snapshot(request)

    assert snapshot["versions"] == {
        "research_pipeline": RESEARCH_PIPELINE_VERSION,
        "configuration_snapshot": CONFIGURATION_SNAPSHOT_VERSION,
        "experiment_schema": EXPERIMENT_SCHEMA_VERSION,
    }
    assert snapshot["data_sample"]["rows"] == len(request.prices)
    assert snapshot["data_sample"]["columns"] == len(request.prices.columns)
    assert snapshot["data_sample"]["price_content_digest"] == price_content_digest(
        request.prices
    )
    signal = snapshot["effective_behavior"]["signal"]
    assert signal["missing_policy"] == "hold"
    assert signal["zscore_ddof"] == 1
    walk_forward = snapshot["effective_behavior"]["walk_forward"]
    assert walk_forward["step_size"] == request.config.walk_forward.trading_days
    assert walk_forward["expanding"] is False
    assert walk_forward["independent_of_full_sample_selection"] is True
    assert walk_forward["inactive_capital_policy"] == (
        "zero_return_cash_for_no_selection_rows"
    )
    diagnostic = snapshot["effective_behavior"]["diagnostic_execution"]
    assert "entry_and_rebalance_unavailable" in diagnostic[
        "non_positive_beta_policy"
    ]
    assert snapshot["effective_behavior"]["robustness"][
        "headline_metric_basis"
    ] == "structural_common_scheduled_horizon"
    assert snapshot["effective_behavior"]["validation"][
        "effective_random_seed"
    ] == 91
    assert "strategy.target_annual_vol" in snapshot["configured_but_not_enforced"]


def test_price_digest_ignores_datetime_frequency_and_is_cross_process_stable() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", name="Date")
    prices = pd.DataFrame(
        {"BBB": [2.0, 2.1, 2.2, 2.3], "AAA": [1.0, 1.1, 1.2, 1.3]},
        index=index,
    )
    without_frequency = prices.copy(deep=True)
    without_frequency.index = pd.DatetimeIndex(
        without_frequency.index.to_numpy(), name="Date", freq=None
    )
    expected = price_content_digest(prices)

    assert price_content_digest(without_frequency) == expected
    config = _config({"group": ["AAA", "BBB"]})
    with_frequency_request = ResearchExperimentRequest("freq", prices, config)
    without_frequency_request = ResearchExperimentRequest(
        "no-freq", without_frequency, config, metadata={"display": "different"}
    )
    assert research_content_digest(
        build_configuration_snapshot(with_frequency_request)
    ) == research_content_digest(build_configuration_snapshot(without_frequency_request))
    script = "\n".join(
        (
            "import pandas as pd",
            "from pairs_trading.pipeline import price_content_digest",
            "idx=pd.date_range('2024-01-01',periods=4,freq='D',name='Date')",
            "p=pd.DataFrame({'BBB':[2.0,2.1,2.2,2.3],'AAA':[1.0,1.1,1.2,1.3]},index=idx)",
            "print(price_content_digest(p))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == expected


def test_content_digest_excludes_name_metadata_and_includes_effective_settings() -> None:
    baseline = _request()
    renamed = replace(baseline, experiment_name="display-only")
    remetadata = replace(baseline, metadata={"description": "display-only"})
    resized = replace(baseline, target_gross_notional=25_000.0)

    baseline_digest = research_content_digest(build_configuration_snapshot(baseline))
    assert research_content_digest(build_configuration_snapshot(renamed)) == baseline_digest
    assert research_content_digest(build_configuration_snapshot(remetadata)) == baseline_digest
    assert research_content_digest(build_configuration_snapshot(resized)) != baseline_digest


def test_content_digest_changes_with_mask_and_pipeline_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    baseline = research_content_digest(build_configuration_snapshot(request))
    masked_prices = request.prices.copy(deep=True)
    mask = pd.DataFrame(
        True,
        index=masked_prices.index,
        columns=masked_prices.columns,
        dtype=bool,
    )
    mask.iloc[-1, 0] = False
    masked_prices.attrs[OBSERVED_PRICE_MASK_ATTR] = mask
    changed_mask = replace(request, prices=masked_prices)
    assert research_content_digest(build_configuration_snapshot(changed_mask)) != baseline

    monkeypatch.setattr(pipeline_module, "RESEARCH_PIPELINE_VERSION", "10A.test")
    assert research_content_digest(build_configuration_snapshot(request)) != baseline


def test_price_digest_rejects_arbitrary_unstable_provenance_objects() -> None:
    request = _request()
    prices = request.prices.copy(deep=True)
    prices.attrs["valuation_policy"] = object()
    with pytest.raises(TypeError, match="valuation_policy"):
        price_content_digest(prices)
    prices = request.prices.copy(deep=True)
    prices.index.name = object()
    with pytest.raises(TypeError, match="canonical representation"):
        price_content_digest(prices)


def test_non_positive_beta_preserves_signal_but_blocks_execution_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    original_statistics = pipeline_module.rolling_ols_spread
    original_backtest = pipeline_module.run_pair_backtest
    captured: dict[str, pd.Series] = {}

    def sign_changing_statistics(
        y: pd.Series, x: pd.Series, lookback: int
    ) -> pd.DataFrame:
        estimates = original_statistics(y, x, lookback)
        estimates.iloc[-80:-70, estimates.columns.get_loc("beta")] = -0.25
        return estimates

    def capture_backtest(*args: Any, **kwargs: Any) -> BacktestResult:
        captured["execution_beta"] = args[2].copy(deep=True)
        captured["zscore"] = kwargs["zscore"].copy(deep=True)
        return original_backtest(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module, "rolling_ols_spread", sign_changing_statistics
    )
    monkeypatch.setattr(pipeline_module, "run_pair_backtest", capture_backtest)
    result = run_research_pipeline(request)

    assert result.hedge_estimates is not None
    assert result.zscore is not None
    assert isinstance(result.diagnostic_execution_coverage, DiagnosticExecutionCoverage)
    negative_rows = result.hedge_estimates["beta"].lt(0.0)
    assert negative_rows.any()
    assert result.zscore.loc[negative_rows].notna().any()
    assert captured["execution_beta"].loc[negative_rows].isna().all()
    assert (
        result.diagnostic_execution_coverage.non_positive_execution_beta_rows
        == int(negative_rows.sum())
    )
    assert not result.backtest_result.positions["rebalance"].any()  # type: ignore[union-attr]


def test_missing_execution_beta_blocks_entry_but_does_not_block_existing_close() -> None:
    index = pd.RangeIndex(7)
    price_y = pd.Series([100.0] * 7, index=index)
    price_x = pd.Series([50.0] * 7, index=index)
    blocked = run_pair_backtest(
        price_y,
        price_x,
        pd.Series([1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0], index=index),
        10_000.0,
        zscore=pd.Series([0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0], index=index),
        hedge_ratio_lag=1,
        execution_lag=1,
        force_liquidation=False,
    )
    assert blocked.positions["executed_state"].eq("FLAT").all()

    closing = run_pair_backtest(
        price_y,
        price_x,
        pd.Series([1.0, 1.0, 1.0, np.nan, np.nan, 1.0, 1.0], index=index),
        10_000.0,
        zscore=pd.Series([0.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0], index=index),
        hedge_ratio_lag=1,
        execution_lag=1,
        force_liquidation=False,
    )
    assert closing.positions["executed_state"].iat[2] == "SHORT_SPREAD"
    assert closing.positions["executed_state"].iat[4] == "FLAT"


def test_real_walk_forward_selection_survives_mocked_full_sample_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_module, "screen_pairs", lambda *args, **kwargs: (_rejected_result(),)
    )
    result = run_research_pipeline(_request())

    assert result.selected_pair is None
    assert result.analytics_stage is PipelineStageStatus.UNAVAILABLE
    assert result.walk_forward_result is not None
    assert result.walk_forward_result.completed_fold_count > 0
    assert result.walk_forward_result.selected_oos_observations > 0
    validate_research_experiment_result(result)


def test_validator_rejects_unknown_status_and_stage_values(
    real_pipeline_case: tuple[ResearchExperimentRequest, ResearchExperimentResult],
) -> None:
    _, result = real_pipeline_case
    with pytest.raises(TypeError, match="ResearchPipelineStatus"):
        validate_research_experiment_result(replace(result, status="UNKNOWN"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PipelineStageStatus"):
        validate_research_experiment_result(
            replace(result, analytics_stage="UNKNOWN")  # type: ignore[arg-type]
        )


def test_pipeline_public_api_contains_no_optimizer_or_live_execution() -> None:
    public = set(pipeline_module.__all__)
    assert not any("optim" in name.casefold() for name in public)
    assert not any(
        term in name.casefold()
        for name in public
        for term in ("broker", "alert", "live", "order", "tracker")
    )
