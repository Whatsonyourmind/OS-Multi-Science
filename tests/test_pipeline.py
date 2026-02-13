"""Tests for the orchestrator pipeline."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from framework.aesc_profiler import (
    EPIDEMIC_NETWORK,
    FINANCIAL_ENERGY_SYSTEM,
    SUPPLY_CHAIN,
    create_profile,
)
from framework.config import OSMultiScienceConfig
from framework.types import (
    DecisionAction,
    ICMResult,
    MethodProfile,
    RouterSelection,
    SystemProfile,
)
from orchestrator.pipeline import Pipeline, PipelineResult, StepResult


# ===================================================================
# Helpers
# ===================================================================

N_SAMPLES = 50


def _make_profile() -> SystemProfile:
    """Return a lightweight profile for testing."""
    return create_profile("test-system", "unit test system")


def _dummy_executor(X, profile):
    """Return random predictions seeded from X."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(len(X))


def _dummy_executor_b(X, profile):
    """A second dummy executor with a different seed."""
    rng = np.random.default_rng(99)
    return rng.standard_normal(len(X))


def _dummy_executor_c(X, profile):
    """A third dummy executor."""
    rng = np.random.default_rng(7)
    return rng.standard_normal(len(X))


def _failing_executor(X, profile):
    """An executor that always raises."""
    raise RuntimeError("model exploded")


def _slow_executor(X, profile):
    """An executor that sleeps briefly to test parallelism."""
    time.sleep(0.05)
    return np.ones(len(X))


# ===================================================================
# Construction and configuration
# ===================================================================


class TestPipelineConstruction:
    def test_default_config(self):
        p = Pipeline()
        assert p.config is not None
        assert p.config.random_seed == 42

    def test_custom_config(self):
        cfg = OSMultiScienceConfig(random_seed=123)
        p = Pipeline(config=cfg)
        assert p.config.random_seed == 123

    def test_empty_model_registry(self):
        p = Pipeline()
        assert len(p._model_executors) == 0

    def test_empty_hooks(self):
        p = Pipeline()
        assert len(p._hooks) == 0


# ===================================================================
# Model registration
# ===================================================================


class TestModelRegistration:
    def test_register_single_model(self):
        p = Pipeline()
        p.register_model("m1", _dummy_executor)
        assert "m1" in p._model_executors

    def test_register_multiple_models(self):
        p = Pipeline()
        p.register_model("m1", _dummy_executor)
        p.register_model("m2", _dummy_executor_b)
        assert len(p._model_executors) == 2

    def test_register_overwrite(self):
        p = Pipeline()
        p.register_model("m1", _dummy_executor)
        p.register_model("m1", _dummy_executor_b)
        assert p._model_executors["m1"] is _dummy_executor_b


# ===================================================================
# Hook registration
# ===================================================================


class TestHooks:
    def test_add_post_hook(self):
        p = Pipeline()
        p.add_hook("route", lambda sr: None, when="post")
        assert len(p._hooks.get("post_route", [])) == 1

    def test_add_pre_hook(self):
        p = Pipeline()
        p.add_hook("icm", lambda name: None, when="pre")
        assert len(p._hooks.get("pre_icm", [])) == 1

    def test_hooks_invoked_on_run_step(self):
        p = Pipeline()
        calls: list[str] = []
        p.add_hook("test_step", lambda name: calls.append("pre"), when="pre")
        p.add_hook("test_step", lambda sr: calls.append("post"), when="post")

        sr = p.run_step("test_step", lambda: 42)
        assert sr.status == "success"
        assert sr.output == 42
        assert calls == ["pre", "post"]

    def test_failing_hook_does_not_crash_step(self):
        p = Pipeline()

        def bad_hook(sr):
            raise ValueError("hook error")

        p.add_hook("test_step", bad_hook, when="post")
        sr = p.run_step("test_step", lambda: "ok")
        assert sr.status == "success"
        assert sr.output == "ok"


# ===================================================================
# run_step
# ===================================================================


class TestRunStep:
    def test_success(self):
        p = Pipeline()
        sr = p.run_step("add", lambda a, b: a + b, 2, 3)
        assert sr.step_name == "add"
        assert sr.status == "success"
        assert sr.output == 5
        assert sr.duration_seconds >= 0
        assert sr.error is None

    def test_failure(self):
        p = Pipeline()
        sr = p.run_step("boom", _failing_executor, [], None)
        assert sr.status == "failed"
        assert sr.error is not None
        assert "exploded" in sr.error

    def test_timing(self):
        p = Pipeline()
        sr = p.run_step("sleep", lambda: time.sleep(0.05) or "done")
        assert sr.duration_seconds >= 0.04


# ===================================================================
# Parallel model execution
# ===================================================================


class TestParallelExecution:
    def test_parallel_runs_multiple_models(self):
        p = Pipeline()
        p.register_model("m1", _dummy_executor)
        p.register_model("m2", _dummy_executor_b)

        X = np.zeros(N_SAMPLES)
        profile = _make_profile()
        # Build a fake MethodProfile list whose names match
        methods = [
            MethodProfile(name="m1", family=__import__("framework.types", fromlist=["EpistemicFamily"]).EpistemicFamily.BASELINE, description=""),
            MethodProfile(name="m2", family=__import__("framework.types", fromlist=["EpistemicFamily"]).EpistemicFamily.BASELINE, description=""),
        ]
        results = p._execute_models_parallel(methods, X, profile)
        assert "m1" in results
        assert "m2" in results
        assert len(results["m1"]) == N_SAMPLES

    def test_parallel_handles_failure(self):
        p = Pipeline()
        p.register_model("good", _dummy_executor)
        p.register_model("bad", _failing_executor)

        from framework.types import EpistemicFamily

        methods = [
            MethodProfile(name="good", family=EpistemicFamily.BASELINE, description=""),
            MethodProfile(name="bad", family=EpistemicFamily.BASELINE, description=""),
        ]
        X = np.zeros(N_SAMPLES)
        results = p._execute_models_parallel(methods, X, _make_profile())
        assert "good" in results
        assert "bad" not in results

    def test_parallelism_faster_than_sequential(self):
        """Two slow models should finish in ~1x sleep time, not 2x."""
        p = Pipeline()
        p.register_model("s1", _slow_executor)
        p.register_model("s2", _slow_executor)

        from framework.types import EpistemicFamily

        methods = [
            MethodProfile(name="s1", family=EpistemicFamily.BASELINE, description=""),
            MethodProfile(name="s2", family=EpistemicFamily.BASELINE, description=""),
        ]
        X = np.zeros(10)
        t0 = time.monotonic()
        results = p._execute_models_parallel(methods, X, _make_profile(), max_workers=4)
        elapsed = time.monotonic() - t0
        assert len(results) == 2
        # Sequential would be >= 0.10; parallel should be < 0.09
        assert elapsed < 0.09

    def test_fallback_to_all_executors_when_none_match(self):
        """When none of the selected method names match registered names,
        the pipeline falls back to running all registered executors."""
        p = Pipeline()
        p.register_model("custom_a", _dummy_executor)

        from framework.types import EpistemicFamily

        methods = [
            MethodProfile(name="NoMatch", family=EpistemicFamily.BASELINE, description=""),
        ]
        X = np.zeros(N_SAMPLES)
        results = p._execute_models_parallel(methods, X, _make_profile())
        assert "custom_a" in results


# ===================================================================
# Full pipeline run
# ===================================================================


class TestFullPipelineRun:
    def _build_pipeline(self) -> Pipeline:
        p = Pipeline()
        p.register_model("m1", _dummy_executor)
        p.register_model("m2", _dummy_executor_b)
        p.register_model("m3", _dummy_executor_c)
        return p

    def test_run_without_y_true(self):
        p = self._build_pipeline()
        profile = _make_profile()
        X = np.zeros(N_SAMPLES)

        result = p.run(profile, X)
        assert isinstance(result, PipelineResult)
        assert result.system_profile is profile
        assert result.selected_kit is not None
        assert result.predictions is not None
        assert result.icm_result is not None
        # CRC and anti_spurious should be skipped
        assert result.crc_decision is None
        assert result.re_score is None
        assert result.anti_spurious is None
        assert len(result.decision_cards) > 0
        assert result.total_duration > 0

    def test_run_with_y_true(self):
        p = self._build_pipeline()
        profile = _make_profile()
        X = np.zeros(N_SAMPLES)
        y_true = np.random.default_rng(0).standard_normal(N_SAMPLES)

        result = p.run(profile, X, y_true=y_true)
        assert result.icm_result is not None
        assert result.re_score is not None
        assert result.crc_decision is not None
        assert isinstance(result.crc_decision, DecisionAction)

    def test_is_success_true_on_good_run(self):
        p = self._build_pipeline()
        result = p.run(_make_profile(), np.zeros(N_SAMPLES))
        assert result.is_success is True

    def test_step_results_populated(self):
        p = self._build_pipeline()
        result = p.run(_make_profile(), np.zeros(N_SAMPLES))
        step_names = [s.step_name for s in result.step_results]
        assert "route" in step_names
        assert "execute" in step_names
        assert "icm" in step_names

    def test_all_steps_timed(self):
        p = self._build_pipeline()
        result = p.run(_make_profile(), np.zeros(N_SAMPLES))
        for sr in result.step_results:
            assert sr.duration_seconds >= 0


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    def test_failing_model_does_not_crash_pipeline(self):
        p = Pipeline()
        p.register_model("bad", _failing_executor)
        p.register_model("good", _dummy_executor)

        result = p.run(_make_profile(), np.zeros(N_SAMPLES))
        # The pipeline should still run to completion
        assert result.predictions is not None
        assert "good" in result.predictions
        assert "bad" not in result.predictions

    def test_all_models_fail_marks_execute_failed(self):
        p = Pipeline()
        p.register_model("bad1", _failing_executor)
        p.register_model("bad2", _failing_executor)

        result = p.run(_make_profile(), np.zeros(N_SAMPLES))
        # Execute step should be marked failed (empty predictions)
        exec_step = next(
            s for s in result.step_results if s.step_name == "execute"
        )
        assert exec_step.status == "failed"
        # Downstream steps should not be present beyond execute
        assert result.icm_result is None

    def test_is_success_false_on_failure(self):
        p = Pipeline()
        p.register_model("bad", _failing_executor)

        result = p.run(_make_profile(), np.zeros(N_SAMPLES))
        assert result.is_success is False


# ===================================================================
# skip_anti_spurious flag
# ===================================================================


class TestSkipAntiSpurious:
    def test_skip_anti_spurious_true(self):
        p = Pipeline()
        p.register_model("m1", _dummy_executor)
        p.register_model("m2", _dummy_executor_b)

        X = np.zeros(N_SAMPLES)
        y = np.random.default_rng(1).standard_normal(N_SAMPLES)

        result = p.run(
            _make_profile(), X, y_true=y, skip_anti_spurious=True
        )
        anti_step = next(
            s for s in result.step_results if s.step_name == "anti_spurious"
        )
        assert anti_step.status == "skipped"
        assert result.anti_spurious is None

    def test_skip_anti_spurious_false_with_y_true(self):
        p = Pipeline()
        p.register_model("m1", _dummy_executor)
        p.register_model("m2", _dummy_executor_b)
        p.register_model("m3", _dummy_executor_c)

        X = np.zeros(N_SAMPLES)
        y = np.random.default_rng(1).standard_normal(N_SAMPLES)

        # Use fewer permutations for speed
        cfg = OSMultiScienceConfig()
        cfg.anti_spurious.n_permutations = 10
        cfg.anti_spurious.n_negative_controls = 5
        p.config = cfg

        result = p.run(
            _make_profile(), X, y_true=y, skip_anti_spurious=False
        )
        anti_step = next(
            s for s in result.step_results if s.step_name == "anti_spurious"
        )
        assert anti_step.status == "success"
        assert result.anti_spurious is not None


# ===================================================================
# PipelineResult properties
# ===================================================================


class TestPipelineResultProperties:
    def test_is_success_all_good(self):
        pr = PipelineResult(
            step_results=[
                StepResult("a", "success", None, 0.1),
                StepResult("b", "success", None, 0.2),
                StepResult("c", "skipped", None, 0.0),
            ]
        )
        assert pr.is_success is True

    def test_is_success_one_failed(self):
        pr = PipelineResult(
            step_results=[
                StepResult("a", "success", None, 0.1),
                StepResult("b", "failed", None, 0.2, error="oops"),
            ]
        )
        assert pr.is_success is False

    def test_is_success_empty(self):
        pr = PipelineResult()
        # No steps => vacuously true
        assert pr.is_success is True

    def test_default_fields(self):
        pr = PipelineResult()
        assert pr.system_profile is None
        assert pr.selected_kit is None
        assert pr.predictions is None
        assert pr.icm_result is None
        assert pr.crc_decision is None
        assert pr.re_score is None
        assert pr.anti_spurious is None
        assert pr.decision_cards == []
        assert pr.step_results == []
        assert pr.total_duration == 0.0
