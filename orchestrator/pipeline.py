"""Orchestrator pipeline for OS Multi-Science.

Executes the full analysis flow: Profile -> Route -> Execute -> ICM -> CRC -> Decide.
Each step is timed, logged, and wrapped in error handling so that a single
failure does not crash the entire pipeline.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from framework.anti_spurious import generate_anti_spurious_report
from framework.catalog import get_catalog
from framework.config import OSMultiScienceConfig
from framework.crc_gating import (
    compute_re,
    conformalize,
    decision_gate,
    fit_isotonic,
)
from framework.icm import compute_icm_from_predictions
from framework.router import generate_decision_cards, select_kit
from framework.types import (
    AntiSpuriousReport,
    DecisionAction,
    DecisionCard,
    ICMResult,
    MethodProfile,
    RouterSelection,
    SystemProfile,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Result data classes
# ===================================================================


@dataclass
class StepResult:
    """Result from a single pipeline step."""

    step_name: str
    status: str  # "success", "failed", "skipped"
    output: Any
    duration_seconds: float
    error: str | None = None


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""

    system_profile: SystemProfile | None = None
    selected_kit: RouterSelection | None = None
    predictions: dict[str, Any] | None = None
    icm_result: ICMResult | None = None
    crc_decision: DecisionAction | None = None
    re_score: float | None = None
    anti_spurious: AntiSpuriousReport | None = None
    decision_cards: list[dict] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def is_success(self) -> bool:
        """True when no step has failed."""
        return all(s.status != "failed" for s in self.step_results)


# ===================================================================
# Pipeline
# ===================================================================


class Pipeline:
    """Main orchestration pipeline.

    Flow
    ----
    1. **Route** -- select a method kit via the Router.
    2. **Execute** -- run registered model executors in parallel.
    3. **ICM** -- compute the Index of Convergence Multi-epistemic.
    4. **CRC** -- Conformal Risk Control gating (requires ``y_true``).
    5. **Anti-spurious** -- anti-spurious convergence check (optional).
    6. **Decision cards** -- generate per-method decision cards.
    """

    def __init__(self, config: OSMultiScienceConfig | None = None) -> None:
        self.config = config or OSMultiScienceConfig()
        self._model_executors: dict[str, Callable] = {}
        self._hooks: dict[str, list[Callable]] = {}

    # ---------------------------------------------------------------
    # Registration helpers
    # ---------------------------------------------------------------

    def register_model(self, name: str, executor: Callable) -> None:
        """Register a model executor function.

        Parameters
        ----------
        name : str
            A unique identifier for the model (should match the method
            name from the catalog when possible).
        executor : callable
            Signature ``(X, profile) -> predictions_array``.
        """
        self._model_executors[name] = executor

    def add_hook(self, step: str, hook: Callable, when: str = "post") -> None:
        """Add a pre- or post-hook to a named pipeline step.

        Parameters
        ----------
        step : str
            The step name (e.g. ``"route"``, ``"execute"``, ``"icm"``).
        hook : callable
            Will be called with the ``StepResult`` (post) or the step
            name string (pre).
        when : str
            ``"pre"`` or ``"post"``.
        """
        key = f"{when}_{step}"
        self._hooks.setdefault(key, []).append(hook)

    # ---------------------------------------------------------------
    # Public run API
    # ---------------------------------------------------------------

    def run(
        self,
        system_profile: SystemProfile,
        X: Any,
        y_true: Any | None = None,
        features: Any | None = None,
        skip_anti_spurious: bool = False,
    ) -> PipelineResult:
        """Execute the full pipeline.

        Parameters
        ----------
        system_profile : SystemProfile
            The AESC system profile describing the problem domain.
        X : array-like
            Input data passed to each registered model executor.
        y_true : array-like, optional
            Ground-truth labels.  Required for CRC gating and
            anti-spurious checks.
        features : array-like, optional
            Feature matrix for anti-spurious analysis.
        skip_anti_spurious : bool
            When ``True`` the anti-spurious step is skipped even if
            ``y_true`` is provided.

        Returns
        -------
        PipelineResult
        """
        t0 = time.monotonic()
        result = PipelineResult(system_profile=system_profile)

        # --- Step 1: Route ---------------------------------------------------
        step = self.run_step(
            "route",
            self._step_route,
            system_profile,
        )
        result.step_results.append(step)
        if step.status == "failed":
            result.total_duration = time.monotonic() - t0
            return result
        result.selected_kit = step.output

        # --- Step 2: Execute models ------------------------------------------
        selected_methods = result.selected_kit.selected_methods
        step = self.run_step(
            "execute",
            self._execute_models_parallel,
            selected_methods,
            X,
            system_profile,
        )
        result.step_results.append(step)
        if step.status == "failed":
            result.total_duration = time.monotonic() - t0
            return result
        result.predictions = step.output

        # If no predictions succeeded, skip downstream
        if not result.predictions:
            step_empty = StepResult(
                step_name="execute",
                status="failed",
                output=None,
                duration_seconds=0.0,
                error="No model executors produced predictions.",
            )
            # Replace the last step result
            result.step_results[-1] = step_empty
            result.total_duration = time.monotonic() - t0
            return result

        # --- Step 3: ICM ------------------------------------------------------
        step = self.run_step(
            "icm",
            self._step_icm,
            result.predictions,
        )
        result.step_results.append(step)
        if step.status == "failed":
            result.total_duration = time.monotonic() - t0
            return result
        result.icm_result = step.output

        # --- Step 4: CRC gating (only if y_true provided) ---------------------
        if y_true is not None:
            step = self.run_step(
                "crc",
                self._step_crc,
                result.icm_result,
                result.predictions,
                y_true,
            )
            result.step_results.append(step)
            if step.status == "success":
                crc_out = step.output
                result.re_score = crc_out["re_score"]
                result.crc_decision = crc_out["decision"]
        else:
            result.step_results.append(
                StepResult(
                    step_name="crc",
                    status="skipped",
                    output=None,
                    duration_seconds=0.0,
                )
            )

        # --- Step 5: Anti-spurious check --------------------------------------
        if y_true is not None and not skip_anti_spurious:
            step = self.run_step(
                "anti_spurious",
                self._step_anti_spurious,
                result.predictions,
                y_true,
                features,
            )
            result.step_results.append(step)
            if step.status == "success":
                result.anti_spurious = step.output
        else:
            result.step_results.append(
                StepResult(
                    step_name="anti_spurious",
                    status="skipped",
                    output=None,
                    duration_seconds=0.0,
                )
            )

        # --- Step 6: Decision cards -------------------------------------------
        step = self.run_step(
            "decision_cards",
            self._step_decision_cards,
            result.selected_kit,
            system_profile,
        )
        result.step_results.append(step)
        if step.status == "success":
            result.decision_cards = step.output

        result.total_duration = time.monotonic() - t0
        return result

    # ---------------------------------------------------------------
    # Step runner with timing, hooks, and error handling
    # ---------------------------------------------------------------

    def run_step(
        self, step_name: str, fn: Callable, *args: Any, **kwargs: Any
    ) -> StepResult:
        """Run a single step with timing, hooks, and error handling.

        Parameters
        ----------
        step_name : str
            Human-readable name for the step.
        fn : callable
            The function to execute.
        *args, **kwargs
            Forwarded to *fn*.

        Returns
        -------
        StepResult
        """
        # Pre-hooks
        for hook in self._hooks.get(f"pre_{step_name}", []):
            try:
                hook(step_name)
            except Exception:
                logger.warning("Pre-hook for %s failed", step_name, exc_info=True)

        t0 = time.monotonic()
        try:
            output = fn(*args, **kwargs)
            duration = time.monotonic() - t0
            step_result = StepResult(
                step_name=step_name,
                status="success",
                output=output,
                duration_seconds=duration,
            )
            logger.info(
                "Step '%s' completed in %.4f s", step_name, duration
            )
        except Exception as exc:
            duration = time.monotonic() - t0
            step_result = StepResult(
                step_name=step_name,
                status="failed",
                output=None,
                duration_seconds=duration,
                error=str(exc),
            )
            logger.error(
                "Step '%s' failed after %.4f s: %s",
                step_name,
                duration,
                exc,
                exc_info=True,
            )

        # Post-hooks
        for hook in self._hooks.get(f"post_{step_name}", []):
            try:
                hook(step_result)
            except Exception:
                logger.warning("Post-hook for %s failed", step_name, exc_info=True)

        return step_result

    # ---------------------------------------------------------------
    # Parallel model execution
    # ---------------------------------------------------------------

    def _execute_models_parallel(
        self,
        selected_methods: list[MethodProfile],
        X: Any,
        profile: SystemProfile,
        max_workers: int = 4,
    ) -> dict[str, Any]:
        """Execute registered models in parallel via ThreadPoolExecutor.

        Only models that (a) are in the selected kit AND (b) have a
        registered executor will be executed.  Models whose executors
        raise are logged but do not prevent other models from running.

        Returns
        -------
        dict[str, array]
            ``{model_name: predictions_array}`` for each model that
            succeeded.
        """
        selected_names = {m.name for m in selected_methods}
        to_run = {
            name: fn
            for name, fn in self._model_executors.items()
            if name in selected_names
        }

        # If no registered executors match any selected method, run all
        # registered executors (the user may have registered under
        # custom names).
        if not to_run:
            to_run = dict(self._model_executors)

        results: dict[str, Any] = {}

        if not to_run:
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_name = {
                pool.submit(fn, X, profile): name
                for name, fn in to_run.items()
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.warning(
                        "Model '%s' raised: %s", name, exc, exc_info=True
                    )

        return results

    # ---------------------------------------------------------------
    # Individual step implementations
    # ---------------------------------------------------------------

    def _step_route(self, profile: SystemProfile) -> RouterSelection:
        catalog = get_catalog()
        return select_kit(profile, catalog=catalog, config=self.config.router)

    def _step_icm(self, predictions: dict[str, Any]) -> ICMResult:
        result = compute_icm_from_predictions(
            predictions, config=self.config.icm, distance_fn="wasserstein",
        )
        # Guard against NaN scores (can occur with degenerate inputs)
        if not np.isfinite(result.icm_score):
            result.icm_score = 0.5
        return result

    def _step_crc(
        self,
        icm_result: ICMResult,
        predictions: dict[str, Any],
        y_true: Any,
    ) -> dict[str, Any]:
        """Run CRC gating.

        We build a simple loss vector from prediction errors and use it
        to fit the isotonic regression + conformal calibration pipeline.
        """
        y_true = np.asarray(y_true, dtype=float)

        # Build per-sample ICM proxies and losses from the predictions.
        pred_arrays = np.array(
            [np.asarray(v, dtype=float) for v in predictions.values()]
        )
        ensemble_mean = np.mean(pred_arrays, axis=0)

        # Loss = squared error of ensemble mean
        losses = (ensemble_mean - y_true) ** 2
        # Ensure losses are finite
        losses = np.where(np.isfinite(losses), losses, 0.0)

        # Use ensemble spread as a proxy for per-sample ICM.
        # Lower spread => higher convergence.
        ensemble_std = np.std(pred_arrays, axis=0)
        max_std = float(np.nanmax(ensemble_std))
        if max_std < 1e-12:
            # All models agree perfectly at every point -- uniform high
            # convergence with small jitter to give isotonic non-constant
            # input.
            rng = np.random.default_rng(self.config.random_seed)
            c_proxy = 0.9 + 0.1 * rng.random(size=ensemble_std.shape)
        else:
            c_proxy = 1.0 - ensemble_std / max_std
            c_proxy = np.where(np.isfinite(c_proxy), c_proxy, 0.5)
            # Add tiny jitter so isotonic regression gets non-constant X
            rng = np.random.default_rng(self.config.random_seed)
            c_proxy = c_proxy + rng.uniform(-1e-6, 1e-6, size=c_proxy.shape)

        c_proxy = np.clip(c_proxy, 0.0, 1.0)

        # Split into fit / calibration  (need at least 2 points each)
        n = len(losses)
        split = max(n // 2, 2)
        g = fit_isotonic(c_proxy[:split], losses[:split])
        g_alpha = conformalize(
            g, c_proxy[split:], losses[split:], alpha=self.config.crc.alpha
        )

        re = compute_re(icm_result.icm_score, g_alpha)
        decision = decision_gate(
            icm_result.icm_score, re, self.config.crc
        )

        return {
            "re_score": re,
            "decision": decision,
        }

    def _step_anti_spurious(
        self,
        predictions: dict[str, Any],
        y_true: Any,
        features: Any | None,
    ) -> AntiSpuriousReport:
        labels = np.asarray(y_true, dtype=float)
        if features is None:
            # Create a dummy feature matrix if none provided
            n = len(labels)
            features_arr = np.arange(n, dtype=float).reshape(-1, 1)
        else:
            features_arr = np.asarray(features, dtype=float)

        # Build predictions dict with arrays of matching length
        pred_dict: dict[str, np.ndarray] = {}
        for name, preds in predictions.items():
            pred_dict[name] = np.asarray(preds, dtype=float)

        def _icm_fn(p: dict[str, np.ndarray]) -> float:
            res = compute_icm_from_predictions(p, config=self.config.icm)
            return res.icm_score

        return generate_anti_spurious_report(
            pred_dict,
            labels,
            features_arr,
            config=self.config.anti_spurious,
            icm_fn=_icm_fn,
        )

    def _step_decision_cards(
        self,
        selection: RouterSelection,
        profile: SystemProfile,
    ) -> list[dict]:
        return generate_decision_cards(selection, profile)
