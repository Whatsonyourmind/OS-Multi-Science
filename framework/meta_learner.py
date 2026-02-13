"""Meta-learner for ICM weight optimization (Q8).

Finds optimal component weights (w_A, w_D, w_U, w_C, lambda) that minimize
epistemic risk subject to coverage and monotonicity constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import qmc, spearmanr
from sklearn.metrics import roc_auc_score

from framework.config import ICMConfig, CRCConfig
from framework.types import ICMComponents, ICMResult
from framework.icm import compute_icm, compute_icm_from_predictions
from framework.crc_gating import fit_isotonic, conformalize


# Weight-space bounds: (lower, upper) for each parameter
_WEIGHT_BOUNDS = {
    "w_A": (0.05, 0.50),
    "w_D": (0.05, 0.50),
    "w_U": (0.05, 0.50),
    "w_C": (0.05, 0.50),
    "lam": (0.05, 0.30),
}

_WEIGHT_NAMES = list(_WEIGHT_BOUNDS.keys())

# Composite objective sub-weights
_OBJ_MONOTONICITY = 0.4
_OBJ_DISCRIMINATION = 0.3
_OBJ_COVERAGE = 0.3


def _weights_to_dict(x: NDArray) -> dict[str, float]:
    """Convert a 5-element vector to a weight dict."""
    return {name: float(x[i]) for i, name in enumerate(_WEIGHT_NAMES)}


def _dict_to_vector(w: dict[str, float]) -> NDArray:
    """Convert a weight dict to a 5-element vector."""
    return np.array([w[name] for name in _WEIGHT_NAMES], dtype=np.float64)


def _make_config_from_weights(weights: dict[str, float]) -> ICMConfig:
    """Build an ICMConfig from a weight dict, keeping defaults for other fields."""
    return ICMConfig(
        w_A=weights["w_A"],
        w_D=weights["w_D"],
        w_U=weights["w_U"],
        w_C=weights["w_C"],
        lam=weights["lam"],
    )


def _compute_scenario_icm(
    scenario: dict,
    config: ICMConfig,
) -> float:
    """Compute an ICM score for one scenario using the given config."""
    result = compute_icm_from_predictions(
        predictions_dict=scenario["predictions_dict"],
        config=config,
        intervals=scenario.get("intervals"),
        signs=scenario.get("signs"),
        pre_scores=scenario.get("pre_scores"),
        post_scores=scenario.get("post_scores"),
        residuals=scenario.get("residuals"),
        features=scenario.get("features"),
        gradients=scenario.get("gradients"),
    )
    return result.icm_score


class MetaLearner:
    """Optimizes ICM configuration weights from historical data.

    Approach:
    1. Generate multiple scenarios with known ground truth
    2. For each weight configuration, compute ICM and evaluate:
       - Monotonicity: Spearman correlation between ICM and -loss
       - Coverage: conformal coverage at given alpha
       - Discrimination: AUC between high/low convergence scenarios
    3. Find weights maximizing a composite objective
    """

    def __init__(self, config: ICMConfig | None = None):
        self.config = config or ICMConfig()
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_weights(
        self,
        weights: dict[str, float],
        scenarios: list[dict],
    ) -> dict[str, float]:
        """Evaluate a weight configuration on multiple scenarios.

        Parameters
        ----------
        weights : dict with keys w_A, w_D, w_U, w_C, lam.
        scenarios : list of scenario dicts, each containing:
            - predictions_dict: {model_name: NDArray}
            - loss: float (observed loss for this scenario)
            - label: int (1 = high convergence, 0 = low convergence)
            plus optional ICM inputs (intervals, signs, etc.)

        Returns
        -------
        dict with keys: monotonicity, coverage, discrimination, composite
        """
        cfg = _make_config_from_weights(weights)

        icm_scores = np.array([
            _compute_scenario_icm(s, cfg) for s in scenarios
        ])
        losses = np.array([s["loss"] for s in scenarios])
        labels = np.array([s["label"] for s in scenarios])

        # --- Monotonicity: Spearman(ICM, -loss) ---
        # Higher ICM should correspond to lower loss
        if len(np.unique(icm_scores)) < 2 or len(np.unique(losses)) < 2:
            monotonicity = 0.0
        else:
            rho, _ = spearmanr(icm_scores, -losses)
            # Map from [-1, 1] to [0, 1]
            monotonicity = float(np.clip((rho + 1.0) / 2.0, 0.0, 1.0))

        # --- Discrimination: AUC between high/low convergence ---
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            discrimination = 0.5
        else:
            try:
                discrimination = float(roc_auc_score(labels, icm_scores))
            except ValueError:
                discrimination = 0.5

        # --- Coverage: conformal coverage at alpha = 0.10 ---
        coverage = self._compute_coverage(icm_scores, losses, alpha=0.10)

        composite = (
            _OBJ_MONOTONICITY * monotonicity
            + _OBJ_DISCRIMINATION * discrimination
            + _OBJ_COVERAGE * coverage
        )

        result = {
            "monotonicity": monotonicity,
            "coverage": coverage,
            "discrimination": discrimination,
            "composite": composite,
        }

        # Record history
        self.history.append({"weights": dict(weights), **result})

        return result

    def _compute_coverage(
        self,
        icm_scores: NDArray,
        losses: NDArray,
        alpha: float = 0.10,
    ) -> float:
        """Compute empirical conformal coverage.

        Splits data in half: first half for fitting isotonic + conformalize,
        second half for evaluating coverage.
        """
        n = len(icm_scores)
        if n < 6:
            # Not enough data for meaningful split
            return 0.5

        mid = n // 2
        # Use first half for fitting, second half for evaluation
        C_fit, L_fit = icm_scores[:mid], losses[:mid]
        C_eval, L_eval = icm_scores[mid:], losses[mid:]

        # Split the fit portion further for isotonic + conformal
        fit_mid = len(C_fit) // 2
        if fit_mid < 2:
            return 0.5

        try:
            g = fit_isotonic(C_fit[:fit_mid], L_fit[:fit_mid])
            g_alpha = conformalize(
                g, C_fit[fit_mid:], L_fit[fit_mid:], alpha=alpha
            )

            # Coverage: fraction of evaluation points where L <= g_alpha(C)
            risk_bounds = np.array([g_alpha(c) for c in C_eval])
            covered = np.mean(L_eval <= risk_bounds)
            return float(covered)
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def grid_search(
        self,
        scenarios: list[dict],
        n_points: int = 50,
        seed: int = 42,
    ) -> dict:
        """Grid search over weight space using Latin hypercube sampling.

        Parameters
        ----------
        scenarios : list of scenario dicts.
        n_points : number of sample points in the Latin hypercube.
        seed : random seed for reproducibility.

        Returns
        -------
        dict with keys: best_weights, best_score, all_results
        """
        # Latin hypercube sampling in [0, 1]^5, then scale to bounds
        sampler = qmc.LatinHypercube(d=len(_WEIGHT_NAMES), seed=seed)
        samples = sampler.random(n=n_points)

        lower = np.array([_WEIGHT_BOUNDS[n][0] for n in _WEIGHT_NAMES])
        upper = np.array([_WEIGHT_BOUNDS[n][1] for n in _WEIGHT_NAMES])
        scaled = qmc.scale(samples, lower, upper)

        all_results: list[dict] = []
        best_score = -np.inf
        best_weights: dict[str, float] = {}

        for i in range(n_points):
            w = _weights_to_dict(scaled[i])
            scores = self.evaluate_weights(w, scenarios)
            entry = {"weights": w, **scores}
            all_results.append(entry)

            if scores["composite"] > best_score:
                best_score = scores["composite"]
                best_weights = dict(w)

        return {
            "best_weights": best_weights,
            "best_score": float(best_score),
            "all_results": all_results,
        }

    # ------------------------------------------------------------------
    # Scipy-based optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        scenarios: list[dict],
        method: str = "nelder-mead",
        n_restarts: int = 5,
    ) -> dict:
        """Scipy-based optimization of weights.

        Parameters
        ----------
        scenarios : list of scenario dicts.
        method : optimization method for scipy.optimize.minimize.
        n_restarts : number of random restarts to escape local optima.

        Returns
        -------
        dict with keys: best_weights, best_score, optimization_result
        """
        lower = np.array([_WEIGHT_BOUNDS[n][0] for n in _WEIGHT_NAMES])
        upper = np.array([_WEIGHT_BOUNDS[n][1] for n in _WEIGHT_NAMES])
        bounds = list(zip(lower, upper))

        rng = np.random.default_rng(42)

        best_score = -np.inf
        best_weights: dict[str, float] = {}
        best_opt_result: Any = None

        def objective(x: NDArray) -> float:
            w = _weights_to_dict(np.clip(x, lower, upper))
            scores = self.evaluate_weights(w, scenarios)
            return -scores["composite"]  # Minimize negative composite

        for restart in range(n_restarts):
            # Random starting point within bounds
            x0 = rng.uniform(lower, upper)

            try:
                if method.lower() == "nelder-mead":
                    result = minimize(
                        objective,
                        x0,
                        method="Nelder-Mead",
                        options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4},
                    )
                else:
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={"maxiter": 200},
                    )

                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_weights = _weights_to_dict(
                        np.clip(result.x, lower, upper)
                    )
                    best_opt_result = result
            except Exception:
                continue

        # Fallback: if all restarts failed, use default config weights
        if not best_weights:
            default_w = {
                "w_A": self.config.w_A,
                "w_D": self.config.w_D,
                "w_U": self.config.w_U,
                "w_C": self.config.w_C,
                "lam": self.config.lam,
            }
            scores = self.evaluate_weights(default_w, scenarios)
            best_weights = default_w
            best_score = scores["composite"]

        return {
            "best_weights": best_weights,
            "best_score": float(best_score),
            "optimization_result": best_opt_result,
        }

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        scenarios: list[dict],
        n_folds: int = 5,
    ) -> dict:
        """K-fold cross-validation of optimized weights.

        For each fold, optimize on train split and evaluate on test split.

        Parameters
        ----------
        scenarios : list of scenario dicts.
        n_folds : number of folds.

        Returns
        -------
        dict with keys: mean_score, std_score, fold_scores, best_weights
        """
        n = len(scenarios)
        n_folds = min(n_folds, n)
        if n_folds < 2:
            raise ValueError(
                f"Need at least 2 scenarios for cross-validation, got {n}"
            )

        rng = np.random.default_rng(42)
        indices = rng.permutation(n)

        fold_size = n // n_folds
        fold_scores: list[float] = []
        fold_weights: list[dict[str, float]] = []

        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else n

            test_idx = set(indices[start:end].tolist())
            train = [s for i, s in enumerate(scenarios) if i not in test_idx]
            test = [s for i, s in enumerate(scenarios) if i in test_idx]

            if len(train) < 2 or len(test) < 1:
                continue

            # Optimize on train, evaluate on test
            opt_result = self.optimize(train, n_restarts=3)
            test_scores = self.evaluate_weights(opt_result["best_weights"], test)

            fold_scores.append(test_scores["composite"])
            fold_weights.append(opt_result["best_weights"])

        if not fold_scores:
            raise ValueError("Cross-validation produced no valid folds")

        # Pick weights from the fold with the best test score
        best_fold_idx = int(np.argmax(fold_scores))

        return {
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "fold_scores": fold_scores,
            "best_weights": fold_weights[best_fold_idx],
        }

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def generate_training_scenarios(
        self,
        n_scenarios: int = 20,
        seed: int = 42,
    ) -> list[dict]:
        """Generate diverse training scenarios for meta-learning.

        Creates scenarios spanning high-convergence (agreeing models, low loss)
        and low-convergence (disagreeing models, high loss) regimes.

        Parameters
        ----------
        n_scenarios : number of scenarios to generate.
        seed : random seed.

        Returns
        -------
        list of scenario dicts, each with:
            predictions_dict, loss, label, intervals, signs, features
        """
        rng = np.random.default_rng(seed)
        scenarios: list[dict] = []
        n_classes = 3

        for i in range(n_scenarios):
            # Alternate between high and low convergence
            is_high = i < n_scenarios // 2
            n_models = rng.integers(3, 7)

            # Base ground-truth distribution
            base = rng.dirichlet(np.ones(n_classes) * 2.0)

            predictions: dict[str, NDArray] = {}
            signs_list: list[float] = []
            intervals: list[tuple[float, float]] = []
            feature_sets: list[set[str]] = []
            all_features = [
                "f_a", "f_b", "f_c", "f_d", "f_e",
                "f_f", "f_g", "f_h", "f_i", "f_j",
            ]

            if is_high:
                # High convergence: small noise, same direction
                noise_scale = rng.uniform(0.01, 0.08)
                loss = rng.uniform(0.05, 0.3)
                label = 1
            else:
                # Low convergence: large noise, mixed directions
                noise_scale = rng.uniform(0.3, 0.8)
                loss = rng.uniform(0.5, 1.0)
                label = 0

            for m in range(n_models):
                noisy = base + rng.normal(0, noise_scale, n_classes)
                noisy = np.abs(noisy)
                noisy /= noisy.sum()
                predictions[f"model_{m}"] = noisy

                # Signs: high convergence = same sign, low = mixed
                if is_high:
                    signs_list.append(1.0)
                else:
                    signs_list.append(rng.choice([-1.0, 1.0]))

                # Intervals
                center = float(noisy.max())
                width = noise_scale * rng.uniform(0.5, 2.0)
                intervals.append((center - width, center + width))

                # Feature sets: high convergence = disjoint, low = overlapping
                if is_high:
                    n_feat = rng.integers(2, 5)
                    feat = set(rng.choice(all_features, size=n_feat, replace=False))
                else:
                    # More overlap for low convergence scenarios
                    common = set(rng.choice(all_features[:4], size=2, replace=False))
                    extra = set(rng.choice(all_features, size=rng.integers(1, 3), replace=False))
                    feat = common | extra
                feature_sets.append(feat)

            scenario = {
                "predictions_dict": predictions,
                "loss": float(loss),
                "label": int(label),
                "intervals": intervals,
                "signs": np.array(signs_list),
                "features": feature_sets,
            }
            scenarios.append(scenario)

        return scenarios

    # ------------------------------------------------------------------
    # Comparison with defaults
    # ------------------------------------------------------------------

    def compare_with_default(
        self,
        scenarios: list[dict],
        optimized_weights: dict[str, float],
    ) -> dict:
        """Compare optimized vs default weights.

        Parameters
        ----------
        scenarios : list of scenario dicts.
        optimized_weights : the optimized weight dict.

        Returns
        -------
        dict with keys: default_scores, optimized_scores, improvement_pct
        """
        default_weights = {
            "w_A": self.config.w_A,
            "w_D": self.config.w_D,
            "w_U": self.config.w_U,
            "w_C": self.config.w_C,
            "lam": self.config.lam,
        }

        default_scores = self.evaluate_weights(default_weights, scenarios)
        optimized_scores = self.evaluate_weights(optimized_weights, scenarios)

        # Compute improvement percentages
        improvement_pct: dict[str, float] = {}
        for key in ["monotonicity", "coverage", "discrimination", "composite"]:
            dv = default_scores[key]
            ov = optimized_scores[key]
            if abs(dv) < 1e-12:
                improvement_pct[key] = 0.0 if abs(ov) < 1e-12 else 100.0
            else:
                improvement_pct[key] = ((ov - dv) / abs(dv)) * 100.0

        return {
            "default_scores": default_scores,
            "optimized_scores": optimized_scores,
            "improvement_pct": improvement_pct,
        }
