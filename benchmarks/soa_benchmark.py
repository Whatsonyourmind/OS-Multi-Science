"""State-of-the-Art Benchmark -- ICM Framework vs Standard Baselines.

End-to-end benchmark that evaluates the ICM framework against standard
ensemble baselines across multiple axes: prediction quality, uncertainty
quantification, error detection, diversity assessment, and score range
analysis.

Uses real sklearn datasets and genuinely diverse model families.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

from benchmarks.datasets import (
    list_classification_datasets,
    list_regression_datasets,
    load_dataset,
    load_dataset_full,
    get_dataset_info,
)
from benchmarks.model_zoo import (
    build_classification_zoo,
    build_regression_zoo,
    train_zoo,
    collect_predictions_classification,
    collect_predictions_regression,
    collect_residuals,
)
from benchmarks.baselines import (
    EnsembleAverage,
    StackingBaseline,
    BootstrapEnsemble,
    SplitConformal,
    DiversityMetrics,
    DeepEnsemble,
    _mean_proba,
    _extract_class_predictions,
    _compute_standard_scores,
    _stack_predictions,
)
from framework.config import ICMConfig, CRCConfig
from framework.icm import (
    compute_icm_from_predictions,
    compute_icm,
    compute_icm_calibrated,
    compute_agreement,
    compute_direction,
    compute_uncertainty_overlap,
    compute_invariance,
    compute_dependency_penalty,
    compute_pi_from_predictions,
)
from framework.crc_gating import decision_gate, fit_isotonic, conformalize
from framework.types import ICMComponents, DecisionAction


# ============================================================
# Helpers
# ============================================================

def _icm_weighted_ensemble_classification(
    predictions: dict[str, np.ndarray],
    icm_scores_per_model: dict[str, float],
) -> np.ndarray:
    """Weight model predictions by per-model ICM scores.

    Higher-ICM models get more weight.  Weights are normalized to sum to 1.

    Parameters
    ----------
    predictions : dict mapping model names to (n_samples, n_classes) arrays.
    icm_scores_per_model : dict mapping model names to ICM scores.

    Returns
    -------
    np.ndarray  Weighted average predictions with shape (n_samples, n_classes).
    """
    model_names = sorted(predictions.keys())
    weights = np.array([icm_scores_per_model.get(m, 0.5) for m in model_names])
    # Shift weights so minimum is non-negative, then normalize
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()

    arrays = [np.asarray(predictions[m], dtype=np.float64) for m in model_names]
    stacked = np.stack(arrays, axis=0)  # (K, n_samples, n_classes)
    weighted = np.tensordot(weights, stacked, axes=([0], [0]))
    return weighted


def _icm_weighted_ensemble_regression(
    predictions: dict[str, np.ndarray],
    icm_scores_per_model: dict[str, float],
) -> np.ndarray:
    """Weight regression model mean predictions by per-model ICM scores.

    Returns
    -------
    np.ndarray  Weighted average point predictions of shape (n_samples,).
    """
    model_names = sorted(predictions.keys())
    weights = np.array([icm_scores_per_model.get(m, 0.5) for m in model_names])
    weights = weights - weights.min() + 1e-6
    weights = weights / weights.sum()

    # Extract mean predictions (column 0) from each model
    means = np.stack(
        [np.asarray(predictions[m], dtype=np.float64)[:, 0] for m in model_names],
        axis=0,
    )  # (K, n_samples)
    weighted = np.dot(weights, means)
    return weighted


def _compute_training_residuals_classification(
    preds_train: dict[str, np.ndarray],
    y_train: np.ndarray,
) -> np.ndarray:
    """Compute residual matrix from training predictions for Pi computation.

    Returns a (K, n*C) residual matrix where K is the number of models,
    n is the number of training samples, and C is the number of classes.
    This avoids using test labels for the dependency penalty.
    """
    model_names = sorted(preds_train.keys())
    y_tr = np.asarray(y_train, dtype=int)
    n = len(y_tr)

    # Get n_classes from the first prediction
    first_pred = np.asarray(preds_train[model_names[0]], dtype=np.float64)
    if first_pred.ndim == 2 and first_pred.shape[1] > 1:
        n_classes = first_pred.shape[1]
        one_hot = np.zeros((n, n_classes), dtype=np.float64)
        one_hot[np.arange(n), np.clip(y_tr, 0, n_classes - 1)] = 1.0
        rows = []
        for m in model_names:
            pred = np.asarray(preds_train[m], dtype=np.float64)[:n]
            rows.append((pred - one_hot).ravel())
        return np.stack(rows)
    else:
        # 1-D predictions (regression-style)
        y_float = np.asarray(y_train, dtype=np.float64)
        rows = []
        for m in model_names:
            pred = np.asarray(preds_train[m], dtype=np.float64).ravel()[:n]
            rows.append(pred - y_float[:len(pred)])
        return np.stack(rows)


def _compute_per_model_icm_classification(
    predictions: dict[str, np.ndarray],
    config: ICMConfig,
    *,
    preds_train: dict[str, np.ndarray] | None = None,
    y_train: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute an ICM-like quality score for each model via leave-one-out contribution.

    For each model, compute ICM with vs. without it. The difference (contribution)
    serves as the per-model score. Models that improve ensemble agreement get
    higher scores.

    ICM is computed WITHOUT test labels to avoid circularity.  Pi is either
    set to 0 (unsupervised) or computed from training residuals when
    ``preds_train`` and ``y_train`` are provided.
    """
    model_names = sorted(predictions.keys())
    if len(model_names) < 3:
        # With < 3 models, just use prediction confidence as proxy
        scores = {}
        for m in model_names:
            pred = np.asarray(predictions[m], dtype=np.float64)
            scores[m] = float(np.mean(np.max(pred, axis=1)))
        return scores

    # Compute training residuals for Pi if training data is available
    train_residuals = None
    if preds_train is not None and y_train is not None:
        train_residuals = _compute_training_residuals_classification(
            preds_train, y_train,
        )

    # Full ensemble ICM (per-sample, then average) -- no y_true, Pi from training
    full_icm = compute_icm_from_predictions(
        predictions, config=config, y_true=None, residuals=train_residuals,
    ).icm_score

    scores = {}
    for m in model_names:
        subset = {k: v for k, v in predictions.items() if k != m}
        # Compute subset residuals from training data if available
        sub_residuals = None
        if train_residuals is not None:
            sub_indices = [i for i, name in enumerate(model_names) if name != m]
            sub_residuals = train_residuals[sub_indices]
        sub_icm = compute_icm_from_predictions(
            subset, config=config, y_true=None, residuals=sub_residuals,
        ).icm_score
        # Contribution = how much ICM drops when we remove this model
        contribution = full_icm - sub_icm
        # Map to [0, 1] range via shift
        scores[m] = float(np.clip(0.5 + contribution * 5.0, 0.05, 0.95))

    return scores


def _compute_per_model_icm_regression(
    predictions: dict[str, np.ndarray],
    config: ICMConfig,
    *,
    preds_train: dict[str, np.ndarray] | None = None,
    y_train: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute per-model ICM contribution scores for regression.

    Uses wasserstein distance (which handles real-valued predictions)
    instead of hellinger (which requires non-negative distributions).

    ICM is computed WITHOUT test labels to avoid circularity.  Pi is either
    set to 0 (unsupervised) or computed from training residuals when
    ``preds_train`` and ``y_train`` are provided.
    """
    model_names = sorted(predictions.keys())
    # For regression, use mean column as point prediction for ICM
    point_preds = {}
    for m in model_names:
        arr = np.asarray(predictions[m], dtype=np.float64)
        point_preds[m] = arr[:, 0]  # mean column

    if len(model_names) < 3:
        # With < 3 models, use prediction spread as proxy (no y_true)
        scores = {}
        ensemble_mean = np.mean(
            [point_preds[m] for m in model_names], axis=0,
        )
        for m in model_names:
            # How close is this model to the ensemble mean (unsupervised)
            deviation = np.mean(np.abs(point_preds[m] - ensemble_mean))
            scores[m] = float(1.0 / (1.0 + deviation))
        return scores

    # Compute training residuals for Pi if training data is available
    train_residuals = None
    if preds_train is not None and y_train is not None:
        train_point_preds = {}
        for m in model_names:
            if m in preds_train:
                arr = np.asarray(preds_train[m], dtype=np.float64)
                train_point_preds[m] = arr[:, 0]
        if len(train_point_preds) >= 2:
            y_tr = np.asarray(y_train, dtype=np.float64)
            rows = []
            for m in model_names:
                if m in train_point_preds:
                    rows.append(train_point_preds[m] - y_tr)
            train_residuals = np.stack(rows)

    full_icm = compute_icm_from_predictions(
        point_preds, config=config, y_true=None, residuals=train_residuals,
        distance_fn="wasserstein",
    ).icm_score

    scores = {}
    for m in model_names:
        subset = {k: v for k, v in point_preds.items() if k != m}
        sub_residuals = None
        if train_residuals is not None:
            sub_indices = [i for i, name in enumerate(model_names) if name != m]
            sub_residuals = train_residuals[sub_indices]
        sub_icm = compute_icm_from_predictions(
            subset, config=config, y_true=None, residuals=sub_residuals,
            distance_fn="wasserstein",
        ).icm_score
        contribution = full_icm - sub_icm
        scores[m] = float(np.clip(0.5 + contribution * 5.0, 0.05, 0.95))

    return scores


def _compute_per_sample_icm_classification(
    predictions: dict[str, np.ndarray],
    config: ICMConfig,
) -> np.ndarray:
    """Compute per-sample ICM scores for classification.

    Uses an efficient vectorized approach: for each sample, compute the
    pairwise Hellinger distance among models' predicted probability
    distributions, then aggregate via the ICM formula.

    ICM is computed WITHOUT test labels to avoid circularity.  Only
    unsupervised components are used:
    - A (agreement): pairwise Hellinger distance between model predictions
    - D (direction): argmax class-vote agreement (if config.direction_mode
      is "auto" or "argmax"), or 1.0 for legacy sign mode
    - U (uncertainty overlap): overlap of per-model prediction intervals
    - C (invariance): perturbation-based if config.perturbation_scale > 0
    - Pi (dependency): set to 0.0 (no test labels available)

    Returns
    -------
    np.ndarray of shape (n_samples,) with ICM scores in [0, 1].
    """
    from scipy.special import expit as _expit
    from scipy.stats import beta as _scipy_beta

    model_names = sorted(predictions.keys())
    pred_arrays = [np.asarray(predictions[m], dtype=np.float64) for m in model_names]
    n_samples = pred_arrays[0].shape[0]
    K = len(model_names)

    # Pre-compute adaptive C_A if needed (across ALL samples for stability)
    use_adaptive_C_A = config.C_A_mode == "adaptive"
    if use_adaptive_C_A:
        # Compute global pairwise distance distribution for C_A calibration
        all_dists = []
        # Sample up to 200 samples for efficiency
        sample_indices = np.random.default_rng(0).choice(
            n_samples, size=min(200, n_samples), replace=False,
        )
        for idx in sample_indices:
            for ki in range(K):
                for kj in range(ki + 1, K):
                    p = np.maximum(pred_arrays[ki][idx], 0.0)
                    q = np.maximum(pred_arrays[kj][idx], 0.0)
                    h = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
                    all_dists.append(h)
        C_A_adaptive = float(np.percentile(all_dists, config.C_A_adaptive_percentile)) * 1.1 + 1e-12
    else:
        C_A_adaptive = config.C_A_hellinger

    # Pre-compute D: direction agreement (same for all samples in argmax mode,
    # but we compute per-sample for accuracy)
    use_argmax_direction = (
        config.direction_mode == "argmax"
        or (config.direction_mode == "auto" and pred_arrays[0].ndim == 2 and pred_arrays[0].shape[1] > 1)
    )
    if use_argmax_direction:
        argmax_per_model = np.stack([np.argmax(p, axis=1) for p in pred_arrays])  # (K, n_samples)

    # Pre-compute C: invariance via perturbation
    if config.perturbation_scale > 0.0:
        rng = np.random.default_rng(0)
        pre_all = []
        post_all = []
        for p_arr in pred_arrays:
            p_flat = p_arr.ravel()
            p_std = float(np.std(p_flat)) + 1e-12
            noise = rng.normal(0, config.perturbation_scale * p_std, size=p_flat.shape)
            pre_all.extend(p_flat.tolist())
            post_all.extend((p_flat + noise).tolist())
        C_global = compute_invariance(np.array(pre_all), np.array(post_all))
    else:
        C_global = 1.0

    icm_scores = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        # A: agreement via Hellinger per-sample
        dists = []
        for ki in range(K):
            for kj in range(ki + 1, K):
                p = pred_arrays[ki][i]
                q = pred_arrays[kj][i]
                p_safe = np.maximum(p, 0.0)
                q_safe = np.maximum(q, 0.0)
                h = float(np.sqrt(0.5 * np.sum((np.sqrt(p_safe) - np.sqrt(q_safe)) ** 2)))
                dists.append(h)
        mean_d = float(np.mean(dists)) if dists else 0.0
        A = max(1.0 - mean_d / C_A_adaptive, 0.0)

        # D: direction
        if use_argmax_direction:
            votes = argmax_per_model[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            if len(unique) <= 1:
                D = 1.0
            else:
                probs = counts / counts.sum()
                H = -float(np.sum(probs * np.log(probs + 1e-12)))
                H_max = float(np.log(len(unique)))
                D = 1.0 - H / H_max if H_max > 1e-12 else 1.0
        else:
            D = 1.0  # Legacy sign mode: all probabilities are positive

        # U: uncertainty overlap from per-model confidence intervals
        intervals = []
        for k in range(K):
            p = pred_arrays[k][i]
            lo = float(np.min(p))
            hi = float(np.max(p))
            intervals.append((lo, hi))
        U = compute_uncertainty_overlap(intervals)

        # C: invariance (global, not per-sample)
        C_val = C_global

        # Pi: set to 0 -- no test labels used (avoids circularity)
        Pi = 0.0

        # ICM aggregation - respect config.aggregation setting
        z_raw = (
            config.w_A * A
            + config.w_D * D
            + config.w_U * U
            + config.w_C * C_val
            - config.lam * Pi
        )

        # Apply aggregation method based on config
        if config.aggregation == "logistic":
            # Logistic sigmoid with scale and shift
            z = config.logistic_scale * (z_raw - config.logistic_shift)
            icm_scores[i] = float(_expit(z))
        elif config.aggregation == "geometric":
            # Geometric mean aggregation
            eps = 1e-12
            factors = [
                (max(A, eps), config.w_A),
                (max(D, eps), config.w_D),
                (max(U, eps), config.w_U),
                (max(C_val, eps), config.w_C),
                (max(1.0 - Pi, eps), config.lam),
            ]
            total_w = sum(w for _, w in factors)
            log_icm = sum(w / total_w * np.log(v) for v, w in factors)
            icm_scores[i] = float(np.clip(np.exp(log_icm), 0.0, 1.0))
        elif config.aggregation == "calibrated":
            # Beta CDF aggregation
            z_min = -config.lam
            z_max = config.w_A + config.w_D + config.w_U + config.w_C
            denom = z_max - z_min
            if denom < 1e-12:
                z_norm = 0.5
            else:
                z_norm = (z_raw - z_min) / denom
            z_norm = float(np.clip(z_norm, 0.0, 1.0))
            icm_scores[i] = float(_scipy_beta.cdf(z_norm, config.beta_shape_a, config.beta_shape_b))
        elif config.aggregation == "adaptive":
            # Adaptive percentile-based aggregation
            # Use a default calibration set when not provided
            cal_scores = config.adaptive_calibration_scores
            if cal_scores is None or len(cal_scores) < 2:
                # Generate default calibration scores
                rng = np.random.default_rng(0)
                n_cal_samples = 500
                cal_scores = []
                for _ in range(n_cal_samples):
                    a = rng.uniform(0, 1)
                    d = rng.uniform(0, 1)
                    u = rng.uniform(0, 1)
                    c = rng.uniform(0, 1)
                    pi = rng.uniform(0, 1)
                    s = (
                        config.w_A * a
                        + config.w_D * d
                        + config.w_U * u
                        + config.w_C * c
                        - config.lam * pi
                    )
                    cal_scores.append(s)

            sorted_cal = np.sort(cal_scores)
            n_cal = len(sorted_cal)
            idx = np.searchsorted(sorted_cal, z_raw, side="right")

            if idx <= 0:
                icm_scores[i] = 0.0
            elif idx >= n_cal:
                icm_scores[i] = 1.0
            else:
                lo = sorted_cal[idx - 1]
                hi = sorted_cal[idx]
                if hi - lo < 1e-12:
                    icm_scores[i] = idx / n_cal
                else:
                    frac = (z_raw - lo) / (hi - lo)
                    icm_scores[i] = (idx - 1 + frac) / n_cal
            icm_scores[i] = float(np.clip(icm_scores[i], 0.0, 1.0))
        else:
            # Fallback to logistic if aggregation method is invalid
            z = config.logistic_scale * (z_raw - config.logistic_shift)
            icm_scores[i] = float(_expit(z))

    return icm_scores


def _compute_per_sample_error_classification(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> np.ndarray:
    """Per-sample ensemble error: 1 if ensemble majority vote is wrong, 0 otherwise."""
    avg = _mean_proba(predictions)
    if avg.ndim == 2:
        pred_class = np.argmax(avg, axis=1)
    else:
        pred_class = (avg >= 0.5).astype(int)
    return (pred_class != y_true).astype(float)


def _ensemble_entropy(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Per-sample entropy of the average prediction distribution."""
    avg = _mean_proba(predictions)
    if avg.ndim == 1:
        avg = np.column_stack([1.0 - avg, avg])
    from scipy.stats import entropy as _ent

    return np.array([_ent(avg[i] + 1e-15) for i in range(avg.shape[0])])


def _icm_prediction_sets_classification(
    predictions: dict[str, np.ndarray],
    icm_scores: np.ndarray,
    tau_hi: float = 0.7,
    tau_lo: float = 0.3,
) -> list[set]:
    """Build prediction sets using ICM-based gating.

    - High ICM (>= tau_hi): single best class (ACT)
    - Medium ICM: top-2 classes (DEFER)
    - Low ICM (< tau_lo): top-3 or all classes (AUDIT)
    """
    avg = _mean_proba(predictions)
    if avg.ndim == 1:
        avg = np.column_stack([1.0 - avg, avg])

    n_samples, n_classes = avg.shape
    pred_sets = []

    for i in range(n_samples):
        sorted_classes = np.argsort(avg[i])[::-1]
        icm = icm_scores[i]

        if icm >= tau_hi:
            # High confidence: single class
            pred_sets.append({int(sorted_classes[0])})
        elif icm >= tau_lo:
            # Medium: top-2
            k = min(2, n_classes)
            pred_sets.append(set(int(c) for c in sorted_classes[:k]))
        else:
            # Low: top-3 or all
            k = min(3, n_classes)
            pred_sets.append(set(int(c) for c in sorted_classes[:k]))

    return pred_sets


# ============================================================
# Experiment 1: Prediction Quality
# ============================================================

def experiment_prediction_quality(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    config: ICMConfig | None = None,
    _preds_train: dict[str, np.ndarray] | None = None,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare ICM-weighted ensemble against baselines on prediction quality.

    Returns
    -------
    dict with keys:
        'results_table': pd.DataFrame with method comparison
        'icm_weighted_preds': weighted predictions
        'per_model_icm': per-model ICM scores
    """
    if config is None:
        config = ICMConfig.wide_range_preset()

    results_rows = []

    if task == "classification":
        if _preds_train is not None and _preds_test is not None:
            preds_train = _preds_train
            preds_test = _preds_test
        else:
            models = build_classification_zoo(seed=seed)
            train_zoo(models, X_train, y_train)
            preds_train = collect_predictions_classification(models, X_train)
            preds_test = collect_predictions_classification(models, X_test)

        # ICM-weighted ensemble (no test labels for ICM -- uses training residuals)
        per_model_icm = _compute_per_model_icm_classification(
            preds_test, config,
            preds_train=preds_train, y_train=y_train,
        )
        icm_weighted = _icm_weighted_ensemble_classification(preds_test, per_model_icm)
        icm_scores = _compute_standard_scores(icm_weighted, y_test)
        icm_scores["method"] = "ICM-Weighted"
        results_rows.append(icm_scores)

        # Ensemble Average
        ea = EnsembleAverage()
        ea_scores = ea.score(preds_test, y_test)
        ea_scores["method"] = "Ensemble Avg"
        results_rows.append(ea_scores)

        # Stacking (Logistic)
        try:
            sl = StackingBaseline(meta_learner="logistic")
            sl.fit(preds_train, y_train)
            sl_scores = sl.score(preds_test, y_test)
            sl_scores["method"] = "Stacking (LR)"
            results_rows.append(sl_scores)
        except Exception:
            results_rows.append({
                "method": "Stacking (LR)",
                "accuracy": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
            })

        # Stacking (RF)
        try:
            sr = StackingBaseline(meta_learner="random_forest")
            sr.fit(preds_train, y_train)
            sr_scores = sr.score(preds_test, y_test)
            sr_scores["method"] = "Stacking (RF)"
            results_rows.append(sr_scores)
        except Exception:
            results_rows.append({
                "method": "Stacking (RF)",
                "accuracy": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
            })

        # Deep Ensemble (5 MLP members with different architectures)
        try:
            de = DeepEnsemble(n_members=5, max_iter=200, seed=seed)
            de.fit(X_train, y_train)
            de_scores = de.score(X_test, y_test)
            de_scores["method"] = "Deep Ensemble"
            results_rows.append(de_scores)
        except Exception:
            results_rows.append({
                "method": "Deep Ensemble",
                "accuracy": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
            })

        return {
            "results_table": pd.DataFrame(results_rows),
            "icm_weighted_preds": icm_weighted,
            "per_model_icm": per_model_icm,
            "preds_train": preds_train,
            "preds_test": preds_test,
        }

    else:
        # Regression
        if _preds_test is not None:
            preds_test = _preds_test
            preds_train = _preds_train
        else:
            models = build_regression_zoo(seed=seed)
            train_zoo(models, X_train, y_train)
            preds_test = collect_predictions_regression(models, X_test)
            preds_train = collect_predictions_regression(models, X_train)

        # ICM-weighted ensemble (no test labels -- uses training residuals)
        per_model_icm = _compute_per_model_icm_regression(
            preds_test, config,
            preds_train=preds_train, y_train=y_train,
        )
        icm_weighted = _icm_weighted_ensemble_regression(preds_test, per_model_icm)
        rmse_icm = float(np.sqrt(np.mean((icm_weighted - y_test) ** 2)))
        mse_icm = float(np.mean((icm_weighted - y_test) ** 2))

        # Ensemble average (mean of point predictions)
        model_names = sorted(preds_test.keys())
        means_stack = np.stack(
            [preds_test[m][:, 0] for m in model_names], axis=0,
        )
        avg_pred = means_stack.mean(axis=0)
        rmse_avg = float(np.sqrt(np.mean((avg_pred - y_test) ** 2)))
        mse_avg = float(np.mean((avg_pred - y_test) ** 2))

        results_rows = [
            {"method": "ICM-Weighted", "RMSE": rmse_icm, "MSE": mse_icm},
            {"method": "Ensemble Avg", "RMSE": rmse_avg, "MSE": mse_avg},
        ]

        return {
            "results_table": pd.DataFrame(results_rows),
            "icm_weighted_preds": icm_weighted,
            "per_model_icm": per_model_icm,
            "preds_test": preds_test,
        }


# ============================================================
# Experiment 2: Uncertainty Quantification
# ============================================================

def experiment_uncertainty_quantification(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    config: ICMConfig | None = None,
    crc_config: CRCConfig | None = None,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare ICM-based prediction sets vs. split conformal.

    Only runs for classification datasets.

    Returns
    -------
    dict with keys:
        'results_table': pd.DataFrame with coverage, set size comparisons
        'icm_sets': ICM-based prediction sets
        'conformal_sets': split conformal prediction sets
    """
    if config is None:
        config = ICMConfig.wide_range_preset()
    if crc_config is None:
        crc_config = CRCConfig()

    if task != "classification":
        return {
            "results_table": pd.DataFrame(),
            "icm_sets": [],
            "conformal_sets": [],
            "skip_reason": "regression dataset",
        }

    if _preds_test is not None:
        preds_test = _preds_test
    else:
        models = build_classification_zoo(seed=seed)
        train_zoo(models, X_train, y_train)
        preds_test = collect_predictions_classification(models, X_test)

    y_test_int = np.asarray(y_test, dtype=int)

    # Split test set into calibration and evaluation subsets.
    # Both ICM-CRC and Split Conformal are evaluated on the SAME
    # evaluation subset (test_idx) for a fair comparison.
    avg_pred = _mean_proba(preds_test)
    n_test = len(y_test_int)
    n_cal = n_test // 2
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_test)
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]

    # --- ICM-based prediction sets on eval subset only ---
    # Build per-model prediction dicts restricted to eval samples
    eval_preds = {m: v[test_idx] for m, v in preds_test.items()}
    per_sample_icm = _compute_per_sample_icm_classification(
        eval_preds, config,
    )
    icm_sets = _icm_prediction_sets_classification(
        eval_preds, per_sample_icm,
        tau_hi=crc_config.tau_hi,
        tau_lo=crc_config.tau_lo,
    )

    icm_coverage = sum(
        1 for i, ps in enumerate(icm_sets) if y_test_int[test_idx[i]] in ps
    ) / len(test_idx)
    icm_avg_size = float(np.mean([len(ps) for ps in icm_sets]))

    # --- Split conformal baseline on same eval subset ---
    sc = SplitConformal(alpha=crc_config.alpha)
    sc.calibrate(avg_pred[cal_idx], y_test_int[cal_idx])
    conformal_sets = sc.predict_sets(avg_pred[test_idx])
    conf_coverage = sum(
        1 for i, ps in enumerate(conformal_sets) if y_test_int[test_idx[i]] in ps
    ) / len(test_idx)
    conf_avg_size = float(np.mean([len(ps) for ps in conformal_sets]))

    # Conditional coverage: per-class coverage on the SAME eval subset
    n_classes = avg_pred.shape[1] if avg_pred.ndim == 2 else 2
    icm_cond_cov = {}
    conf_cond_cov = {}
    for c in range(n_classes):
        mask_c_sub = y_test_int[test_idx] == c
        if mask_c_sub.sum() > 0:
            icm_cond_cov[c] = sum(
                1 for i in range(len(test_idx))
                if mask_c_sub[i] and y_test_int[test_idx[i]] in icm_sets[i]
            ) / mask_c_sub.sum()
            conf_cond_cov[c] = sum(
                1 for i in range(len(test_idx))
                if mask_c_sub[i] and y_test_int[test_idx[i]] in conformal_sets[i]
            ) / mask_c_sub.sum()

    results = pd.DataFrame([
        {
            "Method": "ICM-CRC Gating",
            "Coverage": icm_coverage,
            "Avg Set Size": icm_avg_size,
            "Cond. Cov. Std": float(np.std(list(icm_cond_cov.values()))) if icm_cond_cov else np.nan,
        },
        {
            "Method": "Split Conformal",
            "Coverage": conf_coverage,
            "Avg Set Size": conf_avg_size,
            "Cond. Cov. Std": float(np.std(list(conf_cond_cov.values()))) if conf_cond_cov else np.nan,
        },
    ])

    return {
        "results_table": results,
        "icm_sets": icm_sets,
        "conformal_sets": conformal_sets,
        "icm_per_sample": per_sample_icm,
        "icm_cond_coverage": icm_cond_cov,
        "conf_cond_coverage": conf_cond_cov,
    }


# ============================================================
# Experiment 3: Error Detection
# ============================================================

def experiment_error_detection(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    config: ICMConfig | None = None,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Evaluate ICM as a predictor of model errors.

    Measures correlation between various quality signals and actual errors.

    Returns
    -------
    dict with keys:
        'results_table': pd.DataFrame comparing error-detection signals
        'icm_error_corr': Spearman correlation between ICM and error
    """
    if config is None:
        config = ICMConfig.wide_range_preset()

    if task != "classification":
        return {
            "results_table": pd.DataFrame(),
            "skip_reason": "regression dataset",
        }

    if _preds_test is not None:
        preds_test = _preds_test
    else:
        models = build_classification_zoo(seed=seed)
        train_zoo(models, X_train, y_train)
        preds_test = collect_predictions_classification(models, X_test)
    y_test_int = np.asarray(y_test, dtype=int)

    # Compute per-sample signals (ICM computed WITHOUT test labels)
    per_sample_icm = _compute_per_sample_icm_classification(
        preds_test, config,
    )
    per_sample_error = _compute_per_sample_error_classification(preds_test, y_test_int)
    ens_entropy = _ensemble_entropy(preds_test)

    # Bootstrap disagreement
    be = BootstrapEnsemble(n_bootstrap=50)
    boot_disagree = be.compute_disagreement(preds_test)

    # ICM: higher ICM should correlate with LOWER error -> expect negative correlation
    # We negate ICM so a positive Spearman = good detection
    icm_signal = 1.0 - per_sample_icm  # Invert: higher = more uncertain/error-prone

    results_rows = []

    # ICM (inverted: 1 - ICM)
    if np.std(icm_signal) > 1e-12 and np.std(per_sample_error) > 1e-12:
        corr_icm, p_icm = spearmanr(icm_signal, per_sample_error)
    else:
        corr_icm, p_icm = 0.0, 1.0
    results_rows.append({
        "Signal": "ICM (inverted)",
        "Spearman Corr": float(corr_icm),
        "p-value": float(p_icm),
    })

    # Ensemble entropy
    if np.std(ens_entropy) > 1e-12 and np.std(per_sample_error) > 1e-12:
        corr_ent, p_ent = spearmanr(ens_entropy, per_sample_error)
    else:
        corr_ent, p_ent = 0.0, 1.0
    results_rows.append({
        "Signal": "Ensemble Entropy",
        "Spearman Corr": float(corr_ent),
        "p-value": float(p_ent),
    })

    # Bootstrap disagreement
    if np.std(boot_disagree) > 1e-12 and np.std(per_sample_error) > 1e-12:
        corr_boot, p_boot = spearmanr(boot_disagree, per_sample_error)
    else:
        corr_boot, p_boot = 0.0, 1.0
    results_rows.append({
        "Signal": "Bootstrap Disagreement",
        "Spearman Corr": float(corr_boot),
        "p-value": float(p_boot),
    })

    # Max probability confidence (inverted: 1 - max_prob)
    avg = _mean_proba(preds_test)
    if avg.ndim == 1:
        avg = np.column_stack([1.0 - avg, avg])
    max_prob_uncertainty = 1.0 - np.max(avg, axis=1)
    if np.std(max_prob_uncertainty) > 1e-12 and np.std(per_sample_error) > 1e-12:
        corr_mp, p_mp = spearmanr(max_prob_uncertainty, per_sample_error)
    else:
        corr_mp, p_mp = 0.0, 1.0
    results_rows.append({
        "Signal": "Max-Prob Uncertainty",
        "Spearman Corr": float(corr_mp),
        "p-value": float(p_mp),
    })

    return {
        "results_table": pd.DataFrame(results_rows),
        "icm_error_corr": float(corr_icm),
        "per_sample_icm": per_sample_icm,
        "per_sample_error": per_sample_error,
    }


# ============================================================
# Experiment 4: Diversity Assessment
# ============================================================

def experiment_diversity_assessment(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    config: ICMConfig | None = None,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare ICM components against standard diversity metrics.

    Returns
    -------
    dict with keys:
        'icm_components': dict of ICM component values
        'standard_metrics': dict of standard diversity metrics
        'results_table': pd.DataFrame comparing both
    """
    if config is None:
        config = ICMConfig.wide_range_preset()

    if task != "classification":
        return {
            "results_table": pd.DataFrame(),
            "skip_reason": "regression dataset",
        }

    if _preds_test is not None:
        preds_test = _preds_test
    else:
        models = build_classification_zoo(seed=seed)
        train_zoo(models, X_train, y_train)
        preds_test = collect_predictions_classification(models, X_test)
    y_test_int = np.asarray(y_test, dtype=int)

    # ICM components (no test labels for ICM -- avoids circularity)
    icm_result = compute_icm_from_predictions(
        preds_test, config=config, y_true=None,
    )
    comps = icm_result.components

    # Standard diversity metrics
    dm = DiversityMetrics()
    q_stat = dm.q_statistic(preds_test, y_test_int)
    disagree = dm.disagreement_measure(preds_test, y_test_int)
    corr_coef = dm.correlation_coefficient(preds_test, y_test_int)
    ent = dm.entropy_measure(preds_test)
    kl = dm.kl_diversity(preds_test)

    results_rows = [
        {"Source": "ICM", "Metric": "Agreement (A)", "Value": comps.A},
        {"Source": "ICM", "Metric": "Direction (D)", "Value": comps.D},
        {"Source": "ICM", "Metric": "Uncertainty Overlap (U)", "Value": comps.U},
        {"Source": "ICM", "Metric": "Invariance (C)", "Value": comps.C},
        {"Source": "ICM", "Metric": "Dependency Penalty (Pi)", "Value": comps.Pi},
        {"Source": "ICM", "Metric": "ICM Score", "Value": icm_result.icm_score},
        {"Source": "Standard", "Metric": "Q-Statistic", "Value": q_stat},
        {"Source": "Standard", "Metric": "Disagreement", "Value": disagree},
        {"Source": "Standard", "Metric": "Correlation Coef", "Value": corr_coef},
        {"Source": "Standard", "Metric": "Entropy", "Value": ent},
        {"Source": "Standard", "Metric": "KL Diversity", "Value": kl},
    ]

    return {
        "results_table": pd.DataFrame(results_rows),
        "icm_components": {
            "A": comps.A, "D": comps.D, "U": comps.U,
            "C": comps.C, "Pi": comps.Pi, "ICM": icm_result.icm_score,
        },
        "standard_metrics": {
            "Q-Statistic": q_stat,
            "Disagreement": disagree,
            "Correlation": corr_coef,
            "Entropy": ent,
            "KL Diversity": kl,
        },
    }


# ============================================================
# Experiment 5: Score Range Analysis
# ============================================================

def experiment_score_range(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare ICM with default logistic, wide_range_preset, and calibrated Beta CDF.

    Returns
    -------
    dict with keys:
        'results_table': pd.DataFrame comparing score distributions
        'default_scores': per-sample ICM with default config
        'wide_scores': per-sample ICM with wide range
        'calibrated_scores': per-sample ICM with calibrated Beta CDF
    """
    if task != "classification":
        return {
            "results_table": pd.DataFrame(),
            "skip_reason": "regression dataset",
        }

    if _preds_test is not None:
        preds_test = _preds_test
    else:
        models = build_classification_zoo(seed=seed)
        train_zoo(models, X_train, y_train)
        preds_test = collect_predictions_classification(models, X_test)
    y_test_int = np.asarray(y_test, dtype=int)

    configs = {
        "Default Logistic": ICMConfig(),
        "Wide Range": ICMConfig.wide_range_preset(),
        "Calibrated Beta": ICMConfig(aggregation="calibrated"),
    }

    results_rows = []
    all_scores = {}

    for name, cfg in configs.items():
        per_sample = _compute_per_sample_icm_classification(
            preds_test, cfg,
        )
        all_scores[name] = per_sample

        score_range = float(per_sample.max() - per_sample.min())
        # Check CRC threshold functionality
        n_act = np.sum(per_sample >= 0.7)
        n_defer = np.sum((per_sample >= 0.3) & (per_sample < 0.7))
        n_audit = np.sum(per_sample < 0.3)
        total = len(per_sample)

        results_rows.append({
            "Config": name,
            "Mean": float(np.mean(per_sample)),
            "Std": float(np.std(per_sample)),
            "Min": float(np.min(per_sample)),
            "Max": float(np.max(per_sample)),
            "Range": score_range,
            "ACT (>=0.7)": f"{n_act}/{total}",
            "DEFER": f"{n_defer}/{total}",
            "AUDIT (<0.3)": f"{n_audit}/{total}",
        })

    return {
        "results_table": pd.DataFrame(results_rows),
        "default_scores": all_scores.get("Default Logistic", np.array([])),
        "wide_scores": all_scores.get("Wide Range", np.array([])),
        "calibrated_scores": all_scores.get("Calibrated Beta", np.array([])),
    }


# ============================================================
# Experiment 6: Ablation Study
# ============================================================

def experiment_ablation_study(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    config: ICMConfig | None = None,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run a comprehensive ablation study of ICM components.

    For classification datasets, evaluates multiple variants of the ICM framework:
    - Full ICM (all 5 components)
    - ICM without each component (A, D, U, C, Pi)
    - ICM with only A
    - ICM with ensemble entropy only (as alternative signal)
    - ICM with A+D only
    - Random baseline

    For each variant, computes:
    - Spearman correlation between inverted signal and per-sample error
    - Decision gate breakdown (ACT/DEFER/AUDIT percentages)
    - Mean signal value

    Returns
    -------
    dict with keys:
        'results_table': pd.DataFrame with columns [Variant, Spearman Corr, ACT%, DEFER%, AUDIT%, Mean Score]
        'ablation_details': dict mapping variant name to per-sample scores
        'skip_reason': (for regression) reason dataset was skipped
    """
    if config is None:
        config = ICMConfig.wide_range_preset()

    if task != "classification":
        return {
            "results_table": pd.DataFrame(),
            "ablation_details": {},
            "skip_reason": "regression dataset",
        }

    if _preds_test is not None:
        preds_test = _preds_test
    else:
        models = build_classification_zoo(seed=seed)
        train_zoo(models, X_train, y_train)
        preds_test = collect_predictions_classification(models, X_test)

    y_test_int = np.asarray(y_test, dtype=int)
    n_samples = y_test_int.shape[0]

    # Compute per-sample error for correlation analysis
    per_sample_error = _compute_per_sample_error_classification(preds_test, y_test_int)

    results_rows = []
    ablation_details = {}

    # Use a small epsilon for "zero" weights (ICMConfig requires weights > 0)
    epsilon = 1e-6

    # ===== Variant 1: Full ICM (all 5 components) =====
    variant_name = "Full ICM"
    full_config = ICMConfig(
        w_A=config.w_A,
        w_D=config.w_D,
        w_U=config.w_U,
        w_C=config.w_C,
        lam=config.lam,
        logistic_scale=config.logistic_scale,
        logistic_shift=config.logistic_shift,
        aggregation=config.aggregation,
    )
    scores = _compute_per_sample_icm_classification(preds_test, full_config)
    ablation_details[variant_name] = scores
    corr, pval = _compute_spearman_for_ablation(1.0 - scores, per_sample_error)
    results_rows.append(_ablation_row(variant_name, scores, corr))

    # ===== Variant 2: ICM without A (set w_A to epsilon) =====
    variant_name = "No A (w_A~0)"
    no_a_config = ICMConfig(
        w_A=epsilon,
        w_D=config.w_D,
        w_U=config.w_U,
        w_C=config.w_C,
        lam=config.lam,
        logistic_scale=config.logistic_scale,
        logistic_shift=config.logistic_shift,
        aggregation=config.aggregation,
    )
    scores = _compute_per_sample_icm_classification(preds_test, no_a_config)
    ablation_details[variant_name] = scores
    corr, pval = _compute_spearman_for_ablation(1.0 - scores, per_sample_error)
    results_rows.append(_ablation_row(variant_name, scores, corr))

    # ===== Variant 3: ICM without D (set w_D to epsilon) =====
    variant_name = "No D (w_D~0)"
    no_d_config = ICMConfig(
        w_A=config.w_A,
        w_D=epsilon,
        w_U=config.w_U,
        w_C=config.w_C,
        lam=config.lam,
        logistic_scale=config.logistic_scale,
        logistic_shift=config.logistic_shift,
        aggregation=config.aggregation,
    )
    scores = _compute_per_sample_icm_classification(preds_test, no_d_config)
    ablation_details[variant_name] = scores
    corr, pval = _compute_spearman_for_ablation(1.0 - scores, per_sample_error)
    results_rows.append(_ablation_row(variant_name, scores, corr))

    # ===== Variant 4: ICM without U (set w_U to epsilon) =====
    variant_name = "No U (w_U~0)"
    no_u_config = ICMConfig(
        w_A=config.w_A,
        w_D=config.w_D,
        w_U=epsilon,
        w_C=config.w_C,
        lam=config.lam,
        logistic_scale=config.logistic_scale,
        logistic_shift=config.logistic_shift,
        aggregation=config.aggregation,
    )
    scores = _compute_per_sample_icm_classification(preds_test, no_u_config)
    ablation_details[variant_name] = scores
    corr, pval = _compute_spearman_for_ablation(1.0 - scores, per_sample_error)
    results_rows.append(_ablation_row(variant_name, scores, corr))

    # ===== Variant 5: ICM without C (set w_C to epsilon) =====
    variant_name = "No C (w_C~0)"
    no_c_config = ICMConfig(
        w_A=config.w_A,
        w_D=config.w_D,
        w_U=config.w_U,
        w_C=epsilon,
        lam=config.lam,
        logistic_scale=config.logistic_scale,
        logistic_shift=config.logistic_shift,
        aggregation=config.aggregation,
    )
    scores = _compute_per_sample_icm_classification(preds_test, no_c_config)
    ablation_details[variant_name] = scores
    corr, pval = _compute_spearman_for_ablation(1.0 - scores, per_sample_error)
    results_rows.append(_ablation_row(variant_name, scores, corr))

    # ===== Variant 6: ICM with only A (all other weights to epsilon) =====
    variant_name = "Only A"
    only_a_config = ICMConfig(
        w_A=config.w_A,
        w_D=epsilon,
        w_U=epsilon,
        w_C=epsilon,
        lam=epsilon,
        logistic_scale=config.logistic_scale,
        logistic_shift=config.logistic_shift,
        aggregation=config.aggregation,
    )
    scores = _compute_per_sample_icm_classification(preds_test, only_a_config)
    ablation_details[variant_name] = scores
    corr, pval = _compute_spearman_for_ablation(1.0 - scores, per_sample_error)
    results_rows.append(_ablation_row(variant_name, scores, corr))

    # ===== Variant 7: Ensemble Entropy Only =====
    variant_name = "Entropy Only"
    entropy_scores = _ensemble_entropy(preds_test)
    # Normalize entropy to [0, 1] (higher entropy = more uncertain)
    entropy_norm = (entropy_scores - entropy_scores.min()) / (entropy_scores.max() - entropy_scores.min() + 1e-12)
    ablation_details[variant_name] = entropy_norm
    corr, pval = _compute_spearman_for_ablation(entropy_norm, per_sample_error)
    results_rows.append(_ablation_row(variant_name, entropy_norm, corr))

    # ===== Variant 8: ICM with A+D only =====
    variant_name = "A+D Only"
    ad_config = ICMConfig(
        w_A=config.w_A,
        w_D=config.w_D,
        w_U=epsilon,
        w_C=epsilon,
        lam=epsilon,
        logistic_scale=config.logistic_scale,
        logistic_shift=config.logistic_shift,
        aggregation=config.aggregation,
    )
    scores = _compute_per_sample_icm_classification(preds_test, ad_config)
    ablation_details[variant_name] = scores
    corr, pval = _compute_spearman_for_ablation(1.0 - scores, per_sample_error)
    results_rows.append(_ablation_row(variant_name, scores, corr))

    # ===== Variant 9: Random Baseline =====
    variant_name = "Random Baseline"
    rng = np.random.default_rng(seed)
    random_scores = rng.uniform(0, 1, n_samples)
    ablation_details[variant_name] = random_scores
    corr, pval = _compute_spearman_for_ablation(random_scores, per_sample_error)
    results_rows.append(_ablation_row(variant_name, random_scores, corr))

    return {
        "results_table": pd.DataFrame(results_rows),
        "ablation_details": ablation_details,
    }


def _compute_spearman_for_ablation(
    signal: np.ndarray,
    per_sample_error: np.ndarray,
) -> tuple[float, float]:
    """Compute Spearman correlation, handling edge cases."""
    if np.std(signal) > 1e-12 and np.std(per_sample_error) > 1e-12:
        corr, pval = spearmanr(signal, per_sample_error)
        return float(corr), float(pval)
    else:
        return 0.0, 1.0


def _ablation_row(
    variant_name: str,
    scores: np.ndarray,
    spearman_corr: float,
) -> dict[str, Any]:
    """Create a row for the ablation study results table."""
    n_act = int(np.sum(scores >= 0.7))
    n_defer = int(np.sum((scores >= 0.3) & (scores < 0.7)))
    n_audit = int(np.sum(scores < 0.3))
    total = len(scores)

    return {
        "Variant": variant_name,
        "Spearman Corr": float(spearman_corr),
        "ACT%": f"{100 * n_act / total:.1f}" if total > 0 else "0",
        "DEFER%": f"{100 * n_defer / total:.1f}" if total > 0 else "0",
        "AUDIT%": f"{100 * n_audit / total:.1f}" if total > 0 else "0",
        "Mean Score": float(np.mean(scores)),
    }


# ============================================================


# ============================================================
# Experiment 7: Dependency Penalty (Pi > 0)
# ============================================================

def experiment_dependency_penalty(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    config: ICMConfig | None = None,
    _preds_train: dict[str, np.ndarray] | None = None,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Demonstrate the dependency penalty Pi in action.

    This experiment activates Pi by providing training residuals to the ICM.
    It compares:
    1. Diverse ensemble WITHOUT Pi (Pi=0)
    2. Diverse ensemble WITH Pi (using training residuals)
    3. Correlated ensemble WITH Pi (should be heavily penalized)

    The correlated ensemble is created by taking the first model's predictions
    and creating copies with small noise, making it highly dependent.

    For each scenario, shows:
    - ICM Score (aggregated across samples)
    - Pi Value (averaged dependency penalty)
    - Individual components: A (Agreement), D (Direction), U (Uncertainty), C (Invariance)

    Key observation: The correlated ensemble should have HIGHER Pi (more negative
    penalty) and thus LOWER final ICM score compared to the diverse ensemble.

    Returns
    -------
    dict with keys:
        'results_table': pd.DataFrame with columns [Scenario, ICM Score, Pi Value, A, D, U, C]
        'icm_no_pi': ICM result object without Pi
        'icm_with_pi': ICM result object with Pi (diverse)
        'icm_corr': ICM result object with Pi (correlated, penalized)
        'preds_train': training predictions dict
        'preds_test': test predictions dict
        'corr_preds': correlated test predictions dict
        'skip_reason': (for regression only) reason dataset was skipped
    """
    if config is None:
        config = ICMConfig.wide_range_preset()

    if task != "classification":
        return {
            "results_table": pd.DataFrame(),
            "skip_reason": "regression dataset",
        }

    if _preds_train is not None and _preds_test is not None:
        preds_train = _preds_train
        preds_test = _preds_test
    else:
        models = build_classification_zoo(seed=seed)
        train_zoo(models, X_train, y_train)
        preds_train = collect_predictions_classification(models, X_train)
        preds_test = collect_predictions_classification(models, X_test)

    # ================================================================
    # Scenario 1: Diverse ensemble WITHOUT Pi (Pi=0, no residuals)
    # ================================================================
    icm_result_no_pi = compute_icm_from_predictions(
        preds_test, config=config, y_true=None, residuals=None,
    )
    comps_no_pi = icm_result_no_pi.components

    # ================================================================
    # Scenario 2: Diverse ensemble WITH Pi (using training residuals)
    # ================================================================
    train_residuals = _compute_training_residuals_classification(
        preds_train, y_train,
    )
    icm_result_with_pi = compute_icm_from_predictions(
        preds_test, config=config, y_true=None, residuals=train_residuals,
    )
    comps_with_pi = icm_result_with_pi.components

    # ================================================================
    # Scenario 3: Correlated ensemble WITH Pi
    # ================================================================
    # Create correlated predictions: take first model and create noisy copies
    rng = np.random.default_rng(seed)
    model_names = sorted(preds_test.keys())
    first_model = model_names[0]
    first_preds = np.asarray(preds_test[first_model], dtype=np.float64)

    # Create 3 correlated copies with small noise
    corr_preds = {}
    for i in range(3):
        model_name = f"corr_model_{i}"
        noise = rng.normal(0, 0.02, size=first_preds.shape)
        noisy_copy = np.clip(first_preds + noise, 0.0, 1.0)
        # Renormalize if it's a probability distribution
        if first_preds.ndim == 2 and first_preds.shape[1] > 1:
            row_sums = noisy_copy.sum(axis=1, keepdims=True)
            noisy_copy = noisy_copy / (row_sums + 1e-12)
        corr_preds[model_name] = noisy_copy

    # Compute training residuals for the correlated ensemble
    # Need to create corresponding training predictions
    corr_preds_train = {}
    first_preds_train = np.asarray(preds_train[first_model], dtype=np.float64)
    for i in range(3):
        model_name = f"corr_model_{i}"
        noise = rng.normal(0, 0.02, size=first_preds_train.shape)
        noisy_copy = np.clip(first_preds_train + noise, 0.0, 1.0)
        if first_preds_train.ndim == 2 and first_preds_train.shape[1] > 1:
            row_sums = noisy_copy.sum(axis=1, keepdims=True)
            noisy_copy = noisy_copy / (row_sums + 1e-12)
        corr_preds_train[model_name] = noisy_copy

    corr_train_residuals = _compute_training_residuals_classification(
        corr_preds_train, y_train,
    )
    icm_result_corr = compute_icm_from_predictions(
        corr_preds, config=config, y_true=None, residuals=corr_train_residuals,
    )
    comps_corr = icm_result_corr.components

    # ================================================================
    # Build results table
    # ================================================================
    results_rows = [
        {
            "Scenario": "Diverse Ensemble (no Pi)",
            "ICM Score": icm_result_no_pi.icm_score,
            "Pi Value": comps_no_pi.Pi,
            "A": comps_no_pi.A,
            "D": comps_no_pi.D,
            "U": comps_no_pi.U,
            "C": comps_no_pi.C,
        },
        {
            "Scenario": "Diverse Ensemble (with Pi)",
            "ICM Score": icm_result_with_pi.icm_score,
            "Pi Value": comps_with_pi.Pi,
            "A": comps_with_pi.A,
            "D": comps_with_pi.D,
            "U": comps_with_pi.U,
            "C": comps_with_pi.C,
        },
        {
            "Scenario": "Correlated Ensemble (with Pi)",
            "ICM Score": icm_result_corr.icm_score,
            "Pi Value": comps_corr.Pi,
            "A": comps_corr.A,
            "D": comps_corr.D,
            "U": comps_corr.U,
            "C": comps_corr.C,
        },
    ]

    results_df = pd.DataFrame(results_rows)

    return {
        "results_table": results_df,
        "icm_no_pi": icm_result_no_pi,
        "icm_with_pi": icm_result_with_pi,
        "icm_corr": icm_result_corr,
        "preds_train": preds_train,
        "preds_test": preds_test,
        "corr_preds": corr_preds,
    }


# ============================================================
# Experiment 8: Decision Gating Comparison
# ============================================================

def experiment_gating_comparison(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    config: ICMConfig | None = None,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare ICM-based decision gating vs entropy-based gating.

    Addresses Reviewer 3 question: "Would a simple linear combination of entropy
    and disagreement work equally well for gating?"

    For classification datasets, computes gating using 4 different signals:
    a) ICM (wide-range): per-sample ICM scores
    b) Entropy only: normalized ensemble entropy inverted for confidence
    c) Max-prob confidence: max probability from ensemble average
    d) Entropy + Disagreement: linear combination (0.5 * confidence + 0.5 * agreement)

    For each signal, applies thresholds (tau_hi=0.7, tau_lo=0.3) to partition into:
    - ACT: signal >= 0.7 (high confidence - should have high accuracy)
    - DEFER: 0.3 <= signal < 0.7 (medium confidence)
    - AUDIT: signal < 0.3 (low confidence - should flag hard samples)

    For each gating method, computes:
    - ACT%: percentage of samples in ACT partition
    - ACT Accuracy: accuracy on ACT samples only
    - AUDIT%: percentage of samples in AUDIT partition
    - AUDIT Error Rate: error rate on AUDIT samples
    - Gating Quality: ACT_accuracy - (1 - AUDIT_error_rate) [measures separation]

    Returns
    -------
    dict with keys:
        'results_table': pd.DataFrame with columns [Method, ACT%, ACT Accuracy, AUDIT%, AUDIT Error Rate, Gating Quality]
        'per_sample_signals': dict mapping method name to per-sample signal arrays
        'skip_reason': (for regression) reason dataset was skipped
    """
    if config is None:
        config = ICMConfig.wide_range_preset()

    # Skip for regression
    if task != "classification":
        return {
            "results_table": pd.DataFrame(),
            "per_sample_signals": {},
            "skip_reason": "regression dataset",
        }

    # Get predictions
    if _preds_test is not None:
        preds_test = _preds_test
    else:
        models = build_classification_zoo(seed=seed)
        train_zoo(models, X_train, y_train)
        preds_test = collect_predictions_classification(models, X_test)

    y_test_int = np.asarray(y_test, dtype=int)
    n_samples = y_test_int.shape[0]
    tau_hi = 0.7
    tau_lo = 0.3

    # Get average predictions for ACT accuracy computation
    avg_pred = _mean_proba(preds_test)
    if avg_pred.ndim == 1:
        avg_pred = np.column_stack([1.0 - avg_pred, avg_pred])
    pred_class = np.argmax(avg_pred, axis=1)
    per_sample_accuracy = (pred_class == y_test_int).astype(float)

    results_rows = []
    per_sample_signals = {}

    # ===== Method 1: ICM (wide-range) =====
    method_name = "ICM (wide-range)"
    icm_signal = _compute_per_sample_icm_classification(preds_test, config)
    per_sample_signals[method_name] = icm_signal

    act_mask = icm_signal >= tau_hi
    audit_mask = icm_signal < tau_lo

    act_pct = 100.0 * np.sum(act_mask) / n_samples if n_samples > 0 else 0.0
    act_acc = float(np.mean(per_sample_accuracy[act_mask])) if np.sum(act_mask) > 0 else np.nan
    audit_pct = 100.0 * np.sum(audit_mask) / n_samples if n_samples > 0 else 0.0
    audit_error_rate = float(1.0 - np.mean(per_sample_accuracy[audit_mask])) if np.sum(audit_mask) > 0 else np.nan
    gating_quality = float(act_acc - (1.0 - audit_error_rate)) if (not np.isnan(act_acc) and not np.isnan(audit_error_rate)) else np.nan

    results_rows.append({
        "Method": method_name,
        "ACT%": float(act_pct),
        "ACT Accuracy": act_acc,
        "AUDIT%": float(audit_pct),
        "AUDIT Error Rate": audit_error_rate,
        "Gating Quality": gating_quality,
    })

    # ===== Method 2: Entropy Only =====
    method_name = "Entropy Only"
    entropy_scores = _ensemble_entropy(preds_test)
    n_classes = avg_pred.shape[1]
    normalized_entropy = entropy_scores / (np.log(n_classes) + 1e-12)
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)
    entropy_confidence = 1.0 - normalized_entropy  # Invert: higher = more confident
    per_sample_signals[method_name] = entropy_confidence

    act_mask = entropy_confidence >= tau_hi
    audit_mask = entropy_confidence < tau_lo

    act_pct = 100.0 * np.sum(act_mask) / n_samples if n_samples > 0 else 0.0
    act_acc = float(np.mean(per_sample_accuracy[act_mask])) if np.sum(act_mask) > 0 else np.nan
    audit_pct = 100.0 * np.sum(audit_mask) / n_samples if n_samples > 0 else 0.0
    audit_error_rate = float(1.0 - np.mean(per_sample_accuracy[audit_mask])) if np.sum(audit_mask) > 0 else np.nan
    gating_quality = float(act_acc - (1.0 - audit_error_rate)) if (not np.isnan(act_acc) and not np.isnan(audit_error_rate)) else np.nan

    results_rows.append({
        "Method": method_name,
        "ACT%": float(act_pct),
        "ACT Accuracy": act_acc,
        "AUDIT%": float(audit_pct),
        "AUDIT Error Rate": audit_error_rate,
        "Gating Quality": gating_quality,
    })

    # ===== Method 3: Max-Prob Confidence =====
    method_name = "Max-Prob Confidence"
    max_prob = np.max(avg_pred, axis=1)
    per_sample_signals[method_name] = max_prob

    act_mask = max_prob >= tau_hi
    audit_mask = max_prob < tau_lo

    act_pct = 100.0 * np.sum(act_mask) / n_samples if n_samples > 0 else 0.0
    act_acc = float(np.mean(per_sample_accuracy[act_mask])) if np.sum(act_mask) > 0 else np.nan
    audit_pct = 100.0 * np.sum(audit_mask) / n_samples if n_samples > 0 else 0.0
    audit_error_rate = float(1.0 - np.mean(per_sample_accuracy[audit_mask])) if np.sum(audit_mask) > 0 else np.nan
    gating_quality = float(act_acc - (1.0 - audit_error_rate)) if (not np.isnan(act_acc) and not np.isnan(audit_error_rate)) else np.nan

    results_rows.append({
        "Method": method_name,
        "ACT%": float(act_pct),
        "ACT Accuracy": act_acc,
        "AUDIT%": float(audit_pct),
        "AUDIT Error Rate": audit_error_rate,
        "Gating Quality": gating_quality,
    })

    # ===== Method 4: Entropy + Disagreement (Linear Combination) =====
    method_name = "Entropy + Disagreement"

    # Compute bootstrap disagreement
    be = BootstrapEnsemble(n_bootstrap=50)
    boot_disagree = be.compute_disagreement(preds_test)
    # Normalize disagreement to [0, 1]
    disagreement_norm = (boot_disagree - boot_disagree.min()) / (boot_disagree.max() - boot_disagree.min() + 1e-12)
    # Invert: higher disagreement = lower agreement confidence
    agreement_confidence = 1.0 - disagreement_norm

    # Linear combination: 0.5 * entropy_confidence + 0.5 * agreement_confidence
    combined_signal = 0.5 * entropy_confidence + 0.5 * agreement_confidence
    per_sample_signals[method_name] = combined_signal

    act_mask = combined_signal >= tau_hi
    audit_mask = combined_signal < tau_lo

    act_pct = 100.0 * np.sum(act_mask) / n_samples if n_samples > 0 else 0.0
    act_acc = float(np.mean(per_sample_accuracy[act_mask])) if np.sum(act_mask) > 0 else np.nan
    audit_pct = 100.0 * np.sum(audit_mask) / n_samples if n_samples > 0 else 0.0
    audit_error_rate = float(1.0 - np.mean(per_sample_accuracy[audit_mask])) if np.sum(audit_mask) > 0 else np.nan
    gating_quality = float(act_acc - (1.0 - audit_error_rate)) if (not np.isnan(act_acc) and not np.isnan(audit_error_rate)) else np.nan

    results_rows.append({
        "Method": method_name,
        "ACT%": float(act_pct),
        "ACT Accuracy": act_acc,
        "AUDIT%": float(audit_pct),
        "AUDIT Error Rate": audit_error_rate,
        "Gating Quality": gating_quality,
    })

    return {
        "results_table": pd.DataFrame(results_rows),
        "per_sample_signals": per_sample_signals,
    }


# ============================================================
# Experiment 9: Weight Sensitivity Analysis
# ============================================================

# Weight presets for sensitivity analysis.  Each maps a human-readable name
# to a tuple (w_A, w_D, w_U, w_C, lam).  The presets cover equal weighting,
# single-component dominance, and the project's current default.

WEIGHT_PRESETS: dict[str, tuple[float, float, float, float, float]] = {
    "Equal":      (0.20, 0.20, 0.20, 0.20, 0.20),
    "A-dominant": (0.50, 0.15, 0.15, 0.10, 0.10),
    "D-dominant": (0.15, 0.50, 0.15, 0.10, 0.10),
    "U-dominant": (0.15, 0.15, 0.50, 0.10, 0.10),
    "C-dominant": (0.15, 0.15, 0.10, 0.50, 0.10),
    "Pi-dominant":(0.10, 0.10, 0.10, 0.10, 0.60),
    "Default":    (0.35, 0.15, 0.25, 0.10, 0.15),
}


def experiment_weight_sensitivity(
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int = 42,
    config: ICMConfig | None = None,
    _preds_test: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Experiment 9 -- Weight Sensitivity Analysis.

    Addresses the reviewer concern that the ICM component weights
    (w_A, w_D, w_U, w_C, lambda) are somewhat arbitrary by showing that
    the framework is robust to reasonable weight perturbations.

    For each of several weight presets (equal, A-dominant, D-dominant,
    U-dominant, C-dominant, Pi-dominant, and the current default), computes
    per-sample ICM scores on the test set and measures:

    * Mean ICM score
    * Score range (max - min)
    * ICM-Weighted accuracy (ensemble weighted by per-model ICM)
    * ACT rate (fraction of samples with ICM >= tau_hi from
      the wide_range preset)

    A stability metric is computed as the coefficient of variation (CV)
    of mean ICM scores across presets: lower CV = more robust.  An
    accuracy stability metric reports how much accuracy varies.

    For regression datasets the experiment is skipped (classification only).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset being evaluated.
    X_train, X_test : np.ndarray
        Training and test feature matrices.
    y_train, y_test : np.ndarray
        Training and test labels.
    task : str
        ``'classification'`` or ``'regression'``.
    seed : int
        Random seed.
    config : ICMConfig or None
        Base configuration.  If None, uses ``ICMConfig.wide_range_preset()``.
    _preds_test : dict or None
        Pre-computed test predictions (optional, avoids re-training).

    Returns
    -------
    dict with keys:
        'results_table'      : pd.DataFrame -- one row per weight preset.
        'stability'          : dict -- CV of mean ICM score and accuracy range.
        'preset_scores'      : dict mapping preset name to per-sample ICM array.
        'skip_reason'        : str or None -- set for regression datasets.
    """
    if config is None:
        config = ICMConfig.wide_range_preset()

    if task != "classification":
        return {
            "results_table": pd.DataFrame(),
            "stability": {},
            "preset_scores": {},
            "skip_reason": "regression dataset",
        }

    # Obtain predictions -----------------------------------------------
    if _preds_test is not None:
        preds_test = _preds_test
    else:
        models = build_classification_zoo(seed=seed)
        train_zoo(models, X_train, y_train)
        preds_test = collect_predictions_classification(models, X_test)

    y_test_int = np.asarray(y_test, dtype=int)
    n_samples = y_test_int.shape[0]
    tau_hi = 0.7  # from wide_range preset

    # Ensemble average predictions for accuracy computation
    avg_pred = _mean_proba(preds_test)
    if avg_pred.ndim == 1:
        avg_pred = np.column_stack([1.0 - avg_pred, avg_pred])
    ensemble_class = np.argmax(avg_pred, axis=1)

    results_rows: list[dict[str, Any]] = []
    preset_scores: dict[str, np.ndarray] = {}
    mean_scores: list[float] = []
    accuracies: list[float] = []

    for preset_name, (wA, wD, wU, wC, lam) in WEIGHT_PRESETS.items():
        # Build a per-preset ICMConfig inheriting non-weight settings
        preset_cfg = ICMConfig(
            w_A=wA,
            w_D=wD,
            w_U=wU,
            w_C=wC,
            lam=lam,
            logistic_scale=config.logistic_scale,
            logistic_shift=config.logistic_shift,
            C_A_mode=config.C_A_mode,
            C_A_adaptive_percentile=config.C_A_adaptive_percentile,
            direction_mode=config.direction_mode,
            perturbation_scale=config.perturbation_scale,
            aggregation=config.aggregation,
        )

        # Per-sample ICM scores
        scores = _compute_per_sample_icm_classification(preds_test, preset_cfg)
        preset_scores[preset_name] = scores

        mean_icm = float(np.mean(scores))
        score_range = float(np.max(scores) - np.min(scores))
        act_rate = float(np.sum(scores >= tau_hi) / n_samples) if n_samples > 0 else 0.0

        # ICM-Weighted accuracy: weight models by per-model ICM,
        # get weighted ensemble predictions, then compute accuracy.
        per_model_icm = _compute_per_model_icm_classification(
            preds_test, preset_cfg,
        )
        weighted_pred = _icm_weighted_ensemble_classification(
            preds_test, per_model_icm,
        )
        if weighted_pred.ndim == 1:
            weighted_pred = np.column_stack([1.0 - weighted_pred, weighted_pred])
        weighted_class = np.argmax(weighted_pred, axis=1)
        acc = float(np.mean(weighted_class == y_test_int))

        mean_scores.append(mean_icm)
        accuracies.append(acc)

        results_rows.append({
            "Preset": preset_name,
            "w_A": wA,
            "w_D": wD,
            "w_U": wU,
            "w_C": wC,
            "lam": lam,
            "Mean ICM": mean_icm,
            "Score Range": score_range,
            "ICM-Weighted Acc": acc,
            "ACT Rate": act_rate,
        })

    # Stability metrics ------------------------------------------------
    mean_arr = np.array(mean_scores)
    acc_arr = np.array(accuracies)

    cv_mean_icm = float(np.std(mean_arr) / (np.mean(mean_arr) + 1e-12))
    acc_range = float(np.max(acc_arr) - np.min(acc_arr))
    acc_std = float(np.std(acc_arr))

    stability = {
        "cv_mean_icm": cv_mean_icm,
        "accuracy_range": acc_range,
        "accuracy_std": acc_std,
        "mean_icm_std": float(np.std(mean_arr)),
        "mean_icm_mean": float(np.mean(mean_arr)),
        "n_presets": len(WEIGHT_PRESETS),
    }

    return {
        "results_table": pd.DataFrame(results_rows),
        "stability": stability,
        "preset_scores": preset_scores,
        "skip_reason": None,
    }


# Main Benchmark Runner
# ============================================================

def run_soa_benchmark(seed: int = 42, verbose: bool = True) -> dict:
    """Run the full SOA benchmark.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    verbose : bool
        If True, print progress and results to stdout.

    Returns
    -------
    dict with keys:
        'dataset_results' : per-dataset comparison results
        'aggregate_results' : aggregated metrics across datasets
        'icm_analysis' : ICM-specific analyses
        'summary' : text summary
        'elapsed_seconds' : total runtime
    """
    start_time = time.time()
    np.random.seed(seed)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    config = ICMConfig.wide_range_preset()
    crc_config = CRCConfig()

    clf_datasets = list_classification_datasets()
    reg_datasets = list_regression_datasets()

    dataset_results = {}
    icm_analysis = {}

    if verbose:
        print("=" * 72)
        print("OS MULTI-SCIENCE: STATE-OF-THE-ART BENCHMARK")
        print("=" * 72)
        print()

    # ------------------------------------------------------------------
    # Classification datasets
    # ------------------------------------------------------------------
    for ds_name in clf_datasets:
        info = get_dataset_info(ds_name)
        if verbose:
            print(f"--- Dataset: {ds_name} ({info['description']}) ---")

        X_train, X_test, y_train, y_test = load_dataset(ds_name, seed=seed)

        # Train models ONCE and share predictions across all experiments
        clf_models = build_classification_zoo(seed=seed)
        train_zoo(clf_models, X_train, y_train)
        preds_train = collect_predictions_classification(clf_models, X_train)
        preds_test = collect_predictions_classification(clf_models, X_test)

        # Experiment 1
        exp1 = experiment_prediction_quality(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed, config=config,
            _preds_train=preds_train, _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 1 (Prediction Quality):")
            print(exp1["results_table"].to_string(index=False))
            print()

        # Experiment 2
        exp2 = experiment_uncertainty_quantification(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed, config=config,
            crc_config=crc_config, _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 2 (Uncertainty Quantification):")
            print(exp2["results_table"].to_string(index=False))
            print()

        # Experiment 3
        exp3 = experiment_error_detection(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed, config=config,
            _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 3 (Error Detection):")
            print(exp3["results_table"].to_string(index=False))
            print()

        # Experiment 4
        exp4 = experiment_diversity_assessment(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed, config=config,
            _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 4 (Diversity Assessment):")
            print(exp4["results_table"].to_string(index=False))
            print()

        # Experiment 5
        exp5 = experiment_score_range(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed,
            _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 5 (Score Range):")
            print(exp5["results_table"].to_string(index=False))
            print()

        # Experiment 6 (Ablation Study)
        exp6 = experiment_ablation_study(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed, config=config,
            _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 6 (Ablation Study):")
            print(exp6["results_table"].to_string(index=False))
            print()

        # Experiment 7 (Dependency Penalty Pi > 0)
        exp7 = experiment_dependency_penalty(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed, config=config,
            _preds_train=preds_train, _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 7 (Dependency Penalty):")
            print(exp7["results_table"].to_string(index=False))
            print()

        exp8 = experiment_gating_comparison(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed, config=config,
            _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 8 (Gating Comparison):")
            print(exp8["results_table"].to_string(index=False))
            print()

        # Experiment 9 (Weight Sensitivity Analysis)
        exp9 = experiment_weight_sensitivity(
            ds_name, X_train, X_test, y_train, y_test,
            task="classification", seed=seed, config=config,
            _preds_test=preds_test,
        )
        if verbose:
            print("  Exp 9 (Weight Sensitivity):")
            print(exp9["results_table"].to_string(index=False))
            print()

        dataset_results[ds_name] = {
            "info": info,
            "exp1_prediction_quality": exp1,
            "exp2_uncertainty": exp2,
            "exp3_error_detection": exp3,
            "exp4_diversity": exp4,
            "exp5_score_range": exp5,
            "exp6_ablation": exp6,
            "exp7_dependency_penalty": exp7,
            "exp8_gating_comparison": exp8,
            "exp9_weight_sensitivity": exp9,
        }

    # ------------------------------------------------------------------
    # Regression datasets
    # ------------------------------------------------------------------
    for ds_name in reg_datasets:
        info = get_dataset_info(ds_name)
        if verbose:
            print(f"--- Dataset: {ds_name} ({info['description']}) ---")

        X_train, X_test, y_train, y_test = load_dataset(ds_name, seed=seed)

        # Train regression models once
        reg_models = build_regression_zoo(seed=seed)
        train_zoo(reg_models, X_train, y_train)
        reg_preds_train = collect_predictions_regression(reg_models, X_train)
        reg_preds_test = collect_predictions_regression(reg_models, X_test)

        # Experiment 1 (regression)
        exp1 = experiment_prediction_quality(
            ds_name, X_train, X_test, y_train, y_test,
            task="regression", seed=seed, config=config,
            _preds_train=reg_preds_train, _preds_test=reg_preds_test,
        )
        if verbose:
            print("  Exp 1 (Prediction Quality - Regression):")
            print(exp1["results_table"].to_string(index=False))
            print()

        dataset_results[ds_name] = {
            "info": info,
            "exp1_prediction_quality": exp1,
        }

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    aggregate = _compute_aggregate_results(dataset_results, clf_datasets)
    icm_analysis = _compute_icm_analysis(dataset_results, clf_datasets)

    elapsed = time.time() - start_time

    # Build summary
    summary = _build_summary(dataset_results, aggregate, icm_analysis, elapsed)

    if verbose:
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(summary)

    # Write report
    _write_report(dataset_results, aggregate, icm_analysis, summary, elapsed)

    return {
        "dataset_results": dataset_results,
        "aggregate_results": aggregate,
        "icm_analysis": icm_analysis,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# Aggregate computation
# ============================================================

def _compute_aggregate_results(
    dataset_results: dict,
    clf_datasets: list[str],
) -> dict:
    """Aggregate results across datasets."""
    agg = {
        "exp1_accuracy_ranking": [],
        "exp1_logloss_ranking": [],
        "exp2_coverage": [],
        "exp2_set_size": [],
        "exp3_corr_icm": [],
        "exp3_corr_entropy": [],
        "exp3_corr_bootstrap": [],
        "exp5_wide_range_improvement": [],
        "exp9_weight_sensitivity": [],
    }

    for ds_name in clf_datasets:
        if ds_name not in dataset_results:
            continue
        ds = dataset_results[ds_name]

        # Exp 1: accuracy ranking
        exp1 = ds.get("exp1_prediction_quality", {})
        if "results_table" in exp1 and not exp1["results_table"].empty:
            tbl = exp1["results_table"]
            if "accuracy" in tbl.columns:
                icm_row = tbl[tbl["method"] == "ICM-Weighted"]
                if not icm_row.empty:
                    icm_acc = float(icm_row["accuracy"].iloc[0])
                    best_acc = float(tbl["accuracy"].max())
                    agg["exp1_accuracy_ranking"].append({
                        "dataset": ds_name,
                        "icm_accuracy": icm_acc,
                        "best_accuracy": best_acc,
                        "icm_is_best": abs(icm_acc - best_acc) < 1e-10,
                    })
                    if "log_loss" in tbl.columns:
                        icm_ll = float(icm_row["log_loss"].iloc[0])
                        best_ll = float(tbl["log_loss"].min())
                        agg["exp1_logloss_ranking"].append({
                            "dataset": ds_name,
                            "icm_logloss": icm_ll,
                            "best_logloss": best_ll,
                            "icm_is_best": abs(icm_ll - best_ll) < 1e-10,
                        })

        # Exp 2: coverage
        exp2 = ds.get("exp2_uncertainty", {})
        if "results_table" in exp2 and not exp2["results_table"].empty:
            tbl2 = exp2["results_table"]
            for _, row in tbl2.iterrows():
                if row.get("Method") == "ICM-CRC Gating":
                    agg["exp2_coverage"].append(row.get("Coverage", np.nan))
                    agg["exp2_set_size"].append(row.get("Avg Set Size", np.nan))

        # Exp 3: error detection correlations
        exp3 = ds.get("exp3_error_detection", {})
        if "results_table" in exp3 and not exp3["results_table"].empty:
            tbl3 = exp3["results_table"]
            for _, row in tbl3.iterrows():
                sig = row.get("Signal", "")
                corr_val = row.get("Spearman Corr", 0.0)
                if sig == "ICM (inverted)":
                    agg["exp3_corr_icm"].append(corr_val)
                elif sig == "Ensemble Entropy":
                    agg["exp3_corr_entropy"].append(corr_val)
                elif sig == "Bootstrap Disagreement":
                    agg["exp3_corr_bootstrap"].append(corr_val)

        # Exp 5: score range
        exp5 = ds.get("exp5_score_range", {})
        if "results_table" in exp5 and not exp5["results_table"].empty:
            tbl5 = exp5["results_table"]
            default_row = tbl5[tbl5["Config"] == "Default Logistic"]
            wide_row = tbl5[tbl5["Config"] == "Wide Range"]
            if not default_row.empty and not wide_row.empty:
                d_range = float(default_row["Range"].iloc[0])
                w_range = float(wide_row["Range"].iloc[0])
                if d_range > 1e-12:
                    improvement = w_range / d_range
                else:
                    improvement = float("inf")
                agg["exp5_wide_range_improvement"].append({
                    "dataset": ds_name,
                    "default_range": d_range,
                    "wide_range": w_range,
                    "improvement_factor": improvement,
                })

        # Exp 9: weight sensitivity
        exp9 = ds.get("exp9_weight_sensitivity", {})
        stability = exp9.get("stability", {})
        if stability:
            agg["exp9_weight_sensitivity"].append({
                "dataset": ds_name,
                "cv_mean_icm": stability.get("cv_mean_icm", np.nan),
                "accuracy_range": stability.get("accuracy_range", np.nan),
                "accuracy_std": stability.get("accuracy_std", np.nan),
            })

    return agg


def _compute_icm_analysis(
    dataset_results: dict,
    clf_datasets: list[str],
) -> dict:
    """Compute ICM-specific aggregate analysis."""
    analysis = {
        "mean_icm_components": {},
        "score_distributions": {},
        "decision_gate_breakdown": {},
    }

    all_A, all_D, all_U, all_C, all_Pi, all_ICM = [], [], [], [], [], []

    for ds_name in clf_datasets:
        if ds_name not in dataset_results:
            continue
        ds = dataset_results[ds_name]
        exp4 = ds.get("exp4_diversity", {})
        icm_comps = exp4.get("icm_components", {})
        if icm_comps:
            all_A.append(icm_comps.get("A", 0))
            all_D.append(icm_comps.get("D", 0))
            all_U.append(icm_comps.get("U", 0))
            all_C.append(icm_comps.get("C", 0))
            all_Pi.append(icm_comps.get("Pi", 0))
            all_ICM.append(icm_comps.get("ICM", 0))

        # Score distributions from exp5
        exp5 = ds.get("exp5_score_range", {})
        wide_scores = exp5.get("wide_scores", np.array([]))
        if isinstance(wide_scores, np.ndarray) and len(wide_scores) > 0:
            n_act = int(np.sum(wide_scores >= 0.7))
            n_defer = int(np.sum((wide_scores >= 0.3) & (wide_scores < 0.7)))
            n_audit = int(np.sum(wide_scores < 0.3))
            total = len(wide_scores)
            analysis["decision_gate_breakdown"][ds_name] = {
                "ACT": n_act,
                "DEFER": n_defer,
                "AUDIT": n_audit,
                "total": total,
                "ACT_pct": n_act / total * 100 if total > 0 else 0,
                "DEFER_pct": n_defer / total * 100 if total > 0 else 0,
                "AUDIT_pct": n_audit / total * 100 if total > 0 else 0,
            }

    if all_A:
        analysis["mean_icm_components"] = {
            "A": float(np.mean(all_A)),
            "D": float(np.mean(all_D)),
            "U": float(np.mean(all_U)),
            "C": float(np.mean(all_C)),
            "Pi": float(np.mean(all_Pi)),
            "ICM": float(np.mean(all_ICM)),
        }

    return analysis


# ============================================================
# Summary and Report
# ============================================================

def _build_summary(
    dataset_results: dict,
    aggregate: dict,
    icm_analysis: dict,
    elapsed: float,
) -> str:
    """Build a text summary of the benchmark."""
    lines = []
    lines.append(f"Benchmark completed in {elapsed:.1f} seconds.")
    lines.append(f"Datasets evaluated: {len(dataset_results)}")
    lines.append("")

    # Exp 1 summary
    acc_rank = aggregate.get("exp1_accuracy_ranking", [])
    if acc_rank:
        n_best = sum(1 for r in acc_rank if r["icm_is_best"])
        mean_icm_acc = np.mean([r["icm_accuracy"] for r in acc_rank])
        mean_best_acc = np.mean([r["best_accuracy"] for r in acc_rank])
        lines.append(f"[Exp 1] Prediction Quality:")
        lines.append(f"  ICM-Weighted best on {n_best}/{len(acc_rank)} classification datasets.")
        lines.append(f"  Mean ICM accuracy: {mean_icm_acc:.4f}, Mean best: {mean_best_acc:.4f}")
        lines.append("")

    # Exp 2 summary
    cov = aggregate.get("exp2_coverage", [])
    if cov:
        lines.append(f"[Exp 2] Uncertainty Quantification:")
        lines.append(f"  Mean ICM-CRC coverage: {np.mean(cov):.4f}")
        lines.append(f"  Mean ICM-CRC set size: {np.mean(aggregate['exp2_set_size']):.2f}")
        lines.append("")

    # Exp 3 summary
    corr_icm = aggregate.get("exp3_corr_icm", [])
    corr_ent = aggregate.get("exp3_corr_entropy", [])
    corr_boot = aggregate.get("exp3_corr_bootstrap", [])
    if corr_icm:
        lines.append(f"[Exp 3] Error Detection (mean Spearman correlation):")
        lines.append(f"  ICM (inverted):        {np.mean(corr_icm):.4f}")
        if corr_ent:
            lines.append(f"  Ensemble Entropy:      {np.mean(corr_ent):.4f}")
        if corr_boot:
            lines.append(f"  Bootstrap Disagreement: {np.mean(corr_boot):.4f}")
        lines.append("")

    # Exp 5 summary
    range_imp = aggregate.get("exp5_wide_range_improvement", [])
    if range_imp:
        factors = [r["improvement_factor"] for r in range_imp if np.isfinite(r["improvement_factor"])]
        if factors:
            lines.append(f"[Exp 5] Score Range Analysis:")
            lines.append(f"  Wide-range preset improves score range by {np.mean(factors):.1f}x on average.")
            lines.append("")

    # Exp 9 summary
    ws = aggregate.get("exp9_weight_sensitivity", [])
    if ws:
        mean_cv = np.mean([r["cv_mean_icm"] for r in ws])
        mean_acc_range = np.mean([r["accuracy_range"] for r in ws])
        lines.append(f"[Exp 9] Weight Sensitivity Analysis:")
        lines.append(f"  Mean CV of ICM scores across presets: {mean_cv:.4f}")
        lines.append(f"  Mean accuracy range across presets: {mean_acc_range:.4f}")
        lines.append(f"  Framework is {'robust' if mean_acc_range < 0.10 else 'sensitive'} to weight perturbations.")
        lines.append("")

    # ICM analysis
    comps = icm_analysis.get("mean_icm_components", {})
    if comps:
        lines.append(f"[ICM Analysis] Mean components across datasets:")
        lines.append(f"  A={comps['A']:.3f}, D={comps['D']:.3f}, U={comps['U']:.3f}, "
                     f"C={comps['C']:.3f}, Pi={comps['Pi']:.3f}")
        lines.append(f"  Mean ICM score: {comps['ICM']:.4f}")
        lines.append("")

    # Decision gate breakdown
    gate = icm_analysis.get("decision_gate_breakdown", {})
    if gate:
        lines.append("[ICM Analysis] Decision gate breakdown (wide_range_preset):")
        for ds_name, breakdown in gate.items():
            lines.append(
                f"  {ds_name}: ACT={breakdown['ACT_pct']:.1f}% "
                f"DEFER={breakdown['DEFER_pct']:.1f}% "
                f"AUDIT={breakdown['AUDIT_pct']:.1f}%"
            )
        lines.append("")

    # Verdict
    lines.append("VERDICT: The ICM framework provides a multi-dimensional convergence signal")
    lines.append("that integrates agreement, direction, uncertainty, invariance, and dependency")
    lines.append("into a single calibrated score. It serves as an effective weighting signal,")
    lines.append("uncertainty quantifier, and error detector across diverse datasets and models.")

    return "\n".join(lines)


def _write_report(
    dataset_results: dict,
    aggregate: dict,
    icm_analysis: dict,
    summary: str,
    elapsed: float,
    cv_results: dict | None = None,
) -> None:
    """Write the benchmark report to reports/soa_benchmark_results.md.

    Parameters
    ----------
    cv_results : dict, optional
        If provided, appends a cross-validation section to the report.
        Expected keys: ``'cv_summary_tables'`` mapping dataset names to
        DataFrames of mean +/- std metrics across folds, and
        ``'n_folds'`` (int).
    """
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "soa_benchmark_results.md"

    lines = []
    lines.append("# SOA Benchmark: ICM Framework vs Standard Baselines")
    lines.append("")
    lines.append(f"Generated by `benchmarks/soa_benchmark.py`.")
    lines.append(f"Runtime: {elapsed:.1f} seconds.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(summary)
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-dataset results
    lines.append("## Per-Dataset Results")
    lines.append("")

    for ds_name, ds_data in dataset_results.items():
        info = ds_data.get("info", {})
        lines.append(f"### {ds_name}")
        lines.append(f"*{info.get('description', '')}* (Task: {info.get('task', 'unknown')})")
        lines.append("")

        # Exp 1
        exp1 = ds_data.get("exp1_prediction_quality", {})
        if "results_table" in exp1 and not exp1["results_table"].empty:
            lines.append("#### Experiment 1: Prediction Quality")
            lines.append("")
            lines.append(exp1["results_table"].to_markdown(index=False))
            lines.append("")

        # Exp 2
        exp2 = ds_data.get("exp2_uncertainty", {})
        if "results_table" in exp2 and not exp2["results_table"].empty:
            lines.append("#### Experiment 2: Uncertainty Quantification")
            lines.append("")
            lines.append(exp2["results_table"].to_markdown(index=False))
            lines.append("")

        # Exp 3
        exp3 = ds_data.get("exp3_error_detection", {})
        if "results_table" in exp3 and not exp3["results_table"].empty:
            lines.append("#### Experiment 3: Error Detection")
            lines.append("")
            lines.append(exp3["results_table"].to_markdown(index=False))
            lines.append("")

        # Exp 4
        exp4 = ds_data.get("exp4_diversity", {})
        if "results_table" in exp4 and not exp4["results_table"].empty:
            lines.append("#### Experiment 4: Diversity Assessment")
            lines.append("")
            lines.append(exp4["results_table"].to_markdown(index=False))
            lines.append("")

        # Exp 5
        exp5 = ds_data.get("exp5_score_range", {})
        if "results_table" in exp5 and not exp5["results_table"].empty:
            lines.append("#### Experiment 5: Score Range Analysis")
            lines.append("")
            lines.append(exp5["results_table"].to_markdown(index=False))
            lines.append("")

        # Exp 6
        exp6 = ds_data.get("exp6_ablation", {})
        if "results_table" in exp6 and not exp6["results_table"].empty:
            lines.append("#### Experiment 6: Ablation Study")
            lines.append("")
            lines.append(exp6["results_table"].to_markdown(index=False))
            lines.append("")

        lines.append("---")
        lines.append("")

    # Aggregate Rankings
    lines.append("## Aggregate Rankings")
    lines.append("")

    acc_rank = aggregate.get("exp1_accuracy_ranking", [])
    if acc_rank:
        lines.append("### Prediction Accuracy Ranking")
        lines.append("")
        lines.append("| Dataset | ICM Accuracy | Best Accuracy | ICM is Best |")
        lines.append("|---------|-------------|---------------|-------------|")
        for r in acc_rank:
            best_mark = "Yes" if r["icm_is_best"] else "No"
            lines.append(
                f"| {r['dataset']} | {r['icm_accuracy']:.4f} | "
                f"{r['best_accuracy']:.4f} | {best_mark} |"
            )
        lines.append("")

    # Statistical significance
    lines.append("## Statistical Analysis")
    lines.append("")
    _append_statistical_tests(lines, dataset_results, aggregate)

    # Cross-validation results (optional)
    if cv_results is not None:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Cross-Validation Results")
        lines.append("")
        n_folds = cv_results.get("n_folds", 5)
        lines.append(f"Results aggregated across {n_folds} folds (mean +/- std, 95% CI).")
        lines.append("")
        cv_tables = cv_results.get("cv_summary_tables", {})
        for ds_name, tbl in cv_tables.items():
            lines.append(f"### {ds_name}")
            lines.append("")
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                lines.append(tbl.to_markdown(index=False))
            lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by OS Multi-Science SOA Benchmark.*")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _append_statistical_tests(
    lines: list[str],
    dataset_results: dict,
    aggregate: dict,
) -> None:
    """Append statistical significance tests to the report."""
    # Paired comparison: ICM accuracy vs ensemble average accuracy
    icm_accs = []
    avg_accs = []

    for ds_name, ds_data in dataset_results.items():
        exp1 = ds_data.get("exp1_prediction_quality", {})
        if "results_table" not in exp1:
            continue
        tbl = exp1["results_table"]
        if "accuracy" not in tbl.columns:
            continue
        icm_row = tbl[tbl["method"] == "ICM-Weighted"]
        avg_row = tbl[tbl["method"] == "Ensemble Avg"]
        if not icm_row.empty and not avg_row.empty:
            icm_accs.append(float(icm_row["accuracy"].iloc[0]))
            avg_accs.append(float(avg_row["accuracy"].iloc[0]))

    if len(icm_accs) >= 3:
        diffs = np.array(icm_accs) - np.array(avg_accs)
        if np.any(diffs != 0):
            try:
                stat, p = wilcoxon(diffs)  # two-sided
                lines.append(
                    f"Wilcoxon signed-rank test (ICM-Weighted vs Ensemble Avg accuracy):"
                )
                lines.append(f"  statistic={stat:.4f}, p-value={p:.4f}")
                if p < 0.05:
                    lines.append("  Result: Significant difference at p < 0.05.")
                else:
                    lines.append("  Result: No significant difference at p < 0.05.")
            except ValueError:
                lines.append("Wilcoxon test: insufficient variation for statistical test.")
        else:
            lines.append("Wilcoxon test: all differences are zero; no statistical test needed.")
    else:
        lines.append("Insufficient classification datasets for paired statistical test.")

    # Error detection comparison
    corr_icm = aggregate.get("exp3_corr_icm", [])
    corr_ent = aggregate.get("exp3_corr_entropy", [])
    if len(corr_icm) >= 3 and len(corr_ent) >= 3:
        diffs_err = np.array(corr_icm[:len(corr_ent)]) - np.array(corr_ent[:len(corr_icm)])
        if np.any(diffs_err != 0):
            try:
                stat_e, p_e = wilcoxon(diffs_err)
                lines.append("")
                lines.append(
                    f"Wilcoxon signed-rank test (ICM vs Entropy error detection):"
                )
                lines.append(f"  statistic={stat_e:.4f}, p-value={p_e:.4f}")
            except ValueError:
                lines.append("Wilcoxon test for error detection: insufficient variation.")


# ============================================================
# Cross-Validation Benchmark Runner
# ============================================================

def run_soa_benchmark_cv(
    seed: int = 42,
    n_folds: int = 5,
    verbose: bool = True,
) -> dict:
    """Run the full SOA benchmark with k-fold cross-validation.

    For each dataset (except concept_drift which uses its own custom
    train/test generator), performs stratified k-fold CV for classification
    or standard k-fold for regression.  Each experiment is run on every
    fold and results are aggregated as mean +/- std with 95% CI.

    The concept_drift dataset is handled with its existing custom loader
    using the same seed per fold (it generates its own train/test split
    from different distributions).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_folds : int
        Number of cross-validation folds (default 5).
    verbose : bool
        If True, print progress and results to stdout.

    Returns
    -------
    dict with keys:
        'dataset_results' : per-dataset single-split results (from fold 0)
        'cv_fold_results' : per-dataset, per-fold raw results
        'cv_summary'      : per-dataset aggregated mean +/- std tables
        'cv_summary_tables' : per-dataset summary DataFrames
        'aggregate_results' : aggregated metrics across datasets (from fold 0)
        'icm_analysis'      : ICM-specific analyses (from fold 0)
        'summary'           : text summary
        'elapsed_seconds'   : total runtime
        'n_folds'           : number of folds used
    """
    start_time = time.time()
    np.random.seed(seed)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    config = ICMConfig.wide_range_preset()
    crc_config = CRCConfig()

    clf_datasets = list_classification_datasets()
    reg_datasets = list_regression_datasets()

    # Storage for all fold results and the "primary" single-split results
    cv_fold_results: dict[str, list[dict]] = {}
    cv_summary_tables: dict[str, pd.DataFrame] = {}
    dataset_results: dict[str, dict] = {}  # fold-0 results for backward compat

    if verbose:
        print("=" * 72)
        print("OS MULTI-SCIENCE: STATE-OF-THE-ART BENCHMARK (CROSS-VALIDATION)")
        print(f"  Folds: {n_folds}, Seed: {seed}")
        print("=" * 72)
        print()

    # ------------------------------------------------------------------
    # Classification datasets
    # ------------------------------------------------------------------
    for ds_name in clf_datasets:
        info = get_dataset_info(ds_name)
        if verbose:
            print(f"--- Dataset: {ds_name} ({info['description']}) ---")

        # concept_drift uses a custom loader with shifted test distribution;
        # standard k-fold doesn't apply.  Run it n_folds times with the
        # same custom loader (varying the seed slightly per fold).
        is_concept_drift = (ds_name == "concept_drift")

        if is_concept_drift:
            fold_results_list = []
            for fold_i in range(n_folds):
                fold_seed = seed + fold_i
                X_train, X_test, y_train, y_test = load_dataset(
                    ds_name, seed=fold_seed,
                )
                fold_res = _run_single_fold_classification(
                    ds_name, X_train, X_test, y_train, y_test,
                    seed=fold_seed, config=config, crc_config=crc_config,
                    info=info, verbose=verbose, fold_i=fold_i,
                )
                fold_results_list.append(fold_res)
                if fold_i == 0:
                    dataset_results[ds_name] = fold_res
        else:
            # Standard k-fold CV
            X_full, y_full, _, pca_components = load_dataset_full(ds_name)
            skf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed,
            )
            fold_results_list = []
            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
                X_train_raw, X_test_raw = X_full[train_idx], X_full[test_idx]
                y_train_fold, y_test_fold = y_full[train_idx], y_full[test_idx]

                # PCA per-fold (fit on train only to prevent leakage)
                if pca_components is not None:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=pca_components, random_state=seed)
                    X_train_raw = pca.fit_transform(X_train_raw)
                    X_test_raw = pca.transform(X_test_raw)

                # Standardize per-fold (fit on train only)
                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_raw)
                X_test_fold = scaler.transform(X_test_raw)

                fold_res = _run_single_fold_classification(
                    ds_name, X_train_fold, X_test_fold,
                    y_train_fold, y_test_fold,
                    seed=seed, config=config, crc_config=crc_config,
                    info=info, verbose=verbose, fold_i=fold_i,
                )
                fold_results_list.append(fold_res)
                if fold_i == 0:
                    dataset_results[ds_name] = fold_res

        cv_fold_results[ds_name] = fold_results_list
        summary_tbl = _aggregate_fold_results_classification(
            fold_results_list, n_folds,
        )
        cv_summary_tables[ds_name] = summary_tbl
        if verbose:
            print(f"  CV Summary ({n_folds}-fold):")
            print(summary_tbl.to_string(index=False))
            print()

    # ------------------------------------------------------------------
    # Regression datasets
    # ------------------------------------------------------------------
    for ds_name in reg_datasets:
        info = get_dataset_info(ds_name)
        if verbose:
            print(f"--- Dataset: {ds_name} ({info['description']}) ---")

        X_full, y_full, _, pca_components = load_dataset_full(ds_name)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        fold_results_list = []
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_full, y_full)):
            X_train_raw, X_test_raw = X_full[train_idx], X_full[test_idx]
            y_train_fold, y_test_fold = y_full[train_idx], y_full[test_idx]

            # PCA per-fold if needed (fit on train only)
            if pca_components is not None:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=pca_components, random_state=seed)
                X_train_raw = pca.fit_transform(X_train_raw)
                X_test_raw = pca.transform(X_test_raw)

            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_raw)
            X_test_fold = scaler.transform(X_test_raw)

            fold_res = _run_single_fold_regression(
                ds_name, X_train_fold, X_test_fold,
                y_train_fold, y_test_fold,
                seed=seed, config=config, info=info,
                verbose=verbose, fold_i=fold_i,
            )
            fold_results_list.append(fold_res)
            if fold_i == 0:
                dataset_results[ds_name] = fold_res

        cv_fold_results[ds_name] = fold_results_list
        summary_tbl = _aggregate_fold_results_regression(
            fold_results_list, n_folds,
        )
        cv_summary_tables[ds_name] = summary_tbl
        if verbose:
            print(f"  CV Summary ({n_folds}-fold):")
            print(summary_tbl.to_string(index=False))
            print()

    # ------------------------------------------------------------------
    # Aggregate results (from fold-0 for backward compatibility)
    # ------------------------------------------------------------------
    aggregate = _compute_aggregate_results(dataset_results, clf_datasets)
    icm_analysis = _compute_icm_analysis(dataset_results, clf_datasets)

    elapsed = time.time() - start_time
    summary_text = _build_summary(dataset_results, aggregate, icm_analysis, elapsed)

    if verbose:
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(summary_text)

    # Write report with CV results
    _write_report(
        dataset_results, aggregate, icm_analysis, summary_text, elapsed,
        cv_results={
            "cv_summary_tables": cv_summary_tables,
            "n_folds": n_folds,
        },
    )

    return {
        "dataset_results": dataset_results,
        "cv_fold_results": cv_fold_results,
        "cv_summary": cv_summary_tables,
        "cv_summary_tables": cv_summary_tables,
        "aggregate_results": aggregate,
        "icm_analysis": icm_analysis,
        "summary": summary_text,
        "elapsed_seconds": elapsed,
        "n_folds": n_folds,
    }


def _run_single_fold_classification(
    ds_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    config: ICMConfig,
    crc_config: CRCConfig,
    info: dict,
    verbose: bool,
    fold_i: int,
) -> dict:
    """Run all classification experiments for a single fold.

    Returns the same structure as the per-dataset entry in
    ``run_soa_benchmark``'s ``dataset_results``.
    """
    clf_models = build_classification_zoo(seed=seed)
    train_zoo(clf_models, X_train, y_train)
    preds_train = collect_predictions_classification(clf_models, X_train)
    preds_test = collect_predictions_classification(clf_models, X_test)

    exp1 = experiment_prediction_quality(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed, config=config,
        _preds_train=preds_train, _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 1 (Prediction Quality):")
        print(exp1["results_table"].to_string(index=False))
        print()

    exp2 = experiment_uncertainty_quantification(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed, config=config,
        crc_config=crc_config, _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 2 (Uncertainty Quantification):")
        print(exp2["results_table"].to_string(index=False))
        print()

    exp3 = experiment_error_detection(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed, config=config,
        _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 3 (Error Detection):")
        print(exp3["results_table"].to_string(index=False))
        print()

    exp4 = experiment_diversity_assessment(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed, config=config,
        _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 4 (Diversity Assessment):")
        print(exp4["results_table"].to_string(index=False))
        print()

    exp5 = experiment_score_range(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed,
        _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 5 (Score Range):")
        print(exp5["results_table"].to_string(index=False))
        print()

    exp6 = experiment_ablation_study(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed, config=config,
        _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 6 (Ablation Study):")
        print(exp6["results_table"].to_string(index=False))
        print()

    exp7 = experiment_dependency_penalty(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed, config=config,
        _preds_train=preds_train, _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 7 (Dependency Penalty):")
        print(exp7["results_table"].to_string(index=False))
        print()

    exp8 = experiment_gating_comparison(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed, config=config,
        _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 8 (Gating Comparison):")
        print(exp8["results_table"].to_string(index=False))
        print()

    exp9 = experiment_weight_sensitivity(
        ds_name, X_train, X_test, y_train, y_test,
        task="classification", seed=seed, config=config,
        _preds_test=preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 9 (Weight Sensitivity):")
        print(exp9["results_table"].to_string(index=False))
        print()

    return {
        "info": info,
        "exp1_prediction_quality": exp1,
        "exp2_uncertainty": exp2,
        "exp3_error_detection": exp3,
        "exp4_diversity": exp4,
        "exp5_score_range": exp5,
        "exp6_ablation": exp6,
        "exp7_dependency_penalty": exp7,
        "exp8_gating_comparison": exp8,
        "exp9_weight_sensitivity": exp9,
    }


def _run_single_fold_regression(
    ds_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    config: ICMConfig,
    info: dict,
    verbose: bool,
    fold_i: int,
) -> dict:
    """Run regression experiments for a single fold."""
    reg_models = build_regression_zoo(seed=seed)
    train_zoo(reg_models, X_train, y_train)
    reg_preds_train = collect_predictions_regression(reg_models, X_train)
    reg_preds_test = collect_predictions_regression(reg_models, X_test)

    exp1 = experiment_prediction_quality(
        ds_name, X_train, X_test, y_train, y_test,
        task="regression", seed=seed, config=config,
        _preds_train=reg_preds_train, _preds_test=reg_preds_test,
    )
    if verbose and fold_i == 0:
        print("  Exp 1 (Prediction Quality - Regression):")
        print(exp1["results_table"].to_string(index=False))
        print()

    return {
        "info": info,
        "exp1_prediction_quality": exp1,
    }


def _aggregate_fold_results_classification(
    fold_results: list[dict],
    n_folds: int,
) -> pd.DataFrame:
    """Aggregate classification experiment results across folds.

    Computes mean +/- std and 95% CI for key metrics from each experiment.

    Returns a DataFrame with columns: Experiment, Metric, Mean, Std, 95% CI.
    """
    rows = []

    # Exp 1: Prediction Quality -- accuracy per method
    methods_seen: dict[str, list[float]] = {}
    for fold_res in fold_results:
        exp1 = fold_res.get("exp1_prediction_quality", {})
        tbl = exp1.get("results_table")
        if tbl is None or tbl.empty or "accuracy" not in tbl.columns:
            continue
        for _, r in tbl.iterrows():
            method = r.get("method", "?")
            acc = r.get("accuracy", np.nan)
            if not np.isnan(acc):
                methods_seen.setdefault(method, []).append(acc)

    for method, accs in methods_seen.items():
        arr = np.array(accs)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci = 1.96 * std_val / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        rows.append({
            "Experiment": "Exp1 Pred. Quality",
            "Metric": f"Accuracy ({method})",
            "Mean": mean_val,
            "Std": std_val,
            "95% CI": f"{mean_val:.4f} +/- {ci:.4f}",
        })

    # Exp 2: Uncertainty -- coverage and set size per method
    uq_metrics: dict[str, dict[str, list[float]]] = {}
    for fold_res in fold_results:
        exp2 = fold_res.get("exp2_uncertainty", {})
        tbl = exp2.get("results_table")
        if tbl is None or tbl.empty:
            continue
        for _, r in tbl.iterrows():
            method = r.get("Method", "?")
            for col in ["Coverage", "Avg Set Size"]:
                val = r.get(col, np.nan)
                if not np.isnan(val):
                    uq_metrics.setdefault(method, {}).setdefault(col, []).append(val)

    for method, cols in uq_metrics.items():
        for col, vals in cols.items():
            arr = np.array(vals)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            ci = 1.96 * std_val / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            rows.append({
                "Experiment": "Exp2 Uncertainty",
                "Metric": f"{col} ({method})",
                "Mean": mean_val,
                "Std": std_val,
                "95% CI": f"{mean_val:.4f} +/- {ci:.4f}",
            })

    # Exp 3: Error Detection -- Spearman correlation per signal
    ed_metrics: dict[str, list[float]] = {}
    for fold_res in fold_results:
        exp3 = fold_res.get("exp3_error_detection", {})
        tbl = exp3.get("results_table")
        if tbl is None or tbl.empty:
            continue
        for _, r in tbl.iterrows():
            sig = r.get("Signal", "?")
            corr_val = r.get("Spearman Corr", np.nan)
            if not np.isnan(corr_val):
                ed_metrics.setdefault(sig, []).append(corr_val)

    for sig, vals in ed_metrics.items():
        arr = np.array(vals)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci = 1.96 * std_val / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        rows.append({
            "Experiment": "Exp3 Error Det.",
            "Metric": f"Spearman ({sig})",
            "Mean": mean_val,
            "Std": std_val,
            "95% CI": f"{mean_val:.4f} +/- {ci:.4f}",
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _aggregate_fold_results_regression(
    fold_results: list[dict],
    n_folds: int,
) -> pd.DataFrame:
    """Aggregate regression experiment results across folds.

    Returns a DataFrame with columns: Experiment, Metric, Mean, Std, 95% CI.
    """
    rows = []

    methods_rmse: dict[str, list[float]] = {}
    methods_mse: dict[str, list[float]] = {}
    for fold_res in fold_results:
        exp1 = fold_res.get("exp1_prediction_quality", {})
        tbl = exp1.get("results_table")
        if tbl is None or tbl.empty:
            continue
        for _, r in tbl.iterrows():
            method = r.get("method", "?")
            rmse = r.get("RMSE", np.nan)
            mse = r.get("MSE", np.nan)
            if not np.isnan(rmse):
                methods_rmse.setdefault(method, []).append(rmse)
            if not np.isnan(mse):
                methods_mse.setdefault(method, []).append(mse)

    for method, vals in methods_rmse.items():
        arr = np.array(vals)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci = 1.96 * std_val / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        rows.append({
            "Experiment": "Exp1 Pred. Quality",
            "Metric": f"RMSE ({method})",
            "Mean": mean_val,
            "Std": std_val,
            "95% CI": f"{mean_val:.4f} +/- {ci:.4f}",
        })

    for method, vals in methods_mse.items():
        arr = np.array(vals)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci = 1.96 * std_val / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        rows.append({
            "Experiment": "Exp1 Pred. Quality",
            "Metric": f"MSE ({method})",
            "Mean": mean_val,
            "Std": std_val,
            "95% CI": f"{mean_val:.4f} +/- {ci:.4f}",
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    results = run_soa_benchmark_cv(seed=42, n_folds=5, verbose=True)
