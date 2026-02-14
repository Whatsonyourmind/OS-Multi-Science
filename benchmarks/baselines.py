"""Standard baselines for comparison with the ICM framework.

Implements five baseline methods that serve as reference points for
evaluating the ICM multi-epistemic convergence approach:

1. EnsembleAverage      -- simple probability averaging
2. StackingBaseline     -- two-stage stacking with configurable meta-learner
3. BootstrapEnsemble    -- bootstrap-based disagreement metric
4. SplitConformal       -- split conformal prediction for uncertainty
5. DiversityMetrics     -- standard ensemble diversity measures

All baselines expect predictions in the same format used by
``compute_icm_from_predictions``: ``dict[str, np.ndarray]`` mapping
model names to probability arrays of shape ``(n_samples, n_classes)``
or ``(n_samples,)`` for binary/regression outputs.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy as _scipy_entropy
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


# ============================================================
# Helpers
# ============================================================

def _stack_predictions(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Stack model predictions into a 2-D matrix (n_samples, n_models * n_classes).

    For 1-D predictions (n_samples,), each model contributes one column.
    For 2-D predictions (n_samples, n_classes), columns are concatenated.
    """
    arrays = []
    for name in sorted(predictions.keys()):
        arr = np.asarray(predictions[name], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
    return np.hstack(arrays)


def _extract_class_predictions(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Return (n_samples, n_models) matrix of hard class predictions."""
    cols = []
    for name in sorted(predictions.keys()):
        arr = np.asarray(predictions[name], dtype=np.float64)
        if arr.ndim == 2:
            cols.append(np.argmax(arr, axis=1))
        else:
            cols.append((arr >= 0.5).astype(int))
    return np.column_stack(cols)


def _mean_proba(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Return averaged probability predictions across models.

    Returns shape (n_samples,) for binary, (n_samples, n_classes) for multi-class.
    """
    arrays = []
    for name in sorted(predictions.keys()):
        arrays.append(np.asarray(predictions[name], dtype=np.float64))
    stacked = np.stack(arrays, axis=0)  # (n_models, n_samples, ...) or (n_models, n_samples)
    return stacked.mean(axis=0)


def _compute_standard_scores(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy, log_loss, and brier_score for predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)

    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        # Clip probabilities for log_loss stability
        proba_clipped = np.clip(y_pred_proba, 1e-15, 1.0 - 1e-15)
        n_classes = y_pred_proba.shape[1]
        ll = log_loss(y_true, proba_clipped, labels=list(range(n_classes)))
        # Brier score: multi-class via one-hot
        one_hot = np.eye(n_classes)[y_true]
        bs = float(np.mean(np.sum((proba_clipped - one_hot) ** 2, axis=1)))
    else:
        # Binary case
        proba = y_pred_proba.ravel()
        proba_clipped = np.clip(proba, 1e-15, 1.0 - 1e-15)
        y_pred_class = (proba >= 0.5).astype(int)
        ll = log_loss(y_true, proba_clipped, labels=[0, 1])
        bs = brier_score_loss(y_true, proba_clipped)

    acc = accuracy_score(y_true, y_pred_class)
    return {"accuracy": float(acc), "log_loss": float(ll), "brier_score": float(bs)}


# ============================================================
# Baseline 1: Simple Ensemble Averaging
# ============================================================

class EnsembleAverage:
    """Simple average of all model predictions."""

    def aggregate(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Average probability predictions across models.

        Parameters
        ----------
        predictions : dict mapping model names to arrays of shape
            (n_samples,) or (n_samples, n_classes).

        Returns
        -------
        np.ndarray  Averaged predictions with the same shape as
            individual model outputs.
        """
        return _mean_proba(predictions)

    def score(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Return accuracy, log_loss, brier_score.

        Parameters
        ----------
        predictions : dict of model predictions.
        y_true : true labels of shape (n_samples,).

        Returns
        -------
        dict with keys 'accuracy', 'log_loss', 'brier_score'.
        """
        avg = self.aggregate(predictions)
        return _compute_standard_scores(avg, y_true)


# ============================================================
# Baseline 2: Sklearn Stacking
# ============================================================

class StackingBaseline:
    """Two-stage stacking: base models -> meta-learner.

    The meta-learner is trained on the concatenated probability
    outputs of the base models.

    Parameters
    ----------
    meta_learner : str
        One of 'logistic', 'ridge', 'random_forest'.
    """

    _META_MAP = {
        "logistic": lambda: LogisticRegression(
            max_iter=1000, solver="lbfgs"
        ),
        "ridge": lambda: RidgeClassifier(alpha=1.0),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
    }

    def __init__(self, meta_learner: str = "logistic") -> None:
        if meta_learner not in self._META_MAP:
            raise ValueError(
                f"Unknown meta_learner: {meta_learner!r}. "
                f"Choose from {list(self._META_MAP.keys())}"
            )
        self.meta_learner_name = meta_learner
        self._model = self._META_MAP[meta_learner]()
        self._fitted = False

    def fit(
        self,
        predictions_train: dict[str, np.ndarray],
        y_train: np.ndarray,
    ) -> None:
        """Train meta-learner on base predictions.

        Parameters
        ----------
        predictions_train : dict of base model training predictions.
        y_train : true labels for training set.
        """
        X = _stack_predictions(predictions_train)
        y_train = np.asarray(y_train, dtype=int)
        self._model.fit(X, y_train)
        self._fitted = True

    def predict(self, predictions_test: dict[str, np.ndarray]) -> np.ndarray:
        """Produce stacked prediction probabilities.

        Parameters
        ----------
        predictions_test : dict of base model test predictions.

        Returns
        -------
        np.ndarray  Predicted probabilities from the meta-learner.
            Shape (n_samples, n_classes) or (n_samples,) for binary.
        """
        if not self._fitted:
            raise RuntimeError("StackingBaseline must be fit() before predict().")
        X = _stack_predictions(predictions_test)
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]  # binary: return P(class=1)
            return proba
        else:
            # RidgeClassifier: use decision_function -> sigmoid
            from scipy.special import expit
            decision = self._model.decision_function(X)
            if decision.ndim == 1:
                return expit(decision)
            # Multi-class: softmax
            exp_d = np.exp(decision - decision.max(axis=1, keepdims=True))
            return exp_d / exp_d.sum(axis=1, keepdims=True)

    def score(
        self,
        predictions_test: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Return accuracy, log_loss, brier_score.

        Parameters
        ----------
        predictions_test : dict of base model test predictions.
        y_true : true labels for test set.

        Returns
        -------
        dict with keys 'accuracy', 'log_loss', 'brier_score'.
        """
        proba = self.predict(predictions_test)
        return _compute_standard_scores(proba, y_true)


# ============================================================
# Baseline 3: Bootstrap Ensemble with Disagreement
# ============================================================

class BootstrapEnsemble:
    """Bootstrap-based disagreement metric.

    Measures inter-model disagreement by repeatedly bootstrap-sampling
    subsets of models and computing the entropy of the resulting
    prediction distribution.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    """

    def __init__(self, n_bootstrap: int = 100) -> None:
        self.n_bootstrap = n_bootstrap

    def compute_disagreement(
        self,
        predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Per-sample disagreement score (entropy of bootstrap distribution).

        For each bootstrap sample, a random subset of models is drawn
        (with replacement), their predictions averaged, and the argmax
        class recorded.  The entropy of the bootstrap class distribution
        is the disagreement score per sample.

        Parameters
        ----------
        predictions : dict of model predictions.

        Returns
        -------
        np.ndarray  Per-sample disagreement scores, shape (n_samples,).
            Values in [0, log(n_classes)] for multi-class, [0, log(2)]
            for binary.
        """
        model_names = sorted(predictions.keys())
        n_models = len(model_names)
        sample_arr = np.asarray(predictions[model_names[0]], dtype=np.float64)
        if sample_arr.ndim == 1:
            n_samples = sample_arr.shape[0]
            n_classes = 2
        else:
            n_samples, n_classes = sample_arr.shape

        rng = np.random.default_rng(42)
        # Count bootstrap class votes per sample
        vote_counts = np.zeros((n_samples, n_classes), dtype=np.float64)

        for _ in range(self.n_bootstrap):
            # Bootstrap-sample model indices
            idx = rng.integers(0, n_models, size=n_models)
            # Average predictions of sampled models
            sampled = []
            for i in idx:
                arr = np.asarray(predictions[model_names[i]], dtype=np.float64)
                sampled.append(arr)
            stacked = np.stack(sampled, axis=0)
            avg = stacked.mean(axis=0)

            if avg.ndim == 1:
                classes = (avg >= 0.5).astype(int)
            else:
                classes = np.argmax(avg, axis=1)

            for s in range(n_samples):
                vote_counts[s, classes[s]] += 1

        # Normalize to probabilities and compute entropy
        probs = vote_counts / self.n_bootstrap
        # Entropy per sample
        disagreement = np.array([
            _scipy_entropy(probs[s] + 1e-15) for s in range(n_samples)
        ])
        return disagreement

    def uncertainty_vs_error(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Return correlation between disagreement and actual error.

        Parameters
        ----------
        predictions : dict of model predictions.
        y_true : true labels, shape (n_samples,).

        Returns
        -------
        dict with keys:
            'pearson_correlation' : Pearson correlation between
                disagreement and per-sample error.
            'mean_disagreement' : average disagreement across samples.
        """
        y_true = np.asarray(y_true, dtype=int)
        disagreement = self.compute_disagreement(predictions)
        avg = _mean_proba(predictions)

        # Per-sample error: 0 if correct, 1 if incorrect
        if avg.ndim == 2:
            pred_class = np.argmax(avg, axis=1)
        else:
            pred_class = (avg >= 0.5).astype(int)

        errors = (pred_class != y_true).astype(float)

        # Pearson correlation
        if np.std(disagreement) < 1e-12 or np.std(errors) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(disagreement, errors)[0, 1])

        return {
            "pearson_correlation": corr,
            "mean_disagreement": float(np.mean(disagreement)),
        }


# ============================================================
# Baseline 4: Split Conformal Prediction
# ============================================================

class SplitConformal:
    """Standard split conformal prediction for uncertainty quantification.

    Uses the non-conformity score  1 - p(y_true)  to build prediction
    sets with guaranteed marginal coverage.

    Parameters
    ----------
    alpha : float
        Desired miscoverage rate.  Prediction sets will cover the true
        class with probability >= 1 - alpha.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self._quantile: float | None = None

    def calibrate(
        self,
        predictions_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Compute conformal quantile from calibration set.

        Parameters
        ----------
        predictions_cal : array of shape (n_cal, n_classes) or (n_cal,).
            Model probability predictions on calibration data.
        y_cal : array of shape (n_cal,).
            True labels for calibration data.
        """
        predictions_cal = np.asarray(predictions_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=int)
        n = len(y_cal)

        # Non-conformity scores: 1 - softmax(true class)
        if predictions_cal.ndim == 2:
            scores = 1.0 - predictions_cal[np.arange(n), y_cal]
        else:
            # Binary: P(class=1) for class 1, 1-P for class 0
            p1 = predictions_cal.ravel()
            scores = np.where(y_cal == 1, 1.0 - p1, p1)

        # Conformal quantile: ceil((1 - alpha) * (n + 1)) / n
        quantile_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        quantile_level = min(quantile_level, 1.0)
        self._quantile = float(np.quantile(scores, quantile_level))

    def predict_sets(
        self,
        predictions_test: np.ndarray,
    ) -> list[set]:
        """Return prediction sets for each test point.

        A class c is included in the prediction set if its
        non-conformity score 1 - p(c) <= quantile.

        Parameters
        ----------
        predictions_test : array of shape (n_test, n_classes) or (n_test,).

        Returns
        -------
        list of sets, one per test sample. Each set contains the
        class indices in the prediction set.
        """
        if self._quantile is None:
            raise RuntimeError("SplitConformal must be calibrate()d first.")

        predictions_test = np.asarray(predictions_test, dtype=np.float64)

        if predictions_test.ndim == 2:
            n_test, n_classes = predictions_test.shape
            result = []
            for i in range(n_test):
                pred_set = set()
                for c in range(n_classes):
                    score = 1.0 - predictions_test[i, c]
                    if score <= self._quantile:
                        pred_set.add(c)
                # Guarantee at least one class in the set
                if not pred_set:
                    pred_set.add(int(np.argmax(predictions_test[i])))
                result.append(pred_set)
            return result
        else:
            # Binary
            p1 = predictions_test.ravel()
            result = []
            for i in range(len(p1)):
                pred_set = set()
                if p1[i] <= self._quantile:          # score for class 0 = p1
                    pred_set.add(0)
                if (1.0 - p1[i]) <= self._quantile:  # score for class 1 = 1-p1
                    pred_set.add(1)
                if not pred_set:
                    pred_set.add(int(p1[i] >= 0.5))
                result.append(pred_set)
            return result

    def coverage(
        self,
        predictions_test: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Empirical coverage on test set.

        Parameters
        ----------
        predictions_test : model predictions on test set.
        y_true : true labels, shape (n_test,).

        Returns
        -------
        float  Fraction of test points where y_true is in the
            prediction set.
        """
        y_true = np.asarray(y_true, dtype=int)
        pred_sets = self.predict_sets(predictions_test)
        covered = sum(1 for i, ps in enumerate(pred_sets) if y_true[i] in ps)
        return float(covered / len(y_true))

    def avg_set_size(
        self,
        predictions_test: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Average prediction set size (efficiency).

        Smaller sets with valid coverage indicate better efficiency.

        Parameters
        ----------
        predictions_test : model predictions on test set.
        y_true : true labels (used only for interface consistency).

        Returns
        -------
        float  Average number of classes per prediction set.
        """
        pred_sets = self.predict_sets(predictions_test)
        return float(np.mean([len(ps) for ps in pred_sets]))


# ============================================================
# Baseline 5: Prediction Diversity Metrics
# ============================================================

class DiversityMetrics:
    """Standard ensemble diversity metrics for comparison with ICM.

    All pairwise metrics are averaged over all unique pairs of models.
    """

    def q_statistic(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> float:
        """Yule's Q-statistic measuring pairwise diversity.

        For each pair of models, construct the 2x2 contingency table of
        correct/incorrect classifications, then compute Q = (ad - bc) / (ad + bc).
        Average over all pairs.

        Values near 0 indicate independent models; 1 indicates identical
        models; -1 indicates maximally diverse models.

        Parameters
        ----------
        predictions : dict of model predictions.
        y_true : true labels.

        Returns
        -------
        float  Mean Q-statistic, in [-1, 1].
        """
        y_true = np.asarray(y_true, dtype=int)
        hard = _extract_class_predictions(predictions)
        n_models = hard.shape[1]

        correct = (hard == y_true[:, np.newaxis]).astype(int)

        q_values = []
        for i, j in combinations(range(n_models), 2):
            a = np.sum((correct[:, i] == 1) & (correct[:, j] == 1))
            b = np.sum((correct[:, i] == 1) & (correct[:, j] == 0))
            c = np.sum((correct[:, i] == 0) & (correct[:, j] == 1))
            d = np.sum((correct[:, i] == 0) & (correct[:, j] == 0))
            ad = a * d
            bc = b * c
            denom = ad + bc
            if denom == 0:
                q_values.append(0.0)
            else:
                q_values.append((ad - bc) / denom)

        if not q_values:
            return 0.0
        return float(np.mean(q_values))

    def disagreement_measure(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> float:
        """Proportion of samples where models disagree.

        For each pair of models, compute the fraction of samples where
        one is correct and the other is incorrect.

        Parameters
        ----------
        predictions : dict of model predictions.
        y_true : true labels.

        Returns
        -------
        float  Mean disagreement proportion, in [0, 1].
        """
        y_true = np.asarray(y_true, dtype=int)
        hard = _extract_class_predictions(predictions)
        n_models = hard.shape[1]
        n_samples = hard.shape[0]

        correct = (hard == y_true[:, np.newaxis]).astype(int)

        dis_values = []
        for i, j in combinations(range(n_models), 2):
            b = np.sum((correct[:, i] == 1) & (correct[:, j] == 0))
            c = np.sum((correct[:, i] == 0) & (correct[:, j] == 1))
            dis_values.append((b + c) / n_samples)

        if not dis_values:
            return 0.0
        return float(np.mean(dis_values))

    def correlation_coefficient(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> float:
        """Mean pairwise correlation of correct/incorrect predictions.

        Uses the Pearson correlation of binary correct/incorrect vectors
        across model pairs.

        Parameters
        ----------
        predictions : dict of model predictions.
        y_true : true labels.

        Returns
        -------
        float  Mean correlation coefficient, in [-1, 1].
        """
        y_true = np.asarray(y_true, dtype=int)
        hard = _extract_class_predictions(predictions)
        n_models = hard.shape[1]

        correct = (hard == y_true[:, np.newaxis]).astype(float)

        corr_values = []
        for i, j in combinations(range(n_models), 2):
            std_i = np.std(correct[:, i])
            std_j = np.std(correct[:, j])
            if std_i < 1e-12 or std_j < 1e-12:
                corr_values.append(0.0)
            else:
                corr_values.append(float(np.corrcoef(correct[:, i], correct[:, j])[0, 1]))

        if not corr_values:
            return 0.0
        return float(np.mean(corr_values))

    def entropy_measure(
        self,
        predictions: dict[str, np.ndarray],
    ) -> float:
        """Mean entropy of prediction distribution per sample.

        For each sample, compute the distribution of class votes across
        models, then take the Shannon entropy.  Averaged over all samples.

        Parameters
        ----------
        predictions : dict of model predictions.

        Returns
        -------
        float  Mean entropy, >= 0.
        """
        hard = _extract_class_predictions(predictions)
        n_samples, n_models = hard.shape
        n_classes = int(hard.max()) + 1

        entropies = []
        for s in range(n_samples):
            counts = np.bincount(hard[s].astype(int), minlength=n_classes).astype(float)
            probs = counts / n_models
            entropies.append(_scipy_entropy(probs + 1e-15))

        return float(np.mean(entropies))

    def kl_diversity(
        self,
        predictions: dict[str, np.ndarray],
    ) -> float:
        """Mean pairwise KL divergence between models.

        For multi-class predictions, computes KL(p_i || p_j) for each
        pair (i, j) and averages.  For 1-D predictions, converts to
        [1-p, p] distributions.

        Parameters
        ----------
        predictions : dict of model predictions.

        Returns
        -------
        float  Mean KL divergence, >= 0.
        """
        model_names = sorted(predictions.keys())
        arrays = []
        for name in model_names:
            arr = np.asarray(predictions[name], dtype=np.float64)
            if arr.ndim == 1:
                arr = np.column_stack([1.0 - arr, arr])
            arrays.append(arr)

        n_models = len(arrays)
        n_samples = arrays[0].shape[0]

        kl_values = []
        for i, j in combinations(range(n_models), 2):
            # Per-sample KL, then average
            kl_per_sample = []
            for s in range(n_samples):
                p = np.clip(arrays[i][s], 1e-15, 1.0)
                q = np.clip(arrays[j][s], 1e-15, 1.0)
                p = p / p.sum()
                q = q / q.sum()
                kl = float(_scipy_entropy(p, q))
                kl_per_sample.append(kl)
            kl_values.append(np.mean(kl_per_sample))

        if not kl_values:
            return 0.0
        return float(np.mean(kl_values))


# ============================================================
# Baseline 6: Deep Ensemble (MLP-based)
# ============================================================

class DeepEnsemble:
    """Simulated deep ensemble using sklearn MLPClassifier.

    Trains 5 MLP models with different architectures (varying hidden
    layer sizes) and different random seeds to simulate the diversity
    found in deep ensembles.  Predictions are averaged across all
    ensemble members.

    This mirrors the deep ensemble approach from Lakshminarayanan et al.
    (2017) without requiring PyTorch or TensorFlow.

    Parameters
    ----------
    n_members : int
        Number of ensemble members (default 5).
    max_iter : int
        Maximum training iterations per member.
    seed : int
        Base random seed (each member uses seed + i).
    """

    # 5 architecturally diverse MLP configurations
    _ARCHITECTURES = [
        (32,),              # Shallow, narrow
        (64, 32),           # Medium depth
        (128, 64, 32),      # Deep
        (64, 64),           # Uniform width
        (128, 32),          # Wide-then-narrow
    ]

    def __init__(
        self,
        n_members: int = 5,
        max_iter: int = 300,
        seed: int = 42,
    ) -> None:
        self.n_members = min(n_members, len(self._ARCHITECTURES))
        self.max_iter = max_iter
        self.seed = seed
        self._models: list[MLPClassifier] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train all ensemble members on the same data.

        Each member uses a different architecture and random seed,
        creating genuine diversity in the ensemble.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._models = []

        for i in range(self.n_members):
            mlp = MLPClassifier(
                hidden_layer_sizes=self._ARCHITECTURES[i],
                max_iter=self.max_iter,
                random_state=self.seed + i,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
            )
            mlp.fit(X, y)
            self._models.append(mlp)

        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged probability predictions from all members.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("DeepEnsemble must be fit() before predict_proba().")

        X = np.asarray(X, dtype=np.float64)
        probas = [m.predict_proba(X) for m in self._models]
        return np.mean(probas, axis=0)

    def predict_proba_per_member(self, X: np.ndarray) -> list[np.ndarray]:
        """Return individual member probability predictions.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        list of np.ndarray, each of shape (n_samples, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("DeepEnsemble must be fit() before predict_proba_per_member().")

        X = np.asarray(X, dtype=np.float64)
        return [m.predict_proba(X) for m in self._models]

    def score(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Return accuracy, log_loss, brier_score for ensemble predictions.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y_true : true labels of shape (n_samples,)

        Returns
        -------
        dict with keys 'accuracy', 'log_loss', 'brier_score'.
        """
        proba = self.predict_proba(X)
        return _compute_standard_scores(proba, y_true)

    def disagreement(self, X: np.ndarray) -> np.ndarray:
        """Per-sample disagreement among ensemble members.

        Measured as the mean pairwise Jensen-Shannon divergence
        across member predictions.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with disagreement scores >= 0.
        """
        probas = self.predict_proba_per_member(X)
        n_members = len(probas)
        n_samples = probas[0].shape[0]
        disagreement = np.zeros(n_samples)

        count = 0
        for i in range(n_members):
            for j in range(i + 1, n_members):
                p = np.clip(probas[i], 1e-15, 1.0)
                q = np.clip(probas[j], 1e-15, 1.0)
                m = 0.5 * (p + q)
                jsd = 0.5 * np.sum(p * np.log(p / m), axis=1) + \
                      0.5 * np.sum(q * np.log(q / m), axis=1)
                disagreement += jsd
                count += 1

        if count > 0:
            disagreement /= count

        return disagreement


# ============================================================
# Comparison Function
# ============================================================

def run_baseline_comparison(
    predictions_train: dict[str, np.ndarray],
    predictions_test: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    icm_scores: dict[str, float] | None = None,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Run all baselines and produce a comparison table.

    Parameters
    ----------
    predictions_train : dict of base model training predictions.
    predictions_test : dict of base model test predictions.
    y_train : true labels for training set.
    y_test : true labels for test set.
    icm_scores : optional dict with pre-computed ICM framework scores.
        Expected keys: 'accuracy', 'log_loss', 'brier_score'.
        Additional keys 'coverage' and 'set_size' are used if present.
    alpha : conformal miscoverage level.

    Returns
    -------
    pd.DataFrame  Comparison table with columns:
        Method, Accuracy, Log Loss, Brier, Coverage, Set Size, Diversity Corr
    """
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    rows = []

    # --- ICM Framework (user-supplied or placeholder) ---
    if icm_scores is not None:
        rows.append({
            "Method": "ICM Framework",
            "Accuracy": icm_scores.get("accuracy", np.nan),
            "Log Loss": icm_scores.get("log_loss", np.nan),
            "Brier": icm_scores.get("brier_score", np.nan),
            "Coverage": icm_scores.get("coverage", np.nan),
            "Set Size": icm_scores.get("set_size", np.nan),
            "Diversity Corr": icm_scores.get("diversity_corr", np.nan),
        })

    # --- Ensemble Average ---
    ea = EnsembleAverage()
    ea_scores = ea.score(predictions_test, y_test)
    rows.append({
        "Method": "Ensemble Avg",
        "Accuracy": ea_scores["accuracy"],
        "Log Loss": ea_scores["log_loss"],
        "Brier": ea_scores["brier_score"],
        "Coverage": np.nan,
        "Set Size": np.nan,
        "Diversity Corr": np.nan,
    })

    # --- Stacking (Logistic) ---
    try:
        sl = StackingBaseline(meta_learner="logistic")
        sl.fit(predictions_train, y_train)
        sl_scores = sl.score(predictions_test, y_test)
        rows.append({
            "Method": "Stacking (LR)",
            "Accuracy": sl_scores["accuracy"],
            "Log Loss": sl_scores["log_loss"],
            "Brier": sl_scores["brier_score"],
            "Coverage": np.nan,
            "Set Size": np.nan,
            "Diversity Corr": np.nan,
        })
    except Exception:
        rows.append({
            "Method": "Stacking (LR)",
            "Accuracy": np.nan, "Log Loss": np.nan, "Brier": np.nan,
            "Coverage": np.nan, "Set Size": np.nan, "Diversity Corr": np.nan,
        })

    # --- Stacking (Random Forest) ---
    try:
        sr = StackingBaseline(meta_learner="random_forest")
        sr.fit(predictions_train, y_train)
        sr_scores = sr.score(predictions_test, y_test)
        rows.append({
            "Method": "Stacking (RF)",
            "Accuracy": sr_scores["accuracy"],
            "Log Loss": sr_scores["log_loss"],
            "Brier": sr_scores["brier_score"],
            "Coverage": np.nan,
            "Set Size": np.nan,
            "Diversity Corr": np.nan,
        })
    except Exception:
        rows.append({
            "Method": "Stacking (RF)",
            "Accuracy": np.nan, "Log Loss": np.nan, "Brier": np.nan,
            "Coverage": np.nan, "Set Size": np.nan, "Diversity Corr": np.nan,
        })

    # --- Bootstrap Ensemble ---
    be = BootstrapEnsemble(n_bootstrap=100)
    be_result = be.uncertainty_vs_error(predictions_test, y_test)
    rows.append({
        "Method": "Bootstrap",
        "Accuracy": np.nan,
        "Log Loss": np.nan,
        "Brier": np.nan,
        "Coverage": np.nan,
        "Set Size": np.nan,
        "Diversity Corr": be_result["pearson_correlation"],
    })

    # --- Split Conformal ---
    avg_pred = _mean_proba(predictions_test)
    # For conformal, we need a calibration split from test
    n_test = len(y_test)
    n_cal = n_test // 2
    if n_cal >= 2:
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_test)
        cal_idx = indices[:n_cal]
        test_idx = indices[n_cal:]

        if avg_pred.ndim == 2:
            cal_pred = avg_pred[cal_idx]
            test_pred = avg_pred[test_idx]
        else:
            cal_pred = avg_pred[cal_idx]
            test_pred = avg_pred[test_idx]

        sc = SplitConformal(alpha=alpha)
        sc.calibrate(cal_pred, y_test[cal_idx])
        cov = sc.coverage(test_pred, y_test[test_idx])
        avg_ss = sc.avg_set_size(test_pred, y_test[test_idx])
    else:
        cov = np.nan
        avg_ss = np.nan

    rows.append({
        "Method": "Split Conformal",
        "Accuracy": np.nan,
        "Log Loss": np.nan,
        "Brier": np.nan,
        "Coverage": cov,
        "Set Size": avg_ss,
        "Diversity Corr": np.nan,
    })

    df = pd.DataFrame(rows)
    df = df[["Method", "Accuracy", "Log Loss", "Brier", "Coverage",
             "Set Size", "Diversity Corr"]]
    return df
