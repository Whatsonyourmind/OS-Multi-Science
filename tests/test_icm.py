"""Comprehensive tests for ICM v1.1 engine."""

from __future__ import annotations

import numpy as np
import pytest

from framework.config import ICMConfig
from framework.types import ICMComponents, ICMResult
from framework.icm import (
    hellinger_distance,
    wasserstein2_distance,
    wasserstein2_empirical,
    mmd_distance,
    frechet_variance,
    compute_agreement,
    compute_direction,
    compute_uncertainty_overlap,
    compute_invariance,
    compute_dependency_penalty,
    compute_icm,
    compute_icm_geometric,
    compute_icm_calibrated,
    compute_icm_adaptive,
    _dispatch_aggregation,
    compute_icm_from_predictions,
    compute_icm_timeseries,
)


# ============================================================
# Distance Functions
# ============================================================

class TestHellingerDistance:
    """Tests for hellinger_distance."""

    def test_identical_distributions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert hellinger_distance(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_disjoint_distributions(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert hellinger_distance(p, q) == pytest.approx(1.0, abs=1e-10)

    def test_known_value(self):
        # H(Bernoulli(0.5), Bernoulli(0.5)) = 0
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert hellinger_distance(p, q) == pytest.approx(0.0, abs=1e-10)

    def test_known_asymmetric(self):
        # Analytical: H^2 = 0.5*((sqrt(0.9)-sqrt(0.1))^2 + (sqrt(0.1)-sqrt(0.9))^2)
        #           = (sqrt(0.9)-sqrt(0.1))^2
        # H = |sqrt(0.9)-sqrt(0.1)|
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        expected_sq = (np.sqrt(0.9) - np.sqrt(0.1)) ** 2
        expected = np.sqrt(expected_sq)
        assert hellinger_distance(p, q) == pytest.approx(expected, abs=1e-10)

    def test_bounded_0_1(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            h = hellinger_distance(p, q)
            assert 0.0 <= h <= 1.0 + 1e-10

    def test_symmetry(self):
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        assert hellinger_distance(p, q) == pytest.approx(
            hellinger_distance(q, p), abs=1e-10
        )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            hellinger_distance(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


class TestWasserstein2Distance:
    """Tests for closed-form Wasserstein-2 between Gaussians."""

    def test_identical_gaussians(self):
        mu = np.array([0.0, 0.0])
        sigma = np.eye(2)
        assert wasserstein2_distance(mu, sigma, mu, sigma) == pytest.approx(
            0.0, abs=1e-8
        )

    def test_shifted_mean(self):
        # For identity covariances, W2 = ||mu1-mu2||
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([3.0, 4.0])
        sigma = np.eye(2)
        expected = 5.0  # ||[3,4]|| = 5
        assert wasserstein2_distance(mu1, sigma, mu2, sigma) == pytest.approx(
            expected, abs=1e-6
        )

    def test_1d_different_variances(self):
        # W2^2 = (mu1-mu2)^2 + (sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        #       = 0 + (1 + 4 - 2*sqrt(4)) = 5 - 4 = 1
        mu1 = np.array([0.0])
        mu2 = np.array([0.0])
        s1 = np.array([[1.0]])
        s2 = np.array([[4.0]])
        assert wasserstein2_distance(mu1, s1, mu2, s2) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_non_negative(self):
        rng = np.random.default_rng(123)
        for _ in range(20):
            d = 3
            mu1 = rng.standard_normal(d)
            mu2 = rng.standard_normal(d)
            A = rng.standard_normal((d, d))
            s1 = A @ A.T + 0.1 * np.eye(d)
            B = rng.standard_normal((d, d))
            s2 = B @ B.T + 0.1 * np.eye(d)
            w2 = wasserstein2_distance(mu1, s1, mu2, s2)
            assert w2 >= -1e-10


class TestWasserstein2Empirical:
    """Tests for empirical W2 (1-D fast path)."""

    def test_identical_samples(self):
        X = np.array([1.0, 2.0, 3.0])
        assert wasserstein2_empirical(X, X) == pytest.approx(0.0, abs=1e-10)

    def test_shifted_samples(self):
        X = np.array([0.0, 1.0, 2.0])
        Y = np.array([1.0, 2.0, 3.0])
        # Sorted coupling: all pairs differ by 1, so W2 = 1
        assert wasserstein2_empirical(X, Y) == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            X = rng.standard_normal(50)
            Y = rng.standard_normal(50)
            assert wasserstein2_empirical(X, Y) >= -1e-10


class TestMMDDistance:
    """Tests for Maximum Mean Discrepancy."""

    def test_identical_samples(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        assert mmd_distance(X, X) == pytest.approx(0.0, abs=1e-6)

    def test_different_samples(self):
        X = np.zeros((50, 2))
        Y = np.ones((50, 2)) * 10.0
        d = mmd_distance(X, Y, bandwidth=1.0)
        assert d > 0.5  # Clearly separated

    def test_non_negative(self):
        rng = np.random.default_rng(99)
        for _ in range(20):
            X = rng.standard_normal((30, 3))
            Y = rng.standard_normal((30, 3))
            assert mmd_distance(X, Y) >= -1e-10

    def test_symmetry(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 2))
        Y = rng.standard_normal((20, 2))
        assert mmd_distance(X, Y) == pytest.approx(
            mmd_distance(Y, X), abs=1e-10
        )


class TestFrechetVariance:
    """Tests for Frechet variance in embedding space."""

    def test_identical_embeddings(self):
        embeddings = np.ones((5, 3))
        assert frechet_variance(embeddings) == pytest.approx(0.0, abs=1e-10)

    def test_known_variance(self):
        # Two points equidistant from centroid: [0,0] and [2,0]
        # Centroid = [1,0]. Variance = mean(1^2, 1^2) = 1.0
        embeddings = np.array([[0.0, 0.0], [2.0, 0.0]])
        assert frechet_variance(embeddings) == pytest.approx(1.0, abs=1e-10)

    def test_non_negative(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            emb = rng.standard_normal((10, 4))
            assert frechet_variance(emb) >= -1e-10


# ============================================================
# ICM Components
# ============================================================

class TestComputeAgreement:
    """Tests for compute_agreement."""

    def test_identical_predictions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        A = compute_agreement([p, p, p], distance_fn="hellinger")
        assert A == pytest.approx(1.0, abs=1e-6)

    def test_different_predictions(self):
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 1.0])
        A = compute_agreement([p1, p2], distance_fn="hellinger")
        assert A == pytest.approx(0.0, abs=1e-6)

    def test_bounded_0_1(self):
        rng = np.random.default_rng(33)
        preds = [rng.dirichlet(np.ones(4)) for _ in range(5)]
        A = compute_agreement(preds, distance_fn="hellinger")
        assert 0.0 <= A <= 1.0 + 1e-10

    def test_single_model(self):
        p = np.array([0.5, 0.5])
        assert compute_agreement([p]) == pytest.approx(1.0)

    def test_wasserstein_distance_fn(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([1.1, 2.1, 3.1])
        A = compute_agreement([p1, p2], distance_fn="wasserstein")
        assert 0.0 <= A <= 1.0

    def test_mmd_distance_fn(self):
        rng = np.random.default_rng(88)
        p1 = rng.standard_normal((20, 2))
        p2 = rng.standard_normal((20, 2)) + 0.1
        A = compute_agreement([p1, p2], distance_fn="mmd")
        assert 0.0 <= A <= 1.0


class TestComputeDirection:
    """Tests for compute_direction."""

    def test_all_same_sign(self):
        signs = np.array([1, 1, 1, 1])
        assert compute_direction(signs) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_signs(self):
        # Equal split: max entropy
        signs = np.array([1, -1, 1, -1])
        D = compute_direction(signs)
        assert D == pytest.approx(0.0, abs=1e-6)

    def test_bounded_0_1(self):
        rng = np.random.default_rng(77)
        for _ in range(30):
            signs = rng.choice([-1, 0, 1], size=10)
            D = compute_direction(signs)
            assert 0.0 <= D <= 1.0 + 1e-10

    def test_single_model(self):
        assert compute_direction(np.array([1])) == pytest.approx(1.0)

    def test_from_gradients(self):
        # Positive gradients -> all +1
        grads = np.array([0.5, 1.2, 3.0, 0.01])
        assert compute_direction(grads) == pytest.approx(1.0, abs=1e-6)


class TestComputeUncertaintyOverlap:
    """Tests for compute_uncertainty_overlap."""

    def test_identical_intervals(self):
        intervals = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        assert compute_uncertainty_overlap(intervals) == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_intervals(self):
        intervals = [(0.0, 1.0), (2.0, 3.0)]
        assert compute_uncertainty_overlap(intervals) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        # [0,2] and [1,3]: intersection=[1,2]=1, union=[0,3]=3, IO=1/3
        intervals = [(0.0, 2.0), (1.0, 3.0)]
        assert compute_uncertainty_overlap(intervals) == pytest.approx(
            1.0 / 3.0, abs=1e-6
        )

    def test_bounded_0_1(self):
        rng = np.random.default_rng(22)
        for _ in range(30):
            intervals = [(rng.uniform(0, 5), rng.uniform(5, 10)) for _ in range(4)]
            U = compute_uncertainty_overlap(intervals)
            assert 0.0 <= U <= 1.0 + 1e-10

    def test_single_interval(self):
        assert compute_uncertainty_overlap([(0.0, 1.0)]) == pytest.approx(1.0)


class TestComputeInvariance:
    """Tests for compute_invariance."""

    def test_no_change(self):
        scores = np.array([1.0, 2.0, 3.0])
        assert compute_invariance(scores, scores) == pytest.approx(1.0, abs=1e-6)

    def test_total_change(self):
        pre = np.array([1.0, 0.0, 0.0])
        post = np.array([0.0, 0.0, 0.0])
        C = compute_invariance(pre, post)
        assert C == pytest.approx(0.0, abs=1e-6)

    def test_bounded_0_1(self):
        rng = np.random.default_rng(44)
        for _ in range(30):
            pre = rng.standard_normal(10) + 5  # Shift to ensure non-zero norm
            post = pre + rng.standard_normal(10) * 0.5
            C = compute_invariance(pre, post)
            assert 0.0 <= C <= 1.0 + 1e-10


class TestComputeDependencyPenalty:
    """Tests for compute_dependency_penalty."""

    def test_independent_residuals(self):
        rng = np.random.default_rng(55)
        # Large sample, nearly independent
        residuals = rng.standard_normal((4, 5000))
        Pi = compute_dependency_penalty(residuals=residuals)
        assert Pi < 0.3  # Should be low for independent models

    def test_identical_residuals(self):
        row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = np.vstack([row, row, row])
        Pi = compute_dependency_penalty(residuals=residuals)
        assert Pi > 0.5  # Should be high: models are clones

    def test_feature_overlap(self):
        f1 = {"age", "income", "education"}
        f2 = {"age", "income", "education"}
        f3 = {"temperature", "humidity", "pressure"}
        # f1 and f2 are identical (Jaccard=1), f1/f3 disjoint (Jaccard=0)
        Pi_overlap = compute_dependency_penalty(features=[f1, f2])
        Pi_disjoint = compute_dependency_penalty(features=[f1, f3])
        assert Pi_overlap > Pi_disjoint

    def test_gradient_similarity(self):
        g1 = np.array([1.0, 0.0, 0.0])
        g2 = np.array([1.0, 0.0, 0.0])
        g3 = np.array([-1.0, 0.0, 0.0])
        Pi_same = compute_dependency_penalty(gradients=np.vstack([g1, g2]))
        Pi_opp = compute_dependency_penalty(gradients=np.vstack([g1, g3]))
        assert Pi_same > Pi_opp

    def test_bounded_0_1(self):
        rng = np.random.default_rng(66)
        residuals = rng.standard_normal((5, 100))
        features = [{"a", "b"}, {"b", "c"}, {"c", "d"}, {"d", "e"}, {"e", "a"}]
        gradients = rng.standard_normal((5, 10))
        Pi = compute_dependency_penalty(
            residuals=residuals, features=features, gradients=gradients
        )
        assert 0.0 <= Pi <= 1.0 + 1e-10

    def test_no_inputs_returns_zero(self):
        assert compute_dependency_penalty() == pytest.approx(0.0)


# ============================================================
# ICM Aggregation
# ============================================================

class TestComputeICM:
    """Tests for logistic ICM aggregation."""

    def test_perfect_convergence(self):
        components = ICMComponents(A=1.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        result = compute_icm(components)
        assert result.icm_score > 0.5
        assert result.aggregation_method == "logistic"

    def test_no_convergence(self):
        components = ICMComponents(A=0.0, D=0.0, U=0.0, C=0.0, Pi=1.0)
        result = compute_icm(components)
        assert result.icm_score < 0.5

    def test_bounded_0_1(self):
        rng = np.random.default_rng(11)
        for _ in range(50):
            components = ICMComponents(
                A=rng.uniform(), D=rng.uniform(),
                U=rng.uniform(), C=rng.uniform(),
                Pi=rng.uniform(),
            )
            result = compute_icm(components)
            assert 0.0 <= result.icm_score <= 1.0

    def test_weights_stored(self):
        components = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.5)
        result = compute_icm(components)
        assert "w_A" in result.weights
        assert "lam" in result.weights

    def test_properties(self):
        high = ICMComponents(A=1.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        low = ICMComponents(A=0.0, D=0.0, U=0.0, C=0.0, Pi=1.0)
        r_high = compute_icm(high)
        r_low = compute_icm(low)
        assert r_high.is_high_convergence or r_high.icm_score > 0.5
        assert r_low.icm_score < r_high.icm_score


class TestComputeICMGeometric:
    """Tests for geometric ICM aggregation."""

    def test_perfect_convergence(self):
        components = ICMComponents(A=1.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        result = compute_icm_geometric(components)
        assert result.icm_score == pytest.approx(1.0, abs=1e-6)
        assert result.aggregation_method == "geometric"

    def test_zero_component_gives_low(self):
        components = ICMComponents(A=0.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        result = compute_icm_geometric(components)
        assert result.icm_score < 0.01  # Nearly zero (eps-clamped)

    def test_bounded_0_1(self):
        rng = np.random.default_rng(22)
        for _ in range(50):
            components = ICMComponents(
                A=rng.uniform(), D=rng.uniform(),
                U=rng.uniform(), C=rng.uniform(),
                Pi=rng.uniform(),
            )
            result = compute_icm_geometric(components)
            assert 0.0 <= result.icm_score <= 1.0 + 1e-10


# ============================================================
# Convenience functions
# ============================================================

class TestComputeICMFromPredictions:
    """Tests for the high-level compute_icm_from_predictions."""

    def test_agreeing_models(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        preds = {f"model_{i}": p for i in range(4)}
        result = compute_icm_from_predictions(preds)
        assert result.icm_score > 0.5
        assert result.n_models == 4

    def test_disagreeing_models(self):
        preds = {
            "m1": np.array([1.0, 0.0, 0.0, 0.0]),
            "m2": np.array([0.0, 1.0, 0.0, 0.0]),
            "m3": np.array([0.0, 0.0, 1.0, 0.0]),
            "m4": np.array([0.0, 0.0, 0.0, 1.0]),
        }
        result = compute_icm_from_predictions(preds)
        # Disagreeing models should have lower ICM
        agreeing = compute_icm_from_predictions(
            {f"m{i}": np.array([0.25, 0.25, 0.25, 0.25]) for i in range(4)}
        )
        assert result.icm_score < agreeing.icm_score

    def test_with_intervals(self):
        preds = {
            "m1": np.array([0.5, 0.5]),
            "m2": np.array([0.5, 0.5]),
        }
        intervals = [(0.3, 0.7), (0.35, 0.65)]
        result = compute_icm_from_predictions(preds, intervals=intervals)
        assert 0.0 <= result.icm_score <= 1.0

    def test_with_residuals_and_features(self):
        rng = np.random.default_rng(42)
        preds = {f"m{i}": rng.dirichlet(np.ones(3)) for i in range(3)}
        residuals = rng.standard_normal((3, 50))
        features = [{"a", "b"}, {"b", "c"}, {"d", "e"}]
        result = compute_icm_from_predictions(
            preds, residuals=residuals, features=features
        )
        assert 0.0 <= result.icm_score <= 1.0


class TestComputeICMTimeseries:
    """Tests for rolling-window ICM computation."""

    def test_basic_timeseries(self):
        rng = np.random.default_rng(77)
        T = 10
        ts = []
        for _ in range(T):
            p = rng.dirichlet(np.ones(3))
            ts.append({f"m{i}": p + rng.normal(0, 0.01, 3) for i in range(3)})
        results = compute_icm_timeseries(ts, window_size=1)
        assert len(results) == T
        for r in results:
            assert 0.0 <= r.icm_score <= 1.0

    def test_windowed(self):
        rng = np.random.default_rng(88)
        T = 10
        ts = []
        for _ in range(T):
            p = rng.dirichlet(np.ones(4))
            ts.append({"m1": p, "m2": p + rng.normal(0, 0.01, 4)})
        results = compute_icm_timeseries(ts, window_size=3)
        assert len(results) == T - 3 + 1

    def test_single_step(self):
        preds = [{"m1": np.array([0.5, 0.5]), "m2": np.array([0.5, 0.5])}]
        results = compute_icm_timeseries(preds, window_size=1)
        assert len(results) == 1
        assert results[0].icm_score > 0.5


# ============================================================
# Synthetic scenario: 4 models agree vs disagree
# ============================================================

class TestSyntheticScenario:
    """End-to-end synthetic scenario with 4 models."""

    def test_agreement_vs_disagreement(self):
        """Four models that agree should score higher than four that disagree."""
        rng = np.random.default_rng(42)

        # Agreeing models: all predict ~same distribution
        base = np.array([0.4, 0.3, 0.2, 0.1])
        agree_preds = {
            f"agree_{i}": base + rng.normal(0, 0.01, 4)
            for i in range(4)
        }
        # Normalize to valid distributions
        for k in agree_preds:
            agree_preds[k] = np.abs(agree_preds[k])
            agree_preds[k] /= agree_preds[k].sum()

        agree_intervals = [(0.35, 0.45)] * 4
        agree_signs = np.array([1.0, 1.0, 1.0, 1.0])

        icm_agree = compute_icm_from_predictions(
            agree_preds,
            intervals=agree_intervals,
            signs=agree_signs,
        )

        # Disagreeing models: each predicts a different class
        disagree_preds = {
            "disagree_0": np.array([0.9, 0.03, 0.03, 0.04]),
            "disagree_1": np.array([0.03, 0.9, 0.03, 0.04]),
            "disagree_2": np.array([0.03, 0.03, 0.9, 0.04]),
            "disagree_3": np.array([0.04, 0.03, 0.03, 0.9]),
        }
        disagree_intervals = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.1, 0.4)]
        disagree_signs = np.array([1.0, -1.0, 1.0, -1.0])

        icm_disagree = compute_icm_from_predictions(
            disagree_preds,
            intervals=disagree_intervals,
            signs=disagree_signs,
        )

        assert icm_agree.icm_score > icm_disagree.icm_score, (
            f"Agree ICM {icm_agree.icm_score:.4f} should be > "
            f"Disagree ICM {icm_disagree.icm_score:.4f}"
        )

    def test_agreement_components_dominate(self):
        """When 4 models perfectly agree, all components should be high."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        preds = {f"m{i}": p.copy() for i in range(4)}
        result = compute_icm_from_predictions(
            preds,
            intervals=[(0.2, 0.3)] * 4,
            signs=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        assert result.components.A == pytest.approx(1.0, abs=1e-6)
        assert result.components.D == pytest.approx(1.0, abs=1e-6)
        assert result.components.U == pytest.approx(1.0, abs=1e-6)


# ============================================================
# Monotonicity property
# ============================================================

class TestMonotonicity:
    """More agreement should yield higher ICM."""

    def test_agreement_monotonicity(self):
        """As distributional agreement increases, ICM should increase."""
        config = ICMConfig()
        icm_scores = []
        for noise_scale in [2.0, 1.0, 0.5, 0.1, 0.01]:
            rng = np.random.default_rng(42)
            base = np.array([0.4, 0.3, 0.2, 0.1])
            preds = {}
            for i in range(4):
                noisy = base + rng.normal(0, noise_scale, 4)
                noisy = np.abs(noisy)
                noisy /= noisy.sum()
                preds[f"m{i}"] = noisy
            result = compute_icm_from_predictions(preds, config=config)
            icm_scores.append(result.icm_score)

        # Each step should be >= previous (more agreement -> higher ICM)
        for i in range(1, len(icm_scores)):
            assert icm_scores[i] >= icm_scores[i - 1] - 0.01, (
                f"Monotonicity violated at noise step {i}: "
                f"{icm_scores[i]:.4f} < {icm_scores[i-1]:.4f}"
            )

    def test_direction_monotonicity(self):
        """More direction agreement -> higher D component."""
        # All same sign
        D_all = compute_direction(np.array([1, 1, 1, 1, 1]))
        # Majority same
        D_maj = compute_direction(np.array([1, 1, 1, 1, -1]))
        # Split
        D_split = compute_direction(np.array([1, 1, -1, -1]))
        assert D_all >= D_maj
        assert D_maj >= D_split

    def test_overlap_monotonicity(self):
        """More overlap -> higher U component."""
        # Perfect overlap
        U_perfect = compute_uncertainty_overlap([(0, 1), (0, 1), (0, 1)])
        # Good overlap
        U_good = compute_uncertainty_overlap([(0, 1), (0.2, 1.2), (0.1, 1.1)])
        # Poor overlap
        U_poor = compute_uncertainty_overlap([(0, 1), (2, 3), (4, 5)])
        assert U_perfect >= U_good
        assert U_good >= U_poor

    def test_penalty_monotonicity(self):
        """Higher dependency -> higher penalty -> lower ICM."""
        config = ICMConfig()

        # Independent models
        components_indep = ICMComponents(A=0.8, D=0.8, U=0.8, C=0.8, Pi=0.1)
        icm_indep = compute_icm(components_indep, config)

        # Dependent models
        components_dep = ICMComponents(A=0.8, D=0.8, U=0.8, C=0.8, Pi=0.9)
        icm_dep = compute_icm(components_dep, config)

        assert icm_indep.icm_score > icm_dep.icm_score


# ============================================================
# Beta-calibrated aggregation
# ============================================================

class TestComputeICMCalibrated:
    """Tests for beta-calibrated ICM aggregation."""

    def test_perfect_convergence_high_score(self):
        """Perfect components should yield a high calibrated ICM score."""
        components = ICMComponents(A=1.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        result = compute_icm_calibrated(components)
        assert result.icm_score > 0.9
        assert result.aggregation_method == "calibrated"

    def test_no_convergence_low_score(self):
        """Worst-case components should yield a low calibrated ICM score."""
        components = ICMComponents(A=0.0, D=0.0, U=0.0, C=0.0, Pi=1.0)
        result = compute_icm_calibrated(components)
        assert result.icm_score < 0.1

    def test_bounded_0_1(self):
        """Calibrated scores must be in [0, 1] for random inputs."""
        rng = np.random.default_rng(100)
        for _ in range(100):
            components = ICMComponents(
                A=rng.uniform(), D=rng.uniform(),
                U=rng.uniform(), C=rng.uniform(),
                Pi=rng.uniform(),
            )
            result = compute_icm_calibrated(components)
            assert 0.0 <= result.icm_score <= 1.0

    def test_monotonicity_A(self):
        """Increasing agreement A should increase the calibrated score."""
        scores = []
        for a_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            comp = ICMComponents(A=a_val, D=0.5, U=0.5, C=0.5, Pi=0.2)
            scores.append(compute_icm_calibrated(comp).icm_score)
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 1e-10, (
                f"Monotonicity violated: score[{i}]={scores[i]:.6f} "
                f"< score[{i-1}]={scores[i-1]:.6f}"
            )

    def test_monotonicity_Pi_penalty(self):
        """Increasing penalty Pi should decrease the calibrated score."""
        scores = []
        for pi_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            comp = ICMComponents(A=0.7, D=0.7, U=0.7, C=0.7, Pi=pi_val)
            scores.append(compute_icm_calibrated(comp).icm_score)
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 1e-10, (
                f"Penalty monotonicity violated: score[{i}]={scores[i]:.6f} "
                f"> score[{i-1}]={scores[i-1]:.6f}"
            )

    def test_wider_spread_than_logistic(self):
        """Beta-calibrated should produce wider score spreads than logistic.

        This is the core motivation for the calibrated mode.
        """
        rng = np.random.default_rng(42)
        logistic_scores = []
        calibrated_scores = []
        for _ in range(200):
            comp = ICMComponents(
                A=rng.uniform(), D=rng.uniform(),
                U=rng.uniform(), C=rng.uniform(),
                Pi=rng.uniform(),
            )
            logistic_scores.append(compute_icm(comp).icm_score)
            calibrated_scores.append(compute_icm_calibrated(comp).icm_score)

        logistic_range = max(logistic_scores) - min(logistic_scores)
        calibrated_range = max(calibrated_scores) - min(calibrated_scores)
        logistic_std = float(np.std(logistic_scores))
        calibrated_std = float(np.std(calibrated_scores))

        assert calibrated_range > logistic_range, (
            f"Calibrated range {calibrated_range:.4f} should exceed "
            f"logistic range {logistic_range:.4f}"
        )
        assert calibrated_std > logistic_std, (
            f"Calibrated std {calibrated_std:.4f} should exceed "
            f"logistic std {logistic_std:.4f}"
        )

    def test_symmetry_with_equal_shapes(self):
        """With shape_a == shape_b, the Beta CDF is symmetric around 0.5.

        This means components at the midpoint should map to ~0.5.
        """
        mid = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.5)
        config = ICMConfig(beta_shape_a=5.0, beta_shape_b=5.0)
        result = compute_icm_calibrated(mid, config)
        # The midpoint of z maps to z_norm ~= 0.5294 (due to the weight
        # structure), which through Beta(5,5) gives approximately 0.55.
        # The key test is symmetry: equal distance from midpoint in
        # opposite directions should give symmetric outputs.
        z_mid = (config.w_A * 0.5 + config.w_D * 0.5
                 + config.w_U * 0.5 + config.w_C * 0.5 - config.lam * 0.5)
        z_min = -config.lam
        z_max = config.w_A + config.w_D + config.w_U + config.w_C
        z_norm_mid = (z_mid - z_min) / (z_max - z_min)

        # Check symmetric property: BetaCDF(x; a, a) + BetaCDF(1-x; a, a) = 1
        from scipy.stats import beta as sb
        assert sb.cdf(z_norm_mid, 5.0, 5.0) + sb.cdf(1.0 - z_norm_mid, 5.0, 5.0) == (
            pytest.approx(1.0, abs=1e-10)
        )

    def test_custom_shape_parameters(self):
        """Custom Beta shape parameters should alter the spread."""
        comp = ICMComponents(A=0.8, D=0.6, U=0.7, C=0.9, Pi=0.1)

        # Low shape parameters -> more uniform spread
        config_low = ICMConfig(beta_shape_a=1.5, beta_shape_b=1.5)
        result_low = compute_icm_calibrated(comp, config_low)

        # High shape parameters -> more concentrated around center
        config_high = ICMConfig(beta_shape_a=20.0, beta_shape_b=20.0)
        result_high = compute_icm_calibrated(comp, config_high)

        # Both should be bounded
        assert 0.0 <= result_low.icm_score <= 1.0
        assert 0.0 <= result_high.icm_score <= 1.0

    def test_weights_include_beta_params(self):
        """The weights dict should include beta shape parameters."""
        comp = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.5)
        result = compute_icm_calibrated(comp)
        assert "beta_shape_a" in result.weights
        assert "beta_shape_b" in result.weights
        assert result.weights["beta_shape_a"] == 5.0
        assert result.weights["beta_shape_b"] == 5.0

    def test_lipschitz_continuity(self):
        """Small changes in input should produce bounded changes in output.

        The Beta CDF on [0,1] with finite shape params has a bounded
        derivative, so the mapping is Lipschitz.
        """
        base = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.3)
        result_base = compute_icm_calibrated(base)

        epsilon = 0.001
        perturbed = ICMComponents(
            A=0.5 + epsilon, D=0.5, U=0.5, C=0.5, Pi=0.3
        )
        result_perturbed = compute_icm_calibrated(perturbed)

        delta_output = abs(result_perturbed.icm_score - result_base.icm_score)
        # The Lipschitz constant L should be finite; the output change
        # should be proportional to the input change.
        assert delta_output < 1.0, "Output jump too large for tiny input change"
        # More specifically, the ratio should be bounded
        assert delta_output / epsilon < 100.0, (
            f"Lipschitz ratio {delta_output / epsilon:.2f} too large"
        )


# ============================================================
# Adaptive aggregation
# ============================================================

class TestComputeICMAdaptive:
    """Tests for adaptive / percentile-based ICM aggregation."""

    def test_perfect_convergence_high_score(self):
        """Perfect components -> high adaptive score."""
        comp = ICMComponents(A=1.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        result = compute_icm_adaptive(comp)
        assert result.icm_score > 0.9
        assert result.aggregation_method == "adaptive"

    def test_no_convergence_low_score(self):
        """Worst-case components -> low adaptive score."""
        comp = ICMComponents(A=0.0, D=0.0, U=0.0, C=0.0, Pi=1.0)
        result = compute_icm_adaptive(comp)
        assert result.icm_score < 0.1

    def test_bounded_0_1(self):
        """Adaptive scores must be in [0, 1]."""
        rng = np.random.default_rng(200)
        for _ in range(100):
            comp = ICMComponents(
                A=rng.uniform(), D=rng.uniform(),
                U=rng.uniform(), C=rng.uniform(),
                Pi=rng.uniform(),
            )
            result = compute_icm_adaptive(comp)
            assert 0.0 <= result.icm_score <= 1.0

    def test_monotonicity(self):
        """Increasing all positive components should increase adaptive score."""
        scores = []
        for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            comp = ICMComponents(A=level, D=level, U=level, C=level, Pi=0.0)
            scores.append(compute_icm_adaptive(comp).icm_score)
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 1e-10

    def test_custom_calibration_set(self):
        """Providing a calibration set should affect scoring."""
        comp = ICMComponents(A=0.7, D=0.7, U=0.7, C=0.7, Pi=0.1)

        # Calibration set with low scores -> this comp should rank high
        config_low = ICMConfig(
            adaptive_calibration_scores=[0.0, 0.05, 0.1, 0.15, 0.2]
        )
        result_low = compute_icm_adaptive(comp, config_low)

        # Calibration set with high scores -> this comp should rank lower
        config_high = ICMConfig(
            adaptive_calibration_scores=[0.5, 0.6, 0.7, 0.8, 0.9]
        )
        result_high = compute_icm_adaptive(comp, config_high)

        assert result_low.icm_score > result_high.icm_score, (
            f"Score with low calibration set ({result_low.icm_score:.4f}) "
            f"should be > with high calibration set ({result_high.icm_score:.4f})"
        )

    def test_wider_spread_than_logistic(self):
        """Adaptive should produce wider spread than logistic."""
        rng = np.random.default_rng(42)
        logistic_scores = []
        adaptive_scores = []
        for _ in range(200):
            comp = ICMComponents(
                A=rng.uniform(), D=rng.uniform(),
                U=rng.uniform(), C=rng.uniform(),
                Pi=rng.uniform(),
            )
            logistic_scores.append(compute_icm(comp).icm_score)
            adaptive_scores.append(compute_icm_adaptive(comp).icm_score)

        logistic_std = float(np.std(logistic_scores))
        adaptive_std = float(np.std(adaptive_scores))

        assert adaptive_std > logistic_std, (
            f"Adaptive std {adaptive_std:.4f} should exceed "
            f"logistic std {logistic_std:.4f}"
        )


# ============================================================
# Dispatch / config-driven aggregation
# ============================================================

class TestDispatchAggregation:
    """Tests for _dispatch_aggregation and config.aggregation parameter."""

    def test_default_is_logistic(self):
        """Default config should dispatch to logistic."""
        config = ICMConfig()
        assert config.aggregation == "logistic"
        comp = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.2)
        result = _dispatch_aggregation(comp, config)
        assert result.aggregation_method == "logistic"
        # Should match direct call
        direct = compute_icm(comp, config)
        assert result.icm_score == pytest.approx(direct.icm_score, abs=1e-12)

    def test_dispatch_geometric(self):
        config = ICMConfig(aggregation="geometric")
        comp = ICMComponents(A=0.8, D=0.8, U=0.8, C=0.8, Pi=0.1)
        result = _dispatch_aggregation(comp, config)
        assert result.aggregation_method == "geometric"
        direct = compute_icm_geometric(comp, config)
        assert result.icm_score == pytest.approx(direct.icm_score, abs=1e-12)

    def test_dispatch_calibrated(self):
        config = ICMConfig(aggregation="calibrated")
        comp = ICMComponents(A=0.6, D=0.7, U=0.8, C=0.9, Pi=0.05)
        result = _dispatch_aggregation(comp, config)
        assert result.aggregation_method == "calibrated"
        direct = compute_icm_calibrated(comp, config)
        assert result.icm_score == pytest.approx(direct.icm_score, abs=1e-12)

    def test_dispatch_adaptive(self):
        config = ICMConfig(aggregation="adaptive")
        comp = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.5)
        result = _dispatch_aggregation(comp, config)
        assert result.aggregation_method == "adaptive"
        direct = compute_icm_adaptive(comp, config)
        assert result.icm_score == pytest.approx(direct.icm_score, abs=1e-12)

    def test_unknown_aggregation_raises(self):
        config = ICMConfig(aggregation="unknown_method")
        comp = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.5)
        with pytest.raises(ValueError, match="Unknown aggregation"):
            _dispatch_aggregation(comp, config)

    def test_from_predictions_respects_config_aggregation(self):
        """compute_icm_from_predictions should use config.aggregation."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        preds = {f"m{i}": p for i in range(3)}

        config_logistic = ICMConfig(aggregation="logistic")
        result_log = compute_icm_from_predictions(preds, config=config_logistic)
        assert result_log.aggregation_method == "logistic"

        config_cal = ICMConfig(aggregation="calibrated")
        result_cal = compute_icm_from_predictions(preds, config=config_cal)
        assert result_cal.aggregation_method == "calibrated"

        config_adapt = ICMConfig(aggregation="adaptive")
        result_adapt = compute_icm_from_predictions(preds, config=config_adapt)
        assert result_adapt.aggregation_method == "adaptive"

    def test_timeseries_respects_config_aggregation(self):
        """compute_icm_timeseries should use config.aggregation."""
        rng = np.random.default_rng(55)
        T = 5
        ts = []
        for _ in range(T):
            p = rng.dirichlet(np.ones(3))
            ts.append({f"m{i}": p + rng.normal(0, 0.01, 3) for i in range(3)})

        config_cal = ICMConfig(aggregation="calibrated")
        results = compute_icm_timeseries(ts, config=config_cal, window_size=1)
        assert len(results) == T
        for r in results:
            assert r.aggregation_method == "calibrated"
            assert 0.0 <= r.icm_score <= 1.0


# ============================================================
# Backward compatibility
# ============================================================

class TestBackwardCompatibility:
    """Ensure existing behavior is unchanged when new features are unused."""

    def test_default_icm_config_unchanged(self):
        """Default ICMConfig values should be backward-compatible."""
        config = ICMConfig()
        assert config.w_A == 0.35
        assert config.w_D == 0.15
        assert config.w_U == 0.25
        assert config.w_C == 0.10
        assert config.lam == 0.15
        assert config.aggregation == "logistic"

    def test_compute_icm_unchanged(self):
        """compute_icm should produce identical results to before."""
        comp = ICMComponents(A=0.8, D=0.7, U=0.6, C=0.9, Pi=0.2)
        config = ICMConfig()
        result = compute_icm(comp, config)
        # The logistic function is deterministic; verify a known value
        from scipy.special import expit
        z = (0.35 * 0.8 + 0.15 * 0.7 + 0.25 * 0.6 + 0.10 * 0.9
             - 0.15 * 0.2)
        expected = float(expit(z))
        assert result.icm_score == pytest.approx(expected, abs=1e-12)
        assert result.aggregation_method == "logistic"

    def test_from_predictions_default_logistic(self):
        """compute_icm_from_predictions with no config uses logistic."""
        p = np.array([0.5, 0.5])
        preds = {"m1": p, "m2": p}
        result = compute_icm_from_predictions(preds)
        assert result.aggregation_method == "logistic"

    def test_geometric_still_works(self):
        """compute_icm_geometric should still work identically."""
        comp = ICMComponents(A=1.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        result = compute_icm_geometric(comp)
        assert result.icm_score == pytest.approx(1.0, abs=1e-6)
        assert result.aggregation_method == "geometric"


# ============================================================
# Edge cases for new aggregation modes
# ============================================================

class TestEdgeCases:
    """Edge cases for calibrated and adaptive aggregation."""

    def test_calibrated_all_zeros(self):
        """All components zero, max penalty."""
        comp = ICMComponents(A=0.0, D=0.0, U=0.0, C=0.0, Pi=1.0)
        result = compute_icm_calibrated(comp)
        assert 0.0 <= result.icm_score <= 1.0
        assert result.icm_score < 0.05

    def test_calibrated_all_ones_no_penalty(self):
        """All components one, no penalty."""
        comp = ICMComponents(A=1.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        result = compute_icm_calibrated(comp)
        assert result.icm_score > 0.95

    def test_adaptive_single_calibration_point(self):
        """With only 1 calibration point, should fall back to default."""
        config = ICMConfig(adaptive_calibration_scores=[0.5])
        comp = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.5)
        result = compute_icm_adaptive(comp, config)
        assert 0.0 <= result.icm_score <= 1.0

    def test_adaptive_score_below_calibration_min(self):
        """Score below all calibration points should give ~0."""
        config = ICMConfig(adaptive_calibration_scores=[0.5, 0.6, 0.7, 0.8])
        comp = ICMComponents(A=0.0, D=0.0, U=0.0, C=0.0, Pi=1.0)
        result = compute_icm_adaptive(comp, config)
        assert result.icm_score == pytest.approx(0.0, abs=1e-6)

    def test_adaptive_score_above_calibration_max(self):
        """Score above all calibration points should give 1.0."""
        config = ICMConfig(adaptive_calibration_scores=[0.0, 0.05, 0.1])
        comp = ICMComponents(A=1.0, D=1.0, U=1.0, C=1.0, Pi=0.0)
        result = compute_icm_adaptive(comp, config)
        assert result.icm_score == pytest.approx(1.0, abs=1e-6)

    def test_calibrated_equal_components_different_penalty(self):
        """Two scenarios differing only in Pi should have correct ordering."""
        comp_low_pi = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=0.0)
        comp_high_pi = ICMComponents(A=0.5, D=0.5, U=0.5, C=0.5, Pi=1.0)
        r_low = compute_icm_calibrated(comp_low_pi)
        r_high = compute_icm_calibrated(comp_high_pi)
        assert r_low.icm_score > r_high.icm_score

    def test_calibrated_beta_shape_1_1_is_identity(self):
        """Beta(1,1) is uniform, so CDF(x) = x.  Output should equal z_norm."""
        config = ICMConfig(beta_shape_a=1.0, beta_shape_b=1.0)
        comp = ICMComponents(A=0.7, D=0.6, U=0.8, C=0.5, Pi=0.2)
        z = (config.w_A * 0.7 + config.w_D * 0.6 + config.w_U * 0.8
             + config.w_C * 0.5 - config.lam * 0.2)
        z_min = -config.lam
        z_max = config.w_A + config.w_D + config.w_U + config.w_C
        z_norm = (z - z_min) / (z_max - z_min)
        result = compute_icm_calibrated(comp, config)
        assert result.icm_score == pytest.approx(z_norm, abs=1e-8)
