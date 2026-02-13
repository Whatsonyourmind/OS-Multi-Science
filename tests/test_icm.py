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
