# Experiment Q6: Structural Invariants vs Scalar ICM Stability

## Hypothesis

> Partial output invariants (ranking preservation, sign consistency,
> monotonicity patterns) are more stable across perturbations and
> domains than scalar ICM convergence scores.

## Verdict: NOT SUPPORTED

---

## Experimental Design

| Parameter           | Value                                      |
|---------------------|--------------------------------------------|
| Perturbation levels | 8 (from 0.02 to 0.80)     |
| Repetitions         | 12                                         |
| Models per trial    | 5                                          |
| Samples per scenario| 60                                         |
| Top-k fraction      | 0.2                                       |
| Stability metric    | Coefficient of Variation (CV = std/|mean|) |
| Statistical test    | Wilcoxon signed-rank (one-sided, alpha=0.05)|

### Scenarios

1. **Classification (3-class Gaussian)** -- K models produce class-
   probability vectors with noise-controlled accuracy. ICM uses
   Hellinger distance for distributional agreement.

2. **Regression (1-D noisy sine)** -- Ground truth y = sin(2*pi*x).
   Each model predicts y + Normal(0, perturbation). ICM uses
   Wasserstein distance.

3. **Network cascade (Erdos-Renyi contagion)** -- Cascade on a
   40-node graph with edge perturbations proportional to noise.
   ICM uses Wasserstein distance.

### Structural Invariants

1. **Ranking invariant** -- Mean pairwise Kendall-tau between model
   rankings of instances by predicted value.
2. **Sign invariant** -- Fraction of instances where all models agree
   on the sign of the prediction.
3. **Monotonicity invariant** -- Mean pairwise Spearman correlation
   among model predictions.
4. **Ordering invariant** -- Mean pairwise Jaccard similarity of
   top-20% instance sets ranked by predicted value.

---

## Aggregate Stability Results (CV)

| Scenario | CV(ICM) | CV(Ranking) | CV(Sign) | CV(Monotonicity) | CV(Ordering) |
|----------|---------|-------------|----------|------------------|--------------|
| classification | 0.0136 +/- 0.0035 | 5.5891 +/- 3.7030 | 0.0000 +/- 0.0000 | 5.4244 +/- 4.0747 | 0.1557 +/- 0.0464 |
| regression | 0.0286 +/- 0.0052 | 0.3683 +/- 0.0111 | 0.3289 +/- 0.0261 | 0.2580 +/- 0.0124 | 0.4877 +/- 0.0361 |
| cascade | 0.0181 +/- 0.0052 | 0.9123 +/- 0.1514 | 0.0000 +/- 0.0000 | 0.9133 +/- 0.1494 | 0.0000 +/- 0.0000 |

Lower CV indicates higher stability. Structural invariants with
lower CV than ICM are more stable across perturbations.

## Stability Ratios: CV(ICM) / CV(Invariant)

Values > 1.0 indicate the invariant is MORE stable than scalar ICM.

| Scenario | Ranking | Sign | Monotonicity | Ordering |
|----------|---------|------|--------------|----------|
| classification | 0.0042 | inf | 0.0042 | 0.1002 |
| regression | 0.0775 | 0.0875 | 0.1109 | 0.0589 |
| cascade | 0.0199 | inf | 0.0199 | inf |

## Statistical Tests (Wilcoxon Signed-Rank)

H0: CV(ICM) = CV(invariant); H1: CV(ICM) > CV(invariant)

| Scenario | Invariant | W-statistic | p-value | Significant |
|----------|-----------|-------------|---------|-------------|
| classification | ranking | 0.0 | 1.0000 | no |
| classification | sign | 78.0 | 0.0002 | YES |
| classification | monotonicity | 0.0 | 1.0000 | no |
| classification | ordering | 0.0 | 1.0000 | no |
| regression | ranking | 0.0 | 1.0000 | no |
| regression | sign | 0.0 | 1.0000 | no |
| regression | monotonicity | 0.0 | 1.0000 | no |
| regression | ordering | 0.0 | 1.0000 | no |
| cascade | ranking | 0.0 | 1.0000 | no |
| cascade | sign | 78.0 | 0.0002 | YES |
| cascade | monotonicity | 0.0 | 1.0000 | no |
| cascade | ordering | 78.0 | 0.0002 | YES |

## Cross-Domain Summary

| Invariant | Mean Stability Ratio | Significant Tests | Verdict |
|-----------|---------------------|-------------------|---------|
| ranking | 0.0339 | 0/3 | LESS STABLE |
| sign | 0.0875 (2 inf) | 2/3 | MORE STABLE |
| monotonicity | 0.0450 | 0/3 | LESS STABLE |
| ordering | 0.0795 (1 inf) | 1/3 | LESS STABLE |

---

## Interpretation

### Why the ICM score is more stable than structural invariants

The scalar ICM score demonstrates lower coefficient of variation
(higher stability) across perturbation levels than most structural
invariants. This is explained by two key factors:

1. **Logistic compression**: The ICM score maps through a sigmoid,
   which compresses variation in the component scores into a narrow
   output range (typically 0.6--0.67). This inherently reduces CV.

2. **Invariant sensitivity to perturbation**: Structural invariants
   like ranking (Kendall-tau) and monotonicity (Spearman) measure
   inter-model agreement directly. As perturbation grows, models
   diverge, and these metrics drop from near-perfect (~1.0) to
   near-zero, producing large CV values.

3. **Notable exception -- Sign invariant**: In classification and
   cascade scenarios, the sign invariant achieves perfect stability
   (CV = 0) because all predictions remain positive. This invariant
   IS more stable than ICM in those domains.

### What the invariants reveal instead

Although structural invariants are not more stable in the CV sense,
they provide **more informative signals** about the nature of model
disagreement. The ranking and monotonicity invariants degrade
monotonically with perturbation magnitude, offering a clear
diagnostic of how perturbation affects inter-model relationships.
The ICM score, by contrast, remains relatively flat, which may
mask genuine structural degradation.

### Practical implications

The ICM score's stability is partly an artifact of logistic
compression rather than genuine robustness. Structural invariants,
while having higher CV, are more sensitive diagnostic tools for
detecting when perturbations fundamentally alter multi-model
agreement patterns. They are complementary to ICM, not
replacements.

## Conclusion

The hypothesis is **NOT SUPPORTED**: 1/4 structural
invariants show higher stability (lower CV) than the scalar ICM score
across perturbations.

The scalar ICM score is more stable (lower CV) than ranking,
monotonicity, and ordering invariants, primarily due to logistic
compression. However, the sign invariant achieves perfect stability
(CV = 0) in classification and cascade scenarios where all
predictions remain positive.

Structural invariants are not more stable in the CV sense, but
they are more informative diagnostics: they degrade monotonically
with perturbation magnitude while the ICM score remains relatively
flat. This suggests they should be used as complementary tools
alongside ICM, not as replacements.

---

## Reproducibility

- Script: `experiments/q6_structural_invariants.py`
- Command: `python experiments/q6_structural_invariants.py`
- Dependencies: numpy, scipy, sklearn
- Runtime: approximately 5 seconds
- Base seed: 2026 (deterministic results)
