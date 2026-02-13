# Computational Complexity Analysis -- OS Multi-Science

This document provides a rigorous analysis of the time and space complexity
for every computational stage of the OS Multi-Science framework. Notation
conventions used throughout:

| Symbol | Meaning |
|--------|---------|
| K | Number of models (epistemic methods) |
| N | Number of samples (observations) |
| D | Dimensionality of features / embedding space |
| C | Number of classes (for discrete distributions) |
| T | Number of time steps |
| P | Number of permutations (HSIC / negative controls) |
| M | Calibration set size (adaptive aggregation) |
| W | Rolling window size |

---

## 1. Per-Component Complexity

### 1.1 Agreement -- A

**Function:** `compute_agreement(predictions, distance_fn, ...)`

The agreement component computes all pairwise distances among K model
predictions and averages them.

- **Pairs computed:** K(K-1)/2
- **Per-pair cost:** depends on the distance function (see Section 2)

With Hellinger distance (the default, operating on C-dimensional probability
vectors):

| Metric | Complexity |
|--------|------------|
| Time | O(K^2 * C) |
| Space | O(K^2 + K*C) |

**Bottleneck:** The double loop over K model pairs. For Hellinger the per-pair
cost is O(C) (element-wise sqrt, subtract, square, sum). The quadratic growth
in K dominates when K is large.

With MMD distance on N-sample embeddings, the per-pair cost becomes O(N^2*D),
making the total O(K^2 * N^2 * D) -- a severe bottleneck.

### 1.2 Direction -- D

**Function:** `compute_direction(signs_or_gradients)`

Extracts signs from K gradient values, computes `np.unique` to get the
empirical sign distribution, then computes Shannon entropy.

| Metric | Complexity |
|--------|------------|
| Time | O(K log K) |
| Space | O(K) |

**Bottleneck:** `np.unique` with sorting is O(K log K). The entropy
computation over at most 3 categories (+1, 0, -1) is O(1).

This is the cheapest ICM component.

### 1.3 Uncertainty Overlap -- U

**Function:** `compute_uncertainty_overlap(intervals)`

Computes pairwise interval-overlap (Jaccard on intervals) for K intervals.

| Metric | Complexity |
|--------|------------|
| Time | O(K^2) |
| Space | O(K^2) |

**Bottleneck:** K(K-1)/2 pairwise comparisons, each O(1) (min/max
operations on interval endpoints).

### 1.4 Invariance -- C

**Function:** `compute_invariance(pre_scores, post_scores)`

Computes the L2-norm of the difference between pre- and post-perturbation
score vectors of length K.

| Metric | Complexity |
|--------|------------|
| Time | O(K) |
| Space | O(K) |

**Bottleneck:** Single pass computing the vector norm. This is trivially
cheap.

### 1.5 Dependency Penalty -- Pi

**Function:** `compute_dependency_penalty(residuals, features, gradients, ...)`

Pi is a weighted combination of three sub-components:

#### 1.5.1 Residual correlation (Ledoit-Wolf)

Given a (K, N) residual matrix:

1. Standardize each row: O(K * N)
2. Compute sample correlation S = Z @ Z^T / N: O(K^2 * N)
3. Compute shrinkage intensity (trace(S^2), trace(S)^2): O(K^2) for
   the matrix product S @ S, then O(K) for traces
4. Shrink: O(K^2)

| Metric | Complexity |
|--------|------------|
| Time | O(K^2 * N) |
| Space | O(K * N + K^2) |

#### 1.5.2 Feature/provenance overlap (Jaccard)

For K sets of feature names, with average set size F:

| Metric | Complexity |
|--------|------------|
| Time | O(K^2 * F) |
| Space | O(K * F) |

#### 1.5.3 Gradient similarity (cosine)

For K gradient vectors of dimension D:

| Metric | Complexity |
|--------|------------|
| Time | O(K^2 * D) |
| Space | O(K * D) |

#### 1.5.4 Overall Pi

| Metric | Complexity |
|--------|------------|
| Time | O(K^2 * N + K^2 * F + K^2 * D) = O(K^2 * max(N, F, D)) |
| Space | O(K * N + K^2) |

**Bottleneck:** The Ledoit-Wolf residual correlation (the matrix product
Z @ Z^T with shape K x N times N x K). When N >> K the matrix multiply
dominates.

### 1.6 Component Summary Table

| Component | Time | Space | Bottleneck |
|-----------|------|-------|------------|
| A (Agreement) | O(K^2 * C) [Hellinger] | O(K^2 + K*C) | Pairwise distance loop |
| D (Direction) | O(K log K) | O(K) | Sorting for np.unique |
| U (Uncertainty) | O(K^2) | O(K^2) | Pairwise interval overlap |
| C (Invariance) | O(K) | O(K) | Vector norm |
| Pi (Dependency) | O(K^2 * N) | O(K*N + K^2) | Ledoit-Wolf correlation |

---

## 2. Distance Function Complexity

All distance functions measure discrepancy between two distributions or
sample sets.

| Distance | Time | Space | Notes |
|----------|------|-------|-------|
| Hellinger | O(C) | O(C) | Element-wise sqrt, subtract, square, sum over C bins. Extremely fast; sub-0.02 ms even at C=500. |
| Wasserstein-2 (Gaussian) | O(D^3) | O(D^2) | Requires two matrix square roots via eigendecomposition (eigh) of D x D matrices: O(D^3) each. Traces and matrix products are O(D^2). Dominant cost is eigh. At D=100 this reaches ~8 ms. |
| Wasserstein-2 (Empirical, 1-D) | O(N log N) | O(N) | Sort both samples, interpolate to common grid, compute L2. Cheap: <0.25 ms at N=5000. |
| Wasserstein-2 (Empirical, multi-D) | O(N^3 log N) | O(N^2) | Uses POT library's EMD solver. Builds N x M cost matrix O(N*M*D), then solves linear program. The LP has super-cubic worst-case complexity. |
| MMD (RBF kernel) | O(N^2 * D) | O(N^2) | Builds three kernel matrices (N x N, M x M, N x M) via pairwise squared distances. Quadratic in N, which becomes the dominant bottleneck: 148 ms at N=1000. |

### Distance Function Scaling Behavior (Empirical)

```
Hellinger:     C=5 -> 0.015 ms,  C=500 -> 0.018 ms     (~constant)
W2-Gaussian:   D=2 -> 0.13 ms,   D=100 -> 7.9 ms        (~cubic in D)
W2-Emp-1D:     N=100 -> 0.06 ms, N=5000 -> 0.25 ms      (~N log N)
MMD-RBF:       N=50 -> 0.2 ms,   N=1000 -> 148 ms        (~quadratic in N)
```

---

## 3. Aggregation Complexity

All aggregation methods take the 5 pre-computed component scalars (A, D, U,
C, Pi) and produce a single ICM score.

| Method | Time per sample | Space | Notes |
|--------|-----------------|-------|-------|
| **Logistic** | O(1) | O(1) | Weighted sum + sigmoid (scipy.special.expit). 5 multiplications, 1 addition, 1 exp. |
| **Geometric** | O(1) | O(1) | 5 log operations + weighted sum + exp. Marginally more expensive than logistic due to log/exp, but still constant. |
| **Calibrated Beta CDF** | O(1) | O(1) | Weighted sum + normalization + scipy.stats.beta.cdf. The Beta CDF evaluation involves an incomplete beta function which is O(1) (pre-tabulated / series expansion). |
| **Adaptive** | O(log M) | O(M) | Weighted sum + np.searchsorted on a sorted calibration array of size M + linear interpolation. Dominated by the binary search O(log M). If no calibration set is provided, M defaults to 500 (generated once). |

The aggregation step is negligible compared to component computation in all
practical scenarios. Even the adaptive method's O(log M) binary search adds
only microseconds.

---

## 4. Full Pipeline Complexity

### 4.1 ICM Computation -- Single Sample

When computing ICM for a single observation with K models producing
C-dimensional probability vectors:

```
T_ICM = T_A + T_D + T_U + T_C + T_Pi + T_agg
      = O(K^2*C) + O(K log K) + O(K^2) + O(K) + O(K^2*N) + O(1)
      = O(K^2 * max(C, N))
```

In the common case where predictions are averaged distributions (C-dimensional
vectors, not per-sample), the Pi term with Ledoit-Wolf on (K, N) residuals
dominates.

If no residuals are provided (Pi defaults to 0), the dominant term is the
agreement computation: O(K^2 * C).

### 4.2 ICM Computation -- N Samples (Time Series)

`compute_icm_timeseries` with window_size W over T time steps:

- Number of windows: T - W + 1
- Per window: merge predictions (O(K * W * C) concatenation) + full ICM

```
T_timeseries = (T - W + 1) * O(K^2 * max(C, N_window))
```

where N_window = W * N_per_step.

### 4.3 CRC Gating

| Stage | Time | Space |
|-------|------|-------|
| Isotonic regression fit | O(N log N) | O(N) |
| Conformal calibration | O(N_cal) for residuals + O(N_cal log N_cal) for quantile | O(N_cal) |
| Threshold calibration | O(N log N) + O(T_sweep) where T_sweep = 200 thresholds | O(N) |
| Decision gate (single) | O(1) | O(1) |
| Risk-coverage curve | O(N * T_sweep) | O(T_sweep) |

**Total CRC setup:** O(N log N)
**Per-query decision:** O(1) (apply isotonic predict + add conformal quantile + threshold comparison)

The isotonic regression's `predict` method uses binary search internally:
O(log N_fit) per query.

### 4.4 Early Warning System

| Operation | Time | Space |
|-----------|------|-------|
| Rolling ICM (cumsum-based) | O(T) | O(T) |
| Delta ICM (finite differences) | O(T) | O(T) |
| Prediction variance | O(K * T) | O(K * T) for stacking, O(T) for result |
| Composite Z signal | O(T) | O(T) |
| CUSUM detector | O(T) | O(T) |
| Page-Hinkley detector | O(T) | O(T) |

**Total early warning:** O(K * T) dominated by the prediction variance
computation (stacking K arrays of length T).

All detectors are single-pass O(T) with O(1) state per step.

### 4.5 Anti-Spurious Pipeline

This is the most expensive component of the system.

| Stage | Time | Space |
|-------|------|-------|
| Observed pairwise distance | O(K^2 * N) via pdist | O(K * N) |
| Negative controls (P_ctrl sets) | O(P_ctrl * K * N) | O(K * N) per control |
| Baseline D_0 | O(P_ctrl * K^2 * N) | O(P_ctrl) for distances |
| HSIC test (one pair) | O(N^2) kernel build + O(P_perm * N^2) permutations | O(N^2) |
| HSIC all pairs | O(K^2 * P_perm * N^2) since K(K-1)/2 pairs | O(N^2) |
| FDR correction | O(K^2 log K) for sorting | O(K^2) |
| Ablation analysis | O(K * T_ICM) where T_ICM is time for one ICM call | O(K * N) |

**Total anti-spurious:**

```
T_anti = O(P_ctrl * K^2 * N)  +  O(K^2 * P_perm * N^2)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^
                                   DOMINANT TERM (HSIC)
```

The HSIC permutation test is the clear bottleneck. Each permutation requires
permuting an N x N kernel matrix and computing an element-wise product sum,
costing O(N^2). With K(K-1)/2 pairs and P_perm permutations per pair, this
grows rapidly.

### 4.6 Meta-Learner

| Operation | Time | Space |
|-----------|------|-------|
| Single weight evaluation | O(S * T_ICM) where S = number of scenarios | O(S) |
| Grid search (L points) | O(L * S * T_ICM) | O(L * S) |
| Scipy optimize (R restarts, I iterations) | O(R * I * S * T_ICM) | O(S) |
| Cross-validation (F folds) | O(F * R * I * S * T_ICM) | O(S) |

With defaults (L=50, R=5, I=200, F=5, S=20), grid search evaluates
50 * 20 = 1000 ICM computations. Scipy optimize evaluates approximately
5 * 200 * 20 = 20,000. Cross-validation: 5 * 5 * 200 * 20 = 100,000.

### 4.7 End-to-End Summary

For a single query with K models, N calibration samples, C classes:

```
T_total = T_ICM + T_CRC_query + T_early_warning_step
        = O(K^2 * C) + O(log N) + O(1)
        = O(K^2 * C)
```

For the full pipeline setup (one-time cost):

```
T_setup = T_CRC_fit + T_early_warning_init
        = O(N log N) + O(T)
```

For anti-spurious validation (periodic):

```
T_validation = O(K^2 * P_perm * N^2)
```

---

## 5. Empirical Runtime Measurements

All timings measured on Windows 11, NumPy 2.3.5. Times are median over
3-5 repetitions.

### 5.1 ICM Computation -- Varying K (N=500, C=5)

| K | Time (ms) | Pairs | Time/pair (ms) |
|---|-----------|-------|----------------|
| 2 | 0.52 | 1 | 0.518 |
| 3 | 1.28 | 3 | 0.425 |
| 5 | 1.00 | 10 | 0.100 |
| 7 | 1.99 | 21 | 0.095 |
| 10 | 2.56 | 45 | 0.057 |
| 15 | 6.28 | 105 | 0.060 |
| 20 | 7.27 | 190 | 0.038 |

Observations:
- Growth is sub-quadratic in practice due to Python overhead dominating at
  small K and NumPy vectorization amortizing costs at larger K.
- The per-pair time decreases as K grows, indicating fixed overhead per call
  is significant relative to the O(C) Hellinger computation.
- At K=20 (190 pairs), total time is still under 8 ms.

### 5.2 ICM Computation -- Varying N (K=5, C=5)

| N | Time (ms) |
|---|-----------|
| 100 | 0.26 |
| 500 | 0.45 |
| 1,000 | 0.26 |
| 2,000 | 0.31 |
| 5,000 | 0.26 |
| 10,000 | 0.26 |

Observations:
- ICM time is essentially independent of N when using pre-averaged
  distributions (C-dimensional vectors). The agreement computation operates
  on the C-dimensional summary, not raw samples.
- The slight variation is noise at the sub-millisecond level.

### 5.3 Distance Functions

| Distance | Input Size | Time (ms) |
|----------|-----------|-----------|
| Hellinger | C=5 | 0.015 |
| Hellinger | C=500 | 0.018 |
| W2-Gaussian | D=2 | 0.13 |
| W2-Gaussian | D=10 | 0.16 |
| W2-Gaussian | D=50 | 0.74 |
| W2-Gaussian | D=100 | 7.87 |
| W2-Emp-1D | N=100 | 0.06 |
| W2-Emp-1D | N=5000 | 0.25 |
| MMD-RBF | N=50, D=5 | 0.20 |
| MMD-RBF | N=100, D=5 | 0.32 |
| MMD-RBF | N=500, D=5 | 20.8 |
| MMD-RBF | N=1000, D=5 | 147.5 |

Key takeaways:
- **Hellinger** is negligible at all practical sizes.
- **W2-Gaussian** exhibits clear cubic growth in D due to eigendecomposition.
  At D=100 it is 60x slower than at D=2.
- **MMD** exhibits clear quadratic growth in N. At N=1000 it is 750x slower
  than at N=50. Using MMD with K=10 models would require 45 pairs * 148 ms =
  6.6 seconds just for the agreement component.

### 5.4 HSIC Permutation Test

| N | Permutations | Time (ms) | Time/perm (ms) |
|---|-------------|-----------|----------------|
| 50 | 100 | 5.8 | 0.058 |
| 50 | 1,000 | 66.7 | 0.067 |
| 100 | 100 | 12.8 | 0.128 |
| 100 | 1,000 | 83.3 | 0.083 |
| 200 | 500 | 120.7 | 0.241 |
| 200 | 1,000 | 254.2 | 0.254 |
| 500 | 100 | 631.7 | 6.317 |
| 500 | 500 | 2,606 | 5.213 |
| 500 | 1,000 | 4,947 | 4.946 |

Key takeaways:
- The HSIC kernel build is O(N^2) and dominates setup.
- Each permutation costs O(N^2) for the index permutation on the kernel
  matrix and the element-wise product sum.
- At N=500 with 1000 permutations, a single HSIC test takes ~5 seconds.
- For K=5 models (10 pairs), the full HSIC battery takes ~50 seconds.
  This is the single most expensive operation in the framework.

### 5.5 CRC Gating Pipeline

| N_cal | Isotonic fit (ms) | Conformalize (ms) | Threshold calibration (ms) |
|-------|-------------------|-------------------|---------------------------|
| 100 | 1.07 | 0.23 | 17.5 |
| 500 | 0.72 | 0.23 | 16.6 |
| 1,000 | 0.59 | 0.21 | 16.9 |
| 5,000 | 1.74 | 0.39 | 17.3 |

Observations:
- Isotonic regression fit is fast across all sizes (sklearn's implementation
  is O(N) via the pool-adjacent-violators algorithm, despite theoretical
  O(N log N) for sorting).
- Conformalize is sub-millisecond (compute residuals + quantile).
- Threshold calibration is ~17 ms regardless of N, dominated by the 200-point
  threshold sweep (each calling isotonic predict).

### 5.6 Early Warning Detectors

| T | Window | Rolling ICM (ms) | CUSUM (ms) | Page-Hinkley (ms) |
|---|--------|-------------------|------------|-------------------|
| 100 | 20 | 0.05 | 0.05 | 0.04 |
| 500 | 50 | 0.22 | 0.35 | 0.37 |
| 1,000 | 50 | 0.47 | 0.52 | 0.41 |
| 5,000 | 50 | 4.16 | 7.84 | 3.82 |
| 10,000 | 50 | 5.27 | 5.43 | 4.64 |

Observations:
- All detectors scale linearly with T as expected.
- Even at T=10,000 time steps, all three detectors complete in under 8 ms.
- The rolling ICM uses a cumsum-based O(T) algorithm.
- CUSUM and Page-Hinkley are Python-loop-based O(T), which explains the
  slightly higher constant factor.

### 5.7 Anti-Spurious Pipeline

| K | N | n_ctrl | n_perms | Neg ctrl (ms) | Full report (ms) |
|---|---|--------|---------|---------------|------------------|
| 3 | 100 | 50 | 100 | 1.5 | 33.7 |
| 5 | 100 | 100 | 500 | 3.6 | 531 |
| 5 | 500 | 100 | 500 | 13.1 | 26,822 |
| 5 | 500 | 100 | 1,000 | 9.3 | 52,844 |

Observations:
- Negative control generation is cheap: O(n_ctrl * K * N) with simple
  array permutations.
- The full report is dominated by HSIC tests. At N=500 with K=5
  (10 pairs) and 1000 permutations, the full report takes ~53 seconds.
- Doubling permutations roughly doubles runtime (26.8s -> 52.8s),
  confirming linear scaling in P_perm.
- Increasing N from 100 to 500 (5x) increases HSIC cost by ~50x,
  confirming quadratic scaling in N.

### 5.8 Full End-to-End Pipeline

| K | N | Total (ms) | ICM (ms) | CRC (ms) |
|---|---|-----------|----------|----------|
| 3 | 100 | 17.3 | 0.59 | 16.7 |
| 5 | 500 | 19.1 | 1.26 | 17.8 |
| 10 | 1,000 | 21.8 | 5.46 | 16.3 |
| 5 | 5,000 | 21.6 | 1.03 | 20.6 |

For the core ICM + CRC pipeline (excluding anti-spurious validation),
total time is consistently under 25 ms regardless of problem size, making
it suitable for real-time decision support.

---

## 6. Scalability Recommendations

### 6.1 Practical Limits

| Parameter | Practical Maximum | Rationale |
|-----------|-------------------|-----------|
| K (models) | ~50 with Hellinger, ~10 with MMD | ICM agreement is O(K^2 * cost_distance). With Hellinger, K=50 gives 1225 pairs at 0.02 ms each = ~25 ms. With MMD at N=500, K=10 gives 45 pairs at 21 ms each = ~945 ms. |
| N (samples for ICM) | Effectively unlimited for Hellinger | When using averaged distributions, ICM cost is independent of N. For MMD/empirical-W2 the kernel/transport costs limit N to ~1000. |
| N (samples for HSIC) | ~200-300 for interactive use | At N=200 with 1000 permutations, one HSIC test takes ~254 ms. With 10 model pairs: ~2.5 seconds. At N=500 this grows to ~50 seconds. |
| P_perm (HSIC permutations) | 500-1000 | Sufficient for p-value resolution at the 0.05 FDR level. More permutations give finer resolution but linear cost. |
| T (time steps) | 100,000+ | All detectors are O(T) with small constants. Even at T=10,000 they finish in under 10 ms. |

### 6.2 Bottleneck Identification

Ranked by computational impact:

1. **HSIC permutation test** -- O(K^2 * P * N^2). This is the dominant
   bottleneck in any pipeline that includes anti-spurious validation. The
   N^2 kernel computation combined with K^2 pairs and P permutations makes
   this the most expensive operation by orders of magnitude.

2. **MMD distance** -- O(N^2 * D) per pair. Quadratic in sample count.
   Only a bottleneck when MMD is chosen as the distance function.

3. **Wasserstein-2 Gaussian** -- O(D^3) per pair. Only a bottleneck in
   high-dimensional embedding spaces (D > 50).

4. **Ledoit-Wolf correlation** -- O(K^2 * N). Significant when computing
   Pi from raw residuals with large N, but typically much cheaper than HSIC.

5. **CRC threshold calibration** -- O(N log N) one-time cost. The 200-point
   sweep adds ~17 ms of constant overhead.

### 6.3 Optimization Strategies

#### 6.3.1 HSIC Acceleration (Highest Impact)

- **Nystr\u00f6m approximation:** Approximate the N x N kernel matrix using a
  rank-r approximation (r << N). Reduces kernel build from O(N^2) to O(N*r)
  and permutation cost from O(N^2) to O(N*r). Expected speedup: ~N/r.

- **Random Fourier Features (RFF):** Replace the RBF kernel with an explicit
  random feature map of dimension m << N. HSIC becomes O(N*m) per permutation
  instead of O(N^2). With m=100, this would give a ~5x speedup at N=500.

- **Gamma approximation for null:** Instead of permutation testing, use the
  analytical gamma distribution approximation to the HSIC null distribution.
  Eliminates the factor of P entirely, reducing HSIC to O(N^2) total. This
  is the single most impactful optimization.

- **Subsample for HSIC:** When N is large, subsample to N'=200-300 for the
  HSIC test. The test's statistical power is already adequate at these sizes.

#### 6.3.2 MMD Acceleration

- **Linear-time MMD:** Use the linear-time MMD estimator (block-based)
  instead of the quadratic U-statistic. Reduces per-pair cost from O(N^2*D)
  to O(N*D). Loses some statistical efficiency but is sufficient for the
  agreement component.

- **Random Fourier Features:** Same approach as for HSIC. Map to explicit
  feature space, compute MMD via mean embeddings in O(N*m*D).

#### 6.3.3 Caching

- **Distance matrix caching:** When computing ICM repeatedly (e.g., in
  meta-learner optimization), cache pairwise distances if model predictions
  have not changed. The meta-learner changes weights but not the raw
  predictions, so component values (A, D, U, C) can be cached per scenario
  and only the aggregation recomputed.

- **Kernel matrix caching for HSIC:** The centered kernel matrices Kx and Ky
  are computed once and reused across all permutations. This is already
  implemented. However, caching across multiple HSIC calls for the same
  model's residuals would save redundant kernel computations.

- **Isotonic model caching:** The fitted isotonic regression and conformal
  quantile can be cached and reused for all subsequent queries until the
  calibration set is updated.

#### 6.3.4 Parallelism

- **Pairwise distance computation:** The K(K-1)/2 pairwise distances are
  embarrassingly parallel. Using `concurrent.futures.ThreadPoolExecutor` or
  `joblib.Parallel` could provide near-linear speedup with the number of
  cores.

- **HSIC pair parallelism:** The K(K-1)/2 HSIC tests are independent and
  can be parallelized. With K=5 (10 pairs) on a 10-core machine, this could
  reduce wall-clock time by ~10x.

- **Permutation parallelism within HSIC:** The P permutations within each
  HSIC test can be batched using vectorized NumPy operations instead of a
  Python for-loop. Pre-generating all permutation indices and using advanced
  indexing could provide 2-5x speedup.

- **Meta-learner parallelism:** Scenario evaluations within each weight
  configuration are independent. Grid search points are also independent.

#### 6.3.5 Algorithmic Improvements

- **Incremental Ledoit-Wolf:** For streaming/rolling scenarios, maintain a
  running estimate of the correlation matrix instead of recomputing from
  scratch. Update cost: O(K^2) per new sample vs O(K^2 * N) for full
  recomputation.

- **Vectorized CUSUM/Page-Hinkley:** Replace the Python for-loop with NumPy
  vectorized operations where possible (the cumulative max/min structure makes
  full vectorization difficult, but partial vectorization of the accumulation
  step is feasible).

- **Early termination for anti-spurious:** If the first few HSIC tests
  already show significant dependence (rejecting independence), skip the
  remaining pairs. This provides average-case speedup when convergence is
  spurious.

### 6.4 Complexity Budget by Use Case

| Use Case | Components Used | Expected Time | Limiting Factor |
|----------|----------------|---------------|-----------------|
| Real-time scoring | ICM + CRC decision | < 10 ms | None (well within budget) |
| Batch evaluation (1000 samples) | ICM + CRC for all | < 1 second | CRC threshold sweep |
| Monitoring dashboard | ICM + Early warning | < 20 ms per step | Python loop overhead |
| Full validation | ICM + CRC + Anti-spurious | 1-60 seconds | HSIC permutation test |
| Weight optimization | Meta-learner grid search | 5-30 seconds | Number of scenarios x ICM calls |
| Full cross-validation | Meta-learner CV | 2-10 minutes | Nested optimization loops |

### 6.5 Recommended Defaults by Scale

**Small scale (K <= 5, N <= 500):** Use all defaults. Full pipeline including
anti-spurious completes in under 1 second.

**Medium scale (K <= 10, N <= 2000):** Reduce HSIC permutations to 500.
Consider subsampling to N'=300 for HSIC tests. Total anti-spurious: ~10-30
seconds.

**Large scale (K <= 20, N > 5000):** Switch to Hellinger distance (not MMD).
Subsample to N'=200 for HSIC. Use gamma approximation for HSIC null if
available. Use linear-time MMD if MMD distance is required. Parallelize
pairwise computations. Anti-spurious: ~5-15 seconds with optimizations.

**Streaming/real-time:** Pre-fit isotonic regression and cache. ICM + CRC
decision: < 5 ms per query. Run anti-spurious validation offline at periodic
intervals.

---

## 7. Asymptotic Complexity Summary

### Per-query (after setup)

```
T_query = O(K^2 * C)           [ICM with Hellinger]
        + O(log N_fit)          [CRC isotonic predict]
        + O(1)                  [Decision gate]
        = O(K^2 * C)
```

### One-time setup

```
T_setup = O(N_cal * log N_cal)  [Isotonic fit + conformal calibration]
```

### Periodic validation

```
T_validate = O(K^2 * P * N^2)  [HSIC anti-spurious]
           + O(P_ctrl * K^2 * N) [Negative controls]
           = O(K^2 * P * N^2)
```

### Meta-learning (offline)

```
T_meta = O(L * S * K^2 * C)    [Grid search: L points, S scenarios]
```
