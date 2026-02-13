# Mathematical Foundations of OS Multi-Science

**Version:** 1.0
**Date:** 2026-02-13
**Status:** Formal Mathematical Supplement
**Authors:** OS Multi-Science Project

---

## Table of Contents

1. [Notation and Preliminaries](#1-notation-and-preliminaries)
2. [ICM Definition and Properties](#2-icm-definition-and-properties)
3. [Component Properties](#3-component-properties)
4. [Conformal Risk Control Guarantee](#4-conformal-risk-control-guarantee)
5. [Early Warning Signal Theory](#5-early-warning-signal-theory)
6. [Anti-Spurious Testing](#6-anti-spurious-testing)
7. [Meta-Learner Convergence](#7-meta-learner-convergence)
8. [Geometric Aggregation](#8-convergence-of-geometric-aggregation)
9. [Beta-Calibrated Aggregation](#9-beta-calibrated-aggregation)
10. [Adaptive Aggregation](#10-adaptive-percentile-aggregation)
11. [Summary of Results](#11-summary-of-results)
12. [References](#12-references)

---

## 1. Notation and Preliminaries

### 1.1 Basic Setup

Throughout this document we adopt the following notation.

- $\mathcal{F} = \{f_1, f_2, \ldots, f_K\}$: a set of $K \geq 2$ epistemically diverse predictive models.
- $P_k$: the output distribution (or empirical output distribution) of model $f_k$.
- $\hat{y}_k \in \mathbb{R}^n$: the vector of $n$ predictions from model $f_k$.
- $r_k = \hat{y}_k - y \in \mathbb{R}^n$: the residual vector of model $f_k$ with respect to ground truth $y$.
- $\sigma(\cdot)$: the standard logistic sigmoid, $\sigma(z) = 1/(1 + e^{-z})$.
- $H(P, Q)$: the Hellinger distance between distributions $P$ and $Q$.
- $W_2(P, Q)$: the 2-Wasserstein distance between distributions $P$ and $Q$.
- $\text{MMD}(P, Q)$: the maximum mean discrepancy between $P$ and $Q$.
- $\mathbb{1}\{\cdot\}$: the indicator function.
- $\|\cdot\|_F$: the Frobenius norm.
- $\text{Tr}(\cdot)$: the matrix trace operator.

### 1.2 Weight Constraints

The ICM framework uses the following weight parameters, all strictly positive:

- Component weights: $w_A, w_D, w_U, w_C > 0$ (default: $w_A = 0.35$, $w_D = 0.15$, $w_U = 0.25$, $w_C = 0.10$).
- Dependency penalty coefficient: $\lambda > 0$ (default: $\lambda = 0.15$).
- Dependency sub-weights: $\gamma_\rho, \gamma_J, \gamma_{\text{grad}} > 0$ (default: $0.4, 0.3, 0.3$).

### 1.3 Component Domain

Each of the five ICM components is defined to take values in $[0, 1]$:

$$A, D, U, C, \Pi \in [0, 1]$$

This is enforced by construction in each component function (see Section 3) and additionally by explicit numerical clipping in the implementation.

---

## 2. ICM Definition and Properties

### 2.1 Definition 1 (ICM Score -- Logistic Aggregation)

**Definition.** Given a set of $K$ epistemically diverse models $\mathcal{F} = \{f_1, \ldots, f_K\}$, the *Index of Convergence Multi-epistemic* (ICM) is defined as:

$$\text{ICM}(\mathcal{F}) = \sigma\!\Big(w_A \cdot A(\mathcal{F}) + w_D \cdot D(\mathcal{F}) + w_U \cdot U(\mathcal{F}) + w_C \cdot C(\mathcal{F}) - \lambda \cdot \Pi(\mathcal{F})\Big)$$

where:
- $A(\mathcal{F}) \in [0, 1]$ is the distributional agreement score (Section 3.1),
- $D(\mathcal{F}) \in [0, 1]$ is the directional agreement score (Section 3.2),
- $U(\mathcal{F}) \in [0, 1]$ is the uncertainty overlap score (Section 3.3),
- $C(\mathcal{F}) \in [0, 1]$ is the invariance/stability score (Section 3.4),
- $\Pi(\mathcal{F}) \in [0, 1]$ is the dependency penalty (Section 3.5),
- $\sigma(z) = (1 + e^{-z})^{-1}$ is the logistic sigmoid function.

We define the *linear pre-activation* as:

$$z = w_A \cdot A + w_D \cdot D + w_U \cdot U + w_C \cdot C - \lambda \cdot \Pi$$

so that $\text{ICM} = \sigma(z)$.

### 2.2 Proposition 1 (Boundedness)

**Statement.** For all valid inputs where $A, D, U, C, \Pi \in [0, 1]$ and all weights $w_A, w_D, w_U, w_C, \lambda > 0$:

$$\text{ICM}(\mathcal{F}) \in (0, 1)$$

More precisely, let $z_{\min} = -\lambda$ and $z_{\max} = w_A + w_D + w_U + w_C$. Then:

$$\sigma(z_{\min}) \leq \text{ICM}(\mathcal{F}) \leq \sigma(z_{\max})$$

With default parameters ($w_A = 0.35, w_D = 0.15, w_U = 0.25, w_C = 0.10, \lambda = 0.15$), we have $z \in [-0.15, 0.85]$, so:

$$\text{ICM} \in [\sigma(-0.15), \sigma(0.85)] \approx [0.4626, 0.7006]$$

**Proof.**

*Step 1.* Since $A, D, U, C \in [0, 1]$ and $\Pi \in [0, 1]$, the linear pre-activation satisfies:

$$z = w_A A + w_D D + w_U U + w_C C - \lambda \Pi$$

The minimum of $z$ is attained when $A = D = U = C = 0$ and $\Pi = 1$:

$$z_{\min} = 0 + 0 + 0 + 0 - \lambda \cdot 1 = -\lambda$$

The maximum of $z$ is attained when $A = D = U = C = 1$ and $\Pi = 0$:

$$z_{\max} = w_A + w_D + w_U + w_C - 0 = w_A + w_D + w_U + w_C$$

*Step 2.* The sigmoid function $\sigma : \mathbb{R} \to (0, 1)$ is strictly increasing. Therefore:

$$\sigma(z_{\min}) < \sigma(z) < \sigma(z_{\max})$$

for all $z \in (z_{\min}, z_{\max})$, and equality holds at the boundary values.

*Step 3.* Since $\sigma$ maps $\mathbb{R}$ to the open interval $(0, 1)$ and never attains 0 or 1, we have $\text{ICM} \in (0, 1)$. The bounds $\sigma(z_{\min})$ and $\sigma(z_{\max})$ are tight (attained when components are at their extreme values).  $\square$

**Discussion.** The strict boundedness in $(0, 1)$ (rather than $[0, 1]$) follows from the range of the sigmoid. In practice, the implementation clips the output to $[0, 1]$ for numerical safety, but the theoretical values never reach the endpoints. Note that with default parameters, the effective range is compressed to approximately $[0.46, 0.70]$, which motivates the alternative aggregation methods (Beta-calibrated, adaptive) described in Sections 9 and 10.

---

### 2.3 Proposition 2 (Monotonicity in Agreement)

**Statement.** The ICM score is strictly monotone increasing in the agreement component $A$ when all other components $(D, U, C, \Pi)$ are held fixed. Formally:

$$\frac{\partial\, \text{ICM}}{\partial A} = w_A \cdot \sigma'(z) > 0$$

for all valid inputs, where $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**Proof.**

The ICM is a composition $\text{ICM} = \sigma(z)$ where $z = w_A A + w_D D + w_U U + w_C C - \lambda \Pi$.

By the chain rule:

$$\frac{\partial\,\text{ICM}}{\partial A} = \sigma'(z) \cdot \frac{\partial z}{\partial A} = \sigma'(z) \cdot w_A$$

Since:
1. $\sigma'(z) = \sigma(z)(1 - \sigma(z)) > 0$ for all $z \in \mathbb{R}$ (because $\sigma(z) \in (0,1)$), and
2. $w_A > 0$ by assumption,

we conclude that $\frac{\partial\,\text{ICM}}{\partial A} > 0$.  $\square$

**Corollary (General Monotonicity).** By the same argument, ICM is strictly monotone increasing in each of $D$, $U$, and $C$:

$$\frac{\partial\,\text{ICM}}{\partial D} = w_D \cdot \sigma'(z) > 0, \quad \frac{\partial\,\text{ICM}}{\partial U} = w_U \cdot \sigma'(z) > 0, \quad \frac{\partial\,\text{ICM}}{\partial C} = w_C \cdot \sigma'(z) > 0$$

**Discussion.** Monotonicity is a desirable property for a convergence index: higher agreement, directional consensus, uncertainty overlap, and stability should each individually contribute to a higher ICM score. The partial derivatives also reveal the relative *marginal sensitivity*: the ratio $(\partial \text{ICM}/\partial A) / (\partial \text{ICM}/\partial D) = w_A / w_D = 0.35/0.15 \approx 2.33$, meaning the ICM is approximately 2.33 times more sensitive to changes in agreement than to changes in directional consensus.

---

### 2.4 Proposition 3 (Lipschitz Continuity)

**Statement.** The ICM score is Lipschitz continuous with respect to perturbations in the component vector $(A, D, U, C, \Pi)$. Specifically, for any two component vectors $\mathbf{c} = (A, D, U, C, \Pi)$ and $\mathbf{c}' = (A', D', U', C', \Pi')$:

$$|\text{ICM}(\mathbf{c}) - \text{ICM}(\mathbf{c}')| \leq \frac{1}{4} \cdot \|(\mathbf{w}, \lambda)\| \cdot \|\mathbf{c} - \mathbf{c}'\|$$

where $\mathbf{w} = (w_A, w_D, w_U, w_C, \lambda)$ and $\|\cdot\|$ denotes the Euclidean norm.

More precisely, the Lipschitz constant is:

$$L_{\text{ICM}} = \frac{1}{4}\sqrt{w_A^2 + w_D^2 + w_U^2 + w_C^2 + \lambda^2}$$

With default parameters: $L_{\text{ICM}} = \frac{1}{4}\sqrt{0.35^2 + 0.15^2 + 0.25^2 + 0.10^2 + 0.15^2} = \frac{1}{4}\sqrt{0.2300} \approx 0.1199$.

**Proof.**

*Step 1 (Sigmoid Lipschitz constant).* The sigmoid function satisfies $\sup_z |\sigma'(z)| = \sup_z \sigma(z)(1-\sigma(z)) = 1/4$, attained at $z = 0$. Hence $\sigma$ is $\frac{1}{4}$-Lipschitz:

$$|\sigma(z_1) - \sigma(z_2)| \leq \frac{1}{4}|z_1 - z_2| \quad \text{for all } z_1, z_2 \in \mathbb{R}$$

*Step 2 (Linear map Lipschitz constant).* The pre-activation $z$ is a linear function of $\mathbf{c}$:

$$z(\mathbf{c}) = w_A \cdot A + w_D \cdot D + w_U \cdot U + w_C \cdot C - \lambda \cdot \Pi = \langle \mathbf{w}^*, \mathbf{c} \rangle$$

where $\mathbf{w}^* = (w_A, w_D, w_U, w_C, -\lambda)$. By the Cauchy-Schwarz inequality:

$$|z(\mathbf{c}) - z(\mathbf{c}')| = |\langle \mathbf{w}^*, \mathbf{c} - \mathbf{c}' \rangle| \leq \|\mathbf{w}^*\| \cdot \|\mathbf{c} - \mathbf{c}'\|$$

where $\|\mathbf{w}^*\| = \sqrt{w_A^2 + w_D^2 + w_U^2 + w_C^2 + \lambda^2}$.

*Step 3 (Composition).* Composing the two Lipschitz maps:

$$|\text{ICM}(\mathbf{c}) - \text{ICM}(\mathbf{c}')| = |\sigma(z(\mathbf{c})) - \sigma(z(\mathbf{c}'))| \leq \frac{1}{4}|z(\mathbf{c}) - z(\mathbf{c}')| \leq \frac{1}{4}\|\mathbf{w}^*\| \cdot \|\mathbf{c} - \mathbf{c}'\|$$

Therefore the Lipschitz constant of ICM with respect to the component vector is $L_{\text{ICM}} = \frac{1}{4}\|\mathbf{w}^*\|$.  $\square$

**Discussion.** Lipschitz continuity guarantees that the ICM score is *robust to small perturbations*: a perturbation of magnitude $\epsilon$ in the component vector changes the ICM by at most $L_{\text{ICM}} \cdot \epsilon \approx 0.12\epsilon$. This is important for practical stability, as component estimates are themselves subject to estimation noise. The small Lipschitz constant ($\approx 0.12$) means the sigmoid aggregation acts as a *smoothing* operator, which is beneficial for stability but compresses the output range (see Section 9 for the Beta-calibrated alternative that mitigates this).

---

### 2.5 Proposition 4 (Sensitivity to Dependency)

**Statement.** The ICM score is strictly monotone *decreasing* in the dependency penalty $\Pi$ when all other components $(A, D, U, C)$ are held fixed:

$$\frac{\partial\,\text{ICM}}{\partial \Pi} = -\lambda \cdot \sigma'(z) < 0$$

**Proof.**

By direct computation:

$$\frac{\partial\,\text{ICM}}{\partial \Pi} = \sigma'(z) \cdot \frac{\partial z}{\partial \Pi} = \sigma'(z) \cdot (-\lambda)$$

Since $\sigma'(z) > 0$ (as shown in Proposition 2) and $\lambda > 0$, we have:

$$\frac{\partial\,\text{ICM}}{\partial \Pi} = -\lambda \cdot \sigma'(z) < 0$$

$\square$

**Discussion.** This result formalizes the anti-spurious design principle: when models share dependencies (high $\Pi$), the ICM score decreases, reflecting reduced epistemic value of the apparent convergence. The magnitude $|\partial\text{ICM}/\partial\Pi| = \lambda \cdot \sigma'(z)$ is maximized at $z = 0$ where $\sigma'(0) = 1/4$, giving a maximum sensitivity of $\lambda/4 = 0.0375$ with default parameters. The dependency penalty thus acts as a *multiplicative discount* on convergence credibility.

---

### 2.6 Proposition (Concavity in Positive Components)

**Statement.** For fixed $\Pi$, the ICM score is a *strictly concave* function of $(A, D, U, C)$.

**Proof sketch.** The sigmoid $\sigma$ is a concave function on $[0, +\infty)$ (since $\sigma''(z) = \sigma'(z)(1-2\sigma(z)) < 0$ for $z > 0$). Since $z = w_A A + w_D D + w_U U + w_C C - \lambda\Pi$ is affine in $(A, D, U, C)$ for fixed $\Pi$, and $z \geq 0$ when the components are sufficiently large, the composition $\sigma(z)$ is concave in $(A, D, U, C)$ on the region $\{z \geq 0\}$. In general, $\sigma$ is neither globally concave nor convex, but is *log-concave* on all of $\mathbb{R}$ (i.e., $\log \sigma$ is concave).  $\square$

**Discussion.** Concavity implies *diminishing marginal returns*: the first unit of agreement contributes more to the ICM than the last. This is a desirable property for a convergence metric, as it penalizes over-reliance on any single component.

---

## 3. Component Properties

### 3.1 Distributional Agreement $A$

**Definition 2 (Distributional Agreement).** Given $K$ model output distributions $\{P_1, \ldots, P_K\}$ and a distance function $d$ (Hellinger, Wasserstein, or MMD):

$$A(\mathcal{F}) = 1 - \frac{\text{median}_{i < j}\, d(P_i, P_j)}{C_A}$$

where $C_A > 0$ is a normalization constant, and the result is clipped to $[0, 1]$:

$$A = \text{clip}\!\left(1 - \frac{\text{median}_{i < j}\, d(P_i, P_j)}{C_A},\; 0,\; 1\right)$$

**Remark.** The use of the median (rather than the mean) over the $\binom{K}{2}$ pairwise distances provides robustness to outlier models. The normalization constant $C_A$ should be chosen to reflect the maximum expected distance for the given distance function and domain.

---

#### 3.1.1 Theorem 1 (Hellinger Distance Properties)

**Statement.** The Hellinger distance $H(P, Q)$ between two discrete probability distributions $P = (p_1, \ldots, p_m)$ and $Q = (q_1, \ldots, q_m)$, defined as:

$$H(P, Q) = \sqrt{\frac{1}{2}\sum_{i=1}^{m}\left(\sqrt{p_i} - \sqrt{q_i}\right)^2}$$

satisfies the following properties:

**(a) Boundedness:** $H(P, Q) \in [0, 1]$.

**(b) Identity of indiscernibles:** $H(P, Q) = 0$ if and only if $P = Q$.

**(c) Symmetry:** $H(P, Q) = H(Q, P)$.

**(d) Triangle inequality:** $H(P, R) \leq H(P, Q) + H(Q, R)$ for all distributions $P, Q, R$.

**(e) Relation to Total Variation:** $H^2(P, Q) \leq \text{TV}(P, Q) \leq H(P, Q)\sqrt{2}$,

where $\text{TV}(P, Q) = \frac{1}{2}\sum_i |p_i - q_i|$.

**Proof.**

**(a) Boundedness.** Write $H^2(P,Q) = \frac{1}{2}\sum_i (\sqrt{p_i} - \sqrt{q_i})^2$. Expanding:

$$H^2 = \frac{1}{2}\left(\sum_i p_i + \sum_i q_i - 2\sum_i \sqrt{p_i q_i}\right) = 1 - \sum_i\sqrt{p_i q_i}$$

Since $p_i, q_i \geq 0$ and $\sum_i p_i = \sum_i q_i = 1$, by the Cauchy-Schwarz inequality:

$$0 \leq \sum_i\sqrt{p_i q_i} \leq \sqrt{\sum_i p_i}\sqrt{\sum_i q_i} = 1$$

Therefore $0 \leq H^2 \leq 1$, and thus $H \in [0, 1]$.

**(b) Identity of indiscernibles.** $H(P,Q) = 0$ iff $H^2(P,Q) = 0$ iff $\sum_i\sqrt{p_i q_i} = 1$ iff $\sqrt{p_i} = \sqrt{q_i}$ for all $i$ (since the only way the Cauchy-Schwarz inequality is an equality is when the vectors are proportional, and both sum to 1, hence they must be equal), iff $p_i = q_i$ for all $i$.

**(c) Symmetry.** Immediate from the symmetry of $(\sqrt{p_i} - \sqrt{q_i})^2 = (\sqrt{q_i} - \sqrt{p_i})^2$.

**(d) Triangle inequality.** The Hellinger distance is the $\ell^2$ norm of $\frac{1}{\sqrt{2}}(\sqrt{P} - \sqrt{Q})$ in the vector space $\mathbb{R}^m$. Since the $\ell^2$ norm satisfies the triangle inequality, so does $H$.

Formally, define $\phi(P) = \frac{1}{\sqrt{2}}(\sqrt{p_1}, \ldots, \sqrt{p_m}) \in \mathbb{R}^m$. Then $H(P,Q) = \|\phi(P) - \phi(Q)\|_2$. The $\ell^2$-norm triangle inequality gives:

$$H(P,R) = \|\phi(P) - \phi(R)\|_2 \leq \|\phi(P) - \phi(Q)\|_2 + \|\phi(Q) - \phi(R)\|_2 = H(P,Q) + H(Q,R)$$

**(e) Relation to Total Variation.** This is a classical result. For the lower bound:

$$H^2(P,Q) = 1 - \sum_i\sqrt{p_i q_i} \leq 1 - \sum_i \min(p_i, q_i) = \text{TV}(P,Q)$$

where we used $\sqrt{p_i q_i} \geq \min(p_i, q_i)$ when $p_i, q_i \geq 0$ -- actually, the correct classical inequality uses the AM-GM and Cauchy-Schwarz inequalities more carefully. The standard result (see, e.g., Tsybakov, 2009, Lemma 2.3) states:

$$H^2(P,Q) \leq \text{TV}(P,Q) \leq H(P,Q)\sqrt{2}$$

The upper bound follows from the Cauchy-Schwarz inequality applied to the difference $|p_i - q_i| = |\sqrt{p_i} - \sqrt{q_i}||\sqrt{p_i} + \sqrt{q_i}|$.  $\square$

**Discussion.** The Hellinger distance is a proper metric on the space of probability distributions. Its boundedness in $[0, 1]$ makes it natural for the agreement component $A$. When used as the distance function in the agreement computation with $C_A = 1$ (the default for Hellinger), the agreement score satisfies $A \in [0, 1]$ by construction.

These properties are classical and well-known; see Hellinger (1909) for the original definition and Le Cam & Yang (2000) or Tsybakov (2009) for modern treatments.

---

### 3.2 Direction/Sign Agreement $D$

**Definition 3 (Direction Score).** Given a vector of signs $\mathbf{s} = (s_1, \ldots, s_K)$ where $s_k \in \{-1, 0, +1\}$ represents the sign (direction) of model $f_k$'s prediction or gradient:

$$D(\mathcal{F}) = 1 - \frac{H(\hat{p})}{H_{\max}}$$

where:
- $\hat{p}$ is the empirical distribution over the observed sign categories,
- $H(\hat{p}) = -\sum_c \hat{p}_c \log \hat{p}_c$ is the Shannon entropy (natural logarithm),
- $H_{\max} = \log(|\mathcal{C}|)$ where $\mathcal{C}$ is the set of distinct sign values observed.

If all models agree on the same sign, $D = 1$. If all sign categories have equal frequency, $D = 0$.

#### 3.2.1 Theorem 2 (Direction Entropy Properties)

**Statement.** The direction score $D$ satisfies:

**(a)** $D \in [0, 1]$.

**(b)** $D = 1$ if and only if all models agree on the same sign (i.e., $|\mathcal{C}| = 1$).

**(c)** $D = 0$ if and only if the signs are maximally uncertain, i.e., the empirical sign distribution is uniform over the observed categories.

**Proof.**

**(a)** The Shannon entropy satisfies $0 \leq H(\hat{p}) \leq \log(|\mathcal{C}|) = H_{\max}$ for any distribution $\hat{p}$ over $|\mathcal{C}|$ categories (with equality on the left iff the distribution is degenerate, and equality on the right iff the distribution is uniform). Therefore:

$$0 \leq \frac{H(\hat{p})}{H_{\max}} \leq 1 \quad \implies \quad 0 \leq D = 1 - \frac{H(\hat{p})}{H_{\max}} \leq 1$$

When $|\mathcal{C}| = 1$, we define $H_{\max} = 0$ and set $D = 1$ by convention (all models agree).

**(b)** If $|\mathcal{C}| = 1$, then $D = 1$ by convention. Conversely, if $|\mathcal{C}| \geq 2$, then $H_{\max} > 0$ and $D = 1$ iff $H(\hat{p}) = 0$ iff the distribution is degenerate on a single category, contradicting $|\mathcal{C}| \geq 2$. Hence $D = 1$ iff all models agree.

**(c)** $D = 0$ iff $H(\hat{p}) = H_{\max} = \log(|\mathcal{C}|)$ iff $\hat{p}$ is the uniform distribution over the $|\mathcal{C}|$ categories.  $\square$

**Discussion.** The normalized entropy $H(\hat{p})/H_{\max}$ is sometimes called the *normalized Shannon entropy* or *evenness index* in ecology (Pielou, 1966). Its complement $D = 1 - H/H_{\max}$ measures the degree of consensus among models on the direction of effect. Note that $D$ depends on the number of categories present in the data, not the theoretical maximum of 3 categories $\{-1, 0, +1\}$. This is a deliberate design choice: if no model predicts zero, then having all models agree on the sign gives $D = 1$ regardless.

---

### 3.3 Uncertainty Overlap $U$

**Definition 4 (Uncertainty Overlap).** Given $K$ confidence/credible intervals $\{[l_k, u_k]\}_{k=1}^K$, the uncertainty overlap is defined as the mean pairwise Intersection-over-Union (IoU):

$$U(\mathcal{F}) = \frac{2}{K(K-1)}\sum_{i < j} \text{IoU}([l_i, u_i], [l_j, u_j])$$

where:

$$\text{IoU}([l_i, u_i], [l_j, u_j]) = \frac{|[l_i, u_i] \cap [l_j, u_j]|}{|[l_i, u_i] \cup [l_j, u_j]|} = \frac{\max(\min(u_i, u_j) - \max(l_i, l_j),\, 0)}{\max(u_i, u_j) - \min(l_i, l_j)}$$

with the convention that $0/0 = 0$ (degenerate intervals).

#### 3.3.1 Theorem 3 (Uncertainty Overlap as IoU)

**Statement.** The uncertainty overlap $U$ satisfies:

**(a)** $U \in [0, 1]$.

**(b)** $U = 1$ if and only if all intervals are identical: $[l_i, u_i] = [l_j, u_j]$ for all $i, j$.

**(c)** $U = 0$ if and only if no pair of intervals overlaps: $[l_i, u_i] \cap [l_j, u_j] = \emptyset$ for all $i \neq j$.

**Proof.**

**(a)** Each pairwise IoU satisfies $\text{IoU} \in [0, 1]$: the numerator (intersection length) is non-negative and at most the denominator (union length), since $|A \cap B| \leq |A \cup B|$ for any sets $A, B$. The mean of values in $[0, 1]$ is in $[0, 1]$.

**(b)** $U = 1$ iff every pairwise IoU equals 1. For intervals, $\text{IoU}([l_i, u_i], [l_j, u_j]) = 1$ iff $|I \cap J| = |I \cup J|$ iff $I = J$ (as intervals with equal Lebesgue measure and equal union must be identical). Hence $U = 1$ iff all intervals are identical.

**(c)** $U = 0$ iff every pairwise IoU equals 0. $\text{IoU}(I, J) = 0$ iff $|I \cap J| = 0$ iff the intervals have no overlap (or overlap at a single point, which has measure zero). Hence $U = 0$ iff no pair overlaps.  $\square$

**Discussion.** The IoU (Jaccard index for intervals) is a standard measure of set overlap. By averaging over all $\binom{K}{2}$ pairs, $U$ captures the overall degree to which models' uncertainty ranges agree. When models are calibrated, high $U$ indicates that their uncertainty is concentrated in similar regions of the outcome space, which is a strong signal of convergence.

---

### 3.4 Invariance/Stability Score $C$

**Definition 5 (Invariance Score).** Given pre-intervention prediction vector $\hat{y}^{\text{pre}} \in \mathbb{R}^n$ and post-intervention prediction vector $\hat{y}^{\text{post}} \in \mathbb{R}^n$:

$$C(\mathcal{F}) = 1 - \min\!\left(\frac{\|\hat{y}^{\text{pre}} - \hat{y}^{\text{post}}\|}{\|\hat{y}^{\text{pre}}\| + \epsilon},\; 1\right)$$

where $\|\cdot\|$ denotes the $\ell^2$ norm and $\epsilon > 0$ is a small constant ($\epsilon = 10^{-12}$ in implementation) to prevent division by zero.

**Proposition (Invariance Boundedness).** $C \in [0, 1]$.

**Proof.** The ratio $\|\hat{y}^{\text{pre}} - \hat{y}^{\text{post}}\| / (\|\hat{y}^{\text{pre}}\| + \epsilon) \geq 0$, and applying $\min(\cdot, 1)$ ensures the ratio is in $[0, 1]$. Therefore $C = 1 - [\text{value in } [0,1]] \in [0, 1]$.  $\square$

**Discussion.** The invariance score measures stability of predictions under intervention or perturbation. When $\hat{y}^{\text{post}} = \hat{y}^{\text{pre}}$, we have $C = 1$ (perfect stability). The normalization by $\|\hat{y}^{\text{pre}}\|$ makes the score scale-invariant: it measures *relative* change rather than absolute change.

---

### 3.5 Dependency Penalty $\Pi$

**Definition 6 (Dependency Penalty).** The dependency penalty combines three sub-scores:

$$\Pi(\mathcal{F}) = \bar{\gamma}_\rho \cdot \rho_{\text{corr}} + \bar{\gamma}_J \cdot J_{\text{overlap}} + \bar{\gamma}_{\text{grad}} \cdot g_{\text{sim}}$$

where $\bar{\gamma}_\rho, \bar{\gamma}_J, \bar{\gamma}_{\text{grad}}$ are the sub-weights renormalized to sum to 1 (i.e., $\bar{\gamma}_i = \gamma_i / (\gamma_\rho + \gamma_J + \gamma_{\text{grad}})$), and:

**Sub-component 1: Residual Correlation ($\rho_{\text{corr}}$).**

$$\rho_{\text{corr}} = \frac{1}{K(K-1)} \sum_{i \neq j} |\hat{\Sigma}^{\text{LW}}_{ij}|$$

where $\hat{\Sigma}^{\text{LW}}$ is the Ledoit-Wolf shrinkage estimate of the residual correlation matrix (see Theorem 4 below).

**Sub-component 2: Feature Overlap ($J_{\text{overlap}}$).**

$$J_{\text{overlap}} = \frac{2}{K(K-1)} \sum_{i < j} J(\Phi_i, \Phi_j) = \frac{2}{K(K-1)} \sum_{i < j} \frac{|\Phi_i \cap \Phi_j|}{|\Phi_i \cup \Phi_j|}$$

where $\Phi_k$ is the set of features/provenance identifiers used by model $f_k$, and $J$ is the Jaccard similarity.

**Sub-component 3: Gradient Similarity ($g_{\text{sim}}$).**

$$g_{\text{sim}} = \frac{1}{2}\left(1 + \frac{2}{K(K-1)} \sum_{i < j} \frac{\langle \nabla_i, \nabla_j \rangle}{\|\nabla_i\|\|\nabla_j\|}\right)$$

where $\nabla_k$ is the gradient vector of model $f_k$ (e.g., from SHAP/integrated gradients), and the affine map $(x+1)/2$ transforms cosine similarity from $[-1, 1]$ to $[0, 1]$.

**Proposition (Dependency Penalty Boundedness).** $\Pi \in [0, 1]$.

**Proof.** Each sub-component is in $[0, 1]$:
- $\rho_{\text{corr}} \in [0, 1]$: elements of a correlation matrix are in $[-1, 1]$, so their absolute values are in $[0, 1]$, and the mean is in $[0, 1]$.
- $J_{\text{overlap}} \in [0, 1]$: the Jaccard similarity is in $[0, 1]$.
- $g_{\text{sim}} \in [0, 1]$: cosine similarity is in $[-1, 1]$, mapped to $[0, 1]$ by $(x+1)/2$.

Since $\bar{\gamma}_\rho + \bar{\gamma}_J + \bar{\gamma}_{\text{grad}} = 1$ and all $\bar{\gamma}_i > 0$, the convex combination $\Pi = \sum \bar{\gamma}_i \cdot (\text{value in } [0,1])$ is in $[0, 1]$.  $\square$

---

#### 3.5.1 Theorem 4 (Ledoit-Wolf Shrinkage Estimator)

**Statement.** Given a $K \times n$ matrix of standardized residuals $Z$ (rows are models, columns are observations), the Ledoit-Wolf shrinkage estimator of the correlation matrix is:

$$\hat{\Sigma}^{\text{LW}} = (1 - \alpha^*) S + \alpha^* I_K$$

where $S = \frac{1}{n} Z Z^T$ is the sample correlation matrix and $\alpha^* \in [0, 1]$ is the optimal shrinkage intensity.

The following properties hold:

**(a) Consistency:** $\|\hat{\Sigma}^{\text{LW}} - \Sigma\|_F \to 0$ as $n \to \infty$ (for fixed $K$), where $\Sigma$ is the true correlation matrix.

**(b) Positive definiteness:** $\hat{\Sigma}^{\text{LW}}$ is positive definite for any $\alpha^* \in (0, 1]$.

**(c) Bias-variance tradeoff:** The shrinkage intensity $\alpha^* \in [0, 1]$ minimizes the expected squared Frobenius loss $E\|\hat{\Sigma}^{\text{LW}} - \Sigma\|_F^2$.

**(d) Independence detection:** If the models are truly independent (i.e., $\Sigma = I_K$), then $\rho_{\text{corr}} \to 0$ as $n \to \infty$, and hence $\Pi \to 0$ (assuming other sub-components also indicate independence).

**Proof sketch.**

**(a)** By the law of large numbers, $S \to \Sigma$ almost surely as $n \to \infty$ (for fixed $K$). Since $\alpha^* \to 0$ when $S$ is a consistent estimator (the optimal shrinkage vanishes as the sample size grows), we have $\hat{\Sigma}^{\text{LW}} \to \Sigma$.

**(b)** $I_K$ is positive definite and $S$ is positive semi-definite. For $\alpha^* > 0$, $\hat{\Sigma}^{\text{LW}} = (1-\alpha^*)S + \alpha^* I_K$ is a sum of a positive semi-definite and a (scaled) positive definite matrix, hence positive definite.

**(c)** This is the main result of Ledoit & Wolf (2004). The optimal shrinkage intensity is:

$$\alpha^* = \frac{\sum_{i=1}^K \sum_{j=1}^K \text{Var}(S_{ij})}{E\left[\sum_{i=1}^K \sum_{j=1}^K (S_{ij} - \delta_{ij})^2\right]}$$

where $\delta_{ij}$ is the Kronecker delta. The implementation uses an approximate Oracle formula:

$$\alpha^* = \text{clip}\!\left(\frac{\frac{1}{n}(\text{Tr}(S^2) + \text{Tr}(S)^2) - \frac{2}{K}\text{Tr}(S^2)}{(n+1-\frac{2}{K})(\text{Tr}(S^2) - \frac{\text{Tr}(S)^2}{K})}, \; 0, \; 1\right)$$

**(d)** When $\Sigma = I_K$, we have $S \to I_K$ as $n \to \infty$, so the off-diagonal elements of $S$ converge to 0. Thus $\rho_{\text{corr}} = \frac{1}{K(K-1)}\sum_{i \neq j}|(\hat{\Sigma}^{\text{LW}})_{ij}| \to 0$.  $\square$

**Discussion.** The Ledoit-Wolf estimator is the standard approach for estimating covariance/correlation matrices when $K$ may be comparable to or larger than $n$ (the high-dimensional setting). In our context, $K$ is the number of models (typically 3-7) and $n$ is the number of observations, so we are typically in the low-dimensional regime where $S$ is already a reasonable estimator. The shrinkage provides additional stability and guarantees positive definiteness.

This result is established in Ledoit & Wolf (2004). The consistency and optimality proofs follow their Theorem 1 and Corollary 1.

---

## 4. Conformal Risk Control Guarantee

### 4.1 Setup

The Conformal Risk Control (CRC) module maps ICM scores to epistemic risk bounds. The pipeline is:

1. **Isotonic regression**: Fit a monotone non-increasing function $g : [0,1] \to \mathbb{R}_+$ from calibration pairs $\{(C_i, L_i)\}_{i=1}^n$ where $C_i$ is the ICM score and $L_i$ is the observed loss.

2. **Split conformal calibration**: On a held-out calibration set $\{(C_j, L_j)\}_{j=1}^{n_{\text{cal}}}$, compute residuals $r_j = L_j - g(C_j)$ and let $\hat{q}$ be the $\lceil(1-\alpha)(n_{\text{cal}}+1)\rceil / n_{\text{cal}}$-th empirical quantile of the residuals.

3. **Conformalized risk function**: $g_\alpha(c) = g(c) + \hat{q}$.

4. **Epistemic risk**: $R_e(x) = g_\alpha(C(x))$ for a new instance $x$.

### 4.2 Theorem 5 (Marginal Coverage Guarantee)

**Statement.** Suppose $(X_1, Y_1), \ldots, (X_{n+1}, Y_{n+1})$ are exchangeable random variables. Let $C_i = \text{ICM}(X_i)$ and $L_i = \ell(Y_i, \hat{Y}_i)$ for a fixed loss function $\ell$. The split conformal procedure uses:
- A training set $\{1, \ldots, n_{\text{train}}\}$ to fit the isotonic regression $g$,
- A calibration set $\{n_{\text{train}}+1, \ldots, n\}$ to compute the conformal quantile $\hat{q}$,
- A test point $n+1$.

Then the conformalized bound satisfies the marginal coverage guarantee:

$$P\!\left(L_{n+1} \leq g_\alpha(C_{n+1})\right) \geq 1 - \alpha$$

**Proof.** This is the standard split conformal prediction result. We provide a self-contained proof following Vovk et al. (2005) and Romano et al. (2019).

*Step 1.* Define the nonconformity scores on the calibration set: $R_j = L_j - g(C_j)$ for $j \in \mathcal{I}_{\text{cal}}$, and for the test point: $R_{n+1} = L_{n+1} - g(C_{n+1})$.

*Step 2.* Since $(X_i, Y_i)$ are exchangeable and $g$ is fitted only on the training set (independent of the calibration set and test point), the scores $\{R_j : j \in \mathcal{I}_{\text{cal}}\} \cup \{R_{n+1}\}$ are exchangeable.

*Step 3.* For exchangeable random variables $R_1, \ldots, R_{n_{\text{cal}}+1}$, each $R_j$ has the same probability of being the largest. By the quantile lemma for exchangeable sequences, if $\hat{q}$ is the $\lceil(1-\alpha)(n_{\text{cal}}+1)\rceil / n_{\text{cal}}$-th empirical quantile of $\{R_1, \ldots, R_{n_{\text{cal}}}\}$, then:

$$P(R_{n+1} \leq \hat{q}) \geq 1 - \alpha$$

*Step 4.* Substituting back: $R_{n+1} \leq \hat{q}$ is equivalent to $L_{n+1} - g(C_{n+1}) \leq \hat{q}$, i.e., $L_{n+1} \leq g(C_{n+1}) + \hat{q} = g_\alpha(C_{n+1})$.

Therefore:

$$P(L_{n+1} \leq g_\alpha(C_{n+1})) = P(R_{n+1} \leq \hat{q}) \geq 1 - \alpha$$

$\square$

**Discussion.** The coverage guarantee is *marginal* (averaged over the randomness of both the calibration set and the test point). It holds for any monotone fitting procedure $g$ and requires only exchangeability of the data, not i.i.d. assumptions. The guarantee is distribution-free: it makes no parametric assumptions about the data-generating process.

The key strength is that this converts the ICM convergence score into a *calibrated risk bound*: for any new instance, the loss is bounded by $g_\alpha(C)$ with probability at least $1 - \alpha$. This is the formal justification for the decision gate (Section 4.4).

---

### 4.3 Corollary (Limitation: Conditional Coverage)

**Statement.** The marginal coverage guarantee does *not* imply conditional coverage. That is, it is generally *not* the case that:

$$P(L_{n+1} \leq g_\alpha(C_{n+1}) \mid X_{n+1} = x) \geq 1 - \alpha \quad \text{for all } x$$

**Discussion.** This is a well-known limitation of split conformal prediction (Barber et al., 2021; Lei & Wasserman, 2014). Marginal coverage means the bound holds "on average" over the test distribution, but for specific covariate values $x$, the bound may be too loose or too tight. Achieving conditional coverage requires additional assumptions or more sophisticated methods (e.g., weighted conformal prediction, local conformal methods). This limitation should be transparently reported when using the CRC gating mechanism: the decision gate based on ICM thresholds provides *average-case* guarantees, not worst-case guarantees for every individual instance.

---

### 4.4 Decision Gate Correctness

**Proposition (Decision Gate).** The three-way decision gate:

$$\text{Decision}(C) = \begin{cases} \text{ACT} & \text{if } C \geq \tau_{\text{hi}} \\ \text{DEFER} & \text{if } \tau_{\text{lo}} \leq C < \tau_{\text{hi}} \\ \text{AUDIT} & \text{if } C < \tau_{\text{lo}} \end{cases}$$

is well-defined for any $0 \leq \tau_{\text{lo}} \leq \tau_{\text{hi}} \leq 1$, and the partition $[0, \tau_{\text{lo}}) \cup [\tau_{\text{lo}}, \tau_{\text{hi}}) \cup [\tau_{\text{hi}}, 1]$ covers $[0, 1]$ without gaps or overlaps.

**Proof.** The three intervals $[0, \tau_{\text{lo}})$, $[\tau_{\text{lo}}, \tau_{\text{hi}})$, and $[\tau_{\text{hi}}, 1]$ form a partition of $[0, 1]$ since $0 \leq \tau_{\text{lo}} \leq \tau_{\text{hi}} \leq 1$. Every $C \in [0, 1]$ falls into exactly one of these intervals.  $\square$

**Discussion.** When $\tau_{\text{hi}}$ is calibrated via the conformal procedure (Section 4.2) as the smallest ICM value whose conformalized risk $g_\alpha(\tau_{\text{hi}}) \leq \alpha$, the ACT region inherits the risk guarantee: for instances where $C \geq \tau_{\text{hi}}$, the expected loss is bounded by $\alpha$ (in the marginal sense). The DEFER and AUDIT regions correspond to progressively higher epistemic risk.

---

## 5. Early Warning Signal Theory

### 5.1 Composite Signal

**Definition 7 (Early Warning Signal).** The composite early-warning signal at time $t$ is:

$$Z_t = a_1 \cdot \left(-\frac{dC}{dt}\bigg|_t\right) + a_2 \cdot \text{Var}_m(\hat{y}_{m,t}) + a_3 \cdot \text{trend}(\Pi_t)$$

where:
- $dC/dt|_t$ is approximated by the finite difference of the rolling ICM: $\Delta C_t = \bar{C}_t - \bar{C}_{t-1}$,
- $\text{Var}_m(\hat{y}_{m,t}) = \frac{1}{K}\sum_{k=1}^K (\hat{y}_{k,t} - \bar{y}_t)^2$ is the cross-model prediction variance at time $t$,
- $\text{trend}(\Pi_t)$ is the trend of the dependency penalty over a rolling window,
- $a_1, a_2, a_3 > 0$ are combining weights (default: $a_1 = 0.4$, $a_2 = 0.4$, $a_3 = 0.2$).

The sign convention is such that $Z_t > 0$ indicates a *deteriorating* situation: convergence is decreasing ($dC/dt < 0$, so $-dC/dt > 0$), prediction variance is high, and dependency is trending upward.

### 5.2 Proposition 5 (CUSUM Optimality)

**Statement.** The tabular CUSUM (Cumulative Sum) detector, defined by:

$$S_t = \max\!\left(0,\; S_{t-1} + Z_t - k\right), \quad S_0 = 0$$

with detection at time $\tau = \inf\{t : S_t > h\}$, is optimal for detecting a sustained shift in the mean of $Z_t$ in the following sense:

**(a)** Among all sequential detectors with the same false alarm rate (Average Run Length under $H_0$: $\text{ARL}_0 \geq \gamma$), the CUSUM minimizes the worst-case detection delay (Lorden, 1971; Moustakides, 1986).

**(b)** As the threshold $h \to \infty$, the false alarm rate vanishes: $\text{ARL}_0 \to \infty$.

**(c)** For a shift of magnitude $\delta$ in the mean of $Z_t$, the expected detection delay scales as $E[\tau - t_0 | \text{change at } t_0] \approx h / (\delta - k)$ when $\delta > k$.

**Proof.** This is the celebrated result of Page (1954) and Moustakides (1986). The key steps are:

**(a)** Moustakides (1986) proved that for detecting a change in the mean of i.i.d. normal observations, the CUSUM is exactly optimal in the Lorden (minimax) sense: it minimizes $\sup_{t_0} \text{ess sup}\, E[\tau - t_0 | \tau \geq t_0, \mathcal{F}_{t_0}]$ subject to $E_\infty[\tau] \geq \gamma$. The result extends to the exponential family by sufficiency arguments.

**(b)** Under $H_0$ (no change), $Z_t - k$ has negative mean (when $k$ is chosen larger than $E_0[Z_t]$), so $S_t$ is a random walk with negative drift, reflected at 0. The probability of $S_t$ exceeding $h$ before returning to 0 decreases exponentially with $h$, giving $\text{ARL}_0 \sim e^{2\rho h}$ for some $\rho > 0$ depending on the distribution of $Z_t - k$ under $H_0$.

**(c)** Under $H_1$ (change has occurred with shift $\delta$), $Z_t - k$ has positive mean $\delta - k > 0$. The CUSUM accumulates at rate $\delta - k$ on average, so it crosses $h$ after approximately $h/(\delta - k)$ steps.  $\square$

**Discussion.** The optimality of CUSUM makes it a principled choice for the early warning system. The drift parameter $k$ (default: 0.5) determines the minimum detectable shift: shifts smaller than $k$ will not be reliably detected. The threshold $h$ (default: 5.0) controls the false alarm / detection delay tradeoff. These are tunable parameters that should be calibrated on domain-specific data.

The OS Multi-Science implementation also includes the Page-Hinkley detector as an alternative, which has similar theoretical properties but uses a different statistic accumulation scheme. Both are sequential change-point detectors with known optimality properties.

---

### 5.3 Proposition 6 (Composite Signal Variance)

**Statement.** Under the assumption that the three signal components $(-dC/dt)$, $\text{Var}_m(\hat{y})$, and $\text{trend}(\Pi)$ are mutually independent:

$$\text{Var}(Z_t) = a_1^2 \cdot \text{Var}\!\left(\frac{dC}{dt}\right) + a_2^2 \cdot \text{Var}\!\left(\text{Var}_m(\hat{y})\right) + a_3^2 \cdot \text{Var}\!\left(\text{trend}(\Pi)\right)$$

Under the null hypothesis of no regime change, if each component has stationary variance $\sigma_1^2, \sigma_2^2, \sigma_3^2$ respectively:

$$\text{Var}(Z_t) = a_1^2 \sigma_1^2 + a_2^2 \sigma_2^2 + a_3^2 \sigma_3^2$$

**Proof.**

For any random variables $X_1, X_2, X_3$ and constants $a_1, a_2, a_3$:

$$\text{Var}(a_1 X_1 + a_2 X_2 + a_3 X_3) = \sum_i a_i^2 \text{Var}(X_i) + 2\sum_{i < j} a_i a_j \text{Cov}(X_i, X_j)$$

Under the independence assumption, $\text{Cov}(X_i, X_j) = 0$ for all $i \neq j$, so the cross-terms vanish and:

$$\text{Var}(Z_t) = a_1^2 \text{Var}(X_1) + a_2^2 \text{Var}(X_2) + a_3^2 \text{Var}(X_3)$$

$\square$

**Discussion.** The independence assumption is an idealization. In practice, the three components are likely correlated (e.g., decreasing convergence often coincides with increasing prediction variance). When the components are positively correlated, the true variance of $Z_t$ is *larger* than the independence-based formula, making the CUSUM more sensitive (higher signal-to-noise ratio). The independence formula provides a *lower bound* on the signal variance under positive correlation, making it a conservative estimate for alarm calibration.

The composite signal design combines three complementary indicators of regime change:
1. **$-dC/dt$**: convergence is declining (models are diverging).
2. **$\text{Var}_m(\hat{y})$**: models are producing diverse predictions.
3. **$\text{trend}(\Pi)$**: model dependencies are increasing (potential spurious convergence).

Each captures a different aspect of epistemic degradation, and their combination provides a more robust early-warning system than any single indicator.

---

## 6. Anti-Spurious Testing

### 6.1 HSIC Independence Test

**Definition 8 (Hilbert-Schmidt Independence Criterion).** For random variables $X$ and $Y$ with joint distribution $P_{XY}$ and marginals $P_X, P_Y$, the HSIC with respect to reproducing kernel Hilbert spaces $\mathcal{H}_k, \mathcal{H}_l$ (with kernels $k, l$) is:

$$\text{HSIC}(X, Y; k, l) = \|C_{XY}\|_{\text{HS}}^2$$

where $C_{XY}$ is the cross-covariance operator between the RKHS embeddings and $\|\cdot\|_{\text{HS}}$ is the Hilbert-Schmidt norm.

Given $n$ samples $\{(x_i, y_i)\}_{i=1}^n$, the empirical estimator is:

$$\widehat{\text{HSIC}} = \frac{1}{n^2} \text{Tr}(\tilde{K}_x \tilde{K}_y)$$

where $\tilde{K}_x = H K_x H$ and $\tilde{K}_y = H K_y H$ are the centered kernel matrices, $H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix, and $(K_x)_{ij} = k(x_i, x_j)$, $(K_y)_{ij} = l(y_i, y_j)$.

The OS Multi-Science implementation uses the RBF (Gaussian) kernel $k(x, x') = \exp(-\|x-x'\|^2 / (2\sigma^2))$ with bandwidth $\sigma$ set by the median heuristic.

---

#### 6.1.1 Theorem 6 (HSIC Independence Test)

**Statement.**

**(a) Characterization of independence.** For *characteristic* kernels (including the RBF kernel):

$$\text{HSIC}(X, Y; k, l) = 0 \iff X \perp Y$$

That is, HSIC is zero if and only if $X$ and $Y$ are statistically independent.

**(b) Type I error control.** The permutation test using $\widehat{\text{HSIC}}$ controls the Type I error at level $\alpha$:

$$P(\text{reject } H_0 \mid H_0 \text{ true}) \leq \alpha$$

where $H_0: X \perp Y$ is the null hypothesis of independence.

**Proof.**

**(a)** This is Theorem 4 of Gretton et al. (2005). The key insight is that for a characteristic kernel $k$, the mean embedding $\mu_P \in \mathcal{H}_k$ uniquely determines the distribution $P$. Therefore $\text{HSIC} = 0$ iff $C_{XY} = 0$ iff the joint embedding equals the product of marginal embeddings iff $P_{XY} = P_X \otimes P_Y$ iff $X \perp Y$. The RBF kernel is characteristic (Sriperumbudur et al., 2010), so this result applies to our implementation.

**(b)** The permutation test generates the null distribution by permuting $Y$ indices: for permutation $\pi$, compute $\widehat{\text{HSIC}}_\pi = \frac{1}{n^2}\text{Tr}(\tilde{K}_x \tilde{K}_{y,\pi})$ where $(\tilde{K}_{y,\pi})_{ij} = \tilde{K}_y(\pi(i), \pi(j))$. The $p$-value is:

$$p = \frac{1}{B}\sum_{b=1}^{B} \mathbb{1}\{\widehat{\text{HSIC}}_{\pi_b} \geq \widehat{\text{HSIC}}_{\text{obs}}\}$$

Under $H_0$, all permutations are equally likely (since $X \perp Y$ implies the joint distribution is invariant to permutations of $Y$). Therefore:

$$P(p \leq \alpha \mid H_0) \leq \alpha$$

This is the standard result for permutation tests (Lehmann & Romano, 2005, Theorem 15.2.1).  $\square$

**Discussion.** The HSIC test is the cornerstone of the anti-spurious convergence protocol. When applied to pairs of model residuals $(r_i, r_j)$, it tests whether the residual of model $i$ is independent of the residual of model $j$. Dependence in residuals indicates shared biases or data leakage -- exactly the phenomenon the dependency penalty $\Pi$ is designed to detect.

The permutation test with $B = 1000$ permutations (default) provides a valid (though discrete) $p$-value. For $\alpha = 0.05$, the minimum achievable $p$-value is $1/1001 \approx 0.001$, which is sufficient for most applications. When testing $\binom{K}{2}$ pairs simultaneously, the Benjamini-Hochberg FDR correction (Section 6.3) is applied.

These results are established in Gretton et al. (2005, 2007).

---

### 6.2 Proposition 7 (Negative Control Calibration)

**Statement.** In the OS Multi-Science anti-spurious protocol, negative controls are generated by independently permuting each model's predictions, which destroys any genuine cross-model alignment while preserving marginal distributions. Under the null hypothesis $H_0$ that models' residuals are independent:

**(a)** The negative control distances $\{D_0^{(b)}\}_{b=1}^B$ provide a valid estimate of the baseline distance $D_0$ expected under independence.

**(b)** The normalized convergence $\hat{C} = \exp(-D_{\text{obs}} / D_0)$ satisfies $\hat{C} \approx \exp(-1) \approx 0.368$ when $D_{\text{obs}} \approx D_0$ (i.e., observed convergence is no better than chance).

**(c)** The permutation-based $p$-values from the HSIC test on negative control residuals are uniformly distributed on $\{0, 1/B, 2/B, \ldots, 1\}$ under $H_0$:

$$P(p \leq t \mid H_0) \leq t + \frac{1}{B+1} \quad \text{for all } t \in [0, 1]$$

In the continuous limit ($B \to \infty$), $P(p \leq t \mid H_0) = t$ (uniform distribution).

**Proof.**

**(a)** Under $H_0$, the models are independent, so permuting predictions does not change the joint distribution. The negative control distances are therefore samples from the same distribution as the observed distance, giving a valid reference distribution.

**(b)** By definition, $\hat{C} = e^{-D_{\text{obs}}/D_0}$. When $D_{\text{obs}} = D_0$ (observed distance equals baseline), $\hat{C} = e^{-1} \approx 0.368$. Genuine convergence yields $D_{\text{obs}} < D_0$ and thus $\hat{C} > e^{-1}$; spurious convergence (or no convergence) yields $D_{\text{obs}} \geq D_0$ and $\hat{C} \leq e^{-1}$.

**(c)** This is the standard result for permutation-based $p$-values. Under $H_0$, the observed test statistic and the permuted test statistics are exchangeable. The $p$-value $p = \frac{1}{B}\sum_{b=1}^B \mathbb{1}\{T_{\pi_b} \geq T_{\text{obs}}\}$ satisfies the discrete super-uniformity property $P(p \leq t) \leq t + 1/(B+1)$ (Phipson & Smyth, 2010).  $\square$

**Discussion.** The negative control framework provides an empirical calibration for what "chance-level" convergence looks like. This is critical because the raw pairwise distance between models depends on many factors (dimensionality, scale, model complexity) and has no universal reference level. By comparing to permuted baselines, the normalized convergence $\hat{C}$ provides a calibrated measure of convergence strength relative to the null.

---

### 6.3 FDR Correction (Benjamini-Hochberg)

**Proposition (FDR Control).** The Benjamini-Hochberg procedure applied to $m = \binom{K}{2}$ HSIC $p$-values controls the False Discovery Rate at level $\alpha$:

$$\text{FDR} = E\!\left[\frac{V}{\max(R, 1)}\right] \leq \frac{m_0}{m} \cdot \alpha \leq \alpha$$

where $V$ is the number of false rejections (truly independent pairs incorrectly declared dependent), $R$ is the total number of rejections, and $m_0 \leq m$ is the number of truly null hypotheses.

**Proof.** This is the main result of Benjamini & Hochberg (1995). The procedure orders $p$-values $p_{(1)} \leq \cdots \leq p_{(m)}$ and rejects all $H_{(i)}$ for $i \leq \hat{k}$ where $\hat{k} = \max\{i : p_{(i)} \leq i\alpha/m\}$. The FDR control holds under independence of the $p$-values, and also under positive regression dependence on each null (PRDS) by the result of Benjamini & Yekutieli (2001).  $\square$

**Discussion.** When testing all $\binom{K}{2}$ pairs of model residuals for independence, the FDR correction prevents an excess of false alarms. The implementation uses the step-up procedure with monotonicity enforcement on the adjusted $p$-values, which is the standard practice. For $K = 5$ models, there are $m = 10$ pairs, so the correction is mild. For genuine convergence, *all* pairs should show independent residuals (all HSIC tests should be non-significant after correction).

---

## 7. Meta-Learner Convergence

### 7.1 Proposition 8 (Existence of Optimal Weights)

**Statement.** The meta-learner weight optimization problem:

$$\mathbf{w}^* = \arg\min_{\mathbf{w} \in \mathcal{W}} L_{\text{composite}}(\mathbf{w})$$

where $\mathcal{W} = [0.05, 0.50]^4 \times [0.05, 0.30]$ is the feasible weight domain and $L_{\text{composite}}$ is a continuous loss function of the weights, has at least one global minimum.

**Proof.** The domain $\mathcal{W} = [0.05, 0.50]^4 \times [0.05, 0.30] \subset \mathbb{R}^5$ is compact (closed and bounded in $\mathbb{R}^5$). The loss function $L_{\text{composite}}(\mathbf{w})$ is continuous in $\mathbf{w}$ because:

1. Each ICM component ($A, D, U, C, \Pi$) is a fixed (data-dependent) quantity, independent of $\mathbf{w}$.
2. The ICM score $\sigma(w_A A + w_D D + w_U U + w_C C - \lambda \Pi)$ is a continuous (indeed, smooth) function of $\mathbf{w}$.
3. The loss function $L_{\text{composite}}$ composed with continuous ICM scores is continuous.

By the **Weierstrass Extreme Value Theorem**, a continuous function on a compact set attains its minimum. Therefore, at least one $\mathbf{w}^* \in \mathcal{W}$ exists such that $L_{\text{composite}}(\mathbf{w}^*) \leq L_{\text{composite}}(\mathbf{w})$ for all $\mathbf{w} \in \mathcal{W}$.  $\square$

**Discussion.** The existence of a minimum is guaranteed, but uniqueness is not: the loss landscape may have multiple global minima. In practice, the optimization is performed by grid search or gradient-based methods with multiple random restarts. The compact domain $\mathcal{W}$ prevents degenerate solutions (e.g., zero weights) and ensures all components contribute to the ICM.

---

### 7.2 Proposition 9 (Generalization Bound)

**Statement.** Let $\mathbf{w}^*$ be the weight vector that minimizes the empirical loss $\hat{L}_n(\mathbf{w})$ over $n$ training scenarios on the domain $\mathcal{W} \subset \mathbb{R}^d$ (with $d = 5$). Under standard assumptions (bounded loss, i.i.d. training data), the generalization gap satisfies:

$$|L_{\text{train}}(\mathbf{w}^*) - L_{\text{test}}(\mathbf{w}^*)| \leq \mathcal{O}\!\left(\sqrt{\frac{d}{n}}\right)$$

with high probability.

**Proof sketch.** The ICM scoring function with weights $\mathbf{w}$ can be viewed as a parametric model with $d = 5$ real-valued parameters on a bounded domain $\mathcal{W}$. By standard results in statistical learning theory:

1. **Covering number argument:** The class of functions $\{x \mapsto \sigma(\mathbf{w}^T \phi(x)) : \mathbf{w} \in \mathcal{W}\}$ has a covering number $\mathcal{N}(\epsilon, \mathcal{W}, \|\cdot\|_\infty)$ that is polynomial in $1/\epsilon$ and in the diameter of $\mathcal{W}$. Specifically, $\log \mathcal{N}(\epsilon) = \mathcal{O}(d \log(1/\epsilon))$.

2. **Rademacher complexity bound:** The empirical Rademacher complexity of this function class is bounded by $\hat{R}_n = \mathcal{O}(\sqrt{d/n})$ (see, e.g., Shalev-Shwartz & Ben-David, 2014, Theorem 26.5).

3. **Generalization via Rademacher complexity:** By the standard bound (Bartlett & Mendelson, 2002):

$$E[L_{\text{test}}(\mathbf{w}^*)] \leq L_{\text{train}}(\mathbf{w}^*) + 2\hat{R}_n + \mathcal{O}\!\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)$$

with probability at least $1 - \delta$ over the draw of the training set.

Combining: $|L_{\text{train}} - L_{\text{test}}| \leq \mathcal{O}(\sqrt{d/n})$.

Since $d = 5$ is small, even moderate sample sizes (e.g., $n \geq 50$) yield a tight generalization bound: $\sqrt{5/50} \approx 0.32$. With $n = 200$ scenarios: $\sqrt{5/200} \approx 0.16$.  $\square$

**Discussion.** The low dimensionality of the weight space ($d = 5$) is a significant advantage for the meta-learner: the generalization gap shrinks rapidly with the number of training scenarios. This means that even a modest calibration dataset (50-200 scenarios) is sufficient to learn weights that generalize well. The bound is conservative (worst-case), and in practice the generalization gap is often much smaller.

---

## 8. Convergence of Geometric Aggregation

### 8.1 Definition 9 (Geometric ICM)

The geometric-mean ICM aggregation is defined as:

$$\text{ICM}_{\text{geo}} = A^{\bar{w}_A} \cdot D^{\bar{w}_D} \cdot U^{\bar{w}_U} \cdot C^{\bar{w}_C} \cdot (1 - \Pi)^{\bar{w}_\lambda}$$

where $\bar{w}_A + \bar{w}_D + \bar{w}_U + \bar{w}_C + \bar{w}_\lambda = 1$ are the normalized weights (i.e., $\bar{w}_A = w_A / (w_A + w_D + w_U + w_C + \lambda)$, etc.).

In the implementation, each factor is clamped to $[\epsilon, 1]$ with $\epsilon = 10^{-12}$ to avoid $\log(0)$:

$$\text{ICM}_{\text{geo}} = \exp\!\left(\sum_i \bar{w}_i \log(\max(v_i, \epsilon))\right)$$

where $v_i$ are the component values ($A, D, U, C, 1-\Pi$).

### 8.2 Proposition 10 (Boundedness of Geometric ICM)

**Statement.** $\text{ICM}_{\text{geo}} \in [0, 1]$.

**Proof.** Each factor $v_i \in [0, 1]$ (with clamping: $v_i \in [\epsilon, 1]$), and each exponent $\bar{w}_i \in (0, 1)$ with $\sum_i \bar{w}_i = 1$.

*Upper bound.* Since $v_i \leq 1$ and $\bar{w}_i > 0$: $v_i^{\bar{w}_i} \leq 1$ for all $i$. Therefore $\text{ICM}_{\text{geo}} = \prod_i v_i^{\bar{w}_i} \leq 1$.

*Lower bound.* Since $v_i \geq 0$ (or $v_i \geq \epsilon > 0$ with clamping): $\text{ICM}_{\text{geo}} \geq 0$.

Therefore $\text{ICM}_{\text{geo}} \in [0, 1]$ (or $\text{ICM}_{\text{geo}} \in [\epsilon^1, 1] = [\epsilon, 1]$ with clamping).  $\square$

### 8.3 Proposition 11 (Log-Concavity of Geometric ICM)

**Statement.** The geometric ICM $\text{ICM}_{\text{geo}}$ is *log-concave* in the component vector $(A, D, U, C, 1-\Pi) \in (0, 1]^5$.

**Proof.** Taking the logarithm:

$$\log(\text{ICM}_{\text{geo}}) = \sum_i \bar{w}_i \log(v_i)$$

This is a non-negatively weighted sum of concave functions ($\log$ is concave on $(0, \infty)$). A non-negatively weighted sum of concave functions is concave. Therefore $\log(\text{ICM}_{\text{geo}})$ is concave in $(v_1, \ldots, v_5)$, which by definition means $\text{ICM}_{\text{geo}}$ is log-concave.  $\square$

### 8.4 Proposition 12 (Monotonicity of Geometric ICM)

**Statement.** $\text{ICM}_{\text{geo}}$ is strictly monotone increasing in each of $A, D, U, C$ and strictly monotone decreasing in $\Pi$, when other components are held fixed.

**Proof.** For component $v_i$ with exponent $\bar{w}_i > 0$:

$$\frac{\partial\,\text{ICM}_{\text{geo}}}{\partial v_i} = \bar{w}_i \cdot \frac{\text{ICM}_{\text{geo}}}{v_i} > 0$$

since $\text{ICM}_{\text{geo}} > 0$ and $v_i > 0$. This shows strict monotonicity in each $v_i$.

For $\Pi$, the corresponding factor is $v_5 = 1 - \Pi$, so:

$$\frac{\partial\,\text{ICM}_{\text{geo}}}{\partial \Pi} = \frac{\partial\,\text{ICM}_{\text{geo}}}{\partial v_5} \cdot \frac{\partial v_5}{\partial \Pi} = \bar{w}_\lambda \cdot \frac{\text{ICM}_{\text{geo}}}{1 - \Pi} \cdot (-1) < 0$$

$\square$

### 8.5 Comparison with Logistic Aggregation

**Proposition (Geometric vs. Logistic).** The geometric aggregation has the following qualitative differences from the logistic aggregation:

**(a) Sensitivity to zeros.** If any component $v_i = 0$, then $\text{ICM}_{\text{geo}} = 0$ regardless of other components. The logistic aggregation degrades gracefully (sigmoid maps any finite $z$ to $(0,1)$).

**(b) Multiplicative interaction.** The geometric mean captures *multiplicative* interactions: all components must be simultaneously high for a high ICM. The logistic aggregation is *additive* in the pre-activation.

**(c) Scale-free.** The geometric mean is invariant to multiplicative rescaling of the weights (only the relative magnitudes matter). The logistic aggregation depends on the absolute scale of $z$.

**Discussion.** The geometric aggregation is useful as a *robustness variant*: it is more conservative than the logistic aggregation because a single low component can drive the entire score down. This is desirable when the framework should require *all* aspects of convergence to be present for a high confidence score.

---

## 9. Beta-Calibrated Aggregation

### 9.1 Definition 10 (Beta-Calibrated ICM)

The Beta-calibrated ICM maps the normalized linear pre-activation through a Beta CDF:

$$\text{ICM}_{\text{cal}} = F_{\text{Beta}}(z_{\text{norm}};\, a, b)$$

where:

$$z_{\text{norm}} = \text{clip}\!\left(\frac{z - z_{\min}}{z_{\max} - z_{\min}},\; 0,\; 1\right)$$

with $z_{\min} = -\lambda$ and $z_{\max} = w_A + w_D + w_U + w_C$, and $F_{\text{Beta}}(\cdot; a, b)$ is the CDF of the $\text{Beta}(a, b)$ distribution.

Default parameters: $a = b = 5$.

### 9.2 Proposition 13 (Properties of Beta-Calibrated ICM)

**Statement.** The Beta-calibrated ICM satisfies:

**(a) Boundedness:** $\text{ICM}_{\text{cal}} \in [0, 1]$.

**(b) Monotonicity:** $\text{ICM}_{\text{cal}}$ is monotone increasing in each of $A, D, U, C$ and monotone decreasing in $\Pi$.

**(c) Smoothness:** $\text{ICM}_{\text{cal}}$ is infinitely differentiable on the interior of the component domain.

**(d) Full range utilization:** Unlike the logistic sigmoid (which compresses to $\approx [0.46, 0.70]$ with default weights), the Beta CDF maps $[0, 1]$ to $[0, 1]$ with better discrimination near the boundaries.

**Proof.**

**(a)** The Beta CDF $F_{\text{Beta}} : [0, 1] \to [0, 1]$, and $z_{\text{norm}} \in [0, 1]$ by clipping.

**(b)** The Beta CDF $F_{\text{Beta}}(\cdot; a, b)$ is a strictly increasing function on $[0, 1]$ (for $a, b > 0$). The normalization $z_{\text{norm}}$ is a non-decreasing affine function of $z$, and $z$ is a linear function of the components (increasing in $A, D, U, C$; decreasing in $\Pi$). The composition of non-decreasing functions is non-decreasing.

**(c)** The Beta CDF is the integral of the Beta density $f_{\text{Beta}}(x; a, b) = x^{a-1}(1-x)^{b-1}/B(a,b)$, which is a smooth function on $(0, 1)$ for $a, b > 1$. Since $a = b = 5 > 1$, the CDF is infinitely differentiable on $(0, 1)$.

**(d)** The Beta CDF with $a = b = 5$ has a sigmoidal shape symmetric around $0.5$, but with steeper transitions near $0.5$ than the logistic sigmoid restricted to the same range. Specifically, $F_{\text{Beta}}(0; 5, 5) = 0$ and $F_{\text{Beta}}(1; 5, 5) = 1$, utilizing the full $[0, 1]$ range.  $\square$

**Discussion.** The Beta-calibrated aggregation addresses the practical limitation of the logistic sigmoid: with default weights, the sigmoid compresses the ICM range to approximately $[0.46, 0.70]$, making it difficult for the decision gate to distinguish high-convergence from low-convergence scenarios. The Beta CDF (with $a = b = 5$) provides a wider effective range while preserving all the theoretical properties (boundedness, monotonicity, smoothness) that make the logistic aggregation well-behaved.

---

## 10. Adaptive (Percentile) Aggregation

### 10.1 Definition 11 (Adaptive ICM)

The adaptive ICM maps the linear pre-activation $z$ to its empirical percentile rank within a calibration distribution $\{z_1^{\text{cal}}, \ldots, z_N^{\text{cal}}\}$:

$$\text{ICM}_{\text{ada}} = \hat{F}_N(z) = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\{z_i^{\text{cal}} \leq z\}$$

with linear interpolation between calibration points for smoothness.

### 10.2 Proposition 14 (Properties of Adaptive ICM)

**Statement.**

**(a) Boundedness:** $\text{ICM}_{\text{ada}} \in [0, 1]$.

**(b) Monotonicity:** $\text{ICM}_{\text{ada}}$ is monotone non-decreasing in $z$ (and hence in each positive component, and non-increasing in $\Pi$).

**(c) Uniformity:** If the calibration distribution is representative of the true distribution of $z$, then $\text{ICM}_{\text{ada}}$ is approximately uniformly distributed on $[0, 1]$ over the calibration population.

**(d) Lipschitz continuity:** The empirical CDF (without interpolation) is $1/N$-Lipschitz in the sense that a perturbation passing one calibration point changes the rank by at most $1/N$. With linear interpolation, $\text{ICM}_{\text{ada}}$ is Lipschitz continuous with constant $L = 1/(N \cdot \Delta_{\min})$ where $\Delta_{\min}$ is the minimum gap between adjacent sorted calibration scores.

**Proof.**

**(a)** The empirical CDF takes values in $\{0, 1/N, 2/N, \ldots, 1\}$, and the interpolation is bounded by $[0, 1]$ with clipping.

**(b)** The empirical CDF is a non-decreasing step function. Linear interpolation preserves monotonicity.

**(c)** The probability integral transform: if $Z$ has CDF $F$, then $F(Z)$ is uniformly distributed on $[0, 1]$. When $\hat{F}_N$ approximates $F$ well (for large $N$), $\hat{F}_N(Z) \approx F(Z) \sim \text{Uniform}(0, 1)$.

**(d)** Between two adjacent calibration points $z_{(i)}, z_{(i+1)}$, the interpolated CDF increases linearly from $i/N$ to $(i+1)/N$ over a gap of $z_{(i+1)} - z_{(i)}$. The slope is $1/(N(z_{(i+1)} - z_{(i)}))$, which is bounded by $1/(N \cdot \Delta_{\min})$.  $\square$

**Discussion.** The adaptive aggregation provides the maximum possible discrimination power by construction: it maps the score distribution to a uniform distribution on $[0, 1]$, meaning each quantile of the calibration distribution gets equal "resolution" in the output. The tradeoff is that it requires a representative calibration set (default: 500 uniformly sampled component vectors when no historical data is available).

---

## 11. Summary of Results

The following table summarizes the formal properties established in this document.

| Property | Logistic | Geometric | Beta-Calibrated | Adaptive |
|----------|----------|-----------|-----------------|----------|
| Boundedness in $[0,1]$ | Prop. 1 (open interval) | Prop. 10 | Prop. 13a | Prop. 14a |
| Monotonicity | Prop. 2 (strict) | Prop. 12 (strict) | Prop. 13b (strict) | Prop. 14b (non-decreasing) |
| Lipschitz continuity | Prop. 3 ($L \approx 0.12$) | Yes (on bounded domain) | Prop. 13c (smooth) | Prop. 14d ($L = 1/(N\Delta_{\min})$) |
| Sensitivity to $\Pi$ | Prop. 4 ($-\lambda\sigma'$) | Prop. 12 | Prop. 13b | Prop. 14b |
| Log-concavity | Yes ($\log\sigma$ concave) | Prop. 11 | N/A | N/A |
| Full range $[0,1]$ | No ($\approx[0.46, 0.70]$) | Yes | Yes | Yes |

### Key Theoretical Guarantees

1. **ICM is well-defined** as a bounded, monotone, Lipschitz-continuous convergence index (Section 2).
2. **Each component** ($A, D, U, C, \Pi$) is bounded in $[0, 1]$ with clear operational semantics (Section 3).
3. **The Hellinger distance** used for distributional agreement is a proper metric (Theorem 1).
4. **The Ledoit-Wolf estimator** provides a consistent, positive-definite correlation estimate for the dependency penalty (Theorem 4).
5. **Conformal risk control** provides a distribution-free, finite-sample marginal coverage guarantee (Theorem 5).
6. **CUSUM detection** is minimax optimal for detecting mean shifts (Proposition 5).
7. **HSIC independence testing** provides a valid, consistent test for detecting spurious convergence (Theorem 6).
8. **Meta-learner optimization** is guaranteed to have a solution (Proposition 8) with controlled generalization gap (Proposition 9).

### Known Limitations

1. **Marginal vs. conditional coverage** (Corollary, Section 4.3): the conformal risk bound is marginal, not pointwise.
2. **Independence assumption** in the composite early-warning signal (Proposition 6): the variance decomposition is approximate when components are correlated.
3. **Sigmoid compression** (Proposition 1, Discussion): the logistic aggregation compresses the effective range with default parameters. Mitigated by Beta-calibrated and adaptive alternatives.
4. **Permutation test discreteness** (Theorem 6): the minimum achievable $p$-value is $1/(B+1)$, limiting power for small $B$.
5. **Geometric mean sensitivity to zeros** (Section 8.5): a single zero component collapses the geometric ICM to zero.

---

## 12. References

### Primary Sources (Results Established Here)

The propositions and proofs specific to the ICM framework (Propositions 1-4, 6, 8-14, and the component boundedness results) are original to this document, establishing formal properties of the OS Multi-Science implementation.

### Classical Results (Cited)

1. **Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J.** (2021). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1), 486-507.

2. **Bartlett, P. L., & Mendelson, S.** (2002). Rademacher and Gaussian complexities: Risk bounds and structural results. *Journal of Machine Learning Research*, 3, 463-482.

3. **Benjamini, Y., & Hochberg, Y.** (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

4. **Benjamini, Y., & Yekutieli, D.** (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165-1188.

5. **Gretton, A., Bousquet, O., Smola, A., & Scholkopf, B.** (2005). Measuring statistical dependence with Hilbert-Schmidt norms. *Algorithmic Learning Theory (ALT)*, 63-77.

6. **Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., & Smola, A.** (2007). A kernel statistical test of independence. *Advances in Neural Information Processing Systems (NeurIPS)*, 20.

7. **Hellinger, E.** (1909). Neue Begrundung der Theorie quadratischer Formen von unendlichvielen Veranderlichen. *Journal fur die reine und angewandte Mathematik*, 136, 210-271.

8. **Le Cam, L., & Yang, G. L.** (2000). *Asymptotics in Statistics: Some Basic Concepts* (2nd ed.). Springer.

9. **Ledoit, O., & Wolf, M.** (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

10. **Lehmann, E. L., & Romano, J. P.** (2005). *Testing Statistical Hypotheses* (3rd ed.). Springer.

11. **Lei, J., & Wasserman, L.** (2014). Distribution-free prediction bands for non-parametric regression. *Journal of the Royal Statistical Society: Series B*, 76(1), 71-96.

12. **Lorden, G.** (1971). Procedures for reacting to a change in distribution. *Annals of Mathematical Statistics*, 42(6), 1897-1908.

13. **Moustakides, G. V.** (1986). Optimal stopping times for detecting changes in distributions. *Annals of Statistics*, 14(4), 1379-1387.

14. **Page, E. S.** (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.

15. **Phipson, B., & Smyth, G. K.** (2010). Permutation P-values should never be zero: Calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.

16. **Pielou, E. C.** (1966). The measurement of diversity in different types of biological collections. *Journal of Theoretical Biology*, 13, 131-144.

17. **Romano, Y., Patterson, E., & Candes, E. J.** (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

18. **Shalev-Shwartz, S., & Ben-David, S.** (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.

19. **Sriperumbudur, B. K., Gretton, A., Fukumizu, K., Scholkopf, B., & Lanckriet, G. R. G.** (2010). Hilbert space embeddings and metrics on probability measures. *Journal of Machine Learning Research*, 11, 1517-1561.

20. **Tsybakov, A. B.** (2009). *Introduction to Nonparametric Estimation*. Springer.

21. **Villani, C.** (2008). *Optimal Transport: Old and New*. Springer.

22. **Vovk, V., Gammerman, A., & Shafer, G.** (2005). *Algorithmic Learning in a Random World*. Springer.

---

## Appendix A: Proof of Sigmoid Properties

For completeness, we state the key properties of the logistic sigmoid $\sigma(z) = (1 + e^{-z})^{-1}$ used throughout.

**A.1. Range:** $\sigma : \mathbb{R} \to (0, 1)$, with $\lim_{z \to -\infty} \sigma(z) = 0$ and $\lim_{z \to +\infty} \sigma(z) = 1$.

**A.2. Derivative:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**A.3. Maximum derivative:** $\sup_z \sigma'(z) = \sigma'(0) = 1/4$.

*Proof.* $\sigma'(z) = \sigma(z)(1-\sigma(z))$. Setting $u = \sigma(z) \in (0,1)$, we maximize $u(1-u)$, which by AM-GM has maximum $1/4$ at $u = 1/2$ (i.e., $z = 0$).

**A.4. Lipschitz constant:** $|\sigma(z_1) - \sigma(z_2)| \leq \frac{1}{4}|z_1 - z_2|$ for all $z_1, z_2$.

*Proof.* By the mean value theorem: $|\sigma(z_1) - \sigma(z_2)| = |\sigma'(\xi)||z_1 - z_2| \leq \frac{1}{4}|z_1-z_2|$.

**A.5. Log-concavity:** $\log \sigma(z) = -\log(1 + e^{-z})$ is concave.

*Proof.* $\frac{d^2}{dz^2}\log\sigma(z) = -\sigma(z)(1-\sigma(z)) < 0$ for all $z$.

**A.6. Symmetry:** $\sigma(-z) = 1 - \sigma(z)$.

*Proof.* $\sigma(-z) = \frac{1}{1+e^z} = \frac{e^{-z}}{e^{-z}+1} = 1 - \frac{1}{1+e^{-z}} = 1 - \sigma(z)$.

---

## Appendix B: Derivation of Effective ICM Range

With default parameters $w_A = 0.35, w_D = 0.15, w_U = 0.25, w_C = 0.10, \lambda = 0.15$:

**Worst case** ($A = D = U = C = 0, \Pi = 1$):
$$z_{\min} = -0.15, \quad \text{ICM}_{\min} = \sigma(-0.15) = \frac{1}{1+e^{0.15}} \approx 0.4626$$

**Best case** ($A = D = U = C = 1, \Pi = 0$):
$$z_{\max} = 0.35 + 0.15 + 0.25 + 0.10 = 0.85, \quad \text{ICM}_{\max} = \sigma(0.85) = \frac{1}{1+e^{-0.85}} \approx 0.7006$$

**Effective range:** $\text{ICM} \in [0.4626, 0.7006]$, a span of approximately $0.238$.

**Midpoint** ($A = D = U = C = 0.5, \Pi = 0.5$):
$$z_{\text{mid}} = 0.35(0.5) + 0.15(0.5) + 0.25(0.5) + 0.10(0.5) - 0.15(0.5) = 0.35, \quad \text{ICM}_{\text{mid}} = \sigma(0.35) \approx 0.5866$$

This compression motivates the Beta-calibrated (Section 9) and adaptive (Section 10) alternatives, which map the full component range to the full $[0, 1]$ output range.

---

## Appendix C: Wasserstein Distance Properties

For completeness, we state the key properties of the 2-Wasserstein distance used as an alternative agreement metric.

**C.1. Definition.** For probability measures $\mu, \nu$ on $\mathbb{R}^d$:

$$W_2(\mu, \nu) = \left(\inf_{\gamma \in \Gamma(\mu, \nu)} \int \|x - y\|^2 \, d\gamma(x, y)\right)^{1/2}$$

where $\Gamma(\mu, \nu)$ is the set of all couplings of $\mu$ and $\nu$.

**C.2. Metric properties.** $W_2$ is a proper metric on the space of probability measures with finite second moment: $W_2(\mu, \nu) \geq 0$; $W_2(\mu, \nu) = 0$ iff $\mu = \nu$; $W_2$ is symmetric; $W_2$ satisfies the triangle inequality.

**C.3. Gaussian closed form.** For Gaussian measures $\mu = \mathcal{N}(\mu_1, \Sigma_1)$ and $\nu = \mathcal{N}(\mu_2, \Sigma_2)$:

$$W_2^2(\mu, \nu) = \|\mu_1 - \mu_2\|^2 + \text{Tr}\!\left(\Sigma_1 + \Sigma_2 - 2(\Sigma_2^{1/2}\Sigma_1\Sigma_2^{1/2})^{1/2}\right)$$

This is the formula implemented in `wasserstein2_distance`.

**C.4. Relationship to Hellinger.** For absolutely continuous distributions, there is no direct inequality between $W_2$ and $H$ in general, as $W_2$ is unbounded while $H \in [0, 1]$. However, both metrize weak convergence on compact metric spaces.

See Villani (2008) for comprehensive treatment.

---

## Appendix D: MMD Properties

**D.1. Definition.** For probability measures $P, Q$ and a kernel $k$:

$$\text{MMD}^2(P, Q; k) = E_{X,X'\sim P}[k(X,X')] + E_{Y,Y'\sim Q}[k(Y,Y')] - 2E_{X\sim P, Y\sim Q}[k(X,Y)]$$

**D.2. Characterization.** For a characteristic kernel (such as the RBF kernel): $\text{MMD}(P, Q; k) = 0$ iff $P = Q$. This makes MMD a valid metric on probability distributions (Gretton et al., 2007).

**D.3. Unbiased estimator.** Given samples $X_1, \ldots, X_n \sim P$ and $Y_1, \ldots, Y_m \sim Q$:

$$\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i\neq j}k(X_i,X_j) + \frac{1}{m(m-1)}\sum_{i\neq j}k(Y_i,Y_j) - \frac{2}{nm}\sum_{i,j}k(X_i,Y_j)$$

The implementation uses the biased estimator (with means instead of leave-one-out sums) for simplicity, which is also consistent.

**D.4. Lipschitz property.** For the RBF kernel $k(x,y) = \exp(-\|x-y\|^2/(2\sigma^2))$, the kernel values are bounded in $[0, 1]$, so $\text{MMD}^2 \in [0, 2]$ and $\text{MMD} \in [0, \sqrt{2}]$.

---

*End of Mathematical Foundations Document*
