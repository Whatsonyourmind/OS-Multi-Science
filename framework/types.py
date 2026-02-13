"""Core data types for OS Multi-Science."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# --- Enums ---

class Scale(Enum):
    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"
    MULTI_SCALE = "multi_scale"


class Dynamics(Enum):
    STATIC = "static"
    SLOW = "slowly_evolving"
    FAST = "fast"
    CHAOTIC = "chaotic"


class NetworkType(Enum):
    NONE = "none"
    SPARSE = "sparse"
    DENSE = "dense"
    MULTILAYER = "multilayer"


class AgentType(Enum):
    NONE = "none"
    FEW = "few"
    MANY = "many"
    STRATEGIC = "strategic"


class Feedback(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    DELAYED = "delayed"
    NONLINEAR = "nonlinear"


class DataRegime(Enum):
    SCARCE = "scarce"
    MODERATE = "moderate"
    RICH = "rich"
    STREAMING = "streaming"


class Controllability(Enum):
    NONE = "none"
    PARTIAL = "partial"
    HIGH = "high"


class Observability(Enum):
    INPUTS_ONLY = "inputs_only"
    OUTPUTS_ONLY = "outputs_only"
    PARTIAL = "partial"
    DETAILED = "detailed"


class Role(Enum):
    STRUCTURE = "structure"
    BEHAVIOR = "behavior"
    FORECAST = "forecast"
    INTERVENTION = "intervention"
    CAUSAL_ID = "causal_identification"


class EpistemicFamily(Enum):
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    NETWORK = "network_science"
    AGENT_BASED = "agent_based"
    SYSTEM_DYNAMICS = "system_dynamics"
    CAUSAL = "causal_inference"
    OPTIMIZATION = "operations_research"
    EPIDEMIOLOGICAL = "epidemiological"
    TOPOLOGICAL = "topological_data_analysis"
    INFORMATION_THEORY = "information_theory"
    COMPLEX_SYSTEMS = "complex_systems"
    BASELINE = "baseline"


class DecisionAction(Enum):
    ACT = "act"
    DEFER = "defer"
    AUDIT = "audit"


# --- Core Data Types ---

@dataclass
class SystemProfile:
    """AESC system profile with 10 epistemic axes."""
    name: str
    description: str
    scale: Scale = Scale.MESO
    dynamics: Dynamics = Dynamics.SLOW
    network: NetworkType = NetworkType.SPARSE
    agents: AgentType = AgentType.MANY
    feedback: Feedback = Feedback.MODERATE
    data_regime: DataRegime = DataRegime.MODERATE
    controllability: Controllability = Controllability.PARTIAL
    observability: Observability = Observability.PARTIAL
    conservation_laws: list[str] = field(default_factory=list)
    regulatory_context: list[str] = field(default_factory=list)
    primary_roles: list[Role] = field(default_factory=list)
    secondary_roles: list[Role] = field(default_factory=list)


@dataclass
class MethodProfile:
    """Profile of a scientific method/discipline."""
    name: str
    family: EpistemicFamily
    description: str
    assumptions: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    roles_supported: list[Role] = field(default_factory=list)
    data_requirements: str = ""
    computational_cost: str = "medium"  # low/medium/high
    interpretability: str = "medium"  # low/medium/high


@dataclass
class ICMComponents:
    """Individual components of the ICM index."""
    A: float  # Distributional agreement [0, 1]
    D: float  # Direction/sign agreement [0, 1]
    U: float  # Uncertainty overlap [0, 1]
    C: float  # Invariance score [0, 1]
    Pi: float  # Dependency penalty [0, 1]


@dataclass
class ICMResult:
    """Result of ICM computation for a single instance or window."""
    icm_score: float  # Aggregated ICM in [0, 1]
    components: ICMComponents
    pairwise_distances: np.ndarray | None = None
    n_models: int = 0
    aggregation_method: str = "logistic"
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def is_high_convergence(self) -> bool:
        return self.icm_score >= 0.7

    @property
    def is_low_convergence(self) -> bool:
        return self.icm_score < 0.3


@dataclass
class CRCResult:
    """Result of Conformal Risk Control gating."""
    re_score: float  # Epistemic risk bound
    alpha: float  # Confidence level
    decision: DecisionAction
    tau_hi: float
    tau_lo: float
    coverage_empirical: float | None = None
    coverage_nominal: float | None = None


@dataclass
class EarlyWarningSignal:
    """Early warning signal from convergence dynamics."""
    z_score: float  # Composite Z_t signal
    delta_icm: float  # dC/dt
    variance_predictions: float  # Var_m(y_hat)
    dependency_trend: float  # Pi trend
    change_detected: bool = False
    detector_type: str = "cusum"  # cusum / bocpd / page_hinkley


@dataclass
class AntiSpuriousReport:
    """Report from anti-spurious convergence tests."""
    d0_baseline: float  # Negative control baseline distance
    c_normalized: float  # Normalized convergence
    hsic_pvalue: float  # HSIC independence test p-value
    mgc_pvalue: float | None = None  # MGC test p-value
    ablation_results: dict[str, float] = field(default_factory=dict)
    is_genuine: bool = False  # True if passes all tests
    fdr_corrected: bool = False


@dataclass
class RouterSelection:
    """Result of the Router's method selection."""
    selected_methods: list[MethodProfile]
    fit_scores: dict[str, float]
    diversity_matrix: np.ndarray | None = None
    justifications: dict[str, str] = field(default_factory=dict)
    total_fit: float = 0.0
    avg_diversity: float = 0.0


@dataclass
class DecisionCard:
    """One-page summary for decision-makers."""
    problem_snapshot: str
    recommended_kit: list[dict[str, str]]  # [{method, role, why}, ...]
    main_findings: list[str]
    risks_and_limitations: list[str]
    next_actions: list[str]
    icm_summary: ICMResult | None = None
    crc_summary: CRCResult | None = None
