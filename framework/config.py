"""Configuration management for OS Multi-Science."""

from dataclasses import dataclass, field


@dataclass
class ICMConfig:
    """Configuration for ICM computation."""
    # Component weights (logistic aggregation)
    w_A: float = 0.35  # Distributional agreement weight
    w_D: float = 0.15  # Direction/sign weight
    w_U: float = 0.25  # Uncertainty overlap weight
    w_C: float = 0.10  # Invariance weight
    lam: float = 0.15  # Dependency penalty coefficient

    # Dependency penalty sub-weights
    gamma_rho: float = 0.4   # Residual correlation weight
    gamma_J: float = 0.3     # Provenance overlap weight
    gamma_grad: float = 0.3  # Gradient similarity weight

    # Distance normalization constants
    C_A_hellinger: float = 1.0   # Max Hellinger distance
    C_A_wasserstein: float = 10.0  # Wasserstein normalization (domain-dependent)
    C_A_mmd: float = 1.0  # MMD normalization

    # Kernel parameters
    mmd_bandwidth: float = 1.0  # RBF kernel bandwidth for MMD


@dataclass
class CRCConfig:
    """Configuration for Conformal Risk Control."""
    alpha: float = 0.10  # Confidence level (1 - alpha coverage)
    tau_hi: float = 0.7  # High convergence threshold
    tau_lo: float = 0.3  # Low convergence threshold
    calibration_split: float = 0.2  # Fraction for conformal calibration
    n_repeats: int = 10  # Repetitions for stability


@dataclass
class EarlyWarningConfig:
    """Configuration for early warning system."""
    window_size: int = 100  # Rolling window size
    a1: float = 0.4  # Weight for -delta_ICM
    a2: float = 0.4  # Weight for Var(y_hat)
    a3: float = 0.2  # Weight for Pi trend
    cusum_threshold: float = 5.0  # CUSUM detection threshold
    cusum_drift: float = 0.5  # CUSUM drift parameter


@dataclass
class AntiSpuriousConfig:
    """Configuration for anti-spurious convergence tests."""
    n_permutations: int = 1000  # Permutations for HSIC test
    fdr_level: float = 0.05  # FDR correction level
    n_negative_controls: int = 100  # Number of negative control runs


@dataclass
class RouterConfig:
    """Configuration for the method Router."""
    min_methods: int = 3  # Minimum methods in kit
    max_methods: int = 7  # Maximum methods in kit
    diversity_weight: float = 0.4  # Weight of diversity vs fit
    require_interpretable: bool = True  # At least one interpretable method


@dataclass
class OSMultiScienceConfig:
    """Master configuration for the entire system."""
    icm: ICMConfig = field(default_factory=ICMConfig)
    crc: CRCConfig = field(default_factory=CRCConfig)
    early_warning: EarlyWarningConfig = field(default_factory=EarlyWarningConfig)
    anti_spurious: AntiSpuriousConfig = field(default_factory=AntiSpuriousConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    random_seed: int = 42
    verbose: bool = True
