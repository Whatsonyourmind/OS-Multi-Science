"""OS Multi-Science CLI.

Command-line interface exposing the OS Multi-Science framework for
epistemic convergence analysis across scientific disciplines.

Usage:
    python cli.py profile --preset financial
    python cli.py catalog [--role structure] [--family statistical]
    python cli.py route --preset financial [--min-methods 3] [--max-methods 7]
    python cli.py run --preset financial --n-samples 500 [--distance hellinger] [--seed 42]
    python cli.py benchmark --scenario financial|epidemic
    python cli.py meta-learn [--n-scenarios 20] [--method grid|optimize]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
import traceback
from typing import Any

import numpy as np


# ===================================================================
# Text formatting utilities (stdlib only -- no tabulate/rich)
# ===================================================================

_BANNER = r"""
  ___  ____    __  __       _ _   _   ____       _
 / _ \/ ___|  |  \/  |_   _| | |_(_) / ___|  ___(_) ___ _ __   ___ ___
| | | \___ \  | |\/| | | | | | __| | \___ \ / __| |/ _ \ '_ \ / __/ _ \
| |_| |___) | | |  | | |_| | | |_| |  ___) | (__| |  __/ | | | (_|  __/
 \___/|____/  |_|  |_|\__,_|_|\__|_| |____/ \___|_|\___|_| |_|\___\___|
"""


def _header(title: str, width: int = 78) -> str:
    """Return a formatted section header."""
    lines = [
        "",
        "=" * width,
        f"  {title}",
        "=" * width,
    ]
    return "\n".join(lines)


def _subheader(title: str, width: int = 78) -> str:
    """Return a formatted sub-section header."""
    return f"\n--- {title} {'-' * max(0, width - len(title) - 5)}"


def _table(headers: list[str], rows: list[list[str]],
           col_widths: list[int] | None = None, indent: int = 2) -> str:
    """Build a simple text table.

    Parameters
    ----------
    headers : column header strings.
    rows : list of row lists (each element is a string).
    col_widths : explicit per-column widths; auto-computed when *None*.
    indent : number of leading spaces.

    Returns
    -------
    Formatted table as a single string.
    """
    if not rows:
        return "  (no data)"

    n_cols = len(headers)

    if col_widths is None:
        col_widths = []
        for i in range(n_cols):
            max_w = len(str(headers[i]))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    prefix = " " * indent
    hdr_line = prefix + "".join(
        str(h).ljust(w) for h, w in zip(headers, col_widths)
    )
    sep_line = prefix + "-" * sum(col_widths)
    body_lines = []
    for row in rows:
        body_lines.append(
            prefix + "".join(
                str(c).ljust(w) for c, w in zip(row, col_widths)
            )
        )

    return "\n".join([hdr_line, sep_line] + body_lines)


def _kv(key: str, value: Any, indent: int = 4) -> str:
    """Format a key-value pair."""
    return f"{' ' * indent}{key:30s}: {value}"


def _wrap(text: str, width: int = 74, indent: int = 4) -> str:
    """Word-wrap text with indentation."""
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )
    return wrapper.fill(text)


def _ff(v: float, decimals: int = 4) -> str:
    """Format a float."""
    return f"{v:.{decimals}f}"


# ===================================================================
# Preset profile resolver
# ===================================================================

_PRESET_MAP = {
    "financial": "FINANCIAL_ENERGY_SYSTEM",
    "epidemic": "EPIDEMIC_NETWORK",
    "supply_chain": "SUPPLY_CHAIN",
}


def _resolve_preset(preset_name: str):
    """Import and return a preset SystemProfile by short name."""
    from framework.aesc_profiler import (
        FINANCIAL_ENERGY_SYSTEM,
        EPIDEMIC_NETWORK,
        SUPPLY_CHAIN,
    )
    mapping = {
        "financial": FINANCIAL_ENERGY_SYSTEM,
        "epidemic": EPIDEMIC_NETWORK,
        "supply_chain": SUPPLY_CHAIN,
    }
    key = preset_name.lower().replace("-", "_")
    if key not in mapping:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Choose from: {', '.join(mapping.keys())}"
        )
    return mapping[key]


# ===================================================================
# 1. profile subcommand
# ===================================================================

def cmd_profile(args: argparse.Namespace) -> int:
    """Display a system profile."""
    from framework.aesc_profiler import (
        create_profile,
        profile_summary,
        profile_to_dict,
    )
    from framework.types import (
        Scale, Dynamics, NetworkType, AgentType,
        Feedback, DataRegime, Controllability, Observability,
    )

    if args.preset:
        profile = _resolve_preset(args.preset)
    else:
        # Build a custom profile from CLI arguments
        name = args.name or "Custom Profile"
        desc = args.description or "User-defined system profile"

        enum_lookup = {
            "scale": Scale,
            "dynamics": Dynamics,
            "network": NetworkType,
            "agents": AgentType,
            "feedback": Feedback,
            "data_regime": DataRegime,
            "controllability": Controllability,
            "observability": Observability,
        }

        kwargs: dict[str, Any] = {}
        for field, enum_cls in enum_lookup.items():
            raw = getattr(args, field, None)
            if raw is not None:
                try:
                    kwargs[field] = enum_cls(raw)
                except ValueError:
                    valid = [e.value for e in enum_cls]
                    print(f"Error: Invalid value '{raw}' for --{field}. "
                          f"Valid options: {valid}", file=sys.stderr)
                    return 1

        profile = create_profile(name, desc, **kwargs)

    # Print formatted profile
    print(_header("SYSTEM PROFILE"))
    d = profile_to_dict(profile)
    for key, val in d.items():
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val) if val else "(none)"
        print(_kv(key, val))

    print(_subheader("Summary"))
    print(_wrap(profile_summary(profile)))

    return 0


# ===================================================================
# 2. catalog subcommand
# ===================================================================

def cmd_catalog(args: argparse.Namespace) -> int:
    """List available methods from the catalog."""
    from framework.catalog import get_catalog, get_methods_by_role, get_methods_by_family
    from framework.types import Role, EpistemicFamily

    methods = get_catalog()

    # Apply filters
    if args.role:
        try:
            role = Role(args.role)
        except ValueError:
            valid = [r.value for r in Role]
            print(f"Error: Invalid role '{args.role}'. "
                  f"Valid options: {valid}", file=sys.stderr)
            return 1
        methods = [m for m in methods if role in m.roles_supported]

    if args.family:
        try:
            family = EpistemicFamily(args.family)
        except ValueError:
            valid = [f.value for f in EpistemicFamily]
            print(f"Error: Invalid family '{args.family}'. "
                  f"Valid options: {valid}", file=sys.stderr)
            return 1
        methods = [m for m in methods if m.family == family]

    print(_header("METHOD CATALOG"))

    if not methods:
        print("  No methods match the given filters.")
        return 0

    filter_info = []
    if args.role:
        filter_info.append(f"role={args.role}")
    if args.family:
        filter_info.append(f"family={args.family}")
    if filter_info:
        print(f"  Filters: {', '.join(filter_info)}")
    print(f"  Total methods: {len(methods)}")
    print()

    headers = ["Name", "Family", "Roles", "Data Req.", "Cost", "Interpret."]
    col_widths = [24, 26, 34, 18, 8, 12]
    rows = []
    for m in methods:
        roles = ", ".join(r.value for r in m.roles_supported)
        rows.append([
            m.name,
            m.family.value,
            roles,
            m.data_requirements,
            m.computational_cost,
            m.interpretability,
        ])
    print(_table(headers, rows, col_widths))

    # Detailed view
    if args.verbose:
        for m in methods:
            print(_subheader(m.name))
            print(_wrap(m.description))
            print(_kv("Assumptions", ", ".join(m.assumptions)))
            print(_kv("Strengths", ", ".join(m.strengths)))
            print(_kv("Weaknesses", ", ".join(m.weaknesses)))

    return 0


# ===================================================================
# 3. route subcommand
# ===================================================================

def cmd_route(args: argparse.Namespace) -> int:
    """Select a method kit for a given system profile."""
    from framework.config import RouterConfig
    from framework.router import select_kit, generate_decision_cards
    from framework.catalog import get_catalog

    profile = _resolve_preset(args.preset)
    catalog = get_catalog()

    config = RouterConfig(
        min_methods=args.min_methods,
        max_methods=args.max_methods,
        diversity_weight=args.diversity_weight,
    )

    selection = select_kit(profile, catalog=catalog, config=config)

    print(_header("ROUTER: METHOD KIT SELECTION"))
    print(_kv("System", profile.name))
    print(_kv("Min methods", config.min_methods))
    print(_kv("Max methods", config.max_methods))
    print(_kv("Diversity weight", _ff(config.diversity_weight, 2)))
    print(_kv("Total fit score", _ff(selection.total_fit, 4)))
    print(_kv("Avg. diversity", _ff(selection.avg_diversity, 4)))
    print()

    headers = ["#", "Method", "Family", "Fit Score", "Roles"]
    col_widths = [4, 24, 22, 12, 30]
    rows = []
    for i, m in enumerate(selection.selected_methods, 1):
        roles = ", ".join(r.value for r in m.roles_supported)
        rows.append([
            str(i),
            m.name,
            m.family.value,
            _ff(selection.fit_scores[m.name], 4),
            roles,
        ])
    print(_table(headers, rows, col_widths))

    # Justifications
    print(_subheader("Justifications"))
    for name, justification in selection.justifications.items():
        print(f"    [{name}]")
        print(_wrap(justification, indent=6))

    # Decision cards
    cards = generate_decision_cards(selection, profile)
    if cards:
        print(_subheader("Decision Cards"))
        for card in cards:
            print(f"    [{card['method']}]")
            print(_wrap(f"Why: {card['why_selected']}", indent=6))
            print(_wrap(f"Contribution: {card['contribution']}", indent=6))
            assumptions = ", ".join(card["key_assumptions"][:3])
            print(_wrap(f"Key assumptions: {assumptions}", indent=6))

    return 0


# ===================================================================
# 4. run subcommand
# ===================================================================

def cmd_run(args: argparse.Namespace) -> int:
    """Execute the full OS Multi-Science pipeline."""
    from framework.config import OSMultiScienceConfig, RouterConfig, ICMConfig
    from framework.icm import compute_icm_from_predictions
    from framework.types import ICMResult
    from orchestrator.pipeline import Pipeline
    from benchmarks.synthetic.generators import (
        generate_classification_benchmark,
        generate_multi_model_predictions,
    )
    from framework.aesc_profiler import profile_summary

    profile = _resolve_preset(args.preset)
    distance = args.distance

    print(_BANNER)
    print(_header("PIPELINE EXECUTION"))
    print(_kv("System", profile.name))
    print(_kv("N samples", args.n_samples))
    print(_kv("Distance metric", distance))
    print(_kv("Random seed", args.seed))
    print(_kv("Skip anti-spurious", args.skip_anti_spurious))
    print()

    # -- Configuration --
    config = OSMultiScienceConfig(
        icm=ICMConfig(
            C_A_wasserstein=10.0,
        ),
        random_seed=args.seed,
    )

    # -- Synthetic data generation --
    print("  [1/4] Generating synthetic data...")
    t0 = time.monotonic()
    X, y_true, centers = generate_classification_benchmark(
        n_samples=args.n_samples,
        n_classes=3,
        noise=0.1,
        seed=args.seed,
    )
    raw_preds = generate_multi_model_predictions(
        X, y_true,
        n_agreeing=3,
        n_disagreeing=1,
        seed=args.seed,
    )

    # Convert multi-class probability predictions to 1-D scores
    # (probability assigned to the true class) so the pipeline's CRC
    # and anti-spurious steps can work with them.  The ICM still sees
    # the full distributional shape via these per-sample scalars.
    n_classes = int(y_true.max()) + 1
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.float64)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0

    preds: dict[str, np.ndarray] = {}
    for name, prob_matrix in raw_preds.items():
        # Use the predicted probability of the true class as a 1-D score
        preds[name] = prob_matrix[np.arange(len(y_true)), y_true]

    # y_true for CRC: 1.0 for correct (ideal score), so loss = (pred - 1)^2
    y_true_scores = np.ones(len(y_true), dtype=np.float64)

    gen_time = time.monotonic() - t0
    print(f"         Data shape: X={X.shape}, y={y_true.shape}")
    print(f"         Models: {list(preds.keys())}")
    print(f"         Predictions: 1-D scores (P(true class))")
    print(f"         Generation time: {gen_time:.3f}s")

    # -- Build pipeline with user-specified distance metric --
    # Subclass Pipeline to use the CLI-specified distance function
    # instead of the hardcoded "wasserstein" in the base class.
    class CLIPipeline(Pipeline):
        def _step_icm(self, predictions):
            result = compute_icm_from_predictions(
                predictions,
                config=self.config.icm,
                distance_fn=distance,
            )
            if not np.isfinite(result.icm_score):
                result.icm_score = 0.5
            return result

    print("  [2/4] Registering model executors...")
    pipeline = CLIPipeline(config=config)

    for model_name, model_preds in preds.items():
        # Capture model_preds by default argument to avoid late binding
        def make_executor(p):
            def executor(data, prof):
                return p
            return executor
        pipeline.register_model(model_name, make_executor(model_preds))

    # -- Run pipeline --
    print("  [3/4] Running pipeline...")
    t0 = time.monotonic()
    result = pipeline.run(
        system_profile=profile,
        X=X,
        y_true=y_true_scores,
        features=X,
        skip_anti_spurious=args.skip_anti_spurious,
    )
    run_time = time.monotonic() - t0
    print(f"         Pipeline time: {run_time:.3f}s")

    # -- Results --
    print("  [4/4] Collecting results...")

    # Step results table
    print(_subheader("Pipeline Steps"))
    step_headers = ["Step", "Status", "Duration (s)", "Error"]
    step_rows = []
    for sr in result.step_results:
        err_str = sr.error[:50] + "..." if sr.error and len(sr.error) > 50 else (sr.error or "")
        step_rows.append([
            sr.step_name,
            sr.status.upper(),
            _ff(sr.duration_seconds, 4),
            err_str,
        ])
    print(_table(step_headers, step_rows, [22, 10, 14, 40]))

    # Routing results
    if result.selected_kit:
        print(_subheader("Selected Method Kit"))
        kit_headers = ["Method", "Family", "Fit"]
        kit_rows = []
        for m in result.selected_kit.selected_methods:
            kit_rows.append([
                m.name,
                m.family.value,
                _ff(result.selected_kit.fit_scores.get(m.name, 0), 4),
            ])
        print(_table(kit_headers, kit_rows, [24, 22, 10]))
        print(_kv("Total fit", _ff(result.selected_kit.total_fit, 4)))
        print(_kv("Avg. diversity", _ff(result.selected_kit.avg_diversity, 4)))

    # ICM results
    if result.icm_result:
        print(_subheader("ICM (Index of Convergence Multi-epistemic)"))
        icm = result.icm_result
        print(_kv("ICM Score", _ff(icm.icm_score, 4)))
        print(_kv("Convergence Level",
                   "HIGH" if icm.is_high_convergence
                   else "LOW" if icm.is_low_convergence
                   else "MODERATE"))
        print(_kv("Number of models", icm.n_models))
        print(_kv("Aggregation method", icm.aggregation_method))
        print()

        comp = icm.components
        comp_headers = ["Component", "Value", "Description"]
        comp_rows = [
            ["A (Agreement)", _ff(comp.A, 4), "Distributional agreement"],
            ["D (Direction)", _ff(comp.D, 4), "Direction/sign agreement"],
            ["U (Uncertainty)", _ff(comp.U, 4), "Uncertainty overlap"],
            ["C (Invariance)", _ff(comp.C, 4), "Stability under perturbation"],
            ["Pi (Dependency)", _ff(comp.Pi, 4), "Dependency penalty"],
        ]
        print(_table(comp_headers, comp_rows, [20, 10, 34]))

        if icm.weights:
            print()
            wt_headers = ["Weight", "Value"]
            wt_rows = [[k, _ff(v, 4)] for k, v in icm.weights.items()]
            print(_table(wt_headers, wt_rows, [12, 10]))

    # CRC results
    if result.crc_decision:
        print(_subheader("CRC (Conformal Risk Control) Gating"))
        print(_kv("Epistemic risk (Re)", _ff(result.re_score, 4)))
        print(_kv("Decision", result.crc_decision.value.upper()))

        if result.crc_decision.value == "act":
            print(_wrap(
                "Interpretation: High convergence -- the ensemble assessment "
                "can be used for decision-making with confidence."
            ))
        elif result.crc_decision.value == "defer":
            print(_wrap(
                "Interpretation: Moderate convergence -- the assessment should "
                "be reviewed by a human expert before acting."
            ))
        else:
            print(_wrap(
                "Interpretation: Low convergence -- models disagree "
                "significantly. A full audit is recommended."
            ))

    # Anti-spurious results
    if result.anti_spurious:
        print(_subheader("Anti-Spurious Validation"))
        asp = result.anti_spurious
        print(_kv("Baseline distance (D0)", _ff(asp.d0_baseline, 4)))
        print(_kv("Normalized convergence", _ff(asp.c_normalized, 4)))
        print(_kv("HSIC p-value", _ff(asp.hsic_pvalue, 4)))
        print(_kv("Convergence is genuine", str(asp.is_genuine)))
        print(_kv("FDR corrected", str(asp.fdr_corrected)))

        if asp.ablation_results:
            print()
            abl_headers = ["Model Removed", "ICM Change"]
            abl_rows = [[name, _ff(delta, 6)]
                        for name, delta in asp.ablation_results.items()]
            print(_table(abl_headers, abl_rows, [24, 14]))

    # Decision cards
    if result.decision_cards:
        print(_subheader("Decision Cards"))
        for card in result.decision_cards:
            print(f"    [{card['method']}]")
            print(_wrap(f"Why: {card['why_selected']}", indent=6))
            print(_wrap(f"Contribution: {card['contribution']}", indent=6))

    # Summary
    print(_subheader("Execution Summary"))
    print(_kv("Overall status", "SUCCESS" if result.is_success else "FAILED"))
    print(_kv("Total duration", f"{result.total_duration:.3f}s"))
    print(_kv("Steps completed",
              f"{sum(1 for s in result.step_results if s.status == 'success')}"
              f"/{len(result.step_results)}"))

    return 0 if result.is_success else 1


# ===================================================================
# 5. benchmark subcommand
# ===================================================================

def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run a real-world benchmark scenario."""
    scenario = args.scenario.lower().replace("-", "_")

    if scenario == "financial":
        print(_header("FINANCIAL SYSTEMIC RISK BENCHMARK"))
        print("  Loading financial benchmark...")
        print()
        from benchmarks.real_world.financial import run_benchmark
        run_benchmark()
    elif scenario == "epidemic":
        print(_header("EPIDEMIC SPREADING BENCHMARK"))
        print("  Loading epidemic benchmark...")
        print()
        from benchmarks.real_world.epidemic import run_epidemic_benchmark
        run_epidemic_benchmark()
    else:
        print(f"Error: Unknown scenario '{args.scenario}'. "
              f"Choose from: financial, epidemic", file=sys.stderr)
        return 1

    return 0


# ===================================================================
# 6. meta-learn subcommand
# ===================================================================

def cmd_meta_learn(args: argparse.Namespace) -> int:
    """Run the meta-learner to optimize ICM weights."""
    from framework.meta_learner import MetaLearner
    from framework.config import ICMConfig

    print(_header("META-LEARNER: ICM WEIGHT OPTIMIZATION"))

    config = ICMConfig()
    learner = MetaLearner(config)

    # Print defaults
    default_weights = {
        "w_A": config.w_A,
        "w_D": config.w_D,
        "w_U": config.w_U,
        "w_C": config.w_C,
        "lam": config.lam,
    }

    print(_subheader("Default Weights"))
    for k, v in default_weights.items():
        print(_kv(k, _ff(v, 4)))

    # Generate training scenarios
    print(f"\n  Generating {args.n_scenarios} training scenarios...")
    t0 = time.monotonic()
    scenarios = learner.generate_training_scenarios(
        n_scenarios=args.n_scenarios,
        seed=42,
    )
    gen_time = time.monotonic() - t0
    n_high = sum(1 for s in scenarios if s["label"] == 1)
    n_low = len(scenarios) - n_high
    print(f"  Generated {len(scenarios)} scenarios in {gen_time:.3f}s")
    print(f"  High convergence: {n_high}, Low convergence: {n_low}")

    # Run optimization
    print(f"\n  Running optimization (method: {args.method})...")
    t0 = time.monotonic()

    if args.method == "grid":
        result = learner.grid_search(
            scenarios,
            n_points=50,
            seed=42,
        )
    else:
        result = learner.optimize(
            scenarios,
            method="nelder-mead",
            n_restarts=5,
        )

    opt_time = time.monotonic() - t0
    print(f"  Optimization completed in {opt_time:.3f}s")

    # Print optimized weights
    print(_subheader("Optimized Weights"))
    opt_weights = result["best_weights"]
    for k in default_weights:
        default_v = default_weights[k]
        opt_v = opt_weights.get(k, default_v)
        delta = opt_v - default_v
        direction = "+" if delta > 0 else ""
        print(f"    {k:6s}: {_ff(default_v, 4)} -> {_ff(opt_v, 4)}  "
              f"({direction}{_ff(delta, 4)})")

    print()
    print(_kv("Best composite score", _ff(result["best_score"], 4)))

    # Compare with defaults
    print(_subheader("Comparison: Default vs Optimized"))
    comparison = learner.compare_with_default(scenarios, opt_weights)

    comp_headers = ["Metric", "Default", "Optimized", "Improvement"]
    comp_rows = []
    for metric in ["monotonicity", "coverage", "discrimination", "composite"]:
        dv = comparison["default_scores"][metric]
        ov = comparison["optimized_scores"][metric]
        imp = comparison["improvement_pct"][metric]
        comp_rows.append([
            metric.capitalize(),
            _ff(dv, 4),
            _ff(ov, 4),
            f"{imp:+.1f}%",
        ])
    print(_table(comp_headers, comp_rows, [18, 12, 12, 14]))

    # History summary
    print(_subheader("Optimization History"))
    print(_kv("Total evaluations", len(learner.history)))
    if learner.history:
        composites = [h["composite"] for h in learner.history]
        print(_kv("Min composite", _ff(min(composites), 4)))
        print(_kv("Max composite", _ff(max(composites), 4)))
        print(_kv("Mean composite", _ff(np.mean(composites), 4)))
        print(_kv("Std composite", _ff(np.std(composites), 4)))

    return 0


# ===================================================================
# Argument parser
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="osms",
        description=(
            "OS Multi-Science CLI -- Epistemic Operating System for "
            "Cross-Disciplinary Scientific Discovery"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python cli.py profile --preset financial
              python cli.py catalog --role structure --family statistical
              python cli.py route --preset financial --min-methods 3
              python cli.py run --preset financial --n-samples 500 --seed 42
              python cli.py benchmark --scenario financial
              python cli.py meta-learn --n-scenarios 20 --method grid
        """),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- profile ----
    p_profile = subparsers.add_parser(
        "profile",
        help="Display a system profile",
        description="Show detailed information about a system profile, "
                    "either from a preset or custom parameters.",
    )
    p_profile.add_argument(
        "--preset", choices=["financial", "epidemic", "supply_chain"],
        help="Use a preset profile (financial, epidemic, supply_chain)",
    )
    p_profile.add_argument("--name", help="Custom profile name")
    p_profile.add_argument("--description", help="Custom profile description")
    p_profile.add_argument(
        "--scale", choices=["micro", "meso", "macro", "multi_scale"],
        help="System scale",
    )
    p_profile.add_argument(
        "--dynamics", choices=["static", "slowly_evolving", "fast", "chaotic"],
        help="System dynamics",
    )
    p_profile.add_argument(
        "--network", choices=["none", "sparse", "dense", "multilayer"],
        help="Network type",
    )
    p_profile.add_argument(
        "--agents", choices=["none", "few", "many", "strategic"],
        help="Agent type",
    )
    p_profile.add_argument(
        "--feedback", choices=["weak", "moderate", "strong", "delayed", "nonlinear"],
        help="Feedback type",
    )
    p_profile.add_argument(
        "--data-regime", dest="data_regime",
        choices=["scarce", "moderate", "rich", "streaming"],
        help="Data regime",
    )
    p_profile.add_argument(
        "--controllability", choices=["none", "partial", "high"],
        help="Controllability level",
    )
    p_profile.add_argument(
        "--observability", choices=["inputs_only", "outputs_only", "partial", "detailed"],
        help="Observability level",
    )

    # ---- catalog ----
    p_catalog = subparsers.add_parser(
        "catalog",
        help="List available scientific methods",
        description="Display the method catalog with optional filtering "
                    "by epistemic role or method family.",
    )
    p_catalog.add_argument(
        "--role",
        help="Filter by epistemic role (structure, behavior, forecast, "
             "intervention, causal_identification)",
    )
    p_catalog.add_argument(
        "--family",
        help="Filter by epistemic family (statistical, machine_learning, "
             "network_science, agent_based, etc.)",
    )
    p_catalog.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed method descriptions",
    )

    # ---- route ----
    p_route = subparsers.add_parser(
        "route",
        help="Select a method kit for a system",
        description="Use the intelligent router to select a diverse, "
                    "well-fitting method kit for the given system profile.",
    )
    p_route.add_argument(
        "--preset", required=True,
        choices=["financial", "epidemic", "supply_chain"],
        help="System profile preset",
    )
    p_route.add_argument(
        "--min-methods", type=int, default=3,
        help="Minimum number of methods in kit (default: 3)",
    )
    p_route.add_argument(
        "--max-methods", type=int, default=7,
        help="Maximum number of methods in kit (default: 7)",
    )
    p_route.add_argument(
        "--diversity-weight", type=float, default=0.4,
        help="Weight for diversity vs fit in [0,1] (default: 0.4)",
    )

    # ---- run ----
    p_run = subparsers.add_parser(
        "run",
        help="Execute the full pipeline",
        description="Generate synthetic data, run the full OS Multi-Science "
                    "pipeline (Route -> Execute -> ICM -> CRC -> Anti-spurious "
                    "-> Decision Cards), and display comprehensive results.",
    )
    p_run.add_argument(
        "--preset", required=True,
        choices=["financial", "epidemic", "supply_chain"],
        help="System profile preset",
    )
    p_run.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of synthetic data samples (default: 500)",
    )
    p_run.add_argument(
        "--distance", choices=["hellinger", "wasserstein", "mmd"],
        default="hellinger",
        help="Distance metric for ICM agreement (default: hellinger)",
    )
    p_run.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p_run.add_argument(
        "--skip-anti-spurious", action="store_true",
        help="Skip the anti-spurious convergence check",
    )

    # ---- benchmark ----
    p_benchmark = subparsers.add_parser(
        "benchmark",
        help="Run a real-world benchmark scenario",
        description="Execute a full real-world benchmark with multiple "
                    "model simulators, ICM over time, early warning, "
                    "CRC gating, and knowledge graph recording.",
    )
    p_benchmark.add_argument(
        "--scenario", required=True,
        choices=["financial", "epidemic"],
        help="Benchmark scenario to run",
    )

    # ---- meta-learn ----
    p_meta = subparsers.add_parser(
        "meta-learn",
        help="Optimize ICM weights via meta-learning",
        description="Generate training scenarios and optimize the ICM "
                    "component weights (w_A, w_D, w_U, w_C, lambda) "
                    "to maximize a composite objective of monotonicity, "
                    "discrimination, and coverage.",
    )
    p_meta.add_argument(
        "--n-scenarios", type=int, default=20,
        help="Number of training scenarios to generate (default: 20)",
    )
    p_meta.add_argument(
        "--method", choices=["grid", "optimize"], default="grid",
        help="Optimization method: grid search or scipy optimize (default: grid)",
    )

    return parser


# ===================================================================
# Main entry point
# ===================================================================

_COMMAND_MAP = {
    "profile": cmd_profile,
    "catalog": cmd_catalog,
    "route": cmd_route,
    "run": cmd_run,
    "benchmark": cmd_benchmark,
    "meta-learn": cmd_meta_learn,
}


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handler = _COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
