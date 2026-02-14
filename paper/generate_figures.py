#!/usr/bin/env python3
"""Generate all publication-quality figures for the ICM paper.

Usage:
    python paper/generate_figures.py

Produces 7 figures (PDF + PNG) in paper/figures/.

Requirements:
    - matplotlib, numpy, scipy, sklearn
    - The framework/ and benchmarks/ packages must be importable
      (run from the project root).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.datasets import load_dataset, list_classification_datasets
from benchmarks.model_zoo import (
    build_classification_zoo,
    train_zoo,
    collect_predictions_classification,
)
from framework.config import ICMConfig
from framework.icm import compute_icm_from_predictions, compute_icm
from framework.types import ICMComponents

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (Okabe-Ito inspired via tab10 selection)
COLORS = {
    "breast_cancer": "#1f77b4",  # blue
    "digits":        "#ff7f0e",  # orange
    "iris":          "#2ca02c",  # green
    "wine":          "#d62728",  # red
}
DATASET_LABELS = {
    "breast_cancer": "Breast Cancer",
    "digits":        "Digits",
    "iris":          "Iris",
    "wine":          "Wine",
}
DATASETS = ["breast_cancer", "digits", "iris", "wine"]

# Matplotlib global style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "text.usetex": False,
    "mathtext.fontset": "dejavuserif",
})


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF and PNG."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(str(pdf_path), format="pdf")
    fig.savefig(str(png_path), format="png")
    plt.close(fig)
    print(f"  Saved: {pdf_path.name}, {png_path.name}")


# ===================================================================
# BENCHMARK DATA (from reports/soa_benchmark_results.md)
# ===================================================================

# Experiment 4: ICM components per dataset
COMPONENTS = {
    "breast_cancer": {"A": 0.404096, "D": 0.899394, "U": 0.998233, "C": 0.93165,  "Pi": 0.0},
    "digits":        {"A": 0.358726, "D": 0.766726, "U": 0.813205, "C": 0.906094, "Pi": 0.0},
    "iris":          {"A": 0.573092, "D": 0.764378, "U": 0.797433, "C": 0.922882, "Pi": 0.0},
    "wine":          {"A": 0.478056, "D": 0.864044, "U": 0.887422, "C": 0.92229,  "Pi": 0.0},
}

# Experiment 5: Score range analysis
SCORE_RANGE = {
    "breast_cancer": {
        "Default Logistic": {"mean": 0.687106, "std": 0.0171036, "min": 0.636687,  "max": 0.700289, "range": 0.0636028},
        "Wide Range":       {"mean": 0.79049,  "std": 0.292553,  "min": 0.0463728, "max": 0.967313, "range": 0.92094},
        "Calibrated Beta":  {"mean": 0.687106, "std": 0.0171036, "min": 0.636687,  "max": 0.700289, "range": 0.0636028},
    },
    "digits": {
        "Default Logistic": {"mean": 0.678316, "std": 0.0174788, "min": 0.617966,  "max": 0.699293, "range": 0.0813276},
        "Wide Range":       {"mean": 0.772963, "std": 0.242479,  "min": 0.0642574, "max": 0.965168, "range": 0.900911},
        "Calibrated Beta":  {"mean": 0.678316, "std": 0.0174788, "min": 0.617966,  "max": 0.699293, "range": 0.0813276},
    },
    "iris": {
        "Default Logistic": {"mean": 0.669593, "std": 0.0066889, "min": 0.653938,  "max": 0.676362, "range": 0.0224244},
        "Wide Range":       {"mean": 0.748167, "std": 0.158184,  "min": 0.308321,  "max": 0.874773, "range": 0.566451},
        "Calibrated Beta":  {"mean": 0.669593, "std": 0.0066889, "min": 0.653938,  "max": 0.676362, "range": 0.0224244},
    },
    "wine": {
        "Default Logistic": {"mean": 0.676362, "std": 0.0109055, "min": 0.652099,  "max": 0.688001, "range": 0.0359026},
        "Wide Range":       {"mean": 0.776166, "std": 0.221418,  "min": 0.185483,  "max": 0.92276,  "range": 0.737277},
        "Calibrated Beta":  {"mean": 0.676362, "std": 0.0109055, "min": 0.652099,  "max": 0.688001, "range": 0.0359026},
    },
}

# Experiment 3: Error detection Spearman correlations
ERROR_DETECTION = {
    "breast_cancer": {
        "ICM (inverted)":         0.285392,
        "Ensemble Entropy":       0.286841,
        "Bootstrap Disagreement": 0.333693,
        "Max-Prob Uncertainty":   0.286841,
    },
    "digits": {
        "ICM (inverted)":         0.221967,
        "Ensemble Entropy":       0.245593,
        "Bootstrap Disagreement": 0.232197,
        "Max-Prob Uncertainty":   0.245079,
    },
    "iris": {
        "ICM (inverted)":         0.416851,
        "Ensemble Entropy":       0.416851,
        "Bootstrap Disagreement": 0.072796,
        "Max-Prob Uncertainty":   0.416851,
    },
    "wine": {
        "ICM (inverted)":         0.0,
        "Ensemble Entropy":       0.0,
        "Bootstrap Disagreement": 0.0,
        "Max-Prob Uncertainty":   0.0,
    },
}

# Decision gate breakdown (wide_range_preset) - original 4 datasets
DECISION_GATES = {
    "breast_cancer": {"ACT": 80.7, "DEFER":  5.3, "AUDIT": 14.0},
    "digits":        {"ACT": 68.9, "DEFER": 23.6, "AUDIT":  7.5},
    "iris":          {"ACT": 80.0, "DEFER": 20.0, "AUDIT":  0.0},
    "wine":          {"ACT": 80.6, "DEFER": 13.9, "AUDIT":  5.6},
}

# Decision gate breakdown for ALL 9 classification datasets (expanded)
DECISION_GATES_ALL = {
    "concept_drift":   {"ACT": 83.3, "DEFER":  8.3, "AUDIT":  8.3},
    "moons":           {"ACT": 82.5, "DEFER":  7.5, "AUDIT": 10.0},
    "breast_cancer":   {"ACT": 80.7, "DEFER":  5.3, "AUDIT": 14.0},
    "wine":            {"ACT": 80.6, "DEFER": 13.9, "AUDIT":  5.6},
    "iris":            {"ACT": 80.0, "DEFER": 20.0, "AUDIT":  0.0},
    "digits":          {"ACT": 68.9, "DEFER": 23.6, "AUDIT":  7.5},
    "covertype":       {"ACT": 45.8, "DEFER": 54.0, "AUDIT":  0.2},
    "circles":         {"ACT": 32.5, "DEFER": 66.0, "AUDIT":  1.5},
    "olivetti_faces":  {"ACT": 12.5, "DEFER": 46.2, "AUDIT": 41.2},
}
DATASETS_ALL = list(DECISION_GATES_ALL.keys())
DATASET_LABELS_ALL = {
    "concept_drift":  "Concept\nDrift",
    "moons":          "Moons",
    "breast_cancer":  "Breast\nCancer",
    "wine":           "Wine",
    "iris":           "Iris",
    "digits":         "Digits",
    "covertype":      "Cover-\ntype",
    "circles":        "Circles",
    "olivetti_faces": "Olivetti\nFaces",
}

# Experiment 1: Prediction quality (accuracy)
PREDICTION_QUALITY = {
    "breast_cancer": {
        "ICM-Weighted": 0.95614,  "Ensemble Avg": 0.964912,
        "Stacking (LR)": 0.973684, "Stacking (RF)": 0.947368,
        "Bootstrap": 0.964912,
    },
    "digits": {
        "ICM-Weighted": 0.983333, "Ensemble Avg": 0.975,
        "Stacking (LR)": 0.980556, "Stacking (RF)": 0.980556,
        "Bootstrap": 0.975,
    },
    "iris": {
        "ICM-Weighted": 0.933333, "Ensemble Avg": 0.933333,
        "Stacking (LR)": 0.966667, "Stacking (RF)": 0.966667,
        "Bootstrap": 0.933333,
    },
    "wine": {
        "ICM-Weighted": 0.972222, "Ensemble Avg": 1.0,
        "Stacking (LR)": 1.0, "Stacking (RF)": 0.972222,
        "Bootstrap": 1.0,
    },
}


# ===================================================================
# Figure 1: ICM Architecture Diagram
# ===================================================================
def figure1_architecture():
    """Schematic: K models -> 5 components -> sigma -> ICM -> CRC -> decisions."""
    print("Generating Figure 1: ICM Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    # Style helpers
    box_kw = dict(boxstyle="round,pad=0.3", linewidth=1.2)
    arrow_kw = dict(arrowstyle="-|>", color="#333333", linewidth=1.2,
                    connectionstyle="arc3,rad=0.0", mutation_scale=14)

    # --- Column 1: K Models ---
    model_x = 0.6
    model_labels = [r"$f_1$", r"$f_2$", r"$\vdots$", r"$f_K$"]
    model_ys = [3.6, 2.7, 1.8, 1.0]
    for label, y in zip(model_labels, model_ys):
        if label == r"$\vdots$":
            ax.text(model_x, y, label, ha="center", va="center", fontsize=14)
        else:
            bb = FancyBboxPatch((model_x - 0.45, y - 0.28), 0.9, 0.56,
                                facecolor="#E8EAF6", edgecolor="#3F51B5", **box_kw)
            ax.add_patch(bb)
            ax.text(model_x, y, label, ha="center", va="center", fontsize=12,
                    fontstyle="italic")

    ax.text(model_x, 4.3, r"$K$ Models ($\mathcal{F}$)", ha="center",
            va="center", fontsize=11, fontweight="bold")

    # --- Column 2: Five Components ---
    comp_x = 3.2
    comp_labels = [
        (r"$A$", "Agreement", "#C8E6C9", "#388E3C"),
        (r"$D$", "Direction",  "#BBDEFB", "#1976D2"),
        (r"$U$", "Uncertainty","#FFE0B2", "#F57C00"),
        (r"$C$", "Invariance", "#F8BBD0", "#C2185B"),
        (r"$\Pi$","Dependency", "#D7CCC8", "#5D4037"),
    ]
    comp_ys = [3.6, 2.8, 2.0, 1.2, 0.4]
    for (sym, lbl, fc, ec), y in zip(comp_labels, comp_ys):
        bb = FancyBboxPatch((comp_x - 0.7, y - 0.25), 1.4, 0.50,
                            facecolor=fc, edgecolor=ec, **box_kw)
        ax.add_patch(bb)
        ax.text(comp_x, y, f"{sym}  {lbl}", ha="center", va="center",
                fontsize=9.5, fontweight="bold")

    ax.text(comp_x, 4.3, "ICM Components", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Arrows: models -> components
    for my in [3.6, 2.7, 1.0]:
        for cy in comp_ys:
            ax.annotate("", xy=(comp_x - 0.75, cy), xytext=(model_x + 0.5, my),
                        arrowprops=dict(arrowstyle="-", color="#BDBDBD",
                                        linewidth=0.5, alpha=0.4))

    # --- Column 3: Aggregation sigma ---
    agg_x = 5.6
    agg_y = 2.0
    bb = FancyBboxPatch((agg_x - 0.55, agg_y - 0.35), 1.1, 0.70,
                        facecolor="#FFF9C4", edgecolor="#F9A825", **box_kw)
    ax.add_patch(bb)
    ax.text(agg_x, agg_y, r"$\sigma(\mathbf{w}^T\mathbf{c})$",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # Arrows: components -> sigma
    comp_arrow_kw = dict(arrowstyle="-|>", color="#9E9E9E", linewidth=0.8,
                         connectionstyle="arc3,rad=0.0", mutation_scale=14)
    for cy in comp_ys:
        ax.annotate("", xy=(agg_x - 0.6, agg_y), xytext=(comp_x + 0.75, cy),
                    arrowprops=comp_arrow_kw)

    # --- Column 4: ICM Score ---
    icm_x = 7.2
    icm_y = 2.0
    bb = FancyBboxPatch((icm_x - 0.55, icm_y - 0.35), 1.1, 0.70,
                        facecolor="#E1F5FE", edgecolor="#0288D1", **box_kw)
    ax.add_patch(bb)
    ax.text(icm_x, icm_y, "ICM\nScore", ha="center", va="center",
            fontsize=11, fontweight="bold", linespacing=1.1)

    # Arrow: sigma -> ICM
    ax.annotate("", xy=(icm_x - 0.6, icm_y), xytext=(agg_x + 0.6, agg_y),
                arrowprops=arrow_kw)

    # --- Column 5: CRC Gating ---
    crc_x = 8.7
    crc_y = 2.0
    bb = FancyBboxPatch((crc_x - 0.45, crc_y - 0.35), 0.9, 0.70,
                        facecolor="#F3E5F5", edgecolor="#7B1FA2", **box_kw)
    ax.add_patch(bb)
    ax.text(crc_x, crc_y, "CRC\nGating", ha="center", va="center",
            fontsize=10, fontweight="bold", linespacing=1.1)

    # Arrow: ICM -> CRC
    ax.annotate("", xy=(crc_x - 0.5, crc_y), xytext=(icm_x + 0.6, icm_y),
                arrowprops=arrow_kw)

    # --- Column 6: Decision Outputs ---
    dec_x = 10.0
    dec_data = [
        ("ACT",   3.2, "#C8E6C9", "#2E7D32"),
        ("DEFER", 2.0, "#FFF9C4", "#F57F17"),
        ("AUDIT", 0.8, "#FFCDD2", "#C62828"),
    ]
    for label, y, fc, ec in dec_data:
        bb = FancyBboxPatch((dec_x - 0.4, y - 0.25), 0.8, 0.50,
                            facecolor=fc, edgecolor=ec, **box_kw)
        ax.add_patch(bb)
        ax.text(dec_x, y, label, ha="center", va="center", fontsize=10,
                fontweight="bold")

    # Arrows: CRC -> decisions
    dec_arrow_kw = dict(arrowstyle="-|>", color="#333333", linewidth=1.2,
                        connectionstyle="arc3,rad=0.15", mutation_scale=14)
    for _, y, _, _ in dec_data:
        ax.annotate("", xy=(dec_x - 0.45, y), xytext=(crc_x + 0.5, crc_y),
                    arrowprops=dec_arrow_kw)

    # Threshold labels
    ax.text(9.25, 3.0, r"$\geq \tau_{hi}$", fontsize=8, ha="center",
            color="#2E7D32", fontstyle="italic")
    ax.text(9.4, 2.15, r"$\in [\tau_{lo}, \tau_{hi})$", fontsize=7.5,
            ha="center", color="#F57F17", fontstyle="italic")
    ax.text(9.25, 1.05, r"$< \tau_{lo}$", fontsize=8, ha="center",
            color="#C62828", fontstyle="italic")

    save_figure(fig, "fig1_architecture")


# ===================================================================
# Figure 2: Component Radar / Spider Plot
# ===================================================================
def figure2_components():
    """Radar chart showing 5 ICM components per dataset."""
    print("Generating Figure 2: Component Radar Plot...")

    categories = ["A\n(Agreement)", "D\n(Direction)", "U\n(Uncertainty)",
                  "C\n(Invariance)", r"$1-\Pi$" + "\n(Independence)"]
    N = len(categories)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for ds in DATASETS:
        c = COMPONENTS[ds]
        values = [c["A"], c["D"], c["U"], c["C"], 1.0 - c["Pi"]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=COLORS[ds],
                markersize=5, label=DATASET_LABELS[ds])
        ax.fill(angles, values, alpha=0.08, color=COLORS[ds])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8,
                       color="#666666")
    ax.set_rlabel_position(30)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color("#CCCCCC")
    ax.grid(True, color="#E0E0E0", linewidth=0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), framealpha=0.9,
              edgecolor="#CCCCCC")
    ax.set_title("ICM Component Profiles by Dataset", pad=25, fontsize=13,
                 fontweight="bold")

    save_figure(fig, "fig2_components")


# ===================================================================
# Figure 3: Score Range Comparison (Box-style range plots)
# ===================================================================
def figure3_score_range():
    """Range/spread comparison: Default Logistic vs Wide Range vs Calibrated Beta."""
    print("Generating Figure 3: Score Range Comparison...")

    configs = ["Default Logistic", "Wide Range", "Calibrated Beta"]
    config_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    config_hatches = ["", "", ""]

    fig, axes = plt.subplots(1, 4, figsize=(10, 4), sharey=True)

    for ax_idx, ds in enumerate(DATASETS):
        ax = axes[ax_idx]
        y_positions = np.arange(len(configs))

        for i, cfg in enumerate(configs):
            d = SCORE_RANGE[ds][cfg]
            # Draw range bar
            ax.barh(y_positions[i], d["range"], left=d["min"],
                    height=0.6, color=config_colors[i], alpha=0.7,
                    edgecolor="black", linewidth=0.8)
            # Mark mean
            ax.plot(d["mean"], y_positions[i], "D", color="white",
                    markeredgecolor="black", markersize=6, markeredgewidth=1.2,
                    zorder=5)

        ax.set_yticks(y_positions)
        if ax_idx == 0:
            ax.set_yticklabels(configs, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("ICM Score", fontsize=10)
        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
        ax.axvline(x=0.7, color="#999999", linestyle="--", linewidth=0.8,
                   alpha=0.7, label=r"$\tau_{hi}=0.7$" if ax_idx == 0 else None)
        ax.axvline(x=0.3, color="#999999", linestyle=":", linewidth=0.8,
                   alpha=0.7, label=r"$\tau_{lo}=0.3$" if ax_idx == 0 else None)

    # Add legend for threshold lines
    handles = [
        Line2D([0], [0], color="#999999", linestyle="--", linewidth=0.8),
        Line2D([0], [0], color="#999999", linestyle=":", linewidth=0.8),
        Line2D([0], [0], marker="D", color="white", markeredgecolor="black",
               markersize=6, markeredgewidth=1.2, linestyle="None"),
    ]
    labels = [r"$\tau_{hi}=0.7$", r"$\tau_{lo}=0.3$", "Mean"]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor="#CCCCCC")

    fig.suptitle("ICM Score Range by Aggregation Configuration", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig3_score_range")


# ===================================================================
# Figure 4: Error Detection Comparison
# ===================================================================
def figure4_error_detection():
    """Grouped bar chart of Spearman correlations across methods and datasets."""
    print("Generating Figure 4: Error Detection Comparison...")

    methods = ["ICM (inverted)", "Ensemble Entropy", "Bootstrap Disagreement",
               "Max-Prob Uncertainty"]
    method_short = ["ICM\n(inverted)", "Ensemble\nEntropy", "Bootstrap\nDisagree.",
                    "Max-Prob\nUncert."]
    method_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    n_methods = len(methods)
    n_datasets = len(DATASETS)
    x = np.arange(n_datasets)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, method in enumerate(methods):
        values = [ERROR_DETECTION[ds][method] for ds in DATASETS]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=method_short[i],
                      color=method_colors[i], edgecolor="white", linewidth=0.5)
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                        rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=10)
    ax.set_ylabel("Spearman Correlation", fontsize=11)
    ax.set_title("Error Detection: Spearman Correlation with Misclassification",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC", fontsize=9)
    ax.set_ylim(0, 0.52)
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, "fig4_error_detection")


# ===================================================================
# Figure 5: Decision Gate Breakdown
# ===================================================================
def figure5_decision_gates():
    """Stacked bar chart of ACT / DEFER / AUDIT for ALL 9 classification datasets."""
    print("Generating Figure 5: Decision Gate Breakdown (all datasets)...")

    gate_colors = {
        "ACT":   "#4CAF50",  # green
        "DEFER": "#FFC107",  # amber
        "AUDIT": "#F44336",  # red
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(DATASETS_ALL))
    width = 0.6

    act_vals   = [DECISION_GATES_ALL[ds]["ACT"] for ds in DATASETS_ALL]
    defer_vals = [DECISION_GATES_ALL[ds]["DEFER"] for ds in DATASETS_ALL]
    audit_vals = [DECISION_GATES_ALL[ds]["AUDIT"] for ds in DATASETS_ALL]

    bars_act = ax.bar(x, act_vals, width, label="ACT",
                      color=gate_colors["ACT"], edgecolor="white", linewidth=0.8)
    bars_defer = ax.bar(x, defer_vals, width, bottom=act_vals, label="DEFER",
                        color=gate_colors["DEFER"], edgecolor="white", linewidth=0.8)
    bars_audit = ax.bar(x, audit_vals, width,
                        bottom=[a + d for a, d in zip(act_vals, defer_vals)],
                        label="AUDIT", color=gate_colors["AUDIT"],
                        edgecolor="white", linewidth=0.8)

    # Add percentage labels inside or beside bars
    for bars, vals, bottoms, gate_name in [
        (bars_act, act_vals, [0] * len(DATASETS_ALL), "ACT"),
        (bars_defer, defer_vals, act_vals, "DEFER"),
        (bars_audit, audit_vals, [a + d for a, d in zip(act_vals, defer_vals)], "AUDIT"),
    ]:
        for bar, val, bot in zip(bars, vals, bottoms):
            if val >= 10:  # label inside if segment is wide enough
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
            elif val >= 3:  # label outside for narrow segments
                ax.text(bar.get_x() + bar.get_width() + 0.05,
                        bot + val / 2,
                        f"{val:.0f}%", ha="left", va="center",
                        fontsize=6.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS_ALL[ds] for ds in DATASETS_ALL], fontsize=9)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("CRC Decision Gate Breakdown: ICM Scales with Problem Difficulty",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC", fontsize=10)

    # Add difficulty arrow annotation
    ax.annotate("", xy=(8.3, -12), xytext=(-0.3, -12),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.5),
                annotation_clip=False)
    ax.text(4, -17, r"$\longleftarrow$ Easy                                    Hard $\longrightarrow$",
            ha="center", va="top", fontsize=9, color="#666666",
            fontstyle="italic")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    save_figure(fig, "fig5_decision_gates")


# ===================================================================
# Figure 6: Prediction Quality Comparison
# ===================================================================
def figure6_prediction_quality():
    """Grouped bar chart of accuracy across methods and datasets."""
    print("Generating Figure 6: Prediction Quality Comparison...")

    methods = ["ICM-Weighted", "Ensemble Avg", "Stacking (LR)",
               "Stacking (RF)", "Bootstrap"]
    method_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    n_methods = len(methods)
    n_datasets = len(DATASETS)
    x = np.arange(n_datasets)
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, method in enumerate(methods):
        values = [PREDICTION_QUALITY[ds][method] for ds in DATASETS]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=method,
                      color=method_colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Prediction Quality: Accuracy by Method", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0.88, 1.02)
    ax.legend(loc="lower left", framealpha=0.9, edgecolor="#CCCCCC",
              fontsize=9, ncol=2)

    # Add horizontal line at y=1.0
    ax.axhline(y=1.0, color="#CCCCCC", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    save_figure(fig, "fig6_prediction_quality")


# ===================================================================
# Figure 7: ICM Score vs Error Rate (computed from actual data)
# ===================================================================
def figure7_icm_vs_error():
    """Scatter plot: per-sample ICM score vs misclassification indicator.

    This figure generates real data by:
    1. Loading each classification dataset
    2. Training the model zoo
    3. Computing per-sample ICM scores using the wide-range preset
    4. Binning ICM scores and computing error rate per bin
    5. Plotting binned error rates against ICM score bins
    """
    print("Generating Figure 7: ICM Score vs Error Rate...")
    print("  (Training models on actual datasets -- this may take a moment)")

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.5), sharey=True)

    for ax_idx, ds_name in enumerate(DATASETS):
        ax = axes[ax_idx]

        # Load data and train models
        X_train, X_test, y_train, y_test = load_dataset(ds_name, seed=42)
        models = build_classification_zoo(seed=42)
        train_zoo(models, X_train, y_train)

        # Collect per-sample predictions
        preds = collect_predictions_classification(models, X_test)

        # Compute per-sample ICM scores with wide-range preset
        config_wr = ICMConfig.wide_range_preset()
        n_test = len(y_test)
        icm_scores = np.zeros(n_test)
        misclassified = np.zeros(n_test, dtype=bool)

        for i in range(n_test):
            # Build single-sample prediction dict (each model: shape (1, n_classes))
            sample_preds = {}
            for name, proba in preds.items():
                sample_preds[name] = proba[i:i+1, :]

            result = compute_icm_from_predictions(
                sample_preds, config=config_wr, distance_fn="hellinger"
            )
            icm_scores[i] = result.icm_score

            # Ensemble majority-vote prediction
            votes = []
            for name, proba in preds.items():
                votes.append(np.argmax(proba[i]))
            from collections import Counter
            vote_counts = Counter(votes)
            ensemble_pred = vote_counts.most_common(1)[0][0]
            misclassified[i] = (ensemble_pred != y_test[i])

        # Bin ICM scores and compute error rate per bin
        n_bins = 8
        bin_edges = np.linspace(icm_scores.min() - 0.001,
                                icm_scores.max() + 0.001, n_bins + 1)
        bin_centers = []
        error_rates = []
        bin_sizes = []

        for b in range(n_bins):
            mask = (icm_scores >= bin_edges[b]) & (icm_scores < bin_edges[b+1])
            count = mask.sum()
            if count >= 1:
                bin_centers.append((bin_edges[b] + bin_edges[b+1]) / 2)
                error_rates.append(misclassified[mask].mean())
                bin_sizes.append(count)

        bin_centers = np.array(bin_centers)
        error_rates = np.array(error_rates)
        bin_sizes = np.array(bin_sizes)

        # Scatter: size proportional to bin count
        sizes = np.clip(bin_sizes / bin_sizes.max() * 200, 30, 200)
        ax.scatter(bin_centers, error_rates, s=sizes, c=COLORS[ds_name],
                   alpha=0.8, edgecolors="black", linewidths=0.8, zorder=5)

        # Trend line (if enough points and non-trivial error)
        n_errors = misclassified.sum()
        if len(bin_centers) >= 3 and n_errors >= 2:
            z = np.polyfit(bin_centers, error_rates, 1)
            p = np.poly1d(z)
            x_line = np.linspace(bin_centers.min(), bin_centers.max(), 100)
            y_line = p(x_line)
            # Clip trend line to non-negative
            y_line = np.clip(y_line, 0, None)
            ax.plot(x_line, y_line, "--", color=COLORS[ds_name],
                    linewidth=1.5, alpha=0.7)

        ax.set_xlabel("ICM Score", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Error Rate", fontsize=11)
        ax.set_title(DATASET_LABELS[ds_name], fontsize=11, fontweight="bold")
        max_err = error_rates.max() if len(error_rates) > 0 else 0.0
        ax.set_ylim(-0.02, max(0.20, max_err * 1.3))
        ax.set_xlim(bin_edges[0] - 0.05, bin_edges[-1] + 0.05)

        # Add Spearman correlation annotation
        if len(bin_centers) >= 3:
            from scipy.stats import spearmanr
            n_errors = misclassified.sum()
            if n_errors == 0:
                # No errors: perfect accuracy, correlation is undefined
                ax.text(0.97, 0.97, "0 errors",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                              facecolor="white", edgecolor="#CCCCCC",
                                              alpha=0.9))
            else:
                rho, pval = spearmanr(icm_scores, misclassified.astype(float))
                if np.isnan(rho):
                    rho = 0.0
                ax.text(0.97, 0.97, f"$\\rho$={rho:.2f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                              facecolor="white", edgecolor="#CCCCCC",
                                              alpha=0.9))

    fig.suptitle("ICM Score vs. Ensemble Error Rate (Binned)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    save_figure(fig, "fig7_icm_vs_error")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("ICM Paper Figure Generator")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)
    print()

    figure1_architecture()
    print()
    figure2_components()
    print()
    figure3_score_range()
    print()
    figure4_error_detection()
    print()
    figure5_decision_gates()
    print()
    figure6_prediction_quality()
    print()
    figure7_icm_vs_error()

    print()
    print("=" * 60)
    print("All figures generated successfully.")
    generated = list(FIGURES_DIR.glob("*"))
    print(f"Total files: {len(generated)}")
    for f in sorted(generated):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
