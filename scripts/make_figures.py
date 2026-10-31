#!/usr/bin/env python
"""Generate repository figures from the released result tables.

Each panel is drawn from a committed table under ``results/`` or
``supporting_information/data/``. Run from the repository root:

    python scripts/make_figures.py

Outputs are written into ``figures/`` and are kept in the repository:

    wittig_transfer_curve.{pdf,png}             Fig 3D
    bh_representative_curves.{pdf,png}          Fig 4B
    statistical_competitiveness_by_dataset_and_split.{pdf,png}
                                                statistical-competitiveness breakdown
    learnability_random_splits.{pdf,png}        learnability sensitivity
    efficiency_threshold_sensitivity.{pdf,png}  efficiency-threshold sensitivity
    learning_curves/                            complete SI learning-curve pages (9 pages)

The nine SI learning-curve pages come from
``scripts/build_si_learning_curves.py``, which this script invokes.

These files are programmatic visualizations of the released data underlying
the manuscript figures. Standalone renderings may differ from the
hand-composed manuscript panels in layout, labels, or layering. This script
registers the bundled TeX Gyre Heros faces from ``figures/fonts/``, so its
output is font-deterministic on any machine.

Not drawn here, and why: Fig 1, Fig 2 and Fig 4A are schematics; Fig 3A and
Fig 3B are hand-drawn; Fig 3C and Fig 3E are UMAP maps composed with reaction
structures (data in ``results/wittig_qualitative_study/``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"
RESULTS = REPO / "results"
SI_DATA = REPO / "supporting_information/data"

ORDER = ["EPIG", "BALD", "IndvNN", "NN", "Random"]
DISPLAY = {"EPIG": "EPIG", "BALD": "BALD", "IndvNN": "Indiv. NN",
           "NN": "NN", "Random": "Random"}
COLORS = {
    "EPIG": "#117733",
    "BALD": "#4477AA",
    "IndvNN": "#AA3377",
    "NN": "#EE7733",
    "Random": "#707070",
}
ACQ_TO_LABEL = {"epig_gauss": "EPIG", "bald": "BALD", "test_indv_nn": "IndvNN",
                "nn": "NN", "random": "Random"}

FAMILY_ORDER = ["paper_defined", "chemistry_class_or_substrate_holdout",
                "cluster_based", "true_random"]
FAMILY_LABEL = {
    "paper_defined": "Published",
    "chemistry_class_or_substrate_holdout": "Chemical\nfamily",
    "cluster_based": "Cluster",
    "true_random": "Random",
}
DATASET_ORDER = ["BH", "Rh", "SAD", "Wittig"]
DATASET_LABEL = {"BH": "Buchwald–\nHartwig", "Rh": "Rh\nhydrogenation",
                 "SAD": "Sharpless\nAD", "Wittig": "Wittig"}

# Fig 4B: one representative BH task per split family.
BH_PANELS = [
    ("tr_bh_009", "Random split"),
    ("bh_paper_additive_plate3_extrapolation", "Published split"),
    ("bh_cluster_agglomerative_k8_cl4", "Cluster split"),
    ("bh_aryl_halide_holdout_14", "Chemical family split"),
]

PRIMARY_WINDOW = "2-12"


def register_style() -> str:
    """Register the bundled typeface and apply the shared figure style."""
    for font in sorted((FIGURES / "fonts").glob("**/*.[ot]tf")):
        font_manager.fontManager.addfont(str(font))
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = "TeX Gyre Heros" if "TeX Gyre Heros" in available else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E4E4E4",
        "grid.linewidth": 0.6,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return family


def method_handles():
    return [plt.Line2D([], [], color=COLORS[m], lw=2.0) for m in ORDER]


def save(fig, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix in (".pdf", ".png"):
        path = FIGURES / f"{stem}{suffix}"
        fig.savefig(path)
        out.append(path)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  Fig 3D -- Wittig textbook-to-CAS transfer curve
# --------------------------------------------------------------------------- #

def wittig_transfer_curve() -> list[Path]:
    data = pd.read_csv(RESULTS / "wittig_transfer/transfer_curve.csv")
    if set(data["AF"]) != set(ORDER):
        raise ValueError("transfer curve does not carry all five methods")

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for method in ORDER:
        g = data[data["AF"] == method].sort_values("acquisition_fraction")
        x = 100.0 * g["acquisition_fraction"].to_numpy(float)
        y = g["mean_r2"].to_numpy(float)
        sem = g["sem_r2"].to_numpy(float)
        ax.plot(x, y, color=COLORS[method], linewidth=1.5)
        ax.fill_between(x, y - sem, y + sem, color=COLORS[method],
                        alpha=0.13, linewidth=0)
    ax.set_xlabel("Acquired fraction of the CAS pool (%)")
    ax.set_ylabel("Target set $R^2$")
    ax.legend(method_handles(), [DISPLAY[m] for m in ORDER], loc="lower right",
              frameon=False, handlelength=1.4, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout()
    return save(fig, "wittig_transfer_curve")


# --------------------------------------------------------------------------- #
#  Fig 4B -- representative BH learning curves, one per split family
# --------------------------------------------------------------------------- #

def bh_representative_curves() -> list[Path]:
    curves = pd.read_csv(SI_DATA / "all_benchmark_learning_curves.csv")
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5), sharey=True)
    for ax, (split_id, title) in zip(axes, BH_PANELS):
        task = curves[curves["split_id"] == split_id]
        if task.empty:
            raise ValueError(f"no curve rows for {split_id}")
        for method in ORDER:
            g = (task[task["acquisition_function"] == DISPLAY[method]]
                 .sort_values("acquisition_percent"))
            x = g["acquisition_percent"].to_numpy(float)
            y = g["mean_target_r2"].to_numpy(float)
            sem = g["sem_target_r2"].to_numpy(float)
            ax.plot(x, y, color=COLORS[method], linewidth=1.4)
            ax.fill_between(x, y - sem, y + sem, color=COLORS[method],
                            alpha=0.13, linewidth=0)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Acquired pool (%)")
    axes[0].set_ylabel("Target set $R^2$")
    axes[-1].legend(method_handles(), [DISPLAY[m] for m in ORDER],
                    loc="lower right", frameon=False, handlelength=1.4,
                    handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout()
    return save(fig, "bh_representative_curves")


# --------------------------------------------------------------------------- #
#  Statistical competitiveness by dataset and split family
# --------------------------------------------------------------------------- #

def statistical_competitiveness_by_dataset_and_split() -> tuple[list[Path], dict]:
    cls = pd.read_csv(RESULTS / "benchmark/per_task_classification.csv")
    primary = cls[cls["window"] == PRIMARY_WINDOW]
    if primary.empty:
        raise ValueError(f"no rows at window {PRIMARY_WINDOW}")

    counts = {
        "highest_mean": primary.groupby("AF")["is_highest_mean"].sum().to_dict(),
        "competitive": primary.groupby("AF")["not_detected_worse"].sum().to_dict(),
    }

    def panel(ax, keys, labels, group_col, n_by_key):
        width = 0.16
        idx = np.arange(len(keys), dtype=float)
        for j, method in enumerate(ORDER):
            vals = []
            for k in keys:
                sub = primary[(primary[group_col] == k) & (primary["AF"] == method)]
                n = n_by_key.get(k, 0)
                vals.append(100.0 * float(sub["not_detected_worse"].sum()) / n
                            if n else np.nan)
            ax.bar(idx + (j - 2) * width, vals, width, color=COLORS[method],
                   linewidth=0)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 100)

    n_by_dataset = primary[primary["AF"] == "EPIG"].groupby("dataset").size().to_dict()
    n_by_family = primary[primary["AF"] == "EPIG"].groupby("split_family").size().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9))
    panel(axes[0], DATASET_ORDER, [DATASET_LABEL[d] for d in DATASET_ORDER],
          "dataset", n_by_dataset)
    axes[0].set_title("By reaction dataset", fontsize=8)
    axes[0].set_ylabel("Statistically competitive (% of tasks)")
    panel(axes[1], FAMILY_ORDER, [FAMILY_LABEL[f] for f in FAMILY_ORDER],
          "split_family", n_by_family)
    axes[1].set_title("By split family", fontsize=8)
    axes[1].legend(method_handles(), [DISPLAY[m] for m in ORDER],
                   loc="lower right", frameon=False, ncol=2, handlelength=1.4,
                   handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout()
    return save(fig, "statistical_competitiveness_by_dataset_and_split"), counts


# --------------------------------------------------------------------------- #
#  Learnability sensitivity across the 60 Random Split tasks
# --------------------------------------------------------------------------- #

def learnability_random_splits() -> tuple[list[Path], pd.DataFrame]:
    """x = task ceiling (best method mean AUC 2-12%), y = EPIG minus BALD."""
    frames = []
    for path in (RESULTS / "benchmark_inputs/per_seed_window_auc.csv",
                 RESULTS / "benchmark_inputs/rh_random_split_per_seed_window_auc.csv"):
        d = pd.read_csv(path)
        frames.append(d[d["split_family"] == "true_random"][
            ["dataset", "split_id", "acq", "auc_2_12"]])
    per_seed = pd.concat(frames, ignore_index=True)

    means = per_seed.groupby(["dataset", "split_id", "acq"], as_index=False)["auc_2_12"].mean()
    means["AF"] = means["acq"].map(ACQ_TO_LABEL)
    wide = means.pivot_table(index=["dataset", "split_id"], columns="AF",
                             values="auc_2_12").reset_index()
    wide["task_ceiling"] = wide[ORDER].max(axis=1)
    wide["epig_minus_bald"] = wide["EPIG"] - wide["BALD"]
    if len(wide) != 60:
        raise ValueError(f"expected 60 Random Split tasks, found {len(wide)}")

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    markers = {"BH": "o", "Rh": "s", "SAD": "^", "Wittig": "D"}
    palette = {"BH": "#4477AA", "Rh": "#117733", "SAD": "#AA3377", "Wittig": "#EE7733"}
    for dataset in DATASET_ORDER:
        g = wide[wide["dataset"] == dataset]
        if g.empty:
            continue
        ax.scatter(g["task_ceiling"], g["epig_minus_bald"], s=22,
                   marker=markers[dataset], facecolor=palette[dataset],
                   edgecolor="white", linewidth=0.5, label=dataset, zorder=3)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xlabel("Task ceiling: best method mean AUC$_{2-12}$")
    ax.set_ylabel("EPIG $-$ BALD mean AUC$_{2-12}$")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    return save(fig, "learnability_random_splits"), wide


# --------------------------------------------------------------------------- #
#  Efficiency-threshold sensitivity
# --------------------------------------------------------------------------- #

def efficiency_threshold_sensitivity() -> list[Path]:
    data = pd.read_csv(RESULTS / "sensitivity/target_level_counts.csv")
    expected_q = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    if sorted(data["q"].unique().tolist()) != expected_q:
        raise ValueError("unexpected target levels")
    if set(data["AF"]) != set(ORDER):
        raise ValueError("target-level counts do not carry all five methods")
    if set(data["n_tasks"]) != {118}:
        raise ValueError("target-level counts are not over 118 tasks")

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for method in ORDER:
        g = data[data["AF"] == method].sort_values("q")
        ax.plot(g["q"], g["attained"], color=COLORS[method], marker="o",
                markersize=3.4, linewidth=1.45)
    ax.axvline(0.80, color="#777777", linestyle="--", linewidth=0.8)
    ax.text(0.804, 6, "primary\n$q=0.80$", color="#555555", fontsize=7.5, va="bottom")
    ax.set_xlim(0.69, 0.96)
    ax.set_ylim(0, 110)
    ax.set_xticks(expected_q)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Target level $q$ (fraction of achievable improvement)")
    ax.set_ylabel("Tasks reaching the target within\nthe 30% budget (of 118)")
    ax.legend(method_handles(), [DISPLAY[m] for m in ORDER], loc="lower left",
              frameon=False, handlelength=1.4, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout()
    return save(fig, "efficiency_threshold_sensitivity")


# --------------------------------------------------------------------------- #
#  Complete SI learning-curve pages for the 118 tasks
# --------------------------------------------------------------------------- #

def si_learning_curve_pages() -> list[Path]:
    builder = REPO / "scripts/build_si_learning_curves.py"
    proc = subprocess.run([sys.executable, str(builder)], cwd=REPO,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"build_si_learning_curves.py failed:\n{proc.stderr}")
    return sorted((FIGURES / "learning_curves").glob("*.pdf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-learning-curves", action="store_true",
                        help="skip the nine SI learning-curve pages")
    args = parser.parse_args()

    family = register_style()
    print(f"typeface: {family}")

    written: list[Path] = []
    written += wittig_transfer_curve()
    written += bh_representative_curves()
    paths, counts = statistical_competitiveness_by_dataset_and_split()
    written += paths
    paths, learnability = learnability_random_splits()
    written += paths
    written += efficiency_threshold_sensitivity()
    if not args.skip_learning_curves:
        written += si_learning_curve_pages()

    print("\nFig 4C source counts at window 2-12 (of 118 tasks):")
    for method in ORDER:
        print(f"  {DISPLAY[method]:10s} highest mean "
              f"{int(counts['highest_mean'][method]):3d}"
              f"   statistically competitive "
              f"{int(counts['competitive'][method]):3d}")
    print(f"\nLearnability sensitivity: {len(learnability)} Random Split tasks "
          f"({learnability.groupby('dataset').size().to_dict()})")

    print(f"\nwrote {len(written)} files:")
    for p in written:
        print(f"  {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
