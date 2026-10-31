#!/usr/bin/env python3
"""Export and plot every learning curve in the 118-task census.

The script exports the learning curve data and generates the PDF plots from the
same CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
ANALYSIS = SCRIPTS / "analysis"
for _p in (ANALYSIS, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import benchmark_tasks as fc  # noqa: E402


OUTPUT = REPO / "supporting_information/data/all_benchmark_learning_curves.csv"
FIGURE_DIR = REPO / "figures/learning_curves"
WITTIG_TRANSFER_TASK = "wittig_f119_textbook_base_cas_random_85_15"
EXPECTED_DATASETS = {"BH": 44, "Rh": 38, "SAD": 7, "Wittig": 29}
DATASET_ORDER = list(EXPECTED_DATASETS)
DATASET_NAMES = {
    "BH": "Buchwald–Hartwig C–N coupling",
    "Rh": "Rh-catalyzed asymmetric olefin hydrogenation",
    "SAD": "Sharpless asymmetric dihydroxylation",
    "Wittig": "Wittig olefination",
}
METHOD_NAMES = {
    "epig_gauss": "EPIG",
    "bald": "BALD",
    "test_indv_nn": "Indiv. NN",
    "nn": "NN",
    "random": "Random",
}
METHOD_ORDER = list(METHOD_NAMES)
METHOD_COLORS = {
    "EPIG": "#117733",
    "BALD": "#4477AA",
    "Indiv. NN": "#AA3377",
    "NN": "#EE7733",
    "Random": "#707070",
}
SPLIT_FAMILY_NAMES = {
    "paper_defined": "Published Split",
    "chemistry_class_or_substrate_holdout": "Chemical Family Split",
    "cluster_based": "Cluster Split",
    "true_random": "Random Split",
}
SPLIT_FAMILY_ORDER = list(SPLIT_FAMILY_NAMES)
EXPECTED_GRID = np.linspace(0.0, 0.30, 31)
RH_PARTIAL_OOD_LABELS = {
    "rh_paper_ood_case1": "Partial OOD: SM1 + SM2 → SM3",
    "rh_paper_ood_case2": "Partial OOD: SM1 + SM3 → SM2",
    "rh_paper_ood_case3": "Partial OOD: SM2 + SM3 → SM1",
    "rh_paper_ood_case4": "Partial OOD: SM1 + SM2 + SM4 + SM5 → SM3",
    "rh_paper_ood_case5": "Partial OOD: SM1 + SM3 + SM4 + SM5 → SM2",
    "rh_paper_ood_case6": "Partial OOD: SM2 + SM3 + SM4 + SM5 → SM1",
    "rh_paper_ood_case7": "Partial OOD: SM3 + SM5 → SM4",
}
RH_LIGAND_LABELS = {
    "rh_ligand_class_holdout_P": "Monophosphine (P) ligands held out",
    "rh_ligand_class_holdout_PN": "Aminophosphine (PN) ligands held out",
    "rh_ligand_class_holdout_Phosphoramidite": "Phosphoramidite ligands held out",
}
SAD_LABELS = {
    "sad_train_test_Mono": "Monosubstituted alkenes",
    "sad_train_test_Gem": "gem-Disubstituted alkenes",
    "sad_train_test_Cis": "cis-Disubstituted alkenes",
    "sad_train_test_Trans": "trans-Disubstituted alkenes",
    "sad_train_test_TriQ2": "Trisubstituted alkenes (H in Q2)",
    "sad_train_test_TriQ4": "Trisubstituted alkenes (H in Q4)",
    "sad_train_test_Tetra": "Tetrasubstituted alkenes",
}
INTERNAL_LABEL_TOKENS = ("tr_", "f119", "paper_ood_case", "k8", "cl0", "85_15")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def configure_style() -> None:
    for pattern in ("Arial*.TTF", "Arial*.ttf", "arial*.ttf"):
        for path in (Path.home() / ".local/share/fonts/msttcorefonts").glob(pattern):
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 6.3,
            "axes.labelsize": 7,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_final_curves(curves: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "split_id",
        "split_family",
        "acq",
        "acquisition_fraction",
        "mean_r2",
        "sem_r2",
        "n_repetitions",
    }
    _require(required.issubset(curves), f"final curves lack columns: {required - set(curves)}")
    _require(len(curves) == 18_290, f"expected 18,290 curve rows, found {len(curves)}")
    tasks = curves[["dataset", "split_id", "split_family"]].drop_duplicates()
    _require(len(tasks) == 118, f"expected 118 tasks, found {len(tasks)}")
    _require(
        tasks["dataset"].value_counts().to_dict() == EXPECTED_DATASETS,
        f"unexpected dataset census: {tasks['dataset'].value_counts().to_dict()}",
    )
    _require(
        not tasks.duplicated(["dataset", "split_id"]).any(),
        "a task has multiple split families",
    )
    _require(
        set(tasks["split_family"]) == set(SPLIT_FAMILY_NAMES),
        "final curves do not contain the four expected split families",
    )
    _require(curves["split_id"].eq(WITTIG_TRANSFER_TASK).sum() == 155, "Wittig Published-Split task missing")
    sizes = curves.groupby(["dataset", "split_id", "acq"], sort=False).size()
    _require(sizes.eq(31).all(), "a task/method curve does not contain 31 points")
    methods = curves.groupby(["dataset", "split_id"], sort=False)["acq"].apply(set)
    _require(methods.apply(lambda value: value == set(METHOD_ORDER)).all(), "incomplete methods")
    _require(
        np.isfinite(curves[["acquisition_fraction", "mean_r2", "sem_r2"]].to_numpy(float)).all(),
        "non-finite curve value",
    )
    _require((curves["sem_r2"] >= 0).all(), "negative SEM value")
    _require((curves["n_repetitions"] > 0).all(), "non-positive repetition count")
    for key, group in curves.groupby(["dataset", "split_id", "acq"], sort=False):
        observed = group.sort_values("acquisition_fraction")["acquisition_fraction"].to_numpy(float)
        _require(np.array_equal(observed, EXPECTED_GRID), f"{key}: acquisition grid changed")

    wittig_source = pd.read_csv(fc.WITTIG_TRANSFER_CURVES)
    wittig_final = curves[curves["split_id"].eq(WITTIG_TRANSFER_TASK)].reset_index(drop=True)
    columns = [
        "dataset",
        "split_id",
        "split_family",
        "acq",
        "acquisition_fraction",
        "mean_r2",
        "sem_r2",
        "n_repetitions",
    ]
    _require(
        wittig_final[columns].equals(wittig_source[columns]),
        "Wittig curves differ from the committed source table",
    )
    return tasks


def build_table(curves: pd.DataFrame) -> pd.DataFrame:
    output = curves.copy()
    output["split_family"] = output["split_family"].map(SPLIT_FAMILY_NAMES)
    output["acquisition_function"] = output["acq"].map(METHOD_NAMES)
    output["acquisition_fraction"] = output["acquisition_fraction"].round(2)
    output["acquisition_percent"] = (
        100.0 * output["acquisition_fraction"]
    ).round().astype(int)
    output = output.rename(columns={"mean_r2": "mean_target_r2", "sem_r2": "sem_target_r2"})
    return output[
        [
            "dataset",
            "split_id",
            "split_family",
            "acquisition_function",
            "acquisition_fraction",
            "acquisition_percent",
            "mean_target_r2",
            "sem_target_r2",
            "n_repetitions",
        ]
    ]


def validate_output(output: pd.DataFrame) -> None:
    _require(len(output) == 18_290, "output does not contain 18,290 rows")
    tasks = output[["dataset", "split_id", "split_family"]].drop_duplicates()
    _require(len(tasks) == 118, "output does not contain 118 tasks")
    _require(tasks["dataset"].value_counts().to_dict() == EXPECTED_DATASETS, "bad dataset census")
    _require(
        output.groupby(["dataset", "split_id", "acquisition_function"], sort=False)
        .size()
        .eq(31)
        .all(),
        "output does not contain 31 points for every task/method",
    )
    _require(set(output["split_family"]) == set(SPLIT_FAMILY_NAMES.values()), "bad split labels")
    _require(set(output["acquisition_function"]) == set(METHOD_NAMES.values()), "bad method labels")
    _require(output["split_id"].eq(WITTIG_TRANSFER_TASK).sum() == 155, "Wittig Published-Split task missing")


def _balanced_page_sizes(total: int, maximum: int = 15) -> list[int]:
    pages = math.ceil(total / maximum)
    base, remainder = divmod(total, pages)
    return [base + int(index < remainder) for index in range(pages)]


def _numbered_match(split_id: str, pattern: str, label: str) -> str | None:
    match = re.fullmatch(pattern, split_id)
    return f"{label} {int(match.group(1))}" if match else None


def display_label(dataset: str, split_id: str) -> str:
    """Return a reader-facing title without changing the task ID."""
    if dataset == "BH":
        if split_id == "bh_paper_additive_plate3_extrapolation":
            return "Published additive extrapolation (Plate 3)"
        label = _numbered_match(
            split_id, r"bh_aryl_halide_holdout_(\d{2})", "Held-out aryl/heteroaryl halide"
        )
        label = label or _numbered_match(
            split_id, r"bh_cluster_agglomerative_k8_cl(\d+)", "Cluster"
        )
        label = label or _numbered_match(split_id, r"tr_bh_(\d{3})", "Random split")
    elif dataset == "Rh":
        label = RH_PARTIAL_OOD_LABELS.get(split_id) or RH_LIGAND_LABELS.get(split_id)
        label = label or _numbered_match(split_id, r"rh_cluster_k8_cl(\d+)", "Cluster")
        label = label or _numbered_match(split_id, r"tr_rh_(\d{3})", "Random split")
    elif dataset == "SAD":
        label = SAD_LABELS.get(split_id)
    elif dataset == "Wittig":
        if split_id == WITTIG_TRANSFER_TASK:
            return "CAS target-domain transfer"
        label = _numbered_match(split_id, r"wittig_f119_cluster_k8_cl(\d+)", "Cluster")
        label = label or _numbered_match(split_id, r"tr_wittig_(\d{3})", "Random split")
    else:
        label = None
    _require(label is not None, f"no display label for {dataset}/{split_id}")
    return label


def validate_display_labels(data: pd.DataFrame) -> dict[tuple[str, str], str]:
    tasks = data[["dataset", "split_id"]].drop_duplicates()
    labels = {
        (row.dataset, row.split_id): display_label(row.dataset, row.split_id)
        for row in tasks.itertuples(index=False)
    }
    _require(len(labels) == 118, f"expected 118 display labels, found {len(labels)}")
    _require(all(value.strip() for value in labels.values()), "an empty display label was generated")
    for dataset, group in tasks.groupby("dataset", sort=False):
        values = [labels[(dataset, split_id)] for split_id in group["split_id"]]
        _require(len(values) == len(set(values)), f"duplicate display labels in {dataset}")
    bad = {
        key: value
        for key, value in labels.items()
        if any(token in value.lower() for token in INTERNAL_LABEL_TOKENS)
    }
    _require(not bad, f"display labels expose internal tokens: {bad}")
    return labels


def _ordered_tasks(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    tasks = data.loc[data["dataset"].eq(dataset), ["split_id", "split_family"]].drop_duplicates()
    family_rank = {SPLIT_FAMILY_NAMES[value]: index for index, value in enumerate(SPLIT_FAMILY_ORDER)}
    tasks["family_rank"] = tasks["split_family"].map(family_rank)
    return tasks.sort_values(["family_rank", "split_id"], kind="stable").reset_index(drop=True)


def _plot_task(
    axis: plt.Axes,
    data: pd.DataFrame,
    dataset: str,
    split_id: str,
    display_labels: dict[tuple[str, str], str],
) -> None:
    task = data[(data["dataset"] == dataset) & (data["split_id"] == split_id)]
    lower_values: list[np.ndarray] = []
    upper_values: list[np.ndarray] = []
    for method_id in METHOD_ORDER:
        method = METHOD_NAMES[method_id]
        curve = task[task["acquisition_function"].eq(method)].sort_values("acquisition_fraction")
        x = curve["acquisition_percent"].to_numpy(float)
        mean = curve["mean_target_r2"].to_numpy(float)
        sem = curve["sem_target_r2"].to_numpy(float)
        lower = mean - sem
        upper = mean + sem
        lower_values.append(lower)
        upper_values.append(upper)
        axis.fill_between(x, lower, upper, color=METHOD_COLORS[method], alpha=0.10, linewidth=0)
        axis.plot(x, mean, color=METHOD_COLORS[method], linewidth=1.15, solid_capstyle="round")

    y_min = float(np.min(np.concatenate(lower_values)))
    y_max = float(np.max(np.concatenate(upper_values)))
    span = y_max - y_min
    padding = max(0.025, 0.055 * span)
    axis.set_xlim(0, 30)
    axis.set_ylim(y_min - padding, y_max + padding)
    axis.set_xticks([0, 10, 20, 30])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.3, pad=1.5)
    title = textwrap.fill(
        display_labels[(dataset, split_id)],
        width=34,
        break_long_words=False,
        break_on_hyphens=False,
    )
    axis.set_title(title, pad=2.5)


def render_pages(
    data: pd.DataFrame,
    destination: Path,
    preview_directory: Path | None = None,
) -> dict[str, list[Path]]:
    configure_style()
    destination.mkdir(parents=True, exist_ok=True)
    if preview_directory is not None:
        preview_directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}
    plotted: list[tuple[str, str]] = []
    display_labels = validate_display_labels(data)

    handles = [
        plt.Line2D([], [], color=METHOD_COLORS[METHOD_NAMES[value]], linewidth=1.6)
        for value in METHOD_ORDER
    ]
    labels = [METHOD_NAMES[value] for value in METHOD_ORDER]

    for dataset in DATASET_ORDER:
        tasks = _ordered_tasks(data, dataset)
        sizes = _balanced_page_sizes(len(tasks))
        pages: list[Path] = []
        start = 0
        for page_number, size in enumerate(sizes, start=1):
            page_tasks = tasks.iloc[start : start + size]
            ncols = 4 if size <= 8 else 5
            nrows = math.ceil(size / ncols)
            figure, axes = plt.subplots(nrows, ncols, figsize=(11.7, 8.3), squeeze=False)
            for axis, task in zip(axes.flat, page_tasks.itertuples(index=False), strict=False):
                _plot_task(axis, data, dataset, task.split_id, display_labels)
                plotted.append((dataset, task.split_id))
            for axis in axes.flat[size:]:
                axis.set_visible(False)

            figure.suptitle(
                f"Complete learning curves: {DATASET_NAMES[dataset]} "
                f"({page_number} of {len(sizes)})",
                fontsize=10,
                y=0.975,
            )
            figure.supxlabel("Acquisition pool acquired (%)", fontsize=8, y=0.055)
            figure.supylabel("Target-set $R^2$", fontsize=8, x=0.025)
            figure.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.012),
                ncol=5,
                frameon=False,
                handlelength=2.4,
                columnspacing=1.5,
            )
            figure.subplots_adjust(
                left=0.065,
                right=0.985,
                top=0.915,
                bottom=0.105,
                wspace=0.31,
                hspace=0.48,
            )
            path = destination / f"si_learning_curves_{dataset.lower()}_{page_number:02d}.pdf"
            figure.savefig(
                path,
                format="pdf",
                metadata={
                    "Title": f"{DATASET_NAMES[dataset]} learning curves, page {page_number}",
                    "Subject": "Complete benchmark learning curves through 30% acquisition",
                    "Creator": "scripts/build_si_learning_curves.py",
                },
            )
            if preview_directory is not None:
                figure.savefig(
                    preview_directory / f"{path.stem}_rendered.png",
                    format="png",
                    dpi=144,
                    facecolor="white",
                )
            plt.close(figure)
            pages.append(path)
            start += size
        written[dataset] = pages

    expected = set(
        map(
            tuple,
            data[["dataset", "split_id"]].drop_duplicates().to_numpy(),
        )
    )
    _require(len(plotted) == 118, f"plotted {len(plotted)} panels instead of 118")
    _require(len(set(plotted)) == 118, "a task was plotted more than once")
    _require(set(plotted) == expected, "the PDF pages do not cover the exported task census")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--qa-preview-dir",
        type=Path,
        help="optional directory for raster previews of the exact PDF page figures",
    )
    return parser.parse_args()


def write_table(output: pd.DataFrame, destination: Path) -> None:
    serialized = output.copy()
    serialized["acquisition_fraction"] = serialized["acquisition_fraction"].map(
        lambda value: f"{value:.2f}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized.to_csv(destination, index=False, float_format="%.17g", lineterminator="\n")

    with destination.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_fractions = {f"{value / 100:.2f}" for value in range(31)}
    expected_percentages = {str(value) for value in range(31)}
    _require(
        {row["acquisition_fraction"] for row in rows} == expected_fractions,
        "CSV acquisition fractions are not formatted as 0.00 through 0.30",
    )
    _require(
        {row["acquisition_percent"] for row in rows} == expected_percentages,
        "CSV acquisition percentages are not formatted as integers 0 through 30",
    )


def main() -> None:
    args = parse_args()
    fc.summary()
    curves = fc.load_mean_curves()
    validate_final_curves(curves)
    output = build_table(curves)
    validate_output(output)

    destination = args.output.resolve()
    write_table(output, destination)
    exported = pd.read_csv(destination, float_precision="round_trip")
    validate_output(exported)
    numeric_columns = [
        "acquisition_fraction",
        "acquisition_percent",
        "mean_target_r2",
        "sem_target_r2",
        "n_repetitions",
    ]
    _require(
        np.array_equal(output[numeric_columns].to_numpy(), exported[numeric_columns].to_numpy()),
        "CSV round trip changed a numerical curve value",
    )

    preview_directory = args.qa_preview_dir.resolve() if args.qa_preview_dir is not None else None
    pages = render_pages(exported, args.figure_dir.resolve(), preview_directory)
    print("validated 18,290 rows and 118 tasks (BH 44; Rh 38; SAD 7; Wittig 29)")
    print("validated five complete 0-30% curves per task and exact Wittig source values")
    for dataset in DATASET_ORDER:
        print(f"{dataset}: {EXPECTED_DATASETS[dataset]} panels across {len(pages[dataset])} PDF page(s)")
    print(f"wrote {destination}")
    for paths in pages.values():
        for path in paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
