#!/usr/bin/env python3
"""Build the complete manuscript-facing AUC_2-12 pairwise-test export.

This is a downstream analysis of committed per-repetition AUC values.  It does
not read or rerun active learning trajectories.  For the seven protocol-matched
Rh Published Split tasks, the three fixed partitions are averaged within each
repetition before the ten paired values are tested.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
ANALYSIS = SCRIPTS / "analysis"
for path in (SCRIPTS, ANALYSIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_tasks as fc  # noqa: E402
from inputs import analysis_unit_matrix, load_nesting  # noqa: E402
from stats import ALPHA, exact_wilcoxon_signed_rank, holm_adjust  # noqa: E402


OUTPUT = REPO / "supporting_information/data/pairwise_statistical_tests.csv"
METHODS = list(fc.ACQS)
METHOD_NAMES = {
    "epig_gauss": "EPIG",
    "bald": "BALD",
    "test_indv_nn": "Indiv. NN",
    "nn": "NN",
    "random": "Random",
}
METHOD_NAME_ORDER = [METHOD_NAMES[method] for method in METHODS]
PAIRS = list(combinations(METHODS, 2))
EXPECTED_DATASETS = {"BH": 44, "Rh": 38, "SAD": 7, "Wittig": 29}
EXPECTED_HEADLINE_COMPETITIVE = {"EPIG": 105, "BALD": 79}
SPLIT_FAMILY_NAMES = {
    "paper_defined": "Published Split",
    "chemistry_class_or_substrate_holdout": "Chemical Family Split",
    "cluster_based": "Cluster Split",
    "true_random": "Random Split",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_input_census(per_seed: pd.DataFrame, nesting: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "split_id",
        "split_family",
        "seed_id",
        "acq",
        fc.WINDOW_COL,
    }
    _require(required.issubset(per_seed.columns), f"missing input columns: {required - set(per_seed)}")
    _require(np.isfinite(per_seed[fc.WINDOW_COL].to_numpy(float)).all(), "non-finite AUC_2-12 input")

    tasks = (
        per_seed[["dataset", "split_id", "split_family"]]
        .drop_duplicates()
        .sort_values(["dataset", "split_id"], kind="stable")
        .reset_index(drop=True)
    )
    _require(len(tasks) == 118, f"expected 118 tasks, found {len(tasks)}")
    _require(
        tasks["dataset"].value_counts().to_dict() == EXPECTED_DATASETS,
        f"unexpected task census: {tasks['dataset'].value_counts().to_dict()}",
    )
    _require(
        not tasks.duplicated(["dataset", "split_id"]).any(),
        "a task has multiple split-family labels",
    )
    _require(
        set(tasks["split_family"]) == set(SPLIT_FAMILY_NAMES),
        f"unexpected split families: {set(tasks['split_family'])}",
    )

    rh_tasks = set(fc.RH_PROTOCOL_MATCHED_TASKS)
    _require(set(nesting["split_id"]) == rh_tasks, "Rh nesting map does not cover exactly seven tasks")
    for task in tasks.itertuples(index=False):
        group = per_seed[
            (per_seed["dataset"] == task.dataset) & (per_seed["split_id"] == task.split_id)
        ]
        _require(set(group["acq"]) == set(METHODS), f"{task.dataset}/{task.split_id}: incomplete methods")
        seed_sets = {
            method: tuple(sorted(method_rows["seed_id"].astype(int).tolist()))
            for method, method_rows in group.groupby("acq", sort=False)
        }
        _require(len(set(seed_sets.values())) == 1, f"{task.dataset}/{task.split_id}: unpaired repetitions")
        expected_n = 30 if task.split_id in rh_tasks else 10
        expected_ids = tuple(range(expected_n))
        _require(
            all(seed_ids == expected_ids for seed_ids in seed_sets.values()),
            f"{task.dataset}/{task.split_id}: expected repetition ids 0..{expected_n - 1}",
        )
    return tasks


def _task_matrix(
    per_seed: pd.DataFrame,
    task: object,
    nesting: pd.DataFrame,
) -> pd.DataFrame:
    rows = per_seed[
        (per_seed["dataset"] == task.dataset) & (per_seed["split_id"] == task.split_id)
    ]
    matrix = analysis_unit_matrix(rows, fc.WINDOW_COL, nesting, scheme="primary")
    _require(matrix.shape == (10, 5), f"{task.dataset}/{task.split_id}: analysis shape {matrix.shape}")
    _require(list(matrix.columns) == METHODS, f"{task.dataset}/{task.split_id}: method order changed")
    _require(np.isfinite(matrix.to_numpy(float)).all(), f"{task.dataset}/{task.split_id}: non-finite matrix")
    return matrix


def build_table(per_seed: pd.DataFrame, nesting: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rh_tasks = set(fc.RH_PROTOCOL_MATCHED_TASKS)

    for task in tasks.itertuples(index=False):
        matrix = _task_matrix(per_seed, task, nesting)
        means = {method: float(matrix[method].mean()) for method in METHODS}
        task_rows: list[dict[str, object]] = []
        raw_p: list[float] = []

        for method_1, method_2 in PAIRS:
            differences = matrix[method_1].to_numpy(float) - matrix[method_2].to_numpy(float)
            result = exact_wilcoxon_signed_rank(
                differences,
                rank_rounded=True,
                engine="dp",
            )
            mean_difference = float(np.mean(differences))
            if mean_difference > 0:
                favored = METHOD_NAMES[method_1]
            elif mean_difference < 0:
                favored = METHOD_NAMES[method_2]
            else:
                favored = "Tie"
            task_rows.append(
                {
                    "dataset": task.dataset,
                    "split_id": task.split_id,
                    "split_family": SPLIT_FAMILY_NAMES[task.split_family],
                    "method_1": METHOD_NAMES[method_1],
                    "method_2": METHOD_NAMES[method_2],
                    "n_paired_values": int(result["n_pairs"]),
                    "mean_auc_1": means[method_1],
                    "mean_auc_2": means[method_2],
                    "mean_difference": mean_difference,
                    "median_difference": float(np.median(differences)),
                    "n_nonzero_differences": int(result["n_nonzero"]),
                    "wilcoxon_statistic": float(min(result["w_plus"], result["w_minus"])),
                    "wilcoxon_w_plus": float(result["w_plus"]),
                    "wilcoxon_w_minus": float(result["w_minus"]),
                    "rank_biserial": float(result["rank_biserial"]),
                    "wilcoxon_p_raw": float(result["p_value"]),
                    "wilcoxon_p_holm": np.nan,
                    "significant_after_holm": False,
                    "favored_by_mean": favored,
                    "auc_metric": "AUC_2_12",
                    "paired_unit": (
                        "within-repetition mean across 3 fixed partitions"
                        if task.split_id in rh_tasks
                        else "repetition"
                    ),
                    "source_values_per_method": 30 if task.split_id in rh_tasks else 10,
                    "partitions_averaged_within_repetition": 3 if task.split_id in rh_tasks else 1,
                    "holm_family_size": 10,
                    "test_alternative": "two-sided",
                    "alpha": ALPHA,
                }
            )
            raw_p.append(float(result["p_value"]))

        adjusted = holm_adjust(raw_p)
        for row, adjusted_p in zip(task_rows, adjusted, strict=True):
            row["wilcoxon_p_holm"] = float(adjusted_p)
            row["significant_after_holm"] = bool(adjusted_p < ALPHA)
        rows.extend(task_rows)

    return pd.DataFrame(rows)


def _competitive_counts(table: pd.DataFrame) -> dict[str, int]:
    counts = {name: 0 for name in METHOD_NAME_ORDER}
    for (_dataset, _split_id), task in table.groupby(["dataset", "split_id"], sort=False):
        means: dict[str, float] = {}
        for row in task.itertuples(index=False):
            means[row.method_1] = float(row.mean_auc_1)
            means[row.method_2] = float(row.mean_auc_2)
        top_mean = max(means.values())
        highest = {name for name, value in means.items() if value == top_mean}
        top_method = min(highest, key=METHOD_NAME_ORDER.index)
        p_by_pair = {
            frozenset((row.method_1, row.method_2)): float(row.wilcoxon_p_holm)
            for row in task.itertuples(index=False)
        }
        for method in METHOD_NAME_ORDER:
            competitive = method in highest
            if not competitive:
                p_value = p_by_pair[frozenset((top_method, method))]
                competitive = not (means[method] < top_mean and p_value < ALPHA)
            counts[method] += int(competitive)
    return counts


def validate_table(table: pd.DataFrame) -> dict[str, int]:
    task_keys = ["dataset", "split_id"]
    _require(table[task_keys].drop_duplicates().shape[0] == 118, "output does not contain 118 tasks")
    _require(len(table) == 1180, f"expected 1,180 pairwise rows, found {len(table)}")
    sizes = table.groupby(task_keys, sort=False).size()
    _require(sizes.eq(10).all(), "not every task has exactly 10 pairwise rows")
    _require(
        not table.duplicated(task_keys + ["method_1", "method_2"]).any(),
        "duplicate task/method-pair row",
    )
    _require(table["n_paired_values"].eq(10).all(), "a test does not use 10 paired values")
    _require(table["holm_family_size"].eq(10).all(), "Holm family size is not 10")
    _require(
        set(table["split_family"]) == set(SPLIT_FAMILY_NAMES.values()),
        "output does not use the four manuscript-facing split-family names",
    )

    rh_mask = table["split_id"].isin(fc.RH_PROTOCOL_MATCHED_TASKS)
    _require(rh_mask.groupby(table["split_id"]).any().sum() == 7, "output does not contain seven Rh tasks")
    _require(
        table.loc[rh_mask, "source_values_per_method"].eq(30).all()
        and table.loc[rh_mask, "partitions_averaged_within_repetition"].eq(3).all(),
        "Rh rows do not record 30 nested values reduced to 10 repetition means",
    )
    _require(
        table.loc[~rh_mask, "source_values_per_method"].eq(10).all()
        and table.loc[~rh_mask, "partitions_averaged_within_repetition"].eq(1).all(),
        "ordinary tasks do not use 10 repetition-level values",
    )

    for key, task in table.groupby(task_keys, sort=False):
        expected = holm_adjust(task["wilcoxon_p_raw"].to_numpy(float))
        observed = task["wilcoxon_p_holm"].to_numpy(float)
        _require(
            np.array_equal(expected, observed),
            f"{key}: Holm values were not corrected within this task",
        )
        _require(
            np.array_equal(observed < ALPHA, task["significant_after_holm"].to_numpy(bool)),
            f"{key}: significance decisions do not match adjusted p-values",
        )

    competitive = _competitive_counts(table)
    for method, expected in EXPECTED_HEADLINE_COMPETITIVE.items():
        _require(
            competitive[method] == expected,
            f"headline discrepancy: {method} is competitive on {competitive[method]}/118, expected {expected}/118",
        )
    return competitive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed = fc.load_per_seed()
    nesting = load_nesting()
    tasks = _validate_input_census(per_seed, nesting)
    table = build_table(per_seed, nesting, tasks)
    competitive = validate_table(table)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.15g", lineterminator="\n")
    print("validated 118 tasks, 1,180 pairwise tests, and seven Rh partition-averaged tasks")
    print("statistically competitive: " + ", ".join(f"{name} {competitive[name]}/118" for name in METHOD_NAME_ORDER))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
