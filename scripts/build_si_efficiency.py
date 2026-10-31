#!/usr/bin/env python3
"""Build the manuscript-facing 118-task experimental-efficiency export.

The source table, results/efficiency/efficiency_per_task.csv, is generated from
the released native integer-step method-mean curves by
scripts/efficiency_endpoint.py (first integer crossing at q = 0.80). This
script validates that table and packages it with reader-facing labels; it does
not recompute crossings, campaigns, or model outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results/efficiency/efficiency_per_task.csv"
OUTPUT = REPO / "supporting_information/data/experimental_efficiency.csv"

WITTIG_TRANSFER_TASK = "wittig_f119_textbook_base_cas_random_85_15"
EXPECTED_DATASETS = {"BH": 44, "Rh": 38, "SAD": 7, "Wittig": 29}
METHOD_NAMES = {
    "epig_gauss": "EPIG",
    "bald": "BALD",
    "test_indv_nn": "Indiv. NN",
    "nn": "NN",
    "random": "Random",
}
METHOD_ORDER = list(METHOD_NAMES)
SPLIT_FAMILY_NAMES = {
    "paper_defined": "Published Split",
    "chemistry_class_or_substrate_holdout": "Chemical Family Split",
    "cluster_based": "Cluster Split",
    "true_random": "Random Split",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_source(source: pd.DataFrame) -> None:
    required = {
        "dataset",
        "split_id",
        "split_family",
        "acq",
        "AF",
        "pool_total",
        "budget_reactions",
        "target_score",
        "attained",
        "full_budget",
        "n_reactions",
        "pct_pool",
    }
    _require(required.issubset(source), f"source lacks columns: {required - set(source)}")
    _require(len(source) == 590, f"expected 590 source rows, found {len(source)}")
    tasks = source[["dataset", "split_id", "split_family"]].drop_duplicates()
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
        "source does not contain the four expected split families",
    )
    sizes = source.groupby(["dataset", "split_id"], sort=False).size()
    _require(sizes.eq(5).all(), "not every task has five method rows")
    methods = source.groupby(["dataset", "split_id"], sort=False)["acq"].apply(set)
    _require(methods.apply(lambda value: value == set(METHOD_ORDER)).all(), "incomplete methods")
    _require(
        source["AF"].eq(source["acq"].map({**METHOD_NAMES, "test_indv_nn": "IndvNN"})).all(),
        "source method labels disagree with method identifiers",
    )
    _require(source["split_id"].eq(WITTIG_TRANSFER_TASK).sum() == 5, "Wittig Published-Split task missing")
    numeric = [
        "pool_total",
        "budget_reactions",
        "target_score",
        "n_reactions",
        "pct_pool",
    ]
    _require(np.isfinite(source[numeric].to_numpy(float)).all(), "non-finite numeric value")
    _require((source["attained"] == ~source["full_budget"]).all(), "attainment flags disagree")

    means = source.groupby("acq")["n_reactions"].mean()
    _require(np.isclose(means["epig_gauss"], 203.6271186440678), "EPIG overall mean changed")
    _require(np.isclose(means["bald"], 258.60169491525426), "BALD overall mean changed")
    wittig_means = source[source["dataset"].eq("Wittig")].groupby("acq")["n_reactions"].mean()
    _require(np.isclose(wittig_means["epig_gauss"], 106.37931034482759), "Wittig EPIG mean changed")
    _require(np.isclose(wittig_means["bald"], 137.9655172413793), "Wittig BALD mean changed")
    wittig = source[source["split_id"].eq(WITTIG_TRANSFER_TASK)].set_index("acq")
    expected_reactions = {
        "epig_gauss": 24,
        "bald": 48,
        "test_indv_nn": 44,
        "nn": 60,
        "random": 60,
    }
    _require(
        wittig["n_reactions"].to_dict() == expected_reactions,
        f"Wittig Published-Split reaction counts changed: {wittig['n_reactions'].to_dict()}",
    )


def build_table(source: pd.DataFrame) -> pd.DataFrame:
    output = source.copy()
    output["split_family"] = output["split_family"].map(SPLIT_FAMILY_NAMES)
    output["acquisition_function"] = output["acq"].map(METHOD_NAMES)
    output = output.rename(columns={"acq": "acquisition_function_id"})
    return output[
        [
            "dataset",
            "split_id",
            "split_family",
            "acquisition_function",
            "acquisition_function_id",
            "pool_total",
            "budget_reactions",
            "target_score",
            "attained",
            "full_budget",
            "n_reactions",
            "pct_pool",
        ]
    ]


def validate_output(output: pd.DataFrame) -> None:
    _require(len(output) == 590, "output does not contain 590 rows")
    _require(
        output[["dataset", "split_id"]].drop_duplicates().shape[0] == 118,
        "output does not contain 118 tasks",
    )
    _require(
        output.groupby(["dataset", "split_id"], sort=False).size().eq(5).all(),
        "output does not contain five methods per task",
    )
    _require(set(output["split_family"]) == set(SPLIT_FAMILY_NAMES.values()), "bad split labels")
    _require(set(output["acquisition_function"]) == set(METHOD_NAMES.values()), "bad method labels")
    _require(output["split_id"].eq(WITTIG_TRANSFER_TASK).sum() == 5, "Wittig Published-Split task missing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source.resolve())
    validate_source(source)
    output = build_table(source)
    validate_output(output)

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, float_format="%.15g", lineterminator="\n")
    validate_output(pd.read_csv(destination))
    print("validated 590 rows, 118 tasks, and five methods per task")
    print("overall mean reactions: EPIG 203.6; BALD 258.6")
    print("Wittig mean reactions: EPIG 106.4; BALD 138.0")
    print("Published Split reactions: 24 / 48 / 44 / 60 / 60")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
