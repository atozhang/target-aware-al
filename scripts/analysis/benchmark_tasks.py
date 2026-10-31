"""Shared loader for the 118-task benchmark (AUC 2-12% primary window).

The benchmark comprises 118 active learning tasks over four reaction datasets
(BH 44, Rh 38, SAD 7, Wittig 29). Rh has 20 random-target splits; the files
loaded here store ``tr_rh_010`` through ``tr_rh_019`` separately.
Seven Rh Published Split tasks use 30 nested paired replicates (three fixed
96/96 partitions x ten repetition seeds); every other task uses ten.

The Wittig transfer task (Published Split family) is stored in its own
long-format per-seed file and is appended to the benchmark at load time, so the
four datasets keep separate input tables.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PER_SEED = REPO / "results/benchmark_inputs/per_seed_window_auc.csv"
RH_RANDOM_SPLIT_PER_SEED = (
    REPO / "results/benchmark_inputs/rh_random_split_per_seed_window_auc.csv"
)
MEAN_CURVES = REPO / "results/benchmark_inputs/mean_curves_0_30.csv"
RH_RANDOM_SPLIT_CURVES = (
    REPO / "results/benchmark_inputs/rh_random_split_mean_curves_0_30.csv"
)
NATIVE_MEAN_CURVES = (
    REPO / "results/benchmark_inputs/native_method_mean_curves_0_30.csv.gz"
)
NATIVE_MEAN_CURVES_METADATA = (
    REPO / "results/benchmark_inputs/native_method_mean_curves_0_30.metadata.json"
)
LEDGER = REPO / "supporting_information/tables/benchmark_tasks.csv"
POOLS = REPO / "results/benchmark_inputs/task_pool_sizes.csv"
WITTIG_TRANSFER_WINDOW_SCORES = (
    REPO / "results/wittig_transfer/per_seed_window_scores.csv"
)
WITTIG_TRANSFER_CURVES = (
    REPO
    / "results/wittig_transfer/mean_curves_0_30.csv"
)

WITTIG_TRANSFER_TASK = "wittig_f119_textbook_base_cas_random_85_15"
WITTIG_TRANSFER_FAMILY = "paper_defined"
WITTIG_TRANSFER_METADATA = {
    "n_base": 734,
    "pool_total": 198,
    "n_target": 35,
    "repetitions": 10,
}

WINDOW_COL = "auc_2_12"
WITTIG_WINDOW_COLUMNS = {
    "0-10": "auc_0_10",
    "2-12": "auc_2_12",
    "5-15": "auc_5_15",
    "5-20": "auc_5_20",
    "5-30": "auc_5_30",
    "10-20": "auc_10_20",
    "10-30": "auc_10_30",
}
DELTA = 0.02
ACQS = ["epig_gauss", "bald", "test_indv_nn", "nn", "random"]
LABEL = {"epig_gauss": "EPIG", "bald": "BALD", "test_indv_nn": "IndvNN", "nn": "NN", "random": "Random"}
LABEL_ORDER = ["EPIG", "BALD", "IndvNN", "NN", "Random"]
RH_RANDOM_SPLIT_TASKS = [f"tr_rh_{i:03d}" for i in range(10, 20)]
RH_PROTOCOL_MATCHED_TASKS = [f"rh_paper_ood_case{i}" for i in range(1, 8)]

SCOPES = {"benchmark", "rh_random_split"}
NATIVE_PROTOCOL_VERSION = "native-integer-efficiency-v1"
NATIVE_GRID_RECONSTRUCTION_TOL = 5e-12


def _validate_scope(scope: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}")


def load_wittig_transfer_per_seed(
    window_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return the Wittig Published-Split per-seed rows.

    The committed source table is long over all seven acquisition windows (one
    row per seed x method x window). Callers may request the two primary-window
    columns or the complete seven-window view without reimplementing the pivot
    or its validation.
    """
    requested = tuple(window_columns or (WINDOW_COL, "auc_5_30"))
    unknown = set(requested) - set(WITTIG_WINDOW_COLUMNS.values())
    if unknown:
        raise ValueError(f"unknown Wittig transfer window columns: {sorted(unknown)}")

    source = pd.read_csv(WITTIG_TRANSFER_WINDOW_SCORES)
    required = {"dataset", "split_id", "split_family", "seed_id", "acq", "window", "auc"}
    if not required.issubset(source):
        raise ValueError(f"Wittig transfer scores lack {required - set(source)}")
    if len(source) != 350:
        raise ValueError(f"expected 350 Wittig transfer window rows, found {len(source)}")
    if not source["split_id"].eq(WITTIG_TRANSFER_TASK).all():
        raise ValueError("Wittig transfer scores carry an unexpected task identity")
    if not source["split_family"].eq(WITTIG_TRANSFER_FAMILY).all():
        raise ValueError("Wittig transfer scores carry an unexpected split family")
    if not source["dataset"].eq("Wittig").all():
        raise ValueError("Wittig transfer window scores contain another dataset")
    if set(source["acq"]) != set(ACQS):
        raise ValueError("Wittig transfer window scores do not contain all five methods")
    if set(source["seed_id"].astype(int)) != set(range(10)):
        raise ValueError("Wittig transfer window scores do not use paired seeds 0..9")
    if source.duplicated(["seed_id", "acq", "window"]).any():
        raise ValueError("duplicate Wittig transfer seed/method/window row")

    wide = source.pivot(
        index=["dataset", "split_id", "split_family", "seed_id", "acq"],
        columns="window",
        values="auc",
    )
    for window in WITTIG_WINDOW_COLUMNS:
        if window not in wide:
            raise ValueError(f"Wittig transfer window scores lack {window}")
    replacement = wide.reset_index().rename(columns=WITTIG_WINDOW_COLUMNS)
    replacement = replacement[
        ["dataset", "split_id", "split_family", "seed_id", "acq", *requested]
    ]
    if len(replacement) != 50:
        raise ValueError(f"expected 50 Wittig transfer per-seed rows, found {len(replacement)}")
    return replacement.sort_values(["seed_id", "acq"], kind="stable").reset_index(drop=True)


def _load_wittig_transfer_per_seed() -> pd.DataFrame:
    """Primary-window view used by ``load_per_seed``."""
    return load_wittig_transfer_per_seed()


def _load_wittig_transfer_curves() -> pd.DataFrame:
    replacement = pd.read_csv(WITTIG_TRANSFER_CURVES)
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
    missing = set(columns) - set(replacement)
    if missing:
        raise ValueError(f"Wittig transfer curves lack {missing}")
    if len(replacement) != 155:
        raise ValueError(f"expected 155 Wittig transfer curve rows, found {len(replacement)}")
    if not replacement["dataset"].eq("Wittig").all():
        raise ValueError("Wittig transfer curves contain another dataset")
    if not replacement["split_id"].eq(WITTIG_TRANSFER_TASK).all():
        raise ValueError("Wittig transfer curves use an unexpected task identity")
    if not replacement["split_family"].eq(WITTIG_TRANSFER_FAMILY).all():
        raise ValueError("Wittig transfer curves use an unexpected split family")
    if set(replacement["acq"]) != set(ACQS):
        raise ValueError("Wittig transfer curves do not contain all five methods")
    if not replacement["n_repetitions"].eq(10).all():
        raise ValueError("Wittig transfer curves do not record ten repetitions")
    sizes = replacement.groupby("acq").size()
    if not sizes.eq(31).all():
        raise ValueError("a Wittig transfer method does not contain 31 curve points")
    expected_grid = np.linspace(0.0, 0.30, 31)
    for method, group in replacement.groupby("acq", sort=False):
        observed = group.sort_values("acquisition_fraction")["acquisition_fraction"].to_numpy(float)
        if not np.array_equal(observed, expected_grid):
            raise ValueError(f"Wittig transfer {method} curve does not use the 0--30% grid")
    return replacement[columns].copy()


def expected_repetitions(split_id: str) -> int:
    return 30 if split_id in RH_PROTOCOL_MATCHED_TASKS else 10


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    d.update(path.read_bytes())
    return d.hexdigest()


def load_per_seed(scope: str = "benchmark") -> pd.DataFrame:
    """Per-replicate AUC rows for the 118-task benchmark, or the Rh random splits alone."""
    _validate_scope(scope)
    cols = ["dataset", "split_id", "split_family", "seed_id", "acq", WINDOW_COL, "auc_5_30"]
    rh_random = pd.read_csv(RH_RANDOM_SPLIT_PER_SEED)[cols]
    if scope == "rh_random_split":
        return rh_random.reset_index(drop=True)
    base = pd.read_csv(PER_SEED)[cols]
    wittig = _load_wittig_transfer_per_seed()[cols]
    return pd.concat([base, wittig, rh_random], ignore_index=True)


def load_mean_curves(scope: str = "benchmark") -> pd.DataFrame:
    _validate_scope(scope)
    rh_random = pd.read_csv(RH_RANDOM_SPLIT_CURVES)
    if scope == "rh_random_split":
        return rh_random
    base = pd.read_csv(MEAN_CURVES)
    wittig = _load_wittig_transfer_curves()[base.columns]
    return pd.concat([base, wittig, rh_random], ignore_index=True)


def load_native_mean_curves(verify_grid: bool = True) -> pd.DataFrame:
    """Load and validate the released per-acquisition method-level mean curves.

    The artifact contains only method-level target-set R2 means, not private
    source paths, raw reaction records, per-repetition trajectories, or
    acquisition identities. Ordinary tasks are equal means of ten repetitions.
    For the seven Rh Published-Split tasks, each repetition is first averaged
    over its three fixed partitions and the ten repetition means are then
    averaged equally.

    When ``verify_grid`` is true, linearly sampling each native mean curve at
    the retained 1%-of-pool grid must reproduce the separately released grid
    curve. This check validates consistency; the native endpoint itself never
    uses or infers values from that grid.
    """
    metadata = json.loads(NATIVE_MEAN_CURVES_METADATA.read_text())
    observed_sha = sha256(NATIVE_MEAN_CURVES)
    expected_sha = metadata.get("sha256")
    if observed_sha != expected_sha:
        raise ValueError(
            f"native mean-curve checksum mismatch: {observed_sha} != {expected_sha}"
        )

    curves = pd.read_csv(NATIVE_MEAN_CURVES)
    required = {
        "dataset", "split_id", "split_family", "acq", "AF", "pool_total",
        "budget_reactions", "n_acquired", "mean_r2", "weighting_scheme",
        "n_repetition_units", "n_partitions_per_repetition",
        "n_source_trajectories", "protocol_version",
    }
    if not required.issubset(curves):
        raise ValueError(f"native mean curves lack columns: {required - set(curves)}")
    if len(curves) != int(metadata.get("rows", -1)):
        raise ValueError(
            f"native mean curves have {len(curves)} rows, metadata records {metadata.get('rows')}"
        )
    if not curves["protocol_version"].eq(NATIVE_PROTOCOL_VERSION).all():
        raise ValueError("native mean curves carry an unexpected protocol version")
    if not np.isfinite(curves["mean_r2"].to_numpy(float)).all():
        raise ValueError("native mean curves contain non-finite R2 values")
    if curves.duplicated(["dataset", "split_id", "acq", "n_acquired"]).any():
        raise ValueError("native mean curves contain duplicate task/method/step rows")
    if set(curves["acq"]) != set(ACQS):
        raise ValueError("native mean curves do not contain exactly the five methods")
    if not curves["AF"].eq(curves["acq"].map(LABEL)).all():
        raise ValueError("native mean-curve method labels disagree with method identifiers")

    ledger = pd.read_csv(LEDGER)
    task_columns = ["dataset", "split_id", "split_family", "pool_total"]
    tasks = curves[task_columns].drop_duplicates()
    expected_tasks = ledger[task_columns]
    if len(tasks) != 118 or tasks.duplicated(["dataset", "split_id"]).any():
        raise ValueError(f"expected 118 uniquely labelled native tasks, found {len(tasks)}")
    if set(map(tuple, tasks.to_numpy())) != set(map(tuple, expected_tasks.to_numpy())):
        raise ValueError("native mean-curve task metadata disagree with the final task ledger")

    expected_budget = curves["pool_total"].map(lambda n: math.ceil(0.30 * int(n)))
    if not curves["budget_reactions"].eq(expected_budget).all():
        raise ValueError("native mean-curve budgets are not ceil(0.30 * pool_total)")
    groups = curves.groupby(["dataset", "split_id", "acq"], sort=False)
    if groups.ngroups != 118 * len(ACQS):
        raise ValueError(f"expected 590 native task/method curves, found {groups.ngroups}")
    for (dataset, split_id, acq), group in groups:
        group = group.sort_values("n_acquired")
        budget = int(group["budget_reactions"].iloc[0])
        steps = group["n_acquired"].to_numpy(int)
        if not np.array_equal(steps, np.arange(budget + 1)):
            raise ValueError(f"{dataset}/{split_id}/{acq}: steps are not every integer 0..{budget}")
        is_nested = split_id in RH_PROTOCOL_MATCHED_TASKS
        expected_weighting = (
            "mean_3_partitions_within_each_of_10_repetitions"
            if is_nested else "equal_mean_10_repetitions"
        )
        expected_partitions = 3 if is_nested else 1
        expected_sources = 30 if is_nested else 10
        if not group["weighting_scheme"].eq(expected_weighting).all():
            raise ValueError(f"{split_id}/{acq}: incorrect weighting metadata")
        if not group["n_repetition_units"].eq(10).all():
            raise ValueError(f"{split_id}/{acq}: expected ten repetition units")
        if not group["n_partitions_per_repetition"].eq(expected_partitions).all():
            raise ValueError(f"{split_id}/{acq}: incorrect partition-count metadata")
        if not group["n_source_trajectories"].eq(expected_sources).all():
            raise ValueError(f"{split_id}/{acq}: incorrect source-trajectory count")

    if verify_grid:
        grid = load_mean_curves()
        max_gap = 0.0
        for (split_id, acq), native in curves.groupby(["split_id", "acq"], sort=False):
            native = native.sort_values("n_acquired")
            released = grid[(grid["split_id"] == split_id) & (grid["acq"] == acq)].sort_values(
                "acquisition_fraction"
            )
            if len(released) != 31:
                raise ValueError(f"{split_id}/{acq}: expected 31 released-grid points")
            pool = int(native["pool_total"].iloc[0])
            rebuilt = np.interp(
                released["acquisition_fraction"].to_numpy(float) * pool,
                native["n_acquired"].to_numpy(float),
                native["mean_r2"].to_numpy(float),
            )
            gap = float(np.max(np.abs(rebuilt - released["mean_r2"].to_numpy(float))))
            max_gap = max(max_gap, gap)
        if max_gap > NATIVE_GRID_RECONSTRUCTION_TOL:
            raise ValueError(
                f"native curves do not reproduce the retained grid: max gap {max_gap:.3g}"
            )

    return curves


def task_means(per_seed: pd.DataFrame, window_col: str = WINDOW_COL) -> pd.DataFrame:
    """Wide per-task mean windowed AUC (one row per task, columns = methods)."""
    means = per_seed.groupby(["dataset", "split_id", "split_family", "acq"], as_index=False)[
        window_col
    ].mean()
    wide = means.pivot_table(
        index=["dataset", "split_id", "split_family"], columns="acq", values=window_col
    ).reset_index()
    return wide


def summary(scope: str = "benchmark") -> dict:
    """Derive benchmark dimensions and verify invariants; raise on any inconsistency."""
    per_seed = load_per_seed(scope)
    curves = load_mean_curves(scope)
    problems: list[str] = []

    # Identity agreement is checked independently of counts, so a one-for-one
    # task swap cannot pass silently.
    seed_metadata = per_seed[["dataset", "split_id", "split_family"]].drop_duplicates()
    curve_metadata = curves[["dataset", "split_id", "split_family"]].drop_duplicates()
    if seed_metadata.duplicated(["dataset", "split_id"]).any():
        problems.append("per-seed data assign multiple split families to a task")
    if curve_metadata.duplicated(["dataset", "split_id"]).any():
        problems.append("mean curves assign multiple split families to a task")
    seed_identities = set(map(tuple, seed_metadata.to_numpy()))
    curve_identities = set(map(tuple, curve_metadata.to_numpy()))
    if seed_identities != curve_identities:
        missing = sorted(seed_identities - curve_identities)
        extra = sorted(curve_identities - seed_identities)
        problems.append(f"per-seed/curve task identities disagree: missing={missing}, extra={extra}")

    ledger = pd.read_csv(LEDGER)
    ledger_metadata = ledger[["dataset", "split_id", "split_family"]].drop_duplicates()
    ledger_identities = set(map(tuple, ledger_metadata.to_numpy()))
    task_identities_match_ledger = scope != "benchmark" or seed_identities == ledger_identities
    if scope == "benchmark":
        present = set(per_seed["split_id"])
        curve_present = set(curves["split_id"])
        if WITTIG_TRANSFER_TASK not in present or WITTIG_TRANSFER_TASK not in curve_present:
            problems.append("Wittig Published-Split task is missing from the benchmark")
        if not task_identities_match_ledger:
            missing = sorted(ledger_identities - seed_identities)
            extra = sorted(seed_identities - ledger_identities)
            problems.append(f"task identities disagree with the benchmark task table: missing={missing}, extra={extra}")

        wittig_ledger = ledger[ledger["split_id"].eq(WITTIG_TRANSFER_TASK)]
        if len(wittig_ledger) != 1:
            problems.append("task table does not contain exactly one Wittig Published-Split task")
        else:
            row = wittig_ledger.iloc[0]
            expected_metadata = {
                "split_family": WITTIG_TRANSFER_FAMILY,
                **WITTIG_TRANSFER_METADATA,
            }
            for column, expected_value in expected_metadata.items():
                if row[column] != expected_value:
                    problems.append(
                        f"Wittig transfer task-table metadata {column}={row[column]!r}, expected {expected_value!r}"
                    )
        wittig_seed = per_seed[per_seed["split_id"].eq(WITTIG_TRANSFER_TASK)]
        wittig_curves = curves[curves["split_id"].eq(WITTIG_TRANSFER_TASK)]
        if not wittig_seed["split_family"].eq(WITTIG_TRANSFER_FAMILY).all():
            problems.append("Wittig transfer per-seed rows use the wrong split family")
        if set(wittig_seed["seed_id"].astype(int)) != set(range(10)):
            problems.append("Wittig transfer per-seed rows do not preserve seeds 0..9")
        if not wittig_curves["split_family"].eq(WITTIG_TRANSFER_FAMILY).all():
            problems.append("Wittig transfer mean curves use the wrong split family")
        if not wittig_curves["n_repetitions"].eq(WITTIG_TRANSFER_METADATA["repetitions"]).all():
            problems.append("Wittig transfer mean curves do not record ten repetitions")

    # Method / repetition completeness.
    per_task_acq = per_seed.groupby(["dataset", "split_id"])["acq"].apply(lambda s: set(s))
    for key, methods in per_task_acq.items():
        if methods != set(ACQS):
            problems.append(f"{key}: methods {sorted(methods)} != 5 expected")
    reps = per_seed.groupby(["dataset", "split_id", "acq"])["seed_id"].nunique()
    expected = pd.Series(
        [expected_repetitions(index[1]) for index in reps.index], index=reps.index
    )
    if not reps.eq(expected).all():
        bad = reps[~reps.eq(expected)]
        problems.append(f"{len(bad)} task/method groups have the wrong replicate count")
    for (dataset, split_id), group in per_seed.groupby(["dataset", "split_id"]):
        seed_sets = group.groupby("acq")["seed_id"].apply(
            lambda values: tuple(sorted(set(values)))
        )
        expected_seeds = tuple(range(expected_repetitions(str(split_id))))
        if len(set(seed_sets)) != 1 or seed_sets.iloc[0] != expected_seeds:
            problems.append(f"{dataset}/{split_id}: method seed sets are not identically matched")

    # Step-zero pairing: identical step-0 mean R2 across the five methods per task.
    z = curves[np.isclose(curves["acquisition_fraction"], 0.0)]
    spread = z.groupby(["dataset", "split_id"])["mean_r2"].agg(lambda s: float(s.max() - s.min()))
    unpaired = spread[spread > 1e-9]
    if len(unpaired):
        problems.append(f"{len(unpaired)} tasks fail step-0 pairing (mean R2 spread > 1e-9)")

    tasks = per_seed[["dataset", "split_id"]].drop_duplicates()
    n_tasks = len(tasks)

    # All ten Rh random-target splits must be present; no outcome-based filtering.
    if scope == "benchmark":
        present = set(per_seed["split_id"])
        missing = [t for t in RH_RANDOM_SPLIT_TASKS if t not in present]
        if missing:
            problems.append(f"Rh random-split tasks missing from the benchmark: {missing}")
    if scope != "rh_random_split":
        missing_protocol = [
            task for task in RH_PROTOCOL_MATCHED_TASKS if task not in set(per_seed["split_id"])
        ]
        if missing_protocol:
            problems.append(f"protocol-matched Rh tasks missing: {missing_protocol}")

    # Pool/target constant across methods: one manifest per task by construction;
    # cross-check ledger pool sizes are single-valued per task.
    dup_pool = ledger.groupby("split_id")["pool_total"].nunique()
    if (dup_pool > 1).any():
        problems.append("ledger has multiple pool sizes for a task")
    if scope != "rh_random_split":
        promoted = ledger[ledger["split_id"].isin(RH_PROTOCOL_MATCHED_TASKS)]
        if (len(promoted) != 7
                or not promoted["pool_total"].eq(96).all()
                or not promoted["n_target"].eq(96).all()
                or not promoted["repetitions"].eq(30).all()):
            problems.append(
                "task table does not record all seven Rh Published-Split tasks as "
                "96/96 with 30 replicates"
            )

    by_dataset = tasks["dataset"].value_counts().to_dict()
    fam = tasks.merge(
        per_seed[["dataset", "split_id", "split_family"]].drop_duplicates(),
        on=["dataset", "split_id"],
    )
    by_family = fam["split_family"].value_counts().to_dict()
    random_by_dataset = (
        fam[fam["split_family"] == "true_random"]["dataset"].value_counts().to_dict()
    )

    n_traj_expected = int(
        sum(expected_repetitions(split_id) * len(ACQS) for split_id in tasks["split_id"])
    )
    n_traj_completed = len(per_seed)

    result = {
        "scope": scope,
        "total_tasks": int(n_tasks),
        "tasks_by_dataset": {k: int(v) for k, v in sorted(by_dataset.items())},
        "tasks_by_split_family": {k: int(v) for k, v in sorted(by_family.items())},
        "random_target_by_dataset": {k: int(v) for k, v in sorted(random_by_dataset.items())},
        "methods_per_task": sorted({len(m) for m in per_task_acq}),
        "reps_per_task_method": sorted(reps.unique().tolist()),
        "replicate_design": {
            "retained_tasks": "10 matched repetitions",
            "protocol_matched_rh_tasks": (
                "3 fixed partitions x 10 matched repetitions = 30 nested replicates"
            ),
        },
        "trajectories_expected": int(n_traj_expected),
        "trajectories_completed": int(n_traj_completed),
        "step0_pairing_ok": bool(len(unpaired) == 0),
        "task_identities_match_curves": seed_identities == curve_identities,
        "task_identities_match_ledger": task_identities_match_ledger,
        "wittig_transfer_task_present": WITTIG_TRANSFER_TASK in set(per_seed["split_id"]),
        "rh_random_split_tasks_present": scope != "benchmark" or all(
            t in set(per_seed["split_id"]) for t in RH_RANDOM_SPLIT_TASKS
        ),
        "protocol_matched_rh_tasks_present": (
            all(task in set(per_seed["split_id"]) for task in RH_PROTOCOL_MATCHED_TASKS)
            if scope != "rh_random_split"
            else None
        ),
        "inputs_sha256": {
            "per_seed": sha256(PER_SEED),
            "rh_random_split_per_seed": sha256(RH_RANDOM_SPLIT_PER_SEED) if RH_RANDOM_SPLIT_PER_SEED.exists() else None,
            "mean_curves": sha256(MEAN_CURVES),
            "rh_random_split_mean_curves": sha256(RH_RANDOM_SPLIT_CURVES) if RH_RANDOM_SPLIT_CURVES.exists() else None,
            "wittig_transfer_window_scores": sha256(WITTIG_TRANSFER_WINDOW_SCORES),
            "wittig_transfer_mean_curves": sha256(WITTIG_TRANSFER_CURVES),
            "task_ledger": sha256(LEDGER),
        },
        "problems": problems,
    }
    if problems:
        raise ValueError(f"benchmark validation FAILED ({scope}): {problems}")
    return result
