#!/usr/bin/env python
"""
Load and verify inputs for the 118-task benchmark analysis.

The module loads data through ``scripts/analysis/benchmark_tasks.py``, verifies
that all five methods have identical ``seed_id`` sets within each task, records
input-file SHA-256 checksums, and calculates normalized AUC values for all seven
windows. It does not rerun or modify acquisition trajectories.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
for _p in (str(_HERE), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_tasks as fc  # noqa: E402  (the benchmark loader)
from paths import RH_RANDOM_SPLIT_PER_SEED  # noqa: E402  (repo root derived from __file__)

REPO = fc.REPO
ACQS = list(fc.ACQS)                 # ['epig_gauss','bald','test_indv_nn','nn','random']
LABEL = dict(fc.LABEL)
LABEL_ORDER = list(fc.LABEL_ORDER)   # ['EPIG','BALD','IndvNN','NN','Random']
PRIMARY_WINDOW_COL = fc.WINDOW_COL   # 'auc_2_12'
DELTA = fc.DELTA                     # 0.02

# The seven benchmark windows, in fraction-of-pool units.
WINDOWS = [
    ("auc_0_10", 0.00, 0.10, "0-10"),
    ("auc_2_12", 0.02, 0.12, "2-12"),
    ("auc_5_15", 0.05, 0.15, "5-15"),
    ("auc_5_20", 0.05, 0.20, "5-20"),
    ("auc_5_30", 0.05, 0.30, "5-30"),
    ("auc_10_20", 0.10, 0.20, "10-20"),
    ("auc_10_30", 0.10, 0.30, "10-30"),
]
WINDOW_COLS = [w[0] for w in WINDOWS]
WINDOW_BOUNDS = {w[0]: (w[1], w[2]) for w in WINDOWS}
WINDOW_LABEL = {w[0]: w[3] for w in WINDOWS}

INPUT_FILES = {
    "per_seed": fc.PER_SEED,
    "rh_random_split_per_seed": fc.RH_RANDOM_SPLIT_PER_SEED,
    "mean_curves": fc.MEAN_CURVES,
    "rh_random_split_mean_curves": fc.RH_RANDOM_SPLIT_CURVES,
    "wittig_transfer_window_scores": fc.WITTIG_TRANSFER_WINDOW_SCORES,
    "wittig_transfer_mean_curves": fc.WITTIG_TRANSFER_CURVES,
    "native_method_mean_curves": fc.NATIVE_MEAN_CURVES,
    "native_method_mean_curves_metadata": fc.NATIVE_MEAN_CURVES_METADATA,
    "task_ledger": fc.LEDGER,
    "task_pool_sizes": fc.POOLS,
    "efficiency_per_task": REPO / "results/efficiency/efficiency_per_task.csv",
}


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    d.update(Path(path).read_bytes())
    return d.hexdigest()


def input_shas() -> dict[str, dict[str, object]]:
    out = {}
    for name, p in INPUT_FILES.items():
        p = Path(p)
        out[name] = {
            "path": str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p),
            "exists": p.exists(),
            "sha256": sha256(p) if p.exists() else None,
        }
    return out


def normalized_auc(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Windowed-AUC estimator: mean linearly-interpolated R2 on [lo, hi].

    This is the definition the released per-seed window-AUC tables were built
    with, and the one every retained analysis uses.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.min() > lo + 1e-9 or x.max() < hi - 1e-9:
        return float("nan")
    grid = np.linspace(lo, hi, 500)
    values = np.interp(grid, x, y)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(values, grid) / (hi - lo))


def load_per_seed_full() -> pd.DataFrame:
    """Per-replicate rows for all 118 benchmark tasks with all seven windows.

    Replicate counts: 30 nested paired replicates (three fixed 96/96 partitions
    x ten repetitions) on the seven Rh Published-Split tasks, 10 on every other
    task. All three committed input tables carry all seven windows per replicate.
    """
    base_cols = ["dataset", "split_id", "split_family", "seed_id", "acq"]
    base = pd.read_csv(fc.PER_SEED)
    missing_base = [c for c in WINDOW_COLS if c not in base.columns]
    if missing_base:
        raise ValueError(f"per-seed file lacks windows {missing_base}")
    base = base[base_cols + WINDOW_COLS].copy()
    wittig = fc.load_wittig_transfer_per_seed(WINDOW_COLS)[base_cols + WINDOW_COLS]

    if not RH_RANDOM_SPLIT_PER_SEED.exists():
        raise FileNotFoundError(
            f"{RH_RANDOM_SPLIT_PER_SEED} not found; it carries the ten Rh "
            "random-split tasks with all seven windows per replicate"
        )
    rh_random = pd.read_csv(RH_RANDOM_SPLIT_PER_SEED)[base_cols + WINDOW_COLS].copy()

    return pd.concat([base, wittig, rh_random], ignore_index=True)


PARTITION_MAP = REPO / "results/rh_published_split/partition_map.csv"


def load_nesting() -> pd.DataFrame:
    """Authoritative seed_id -> (partition_id, repetition_seed) map for the seven
    protocol-matched Rh tasks, read from the committed partition map."""
    m = pd.read_csv(PARTITION_MAP)
    key = m.drop_duplicates(["split_id", "seed_id"])[
        ["split_id", "seed_id", "partition_id", "repetition_seed"]
    ].rename(columns={"split_id": "split_id",
                      "seed_id": "seed_id"})
    joined = m.rename(columns={"split_id": "split_id",
                               "seed_id": "seed_id"}).merge(
        key, on=["split_id", "seed_id"], suffixes=("", "_ref"))
    if not ((joined["partition_id"] == joined["partition_id_ref"]).all()
            and (joined["repetition_seed"] == joined["repetition_seed_ref"]).all()):
        raise ValueError("nesting map is inconsistent across methods")
    if sorted(key["partition_id"].unique()) != [0, 1, 2]:
        raise ValueError("expected exactly three partitions")
    if sorted(key["repetition_seed"].unique()) != list(range(10)):
        raise ValueError("expected ten repetition seeds")
    return key


def analysis_unit_matrix(task_rows: pd.DataFrame, window_col: str,
                         nesting: pd.DataFrame, scheme: str = "primary") -> pd.DataFrame:
    """units x methods matrix on the PRIMARY analysis unit for this task.

    The seven protocol-matched Rh tasks carry 30 nested replicates: three fixed
    96/96 partitions x ten matched repetition seeds. Those 30 observations are NOT
    exchangeable, so the primary unit averages the three partition-specific values
    within each repetition seed and tests the ten resulting seed-level values.
    Every other task already has ten independent matched repetitions and is
    unchanged.

    scheme:
      "primary"       ten seed-level units on the Rh tasks, ten elsewhere
      "pooled_30"     all 30 nested replicates as units (SI sensitivity)
      "partition_0/1/2"  one 96/96 partition alone, ten repetitions (SI sensitivity)
    """
    sid = str(task_rows["split_id"].iloc[0])
    nested = sid in set(nesting["split_id"])
    if not nested:
        if scheme.startswith("partition_") or scheme in ("primary", "pooled_30"):
            return task_rows.pivot_table(index="seed_id", columns="acq",
                                         values=window_col)[ACQS].sort_index()
        raise ValueError(f"unknown scheme {scheme!r}")

    rows = task_rows.merge(nesting[nesting.split_id == sid], on=["split_id", "seed_id"],
                           how="left")
    if rows["partition_id"].isna().any():
        raise ValueError(f"{sid}: nesting map does not cover every replicate")

    if scheme == "primary":
        counts = rows.pivot_table(index="repetition_seed", columns="acq",
                                  values=window_col, aggfunc="size")[ACQS]
        if not (counts.to_numpy() == 3).all():
            raise ValueError(f"{sid}: a seed unit does not average exactly 3 partitions")
        return rows.pivot_table(index="repetition_seed", columns="acq",
                                values=window_col, aggfunc="mean")[ACQS].sort_index()
    if scheme == "pooled_30":
        return rows.pivot_table(index="seed_id", columns="acq",
                                values=window_col)[ACQS].sort_index()
    if scheme.startswith("partition_"):
        p = int(scheme.split("_")[1])
        sub = rows[rows.partition_id == p]
        return sub.pivot_table(index="repetition_seed", columns="acq",
                               values=window_col)[ACQS].sort_index()
    raise ValueError(f"unknown scheme {scheme!r}")


def load_mean_curves() -> pd.DataFrame:
    return fc.load_mean_curves()


def load_pools() -> pd.DataFrame:
    pools = pd.read_csv(fc.POOLS)
    # the ten Rh random-target splits share a 672-reaction pool and are
    # appended here so every task has a pool size.
    have = set(pools["split_id"])
    ext_rows = pd.DataFrame({
        "dataset": "Rh",
        "split_id": [t for t in fc.RH_RANDOM_SPLIT_TASKS if t not in have],
        "split_family": "true_random",
        "pool_total": 672,
    })
    if len(ext_rows):
        pools = pd.concat([pools, ext_rows], ignore_index=True)
    return pools


def verify_pairing(per_seed: pd.DataFrame) -> dict:
    """Fail loudly unless every task has all five methods paired on an identical
    seed_id set. Runs the benchmark validation first, then the stricter check.
    Returns a summary dict on success; raises ValueError otherwise.
    """
    census = fc.summary()  # raises on 5-method / 10-rep / step-0 / pool problems

    problems: list[str] = []
    identity_cols = ["dataset", "split_id", "split_family"]
    actual_metadata = per_seed[identity_cols].drop_duplicates()
    if actual_metadata.duplicated(["dataset", "split_id"]).any():
        problems.append("analyzed per-seed data assign multiple split families to a task")

    expected_metadata = fc.load_per_seed()[identity_cols].drop_duplicates()
    ledger_metadata = pd.read_csv(fc.LEDGER)[identity_cols].drop_duplicates()
    actual_identities = set(map(tuple, actual_metadata.to_numpy()))
    expected_identities = set(map(tuple, expected_metadata.to_numpy()))
    ledger_identities = set(map(tuple, ledger_metadata.to_numpy()))
    if actual_identities != expected_identities:
        missing = sorted(expected_identities - actual_identities)
        extra = sorted(actual_identities - expected_identities)
        problems.append(
            "analyzed task identities disagree with the benchmark loader: "
            f"missing={missing}, extra={extra}"
        )
    if actual_identities != ledger_identities:
        missing = sorted(ledger_identities - actual_identities)
        extra = sorted(actual_identities - ledger_identities)
        problems.append(
            "analyzed task identities disagree with manuscript ledger: "
            f"missing={missing}, extra={extra}"
        )
    present = set(per_seed["split_id"])
    if fc.WITTIG_TRANSFER_TASK not in present:
        problems.append("Wittig Published-Split task is missing from analyzed data")

    seed_counts: dict[int, int] = {}
    for (dataset, split_id), g in per_seed.groupby(["dataset", "split_id"], sort=False):
        seed_sets = {acq: frozenset(sub["seed_id"].tolist())
                     for acq, sub in g.groupby("acq")}
        if set(seed_sets) != set(ACQS):
            problems.append(f"{dataset}/{split_id}: methods {sorted(seed_sets)} != 5 expected")
            continue
        ref = seed_sets[ACQS[0]]
        for acq in ACQS[1:]:
            if seed_sets[acq] != ref:
                problems.append(
                    f"{dataset}/{split_id}: seed_id set for {acq} differs from {ACQS[0]} "
                    f"(symmetric diff {sorted(ref ^ seed_sets[acq])})"
                )
        # duplicate seed rows within a method would corrupt pairing
        for acq, sub in g.groupby("acq"):
            if sub["seed_id"].duplicated().any():
                problems.append(f"{dataset}/{split_id}/{acq}: duplicate seed_id rows")
        # replicate count must equal the expected count for this task
        # (30 for the seven protocol-matched Rh tasks, 10 otherwise) and the
        # matched replicate ids must be exactly 0..n-1 across all five methods
        n = len(ref)
        expected_n = fc.expected_repetitions(str(split_id))
        if n != expected_n:
            problems.append(
                f"{dataset}/{split_id}: {n} matched replicates, expected {expected_n}")
        elif tuple(sorted(ref)) != tuple(range(expected_n)):
            problems.append(
                f"{dataset}/{split_id}: replicate ids are not 0..{expected_n - 1}")
        seed_counts[n] = seed_counts.get(n, 0) + 1

    if problems:
        raise ValueError("pairing verification FAILED: " + "; ".join(problems[:20]))

    return {
        "census_audit": census,
        "n_tasks": int(per_seed[["dataset", "split_id"]].drop_duplicates().shape[0]),
        "task_identities_match_benchmark_loader": actual_identities == expected_identities,
        "task_identities_match_ledger": actual_identities == ledger_identities,
        "seed_count_distribution": {int(k): int(v) for k, v in sorted(seed_counts.items())},
        "replicate_counts_as_expected": set(seed_counts) <= {10, 30},
    }


def task_window_means_per_seed(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Per (task, method, window) mean windowed AUC over seeds (NaN-skipping),
    matching the published convention (mean-of-per-seed).
    Windows absent from a row's source file yield NaN. Returns long rows with an
    `n_valid_seeds` column.
    """
    rows = []
    for (dataset, split_id, fam, acq), g in per_seed.groupby(
        ["dataset", "split_id", "split_family", "acq"], sort=False
    ):
        for col in WINDOW_COLS:
            vals = g[col].to_numpy(float) if col in g.columns else np.array([np.nan])
            valid = vals[np.isfinite(vals)]
            rows.append({
                "dataset": dataset, "split_id": split_id, "split_family": fam,
                "acq": acq, "window_col": col,
                "a_window": float(valid.mean()) if valid.size else float("nan"),
                "n_valid_seeds": int(valid.size),
            })
    return pd.DataFrame(rows)


def task_window_means(per_seed: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Authoritative per (task, method, window) window mean.

    Primary source = mean-of-per-seed (`task_window_means_per_seed`). Where the
    per-seed file does not store a window (the 10 Rh random-target splits on the five
    windows other than 2-12 and 5-30), fall back to the mean-curve
    `normalized_auc`. A `source` column records which was used.
    """
    keys = ["dataset", "split_id", "split_family", "acq", "window_col"]
    ps = task_window_means_per_seed(per_seed)
    cv = task_window_means_from_curves(curves).rename(columns={"a_window": "a_window_curve"})
    out = ps.merge(cv, on=keys, how="left")
    need = out["a_window"].isna() | (out["n_valid_seeds"] == 0)
    out["source"] = np.where(need & out["a_window_curve"].notna(),
                             "mean_curve_fallback", "per_seed_mean")
    out["a_window"] = np.where(need, out["a_window_curve"], out["a_window"])
    return out.drop(columns=["a_window_curve"])


def task_window_means_from_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Per (task, method) mean window AUC for all seven windows, computed from the
    committed mean curves via `normalized_auc`. Uniform across all 118 tasks.
    Returns long rows: dataset, split_id, split_family, acq, window_col, a_window.
    """
    rows = []
    for (dataset, split_id, fam), g in curves.groupby(
        ["dataset", "split_id", "split_family"], sort=False
    ):
        for acq, sub in g.groupby("acq"):
            sub = sub.sort_values("acquisition_fraction")
            x = sub["acquisition_fraction"].to_numpy(float)
            y = sub["mean_r2"].to_numpy(float)
            for col, lo, hi, _lab in WINDOWS:
                rows.append({
                    "dataset": dataset, "split_id": split_id, "split_family": fam,
                    "acq": acq, "window_col": col,
                    "a_window": normalized_auc(x, y, lo, hi),
                })
    return pd.DataFrame(rows)
