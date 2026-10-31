"""Experimental-savings intervals at 100,000 draws, with resampling-structure sensitivities.

This script produces the published 100,000-draw intervals reported in SI Table
S8 and results/main_results_table.csv, written to
savings_intervals_100k_primary.csv and savings_intervals_100k_by_structure.csv.
`efficiency_pool_spared.py` and `efficiency_reactions_saved.py` compute 5,000-draw
intervals with the same procedure and seeds; at 5,000 draws the cluster structure
below reproduces them exactly.

Draw count
----------
The published intervals use 100,000 draws. A 200,000-draw check is recorded in
savings_intervals_200k_confirmation.csv; near a rounding boundary, a displayed
endpoint can differ by one unit in that check.

Resampling structures
---------------------
    cluster      PRIMARY, and the declared procedure: resample the 12
                 dataset x split_family clusters with replacement and pool the
                 tasks of the selected clusters (`cluster_ci`).
    task         ordinary paired-task bootstrap: resample the 118 task-level
                 differences independently, ignoring the clustering.
    hierarchical two-stage: resample clusters with replacement, then resample the
                 tasks WITHIN each selected cluster with replacement. Propagates
                 both levels of variation rather than treating the tasks inside a
                 selected cluster as fixed.

The summary identifies methods for which the confidence interval excludes zero
under some resampling schemes but not others. Means are identical under all
three; only the intervals differ.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "results/efficiency"
PER_TASK = PKG / "efficiency_per_task.csv"

METHODS = ["EPIG", "BALD", "IndvNN", "NN", "Random"]
CLUSTER_KEYS = ["dataset", "split_family"]
PRIMARY_DRAWS = 100_000
CONFIRM_DRAWS = 200_000
EXPECTED_TASKS = 118

METRICS = {
    "n_reactions": {"label": "mean reactions saved", "dp": 0},
    "pct_pool": {"label": "% acquisition pool saved", "dp": 1},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the seed derivation used for the published intervals ------------------ #

def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def cluster_ci(frame: pd.DataFrame, col: str, reps: int, seed: int) -> tuple[float, float]:
    clusters = [g[col].to_numpy(float) for _, g in frame.groupby(CLUSTER_KEYS)]
    rng = np.random.default_rng(seed)
    n = len(clusters)
    if n < 2:
        values = frame[col].to_numpy(float)
        m = len(values)
        means = values[rng.integers(0, m, size=(reps, m))].mean(axis=1)
        return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    means = np.array([np.concatenate([clusters[j] for j in rng.integers(0, n, n)]).mean()
                      for _ in range(reps)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# --------------------------------------------------------------------------- #

def task_ci(frame: pd.DataFrame, col: str, reps: int, seed: int) -> tuple[float, float]:
    """Ordinary paired-task bootstrap over the 118 differences."""
    values = frame[col].to_numpy(float)
    rng = np.random.default_rng(seed)
    m = len(values)
    means = values[rng.integers(0, m, size=(reps, m))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def hierarchical_ci(frame: pd.DataFrame, col: str, reps: int, seed: int) -> tuple[float, float]:
    """Two-stage: resample clusters, then resample tasks within each drawn cluster."""
    clusters = [g[col].to_numpy(float) for _, g in frame.groupby(CLUSTER_KEYS)]
    rng = np.random.default_rng(seed)
    n = len(clusters)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        picked = rng.integers(0, n, n)
        parts = []
        for j in picked:
            c = clusters[j]
            parts.append(c[rng.integers(0, len(c), len(c))])
        means[i] = np.concatenate(parts).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


STRUCTURES = {"cluster": cluster_ci, "task": task_ci, "hierarchical": hierarchical_ci}


def seed_for(value: str, method: str, structure: str) -> int:
    """Deterministic seeds. The cluster structure retains the seed key documented
    in `efficiency_reactions_saved.py`, so the 5,000-draw intervals remain exactly
    reproducible; the two additional structures extend that key with the structure
    name."""
    base = (value, "Overall", "Random", "d") if method == "EPIG" else \
           (value, "Overall", "Random", method, "d")
    return stable_seed(*base) if structure == "cluster" else stable_seed(*base, structure)


def paired(eff: pd.DataFrame, value: str, method: str) -> pd.DataFrame:
    key = ["dataset", "split_id"]
    base = eff[eff["AF"] == "Random"]
    other = eff[eff["AF"] == method][key + ["split_family", value]].rename(
        columns={value: "other_val"})
    out = base.merge(other, on=key + ["split_family"])
    if len(out) != EXPECTED_TASKS:
        raise ValueError(f"{method}: {len(out)} paired rows")
    out = out.copy()
    out["d"] = out[value] - out["other_val"]
    return out


def fmt(v: float, dp: int) -> str:
    return f"{v:.{dp}f}" if dp else f"{v:.0f}"


def main() -> None:
    eff = pd.read_csv(PER_TASK)
    # pct_pool does not round-trip through CSV bit-for-bit; recompute from the
    # exactly-stored integer columns using the published formula
    derived = 100.0 * eff["n_reactions"] / eff["pool_total"]
    if not np.allclose(derived, eff["pct_pool"], rtol=0, atol=1e-9):
        raise ValueError("recomputed pct_pool disagrees with the stored column")
    eff["pct_pool"] = derived

    rows, confirm_rows = [], []
    for value, meta in METRICS.items():
        for method in METHODS:
            fr = paired(eff, value, method)
            mean = float(fr["d"].mean())
            for structure, fn in STRUCTURES.items():
                seed = seed_for(value, method, structure)
                if method == "Random":
                    lo = hi = 0.0  # Random - Random is identically zero
                else:
                    lo, hi = fn(fr, "d", PRIMARY_DRAWS, seed)
                rows.append({
                    "metric": value, "method": method, "structure": structure,
                    "draws": PRIMARY_DRAWS, "mean": mean, "ci_lo": lo, "ci_hi": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "rounded": f"{fmt(mean, meta['dp'])} [{fmt(lo, meta['dp'])}, "
                               f"{fmt(hi, meta['dp'])}]",
                    "seed": seed,
                })
                # 200,000-draw confirmation of the published (cluster) endpoints
                if structure == "cluster" and method != "Random":
                    clo, chi = fn(fr, "d", CONFIRM_DRAWS, seed)
                    confirm_rows.append({
                        "metric": value, "method": method,
                        "lo_100k": lo, "hi_100k": hi,
                        "lo_200k": clo, "hi_200k": chi,
                        "rounded_100k": f"[{fmt(lo, meta['dp'])}, {fmt(hi, meta['dp'])}]",
                        "rounded_200k": f"[{fmt(clo, meta['dp'])}, {fmt(chi, meta['dp'])}]",
                        "rounded_unchanged": (fmt(lo, meta["dp"]) == fmt(clo, meta["dp"])
                                              and fmt(hi, meta["dp"]) == fmt(chi, meta["dp"])),
                    })
    detail = pd.DataFrame(rows)
    confirm = pd.DataFrame(confirm_rows)
    detail.to_csv(PKG / "savings_intervals_100k_by_structure.csv", index=False)
    confirm.to_csv(PKG / "savings_intervals_200k_confirmation.csv", index=False)

    # ---- published (primary, cluster) table ----
    published = detail[(detail.structure == "cluster")][
        ["metric", "method", "mean", "ci_lo", "ci_hi", "rounded", "excludes_zero", "seed"]]
    published.to_csv(PKG / "savings_intervals_100k_primary.csv", index=False)

    # ---- structure dependence ----
    dep = []
    for (value, method), g in detail.groupby(["metric", "method"], sort=False):
        if method == "Random":
            continue
        ex = {r.structure: bool(r.excludes_zero) for r in g.itertuples()}
        if len(set(ex.values())) > 1:
            dep.append({"metric": value, "method": method,
                        **{f"excludes_zero_{k}": v for k, v in ex.items()}})
    dep = pd.DataFrame(dep)
    dep.to_csv(PKG / "savings_structure_dependence.csv", index=False)

    all_confirmed = bool(confirm["rounded_unchanged"].all())

    # ---------------- report ----------------
    for value, meta in METRICS.items():
        print(f"\n=== {meta['label']} — {PRIMARY_DRAWS:,} draws ===")
        print(f"{'method':8s} {'cluster (PRIMARY)':>28s} {'paired-task':>28s} {'hierarchical':>28s}")
        for method in METHODS:
            g = detail[(detail.metric == value) & (detail.method == method)].set_index("structure")
            cells = ""
            dp = meta["dp"]
            for st in ("cluster", "task", "hierarchical"):
                r = g.loc[st]
                mark = "" if method == "Random" else ("" if r["excludes_zero"] else "  0*")
                interval = f"[{fmt(r.ci_lo, dp)}, {fmt(r.ci_hi, dp)}]{mark}"
                cells += f"{interval:>28s}"
            print(f"{method:8s}{cells}")
        print("  0* = interval includes zero")

    print(f"\n=== {CONFIRM_DRAWS:,}-draw confirmation of the primary endpoints ===")
    for r in confirm.itertuples():
        flag = "unchanged" if r.rounded_unchanged else "CHANGED"
        print(f"  {r.metric:12s} {r.method:8s} {r.rounded_100k:>20s} -> "
              f"{r.rounded_200k:>20s}  {flag}")
    print(f"\nall manuscript-rounded endpoints unchanged at {CONFIRM_DRAWS:,}: {all_confirmed}")

    print("\n=== resampling-structure sensitivity ===")
    if len(dep):
        for r in dep.itertuples():
            print(f"  {r.metric:12s} {r.method:8s} excludes zero: "
                  f"cluster={r.excludes_zero_cluster} task={r.excludes_zero_task} "
                  f"hierarchical={r.excludes_zero_hierarchical}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
