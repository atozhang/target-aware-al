"""Paired reaction savings versus Random on the 118-task benchmark.

reactions_saved[m, t] = n_reactions(Random, t) - n_reactions(m, t)

Positive values mean the method reached the target score using fewer reactions
than Random did on the same task.

Output
------
results/efficiency/reactions_saved_vs_random_summary.csv: per-method mean, SD,
median, a 95 % cluster-bootstrap interval at 5,000 draws, and censoring counts.

The published reaction-savings intervals in SI Table S8 and
results/main_results_table.csv are not these 5,000-draw intervals. They are the
100,000-draw intervals of scripts/efficiency_savings_intervals.py
(results/efficiency/savings_intervals_100k_primary.csv), which use the same
bootstrap procedure and seeds. The means are identical in both files.

Procedure
---------
`cluster_ci`: resample the 12 dataset x split_family clusters with replacement,
pool the tasks of the selected clusters, take the mean, and report the 2.5/97.5
percentiles over 5,000 draws. efficiency_pool_spared.py applies the same procedure
to pct_pool.

Bootstrap seeds
---------------
Fixed and derived deterministically from literal key parts:

  EPIG   stable_seed("n_reactions", "Overall", "Random", "d")
         -- the seed of the Overall/Random paired difference in
            efficiency_summary_reactions.csv (written by efficiency_endpoint.py,
            also at 5,000 draws). The EPIG row must reproduce that row's
            paired_diff_vs_epig and CI; this agreement is asserted.
  other  stable_seed("n_reactions", "Overall", "Random", <method>, "d")

The same seeds are used for the published 100,000-draw cluster intervals. The
resolved integer seeds are pinned in EXPECTED_SEEDS below and asserted, so a
change in the derivation cannot silently move an interval.

The Random row is included for completeness: Random minus Random is identically
zero on every task, so its mean, SD and interval are exactly 0.
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
REFERENCE_5000_TABLE = PKG / "efficiency_summary_reactions.csv"
LEDGER = REPO / "supporting_information/tables/benchmark_tasks.csv"
WITTIG_TRANSFER_TASK = "wittig_f119_textbook_base_cas_random_85_15"

METHODS = ["EPIG", "BALD", "IndvNN", "NN", "Random"]
VALUE = "n_reactions"
SCOPE = "Overall"
CLUSTER_BOOT = 5000  # 5,000 draws; the published intervals use 100,000 (efficiency_savings_intervals.py)
CLUSTER_KEYS = ["dataset", "split_family"]
EXPECTED_TASKS = 118

# Pinned seeds: any change to the derivation must fail loudly rather than move a
# published interval.
EXPECTED_SEEDS = {
    "EPIG": 3382740998,
    "BALD": 2225935390,
    "IndvNN": 4193014655,
    "NN": 1624363823,
    "Random": 1714580582,
}

PIN_DP = 10
ROUNDTRIP_TOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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

def seed_for(method: str) -> tuple[int, tuple[str, ...]]:
    """Primary key for EPIG (the Random-vs-EPIG pair); documented variant otherwise."""
    if method == "EPIG":
        parts = (VALUE, SCOPE, "Random", "d")
    else:
        parts = (VALUE, SCOPE, "Random", method, "d")
    return stable_seed(*parts), parts


def main() -> None:
    eff = pd.read_csv(PER_TASK)
    key = ["dataset", "split_id"]

    # Verify that the efficiency tasks match the task ledger.
    ledger_ids = set(pd.read_csv(LEDGER)["split_id"])
    efficiency_ids = set(eff["split_id"])
    if efficiency_ids != ledger_ids:
        raise ValueError("efficiency_per_task task identities disagree with the final ledger")
    if WITTIG_TRANSFER_TASK not in efficiency_ids:
        raise ValueError("efficiency_per_task does not contain the Wittig transfer task")

    # n_reactions is stored as an exact integer count, so unlike pct_pool it
    # round-trips through CSV without loss and needs no recomputation.
    if not np.array_equal(eff[VALUE].to_numpy(), eff[VALUE].to_numpy().astype(np.int64)):
        raise ValueError("n_reactions is expected to be integral")

    # --- pairing check ----------------------------------------------------
    if eff.duplicated(key + ["AF"]).any():
        raise ValueError("duplicate (task, AF) rows")
    per_task_afs = eff.groupby(key)["AF"].nunique()
    if not (per_task_afs == 5).all():
        raise ValueError("not every task carries all five acquisition functions")
    if len(per_task_afs) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, found {len(per_task_afs)}")

    # Random supplies the base rows, so the row order and the summation order of
    # the bootstrap means match the Overall/Random computation of efficiency_endpoint.py.
    base = eff[eff["AF"] == "Random"]

    rows = []
    for method in METHODS:
        other = eff[eff["AF"] == method][key + ["split_family", VALUE, "full_budget"]].rename(
            columns={VALUE: "other_val", "full_budget": "other_full_budget"})
        paired = base.merge(other, on=key + ["split_family"])
        if len(paired) != EXPECTED_TASKS:
            raise ValueError(f"{method}: merge dropped tasks ({len(paired)} paired)")
        paired["d"] = paired[VALUE] - paired["other_val"]

        values = paired["d"].to_numpy(float)
        seed, parts = seed_for(method)
        if seed != EXPECTED_SEEDS[method]:
            raise AssertionError(
                f"{method}: seed {seed} != pinned {EXPECTED_SEEDS[method]}; the "
                "seed derivation changed and would move a published interval")
        lo, hi = cluster_ci(paired, "d", CLUSTER_BOOT, seed)

        rows.append({
            "method": method,
            "n_tasks": int(len(paired)),
            # pandas Series.mean(), matching float(paired["d"].mean())
            "mean_reactions_saved": float(paired["d"].mean()),
            "sd_reactions_saved": float(paired["d"].std(ddof=1)),
            "median_reactions_saved": float(np.median(values)),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_saved_gt_zero": int((values > 0).sum()),
            "n_saved_zero": int((values == 0).sum()),
            "n_saved_lt_zero": int((values < 0).sum()),
            "n_full_budget_method": int(paired["other_full_budget"].sum()),
            "n_full_budget_random": int(paired["full_budget"].sum()),
            "n_full_budget_both": int((paired["full_budget"].to_numpy(bool)
                                      & paired["other_full_budget"].to_numpy(bool)).sum()),
            "bootstrap_seed_parts": "|".join(parts),
            "bootstrap_seed": seed,
        })

    summary = pd.DataFrame(rows)

    # --- the Random row must be identically zero ---------------------------
    r = summary[summary["method"] == "Random"].iloc[0]
    for col in ("mean_reactions_saved", "sd_reactions_saved", "median_reactions_saved",
                "ci_lo", "ci_hi"):
        if float(r[col]) != 0.0:
            raise AssertionError(f"Random {col} must be exactly 0, got {r[col]!r}")

    # --- exact agreement with the stored 5,000-draw Random-vs-EPIG row -------
    epig = summary[summary["method"] == "EPIG"].iloc[0]
    canon = pd.read_csv(REFERENCE_5000_TABLE)
    crow = canon[(canon["scope"] == SCOPE) & (canon["AF"] == "Random")].iloc[0]
    checks = {
        "mean": (epig["mean_reactions_saved"], float(crow["paired_diff_vs_epig"])),
        "ci_lo": (epig["ci_lo"], float(crow["paired_diff_ci_lo"])),
        "ci_hi": (epig["ci_hi"], float(crow["paired_diff_ci_hi"])),
    }
    for name, (got, from_csv) in checks.items():
        if abs(got - from_csv) > ROUNDTRIP_TOL:
            raise AssertionError(
                f"EPIG {name} diverges from {REFERENCE_5000_TABLE.name} beyond serialization "
                f"noise: {got!r} vs {from_csv!r} (delta {got - from_csv:.3e})")
    print(f"PASS: EPIG row reproduces the 5,000-draw Overall/Random paired reaction "
          f"difference of {REFERENCE_5000_TABLE.name}: {epig['mean_reactions_saved']:.10f} "
          f"[{epig['ci_lo']:.10f}, {epig['ci_hi']:.10f}]\n"
          f"      (max |delta| vs stored CSV "
          f"{max(abs(g - c) for g, c in checks.values()):.3e}, float64 round-trip noise); "
          f"seed parts {epig['bootstrap_seed_parts']}\n")

    summary.to_csv(PKG / "reactions_saved_vs_random_summary.csv", index=False)

    # --- printed cells at 5,000 draws: mean [95% CI], whole reactions --------
    def cell(row) -> str:
        if row.method == "Random":
            return "Reference"
        return (f"{row.mean_reactions_saved:.0f} "
                f"[{row.ci_lo:.0f}, {row.ci_hi:.0f}]".replace("-", "−"))

    main_table = pd.DataFrame({
        "Method": summary["method"],
        "Mean reactions saved vs. Random [95% CI]": [cell(r) for r in summary.itertuples()],
    })

    # --- diagnostics at 5,000 draws: CIs and censoring counts ----------------
    si_table = pd.DataFrame({
        "Method": summary["method"],
        "Mean (SD)": [f"{r.mean_reactions_saved:.2f} ({r.sd_reactions_saved:.2f})"
                      for r in summary.itertuples()],
        "Median": [f"{r.median_reactions_saved:.2f}" for r in summary.itertuples()],
        "95% CI": [f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]" for r in summary.itertuples()],
        "Saved > 0": summary["n_saved_gt_zero"],
        "Saved = 0": summary["n_saved_zero"],
        "Saved < 0": summary["n_saved_lt_zero"],
        "Method at budget cap": summary["n_full_budget_method"],
        "Both at budget cap": summary["n_full_budget_both"],
    })


    pd.set_option("display.width", 160)
    print(summary[["method", "n_tasks", "mean_reactions_saved", "sd_reactions_saved",
                   "median_reactions_saved", "ci_lo", "ci_hi", "bootstrap_seed"]].to_string(index=False))
    print("\n5,000-draw cells (published 100,000-draw intervals: savings_intervals_100k_primary.csv):")
    for r in main_table.itertuples():
        print(f"  {r.Method:8s} {getattr(r, '_2')}")


if __name__ == "__main__":
    main()
