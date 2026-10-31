"""Paired pool sparing versus Random on the 118-task benchmark.

pool_spared[m, t] = pct_pool(Random, t) - pct_pool(m, t)

Positive values mean the method reached the target score on a smaller fraction
of the candidate pool than Random did on the same task.

Output: results/efficiency/pool_spared_vs_random_summary.csv, with 95 %
cluster-bootstrap intervals at 5,000 draws. The published pool-savings intervals
in SI Table S8 and results/main_results_table.csv are the
100,000-draw intervals of scripts/efficiency_savings_intervals.py
(results/efficiency/savings_intervals_100k_primary.csv), which use the same
procedure and seeds; the means are identical.

The EPIG row of this table is the Random-versus-EPIG paired difference stored in
efficiency_summary_pct_pool.csv (scope "Overall", AF "Random", written by
efficiency_endpoint.py at 5,000 draws): pct_pool(Random) - pct_pool(EPIG) is
exactly the quantity that row's `paired_diff_vs_epig` / `paired_diff_ci_*`
columns hold. This script reuses the same `cluster_ci`, `stable_seed`, cluster
labels, and draw count, and asserts that the EPIG row reproduces that stored
mean and interval.

Bootstrap seeds
---------------
`scripts/efficiency_endpoint.py` seeds each paired-difference CI with
`stable_seed(value, scope, af, "d")`, where `af` is the non-EPIG member of the
pair. For the EPIG row here the pair is (Random, EPIG), so the seed is
`stable_seed("pct_pool", "Overall", "Random", "d")`. The BALD/Indiv. NN/NN rows
use the variant key `stable_seed("pct_pool", "Overall", "Random", af, "d")`.
The published 100,000-draw cluster intervals use the same seeds.
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
REFERENCE_5000_TABLE = PKG / "efficiency_summary_pct_pool.csv"

METHODS = ["EPIG", "BALD", "IndvNN", "NN"]
VALUE = "pct_pool"
SCOPE = "Overall"
CLUSTER_BOOT = 5000  # 5,000 draws; the published intervals use 100,000 (efficiency_savings_intervals.py)
CLUSTER_KEYS = ["dataset", "split_family"]
EXPECTED_TASKS = 118

# Stored 5,000-draw Overall/Random paired difference versus EPIG, from
# efficiency_summary_pct_pool.csv as written by scripts/efficiency_endpoint.py.
# (The published 100,000-draw interval is in savings_intervals_100k_primary.csv.)
REFERENCE_5000_MEAN = 6.8966511709
REFERENCE_5000_CI = (4.1043835037, 9.9541237569)
PIN_DP = 10           # precision at which these reference values are pinned
ROUNDTRIP_TOL = 1e-12  # float64 CSV serialization noise in the stored reference


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

    # The stored pct_pool column does not round-trip through CSV bit-for-bit, so
    # recompute it from the exactly-stored integer columns using the defining
    # formula, 100 * n_reactions / pool_total. This is
    # what makes the EPIG row bitwise-identical to the in-memory result of efficiency_endpoint.py;
    # the two differ by ~1e-15, i.e. serialization noise only.
    derived = 100.0 * eff["n_reactions"] / eff["pool_total"]
    if not np.allclose(derived, eff[VALUE], rtol=0, atol=1e-9):
        raise ValueError("recomputed pct_pool disagrees with the stored column")
    eff[VALUE] = derived

    # --- pairing check ----------------------------------------------------
    if eff.duplicated(key + ["AF"]).any():
        raise ValueError("duplicate (task, AF) rows")
    per_task_afs = eff.groupby(key)["AF"].nunique()
    if not (per_task_afs == 5).all():
        raise ValueError("not every task carries all five acquisition functions")
    if len(per_task_afs) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, found {len(per_task_afs)}")

    # Build each paired frame exactly as the generator does: the reference
    # AF supplies the base rows (and therefore the row order and the summation order
    # of the bootstrap means), the compared AF is merged on the full task key, and
    # the difference is taken as reference minus compared. Holding the base rows to
    # the Random frame is what makes the EPIG row bitwise-identical to the stored
    # Overall/Random computation, which used the same orientation.
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
        lo, hi = cluster_ci(paired, "d", CLUSTER_BOOT, seed)

        rows.append({
            "method": method,
            "n_tasks": int(len(paired)),
            # pandas Series.mean(), matching `float(paired["d"].mean())`;
            # numpy's ndarray.mean() uses pairwise summation and differs by ~1 ULP.
            "mean_pool_spared_pp": float(paired["d"].mean()),
            "sd_pool_spared_pp": float(paired["d"].std(ddof=1)),  # sample SD
            "median_pool_spared_pp": float(np.median(values)),
            "ci_lo_pp": lo,
            "ci_hi_pp": hi,
            "n_spared_gt_zero": int((values > 0).sum()),
            "n_spared_zero": int((values == 0).sum()),
            "n_spared_lt_zero": int((values < 0).sum()),
            "n_full_budget_method": int(paired["other_full_budget"].sum()),
            "n_full_budget_random": int(paired["full_budget"].sum()),
            "n_full_budget_both": int((paired["full_budget"].to_numpy(bool)
                                       & paired["other_full_budget"].to_numpy(bool)).sum()),
            "bootstrap_seed_parts": "|".join(parts),
            "bootstrap_seed": seed,
        })

    summary = pd.DataFrame(rows)

    # --- exact agreement with the stored 5,000-draw Random-vs-EPIG row -----
    epig = summary[summary["method"] == "EPIG"].iloc[0]
    canon = pd.read_csv(REFERENCE_5000_TABLE)
    crow = canon[(canon["scope"] == SCOPE) & (canon["AF"] == "Random")].iloc[0]
    checks = {
        "mean": (epig["mean_pool_spared_pp"], float(crow["paired_diff_vs_epig"]), REFERENCE_5000_MEAN),
        "ci_lo": (epig["ci_lo_pp"], float(crow["paired_diff_ci_lo"]), REFERENCE_5000_CI[0]),
        "ci_hi": (epig["ci_hi_pp"], float(crow["paired_diff_ci_hi"]), REFERENCE_5000_CI[1]),
    }
    for name, (got, from_csv, literal) in checks.items():
        # Exact agreement at the pinned precision: both values must round to the
        # same 10-decimal reference figure.
        if round(got, PIN_DP) != literal:
            raise AssertionError(
                f"EPIG {name} does not reproduce the pinned reference value at "
                f"{PIN_DP} dp: {got!r} -> {round(got, PIN_DP)!r} vs {literal!r}")
        # Against the stored CSV, allow float64 serialization noise only. The
        # stored numbers were computed in memory and written out with lossy
        # text formatting, so the reference itself carries ~1 ULP of round-trip
        # error; anything larger would be a real methodological divergence.
        if abs(got - from_csv) > ROUNDTRIP_TOL:
            raise AssertionError(
                f"EPIG {name} diverges from {REFERENCE_5000_TABLE.name} beyond serialization "
                f"noise: {got!r} vs {from_csv!r} (delta {got - from_csv:.3e})")
    print(f"PASS: EPIG row reproduces the 5,000-draw Overall/Random paired difference of {REFERENCE_5000_TABLE.name} "
          f"{REFERENCE_5000_MEAN} [{REFERENCE_5000_CI[0]}, {REFERENCE_5000_CI[1]}] "
          f"exactly at {PIN_DP} dp\n"
          f"      (max |delta| vs stored CSV "
          f"{max(abs(g - c) for g, c, _ in checks.values()):.3e}, float64 round-trip noise); "
          f"seed parts {epig['bootstrap_seed_parts']}\n")

    summary.to_csv(PKG / "pool_spared_vs_random_summary.csv", index=False)

    # --- display tables at 5,000 draws, printed, not written ---------------
    main_table = pd.DataFrame({
        "Method": summary["method"],
        "Mean pool spared vs. Random (pp), mean (SD)": [
            f"{r.mean_pool_spared_pp:.1f} ({r.sd_pool_spared_pp:.1f})" for r in summary.itertuples()],
    })

    # --- diagnostics at 5,000 draws: CIs and censoring counts ---------------
    si_table = pd.DataFrame({
        "Method": summary["method"],
        "Mean (SD)": [f"{r.mean_pool_spared_pp:.2f} ({r.sd_pool_spared_pp:.2f})"
                      for r in summary.itertuples()],
        "Median": [f"{r.median_pool_spared_pp:.2f}" for r in summary.itertuples()],
        "95% CI": [f"[{r.ci_lo_pp:.2f}, {r.ci_hi_pp:.2f}]" for r in summary.itertuples()],
        "Spared > 0": summary["n_spared_gt_zero"],
        "Spared = 0": summary["n_spared_zero"],
        "Spared < 0": summary["n_spared_lt_zero"],
        "Method at budget cap": summary["n_full_budget_method"],
        "Both at budget cap": summary["n_full_budget_both"],
    })


    print("SUMMARY (5,000 draws)")
    print(main_table.to_string(index=False))
    print("\nDIAGNOSTICS (5,000-draw CIs; published 100,000-draw intervals: savings_intervals_100k_primary.csv)")
    print(si_table.to_string(index=False))


if __name__ == "__main__":
    main()
