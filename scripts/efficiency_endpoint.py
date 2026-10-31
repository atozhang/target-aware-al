#!/usr/bin/env python
"""Native integer-step experimental-efficiency endpoint for the 118-task benchmark.

Recomputes, from the released per-acquisition method-mean curves and task table:

  results/efficiency/efficiency_per_task.csv        reactions to target, per task x method
  results/efficiency/efficiency_summary_reactions.csv
  results/efficiency/efficiency_summary_pct_pool.csv
                                                    scope means, IQRs and cluster-bootstrap CIs
  results/efficiency/crossing_rule_sensitivity.csv  primary vs two-consecutive crossing rule
  results/sensitivity/target_level_counts.csv       tasks reaching q = 0.70 ... 0.95

Definition
----------
For each task, R0 is the shared step-zero method mean and Rmax30 is the highest
method-level mean target-set R2 observed for any of the five methods at an
integer acquisition step from 0 through ``B = ceil(0.30 N_pool)``, inclusive.
The target is ``R0 + q (Rmax30 - R0)`` with q = 0.80 for the primary endpoint.
A method's endpoint is the first integer step whose raw method-level mean curve
meets the target. The curve is not smoothed, interpolated, floored, or made
monotone. A method that does not reach the target is charged B and retains a
separate nonattainment flag.

The two-consecutive sensitivity requires the target to be met at an integer
step and at the next integer step, i.e. for one additional acquisition.

Run from the repository root:

    python scripts/efficiency_endpoint.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_ANALYSIS = _SCRIPTS / "analysis"
for _p in (str(_ANALYSIS), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_tasks import (  # noqa: E402
    ACQS,
    LABEL,
    LABEL_ORDER,
    LEDGER,
    load_native_mean_curves,
)
from crossing import (  # noqa: E402
    crossing_or_budget,
    efficiency_target,
    native_first_crossing,
    native_two_consecutive_crossing,
)

REPO = _SCRIPTS.parent
TARGET_Q = 0.80
Q_LEVELS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
N_TASKS = 118
CLUSTER_KEYS = ["dataset", "split_family"]
KEY = ["dataset", "split_id", "split_family"]
SUMMARY_DRAWS = 5_000
SENSITIVITY_DRAWS = 100_000
RULES = {
    "native_first_integer": ("primary", native_first_crossing),
    "native_two_consecutive_integer": ("sensitivity", native_two_consecutive_crossing),
}

# --------------------------------------------------------------------------- #
#  bootstrap (the procedure and seed derivation of the published summaries)
# --------------------------------------------------------------------------- #

def stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def cluster_ci(frame: pd.DataFrame, col: str, reps: int, seed: int) -> tuple[float, float]:
    """Resample dataset x split-family clusters; task bootstrap if only one cluster."""
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


def savings_seed(value: str, method: str) -> int:
    """Seed of the published savings-vs-Random cluster intervals."""
    if method == "EPIG":
        return stable_seed(value, "Overall", "Random", "d")
    return stable_seed(value, "Overall", "Random", method, "d")


# --------------------------------------------------------------------------- #
#  per-task endpoint
# --------------------------------------------------------------------------- #

def load_inputs() -> tuple[pd.DataFrame, pd.Series]:
    curves = load_native_mean_curves()
    tasks = curves[KEY].drop_duplicates()
    if len(tasks) != N_TASKS or tasks.duplicated(["dataset", "split_id"]).any():
        raise ValueError(f"expected {N_TASKS} uniquely labelled tasks, found {len(tasks)}")
    if set(curves["acq"]) != set(ACQS):
        raise ValueError("mean curves do not carry exactly the five methods")
    if curves.groupby(["dataset", "split_id", "acq"]).ngroups != N_TASKS * len(ACQS):
        raise ValueError("native mean curves are not 118 tasks x five methods")
    if not np.isfinite(curves["mean_r2"].to_numpy(float)).all():
        raise ValueError("non-finite mean R2")
    ledger = pd.read_csv(LEDGER)
    if set(ledger["split_id"]) != set(tasks["split_id"]) or not ledger["split_id"].is_unique:
        raise ValueError("task table identities disagree with the mean curves")
    return curves, ledger.set_index("split_id")["pool_total"].astype(int)


def task_efficiency(curves: pd.DataFrame, pools: pd.Series, q: float = TARGET_Q,
                    rule=native_first_crossing) -> pd.DataFrame:
    """Per task x method efficiency table (the released ``efficiency_per_task`` schema)."""
    rows = []
    for (dataset, split_id, family), g in curves.groupby(KEY, sort=False):
        wide = g.pivot(index="n_acquired", columns="acq", values="mean_r2").sort_index()
        pool = int(pools[split_id])
        budgets = g["budget_reactions"].unique()
        if len(budgets) != 1:
            raise ValueError(f"{split_id}: inconsistent native acquisition budgets")
        budget = int(budgets[0])
        if not np.array_equal(wide.index.to_numpy(int), np.arange(budget + 1)):
            raise ValueError(f"{split_id}: native curve is not evaluated at every step 0..{budget}")
        step0 = wide.iloc[0].to_numpy(float)
        if np.ptp(step0) > 1e-10:
            raise ValueError(f"{split_id}: methods do not share a step-zero score")
        initial = float(wide.iloc[0].mean())
        ceiling = float(wide.to_numpy(float).max())      # includes step 0, so >= initial
        target = efficiency_target(initial, ceiling, q)
        for acq in ACQS:
            crossing = rule(wide[acq].to_numpy(float), target)
            attained, n_reactions = crossing_or_budget(crossing, budget)
            rows.append({
                "dataset": dataset, "split_id": split_id, "split_family": family,
                "acq": acq, "AF": LABEL[acq], "pool_total": pool,
                "budget_reactions": budget, "target_score": target,
                "attained": attained, "full_budget": not attained,
                "n_reactions": n_reactions,
                "pct_pool": 100.0 * n_reactions / pool,
            })
    out = pd.DataFrame(rows)
    if len(out) != N_TASKS * len(ACQS):
        raise ValueError("efficiency table is not 118 tasks x 5 methods")
    if (out["n_reactions"] > out["budget_reactions"]).any():
        raise ValueError("a method is charged more than the full budget")
    return out


# --------------------------------------------------------------------------- #
#  summaries
# --------------------------------------------------------------------------- #

def scopes(eff: pd.DataFrame) -> dict[str, tuple[str, pd.DataFrame]]:
    """seed key -> (published scope label, rows)."""
    rh = eff[eff["dataset"] == "Rh"]
    rh_random = rh[rh["split_family"] == "true_random"]
    out: dict[str, tuple[str, pd.DataFrame]] = {"Overall": ("Overall", eff)}
    for dataset in ["BH", "Rh", "SAD", "Wittig"]:
        out[dataset] = (dataset, eff[eff["dataset"] == dataset])
    for family in eff["split_family"].unique():
        out[f"family:{family}"] = (f"family:{family}", eff[eff["split_family"] == family])
    out["Rh_random_all"] = ("Rh random-target", rh_random)
    return out


def efficiency_summary(eff: pd.DataFrame, value: str, reps: int = SUMMARY_DRAWS) -> pd.DataFrame:
    """Per scope and method: mean, median, IQR, cluster CI, and the paired
    difference to EPIG (method minus EPIG) with its cluster CI."""
    rows = []
    for seed_key, (scope, frame) in scopes(eff).items():
        epig = frame[frame["AF"] == "EPIG"][KEY + [value]].rename(columns={value: "epig_val"})
        for af in LABEL_ORDER:
            sub = frame[frame["AF"] == af]
            v = sub[value].to_numpy(float)
            lo, hi = cluster_ci(sub, value, reps, stable_seed(value, seed_key, af))
            if af == "EPIG":
                d_mean = d_lo = d_hi = 0.0
            else:
                paired = sub.merge(epig, on=KEY)
                paired["d"] = paired[value] - paired["epig_val"]
                d_mean = float(paired["d"].mean())
                d_lo, d_hi = cluster_ci(paired, "d", reps, stable_seed(value, seed_key, af, "d"))
            rows.append({
                "scope": scope, "AF": af, "n_tasks": len(sub),
                "mean": float(v.mean()), "median": float(np.median(v)),
                "iqr_lo": float(np.quantile(v, 0.25)), "iqr_hi": float(np.quantile(v, 0.75)),
                "ci_lo": lo, "ci_hi": hi,
                "paired_diff_vs_epig": d_mean, "paired_diff_ci_lo": d_lo, "paired_diff_ci_hi": d_hi,
                "n_full_budget": int(sub["full_budget"].sum()),
            })
    return pd.DataFrame(rows)


def target_level_counts(curves: pd.DataFrame, pools: pd.Series) -> pd.DataFrame:
    rows = []
    for q in Q_LEVELS:
        eff = task_efficiency(curves, pools, q=q)
        for af in LABEL_ORDER:
            sub = eff[eff["AF"] == af]
            n = int(sub["attained"].sum())
            rows.append({"q": q, "AF": af, "n_tasks": len(sub), "attained": n,
                         "pct_attained": 100 * n / len(sub)})
    return pd.DataFrame(rows)


def crossing_rule_sensitivity(curves: pd.DataFrame, pools: pd.Series,
                              reps: int = SENSITIVITY_DRAWS) -> pd.DataFrame:
    """Primary vs two-consecutive rule: attainment, means, and savings vs Random."""
    rows = []
    for rule_name, (role, rule) in RULES.items():
        eff = task_efficiency(curves, pools, rule=rule)
        base = eff[eff["AF"] == "Random"]
        for af in LABEL_ORDER:
            sub = eff[eff["AF"] == af]
            row = {"rule": rule_name, "role": role, "method": af, "n_tasks": len(sub),
                   "n_attained": int(sub["attained"].sum()),
                   "mean_n_reactions": float(sub["n_reactions"].mean()),
                   "mean_pct_pool": float(sub["pct_pool"].mean())}
            for value, tag in (("n_reactions", "reactions_saved"), ("pct_pool", "pool_saved_pp")):
                paired = base.merge(
                    sub[KEY + [value]].rename(columns={value: "other_val"}), on=KEY)
                paired["d"] = paired[value] - paired["other_val"]
                lo, hi = ((0.0, 0.0) if af == "Random"
                          else cluster_ci(paired, "d", reps, savings_seed(value, af)))
                row[f"{tag}_vs_random"] = float(paired["d"].mean())
                row[f"{tag}_ci_lo"] = lo
                row[f"{tag}_ci_hi"] = hi
            row["bootstrap_draws"] = reps
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=REPO,
                        help="write results/... under this root (default: the repository)")
    args = parser.parse_args(argv)
    efficiency_dir = args.out_root / "results/efficiency"
    sensitivity_dir = args.out_root / "results/sensitivity"
    efficiency_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    curves, pools = load_inputs()
    eff = task_efficiency(curves, pools)
    eff.to_csv(efficiency_dir / "efficiency_per_task.csv", index=False)
    for value, tag in (("n_reactions", "reactions"), ("pct_pool", "pct_pool")):
        efficiency_summary(eff, value).to_csv(
            efficiency_dir / f"efficiency_summary_{tag}.csv", index=False)
    target_level_counts(curves, pools).to_csv(
        sensitivity_dir / "target_level_counts.csv", index=False)
    sensitivity = crossing_rule_sensitivity(curves, pools)
    sensitivity.to_csv(efficiency_dir / "crossing_rule_sensitivity.csv", index=False)

    means = eff.groupby("AF")["n_reactions"].mean().reindex(LABEL_ORDER)
    attained = eff.groupby("AF")["attained"].sum().reindex(LABEL_ORDER)
    print(f"native first integer crossing, q = {TARGET_Q:.2f}, {N_TASKS} tasks")
    for af in LABEL_ORDER:
        print(f"  {af:7s} mean reactions {means[af]:8.2f}   attained {int(attained[af]):3d}/{N_TASKS}")
    print("\nsavings vs Random, reactions (100,000-draw cluster CI)")
    for r in sensitivity[sensitivity["method"] != "Random"].itertuples():
        print(f"  {r.rule:20s} {r.method:7s} {r.reactions_saved_vs_random:8.2f} "
              f"[{r.reactions_saved_ci_lo:.2f}, {r.reactions_saved_ci_hi:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
