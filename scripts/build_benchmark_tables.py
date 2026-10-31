#!/usr/bin/env python
"""Regenerate the benchmark summary tables that are derived from the per-task classification.

Writes, from the released per-seed inputs and the efficiency and savings tables:

  results/benchmark/per_task_classification.csv, counts_by_window.csv, counts_by_dataset.csv,
      group_summary.csv
  results/sensitivity/rh_partition/counts.csv, summary.json
  results/efficiency/negative_ceiling_margin.csv
  results/main_results_table.csv
  supporting_information/tables/competitiveness/*.csv (six count tables, two task-level tables)

Every statistic comes from the repository's own modules: ``verify_results.run_window`` /
``classify_task`` / ``counts_from`` (highest mean; exact paired Wilcoxon signed-rank with the
within-task Holm correction), ``inputs.analysis_unit_matrix`` (Rh partition schemes), and the
committed efficiency and savings tables. Only the table assembly is done here.

Run after ``efficiency_endpoint.py`` and ``efficiency_savings_intervals.py`` (the group summary
and the main results table read their outputs), then check with ``verify_results.py``:

    python scripts/build_benchmark_tables.py            # write the tables
    python scripts/build_benchmark_tables.py --check    # regenerate in a temporary directory and
                                                         # compare with the committed files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINUS = "−"
OUTPUTS = [
    "results/benchmark/per_task_classification.csv",
    "results/benchmark/counts_by_window.csv",
    "results/benchmark/counts_by_dataset.csv",
    "results/benchmark/group_summary.csv",
    "results/sensitivity/rh_partition/counts.csv",
    "results/sensitivity/rh_partition/summary.json",
    "results/efficiency/negative_ceiling_margin.csv",
    "results/main_results_table.csv",
    *[f"supporting_information/tables/competitiveness/{s}_by_{k}.csv"
      for s in ("highest_mean", "competitive") for k in ("overall", "dataset", "split_family")],
    "supporting_information/tables/competitiveness/task_level_classifications_all_windows.csv",
    "supporting_information/tables/competitiveness/task_level_classifications_a2_12.csv",
]
SCHEMES = ["primary", "pooled_30", "partition_0", "partition_1", "partition_2"]
# the released header line (two bracketed column names are quoted in the release; data rows are pandas QUOTE_MINIMAL)
MAIN_TABLE_HEADER = ('Acquisition function,"Highest mean AUC2–12, n/N (%)","Statistically competitive, n/N (%)",'
                     '"Mean reactions saved relative to Random [95% CI]","% Acquisition pool saved relative to Random [95% CI]"\n')
FAMILY_DISPLAY = {"true_random": "Random", "cluster_based": "Cluster",
                  "chemistry_class_or_substrate_holdout": "Chemical family", "paper_defined": "Published"}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


REPO = Path(__file__).resolve().parents[1]


def load_repo(repo: Path = REPO):
    for p in (repo / "scripts/analysis", repo / "scripts"):
        sys.path.insert(0, str(p))
    import benchmark_tasks as fc
    import inputs as cio
    import verify_results as V
    if fc.REPO.resolve() != repo.resolve():
        raise SystemExit(f"imported modules resolve to {fc.REPO}, not {repo}")
    return fc, cio, V


def classification(cio, V) -> pd.DataFrame:
    per_seed = cio.load_per_seed_full()
    cio.verify_pairing(per_seed)  # benchmark census + strict pairing; raises on any problem
    nesting = cio.load_nesting()
    cols = ["auc_2_12"] + [w[0] for w in cio.WINDOWS if w[0] != "auc_2_12"]
    parts, mm = [], []
    for col in cols:
        c, m = V.run_window(per_seed, col, nesting)
        parts.append(c)
        mm += m
    if mm:
        raise SystemExit(f"rounding-sensitivity mismatches: {mm}")
    return pd.concat(parts, ignore_index=True), per_seed, nesting


def si_tables(cls: pd.DataFrame, V, out: Path) -> None:
    disp, order = V.DISPLAY, V.LABEL_ORDER
    frame = cls.copy()
    frame["acquisition_function"] = frame["AF"].map(disp)
    frame["split_family_display"] = frame["split_family"].map(FAMILY_DISPLAY)
    windows = [w[3] for w in V.WINDOWS]
    method_order = [disp[m] for m in order]
    task = frame.drop(columns=["split_family"]).rename(columns={
        "split_family_display": "split_family", "is_highest_mean": "highest_mean",
        "not_detected_worse": "statistically_competitive"})[[
            "window", "dataset", "split_family", "split_id", "acquisition_function", "n_analysis_units", "unit_scheme",
            "mean_auc", "gap_to_best", "highest_mean", "detected_worse", "statistically_competitive",
            "holm_wilcoxon_p_vs_top", "ref_vs", "friedman_chi2", "friedman_p"]]
    task["ref_vs"] = task["ref_vs"].map(lambda v: disp.get(v, v))
    task = task.sort_values(["window", "dataset", "split_id", "acquisition_function"])
    d = out / "supporting_information/tables/competitiveness"
    d.mkdir(parents=True, exist_ok=True)
    task.to_csv(d / "task_level_classifications_all_windows.csv", index=False)
    task[task["window"].eq("2-12")].to_csv(d / "task_level_classifications_a2_12.csv", index=False)
    specs = [("overall", [None], None), ("dataset", ["BH", "Rh", "SAD", "Wittig"], "dataset"),
             ("split_family", ["Random", "Cluster", "Chemical family", "Published"], "split_family_display")]
    for kind, scopes, column in specs:
        for stem, value_col in (("highest_mean_by", "is_highest_mean"), ("competitive_by", "not_detected_worse")):
            rows = []
            for scope in scopes:
                sub = frame if column is None else frame[frame[column].eq(scope)]
                n_tasks = int(sub["split_id"].nunique())
                for method in method_order:
                    row = {"scope": "All tasks" if scope is None else scope, "acquisition_function": method}
                    for w in windows:
                        n = int(sub[sub["window"].eq(w) & sub["acquisition_function"].eq(method)][value_col].sum())
                        row[w] = f"{n}/{n_tasks} ({int(round(100 * n / n_tasks))}%)"
                    rows.append(row)
            pd.DataFrame(rows).to_csv(d / f"{stem}_{kind}.csv", index=False)


def group_summary(cls: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    primary = cls[cls["window"].eq("2-12")]
    agg = dict(n_tasks=("split_id", "nunique"), highest_mean=("is_highest_mean", "sum"), competitive=("not_detected_worse", "sum"))
    eagg = dict(mean_reactions=("n_reactions", "mean"), mean_pct_pool=("pct_pool", "mean"))
    cparts, eparts = [], []
    for grouping, col, label in (("Overall", None, "All tasks"), ("Reaction dataset", "dataset", None), ("Split family", "split_family", None)):
        if col is None:
            g = primary.groupby("AF", as_index=False).agg(**agg); g["grouping"] = grouping; g["group"] = label
            e = eff.groupby("AF", as_index=False).agg(**eagg); e["grouping"] = grouping; e["group"] = label
        else:
            g = primary.groupby([col, "AF"], as_index=False).agg(**agg); g["grouping"] = grouping; g["group"] = g[col]
            e = eff.groupby([col, "AF"], as_index=False).agg(**eagg); e["grouping"] = grouping; e["group"] = e[col]
        cparts.append(g[["grouping", "group", "AF", "n_tasks", "highest_mean", "competitive"]])
        eparts.append(e[["grouping", "group", "AF", "mean_reactions", "mean_pct_pool"]])
    s = pd.concat(cparts, ignore_index=True).merge(pd.concat(eparts, ignore_index=True), on=["grouping", "group", "AF"], how="left")
    s["pct_competitive"] = 100 * s["competitive"] / s["n_tasks"]
    return s


def rh_partition(cls, per_seed, nesting, fc, cio, V) -> tuple[pd.DataFrame, dict]:
    rh_tasks = list(fc.RH_PROTOCOL_MATCHED_TASKS)
    disp = V.DISPLAY
    rows = []
    for scheme in SCHEMES:
        for col, _lo, _hi, lab in cio.WINDOWS:
            for task in rh_tasks:
                mat = cio.analysis_unit_matrix(per_seed[per_seed.split_id == task], col, nesting, scheme=scheme)
                res = V.classify_task(mat, rank_rounded=True)
                for acq in cio.ACQS:
                    r = res["cls"][acq]
                    rows.append({"scheme": scheme, "window": lab, "split_id": task, "acquisition_function": disp[cio.LABEL[acq]],
                                 "mean_auc": r["mean_auc"], "highest_mean": bool(r["is_highest_mean"]),
                                 "statistically_competitive": bool(r["not_detected_worse"])})
    rc = pd.DataFrame(rows)
    out = []
    for lab in [w[3] for w in cio.WINDOWS]:
        other = cls[(cls.window == lab) & (~cls.split_id.isin(rh_tasks))]
        if other["split_id"].nunique() != 111:
            raise SystemExit("expected 111 non-Rh-Published tasks")
        bh, bc = other.groupby("AF")["is_highest_mean"].sum(), other.groupby("AF")["not_detected_worse"].sum()
        for scheme in SCHEMES:
            g = rc[(rc.window == lab) & (rc.scheme == scheme)]
            for af in V.LABEL_ORDER:
                sub = g[g.acquisition_function == disp[af]]
                out.append({"window": lab, "scheme": scheme, "acquisition_function": disp[af],
                            "rh_highest_mean": int(sub.highest_mean.sum()), "rh_competitive": int(sub.statistically_competitive.sum()),
                            "overall_highest_mean": int(bh[af] + sub.highest_mean.sum()),
                            "overall_competitive": int(bc[af] + sub.statistically_competitive.sum()), "overall_denominator": 118})
    counts = pd.DataFrame(out)
    key = ["window", "split_id", "acquisition_function"]
    piv = rc.pivot_table(index=key, columns="scheme", values="mean_auc")
    base = rc[rc.scheme == "primary"].set_index(key)
    n_changes = 0
    for scheme, g in rc.groupby("scheme", sort=False):
        if scheme == "primary":
            continue
        g = g.set_index(key)
        n_changes += int(((g.statistically_competitive != base.loc[g.index, "statistically_competitive"])
                          | (g.highest_mean != base.loc[g.index, "highest_mean"])).sum())
    pub = cls[cls.split_id.isin(rh_tasks)]
    mism = sum(bool(base.loc[(r.window, r.split_id, disp[r.AF]), "statistically_competitive"]) != bool(r.not_detected_worse)
               or bool(base.loc[(r.window, r.split_id, disp[r.AF]), "highest_mean"]) != bool(r.is_highest_mean)
               for r in pub.itertuples())
    labs = [w[3] for w in cio.WINDOWS]
    summ = {
        "windows": labs, "rh_tasks": rh_tasks, "nesting_source": "results/rh_published_split/partition_map.csv",
        "primary_scheme": "primary (seed_units_10): ten repetition-seed units, each the mean of the three fixed 96/96 partitions",
        "si_sensitivity_schemes": SCHEMES[1:], "primary_reproduces_published": mism == 0,
        "max_abs_mean_delta_primary_vs_pooled30": float((piv["primary"] - piv["pooled_30"]).abs().max()),
        "n_highest_mean_differences_primary_vs_pooled30": int((rc[rc.scheme == "primary"].set_index(key).highest_mean
                                                               != rc[rc.scheme == "pooled_30"].set_index(key).highest_mean).sum()),
        "n_classification_changes_vs_primary": n_changes,
        "rh_competitive_cells_of_35": {s: {w: int(counts[(counts.scheme == s) & (counts.window == w)].rh_competitive.sum()) for w in labs}
                                       for s in SCHEMES},
        "overall_competitive_a2_12": {s: {r.acquisition_function: int(r.overall_competitive)
                                          for r in counts[(counts.scheme == s) & (counts.window == "2-12")].itertuples()} for s in SCHEMES},
        "overall_highest_mean_a2_12": {s: {r.acquisition_function: int(r.overall_highest_mean)
                                           for r in counts[(counts.scheme == s) & (counts.window == "2-12")].itertuples()} for s in SCHEMES},
        "note": "primary is the published basis; the others are SI sensitivities",
    }
    return counts, summ


def negative_ceiling(cls: pd.DataFrame) -> pd.DataFrame:
    p = cls[cls.window.eq("2-12")]
    wide = p.pivot_table(index=["split_id", "dataset", "split_family"], columns="AF", values="mean_auc").reset_index()
    methods = ["EPIG", "BALD", "IndvNN", "NN", "Random"]
    wide["task_ceiling"] = wide[methods].to_numpy(float).max(axis=1)
    wide["epig_margin_over_alt"] = wide["EPIG"].to_numpy(float) - wide[["BALD", "IndvNN", "NN", "Random"]].to_numpy(float).max(axis=1)
    wide["epig_strict_win"] = wide["epig_margin_over_alt"] > 0
    return wide.sort_values(["dataset", "split_id"])[["split_id", "dataset", "split_family", "task_ceiling",
                                                      "epig_margin_over_alt", "epig_strict_win"]]


def table1(cls: pd.DataFrame, sv: pd.DataFrame, V) -> pd.DataFrame:
    cnt = V.counts_from(cls[cls.window_col.eq("auc_2_12")]).set_index("AF")
    prim = sv[sv.structure.eq("cluster")].set_index(["metric", "method"])["rounded"]

    rows = []
    for af in V.LABEL_ORDER:
        n = int(cnt.loc[af, "n_tasks"])
        cell = lambda k: f"{int(cnt.loc[af, k])}/{n} ({int(round(100.0 * cnt.loc[af, k] / n))}%)"
        ref = af == "Random"
        rows.append({"Acquisition function": V.DISPLAY[af],
                     "Highest mean AUC2–12, n/N (%)": cell("highest_mean"),
                     "Statistically competitive, n/N (%)": cell("competitive"),
                     "Mean reactions saved relative to Random [95% CI]": "Reference" if ref else prim[("n_reactions", af)].replace("-", MINUS),
                     "% Acquisition pool saved relative to Random [95% CI]": "Reference" if ref else prim[("pct_pool", af)].replace("-", MINUS)})
    return pd.DataFrame(rows)


def generate(repo: Path, out: Path, eff_path: Path | None = None) -> dict:
    fc, cio, V = load_repo(repo)
    cls, per_seed, nesting = classification(cio, V)
    (out / "results/benchmark").mkdir(parents=True, exist_ok=True)
    cls.to_csv(out / "results/benchmark/per_task_classification.csv", index=False)
    V.counts_from(cls).to_csv(out / "results/benchmark/counts_by_window.csv", index=False)
    (cls.groupby(["window", "dataset", "AF"]).agg(competitive=("not_detected_worse", "sum"), highest=("is_highest_mean", "sum"),
                                                   n=("split_id", "nunique")).reset_index()
        .to_csv(out / "results/benchmark/counts_by_dataset.csv", index=False))
    eff = pd.read_csv(eff_path or repo / "results/efficiency/efficiency_per_task.csv")
    group_summary(cls, eff).to_csv(out / "results/benchmark/group_summary.csv", index=False)
    # SI tables are assembled from the classification table as written to and read back from CSV
    si_tables(pd.read_csv(out / "results/benchmark/per_task_classification.csv"), V, out)
    counts, summ = rh_partition(cls, per_seed, nesting, fc, cio, V)
    (out / "results/sensitivity/rh_partition").mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / "results/sensitivity/rh_partition/counts.csv", index=False)
    with open(out / "results/sensitivity/rh_partition/summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    (out / "results/efficiency").mkdir(parents=True, exist_ok=True)
    # like the SI tables, assembled from the classification table as written to and read back from CSV
    negative_ceiling(pd.read_csv(out / "results/benchmark/per_task_classification.csv")).to_csv(
        out / "results/efficiency/negative_ceiling_margin.csv", index=False)
    sv = pd.read_csv(repo / "results/efficiency/savings_intervals_100k_by_structure.csv")
    t1 = table1(cls, sv, V)
    (out / "results/main_results_table.csv").write_text(MAIN_TABLE_HEADER + t1.to_csv(index=False, header=False))
    return {"n_tasks": int(cls.split_id.nunique()), "rows": int(len(cls))}


def compare(out: Path, ref: Path) -> dict:
    res = {}
    for rel in OUTPUTS:
        a, b = out / rel, ref / rel
        same = a.read_bytes() == b.read_bytes()
        entry = {"byte_identical": same}
        if not same and rel.endswith(".csv"):
            x = pd.read_csv(a, float_precision="round_trip"); y = pd.read_csv(b, float_precision="round_trip")
            entry["shape"] = [list(x.shape), list(y.shape)]
            if x.shape == y.shape and list(x.columns) == list(y.columns):
                num = [c for c in x.columns if pd.api.types.is_float_dtype(x[c])]
                oth = [c for c in x.columns if c not in num]
                entry["non_float_identical"] = bool(x[oth].astype(str).equals(y[oth].astype(str)))
                entry["float_max_abs_diff"] = float(max([np.nanmax(np.abs(x[c].to_numpy(float) - y[c].to_numpy(float))) for c in num] or [0.0]))
        res[rel] = entry
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temporary directory and compare with the committed tables")
    a = ap.parse_args()
    if not a.check:
        info = generate(REPO, REPO)
        print(json.dumps({**info, "written": OUTPUTS}, indent=1))
        return 0
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        info = generate(REPO, Path(tmp))
        res = compare(Path(tmp), REPO)
    bad = {k: v for k, v in res.items() if not v["byte_identical"]}
    print(json.dumps({**info, "compared": len(res), "differing": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
