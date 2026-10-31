#!/usr/bin/env python
"""Recompute the manuscript benchmark summaries from the released inputs.

This is the reproduction entry point for the per-task summaries: **highest mean
windowed AUC** and **statistical competitiveness** (exact paired two-sided
Wilcoxon signed-rank with a within-task Holm correction), recomputed from the
released per-seed window-AUC inputs and validated against every released
summary table in this repository.

What it does, per task and window:

  * builds the primary paired analysis-unit matrix using
    ``scripts/analysis/benchmark_tasks.py`` and
    ``scripts/analysis/inputs.py``: ten matched paired units
    per task -- the seven protocol-matched Rh tasks carry 30 nested replicates
    (three fixed partitions x ten repetition seeds) and are reduced to ten
    seed-level units by averaging the three partition values within each
    repetition seed; every other task has ten matched repetitions natively;
  * computes the five method means; the highest-mean method(s) are those whose
    mean equals the task maximum exactly;
  * runs all C(5,2) = 10 pairwise exact two-sided Wilcoxon signed-rank tests
    (``scripts/analysis/stats.py``, exact DP engine): paired
    differences rounded to 12 decimals before zero removal and tie ranking,
    exact-zero differences discarded, average ranks on ties;
  * Holm-adjusts the ten-member within-task family (alpha = 0.05);
  * classifies: detected worse = mean below the task maximum AND Holm-adjusted
    two-sided p vs a highest-mean method < 0.05; statistically competitive =
    not detected worse. Nonsignificance is never reported as a tie or as
    equivalence;
  * reruns the classification ranking the UNROUNDED differences (sensitivity)
    and requires the competitive sets to be identical.

Validation targets (all committed in this repository; nothing is hard-coded):

  * results/benchmark/
    per_task_classification.csv  (+ counts_by_window.csv)
  * supporting_information/tables/competitiveness/
    task_level_classifications_a2_12.csv (+ the six wide count tables)
  * results/main_results_table.csv
    (the released main-results table; 2-12 window columns)

The script recomputes classifications from the released results and checks them
against the committed summary tables. Exit code 0 means every per-task
classification, every count, and the task-identity sets match the released
tables exactly.

Usage (from the repository root):

    python scripts/verify_results.py                # all seven windows
    python scripts/verify_results.py --window 2-12  # primary window only
    python scripts/verify_results.py --write-dir OUTDIR --json report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_ANALYSIS = _SCRIPTS / "analysis"
for _p in (str(_ANALYSIS), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inputs as cio  # noqa: E402
import benchmark_tasks as fc  # noqa: E402
from stats import (  # noqa: E402
    ALPHA,
    exact_wilcoxon_signed_rank,
    friedman_diagnostic,
    holm_adjust,
)

REPO = fc.REPO
ACQS = list(cio.ACQS)
LABEL = dict(cio.LABEL)
LABEL_ORDER = list(cio.LABEL_ORDER)
DISPLAY = {"EPIG": "EPIG", "BALD": "BALD", "IndvNN": "Indiv. NN", "NN": "NN", "Random": "Random"}
PAIRS = [(a, b) for a, b in combinations(ACQS, 2)]
WINDOWS = list(cio.WINDOWS)
WINDOW_LABEL = dict(cio.WINDOW_LABEL)
PRIMARY = cio.PRIMARY_WINDOW_COL  # auc_2_12

SENSITIVITY_DIR = REPO / "results/benchmark"
SI_DIR = REPO / "supporting_information/tables/competitiveness"
MAIN_TABLE = REPO / "results/main_results_table.csv"


# --------------------------------------------------------------------------- #
#  recomputation
# --------------------------------------------------------------------------- #

def classify_task(mat: pd.DataFrame, rank_rounded: bool = True) -> dict:
    """Highest-mean + Wilcoxon-Holm competitiveness for one task's unit matrix."""
    means = {acq: float(mat[acq].mean()) for acq in ACQS}
    top_mean = max(means.values())
    highest_set = {acq for acq in ACQS if means[acq] == top_mean}
    # deterministic single reference among the highest-mean methods
    top_method = min(highest_set, key=lambda a: LABEL_ORDER.index(LABEL[a]))

    wil_raw = []
    pair_stats: dict[tuple[str, str], dict] = {}
    for a, b in PAIRS:
        diffs = mat[a].to_numpy(float) - mat[b].to_numpy(float)
        w = exact_wilcoxon_signed_rank(diffs, rank_rounded=rank_rounded, engine="dp")
        pair_stats[(a, b)] = w
        wil_raw.append(w["p_value"])
    wil_holm = holm_adjust(wil_raw)
    holm_by_pair = {pair: float(wil_holm[k]) for k, pair in enumerate(PAIRS)}

    def holm_vs_top(acq: str) -> float:
        if (top_method, acq) in holm_by_pair:
            return holm_by_pair[(top_method, acq)]
        return holm_by_pair[(acq, top_method)]  # two-sided p is orientation-invariant

    cls = {}
    for acq in ACQS:
        is_top = acq in highest_set
        gap = top_mean - means[acq]
        if is_top:
            rec = {
                "mean_auc": means[acq],
                "gap_to_best": gap,
                "is_highest_mean": True,
                "detected_worse": False,
                "not_detected_worse": True,
                "holm_wilcoxon_p_vs_top": float("nan"),
                "ref_vs": LABEL[top_method],
            }
        else:
            holm_p = holm_vs_top(acq)
            lower = means[acq] < top_mean
            detected = bool(lower and holm_p < ALPHA)
            rec = {
                "mean_auc": means[acq],
                "gap_to_best": gap,
                "is_highest_mean": False,
                "detected_worse": detected,
                "not_detected_worse": not detected,
                "holm_wilcoxon_p_vs_top": float(holm_p),
                "ref_vs": LABEL[top_method],
            }
        cls[acq] = rec

    fried = friedman_diagnostic(mat[ACQS].to_numpy(float))
    return {"cls": cls, "friedman": fried}


def run_window(per_seed: pd.DataFrame, window_col: str, nesting: pd.DataFrame):
    """Recompute the per-task classification table for one window."""
    rows, mismatches = [], []
    tasks = per_seed[["dataset", "split_id", "split_family"]].drop_duplicates()
    for _, t in tasks.iterrows():
        d, sid, fam = t["dataset"], t["split_id"], t["split_family"]
        trows = per_seed[(per_seed.dataset == d) & (per_seed.split_id == sid)]
        mat = cio.analysis_unit_matrix(trows, window_col, nesting, scheme="primary")

        res = classify_task(mat, rank_rounded=True)     # primary convention
        res_u = classify_task(mat, rank_rounded=False)  # unrounded sensitivity
        set_r = {a for a in ACQS if res["cls"][a]["not_detected_worse"]}
        set_u = {a for a in ACQS if res_u["cls"][a]["not_detected_worse"]}
        if set_r != set_u:
            mismatches.append({
                "window_col": window_col, "dataset": d, "split_id": sid,
                "rounded": sorted(LABEL[a] for a in set_r),
                "unrounded": sorted(LABEL[a] for a in set_u),
            })

        for a in ACQS:
            r = res["cls"][a]
            rows.append({
                "window_col": window_col,
                "window": WINDOW_LABEL[window_col],
                "dataset": d,
                "split_family": fam,
                "split_id": sid,
                "AF": LABEL[a],
                "n_analysis_units": int(mat.shape[0]),
                "unit_scheme": "primary",
                "mean_auc": r["mean_auc"],
                "gap_to_best": r["gap_to_best"],
                "is_highest_mean": bool(r["is_highest_mean"]),
                "detected_worse": bool(r["detected_worse"]),
                "not_detected_worse": bool(r["not_detected_worse"]),
                "holm_wilcoxon_p_vs_top": r["holm_wilcoxon_p_vs_top"],
                "ref_vs": r["ref_vs"],
                "friedman_chi2": res["friedman"]["statistic"],
                "friedman_p": res["friedman"]["p_value"],
            })
    return pd.DataFrame(rows), mismatches


def counts_from(cls: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (wcol, wlab), g in cls.groupby(["window_col", "window"], sort=False):
        n = g["split_id"].nunique()
        for af in LABEL_ORDER:
            s = g[g.AF == af]
            out.append({
                "window": wlab, "window_col": wcol, "AF": af, "n_tasks": n,
                "highest_mean": int(s["is_highest_mean"].sum()),
                "competitive": int(s["not_detected_worse"].sum()),
                "detected_worse": int(s["detected_worse"].sum()),
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
#  validation against the released tables
# --------------------------------------------------------------------------- #

def _read_committed(path: Path) -> pd.DataFrame:
    """Read a committed CSV with exact (round-trip) float parsing.

    pandas' default float parser can be one ULP off the shortest-round-trip
    text written by ``to_csv``; ``float_precision="round_trip"`` parses the
    committed text back to the exact float64, so recomputed values can be
    compared for bit-exact equality.
    """
    return pd.read_csv(path, float_precision="round_trip")


def _float_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.to_numpy(float)
    b = b.to_numpy(float)
    return pd.Series((a == b) | (np.isnan(a) & np.isnan(b)))


def validate_per_task(recomputed: pd.DataFrame, problems: list) -> dict:
    """Exact per-task comparison against the committed classification table."""
    committed = _read_committed(SENSITIVITY_DIR / "per_task_classification.csv")
    windows_done = sorted(recomputed["window_col"].unique())
    committed = committed[committed["window_col"].isin(windows_done)].copy()

    key = ["window_col", "dataset", "split_id", "AF"]
    rec_ids = set(map(tuple, recomputed[["dataset", "split_id"]].drop_duplicates().to_numpy()))
    com_ids = set(map(tuple, committed[["dataset", "split_id"]].drop_duplicates().to_numpy()))
    if rec_ids != com_ids:
        problems.append(
            "task identities differ from committed classification: "
            f"missing={sorted(com_ids - rec_ids)}, extra={sorted(rec_ids - com_ids)}"
        )
    merged = recomputed.merge(committed, on=key, suffixes=("", "_committed"), how="outer",
                              indicator=True)
    off = merged[merged["_merge"] != "both"]
    if len(off):
        problems.append(f"{len(off)} per-task rows do not pair 1:1 with the committed table")
        merged = merged[merged["_merge"] == "both"]

    exact_cols = ["split_family", "n_analysis_units", "unit_scheme",
                  "is_highest_mean", "detected_worse", "not_detected_worse", "ref_vs"]
    for col in exact_cols:
        bad = merged[merged[col] != merged[f"{col}_committed"]]
        if len(bad):
            problems.append(
                f"{col}: {len(bad)} per-task mismatches vs committed, e.g. "
                + "; ".join(f"{r.window_col}/{r.split_id}/{r.AF}" for r in bad.head(5).itertuples())
            )
    float_cols = ["mean_auc", "gap_to_best", "holm_wilcoxon_p_vs_top",
                  "friedman_chi2", "friedman_p"]
    float_report = {}
    for col in float_cols:
        eq = _float_equal(merged[col], merged[f"{col}_committed"])
        n_bad = int((~eq).sum())
        with np.errstate(invalid="ignore"):
            max_abs = float(np.nanmax(np.abs(
                merged[col].to_numpy(float) - merged[f"{col}_committed"].to_numpy(float)
            ))) if len(merged) else 0.0
        float_report[col] = {"n_unequal": n_bad, "max_abs_diff": max_abs}
        if n_bad:
            problems.append(f"{col}: {n_bad} values differ from committed (max abs {max_abs:g})")
    return {"rows_compared": int(len(merged)), "float_columns": float_report}


def validate_si_a2_12(recomputed: pd.DataFrame, problems: list) -> dict:
    """Comparison of the 2-12 slice against the committed SI task-level table.

    Every statistical statement in the SI table -- the classification columns,
    the analysis-unit design, the reference method, and the Holm-adjusted
    p-values -- must match the recomputation EXACTLY. The four float display
    columns (mean_auc, gap_to_best, friedman_chi2, friedman_p) are additionally
    compared bit-for-bit; where they differ, the difference must be exactly the
    pre-existing last-ULP float-text divergence between the two released
    tables themselves (the SI export was written by a separate upstream
    export path from the classification table; the two committed files
    disagree at <= 2e-14 in these display columns while agreeing exactly on
    every classification and p-value). Any deviation beyond that committed
    divergence is a failure.
    """
    if "auc_2_12" not in set(recomputed["window_col"]):
        return {"rows": 0}
    si = _read_committed(SI_DIR / "task_level_classifications_a2_12.csv")
    sens = _read_committed(SENSITIVITY_DIR / "per_task_classification.csv")
    sens = sens[sens["window_col"] == "auc_2_12"].copy()
    sens["acquisition_function"] = sens["AF"].map(DISPLAY)

    rec = recomputed[recomputed["window_col"] == "auc_2_12"].copy()
    rec["acquisition_function"] = rec["AF"].map(DISPLAY)
    # the SI export writes ref_vs with display names (e.g. "Indiv. NN"),
    # the classification table with internal labels (e.g. "IndvNN")
    rec["ref_vs"] = rec["ref_vs"].map(lambda s: DISPLAY.get(s, s))
    merged = rec.merge(
        si, on=["split_id", "acquisition_function"], suffixes=("", "_si"), how="outer",
        indicator=True,
    )
    off = merged[merged["_merge"] != "both"]
    if len(off):
        problems.append(f"SI a2-12: {len(off)} rows do not pair 1:1 with the committed SI table")
        merged = merged[merged["_merge"] == "both"]
    merged = merged.merge(
        sens[["split_id", "acquisition_function", "mean_auc", "gap_to_best",
              "friedman_chi2", "friedman_p"]],
        on=["split_id", "acquisition_function"], suffixes=("", "_sens"),
    )

    checks = [
        ("is_highest_mean", "highest_mean"),
        ("detected_worse", "detected_worse_si"),
        ("not_detected_worse", "statistically_competitive"),
        ("n_analysis_units", "n_analysis_units_si"),
        ("unit_scheme", "unit_scheme_si"),
        ("ref_vs", "ref_vs_si"),
    ]
    for mine, theirs in checks:
        col = theirs if theirs in merged.columns else theirs.removesuffix("_si")
        bad = merged[merged[mine] != merged[col]]
        if len(bad):
            problems.append(f"SI a2-12 {mine}: {len(bad)} mismatches, e.g. "
                            + "; ".join(f"{r.split_id}/{r.acquisition_function}"
                                        for r in bad.head(5).itertuples()))
    # the Holm-adjusted p-values are statistical content: exact, no exceptions
    eq_p = _float_equal(merged["holm_wilcoxon_p_vs_top"], merged["holm_wilcoxon_p_vs_top_si"])
    if int((~eq_p).sum()):
        problems.append(
            f"SI a2-12 holm_wilcoxon_p_vs_top: {int((~eq_p).sum())} p-values differ"
        )

    # float display columns: bit-equal, or exactly the committed<->committed
    # last-ULP divergence (recomputed value must equal the classification
    # committed value bit-for-bit in every such cell)
    float_note = {}
    for col in ("mean_auc", "gap_to_best", "friedman_chi2", "friedman_p"):
        mine = merged[col].to_numpy(float)
        si_v = merged[f"{col}_si"].to_numpy(float)
        sens_v = merged[f"{col}_sens"].to_numpy(float)
        eq_si = (mine == si_v) | (np.isnan(mine) & np.isnan(si_v))
        eq_sens = (mine == sens_v) | (np.isnan(mine) & np.isnan(sens_v))
        unexplained = ~eq_si & ~eq_sens
        if int(unexplained.sum()):
            problems.append(
                f"SI a2-12 {col}: {int(unexplained.sum())} values differ beyond the "
                "pre-existing committed float-text divergence"
            )
        n_ulp = int((~eq_si & eq_sens).sum())
        with np.errstate(invalid="ignore"):
            max_abs = float(np.nanmax(np.abs(mine - si_v))) if n_ulp else 0.0
        float_note[col] = {"pre_existing_ulp_cells": n_ulp, "max_abs_diff": max_abs}
    return {"rows": int(len(merged)), "si_display_float_note": float_note}


def _parse_n(cell: str) -> int:
    return int(str(cell).split("/")[0])


def validate_si_counts(cnt: pd.DataFrame, problems: list) -> int:
    """Compare recomputed counts against the six committed wide SI count tables."""
    scope_col = {"overall": None, "dataset": "dataset", "split_family": "split_family"}
    si_scope_values = {
        "overall": {"All tasks": None},
        "dataset": {"BH": "BH", "Rh": "Rh", "SAD": "SAD", "Wittig": "Wittig"},
        "split_family": {
            "Random": "true_random",
            "Cluster": "cluster_based",
            "Chemical family": "chemistry_class_or_substrate_holdout",
            "Published": "paper_defined",
        },
    }
    n_checked = 0
    for scope_kind in ("overall", "dataset", "split_family"):
        for stem, value_col in (("highest_mean_by", "highest_mean"),
                                ("competitive_by", "competitive")):
            path = SI_DIR / f"{stem}_{scope_kind}.csv"
            table = pd.read_csv(path)
            for _, row in table.iterrows():
                af_display = row["acquisition_function"]
                af = {v: k for k, v in DISPLAY.items()}[af_display]
                for wlab in cnt["window"].unique():
                    if wlab not in table.columns:
                        continue
                    sub = cnt[(cnt["window"] == wlab) & (cnt["AF"] == af)]
                    if scope_kind == "overall":
                        expected = _parse_n(row[wlab])
                        got = int(sub[value_col].iloc[0])
                    else:
                        continue  # scoped counts handled below from the per-task table
                    n_checked += 1
                    if got != expected:
                        problems.append(
                            f"{path.name} {af_display} {wlab}: recomputed {got} != committed {expected}"
                        )
    return n_checked


def validate_scoped_counts(recomputed: pd.DataFrame, problems: list) -> int:
    """Dataset- and split-family-scoped counts vs the committed wide SI tables."""
    fam_display = {
        "true_random": "Random",
        "cluster_based": "Cluster",
        "chemistry_class_or_substrate_holdout": "Chemical family",
        "paper_defined": "Published",
    }
    n_checked = 0
    for scope_kind, group_col, to_scope in (
        ("dataset", "dataset", lambda v: v),
        ("split_family", "split_family", lambda v: fam_display[v]),
    ):
        for stem, flag in (("highest_mean_by", "is_highest_mean"),
                           ("competitive_by", "not_detected_worse")):
            table = pd.read_csv(SI_DIR / f"{stem}_{scope_kind}.csv")
            lookup = {(r["scope"], r["acquisition_function"]): r for _, r in table.iterrows()}
            for (gval, wlab, af), sub in recomputed.groupby([group_col, "window", "AF"]):
                scope = to_scope(gval)
                row = lookup.get((scope, DISPLAY[af]))
                if row is None or wlab not in table.columns:
                    continue
                got = int(sub[flag].sum())
                expected = _parse_n(row[wlab])
                n_checked += 1
                if got != expected:
                    problems.append(
                        f"{stem}_{scope_kind} {scope}/{DISPLAY[af]}/{wlab}: "
                        f"recomputed {got} != committed {expected}"
                    )
    return n_checked


def validate_committed_counts_table(cnt: pd.DataFrame, problems: list) -> int:
    committed = pd.read_csv(SENSITIVITY_DIR / "counts_by_window.csv")
    committed = committed[committed["window_col"].isin(set(cnt["window_col"]))]
    merged = cnt.merge(committed, on=["window_col", "window", "AF"],
                       suffixes=("", "_committed"), how="outer", indicator=True)
    off = merged[merged["_merge"] != "both"]
    if len(off):
        problems.append(f"counts_by_window.csv: {len(off)} rows do not pair 1:1")
        merged = merged[merged["_merge"] == "both"]
    for col in ("n_tasks", "highest_mean", "competitive", "detected_worse"):
        bad = merged[merged[col] != merged[f"{col}_committed"]]
        if len(bad):
            problems.append(f"counts_by_window.csv {col}: {len(bad)} mismatches")
    return int(len(merged))


def validate_main_table(cnt: pd.DataFrame, problems: list) -> int:
    """The released main-results table's 2-12 classification columns."""
    if "2-12" not in set(cnt["window"]):
        return 0
    table = pd.read_csv(MAIN_TABLE)
    hm_col = next((c for c in table.columns if c.startswith("Highest mean")), None)
    comp_col = next((c for c in table.columns if c.startswith("Statistically competitive")), None)
    if hm_col is None or comp_col is None:
        problems.append("results/main_results_table.csv lacks the expected columns")
        return 0
    sub = cnt[cnt["window"] == "2-12"].set_index("AF")
    n_checked = 0
    for _, row in table.iterrows():
        af = {v: k for k, v in DISPLAY.items()}[row["Acquisition function"]]
        n_tasks = int(sub.loc[af, "n_tasks"])
        for col, value_col in ((hm_col, "highest_mean"), (comp_col, "competitive")):
            got = int(sub.loc[af, value_col])
            expected = _parse_n(row[col])
            pct = int(round(100.0 * got / n_tasks))
            expected_cell = f"{got}/{n_tasks} ({pct}%)"
            n_checked += 1
            if got != expected or str(row[col]) != expected_cell:
                problems.append(
                    f"main table {row['Acquisition function']} {col!r}: recomputed "
                    f"{expected_cell!r} != committed {row[col]!r}"
                )
    return n_checked


# --------------------------------------------------------------------------- #
#  entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", default=None, choices=[w[3] for w in WINDOWS],
                        help="restrict to one window label (default: all seven)")
    parser.add_argument("--write-dir", type=Path, default=None,
                        help="optionally write the recomputed tables to this directory")
    parser.add_argument("--json", type=Path, default=None,
                        help="optionally write a machine-readable verification record")
    args = parser.parse_args(argv)

    print("loading census via benchmark_tasks.py and inputs.py ...")
    per_seed = cio.load_per_seed_full()
    pairing = cio.verify_pairing(per_seed)
    print(f"  tasks: {pairing['n_tasks']}  "
          f"replicates: {pairing['seed_count_distribution']}  "
          f"identities match ledger: {pairing['task_identities_match_ledger']}")
    nesting = cio.load_nesting()

    window_cols = [w[0] for w in WINDOWS]
    if args.window:
        window_cols = [w[0] for w in WINDOWS if w[3] == args.window]

    all_cls, all_mm = [], []
    for wcol in window_cols:
        cls_w, mm = run_window(per_seed, wcol, nesting)
        all_cls.append(cls_w)
        all_mm.extend(mm)
        print(f"  recomputed {WINDOW_LABEL[wcol]:6s} ({len(cls_w)} rows)")
    recomputed = pd.concat(all_cls, ignore_index=True)
    cnt = counts_from(recomputed)

    problems: list[str] = []
    if all_mm:
        problems.append(
            f"{len(all_mm)} rounding-sensitivity mismatches (12-dp-ranked vs unrounded): {all_mm}"
        )

    per_task_report = validate_per_task(recomputed, problems)
    si_report = validate_si_a2_12(recomputed, problems)
    n_counts = validate_committed_counts_table(cnt, problems)
    n_overall = validate_si_counts(cnt, problems)
    n_scoped = validate_scoped_counts(recomputed, problems)
    n_main = validate_main_table(cnt, problems)

    print("\n=== recomputed HIGHEST MEAN (n/118) ===")
    high = cnt.pivot_table(index="AF", columns="window", values="highest_mean",
                           aggfunc="first").reindex(LABEL_ORDER)
    print(high.to_string())
    print("\n=== recomputed STATISTICALLY COMPETITIVE (n/118) ===")
    comp = cnt.pivot_table(index="AF", columns="window", values="competitive",
                           aggfunc="first").reindex(LABEL_ORDER)
    print(comp.to_string())

    print("\n=== validation against the released tables ===")
    print(f"  per-task rows compared (exact): {per_task_report['rows_compared']}")
    print(f"  SI a2-12 rows compared        : {si_report['rows']}")
    print(f"  counts rows compared          : {n_counts}")
    print(f"  SI overall count cells        : {n_overall}")
    print(f"  SI scoped count cells         : {n_scoped}")
    print(f"  main-table cells              : {n_main}")
    print(f"  rounding-sensitivity mismatches: {len(all_mm)}")

    if args.write_dir:
        args.write_dir.mkdir(parents=True, exist_ok=True)
        recomputed.to_csv(args.write_dir / "recomputed_per_task_classification.csv", index=False)
        cnt.to_csv(args.write_dir / "recomputed_counts.csv", index=False)
        print(f"\nwrote recomputed tables to {args.write_dir}")

    if args.json:
        record = {
            "windows": sorted({WINDOW_LABEL[w] for w in window_cols}),
            "n_tasks": int(pairing["n_tasks"]),
            "replicate_distribution": pairing["seed_count_distribution"],
            "census_inputs_sha256": pairing["census_audit"]["inputs_sha256"],
            "per_task_comparison": per_task_report,
            "si_a2_12": si_report,
            "counts_rows": n_counts,
            "overall_count_cells": n_overall,
            "scoped_count_cells": n_scoped,
            "main_table_cells": n_main,
            "rounding_sensitivity_mismatches": all_mm,
            "highest_mean": {af: {w: int(v) for w, v in high.loc[af].items()}
                             for af in LABEL_ORDER},
            "competitive": {af: {w: int(v) for w, v in comp.loc[af].items()}
                            for af in LABEL_ORDER},
            "problems": problems,
            "verdict": "PASS" if not problems else "FAIL",
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote verification record to {args.json}")

    if problems:
        print(f"\nREPRODUCTION FAILED ({len(problems)} problems):")
        for p in problems:
            print("  -", p)
        return 1
    print("\nREPRODUCTION PASSED: every recomputed classification, count, and task "
          "identity matches the released tables exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
