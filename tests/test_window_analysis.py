"""Locks for the seven-window competitiveness reporting.

Covers:
  * the 118-task benchmark and its dataset composition;
  * replicate counts -- 30 nested paired replicates on the seven Rh
    Published-Split tasks, 10 on every other task, with identically matched
    replicate ids;
  * the AUC2-12 results (highest mean AND statistically competitive) as
    published in `results/main_results_table.csv` and in the claim-to-file map
    in README.md;
  * exactness of the DP signed-rank engine against exhaustive enumeration,
    including ties and dropped zeros;
  * the seven-window count tables and the wide SI tables;
  * byte-identical re-derivation of the three SI data exports.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.benchmark_tasks import (
    WITTIG_TRANSFER_TASK,
    RH_PROTOCOL_MATCHED_TASKS,
    summary,
    expected_repetitions,
    load_per_seed,
)
from scripts.analysis import inputs

REPO = Path(__file__).resolve().parents[1]
SENS = REPO / "results/benchmark"
SI = REPO / "supporting_information/tables/competitiveness"
MAIN_RESULTS_CSV = REPO / "results/main_results_table.csv"

WINDOWS = ["0-10", "2-12", "5-15", "5-20", "5-30", "10-20", "10-30"]
DISPLAY_ORDER = ["EPIG", "BALD", "Indiv. NN", "NN", "Random"]

# --- the published AUC2-12 numbers ---
PUBLISHED_HIGHEST = {"EPIG": 74, "BALD": 22, "IndvNN": 12, "NN": 7, "Random": 3}
PUBLISHED_COMPETITIVE = {"EPIG": 105, "BALD": 79, "IndvNN": 74, "NN": 40, "Random": 64}
PUBLISHED_HIGHEST_PCT = {"EPIG": 63, "BALD": 19, "IndvNN": 10, "NN": 6, "Random": 3}
PUBLISHED_COMPETITIVE_PCT = {"EPIG": 89, "BALD": 67, "IndvNN": 63, "NN": 34, "Random": 54}
N_TASKS = 118
BY_DATASET = {"BH": 44, "Rh": 38, "SAD": 7, "Wittig": 29}

# --- the locked, fully reproducible reaction-savings values ---
REACTIONS_SUMMARY = REPO / "results/efficiency/reactions_saved_vs_random_summary.csv"
# mean, ci_lo, ci_hi at the precision the manuscript prints (whole reactions)
# published intervals: cluster bootstrap at 100,000 draws (mean, ci_lo, ci_hi)
PUBLISHED_REACTIONS = {
    "EPIG": (81, 46, 121),
    "BALD": (26, -19, 66),
    "IndvNN": (28, 10, 51),
    "NN": (-81, -249, 35),
    "Random": (0, 0, 0),
}
PUBLISHED_POOL = {           # one decimal place
    "EPIG": (6.9, 4.1, 9.9),
    "BALD": (4.4, 0.6, 8.7),
    "IndvNN": (2.3, 1.0, 3.9),
    "NN": (-2.5, -8.6, 3.5),
}
SAVINGS_PRIMARY = REPO / "results/efficiency/savings_intervals_100k_primary.csv"
SAVINGS_STRUCTURE = REPO / "results/efficiency/savings_intervals_100k_by_structure.csv"
SAVINGS_CONFIRM = REPO / "results/efficiency/savings_intervals_200k_confirmation.csv"
SAVINGS_DEPEND = REPO / "results/efficiency/savings_structure_dependence.csv"
REACTIONS_SEEDS = {"EPIG": 3382740998, "BALD": 2225935390, "IndvNN": 4193014655,
                   "NN": 1624363823, "Random": 1714580582}
@pytest.fixture(scope="module")
def cls() -> pd.DataFrame:
    return pd.read_csv(SENS / "per_task_classification.csv")


@pytest.fixture(scope="module")
def counts() -> pd.DataFrame:
    return pd.read_csv(SENS / "counts_by_window.csv")


# --------------------------------------------------------------------------- #
#  census
# --------------------------------------------------------------------------- #

def test_benchmark_summary_is_clean():
    a = summary()
    assert a["problems"] == []
    assert a["total_tasks"] == N_TASKS
    assert a["tasks_by_dataset"] == BY_DATASET


def test_loader_uses_exact_benchmark_task_identities():
    full = inputs.load_per_seed_full()
    final = load_per_seed()
    columns = ["dataset", "split_id", "split_family"]
    full_ids = set(map(tuple, full[columns].drop_duplicates().to_numpy()))
    final_ids = set(map(tuple, final[columns].drop_duplicates().to_numpy()))
    assert full_ids == final_ids
    assert WITTIG_TRANSFER_TASK in set(full["split_id"])
    wittig = full[full["split_id"].eq(WITTIG_TRANSFER_TASK)]
    assert len(wittig) == 50
    assert np.isfinite(wittig[inputs.WINDOW_COLS].to_numpy(float)).all()


def test_census_is_118_tasks_with_expected_dataset_composition(cls):
    tasks = cls[cls.window == "2-12"].drop_duplicates(["dataset", "split_id"])
    assert len(tasks) == N_TASKS
    assert tasks["dataset"].value_counts().to_dict() == BY_DATASET


def test_seven_protocol_matched_rh_tasks_are_locked():
    assert RH_PROTOCOL_MATCHED_TASKS == [f"rh_paper_ood_case{i}" for i in range(1, 8)]
    assert len(RH_PROTOCOL_MATCHED_TASKS) == 7


@pytest.mark.parametrize("split_id,expected", [
    *[(t, 30) for t in RH_PROTOCOL_MATCHED_TASKS],
    ("tr_rh_010", 10), ("bh_aryl_halide_holdout_00", 10),
])
def test_expected_repetitions(split_id, expected):
    assert expected_repetitions(split_id) == expected


def test_stored_replicate_counts_are_30_for_protocol_matched_and_10_otherwise():
    """The CENSUS stores 30 nested replicates on the seven protocol-matched Rh tasks."""
    from scripts.analysis.benchmark_tasks import load_per_seed
    ps = load_per_seed()
    reps = ps.groupby(["split_id", "acq"]).seed_id.nunique()
    for (sid, _acq), n in reps.items():
        assert n == expected_repetitions(str(sid)), (sid, n)


def test_every_task_contributes_ten_paired_analysis_units(cls):
    """The PRIMARY analysis unit is ten paired units for every task: the nested Rh
    tasks average their three partitions within each repetition seed."""
    assert set(cls["n_analysis_units"]) == {10}, set(cls["n_analysis_units"])
    assert set(cls["unit_scheme"]) == {"primary"}
    rh = set(RH_PROTOCOL_MATCHED_TASKS)
    assert cls[cls.split_id.isin(rh)]["split_id"].nunique() == 7
    assert cls["split_id"].nunique() == N_TASKS


def test_every_task_carries_all_five_methods_in_every_window(cls):
    g = cls.groupby(["window", "split_id"])["AF"].nunique()
    assert set(g) == {5}
    assert len(cls) == len(WINDOWS) * N_TASKS * 5 == 4130


# --------------------------------------------------------------------------- #
#  published AUC2-12 results
# --------------------------------------------------------------------------- #

def test_published_a2_12_highest_mean_counts(counts):
    c = counts[counts.window == "2-12"].set_index("AF")
    for af, n in PUBLISHED_HIGHEST.items():
        assert int(c.loc[af, "highest_mean"]) == n, af
    assert sum(PUBLISHED_HIGHEST.values()) == N_TASKS


def test_published_a2_12_competitive_counts(counts):
    c = counts[counts.window == "2-12"].set_index("AF")
    for af, n in PUBLISHED_COMPETITIVE.items():
        assert int(c.loc[af, "competitive"]) == n, af


def test_competitive_and_detected_worse_partition_every_task(counts):
    for _, r in counts.iterrows():
        assert int(r["competitive"]) + int(r["detected_worse"]) == int(r["n_tasks"])


def test_highest_mean_methods_are_never_detected_worse(cls):
    assert not cls[cls.is_highest_mean & cls.detected_worse].shape[0]


def test_exactly_one_highest_mean_method_per_task_and_window(cls):
    g = cls.groupby(["window", "split_id"])["is_highest_mean"].sum()
    assert set(g) == {1}


def test_main_results_csv_agrees_with_regenerated_counts(counts):
    c = counts[counts.window == "2-12"].set_index("AF")
    csv = pd.read_csv(MAIN_RESULTS_CSV)
    assert list(csv.columns) == [
        "Acquisition function",
        "Highest mean AUC2–12, n/N (%)",
        "Statistically competitive, n/N (%)",
        "Mean reactions saved relative to Random [95% CI]",
        "% Acquisition pool saved relative to Random [95% CI]",
    ]
    disp = {"EPIG": "EPIG", "BALD": "BALD", "IndvNN": "Indiv. NN",
            "NN": "NN", "Random": "Random"}
    assert list(csv["Acquisition function"]) == DISPLAY_ORDER
    for af, n in PUBLISHED_HIGHEST.items():
        cell = f"{n}/{N_TASKS} ({PUBLISHED_HIGHEST_PCT[af]}%)"
        assert int(c.loc[af, "highest_mean"]) == n
        row = csv[csv["Acquisition function"] == disp[af]].iloc[0]
        assert row["Highest mean AUC2–12, n/N (%)"] == cell
    for af, n in PUBLISHED_COMPETITIVE.items():
        cell = f"{n}/{N_TASKS} ({PUBLISHED_COMPETITIVE_PCT[af]}%)"
        assert int(c.loc[af, "competitive"]) == n
        row = csv[csv["Acquisition function"] == disp[af]].iloc[0]
        assert row["Statistically competitive, n/N (%)"] == cell


def test_reaction_savings_5000_draw_summary_is_deterministic_and_pinned():
    """The 5,000-draw table has the published means; published intervals use
    100,000 draws."""
    s = pd.read_csv(REACTIONS_SUMMARY).set_index("method")
    assert list(s.index) == ["EPIG", "BALD", "IndvNN", "NN", "Random"]
    assert set(s["n_tasks"]) == {N_TASKS}
    for af, seed in REACTIONS_SEEDS.items():
        assert int(s.loc[af, "bootstrap_seed"]) == seed, af
    for af, (mean, _lo, _hi) in PUBLISHED_REACTIONS.items():
        assert round(float(s.loc[af, "mean_reactions_saved"])) == mean, af


def test_published_intervals_come_from_the_100k_cluster_bootstrap():
    d = pd.read_csv(SAVINGS_PRIMARY)
    assert set(d["metric"]) == {"n_reactions", "pct_pool"}
    rx = d[d.metric == "n_reactions"].set_index("method")
    pool = d[d.metric == "pct_pool"].set_index("method")
    for af, (mean, lo, hi) in PUBLISHED_REACTIONS.items():
        assert round(float(rx.loc[af, "mean"])) == mean, af
        assert round(float(rx.loc[af, "ci_lo"])) == lo, af
        assert round(float(rx.loc[af, "ci_hi"])) == hi, af
    for af, (mean, lo, hi) in PUBLISHED_POOL.items():
        assert round(float(pool.loc[af, "mean"]), 1) == mean, af
        assert round(float(pool.loc[af, "ci_lo"]), 1) == lo, af
        assert round(float(pool.loc[af, "ci_hi"]), 1) == hi, af


def test_200k_confirmation_records_the_rounding_boundary_shifts():
    c = pd.read_csv(SAVINGS_CONFIRM)
    assert len(c) == 8, len(c)
    changed = c[~c["rounded_unchanged"]]
    assert set(zip(changed["metric"], changed["method"])) == {
        ("n_reactions", "NN"), ("pct_pool", "IndvNN")
    }
    # Both shifts are a single final display unit at a rounding boundary.
    nn = c[(c.metric == "n_reactions") & (c.method == "NN")].iloc[0]
    assert round(nn["lo_100k"]) == -249 and round(nn["lo_200k"]) == -248
    assert round(nn["hi_100k"]) == round(nn["hi_200k"]) == 35
    epig = c[(c.metric == "n_reactions") & (c.method == "EPIG")].iloc[0]
    assert (round(epig["lo_200k"]), round(epig["hi_200k"])) == (46, 121)


def test_all_three_resampling_structures_are_reported():
    d = pd.read_csv(SAVINGS_STRUCTURE)
    assert set(d["structure"]) == {"cluster", "task", "hierarchical"}
    assert set(d["draws"]) == {100_000}
    assert d.groupby(["metric", "method"]).size().eq(3).all()


def test_reaction_savings_resampling_structure_qualifications():
    """Check sensitivity of savings intervals to the resampling structure."""
    d = pd.read_csv(SAVINGS_STRUCTURE)
    rx = d[(d.metric == "n_reactions") & (d.method == "BALD")].set_index("structure")
    assert not rx["excludes_zero"].any(), rx.to_string()
    ind = d[(d.metric == "n_reactions") & (d.method == "IndvNN")].set_index("structure")
    assert bool(ind.loc["cluster", "excludes_zero"]) is True
    assert bool(ind.loc["task", "excludes_zero"]) is False
    assert bool(ind.loc["hierarchical", "excludes_zero"]) is False
    for method in ("BALD", "IndvNN"):
        pool = d[(d.metric == "pct_pool") & (d.method == method)]
        assert pool["excludes_zero"].all(), method
    dep = pd.read_csv(SAVINGS_DEPEND)
    flagged = set(zip(dep["metric"], dep["method"]))
    assert flagged == {("n_reactions", "IndvNN"), ("n_reactions", "NN"), ("pct_pool", "NN")}


def test_epig_savings_are_robust_to_the_resampling_structure():
    d = pd.read_csv(SAVINGS_STRUCTURE)
    for metric in ("n_reactions", "pct_pool"):
        g = d[(d.metric == metric) & (d.method == "EPIG")]
        assert g["excludes_zero"].all(), (metric, g.to_string())


def test_random_reaction_savings_row_is_identically_zero():
    s = pd.read_csv(REACTIONS_SUMMARY).set_index("method")
    for col in ("mean_reactions_saved", "sd_reactions_saved",
                "median_reactions_saved", "ci_lo", "ci_hi"):
        assert float(s.loc["Random", col]) == 0.0, col


def test_epig_reaction_interval_matches_the_published_row():
    s = pd.read_csv(REACTIONS_SUMMARY).set_index("method")
    canon = pd.read_csv(REPO / "results/efficiency/efficiency_summary_reactions.csv")
    row = canon[(canon.scope == "Overall") & (canon.AF == "Random")].iloc[0]
    assert float(s.loc["EPIG", "mean_reactions_saved"]) == pytest.approx(
        float(row["paired_diff_vs_epig"]), abs=1e-12)
    assert float(s.loc["EPIG", "ci_lo"]) == pytest.approx(
        float(row["paired_diff_ci_lo"]), abs=1e-12)
    assert float(s.loc["EPIG", "ci_hi"]) == pytest.approx(
        float(row["paired_diff_ci_hi"]), abs=1e-12)


def test_reaction_intervals_in_main_results_table_match_the_generator():
    csv = pd.read_csv(MAIN_RESULTS_CSV)
    col = "Mean reactions saved relative to Random [95% CI]"
    disp = {"EPIG": "EPIG", "BALD": "BALD", "IndvNN": "Indiv. NN",
            "NN": "NN", "Random": "Random"}
    for af, (mean, lo, hi) in PUBLISHED_REACTIONS.items():
        cell = "Reference" if af == "Random" else (
            f"{mean} [{lo}, {hi}]".replace("-", "−"))
        assert csv[csv["Acquisition function"] == disp[af]].iloc[0][col] == cell, af


def test_reaction_savings_bootstrap_seeds_are_pinned():
    """The published intervals are reproducible only if the deterministic
    bootstrap seeds are exactly these. The seeds derive from the literal scope
    string, so this also guards the scope labels against renaming."""
    summary = pd.read_csv(REACTIONS_SUMMARY)
    assert len(summary) == len(REACTIONS_SEEDS)
    got = {r["method"]: int(r["bootstrap_seed"]) for _, r in summary.iterrows()}
    assert got == REACTIONS_SEEDS
    for _, r in summary.iterrows():
        assert int(r["n_tasks"]) == N_TASKS
        assert "Overall" in r["bootstrap_seed_parts"]


def test_percentages_round_to_nearest_whole_percent():
    for table, pcts in ((PUBLISHED_HIGHEST, PUBLISHED_HIGHEST_PCT),
                        (PUBLISHED_COMPETITIVE, PUBLISHED_COMPETITIVE_PCT)):
        for af, n in table.items():
            assert pcts[af] == int(round(100.0 * n / N_TASKS)), af


# --------------------------------------------------------------------------- #
#  exact DP engine
# --------------------------------------------------------------------------- #

def _stats():
    import sys
    p = str(REPO / "scripts/analysis")
    if p not in sys.path:
        sys.path.insert(0, p)
    import stats
    return stats


@pytest.mark.parametrize("diffs", [
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [-1.0, -2.0, -3.0],
    [0.0, 0.0, 1.0, -1.0, 2.0],            # dropped exact zeros
    [1.0, 1.0, 1.0, 2.0, 2.0, -3.0],       # tied absolute differences
    [2.0] * 6 + [5.0, -5.0],               # even tie group -> half-integer ranks
    [0.0] * 5,                             # all zero
    [7.0],                                 # single pair
    [1.0, -1.0] * 5 + [2.0, -2.0] * 5,     # m = 20
])
def test_dp_matches_exhaustive_enumeration(diffs):
    sc = _stats()
    for rank_rounded in (False, True):
        dp = sc.exact_wilcoxon_signed_rank(
            diffs, rank_rounded=rank_rounded, engine="dp"
        )
        en = sc.exact_wilcoxon_signed_rank(
            diffs, rank_rounded=rank_rounded, engine="enum"
        )
        assert set(dp) == set(en)
        for k in dp:
            if k in ("p_value", "p_greater", "p_less"):
                assert dp[k] == en[k], (k, diffs)   # bit-identical
            else:
                assert dp[k] == pytest.approx(en[k], abs=1e-12), (k, diffs)


def test_all_zero_differences_give_p_equal_one():
    sc = _stats()
    r = sc.exact_wilcoxon_signed_rank([0.0, 0.0, 0.0], engine="dp")
    assert r["p_value"] == 1.0 and r["all_zero"] and r["n_nonzero"] == 0


def test_dp_null_distribution_is_exact_and_symmetric():
    sc = _stats()
    from scipy.stats import rankdata
    for m in (1, 5, 12, 30):
        ranks = rankdata(np.arange(1.0, m + 1), method="average")
        doubled = sc._doubled_ranks(ranks)
        counts = sc._signed_rank_counts(tuple(int(v) for v in doubled))
        assert sum(counts) == 2 ** m
        assert list(counts) == list(counts)[::-1]


def test_dp_handles_thirty_replicates_where_enumeration_is_refused():
    """m = 30 must work under DP and be refused by the enumeration guard."""
    sc = _stats()
    diffs = list(np.arange(1.0, 31.0))
    r = sc.exact_wilcoxon_signed_rank(diffs, engine="dp")
    assert r["n_nonzero"] == 30
    assert 0.0 < r["p_value"] <= 1.0
    assert r["p_value"] == pytest.approx(2.0 / 2 ** 30, rel=1e-12)
    with pytest.raises(ValueError):
        sc.exact_wilcoxon_signed_rank(diffs, engine="enum")


def test_holm_is_applied_across_ten_within_task_comparisons(cls):
    """Every non-highest-mean cell carries a Holm-adjusted p in [0, 1]."""
    sub = cls[~cls.is_highest_mean]
    p = sub["holm_wilcoxon_p_vs_top"].to_numpy(float)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()
    # detected worse iff Holm p < 0.05 (given the mean is below the top)
    assert (sub["detected_worse"] == (p < 0.05)).all()


# --------------------------------------------------------------------------- #
#  seven-window tables and SI artefacts
# --------------------------------------------------------------------------- #

def test_all_seven_windows_present_with_full_denominator(counts):
    assert sorted(counts["window"].unique()) == sorted(WINDOWS)
    assert set(counts["n_tasks"]) == {N_TASKS}


def test_wide_si_count_tables_agree_with_the_regenerated_counts(counts):
    """Every cell of the two overall wide SI tables must be the recomputed count."""
    disp = {"EPIG": "EPIG", "BALD": "BALD", "IndvNN": "Indiv. NN",
            "NN": "NN", "Random": "Random"}
    for col, fname in (("competitive", "competitive_by_overall.csv"),
                       ("highest_mean", "highest_mean_by_overall.csv")):
        wide = pd.read_csv(SI / fname).set_index("acquisition_function")
        for _, r in counts.iterrows():
            cell = str(wide.loc[disp[r["AF"]], r["window"]])
            n = int(cell.split("/", 1)[0])
            pct = int(cell.split("(")[1].rstrip("%)"))
            assert n == int(r[col]), (fname, r["AF"], r["window"], cell)
            assert pct == int(round(100.0 * n / N_TASKS)), (fname, cell)


def test_scoped_wide_si_tables_cover_every_window_and_method():
    for fname in ("competitive_by_dataset.csv", "highest_mean_by_dataset.csv",
                  "competitive_by_split_family.csv", "highest_mean_by_split_family.csv"):
        d = pd.read_csv(SI / fname)
        assert sorted(d["acquisition_function"].unique()) == sorted(DISPLAY_ORDER), fname
        assert [c for c in d.columns if c not in ("scope", "acquisition_function")] == WINDOWS, fname


def test_task_level_si_table_is_complete():
    t = pd.read_csv(SI / "task_level_classifications_all_windows.csv")
    assert len(t) == len(WINDOWS) * N_TASKS * 5 == 4130
    assert t["split_id"].nunique() == N_TASKS
    for c in ("highest_mean", "statistically_competitive", "n_analysis_units",
              "unit_scheme", "mean_auc"):
        assert c in t.columns
    a212 = pd.read_csv(SI / "task_level_classifications_a2_12.csv")
    assert len(a212) == N_TASKS * 5




# --------------------------------------------------------------------------- #
#  SI data exports re-derive byte-identically
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "builder,committed",
    [
        ("build_si_learning_curves.py", "data/all_benchmark_learning_curves.csv"),
        ("build_si_efficiency.py", "data/experimental_efficiency.csv"),
        ("build_si_pairwise_tests.py", "data/pairwise_statistical_tests.csv"),
    ],
)
def test_si_data_exports_re_derive_byte_identically(builder, committed, tmp_path):
    """Each SI data export is regenerated from the released inputs and must come
    back byte-identical to the committed copy."""
    import hashlib
    import subprocess
    import sys

    out = tmp_path / Path(committed).name
    cmd = [sys.executable, str(REPO / "scripts" / builder), "--output", str(out)]
    if builder == "build_si_learning_curves.py":
        cmd += ["--figure-dir", str(tmp_path / "pages")]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.is_file(), out

    committed_path = REPO / "supporting_information" / committed
    assert (hashlib.sha256(out.read_bytes()).hexdigest()
            == hashlib.sha256(committed_path.read_bytes()).hexdigest()), committed
