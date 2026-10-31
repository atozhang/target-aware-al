"""Parity locks for the published headline claims.

Everything asserted here is RECOMPUTED from the released per-seed inputs
through `scripts/verify_results.py` and then compared with the committed
tables; no published count is read from a file and reported as a reproduction.

Covered:
  * exact benchmark identity and agreement with the 118-task table, including
    the Wittig Published-Split task's recorded metadata and the Wittig split
    membership of every task and repetition;
  * Highest Mean AUC(2-12%) recomputation -- EPIG 74, BALD 22, Indiv. NN 12,
    NN 7, Random 3 -- and the per-dataset EPIG split 32 / 16 / 4 / 22;
  * Wilcoxon-Holm competitiveness recomputation -- EPIG 105, BALD 79,
    Indiv. NN 74, NN 40, Random 64;
  * agreement of every committed classification table with the recomputation
    (per-task rows, SI tables, count tables, main table);
  * the committed counts carrying the current headline, and the primary savings
    table being the cluster-bootstrap rows of the by-structure table;
  * released efficiency and savings values, and the negative-ceiling per-task
    agreement with the recomputed classification.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "scripts/analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import verify_results as rr  # noqa: E402
from scripts.analysis import inputs  # noqa: E402
from scripts.analysis.benchmark_tasks import (  # noqa: E402
    LEDGER,
    WITTIG_TRANSFER_TASK,
    summary,
    load_mean_curves,
    load_per_seed,
)

PKG = REPO / "results/efficiency"
SENS = REPO / "results/benchmark"

N_TASKS = 118
BY_DATASET = {"BH": 44, "Rh": 38, "SAD": 7, "Wittig": 29}
# internal method ids, as used by the census loaders and the count tables
HIGHEST_MEAN = {"EPIG": 74, "BALD": 22, "IndvNN": 12, "NN": 7, "Random": 3}
COMPETITIVE = {"EPIG": 105, "BALD": 79, "IndvNN": 74, "NN": 40, "Random": 64}
# display labels, as used by the manuscript and the release documentation
DISPLAY = {"EPIG": "EPIG", "BALD": "BALD", "IndvNN": "Indiv. NN", "NN": "NN",
           "Random": "Random"}
EPIG_HIGHEST_BY_DATASET = {"BH": 32, "Rh": 16, "SAD": 4, "Wittig": 22}
PRIMARY = "2-12"


# --------------------------------------------------------------------------- #
#  recomputation (primary window), shared by the parity tests
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def recomputed():
    """Recompute the primary-window classification from the per-seed inputs."""
    per_seed = inputs.load_per_seed_full()
    nesting = inputs.load_nesting()
    cls, mismatches = rr.run_window(per_seed, rr.PRIMARY, nesting)
    return {
        "per_seed": per_seed,
        "cls": cls,
        "counts": rr.counts_from(cls),
        "rounding_mismatches": mismatches,
        "pairing": inputs.verify_pairing(per_seed),
    }


# --------------------------------------------------------------------------- #
#  census and task-ledger identity
# --------------------------------------------------------------------------- #

def test_benchmark_summary_is_clean_and_exact():
    a = summary()
    assert a["problems"] == []
    assert a["total_tasks"] == N_TASKS
    assert a["tasks_by_dataset"] == BY_DATASET
    assert a["tasks_by_split_family"]["true_random"] == 60
    assert a["random_target_by_dataset"] == {"BH": 20, "Rh": 20, "Wittig": 20}
    assert a["methods_per_task"] == [5]
    assert sorted(a["reps_per_task_method"]) == [10, 30]
    assert a["trajectories_completed"] == a["trajectories_expected"] == 6600
    assert a["step0_pairing_ok"] is True
    assert a["task_identities_match_curves"] is True
    assert a["task_identities_match_ledger"] is True
    assert a["wittig_transfer_task_present"] is True


def test_task_ledger_identity_is_exactly_the_census_identity():
    """The 118 released ledger identities are exactly the census identities."""
    per_seed = load_per_seed()
    curves = load_mean_curves()
    ledger = pd.read_csv(LEDGER)
    seed_ids = set(per_seed["split_id"])
    curve_ids = set(curves["split_id"])
    ledger_ids = set(ledger["split_id"])
    assert len(ledger_ids) == N_TASKS
    assert seed_ids == curve_ids == ledger_ids
    assert WITTIG_TRANSFER_TASK in ledger_ids
    assert ledger.groupby("dataset")["split_id"].nunique().to_dict() == BY_DATASET


def test_wittig_transfer_task_table_metadata():
    """The Wittig Published-Split task's released row, value by value."""
    ledger = pd.read_csv(LEDGER).set_index("split_id")
    wittig = ledger.loc[WITTIG_TRANSFER_TASK]
    assert wittig["dataset"] == "Wittig"
    assert wittig["split_family"] == "paper_defined"
    assert int(wittig["n_base"]) == 734       # the public textbook base training set
    assert int(wittig["pool_total"]) == 198
    assert int(wittig["n_target"]) == 35
    assert int(wittig["repetitions"]) == 10


WITTIG_SPLITS = REPO / "results/benchmark_inputs/wittig_split_membership.csv"


def _wittig_records() -> tuple[pd.DataFrame, list[str]]:
    lit = pd.read_csv(REPO / "data/Wittig/wittig_literature_f119.csv")
    cas = pd.read_csv(REPO / "data/Wittig/wittig_cas_f119.csv")
    return pd.concat([lit, cas], ignore_index=True), lit["record_id"].tolist()


def test_wittig_split_membership_recovers_every_task_and_repetition():
    """Each Wittig task's target, pool, base and initial sets, for all ten
    repetitions, follow from the membership file and have the sizes of the
    118-task table."""
    members = pd.read_csv(WITTIG_SPLITS, dtype={"seed": "Int64"})
    records, literature = _wittig_records()
    every, base_set = set(records["record_id"]), set(literature)
    ledger = pd.read_csv(LEDGER)
    ledger = ledger[ledger["dataset"].eq("Wittig")].set_index("split_id")
    assert set(members["split_id"]) == set(ledger.index)
    assert set(members["role"]) == {"target", "initial"}
    assert set(members["record_id"]) <= every
    for split_id, task in ledger.iterrows():
        rows = members[members["split_id"].eq(split_id)]
        fixed = set(rows.loc[rows["seed"].isna(), "record_id"])
        assert set(rows.loc[rows["seed"].isna(), "role"]) <= {"target"}
        for seed in range(int(task["repetitions"])):
            here = rows[rows["seed"].eq(seed)]
            target = fixed | set(here.loc[here["role"].eq("target"), "record_id"])
            initial = set(here.loc[here["role"].eq("initial"), "record_id"])
            base = base_set if split_id == WITTIG_TRANSFER_TASK else set()
            pool = every - target - base
            assert (len(target), len(pool), len(base), len(initial)) == (
                task["n_target"], task["pool_total"], task["n_base"], task["warm_start"]), (split_id, seed)
            assert initial <= pool, (split_id, seed)


def test_wittig_cluster_targets_are_the_ward_k8_clusters():
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, str(REPO / "data/Wittig"))
    from feature_config import STANDARDIZED_FEATURES

    records, _ = _wittig_records()
    z = StandardScaler().fit_transform(records[STANDARDIZED_FEATURES].to_numpy(float))
    labels = AgglomerativeClustering(n_clusters=8).fit_predict(z)
    members = pd.read_csv(WITTIG_SPLITS, dtype={"seed": "Int64"})
    for cluster in range(8):
        split_id = f"wittig_f119_cluster_k8_cl{cluster}"
        target = set(members.loc[members["split_id"].eq(split_id) & members["role"].eq("target"), "record_id"])
        assert target == set(records.loc[labels == cluster, "record_id"]), split_id


def test_protocol_matched_rh_tasks_are_qualified_and_nested():
    """The seven nested Rh tasks are 96/96 with 30 replicates, validated from
    structured fields rather than from any free-text note."""
    per_seed = load_per_seed()
    ledger = pd.read_csv(LEDGER)
    rh = sorted(ledger.loc[ledger["split_id"].str.startswith("rh_paper_ood_case"),
                           "split_id"].unique())
    assert rh == [f"rh_paper_ood_case{i}" for i in range(1, 8)]
    promoted = ledger[ledger["split_id"].isin(rh)]
    assert promoted["pool_total"].eq(96).all()
    assert promoted["n_target"].eq(96).all()
    assert promoted["repetitions"].eq(30).all()
    replicates = (per_seed[per_seed["split_id"].isin(rh)]
                  .groupby(["split_id", "acq"])["seed_id"].nunique())
    assert replicates.eq(30).all()


def test_recomputed_replicate_design(recomputed):
    p = recomputed["pairing"]
    assert int(p["n_tasks"]) == N_TASKS
    assert {int(k): int(v) for k, v in p["seed_count_distribution"].items()} == {10: 111, 30: 7}
    assert p["task_identities_match_ledger"] is True
    assert set(recomputed["cls"]["n_analysis_units"]) == {10}
    assert set(recomputed["cls"]["unit_scheme"]) == {"primary"}


# --------------------------------------------------------------------------- #
#  Highest Mean and Wilcoxon-Holm competitiveness, recomputed
# --------------------------------------------------------------------------- #

def test_highest_mean_recomputes_to_the_published_counts(recomputed):
    c = recomputed["counts"].set_index("AF")
    got = {af: int(c.loc[af, "highest_mean"]) for af in HIGHEST_MEAN}
    assert got == HIGHEST_MEAN
    assert sum(got.values()) == N_TASKS


def test_wilcoxon_holm_competitiveness_recomputes_to_the_published_counts(recomputed):
    c = recomputed["counts"].set_index("AF")
    got = {af: int(c.loc[af, "competitive"]) for af in COMPETITIVE}
    assert got == COMPETITIVE
    for af in COMPETITIVE:
        assert int(c.loc[af, "competitive"]) + int(c.loc[af, "detected_worse"]) == N_TASKS


def test_per_dataset_epig_highest_mean_recomputes_and_sums_to_the_total(recomputed):
    cls = recomputed["cls"]
    epig = cls[cls.AF == "EPIG"]
    got = epig[epig.is_highest_mean].groupby("dataset")["split_id"].nunique().to_dict()
    assert got == EPIG_HIGHEST_BY_DATASET
    assert sum(got.values()) == HIGHEST_MEAN["EPIG"]
    committed = pd.read_csv(SENS / "counts_by_dataset.csv")
    committed = committed[(committed.AF == "EPIG") & (committed.window == PRIMARY)]
    committed_epig = committed.set_index("dataset")["highest"].astype(int).to_dict()
    assert committed_epig == EPIG_HIGHEST_BY_DATASET


def test_recomputation_is_insensitive_to_the_rounding_convention(recomputed):
    """Ranking the unrounded paired differences must not move any task."""
    assert recomputed["rounding_mismatches"] == []


def test_every_committed_classification_table_agrees_with_the_recomputation(recomputed):
    """The released wrapper's own validation layer, run on the primary window."""
    problems: list[str] = []
    cls, cnt = recomputed["cls"], recomputed["counts"]
    per_task = rr.validate_per_task(cls, problems)
    si = rr.validate_si_a2_12(cls, problems)
    n_counts = rr.validate_committed_counts_table(cnt, problems)
    n_overall = rr.validate_si_counts(cnt, problems)
    n_scoped = rr.validate_scoped_counts(cls, problems)
    n_main = rr.validate_main_table(cnt, problems)
    assert problems == []
    assert per_task["rows_compared"] == N_TASKS * 5
    assert si["rows"] == N_TASKS * 5
    assert (n_counts, n_overall, n_main) == (5, 10, 10)
    assert n_scoped > 0


# --------------------------------------------------------------------------- #
#  committed-table parity
# --------------------------------------------------------------------------- #

def test_committed_counts_carry_the_current_headline(recomputed):
    counts = pd.read_csv(SENS / "counts_by_window.csv")
    primary = counts[counts.window == PRIMARY].set_index("AF")
    assert set(primary["n_tasks"]) == {N_TASKS}
    rec = recomputed["counts"].set_index("AF")
    for af in HIGHEST_MEAN:
        assert int(primary.loc[af, "highest_mean"]) == HIGHEST_MEAN[af], af
        assert int(primary.loc[af, "competitive"]) == COMPETITIVE[af], af
        assert int(rec.loc[af, "highest_mean"]) == int(primary.loc[af, "highest_mean"])
        assert int(rec.loc[af, "competitive"]) == int(primary.loc[af, "competitive"])
    assert sorted(counts["window"].unique()) == sorted(
        ["0-10", "2-12", "5-15", "5-20", "5-30", "10-20", "10-30"])


def test_primary_savings_table_is_the_cluster_bootstrap_rows():
    """The published primary savings table is exactly the cluster-bootstrap
    structure of the by-structure table -- the declared primary procedure."""
    primary = pd.read_csv(PKG / "savings_intervals_100k_primary.csv")
    by_structure = pd.read_csv(PKG / "savings_intervals_100k_by_structure.csv")
    cluster = by_structure[by_structure.structure == "cluster"]
    merged = primary.merge(cluster, on=["metric", "method"], suffixes=("_p", "_r"))
    assert len(merged) == len(primary) == 10
    for col in ("mean", "ci_lo", "ci_hi"):
        assert (merged[f"{col}_p"] - merged[f"{col}_r"]).abs().max() < 1e-9, col
    assert (merged["seed_p"] == merged["seed_r"]).all()
    assert (merged["excludes_zero_p"] == merged["excludes_zero_r"]).all()


# --------------------------------------------------------------------------- #
#  released efficiency / savings values
# --------------------------------------------------------------------------- #

def test_overall_experimental_efficiency_values_and_paired_intervals():
    rx = pd.read_csv(PKG / "efficiency_summary_reactions.csv")
    pool = pd.read_csv(PKG / "efficiency_summary_pct_pool.csv")
    ov_rx = rx[rx.scope == "Overall"].set_index("AF")
    ov_pool = pool[pool.scope == "Overall"].set_index("AF")
    assert set(ov_rx["n_tasks"]) == {N_TASKS}
    assert ov_rx.loc["EPIG", "mean"] == pytest.approx(203.63, abs=0.01)
    assert ov_rx.loc["BALD", "mean"] == pytest.approx(258.60, abs=0.01)
    assert ov_pool.loc["EPIG", "mean"] == pytest.approx(15.18, abs=0.01)
    assert ov_pool.loc["BALD", "mean"] == pytest.approx(17.63, abs=0.01)
    # EPIG uses significantly fewer reactions and less pool than every comparator
    for af in ("BALD", "IndvNN", "NN", "Random"):
        assert ov_rx.loc[af, "paired_diff_ci_lo"] > 0, af
        assert ov_pool.loc[af, "paired_diff_ci_lo"] > 0, af


def test_rh_experimental_efficiency_is_reported_as_a_tie():
    """On Rh the EPIG-vs-BALD paired interval spans zero; it is not a win."""
    rx = pd.read_csv(PKG / "efficiency_summary_reactions.csv")
    rh = rx[rx.scope == "Rh"].set_index("AF")
    assert rh.loc["BALD", "paired_diff_ci_lo"] < 0 < rh.loc["BALD", "paired_diff_ci_hi"]


def test_per_task_efficiency_respects_its_budget_and_pool_bounds():
    per = pd.read_csv(PKG / "efficiency_per_task.csv")
    assert len(per) == N_TASKS * 5
    assert (per["n_reactions"] <= per["budget_reactions"]).all()
    assert (per["pct_pool"]
            <= 100.0 * per["budget_reactions"] / per["pool_total"] + 1e-9).all()


def test_negative_ceiling_margins_agree_with_the_recomputed_classification(recomputed):
    """`epig_strict_win` is the recomputed highest-mean flag, task by task."""
    nc = pd.read_csv(PKG / "negative_ceiling_margin.csv")
    assert len(nc) == N_TASKS
    epig = recomputed["cls"]
    epig = epig[epig.AF == "EPIG"][["split_id", "is_highest_mean"]]
    merged = nc.merge(epig, on="split_id", validate="one_to_one")
    assert len(merged) == N_TASKS
    assert (merged["epig_strict_win"].astype(bool)
            == merged["is_highest_mean"].astype(bool)).all()
    assert int(merged["epig_strict_win"].sum()) == HIGHEST_MEAN["EPIG"]
    # the EPIG advantage is stronger where the task ceiling is positive
    positive = merged[merged["task_ceiling"] > 0]
    assert len(positive) > 0
    assert positive["epig_strict_win"].mean() > merged["epig_strict_win"].mean()
