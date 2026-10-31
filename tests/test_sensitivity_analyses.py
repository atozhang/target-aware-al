"""Invariants of the published sensitivity analyses.

Covered:
  * the 12-decimal rounding convention for ranking and zero removal;
  * Rh aggregation sensitivity: the published rows come from the primary
    scheme, and EPIG's competitive count does not depend on the aggregation.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.benchmark_tasks import (
    RH_PROTOCOL_MATCHED_TASKS,
    load_native_mean_curves,
)

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "scripts/analysis"
SENSITIVITY = REPO / "results/sensitivity"
SENS = REPO / "results/benchmark"

if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

WINDOWS = ["0-10", "2-12", "5-15", "5-20", "5-30", "10-20", "10-30"]
COMPETITIVE_PRIMARY = {"EPIG": 105, "BALD": 79, "Indiv. NN": 74, "NN": 40, "Random": 64}


# --------------------------------------------------------------------------- #
#  rounding convention
# --------------------------------------------------------------------------- #

def test_primary_path_ranks_the_rounded_differences():
    import stats as sc
    sig = inspect.signature(sc.exact_wilcoxon_signed_rank)
    assert sig.parameters["rank_rounded"].default is True
    # Two differences equal at 12 dp but distinct in float64 must tie. The tie
    # group needs MIXED signs: an all-positive group leaves w_plus unchanged.
    d = [1.0, -(1.0 + 1e-15), -2.0, 3.0]
    assert (sc.exact_wilcoxon_signed_rank(d, rank_rounded=True)["w_plus"]
            != sc.exact_wilcoxon_signed_rank(d, rank_rounded=False)["w_plus"]), (
        "rounding must change the ranks for near-identical differences")


def test_zero_removal_uses_the_rounded_differences():
    import stats as sc
    r = sc.exact_wilcoxon_signed_rank([1e-15, -1e-15, 2.0, -3.0])
    assert r["n_pairs"] == 4 and r["n_nonzero"] == 2


def test_rounding_convention_is_twelve_decimals_before_zero_removal():
    """Paired differences are rounded to 12 dp *before* zeros are dropped and
    ties are ranked.

    The unrounded-ranking sensitivity itself is rerun end to end by
    ``scripts/verify_results.py``, which reports the mismatch count against the
    committed per-task table; this pins the convention the engine applies.
    """
    import stats as sc
    assert sc.ZERO_TIE_DECIMALS == 12
    # a difference below the rounding threshold is treated as an exact zero and
    # dropped, not ranked as a tiny nonzero value
    r = sc.exact_wilcoxon_signed_rank([1e-15, 2.0, -3.0])
    assert r["n_pairs"] == 3 and r["n_nonzero"] == 2


# --------------------------------------------------------------------------- #
#  analysis package layout
# --------------------------------------------------------------------------- #

def test_repo_root_is_derived_from_file():
    import paths
    assert paths.REPO == REPO
    assert paths.PACKAGE_DIR == PKG


def test_rh_random_split_per_seed_table_is_complete():
    import paths
    assert paths.RH_RANDOM_SPLIT_PER_SEED.is_file()
    df = pd.read_csv(paths.RH_RANDOM_SPLIT_PER_SEED)
    assert len(df) == 500 and df["split_id"].nunique() == 10


# --------------------------------------------------------------------------- #
#  Rh aggregation sensitivity
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def rh() -> dict:
    return json.loads((SENSITIVITY / "rh_partition/summary.json").read_text())


def test_primary_scheme_reproduces_the_published_rh_rows(rh):
    assert rh["primary_reproduces_published"] is True
    assert rh["si_sensitivity_schemes"] == ["pooled_30", "partition_0",
                                            "partition_1", "partition_2"]
    assert sorted(rh["windows"]) == sorted(WINDOWS)
    assert len(rh["rh_tasks"]) == 7


def test_highest_mean_is_invariant_between_primary_and_pooled30(rh):
    """The mean of ten seed-level means (each over three partitions) equals the
    mean of all thirty values. Verified, not assumed."""
    assert rh["n_highest_mean_differences_primary_vs_pooled30"] == 0
    assert rh["max_abs_mean_delta_primary_vs_pooled30"] < 1e-12


def test_epig_overall_competitive_count_is_stable_across_aggregations(rh):
    """EPIG's published 105 must not depend on the aggregation choice."""
    vals = {k: d["EPIG"] for k, d in rh["overall_competitive_a2_12"].items()}
    assert vals["primary"] == 105
    assert vals["pooled_30"] == 105
    assert min(vals.values()) >= 104 and max(vals.values()) <= 108, vals


def test_published_a2_12_competitive_counts_come_from_the_primary_scheme(rh):
    assert rh["overall_competitive_a2_12"]["primary"] == COMPETITIVE_PRIMARY


def test_aggregation_sensitivity_counts_cover_every_scheme_and_window():
    counts = pd.read_csv(SENSITIVITY / "rh_partition/counts.csv")
    assert sorted(counts["window"].unique()) == sorted(WINDOWS)
    assert {"primary", "pooled_30", "partition_0", "partition_1",
            "partition_2"} <= set(counts["scheme"])


def test_native_endpoint_artifact_records_nested_rh_weighting():
    """The endpoint averages three partitions within each of ten repetitions."""
    native = load_native_mean_curves()
    rh_native = native[native["split_id"].isin(RH_PROTOCOL_MATCHED_TASKS)]
    assert rh_native["split_id"].nunique() == 7
    assert set(rh_native["weighting_scheme"]) == {
        "mean_3_partitions_within_each_of_10_repetitions"
    }
    assert set(rh_native["n_repetition_units"]) == {10}
    assert set(rh_native["n_partitions_per_repetition"]) == {3}
    assert set(rh_native["n_source_trajectories"]) == {30}
