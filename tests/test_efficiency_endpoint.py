"""Native integer-step experimental-efficiency endpoint and published values."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.benchmark_tasks import (
    ACQS,
    NATIVE_MEAN_CURVES,
    NATIVE_MEAN_CURVES_METADATA,
    RH_PROTOCOL_MATCHED_TASKS,
    load_native_mean_curves,
    sha256,
)
from scripts.analysis.crossing import (
    crossing_or_budget,
    efficiency_target,
    native_first_crossing,
    native_two_consecutive_crossing,
)

REPO = Path(__file__).resolve().parents[1]
EFFICIENCY = REPO / "results/efficiency"


def _endpoint():
    import sys

    for path in (REPO / "scripts", REPO / "scripts/analysis"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import efficiency_endpoint

    return efficiency_endpoint


# --------------------------------------------------------------------------- #
# Estimator definition
# --------------------------------------------------------------------------- #


def test_target_is_the_fraction_q_of_the_common_improvement():
    assert efficiency_target(0.2, 0.7, 0.8) == pytest.approx(0.6)
    assert efficiency_target(-1.0, 0.0, 0.8) == pytest.approx(-0.2)


def test_native_first_crossing_is_the_first_integer_meeting_step():
    curve = np.array([0.0, 0.2, 0.61, 0.4, 0.8])
    assert native_first_crossing(curve, 0.6) == 2
    assert native_first_crossing(curve, 0.0) == 0


def test_exact_crossing_at_budget_is_attainment_not_nonattainment():
    curve = np.array([0.0, 0.1, 0.2, 0.6])
    crossing = native_first_crossing(curve, 0.6)
    assert crossing == 3
    assert crossing_or_budget(crossing, budget=3) == (True, 3)


def test_nonattainment_is_charged_the_full_budget_with_a_false_flag():
    crossing = native_first_crossing(np.array([0.0, 0.3, 0.4]), 0.5)
    assert crossing is None
    assert crossing_or_budget(crossing, budget=2) == (False, 2)


def test_raw_native_curve_is_not_smoothed_or_made_monotone():
    curve = np.array([0.0, 0.7, 0.1, 0.8])
    assert native_first_crossing(curve, 0.6) == 1


def test_native_two_consecutive_requires_one_additional_acquisition():
    isolated = np.array([0.0, 0.8, 0.2, 0.9])
    persistent = np.array([0.0, 0.6, 0.7, 0.2])
    assert native_two_consecutive_crossing(isolated, 0.5) is None
    assert native_two_consecutive_crossing(persistent, 0.5) == 1
    # A first meeting only at the budget cannot be confirmed.
    assert native_first_crossing(isolated, 0.9) == 3
    assert native_two_consecutive_crossing(isolated, 0.9) is None


def test_native_two_consecutive_is_never_earlier_than_first_crossing():
    rng = np.random.default_rng(7)
    for _ in range(200):
        curve = np.cumsum(rng.normal(0.02, 0.05, size=31))
        target = float(np.quantile(curve, 0.8))
        first = native_first_crossing(curve, target)
        pair = native_two_consecutive_crossing(curve, target)
        if pair is not None:
            assert first is not None and first <= pair


# --------------------------------------------------------------------------- #
# Released native input artifact
# --------------------------------------------------------------------------- #


def test_native_artifact_checksum_and_public_metadata():
    metadata = json.loads(NATIVE_MEAN_CURVES_METADATA.read_text())
    assert sha256(NATIVE_MEAN_CURVES) == metadata["sha256"]
    assert metadata["tasks"] == 118
    assert metadata["task_method_curves"] == 590
    assert metadata["source_validation"]["recorded_sha256_matches"] == 6290
    assert metadata["source_validation"]["recorded_sha256_mismatches"] == 310
    assert metadata["source_validation"]["all_recorded_hashes_verified"] is False
    assert "private source paths" in metadata["privacy"]


def test_native_artifact_has_every_integer_step_and_weighting_metadata():
    curves = load_native_mean_curves()
    assert curves["split_id"].nunique() == 118
    assert curves.groupby(["split_id", "acq"]).ngroups == 590
    for (split_id, _acq), group in curves.groupby(["split_id", "acq"], sort=False):
        group = group.sort_values("n_acquired")
        budget = int(group["budget_reactions"].iloc[0])
        np.testing.assert_array_equal(group["n_acquired"], np.arange(budget + 1))
        nested = split_id in RH_PROTOCOL_MATCHED_TASKS
        assert set(group["n_repetition_units"]) == {10}
        assert set(group["n_partitions_per_repetition"]) == ({3} if nested else {1})
        assert set(group["n_source_trajectories"]) == ({30} if nested else {10})


# --------------------------------------------------------------------------- #
# Exact regeneration from the released native input
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def regenerated():
    ep = _endpoint()
    curves, pools = ep.load_inputs()
    return {"curves": curves, "pools": pools, "eff": ep.task_efficiency(curves, pools), "ep": ep}


def _csv_text(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def test_per_task_table_regenerates_byte_identically(regenerated):
    committed = (EFFICIENCY / "efficiency_per_task.csv").read_text()
    assert _csv_text(regenerated["eff"]) == committed


def test_target_level_counts_regenerate_byte_identically(regenerated):
    ep = regenerated["ep"]
    counts = ep.target_level_counts(regenerated["curves"], regenerated["pools"])
    committed = (REPO / "results/sensitivity/target_level_counts.csv").read_text()
    assert _csv_text(counts) == committed


def test_scope_summaries_regenerate(regenerated):
    ep = regenerated["ep"]
    for value, tag in (("n_reactions", "reactions"), ("pct_pool", "pct_pool")):
        summary = ep.efficiency_summary(regenerated["eff"], value)
        committed = pd.read_csv(EFFICIENCY / f"efficiency_summary_{tag}.csv")
        assert list(summary[["scope", "AF"]].itertuples(index=False)) == list(
            committed[["scope", "AF"]].itertuples(index=False)
        )
        numeric = [column for column in committed.columns if column not in ("scope", "AF")]
        np.testing.assert_allclose(
            summary[numeric].to_numpy(float),
            committed[numeric].to_numpy(float),
            rtol=0,
            atol=1e-9,
        )


def test_rh_random_target_scope_contains_all_twenty_tasks(regenerated):
    summary = regenerated["ep"].efficiency_summary(regenerated["eff"], "n_reactions")
    rh_scopes = set(summary.loc[summary["scope"].str.startswith("Rh "), "scope"])
    assert rh_scopes == {"Rh random-target"}
    assert set(summary.loc[summary["scope"] == "Rh random-target", "n_tasks"]) == {20}


def test_shared_baseline_native_ceiling_and_first_crossings(regenerated):
    eff = regenerated["eff"].set_index(["split_id", "acq"])
    for split_id, task in regenerated["curves"].groupby("split_id", sort=False):
        wide = task.pivot(index="n_acquired", columns="acq", values="mean_r2").sort_index()
        step0 = wide.iloc[0].to_numpy(float)
        assert np.ptp(step0) <= 1e-10
        initial = float(step0.mean())
        ceiling = float(wide.to_numpy(float).max())
        target = efficiency_target(initial, ceiling, 0.8)
        for acq in ACQS:
            row = eff.loc[(split_id, acq)]
            crossing = native_first_crossing(wide[acq].to_numpy(float), target)
            attained, count = crossing_or_budget(crossing, int(row["budget_reactions"]))
            assert row["target_score"] == pytest.approx(target, abs=1e-14)
            assert bool(row["attained"]) is attained
            assert int(row["n_reactions"]) == count


# --------------------------------------------------------------------------- #
# Published values
# --------------------------------------------------------------------------- #


def test_headline_mean_reactions_and_attainment():
    eff = pd.read_csv(EFFICIENCY / "efficiency_per_task.csv")
    means = eff.groupby("AF")["n_reactions"].mean()
    expected = {
        "EPIG": 203.6271186440678,
        "BALD": 258.60169491525426,
        "IndvNN": 257.00847457627117,
        "NN": 365.5,
        "Random": 284.7881355932203,
    }
    for method, value in expected.items():
        assert means[method] == pytest.approx(value, abs=1e-12)
    wide = eff.pivot_table(index=["dataset", "split_id"], columns="AF", values="n_reactions")
    gap = wide["BALD"] - wide["EPIG"]
    assert round(gap.mean()) == 55
    assert round(gap.loc["BH"].mean()) == 126
    assert round(gap.loc["Wittig"].mean()) == 32
    assert eff.groupby("AF")["attained"].sum().to_dict() == {
        "EPIG": 102,
        "BALD": 96,
        "IndvNN": 80,
        "NN": 45,
        "Random": 68,
    }


def test_wittig_published_split_native_counts_and_threshold():
    eff = pd.read_csv(EFFICIENCY / "efficiency_per_task.csv")
    wittig = eff[eff["split_id"] == "wittig_f119_textbook_base_cas_random_85_15"]
    assert wittig.set_index("AF")["n_reactions"].to_dict() == {
        "EPIG": 24,
        "BALD": 48,
        "IndvNN": 44,
        "NN": 60,
        "Random": 60,
    }
    assert wittig["target_score"].nunique() == 1
    assert float(wittig["target_score"].iloc[0]) == pytest.approx(0.180942, abs=5e-7)


def test_crossing_rule_sensitivity_primary_rows_match_published_intervals():
    sensitivity = pd.read_csv(EFFICIENCY / "crossing_rule_sensitivity.csv")
    assert list(sensitivity["rule"].unique()) == [
        "native_first_integer",
        "native_two_consecutive_integer",
    ]
    assert set(sensitivity.loc[sensitivity["rule"] == "native_first_integer", "role"]) == {"primary"}
    assert set(
        sensitivity.loc[sensitivity["rule"] == "native_two_consecutive_integer", "role"]
    ) == {"sensitivity"}
    primary = sensitivity[sensitivity["rule"] == "native_first_integer"].set_index("method")
    published = pd.read_csv(EFFICIENCY / "savings_intervals_100k_primary.csv")
    for metric, tag in (("n_reactions", "reactions_saved"), ("pct_pool", "pool_saved_pp")):
        expected = published[published["metric"] == metric].set_index("method")
        for method in ("EPIG", "BALD", "IndvNN", "NN"):
            assert primary.loc[method, f"{tag}_vs_random"] == pytest.approx(
                expected.loc[method, "mean"], abs=1e-9
            )
            assert primary.loc[method, f"{tag}_ci_lo"] == pytest.approx(
                expected.loc[method, "ci_lo"], abs=1e-9
            )
            assert primary.loc[method, f"{tag}_ci_hi"] == pytest.approx(
                expected.loc[method, "ci_hi"], abs=1e-9
            )


def test_epig_savings_exclude_zero_under_native_two_consecutive_rule():
    sensitivity = pd.read_csv(EFFICIENCY / "crossing_rule_sensitivity.csv")
    epig = sensitivity[
        (sensitivity["rule"] == "native_two_consecutive_integer")
        & (sensitivity["method"] == "EPIG")
    ].iloc[0]
    assert epig["reactions_saved_ci_lo"] > 0
    assert epig["pool_saved_pp_ci_lo"] > 0
