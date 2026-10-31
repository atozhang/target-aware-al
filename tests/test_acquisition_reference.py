"""Acquisition-reference parity for the released EPIG method layer.

Locks the released `epig_al` acquisition functions against the frozen synthetic
reference in `tests/fixtures/af_reference_v1.npz`, described in
`af_reference_v1.json`. The fixture contains only pseudo-random arrays; no
experimental dataset is involved.

Covered:
  * full score vectors for all five acquisition functions
    (epig_gauss, bald, test_indv_nn, nn, random), rtol 1e-10;
  * the top-ten candidate ordering per acquisition function, exactly;
  * `select_batch` selected-index sequences at k = 5 and k = 10, exactly --
    the deterministic-selection invariant;
  * byte identity of the fixture against its recorded SHA-256.

Tolerances are those declared with the fixture and must not be relaxed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from epig_al import acquisition_scores, select_batch
from epig_al.acquisition.base import AcquireManager

REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
NPZ = FIXTURE_DIR / "af_reference_v1.npz"
SPEC = FIXTURE_DIR / "af_reference_v1.json"

AFS = ["epig_gauss", "bald", "test_indv_nn", "nn", "random"]
SCORE_RTOL = 1e-10


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC.read_text())


@pytest.fixture(scope="module")
def reference():
    with np.load(NPZ, allow_pickle=False) as fh:
        return {k: fh[k] for k in fh.files}


@pytest.fixture(scope="module")
def model(reference):
    """The reference model: exactly the estimator recorded in the spec."""
    return RandomForestRegressor(n_estimators=30, random_state=0, n_jobs=1).fit(
        reference["X_train"], reference["y_train"]
    )


def _score(model, reference, af: str) -> np.ndarray:
    AcquireManager.set_seed(0)
    return np.asarray(
        acquisition_scores(
            model,
            reference["X_pool"],
            reference["X_target"],
            X_train=reference["X_train"],
            y_train=reference["y_train"],
            metric=af,
        ),
        dtype=np.float64,
    )


def test_fixture_file_is_the_recorded_one(spec):
    assert hashlib.sha256(NPZ.read_bytes()).hexdigest() == spec["fixture_sha256"]


@pytest.mark.parametrize("af", AFS)
def test_score_vectors_reproduce_the_reference(af, model, reference):
    got = _score(model, reference, af)
    ref = reference[f"scores__{af}"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=SCORE_RTOL, atol=0.0)


@pytest.mark.parametrize("af", AFS)
def test_top_ten_candidate_ordering_is_exact(af, model, reference, spec):
    got = _score(model, reference, af)
    finite = np.where(np.isfinite(got), got, -np.inf)
    assert np.argsort(finite)[::-1][:10].tolist() == spec["scores"][af]["top10_indices"]


@pytest.mark.parametrize("af", AFS)
@pytest.mark.parametrize("k", [5, 10])
def test_selected_indices_are_exact(af, k, model, reference, spec):
    AcquireManager.set_seed(0)
    idx = select_batch(
        model,
        reference["X_pool"],
        reference["X_target"],
        X_train=reference["X_train"],
        y_train=reference["y_train"],
        k=k,
        metric=af,
    )
    got = np.asarray(idx, dtype=np.int64).tolist()
    assert got == reference[f"batch{k}__{af}"].tolist()
    assert got == spec["batches"][f"{af}_k{k}"]
    assert len(set(got)) == k
