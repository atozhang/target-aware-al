"""Analytic and invariance tests for Gaussian EPIG and regression BALD."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from epig_al.acquisition.metrics import (
    _ensemble_predict,
    bald_regression_gaussian,
    epig_gauss,
    estimate_noise_var,
)


class _ConstantEnsemble:
    """Synthetic ensemble with fixed member predictions."""

    def __init__(self, members: np.ndarray):
        self._members = np.asarray(members, dtype=float)

    def predict_ensemble(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        return np.tile(self._members[:, None], (1, n))


class _TwoPointEnsemble:
    """Return distinct member predictions for sentinel pool/target inputs."""

    def __init__(self, pool_members: np.ndarray, target_members: np.ndarray):
        self.pool_members = np.asarray(pool_members, dtype=float)
        self.target_members = np.asarray(target_members, dtype=float)

    def predict_ensemble(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                self.pool_members if row[0] < 0.5 else self.target_members
                for row in np.asarray(x)
            ]
        )


def test_zero_covariance_gives_zero_information():
    sigma2 = 0.25
    members_x = np.array([-1.0, -1.0, 1.0, 1.0])
    members_t = np.array([-1.0, 1.0, -1.0, 1.0])
    model = _TwoPointEnsemble(members_x, members_t)
    score = epig_gauss(model, None, np.array([[0.0]]), np.array([[1.0]]), sigma2=sigma2)[0]
    assert score == pytest.approx(0.0, abs=1e-12)


def test_known_bivariate_gaussian_information():
    sigma2 = 0.2
    members_x = np.array([-1.0, 0.0, 1.0, 2.0])
    members_t = np.array([-0.5, 0.4, 1.2, 2.4])
    model = _TwoPointEnsemble(members_x, members_t)
    score = epig_gauss(model, None, np.array([[0.0]]), np.array([[1.0]]), sigma2=sigma2)[0]

    var_x = np.var(members_x, ddof=1) + sigma2
    var_t = np.var(members_t, ddof=1) + sigma2
    cov = np.cov(members_x, members_t, ddof=1)[0, 1]
    expected = 0.5 * np.log((var_x * var_t) / (var_x * var_t - cov * cov))
    assert score == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_translation_invariance_on_means():
    sigma2 = 0.1
    shift = 17.0
    members = np.array([0.0, 1.0, 2.0])
    pool = np.zeros((2, 3))
    target = np.ones((2, 3))
    base = epig_gauss(_ConstantEnsemble(members), None, pool, target, sigma2=sigma2)
    shifted = epig_gauss(_ConstantEnsemble(members + shift), None, pool, target, sigma2=sigma2)
    np.testing.assert_allclose(base, shifted, rtol=1e-6, atol=1e-8)


def test_positive_scale_invariance_on_means_and_sigma():
    sigma2 = 0.2
    scale = 3.5
    members = np.array([-0.5, 0.0, 0.8, 1.1])
    pool = np.linspace(-1, 1, 4).reshape(-1, 1)
    target = np.linspace(0, 2, 3).reshape(-1, 1)
    base = epig_gauss(_ConstantEnsemble(members), None, pool, target, sigma2=sigma2)
    scaled = epig_gauss(
        _ConstantEnsemble(members * scale), None, pool, target, sigma2=sigma2 * scale * scale
    )
    np.testing.assert_allclose(base, scaled, rtol=1e-5, atol=1e-6)


def test_target_averaging_is_uniform_mean():
    sigma2 = 0.05
    members = np.array([0.0, 1.0, 2.0])
    pool = np.zeros((1, 1))
    target = np.array([[0.0], [1.0]])
    uniform = epig_gauss(_ConstantEnsemble(members), None, pool, target, sigma2=sigma2)
    weights = np.array([0.25, 0.75])
    weighted = epig_gauss(
        _ConstantEnsemble(members), None, pool, target, sigma2=sigma2, weights=weights
    )
    per_target = [
        epig_gauss(_ConstantEnsemble(members), None, pool, target[j : j + 1], sigma2=sigma2)[0]
        for j in range(2)
    ]
    expected = float(np.mean(per_target))
    assert uniform[0] == pytest.approx(expected, rel=1e-6)
    assert weighted[0] == pytest.approx(float(np.average(per_target, weights=weights)), rel=1e-6)


def test_near_singular_covariance_is_finite():
    sigma2 = 1e-4
    members = np.array([1.0, 1.0 + 1e-9, 1.0 - 1e-9])
    pool = np.zeros((3, 1))
    target = np.zeros((2, 1))
    scores = epig_gauss(_ConstantEnsemble(members), None, pool, target, sigma2=sigma2)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_bald_uses_sample_ddof_aligned_with_epig():
    sigma2 = 0.3
    members = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]])
    bald = bald_regression_gaussian(members, noise_var=sigma2)
    epi = np.var(members, axis=0, ddof=1)
    expected = 0.5 * np.log1p(epi / sigma2)
    np.testing.assert_allclose(bald, expected, rtol=1e-12)


def test_estimate_noise_var_floor():
    y = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.0, 1.0, 2.0])
    assert estimate_noise_var(y, pred, floor=1e-6) == pytest.approx(1e-6)


def test_ensemble_predict_from_sklearn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))
    y = rng.normal(size=5)
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(x, y)
    preds = _ensemble_predict(model, x)
    assert preds.shape == (5, len(x))
    assert np.all(np.isfinite(preds))
