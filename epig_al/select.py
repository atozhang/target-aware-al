"""Selection and scoring interfaces for the five acquisition methods.

The public interfaces are :func:`acquisition_scores`, :func:`select_batch`, and
:class:`ActiveLearner`. EPIG and BALD estimate uncertainty from per-member
predictions and therefore require an ensemble regressor exposing a
one-dimensional collection of fitted estimators, such as
``RandomForestRegressor``, ``ExtraTreesRegressor``, or ``BaggingRegressor``.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .acquisition.base import AcquireManager
from .acquisition.metrics import _ensemble_predict, estimate_noise_var

# Acquisition rules exposed by name. The five reported in the paper:
METRICS = {
    "epig": "epig_gauss",   # target-aware (this work)
    "bald": "bald",         # global epistemic uncertainty
    "nn": "nn",             # nearest-neighbour to the target set
    "indv_nn": "test_indv_nn",
    "random": "random",
}
_MANAGER_METRICS = frozenset(METRICS.values())
# Metrics whose scores benefit from a data-driven observation-noise estimate.
_NOISE_AWARE = {"epig_gauss", "bald"}


def _resolve_metric(metric: str) -> str:
    """Resolve a public alias or one of the five underlying metric names."""
    name = METRICS.get(metric, metric)
    if name not in _MANAGER_METRICS:
        choices = ", ".join(METRICS)
        raise ValueError(f"unknown acquisition metric {metric!r}; choose from {choices}")
    return name


def _estimate_sigma2(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    acquisition_model=None,
) -> float:
    """Estimate homoscedastic noise in the acquisition response space."""
    scorer = acquisition_model if acquisition_model is not None else model
    y_obs = np.asarray(y_train, dtype=float)
    if hasattr(model, "transform_observations"):
        y_obs = np.asarray(model.transform_observations(y_obs), dtype=float)
    if hasattr(scorer, "predict_ensemble"):
        y_pred = _ensemble_predict(scorer, X_train).mean(axis=0)
    else:
        y_pred = np.asarray(scorer.predict(X_train), dtype=float)
    return float(estimate_noise_var(y_obs, y_pred))


def acquisition_scores(
    model,
    X_pool,
    X_target=None,
    *,
    X_train=None,
    y_train=None,
    metric: str = "epig",
    acquisition_model=None,
) -> np.ndarray:
    """Score each candidate reaction in ``X_pool``; higher means more valuable.

    Parameters
    ----------
    model : fitted estimator
        For ``epig``/``bald``, an ensemble regressor exposing ``estimators_``
        (e.g. a fitted ``RandomForestRegressor``).
    acquisition_model : optional
        Ensemble used for EPIG/BALD scoring when it differs from ``model``
        Defaults to ``model``.
    X_pool : array (n_candidates, n_features)
        Unlabeled candidate experiments to score.
    X_target : array (n_target, n_features), optional
        The reactions you ultimately want to predict well. Required for
        ``epig``/``nn``/``indv_nn`` (the target-aware rules).
    X_train, y_train : arrays, optional
        Currently labeled training set. Used to estimate observation noise for
        ``epig``/``bald`` exactly as in the paper; pass them for faithful scores.
    metric : str
        One of ``epig`` (default), ``bald``, ``nn``, ``indv_nn``, ``random``
        or the corresponding low-level manager name.

    Returns
    -------
    np.ndarray of shape (n_candidates,)
    """
    name = _resolve_metric(metric)
    X_pool = np.asarray(X_pool, dtype=float)
    X_target = None if X_target is None else np.asarray(X_target, dtype=float)
    X_train = None if X_train is None else np.asarray(X_train, dtype=float)

    # Match the paper's noise handling: estimate from training residuals for the
    # noise-aware rules, otherwise a small floor.
    if name in _NOISE_AWARE and X_train is not None and y_train is not None:
        sigma2 = _estimate_sigma2(
            model,
            X_train,
            y_train,
            acquisition_model=acquisition_model,
        )
    else:
        sigma2 = 1e-6

    scorer = acquisition_model if acquisition_model is not None else model
    manager = AcquireManager(n_iter=1, k_batch=1, metric=name)
    scores = manager.calc(
        metric=name,
        model=scorer,
        data=X_train,
        rxnspace=X_pool,
        testdata=X_target,
        sigma2=sigma2,
        noise_var=sigma2,
    )
    scores = np.asarray(scores, dtype=float)
    if scores.shape[0] != X_pool.shape[0]:
        raise ValueError(f"metric '{metric}' returned {scores.shape[0]} scores for {X_pool.shape[0]} candidates")
    return scores


def select_batch(
    model,
    X_pool,
    X_target=None,
    *,
    X_train=None,
    y_train=None,
    k: int = 1,
    metric: str = "epig",
    acquisition_model=None,
) -> np.ndarray:
    """Return the indices of the ``k`` highest-scoring candidates in ``X_pool``.

    This is a one-shot top-``k`` of a single scoring. For the strictly greedy,
    refit-after-each-acquisition behavior used in the reported evaluation, use
    :class:`ActiveLearner` with ``batch_size=1``.
    """
    scores = acquisition_scores(
        model,
        X_pool,
        X_target,
        X_train=X_train,
        y_train=y_train,
        metric=metric,
        acquisition_model=acquisition_model,
    )
    k = min(int(k), scores.shape[0])
    return np.argsort(scores)[::-1][:k]


class ActiveLearner:
    """A greedy active learning loop: fit -> score pool -> acquire -> repeat.

    This retrospective helper requires labels for the complete input dataset;
    acquiring a point reveals its stored label. The reported evaluation and
    tutorial use this setting. Prospective campaigns should call
    :func:`select_batch` after each model fit and append measured outcomes
    before the next call.

    Parameters
    ----------
    model_factory : callable returning a fresh, unfitted estimator
        A new model is built and fit each round (e.g. ``lambda: RandomForestRegressor(n_estimators=300)``).
    metric : str
        Acquisition rule (see :data:`METRICS`). Default ``"epig"``.
    batch_size : int
        Number of points acquired per round (default 1, matching the reported evaluation).
    """

    def __init__(self, model_factory: Callable[[], object], metric: str = "epig", batch_size: int = 1):
        self.model_factory = model_factory
        self.metric = metric
        self.batch_size = int(batch_size)

    def run(
        self,
        X,
        y,
        *,
        labeled: Sequence[int],
        pool: Sequence[int],
        target: Sequence[int],
        n_rounds: Optional[int] = None,
        score_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> pd.DataFrame:
        """Run the loop and return a learning-curve trajectory.

        Returns a DataFrame with one row per round: ``round``, ``n_labeled``,
        ``n_acquired``, and ``target_score`` (R^2 on the target set by default).
        """
        from sklearn.metrics import r2_score

        score_fn = score_fn or r2_score
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        labeled = list(labeled)
        pool = list(pool)
        target = list(target)
        X_target = X[target]
        y_target = y[target]

        max_rounds = len(pool) // max(self.batch_size, 1)
        n_rounds = max_rounds if n_rounds is None else min(int(n_rounds), max_rounds)

        rows = []
        for rnd in range(n_rounds + 1):
            model = self.model_factory().fit(X[labeled], y[labeled])
            rows.append({
                "round": rnd,
                "n_labeled": len(labeled),
                "n_acquired": rnd * self.batch_size,
                "target_score": float(score_fn(y_target, model.predict(X_target))),
            })
            if rnd == n_rounds or not pool:
                break
            local = select_batch(
                model, X[pool], X_target,
                X_train=X[labeled], y_train=y[labeled],
                k=self.batch_size, metric=self.metric,
            )
            chosen = [pool[i] for i in local]
            labeled.extend(chosen)
            pool = [p for p in pool if p not in set(chosen)]
        return pd.DataFrame(rows)
