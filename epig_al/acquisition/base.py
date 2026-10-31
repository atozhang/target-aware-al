"""Dispatch the five acquisition functions evaluated in the manuscript."""

from typing import Callable, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from . import metrics as metrics_module


class AcquireManager:
    """Stateful scorer for active learning workflows."""

    RG = np.random.default_rng()

    @staticmethod
    def set_seed(seed: Optional[int] = None):
        """Seed the generator used by Random and randomized Indiv. NN selection."""
        AcquireManager.RG = np.random.default_rng(seed)

    def __init__(self, n_iter, k_batch, metric: str):
        self.metric_functions = {
            "epig_gauss": self.epig_gauss,
            "nn": self.nn,
            "bald": self.bald,
            "random": self.random,
            "test_indv_nn": self.test_indv_nn,
        }
        self.scores = []
        self.model = None
        self.data = None
        self.rxnspace = None
        self.testdata = None
        self.n_iter = n_iter
        self.k_batch = k_batch
        self.metric = metric
        self.already_selected = set()

    def update_params(self, model=None, data=None, rxnspace=None, testdata=None):
        """Update the model and data blocks used for scoring."""
        if model is not None:
            self.model = model
        if data is not None:
            self.data = data
        if rxnspace is not None:
            self.rxnspace = rxnspace
        if testdata is not None:
            self.testdata = testdata

    def get_metric(self, metric: Optional[str] = None) -> Callable[..., np.ndarray]:
        """Return the requested metric function, or the configured default."""
        key = metric if metric is not None else self.metric
        try:
            return self.metric_functions[key]
        except KeyError as exc:
            raise ValueError(f'Unrecognized metric: "{key}"') from exc

    def get_needs(self, metric: str) -> Set[str]:
        """Return the inputs required by a metric."""
        needs = {
            "epig_gauss": {"model", "data", "rxnspace", "testdata"},
            "bald": {"model", "rxnspace"},
            "nn": {"rxnspace", "testdata"},
            "test_indv_nn": {"rxnspace", "testdata"},
            "random": set(),
        }
        return needs.get(metric, set())

    def valid_metrics(self) -> Set[str]:
        """Return the registered metric names."""
        return set(self.metric_functions)

    def random(self, **kwargs):
        """Return independent random scores for the candidate pool."""
        return self.RG.random(self.rxnspace.shape[0])

    def epig_gauss(self, sigma2: float = 1e-6, weights=None, **kwargs):
        """Return Gaussian ensemble EPIG scores."""
        try:
            metrics_module._ensemble_predict(self.model, self.testdata[:1])
        except Exception as exc:
            raise ValueError(
                "epig_gauss requires an ensemble-capable model exposing "
                "estimators_, predict_ensemble, or a list of fitted models."
            ) from exc
        return metrics_module.epig_gauss(
            self.model,
            self.data,
            self.rxnspace,
            self.testdata,
            sigma2=sigma2,
            weights=weights,
        )

    def nn(self, **kwargs):
        """Score candidates by negative mean distance to the target set."""
        distances = euclidean_distances(self.rxnspace, self.testdata)
        return -distances.mean(axis=1)

    def test_indv_nn(self, k_nearest: int = 1, **kwargs):
        """Prioritize the nearest candidate analogs of individual target points.

        Candidates that are not among any target's nearest neighbors receive
        random fallback scores below the nearest-neighbor score range.
        """
        n_rxn = len(self.rxnspace)
        scores = np.full(n_rxn, -np.inf)
        distances = euclidean_distances(self.rxnspace, self.testdata)

        for target_index in range(self.testdata.shape[0]):
            nearest = np.argsort(distances[:, target_index])[:k_nearest]
            for candidate_index in nearest:
                scores[candidate_index] = max(
                    scores[candidate_index], self.RG.uniform(1000, 2000)
                )

        fallback = np.isneginf(scores)
        if np.any(fallback):
            scores[fallback] = self.RG.uniform(0, 1, int(fallback.sum()))
        return scores

    def bald(self, **kwargs):
        """Return regression BALD scores from ensemble predictions."""
        if self.rxnspace is None or self.rxnspace.shape[0] == 0:
            raise ValueError("self.rxnspace is empty; BALD requires candidates")

        n_samples = kwargs.get("n_samples", 30)
        noise_var = kwargs.get("noise_var")
        mu_s, var_s, mode = metrics_module.predict_samples(
            self.model, self.rxnspace, n_samples=n_samples
        )

        if mu_s is None:
            model_type = type(self.model).__name__
            raise ValueError(
                "BALD regression requires predictive samples from an ensemble; "
                f"{model_type} is unsupported ({mode}). Wrap a deterministic "
                "regressor in BaggingRegressor."
            )

        if var_s is None and noise_var is None:
            noise_var = metrics_module.estimate_noise_var_from_training_or_default(
                self, default=1.0
            )
        return metrics_module.bald_regression_gaussian(
            mu_s, var_s=var_s, noise_var=noise_var
        )

    def calc(self, metric: str, model, data, rxnspace, testdata, **kwargs):
        """Update scorer state and compute one acquisition-score vector."""
        self.update_params(model, data, rxnspace, testdata)
        if metric not in self.metric_functions:
            raise ValueError(
                f'Unrecognized metric "{metric}". Expected one of {self.valid_metrics()}'
            )
        return self.metric_functions[metric](**kwargs)
