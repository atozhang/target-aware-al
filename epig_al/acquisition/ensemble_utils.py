"""Build the bagged XGBoost ensemble used by the Wittig study."""

from typing import Optional, Union

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import BaggingRegressor


def _check_xgboost():
    """Raise a clear error when the qualitative-study dependency is absent."""
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for make_bagged_xgb; "
            "install the qualitative extra with pip install -e '.[qualitative]'"
        ) from exc


def make_bagged_xgb(
    base_xgb,
    n_estimators: int = 32,
    max_samples: float = 0.8,
    max_features: float = 1.0,
    bootstrap: bool = True,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    ensure_stochastic: bool = True,
) -> BaggingRegressor:
    """Return an unfitted bagged XGBoost ensemble.

    The qualitative Wittig study fits each member to a bootstrap sample. When
    the supplied XGBoost estimator has no internal row or feature subsampling,
    the default ensure_stochastic=True sets both fractions to 0.8.
    """
    _check_xgboost()
    estimator = clone(base_xgb)
    if ensure_stochastic:
        subsample = getattr(estimator, "subsample", 1.0)
        colsample = getattr(estimator, "colsample_bytree", 1.0)
        subsample = 1.0 if subsample is None else subsample
        colsample = 1.0 if colsample is None else colsample
        if subsample >= 1.0 and colsample >= 1.0:
            estimator.set_params(subsample=0.8, colsample_bytree=0.8)

    return BaggingRegressor(
        estimator=estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=-1,
    )
