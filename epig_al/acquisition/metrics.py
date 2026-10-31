"""Acquisition-score calculations used by the released EPIG benchmarks.

This module implements Gaussian ensemble EPIG and regression BALD, together
with their shared ensemble-prediction and observation-noise helpers.
"""
from typing import Optional

import numpy as np


# Relative numerical floor for the bivariate-Gaussian EPIG denominator.
EPS_DENOM_FACTOR = 1e-12

def _validate_ensemble_preds(preds: np.ndarray, N: int) -> np.ndarray:
    """Validate and normalize ensemble prediction array to (S, N), S >= 2, finite float."""
    preds = np.asarray(preds, dtype=float)
    if preds.ndim != 2:
        raise ValueError(f"Ensemble predictions must be 2D; got shape {preds.shape}.")
    S, n_pts = preds.shape
    if S < 2:
        raise ValueError(f"Ensemble must have at least 2 members (S >= 2); got S={S}.")
    if n_pts != N:
        raise ValueError(
            f"Ensemble prediction second dimension must be N={N}; got {n_pts}."
        )
    if not np.all(np.isfinite(preds)):
        raise ValueError(
            "Ensemble predictions contain NaN or inf; check model or inputs."
        )
    return preds


def _ensemble_predict(model, X: np.ndarray) -> np.ndarray:
    """
    Get ensemble predictive means: (S, N) array with S = ensemble size, N = number of points.

    Supports fitted models exposing ``estimators_`` or ``predict_ensemble``,
    and lists or tuples of fitted estimators. A wrapper's public prediction
    interface is used before its inner ``model`` attribute.

    Parameters
    ----------
    model : object or list/tuple of fitted estimators
        A regressor with estimators_, predict_ensemble(X), or a list of fitted models.
    X : np.ndarray
        Inputs, shape (N, d) or (d,) (single point reshaped to (1, d)).

    Returns
    -------
    np.ndarray
        Shape (S, N). S = number of ensemble members, N = X.shape[0].

    Raises
    ------
    ValueError
        If no recognized ensemble interface, or validation fails (S < 2, non-finite, shape mismatch).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]

    def explicit_predictions(owner, method_name: str) -> np.ndarray:
        method = getattr(owner, method_name)
        if method_name == "predict_samples":
            result = method(X)
            out = result[0] if isinstance(result, tuple) else result
        else:
            out = method(X)
        out = np.asarray(out)
        if out.ndim == 2 and out.shape[0] == N and out.shape[1] != N:
            out = out.T
        try:
            return _validate_ensemble_preds(out, N)
        except Exception as exc:
            raise ValueError(
                f"Explicit {type(owner).__name__}.{method_name} returned invalid predictions"
            ) from exc

    # A wrapper's public interface defines prediction units. Never bypass it by
    # unwrapping .model first, and never silently fall back if it is malformed.
    for method_name in ("predict_ensemble", "predict_samples"):
        if hasattr(model, method_name) and callable(getattr(model, method_name)):
            return explicit_predictions(model, method_name)

    inner = getattr(model, "model", model)

    if isinstance(inner, (list, tuple)):
        preds = np.stack([m.predict(X) for m in inner], axis=0)
        return _validate_ensemble_preds(preds, N)

    for method_name in ("predict_ensemble", "predict_samples"):
        if hasattr(inner, method_name) and callable(getattr(inner, method_name)):
            return explicit_predictions(inner, method_name)

    if hasattr(inner, "estimators_") and inner.estimators_ is not None:
        preds = np.stack([e.predict(X) for e in inner.estimators_], axis=0)
        return _validate_ensemble_preds(preds, N)

    if hasattr(model, "estimators_") and model.estimators_ is not None:
        preds = np.stack([e.predict(X) for e in model.estimators_], axis=0)
        return _validate_ensemble_preds(preds, N)

    raise ValueError(
        "EPIG/BALD require an ensemble-capable model. Use one of: "
        "(1) estimators_ (e.g. RandomForestRegressor, BaggingRegressor), "
        "(2) predict_ensemble(X) returning (S, N) or (N, S), or "
        "(3) model as list/tuple of fitted models. "
        "Wrap deterministic regressors in an ensemble such as BaggingRegressor."
    )


def estimate_noise_var(
    y_true: np.ndarray, y_pred: np.ndarray, floor: float = 1e-8
) -> float:
    """
    Estimate homoscedastic noise variance from residuals (MSE with floor).

    Use for EPIG (sigma2) or BALD (noise_var) when training data is available.
    Floor prevents MI explosions when noise_var is near zero.

    Parameters
    ----------
    y_true : np.ndarray
        Observed targets.
    y_pred : np.ndarray
        Predicted values (same length as y_true).
    floor : float
        Minimum returned variance. Default 1e-8.

    Returns
    -------
    float
        max(mean((y_true - y_pred)**2), floor).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = np.mean((y_true - y_pred) ** 2)
    return max(float(mse), floor)


def bald_regression_gaussian(
    mu_s: np.ndarray,
    var_s: Optional[np.ndarray] = None,
    noise_var: Optional[float] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Approximate BALD for regression via MI = H[p(y|x,D)] - E_theta H[p(y|x,theta)]
    assuming Gaussian conditional predictive distributions and moment-matched mixture entropy.

    Parameters
    ----------
    mu_s : (S, N) array
        Predictive means from S stochastic passes/samples.
    var_s : (S, N) array or None
        Per-sample aleatoric variances (optional, for heteroscedastic models).
    noise_var : float or None
        Scalar homoscedastic noise variance if var_s is None.
    eps : float
        Small epsilon for numerical stability.

    Returns
    -------
    np.ndarray
        BALD scores, shape (N,). Nonnegative.

    Formula:
      epi = Var_s[mu_s]  (epistemic variance)
      If var_s:
          total = mean(var_s, axis=0) + epi
          return 0.5*(log(total) - mean(log(var_s), axis=0))
      Else:
          require noise_var; return 0.5*log1p(epi / noise_var)

    Raises
    ------
    ValueError
        If neither var_s nor noise_var is provided.
    """
    mu_s = np.asarray(mu_s)
    if mu_s.ndim != 2:
        raise ValueError(f"mu_s must have shape (S, N); got {mu_s.shape}")

    # Epistemic variance: Var_s[mu_s]. ddof=1 matches EPIG-Gauss ensemble variance.
    epi = np.var(mu_s, axis=0, ddof=1)  # (N,)

    if var_s is not None:
        var_s = np.asarray(var_s)
        if var_s.shape != mu_s.shape:
            raise ValueError(f"var_s must match mu_s shape (S, N); got {var_s.shape}")

        # Heteroscedastic case: total = E[var_s] + epi
        ale = np.mean(var_s, axis=0)  # (N,)
        total = np.clip(ale + epi, eps, None)
        var_s_clipped = np.clip(var_s, eps, None)

        # BALD = 0.5*(log(total) - E[log(var_s)])
        bald = 0.5 * (np.log(total) - np.mean(np.log(var_s_clipped), axis=0))
        return bald

    # Homoscedastic case
    if noise_var is None:
        raise ValueError("Either var_s or noise_var must be provided")
    noise_var = max(float(noise_var), eps)

    # BALD = 0.5*log1p(epi / noise_var)
    bald = 0.5 * np.log1p(epi / noise_var)
    return bald


def predict_samples(model, X: np.ndarray, n_samples: int = 30):
    """
    Extract stochastic predictive samples from a model for BALD computation.

    Parameters
    ----------
    model : object
        Model wrapper or raw scikit-learn ensemble.
    X : np.ndarray
        Inputs, shape (N, d).
    n_samples : int
        Desired number of samples (default 30). Ignored if model has fixed ensemble size.

    Returns
    -------
    mu_s : (S, N) array or None
        Predictive means from S samples. None if unsupported.
    var_s : (S, N) array or None
        Per-sample aleatoric variances. None for homoscedastic models.
    mode : str
        Description of sampling method: "predict_samples", "ensemble_estimators", "unsupported"

    Strategy order:
    A) If model has method predict_samples(X, n_samples=...): call it.
       Expect mu_s (S,N) and optionally var_s.
    B) If model has attribute estimators_ (sklearn RF/ExtraTrees/Bagging):
       Build mu_s by stacking est.predict(X) for each estimator.
       Ignore n_samples if > #estimators.
       var_s = None
    C) Else: return (None, None, "unsupported")
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N = X.shape[0]

    def explicit_samples(owner, method_name: str):
        try:
            if method_name == "predict_samples":
                try:
                    result = owner.predict_samples(X, n_samples=n_samples)
                except TypeError:
                    result = owner.predict_samples(X)
                if isinstance(result, tuple):
                    mu_s = result[0]
                    var_s = result[1] if len(result) > 1 else None
                else:
                    mu_s, var_s = result, None
            else:
                mu_s, var_s = owner.predict_ensemble(X), None
            mu_s = np.asarray(mu_s)
            if mu_s.ndim == 2 and mu_s.shape[0] == N and mu_s.shape[1] != N:
                mu_s = mu_s.T
            mu_s = _validate_ensemble_preds(mu_s, N)
            if var_s is not None:
                var_s = np.asarray(var_s, dtype=float)
                if var_s.shape != mu_s.shape or not np.isfinite(var_s).all() or np.any(var_s < 0):
                    raise ValueError("var_s must be finite, nonnegative, and match mu_s")
            return mu_s, var_s, method_name
        except Exception as exc:
            raise ValueError(
                f"Explicit {type(owner).__name__}.{method_name} returned invalid predictive samples"
            ) from exc

    # Public wrapper interfaces take priority because they may transform latent
    # member predictions into response space.
    for method_name in ("predict_samples", "predict_ensemble"):
        if hasattr(model, method_name) and callable(getattr(model, method_name)):
            return explicit_samples(model, method_name)

    inner = getattr(model, "model", model)
    for method_name in ("predict_samples", "predict_ensemble"):
        if hasattr(inner, method_name) and callable(getattr(inner, method_name)):
            return explicit_samples(inner, method_name)

    # Strategy B: model has estimators_ attribute (ensemble models)
    if hasattr(inner, "estimators_") and inner.estimators_ is not None:
        estimators = inner.estimators_
        preds = np.stack([e.predict(X) for e in estimators], axis=0)  # (S, N)
        return _validate_ensemble_preds(preds, N), None, "ensemble_estimators"

    if hasattr(model, "estimators_") and model.estimators_ is not None:
        estimators = model.estimators_
        preds = np.stack([e.predict(X) for e in estimators], axis=0)
        return _validate_ensemble_preds(preds, N), None, "ensemble_estimators"

    # Strategy C: unsupported
    return None, None, "unsupported"


def estimate_noise_var_from_training_or_default(acquirer_manager, default: float = 1.0) -> float:
    """
    Estimate noise variance from training residuals if possible, else return default with warning.

    If acquirer_manager has data (X) and training targets (train_targets or data as (X, y) tuple),
    computes MSE on training residuals and returns max(MSE, 1e-8). Otherwise warns and returns
    max(default, 1e-8) to avoid MI explosions.

    Parameters
    ----------
    acquirer_manager : AcquireManager
        Instance with self.model, self.data; optionally self.train_targets or data as (X, y).
    default : float
        Default noise variance if estimation fails.

    Returns
    -------
    float
        Estimated or default noise variance (with safety floor 1e-8).
    """
    import warnings

    NOISE_FLOOR = 1e-8

    try:
        if hasattr(acquirer_manager, "data") and acquirer_manager.data is not None:
            data = acquirer_manager.data
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                X_train, y_train = np.asarray(data[0]), np.asarray(data[1])
            else:
                X_train = np.asarray(data)
                y_train = getattr(acquirer_manager, "train_targets", None)
                if y_train is not None:
                    y_train = np.asarray(y_train)
            if y_train is not None and len(y_train) == len(X_train):
                model = getattr(acquirer_manager, "model", None)
                if model is not None:
                    if hasattr(model, "get_means"):
                        y_pred = model.get_means(X_train)
                    elif hasattr(model, "predict"):
                        y_pred = model.predict(X_train)
                    else:
                        inner = getattr(model, "model", model)
                        y_pred = inner.predict(X_train)
                    return estimate_noise_var(y_train, y_pred, floor=NOISE_FLOOR)
    except Exception:
        pass

    warnings.warn(
        f"BALD: noise_var not provided and cannot estimate from training residuals. "
        f"Using noise_var={max(default, NOISE_FLOOR)}. BALD scores depend on target scaling. "
        f"Recommend passing noise_var explicitly or setting data=(X_train, y_train) or train_targets."
    )
    return max(default, NOISE_FLOOR)


def epig_gauss(
    model,
    train_data,
    rxnspace: np.ndarray,
    testdata: np.ndarray,
    sigma2: float = 1e-6,
    weights: Optional[np.ndarray] = None,
    pool_batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Gaussian ensemble EPIG (Expected Predictive Information Gain).

    The implementation used in the released benchmarks moment-matches ensemble
    predictions for each candidate-target pair to a bivariate Gaussian and adds
    ``sigma2`` observation noise before computing mutual information.

    Predictive model: y = f_θ(x) + ε with ε ~ N(0, σ²). sigma2 is observation/likelihood noise;
    residual variance from training (or held-out) is a reasonable plug-in but is an approximation.

    For RandomForest: treating trees as θ-samples is a heuristic; bootstrap + feature subsampling
    induce diversity but are not a calibrated posterior.

    Parameters
    ----------
    model : object
        Regressor that supplies ensemble predictions through ``estimators_``,
        ``predict_ensemble``, or a list of fitted models.
    train_data : optional
        Unused; retained for compatibility with the acquisition manager.
    rxnspace : np.ndarray
        Candidate pool, shape (N_pool, d).
    testdata : np.ndarray
        Test/target points, shape (M, d).
    sigma2 : float
        Observation noise variance. Recommend estimating from training residuals when possible.
        Very small values can make MI unstable. Default 1e-6.
    weights : np.ndarray, optional
        (M,) nonnegative weights for target points (discrete approximation to p(x*) or importance weights).
        If provided and not all zero, EPIG(x) = weighted average of per-target MI; else plain mean.

    Returns
    -------
    np.ndarray
        Acquisition scores, shape (N_pool,). Higher = more informative candidate.
    """
    rxnspace = np.asarray(rxnspace, dtype=float)
    testdata = np.asarray(testdata, dtype=float)
    if rxnspace.ndim == 1:
        rxnspace = rxnspace.reshape(1, -1)
    N_pool, _ = rxnspace.shape
    M = testdata.shape[0]
    sigma2 = max(float(sigma2), 1e-8)  # floor to avoid MI instability

    m_T = _ensemble_predict(model, testdata)
    S, M_check = m_T.shape
    if S < 2:
        raise ValueError("Ensemble must have at least 2 members (S >= 2) for sample variance.")
    if M_check != M:
        raise ValueError("Ensemble prediction shape (S, M) must match testdata length M.")

    if np.any(np.isnan(m_T)) or np.any(np.isinf(m_T)):
        raise ValueError(
            "Ensemble predictions contain NaN or inf; check model or inputs."
        )

    mT = m_T - m_T.mean(axis=0, keepdims=True)
    var_T = (mT * mT).sum(axis=0) / (S - 1) + sigma2

    if weights is not None:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.shape[0] != M:
            raise ValueError("weights must have length M (number of test points).")
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative.")
        use_weights = np.any(weights > 0)
    else:
        use_weights = False

    # Vectorized path: evaluate ensemble predictions for many pool points at once.
    # This removes the expensive Python loop over candidates and dramatically
    # reduces per-iteration EPIG overhead for large pools.
    if pool_batch_size is None or pool_batch_size <= 0:
        pool_batch_size = N_pool

    scores = np.zeros(N_pool, dtype=float)
    bad_target_var = var_T <= 0
    for p0 in range(0, N_pool, int(pool_batch_size)):
        p1 = min(N_pool, p0 + int(pool_batch_size))
        X_pool = rxnspace[p0:p1]

        # Ensemble means on pool batch: shape (S, B)
        m_X = _ensemble_predict(model, X_pool)
        if m_X.shape[0] != S or m_X.shape[1] != (p1 - p0):
            raise ValueError("Ensemble prediction shape mismatch for rxnspace batch.")
        if np.any(np.isnan(m_X)) or np.any(np.isinf(m_X)):
            raise ValueError("Ensemble predictions contain NaN or inf; check model or inputs.")

        mX = m_X - m_X.mean(axis=0, keepdims=True)  # (S, B)
        var_X = (mX * mX).sum(axis=0) / (S - 1) + sigma2  # (B,)
        var_X = np.maximum(var_X, 0.0)

        # cov(X, T) across ensemble members for all pool points in one matmul.
        cov_XT = (mX.T @ mT) / (S - 1)  # (B, M)
        vxvt = var_X[:, None] * var_T[None, :]  # (B, M)
        denom = vxvt - cov_XT * cov_XT
        # Use a relative guard so changing response units (and scaling sigma2
        # consistently) cannot change the score.
        denom = np.maximum(
            denom,
            EPS_DENOM_FACTOR * np.maximum(vxvt, np.finfo(float).tiny),
        )
        ratio = np.maximum(vxvt / denom, 1.0)

        mi = 0.5 * np.log(ratio)
        mi = np.maximum(mi, 0.0)
        if np.any(bad_target_var):
            mi[:, bad_target_var] = 0.0

        if use_weights:
            scores[p0:p1] = np.average(mi, axis=1, weights=weights)
        else:
            scores[p0:p1] = mi.mean(axis=1)

        # Numerical guard in case an extremely small/negative var_X slipped in.
        scores[p0:p1] = np.where(var_X > 0, scores[p0:p1], 0.0)

    return scores
