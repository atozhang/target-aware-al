"""Native integer-step estimators for the experimental-efficiency endpoint.

A method's efficiency on a task is the first integer number of acquired
reactions at which its method-level mean target-set R2 reaches

    target = R0 + q * (Rmax30 - R0),

where R0 is the shared step-zero mean and Rmax30 is the highest method-level
mean R2 observed for any method at an integer step from zero through the
30%-of-pool budget, inclusive. The curves are not smoothed, interpolated, or
made monotone.

The primary rule credits the first meeting step. The two-consecutive
sensitivity credits the first step that meets the target only when the next
integer step also meets it. Thus, at native resolution, that sensitivity asks
whether the target remains met for one additional acquisition.
"""
from __future__ import annotations

import numpy as np

# A curve value counts as meeting the target if it is within this tolerance.
TARGET_TOL = 1e-12


def efficiency_target(initial_score: float, ceiling: float, q: float) -> float:
    """Return ``R0 + q * (Rmax30 - R0)``."""
    return initial_score + q * (ceiling - initial_score)


def _validate_native_curve(y: np.ndarray) -> np.ndarray:
    curve = np.asarray(y, dtype=float)
    if curve.ndim != 1 or not len(curve):
        raise ValueError("native curve must be a non-empty one-dimensional array")
    if not np.isfinite(curve).all():
        raise ValueError("native curve contains a non-finite score")
    return curve


def native_first_crossing(y: np.ndarray, target: float) -> int | None:
    """First integer step whose raw method-level mean reaches ``target``.

    ``None`` denotes nonattainment within the supplied native-step curve.
    """
    curve = _validate_native_curve(y)
    reached = np.flatnonzero(curve >= float(target) - TARGET_TOL)
    return None if not len(reached) else int(reached[0])


def native_two_consecutive_crossing(y: np.ndarray, target: float) -> int | None:
    """First meeting step for which the next integer step also meets target.

    A first meeting that occurs only at the final budget step cannot be
    confirmed by this sensitivity rule and is therefore nonattainment.
    """
    curve = _validate_native_curve(y)
    meets = curve >= float(target) - TARGET_TOL
    reached = np.flatnonzero(meets[:-1] & meets[1:])
    return None if not len(reached) else int(reached[0])


def crossing_or_budget(crossing: int | None, budget: int) -> tuple[bool, int]:
    """Return ``(attained, acquired reactions)`` under the full-budget rule.

    The separate flag distinguishes nonattainment, which is charged the full
    budget, from an actual first crossing at exactly the budget.
    """
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if crossing is None:
        return False, budget
    step = int(crossing)
    if step < 0 or step > budget:
        raise ValueError(f"crossing {step} lies outside native budget 0..{budget}")
    return True, step
