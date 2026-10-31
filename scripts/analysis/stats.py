#!/usr/bin/env python
"""Exact paired statistical primitives for benchmark comparisons.

All functions are deterministic and dependency-light (numpy + scipy only
for rankdata / friedman diagnostic). Nothing here reads data or touches the
manuscript; this module is pure statistics so it can be unit-tested in isolation.

Protocol (as specified for the task-level competitiveness sensitivity):

  * paired differences are computed directly;
  * absolute differences receive AVERAGE ranks on ties;
  * EXACT zero differences are discarded (Wilcoxon convention);
  * the exact Wilcoxon p-value is obtained by exhaustive sign-flip enumeration
    of the m nonzero differences (all 2**m assignments), NOT a normal
    approximation and NOT an automatic SciPy fallback;
  * p = 1 is returned when every paired difference is zero;
  * rounding to 12 decimals is applied to the paired differences BEFORE zero
    removal and before tie ranking.

Rounding convention
-------------------
The primary path rounds the paired differences to `ZERO_TIE_DECIMALS` first and
uses those rounded values for BOTH zero removal and tie ranking: values
indistinguishable at 12 dp are treated as tied.
`rank_rounded=False` retains the unrounded behaviour as a sensitivity check;
scripts/verify_results.py runs both and asserts the classifications are identical.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np
from scipy.stats import rankdata

ZERO_TIE_DECIMALS = 12
ALPHA = 0.05
# Guard on the exhaustive validation engine, which materialises a (2**m, m)
# sign matrix. The released analysis uses the exact dynamic-programming engine,
# which represents the same null distribution without that matrix.
ENUM_MAX_M = 20


@lru_cache(maxsize=None)
def _sign_matrix(m: int) -> np.ndarray:
    """All 2**m sign assignments in {-1,+1}, one per row (cached)."""
    if m > ENUM_MAX_M:
        raise ValueError(
            f"exhaustive Wilcoxon enumeration requested for m={m} > {ENUM_MAX_M}; "
            "this census is expected to have at most 10 paired seeds per task"
        )
    return np.array(list(product((-1.0, 1.0), repeat=m)), dtype=float)


def _nonzero_diffs(diffs, round_decimals=ZERO_TIE_DECIMALS):
    """
    Split diffs into (kept_unrounded, kept_rounded) after discarding exact zeros.

    Zeros are identified on the ROUNDED differences (12 dp by default) per the
    Wilcoxon convention. Both views of the surviving pairs are returned: the
    primary path ranks `kept_rounded` so that values equal at `round_decimals`
    tie, and `kept_unrounded` supports the unrounded sensitivity check.
    """
    diffs = np.asarray(diffs, dtype=float)
    rounded = np.round(diffs, round_decimals)
    keep = rounded != 0.0
    return diffs[keep], rounded[keep]


def _doubled_ranks(ranks: np.ndarray) -> np.ndarray:
    """Average ranks scaled to exact integers.

    `rankdata(..., method="average")` yields either integers or half-integers
    (a tie group of size k centred on its mean contributes .5 exactly when k is
    even), so 2*rank is always integral. Doubling keeps the DP over subset sums
    purely integer-valued, which is what makes the null distribution exact.
    """
    doubled = 2.0 * np.asarray(ranks, dtype=float)
    nearest = np.rint(doubled)
    if not np.allclose(doubled, nearest, rtol=0.0, atol=1e-9):
        raise ValueError("average ranks are not half-integral; cannot double exactly")
    return nearest.astype(np.int64)


@lru_cache(maxsize=None)
def _signed_rank_counts(doubled_ranks: tuple) -> tuple:
    """Exact counts of subset sums of the doubled ranks.

    Returns a tuple `counts` of exact Python ints where counts[w] is the number
    of the 2**m sign assignments whose positive-rank sum (in doubled units) is
    exactly w. This is the full Wilcoxon signed-rank null distribution: under H0
    each sign is independently +/-1 with probability 1/2, and

        S = sum_i s_i * r_i = 2 * W_plus - total   (all in doubled units)

    so S is a strictly increasing function of W_plus and the two carry identical
    tail probabilities. Cost is O(m * total) instead of O(2**m), and every count
    is an exact integer (Python ints, no overflow), so no precision is lost even
    at m = 30 where enumeration would need ~258 GB.
    """
    total = int(sum(doubled_ranks))
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in doubled_ranks:
        r = int(r)
        # walk downward so each rank is used at most once (0/1 knapsack)
        for w in range(total, r - 1, -1):
            c = counts[w - r]
            if c:
                counts[w] += c
    return tuple(counts)


def _dp_tail_probabilities(ranks: np.ndarray, s_obs_doubled: int) -> tuple:
    """Exact two-sided and one-sided tail probabilities from the DP distribution.

    Mirrors the enumeration comparisons exactly. With S = 2*W_plus - total in
    doubled units, the enumeration's `|S_perm| >= |S_obs|`, `S_perm >= S_obs` and
    `S_perm <= S_obs` become exact integer comparisons -- no tolerance needed,
    because doubled ranks make every attainable S an integer.
    """
    doubled = _doubled_ranks(ranks)
    counts = _signed_rank_counts(tuple(int(v) for v in doubled))
    total = int(doubled.sum())
    n_perm = 2 ** int(doubled.size)

    two_sided = greater = less = 0
    abs_obs = abs(int(s_obs_doubled))
    for w, c in enumerate(counts):
        if not c:
            continue
        s = 2 * w - total
        if abs(s) >= abs_obs:
            two_sided += c
        if s >= s_obs_doubled:
            greater += c
        if s <= s_obs_doubled:
            less += c
    return (two_sided / n_perm, greater / n_perm, less / n_perm)


def exact_wilcoxon_signed_rank(
    diffs, round_decimals=ZERO_TIE_DECIMALS, engine="dp", rank_rounded=True
):
    """
    Exact two-sided Wilcoxon signed-rank test.

    Two interchangeable exact engines:
      * engine="dp"   (default, production) -- exact DP over the signed-rank-sum
        null distribution; ranks are doubled to integers so the recursion is
        exact.
      * engine="enum" -- the original exhaustive 2**m sign-flip enumeration,
        retained as the validation reference and capped at ENUM_MAX_M.

    Conventions are identical under both engines: the paired differences are
    rounded to `round_decimals`, exact-zero differences are then discarded, tied
    absolute differences receive AVERAGE ranks, and p = 1 is returned when every
    paired difference is zero.

    Parameters
    ----------
    diffs : array-like
        Paired differences (already a - b).
    round_decimals : int
        Decimals at which differences are considered indistinguishable, used for
        both zero removal and tie ranking.
    rank_rounded : bool
        True (default, primary path): rank the rounded differences, so values
        equal at `round_decimals` are tied. False: rank the unrounded values --
        the sensitivity variant; scripts/verify_results.py asserts both agree.
    engine : {"dp", "enum"}
        Exact engine to use. Both give identical results; see module docstring.

    Returns
    -------
    dict with keys:
        n_pairs, n_nonzero, w_plus, w_minus, signed_rank_sum, p_value,
        rank_biserial, all_zero
    """
    diffs = np.asarray(diffs, dtype=float)
    n_pairs = int(diffs.size)
    kept_unrounded, kept_rounded = _nonzero_diffs(diffs, round_decimals)
    kept = kept_rounded if rank_rounded else kept_unrounded
    m = int(kept.size)

    if m == 0:
        # Every paired difference is (rounded to) zero.
        return {
            "n_pairs": n_pairs,
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "signed_rank_sum": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
            "all_zero": True,
        }

    ranks = rankdata(np.abs(kept), method="average")
    signs = np.sign(kept)
    signed_ranks = signs * ranks
    s_obs = float(signed_ranks.sum())

    w_plus = float(ranks[kept > 0].sum())
    w_minus = float(ranks[kept < 0].sum())
    total_rank = float(ranks.sum())  # == w_plus + w_minus

    # Null distribution of S_perm = sum(sign_i * rank_i), symmetric about 0:
    #   two-sided p    = P(|S_perm| >= |S_obs|)
    #   one-sided (>)  = P(S_perm >= S_obs)   [H1: a > b, i.e. positive shift]
    #   one-sided (<)  = P(S_perm <= S_obs)   [H1: a < b]
    if engine == "dp":
        # Exact DP; doubled ranks make every comparison exact integer arithmetic.
        s_obs_doubled = int(np.rint(2.0 * s_obs))
        p_value, p_greater, p_less = _dp_tail_probabilities(ranks, s_obs_doubled)
    elif engine == "enum":
        # Exhaustive enumeration of all 2**m sign assignments to the ranks.
        tol = 1e-9
        n_perm = 2 ** m
        s_perm = _sign_matrix(m) @ ranks  # every one of the 2**m assignments
        p_value = float(np.count_nonzero(np.abs(s_perm) >= abs(s_obs) - tol)) / n_perm
        p_greater = float(np.count_nonzero(s_perm >= s_obs - tol)) / n_perm
        p_less = float(np.count_nonzero(s_perm <= s_obs + tol)) / n_perm
    else:
        raise ValueError(f"unknown engine {engine!r}; expected 'dp' or 'enum'")

    rank_biserial = (w_plus - w_minus) / total_rank if total_rank > 0 else 0.0

    return {
        "n_pairs": n_pairs,
        "n_nonzero": m,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "signed_rank_sum": s_obs,
        "p_value": float(p_value),
        "p_greater": float(p_greater),
        "p_less": float(p_less),
        "rank_biserial": float(rank_biserial),
        "all_zero": False,
    }


def holm_adjust(pvalues):
    """
    Holm step-down adjustment across a family of p-values.

    Returns adjusted p-values in the ORIGINAL order. Monotonicity is enforced
    (adjusted values are non-decreasing along the sorted order) and each value
    is capped at 1.0.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    adjusted_sorted = np.empty(n, dtype=float)
    running_max = 0.0
    for i in range(n):
        val = (n - i) * sorted_p[i]  # (n - i) == (n - rank + 1) for 0-based i
        val = min(val, 1.0)
        running_max = max(running_max, val)
        adjusted_sorted[i] = running_max
    out = np.empty(n, dtype=float)
    out[order] = adjusted_sorted
    return out


def friedman_diagnostic(blocks_by_method):
    """
    Friedman omnibus across methods (columns) over paired blocks (rows).

    Parameters
    ----------
    blocks_by_method : 2D array-like, shape (n_blocks, n_methods)

    Returns
    -------
    dict(statistic, p_value, n_blocks, n_methods) or NaNs if degenerate.
    Diagnostic only -- never used to gate pairwise classifications.
    """
    from scipy.stats import friedmanchisquare

    arr = np.asarray(blocks_by_method, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] < 2:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_blocks": int(arr.shape[0]) if arr.ndim == 2 else 0,
            "n_methods": int(arr.shape[1]) if arr.ndim == 2 else 0,
        }
    columns = [arr[:, j] for j in range(arr.shape[1])]
    try:
        stat, p = friedmanchisquare(*columns)
    except Exception:
        stat, p = float("nan"), float("nan")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_blocks": int(arr.shape[0]),
        "n_methods": int(arr.shape[1]),
    }
