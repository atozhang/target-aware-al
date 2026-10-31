"""Acquisition methods and selection interfaces for target-aware active learning.

The package implements EPIG, BALD, Indiv. NN, NN, and Random acquisition and
exports :func:`acquisition_scores`, :func:`select_batch`, and
:class:`ActiveLearner`. Repository datasets, result tables, and figures are not
bundled with the installed package.
"""

__version__ = "1.0.0"

from .select import ActiveLearner, acquisition_scores, select_batch

__all__ = [
    "acquisition_scores",
    "select_batch",
    "ActiveLearner",
    "__version__",
]
