"""Acquisition functions for active learning over reaction space.

The :class:`~epig_al.acquisition.base.AcquireManager` dispatches to every
acquisition rule compared in the manuscript.  The five reported in the main
text map onto manager metrics as follows:

================  ==========================
Paper name        ``AcquireManager`` metric
================  ==========================
Random            ``"random"``
NN                ``"nn"``
Indiv. NN         ``"test_indv_nn"``
BALD              ``"bald"``
EPIG              ``"epig_gauss"``
================  ==========================

The lower-level array math (EPIG, BALD, noise estimation, ...) lives in
:mod:`epig_al.acquisition.metrics`.
"""

from .base import AcquireManager

__all__ = ["AcquireManager"]
