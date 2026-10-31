"""Session-wide test configuration.

Caps the native thread pools to one thread each before any test module imports
the numeric stack. Every test here is small and deterministic and none benefits
from intra-op parallelism, so this is both the right default and a guard against
exhausting the process limit on a shared host. Pre-existing values are respected.
"""

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",         # libgomp / OpenMP (sklearn, scipy)
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

for _var in _THREAD_VARS:
    os.environ.setdefault(_var, "1")
