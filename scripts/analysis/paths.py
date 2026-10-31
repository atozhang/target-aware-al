#!/usr/bin/env python
"""Path resolution for the benchmark analysis package.

No repository path is hard-coded anywhere in this package: the repository root is
derived from this file's location, so the code runs unchanged in any clone,
worktree, or fresh checkout.
"""
from __future__ import annotations

from pathlib import Path

# scripts/analysis/paths.py -> scripts/analysis -> scripts -> repo
PACKAGE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PACKAGE_DIR.parent
REPO = SCRIPTS_DIR.parent

RESULTS = REPO / "results"
BENCHMARK_INPUTS = RESULTS / "benchmark_inputs"
BENCHMARK = RESULTS / "benchmark"
EFFICIENCY = RESULTS / "efficiency"
SENSITIVITY = RESULTS / "sensitivity"
SI_TABLES = REPO / "supporting_information/tables/competitiveness"
FIGURES = REPO / "figures"

# per-replicate window AUCs for the ten Rh random-split tasks, all seven windows
RH_RANDOM_SPLIT_PER_SEED = BENCHMARK_INPUTS / "rh_random_split_per_seed_window_auc.csv"


def rel(path: Path) -> str:
    """Repo-relative string; absolute if the path lies outside the repository."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)
