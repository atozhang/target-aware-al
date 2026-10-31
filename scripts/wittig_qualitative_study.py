#!/usr/bin/env python
"""Reproduce the qualitative Wittig active learning study (Fig 3E).

The study is separate from the 118-task benchmark. It uses all 967 Wittig
reactions (``data/Wittig``: 734 literature + 233 CAS-derived, F119 descriptors):

* target: Ward cluster 12 of 16 (``AgglomerativeClustering`` on standardized
  F119 descriptors), restricted to the cluster's largest connected component
  on the F119 UMAP (points closer than 1.0 UMAP unit are linked): 63 reactions.
  The cluster's other members remain in the pool;
* pool: the other 904 reactions. For each seed (0, 1, 2), 25 pool reactions
  form the initial labeled set, leaving 879 acquisition candidates;
* 25 sequential acquisitions (one reaction per step, highest score) with EPIG,
  BALD and Indiv. NN. The model is a 32-member bagged XGBoost regressor trained
  on logit-transformed %Z; acquisition scores use the ensemble in logit units.

The script regenerates the split and the 225 acquisitions (about 5 min on one
CPU core; requires ``xgboost``) and compares them with
``results/wittig_qualitative_study/``. Run from the repository root:

    python scripts/wittig_qualitative_study.py              # split + acquisitions
    python scripts/wittig_qualitative_study.py --split-only # split only (seconds)
    python scripts/wittig_qualitative_study.py --out DIR    # also write the regenerated tables
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data" / "Wittig"))
from feature_config import STANDARDIZED_FEATURES  # noqa: E402

DATA = [REPO / "data/Wittig/wittig_literature_f119.csv", REPO / "data/Wittig/wittig_cas_f119.csv"]
STUDY = REPO / "results/wittig_qualitative_study"
STUDY_ID = "wittig_f119_qualitative_k16_cl12"
N_CLUSTERS, TARGET_CLUSTER, UMAP_LINK = 16, 12, 1.0
SEEDS, N_INITIAL, N_STEPS, N_BAGS = (0, 1, 2), 25, 25, 32
METHODS = {"epig_gauss": "EPIG", "bald": "BALD", "test_indv_nn": "IndvNN"}
# every initial-set, model and acquisition seed is derived from this label
SEED_LABEL, BASE_SEED = "wittig_case_compact_cl6_main_component", 20260522


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return (BASE_SEED + int(digest[:8], 16)) % (2**32 - 1)


def load_reactions() -> pd.DataFrame:
    """The 967 reactions in released order: literature rows, then CAS rows."""
    frames = [pd.read_csv(path) for path in DATA]
    data = pd.concat(frames, ignore_index=True).copy()
    data["source"] = ["literature"] * len(frames[0]) + ["cas"] * len(frames[1])
    if len(data) != 967 or not data["record_id"].is_unique:
        raise ValueError("expected 967 uniquely identified reactions")
    return data


def study_split(data: pd.DataFrame, umap: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Target, pool and per-seed initial labeled rows (indices into the 967 reactions)."""
    z = StandardScaler().fit_transform(data[STANDARDIZED_FEATURES].to_numpy(float))
    cluster = np.where(AgglomerativeClustering(n_clusters=N_CLUSTERS).fit_predict(z) == TARGET_CLUSTER)[0]
    pairs = np.array(sorted(cKDTree(umap[cluster]).query_pairs(UMAP_LINK)))
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cluster),) * 2)
    component = connected_components(graph, directed=False)[1]
    target = np.sort(cluster[component == np.bincount(component).argmax()])
    pool = np.setdiff1d(np.arange(len(data)), target, assume_unique=True)
    initial = {seed: np.sort(np.random.default_rng(stable_seed("warm_start", "Wittig", SEED_LABEL, seed))
                             .choice(pool, size=N_INITIAL, replace=False)) for seed in SEEDS}
    return target, pool, initial


def _member_mean(members: np.ndarray) -> np.ndarray:
    total = np.zeros(members.shape[1], dtype=np.float64)
    for member in np.asarray(members, dtype=np.float64):      # fixed reduction order
        np.add(total, member, out=total)
    return total / float(len(members))


class BaggedXGB:
    """32-member bagged XGBoost on logit(%Z / 100); predictions are in logit units.

    It deliberately has no ``model`` or ``estimators_`` attribute, so the acquisition
    functions score its ``predict_ensemble`` members."""

    def __init__(self, seed: int):
        from xgboost import XGBRegressor

        from epig_al.acquisition.ensemble_utils import make_bagged_xgb

        base = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, colsample_bytree=0.8,
                            random_state=seed, n_jobs=1, verbosity=0)
        self.bagging = make_bagged_xgb(base, n_estimators=N_BAGS, max_samples=0.8, max_features=1.0,
                                     bootstrap=True, random_state=seed, ensure_stochastic=True).set_params(n_jobs=1)

    @staticmethod
    def logit(y: np.ndarray) -> np.ndarray:
        unit = np.clip(np.asarray(y, dtype=float) / 100.0, 1e-6, 1 - 1e-6)
        return np.log(unit / (1.0 - unit))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaggedXGB":
        self.bagging.fit(x, self.logit(y))
        return self

    def predict_ensemble(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([estimator.predict(x) for estimator in self.bagging.estimators_])

    def predict_samples(self, x: np.ndarray, n_samples: int | None = None):
        return self.predict_ensemble(x), None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _member_mean(self.predict_ensemble(x))


def acquisitions(data: pd.DataFrame, target: np.ndarray, pool: np.ndarray, initial: dict[int, np.ndarray]) -> pd.DataFrame:
    from epig_al.acquisition.base import AcquireManager
    from epig_al.acquisition.metrics import estimate_noise_var

    x, y = data[STANDARDIZED_FEATURES].to_numpy(float), data["z_selectivity_percent"].to_numpy(float)
    position = {int(row): i for i, row in enumerate(pool)}
    rows = []
    for seed in SEEDS:
        for acq, method in METHODS.items():
            acquired = [position[int(row)] for row in initial[seed]]
            manager = AcquireManager(n_iter=N_STEPS + 1, k_batch=1, metric=acq)
            AcquireManager.set_seed(stable_seed("acquisition", "Wittig", SEED_LABEL, seed, acq))
            for step in range(N_STEPS):
                train = np.sort(pool[acquired])
                model = BaggedXGB(stable_seed("model", "Wittig", SEED_LABEL, seed, step)).fit(x[train], y[train])
                taken = set(acquired)
                remaining = np.asarray([i for i in range(len(pool)) if i not in taken], dtype=int)
                sigma2 = (estimate_noise_var(model.logit(y[train]), model.predict(x[train]))
                          if acq in ("epig_gauss", "bald") else 1e-6)
                scores = np.asarray(manager.calc(metric=acq, model=model, data=x[train], rxnspace=x[pool[remaining]],
                                                 testdata=x[target], sigma2=sigma2, noise_var=sigma2), dtype=float)
                best = int(np.argmax(scores))
                acquired.append(int(remaining[best]))
                rows.append({"seed": seed, "method": method, "acquisition_rank": step + 1,
                             "record_id": data["record_id"].iat[pool[remaining[best]]],
                             "acquisition_score": float(scores[best])})
            print(f"  seed {seed} {method}: done", flush=True)
    return pd.DataFrame(rows)


def membership_table(data: pd.DataFrame, target: np.ndarray, initial: dict[int, np.ndarray]) -> pd.DataFrame:
    table = data[["record_id", "source"]].copy()
    for seed in SEEDS:
        role = np.full(len(data), "candidate", dtype=object)
        role[initial[seed]] = "initial"
        role[target] = "target"
        table[f"role_seed{seed}"] = role
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--split-only", action="store_true", help="regenerate and compare only the split")
    parser.add_argument("--out", type=Path, help="also write the regenerated tables to this directory")
    args = parser.parse_args()

    data = load_reactions()
    released = pd.read_csv(STUDY / "split_membership.csv", dtype={"umap_1": str, "umap_2": str})
    if released["record_id"].tolist() != data["record_id"].tolist():
        raise SystemExit("split_membership.csv is not in released reaction order")
    target, pool, initial = study_split(data, released[["umap_1", "umap_2"]].astype(float).to_numpy())
    membership = membership_table(data, target, initial)
    for seed in SEEDS:
        counts = membership[f"role_seed{seed}"].value_counts().to_dict()
        print(f"seed {seed}: {counts.get('target', 0)} target, {counts.get('initial', 0)} initial, "
              f"{counts.get('candidate', 0)} candidates")
    ok = membership.equals(released[membership.columns])
    print(f"split matches split_membership.csv: {ok}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        membership.assign(umap_1=released["umap_1"], umap_2=released["umap_2"]).to_csv(args.out / "split_membership.csv", index=False)
    if not args.split_only:
        sequence = acquisitions(data, target, pool, initial)
        published = pd.read_csv(STUDY / "acquisitions.csv")
        keys = ["seed", "method", "acquisition_rank", "record_id"]
        same = sequence[keys].equals(published[keys])
        close = same and np.allclose(sequence["acquisition_score"], published["acquisition_score"], rtol=1e-9, atol=0)
        print(f"{len(sequence)} acquisitions match acquisitions.csv: {same} (scores equal: {close})")
        ok = ok and close
        if args.out:
            sequence.assign(study_id=STUDY_ID)[["study_id"] + list(sequence.columns)].to_csv(
                args.out / "acquisition_sequences.csv", index=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
