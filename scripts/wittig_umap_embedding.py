#!/usr/bin/env python
"""Regenerate the Wittig UMAP coordinates of Fig 3C and Fig 3E.

The released coordinates (`umap_1`, `umap_2` in
``results/wittig_qualitative_study/split_membership.csv``) are a UMAP embedding of
all 967 Wittig reactions in ``data/Wittig`` (literature rows, then CAS rows):

* input: the 119 F119 descriptors in ``feature_config.STANDARDIZED_FEATURES`` order;
* preprocessing: ``StandardScaler`` fit on all 967 rows;
* embedding: ``umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.18,
  metric="euclidean", random_state=42).fit_transform``, float32 output.

No filtering, transformation or adjustment is applied to the coordinates. They
were produced with umap-learn 0.5.11, numba 0.63.1, llvmlite 0.46.0, pynndescent
0.5.13, scikit-learn 1.8.0 and numpy 1.26.4 on Linux x86-64; with these versions
this script reproduces them exactly. Other versions or platforms can give a
different layout, so the released coordinates remain the reference input of
``scripts/wittig_qualitative_study.py``. The dependency is intentionally
optional; install it with ``pip install -e ".[umap]"``. Run from the repository
root:

    python scripts/wittig_umap_embedding.py            # compare with the released coordinates
    python scripts/wittig_umap_embedding.py --out DIR  # also write the regenerated coordinates
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "Wittig"))
from feature_config import STANDARDIZED_FEATURES  # noqa: E402

DATA = [REPO / "data/Wittig/wittig_literature_f119.csv", REPO / "data/Wittig/wittig_cas_f119.csv"]
RELEASED = REPO / "results/wittig_qualitative_study/split_membership.csv"
UMAP_KW = dict(n_components=2, n_neighbors=20, min_dist=0.18, metric="euclidean", random_state=42)


def embedding(data: pd.DataFrame) -> np.ndarray:
    import umap

    scaled = StandardScaler().fit_transform(data[STANDARDIZED_FEATURES].to_numpy(float))
    return umap.UMAP(**UMAP_KW).fit_transform(scaled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="also write the regenerated coordinates to this directory")
    args = parser.parse_args()

    data = pd.concat([pd.read_csv(path) for path in DATA], ignore_index=True)
    released = pd.read_csv(RELEASED)
    if len(data) != 967 or released["record_id"].tolist() != data["record_id"].tolist():
        raise SystemExit("split_membership.csv is not in released reaction order")
    xy = embedding(data)
    reference = released[["umap_1", "umap_2"]].to_numpy(float).astype(np.float32)
    same = xy.dtype == np.float32 and np.array_equal(xy, reference)
    diff = float(np.abs(xy.astype(float) - reference.astype(float)).max())
    print(f"967 x 119 -> UMAP {xy.shape}; coordinates equal to split_membership.csv: {same} "
          f"(max |difference| {diff:.3g})")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"record_id": data["record_id"], "umap_1": xy[:, 0], "umap_2": xy[:, 1]}).to_csv(
            args.out / "umap_coordinates.csv", index=False)
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
