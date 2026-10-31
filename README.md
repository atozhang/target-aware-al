# A General Strategy for Active Learning When Predicting Reaction Performance in a Target Domain

This repository is the code and data companion to *A General Strategy for Active
Learning When Predicting Reaction Performance in a Target Domain*. It contains
implementations of EPIG, BALD, Indiv. NN, NN, and Random acquisition, the
released datasets and results for all 118 tasks, and scripts that reproduce
selected figures and aggregate analyses.

The evaluation covers 118 active learning tasks from four reaction datasets:
Buchwald–Hartwig coupling, Rh-catalyzed asymmetric olefin hydrogenation,
Sharpless asymmetric dihydroxylation, and Wittig olefination. Using AUC₂–₁₂, the
normalized average R² on the target set over the 2–12 % acquisition interval,
EPIG has the highest mean on 74/118 tasks (63 %) and is statistically competitive
on 105/118 tasks (89 %), meaning that it is not detected as significantly worse
than the per-task highest-mean method. At the experimental-efficiency endpoint
of 80 % of achievable improvement, EPIG saves a mean of 81 reactions [95 % CI
46, 121] relative to Random acquisition, equivalent to 6.9 [4.1, 9.9]
percentage points of the acquisition pool.

## Installation

For reproducibility, use the pinned conda environment:

```bash
conda env create -f environment.yml
conda activate epig-al
```

This installs the repository as an editable package (`epig_al`). Alternatively,
install the package with pip (dependencies may resolve to newer compatible versions):

```bash
pip install -e .                       # method layer only
pip install -e ".[figures,notebook]"   # + figure rendering and the notebooks
```

## Using EPIG

`select_batch` scores a pool of candidate experiments and returns the indices of
the `k` highest-scoring candidates. EPIG is target-aware: it values each
candidate by the predictive information its outcome is expected to provide about
the reactions in `X_target`.

```python
from sklearn.ensemble import RandomForestRegressor
from epig_al import select_batch

model = RandomForestRegressor(n_estimators=300, random_state=0)
model.fit(X_train, y_train)

idx = select_batch(
    model,
    X_pool,                 # unlabeled candidate experiments
    X_target,               # the reactions you want to predict well
    X_train=X_train,
    y_train=y_train,
    k=8,
    metric="epig",          # or "bald", "nn", "indv_nn", "random"
)
```

`ActiveLearner` fits a model, scores the remaining candidates, acquires
reactions, and refits after each round in a retrospective simulation:

```python
from epig_al import ActiveLearner

learner = ActiveLearner(
    model_factory=lambda: RandomForestRegressor(n_estimators=300, random_state=0),
    metric="epig",
    batch_size=1,
)
trajectory = learner.run(
    X, y,
    labeled=initial_indices,
    pool=pool_indices,
    target=target_indices,
    n_rounds=30,
)
```

`trajectory` is a learning curve evaluated after each acquisition round (R² on
the target set by default).
`notebooks/using_epig.ipynb` walks through a complete example on public data.

## Data

| Dataset | Reactions | Source |
|---|---|---|
| Buchwald–Hartwig | 3,955 | Ahneman et al., *Science* **2018** ([10.1126/science.aar5169](https://doi.org/10.1126/science.aar5169)) |
| Rh hydrogenation | 960 | Kalikadien et al., *Chem. Sci.* **2024** ([10.1039/D4SC03647F](https://doi.org/10.1039/D4SC03647F)) |
| Sharpless AD | 1,007 | Ocampo et al., *ACS Cent. Sci.* **2025** ([10.1021/acscentsci.5c00900](https://doi.org/10.1021/acscentsci.5c00900)) |
| Wittig (literature) | 734 | Vedejs & Peterson, *Topics in Stereochemistry* **1994** ([10.1002/9780470147306.ch1](https://doi.org/10.1002/9780470147306.ch1)) |
| Wittig (CAS) | 233 | Publications and patents dated 2010–2015, sourced from the CAS Content Collection™ (CAS is a division of the American Chemical Society); see [CAS Reactions](https://www.cas.org/cas-data/cas-reactions) |

The Rh dataset is cited but not redistributed here; see `data/README.md` and
`data/LICENSES.md` for its source and availability.

`data/README.md` describes the released data and cites each source;
`data/LICENSES.md` records provenance, attribution, and licensing information for
redistributed datasets and assets.

**Wittig data.** The 967 Wittig reactions of the study (734 from the literature
and 233 curated from CAS reaction records) are released with their F119
descriptors in `data/Wittig/`. The CAS records are redistributed with the
permission of CAS (`data/LICENSES.md`).

## Reproducing reported analyses

The analyses use the per-seed inputs and mean learning curves in `results/`; the
original active learning campaigns are not rerun. The main entry points are:

```bash
python scripts/verify_results.py        # recompute task classifications and summary counts
python scripts/efficiency_endpoint.py   # recompute the experimental-efficiency endpoint
python scripts/make_figures.py          # redraw the redistributable figure panels
python scripts/wittig_qualitative_study.py   # rerun the qualitative Wittig study (Fig 3E)
python -m pytest -q                     # run the test suite
```

`verify_results.py` recomputes the highest-mean and Wilcoxon–Holm competitiveness
classifications across all seven budget windows and checks the released summary
tables. `efficiency_endpoint.py` reconstructs the per-task experimental-
efficiency table and its summaries from the released native integer-step
method-mean curves. The endpoint input, its metadata, and checksum are in
`results/benchmark_inputs/native_method_mean_curves_0_30.*`.
`make_figures.py` redraws the redistributable figure panels. Other scripts under
`scripts/` reproduce the savings and sensitivity analyses and SI data exports;
`results/README.md` and `supporting_information/README.md` identify their inputs
and outputs.

For an annotated walkthrough, see `notebooks/reproduce_main_results.ipynb`.

To execute both notebooks headlessly from the repository root without changing
the committed files:

```bash
jupyter execute notebooks/using_epig.ipynb --kernel_name=python3 --timeout=-1 --output=/tmp/using_epig.executed.ipynb
jupyter execute notebooks/reproduce_main_results.ipynb --kernel_name=python3 --timeout=-1 --output=/tmp/reproduce_main_results.executed.ipynb
```

**Scope.** The qualitative Wittig study of Fig 3E is rerun from the released data
(`results/wittig_qualitative_study/README.md`).

### Numerical reproducibility

The released analyses and tests were validated in the pinned conda environment
on Linux x86-64, as used by the GitHub Actions test workflow. Differences among
numerical libraries and operating systems may affect last-digit floating-point
values or the ordering of nearly tied acquisition scores. These differences do
not affect the rounded results or task classifications reported here.

## Repository contents

| Path | Contents |
|---|---|
| `epig_al/` | The EPIG method layer: acquisition functions and the selection API. |
| `data/` | The public reaction datasets with precomputed descriptors (`data/README.md`). |
| `results/` | Released per-seed inputs and every committed result table (`results/README.md`). |
| `supporting_information/` | The 118-task table, SI classification and count tables, and the SI data exports (`supporting_information/README.md`). |
| `figures/` | Redistributable figure panels and the bundled typeface (`figures/README.md`). |
| `notebooks/` | `using_epig.ipynb` (tutorial) and `reproduce_main_results.ipynb` (headline recomputation). |
| `scripts/` | Verification, analysis and figure scripts. |
| `tests/` | The test suite. |

## Tests

```bash
pytest -q
```

The suite covers mathematical invariants of the acquisition methods, released
statistical analyses, sensitivity analyses, and regeneration checks for the SI
data exports. `tests/conftest.py` caps BLAS/OpenMP threads for reproducible
numerical tests.

## Citation

Software citation metadata is in `CITATION.cff`. A citation for the accompanying
manuscript will be added when it is available.

## License

Repository software is MIT-licensed (see `LICENSE`). Data and asset provenance
and licensing are documented in `data/LICENSES.md` and the corresponding asset
license files.
