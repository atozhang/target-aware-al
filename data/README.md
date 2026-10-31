# Datasets and descriptors

This directory contains the publicly redistributed reaction data and descriptors used in the study. All prediction targets are continuous. See `LICENSES.md` for source attribution and licensing information.

| Directory | Reaction | Prediction target | Records | Representation |
|---|---|---|---|---|
| `BH/` | Buchwald–Hartwig C–N coupling | Yield (%) | 3,960 (3,955 labeled) | 120 atom and DFT descriptors |
| `SAD/` | Sharpless asymmetric dihydroxylation | ΔΔG derived from er (kcal/mol) | 1,007 | Quadrant-based descriptors |
| `Wittig/` | Wittig olefination | E/Z selectivity (`%Z`) | 967 (734 literature + 233 CAS) | 119 numerical descriptors (F119) |

The Rh reaction data are not redistributed in this repository. See the Rh section below for the original data source and the location of the released results.

## Buchwald–Hartwig coupling (`BH/`)

**Source:** Ahneman, Estrada, Lin, Dreher, and Doyle, *Science* (2018). [DOI: 10.1126/science.aar5169](https://doi.org/10.1126/science.aar5169).

Redistributed from the authors' MIT-licensed repository: [doylelab/rxnpredict](https://github.com/doylelab/rxnpredict).

- `BH_HTE_scaled_3.csv`: Reaction descriptors and yields for the complete 3,960-reaction experimental design. Five reactions have no reported yield in the original data and are excluded from the evaluation, leaving 3,955 labeled reactions.

The table does not contain reaction identity columns. The corresponding reaction identities are available from the original source.

## Rh-catalyzed reactions

**Source:** Kalikadien, Valsecchi, van Putten, Maes, Muuronen, Dyubankova, Lefort, and Pidko, *Chemical Science* (2024). [DOI: 10.1039/D4SC03647F](https://doi.org/10.1039/D4SC03647F).

The original 960-reaction dataset is available from [4TU.ResearchData](https://doi.org/10.4121/ecbd4b91-c434-4bdf-a0ed-4e9e0fb05e94.v1) under CC BY 4.0 but is not redistributed here.

The representation and data splits used in this study are described in `../results/rh_published_split/README.md`. The corresponding evaluation results are available in `../results/rh_published_split/`.

## Sharpless asymmetric dihydroxylation (`SAD/`)

**Source:** Ocampo, Altundas, Bock, Feiz, and Denmark, *ACS Central Science* (2025). [DOI: 10.1021/acscentsci.5c00900](https://doi.org/10.1021/acscentsci.5c00900).

Redistributed from the authors' MIT-licensed repository: [SEDenmarkLab/SAD](https://github.com/SEDenmarkLab/SAD). The authors request citation of the paper above when using their database.

- `SAD_Database.csv`: The complete database of 1,007 reactions across six olefin types: Trans (325), Tri (252), Mono (182), Gem (164), Cis (53), and Tetra (31).
- `benchmark_splits/*.hdf5`: The authors' training and test splits defining seven prediction tasks. The trisubstituted olefins are divided into TriQ2 and TriQ4; each remaining olefin type defines one task.

Each split file contains `X_Train`, `X_Test`, `Y_Train`, and `Y_Test`. The response is `ddG er (kcal/mol)`, and the descriptors include quadrant-specific and shared features.

The files use the pandas HDF5 format. Reading them with `pandas.read_hdf` requires the optional `tables` package, which is not needed to run the released evaluation code.

## Wittig olefination (`Wittig/`)

**Sources:** Vedejs and Peterson, "Stereochemistry and Mechanism in the Wittig Reaction," *Topics in Stereochemistry*, Vol. 21 (1994). [DOI: 10.1002/9780470147306.ch1](https://doi.org/10.1002/9780470147306.ch1). CAS: reactions from publications and patents dated 2010–2015 sourced from the CAS Content Collection™ (CAS is a division of the American Chemical Society); see [CAS Reactions](https://www.cas.org/cas-data/cas-reactions).

The dataset contains 967 reactions curated in this work, 734 from the literature and 233 from CAS reaction records, with 119 numerical descriptors (F119) computed in this work.

- `wittig_literature_f119.csv`: Literature records with source identifiers, reported Z selectivity, reaction conditions, component and reaction SMILES, and F119 descriptors.
- `wittig_cas_f119.csv`: CAS-derived records with recorded Z selectivity, stage conditions and solvents, component, oxaphosphetane and reaction SMILES, and F119 descriptors.
- `schema.json`: Column definitions for both tables.
- `feature_config.py`: The ordered F119 feature definitions.

See `Wittig/README.md` for details of the identifiers and molecular representation.
