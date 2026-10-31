# Wittig F119 data

This directory contains the 967 Wittig olefination reactions of the study with
their Z selectivity and F119 descriptors: 734 Vedejs and Peterson literature
records and 233 records curated from CAS reaction records. The two tables share
one descriptor block (same names and order); concatenated, literature rows first,
they form the 967-reaction dataset used in the evaluation. `schema.json` defines
every column.

- `wittig_literature_f119.csv` — the 734 Vedejs–Peterson literature records.
  `record_id` is a stable identifier for the released table; `source_table` and
  `source_entry` identify the corresponding entry in Vedejs and Peterson (1994),
  and `condition_index` distinguishes
  multiple experimental conditions reported for the same source entry. The
  table includes reported Z selectivity, temperature, solvent, component SMILES,
  the released unmapped reaction representation, and the 119 F119 descriptors
  used in the evaluation. The descriptor values were computed in this work.
  - `z_selectivity_percent` — the reported Z selectivity, in percent.
  - `solvent_reported` — preserves the transcribed solvent notation;
    `solvent_standardized` is the standardized solvent label.
  - `NA` in `base_smiles` indicates that no separate base is specified in the
    curated reaction record.
  - `rxn_smiles` — the unmapped representation released here: carbonyl component,
    ylide, base and solvent as reactants and the cis- and trans-oxaphosphetane
    intermediates as products.
  - The ordered F119 names are given by `feature_config.py`; the descriptor
    definitions and classes are described in the Supporting Information of the
    accompanying paper.
- `wittig_cas_f119.csv` — the 233 CAS-derived records. `record_id` (prefixes
  `CAS_R` and `X48_`) is a stable identifier assigned in this work; it is not a
  CAS Registry Number or CAS reaction identifier.
  - `z_selectivity_percent` — computed from the amounts of the Z and E alkene
    products recorded in the CAS reaction record, with the geometry assigned at
    the newly formed C=C bond.
  - `rxn_smiles` — the atom-mapped reaction SMILES of the CAS record. It shows one
    recorded alkene product, not necessarily the major isomer; the E/Z ratio is
    `z_selectivity_percent`.
  - `aldehyde_smiles`, `ylide_smiles`, `cis_opa_smiles`, `trans_opa_smiles` — the
    carbonyl component, ylide and cis- and trans-oxaphosphetanes of the reaction.
  - The 119 F119 columns are the precomputed numerical features supplied for
    the benchmark.
  - The temperature, stage, solvent and base columns transcribe the CAS record and
    are not model features.
- `schema.json` — column definitions, units and missing-value conventions for both
  tables.
- `feature_config.py` — the ordered F119 feature schema used by the released
  analysis.

## Sources

- Literature records: E. Vedejs, M. J. Peterson, "Stereochemistry and Mechanism
  in the Wittig Reaction", *Topics in Stereochemistry*, Vol. 21, Wiley, 1994
  ([10.1002/9780470147306.ch1](https://doi.org/10.1002/9780470147306.ch1)).
- CAS records: 233 reactions from publications and patents dated 2010–2015
  sourced from the CAS Content Collection™, a comprehensive human-curated
  scientific data repository (CAS is a division of the American Chemical
  Society). The records were curated in this work and are redistributed with
  the permission of CAS (`../LICENSES.md`); see also
  [CAS Reactions](https://www.cas.org/cas-data/cas-reactions).
