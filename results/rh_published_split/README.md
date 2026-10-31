# Rh Published Split Tasks

This directory contains results for seven Rh tasks (`rh_paper_ood_case1`–`rh_paper_ood_case7`) constructed following the split protocol described in the source publication. The original split assignments were unavailable, so the tasks use reconstructed partitions.

## Files

| File | Contents |
|---|---|
| `task_design.csv` | Reference table of the split protocol for each task: acquisition pool and target sizes, initial training size, the three partition seeds, and the reconstruction qualification. |
| `partition_map.csv` | Maps each of the 30 runs per task to its partition and repetition seed. Read by the analysis layer to average the three partitions within each repetition seed. |
| `trajectories/acquisition_trajectories.csv.gz` | Acquisition trajectories for 1,050 runs (7 tasks × 5 methods × 30 runs), including performance metrics and selected reactions. |

## Source data

Kalikadien et al., *Chem. Sci.* **2024**, [10.1039/D4SC03647F](https://doi.org/10.1039/D4SC03647F).

The reaction dataset is available from [4TU.ResearchData](https://doi.org/10.4121/ecbd4b91-c434-4bdf-a0ed-4e9e0fb05e94.v1) under CC BY 4.0. The authors' modeling code is available at [EPiCs-group/obelix-ml-pipeline](https://github.com/EPiCs-group/obelix-ml-pipeline).

The source workbook is cited rather than redistributed here. See `data/README.md` for details.

## Representation and protocol

The 960 reactions are represented using ligand descriptors and a one-hot encoding of the substrate, giving 41 features (192 reactions per substrate).

The random forest acquisition hyperparameters used in this study were selected from:

- `n_estimators`: 50, 100, 200, 300
- `max_depth`: 3, 5, 10, None
- `max_features`: None, "sqrt", "log2"

Each task holds out one olefin. Its 192 reactions are divided equally into a 96-reaction acquisition pool and a fixed 96-reaction target set using `train_test_split` with `test_size=0.5`.

Three partitions were constructed using seeds 42, 43, and 44. Each partition was evaluated with ten matched repetition seeds, giving 30 trajectories per task and acquisition method.

Each row of the trajectory archive carries both the acquisition identifier used by the code (`acq_fn`, for example `epig_gauss`) and the corresponding manuscript label (`af_label`, for example EPIG), so no separate mapping is needed.

## Joining the trajectories to the task records

The trajectory archive stores one row per acquisition round and records each partition as a separate run. Its identifiers therefore differ from those used in the task records.

**Task identifiers.** Each task has three partitions. For example, `rh_paper_ood_case1` corresponds to `rh_literature_partial_ood_case1_split0`, `rh_literature_partial_ood_case1_split1`, and `rh_literature_partial_ood_case1_split2` in the archive. The `case_id` column holds the case number alone (1–7), so the corresponding task identifier is `rh_paper_ood_case<case_id>`. The `partition_id` column identifies the partition (0, 1, or 2).

**Repetition identifiers.** The archive uses `seed_id` for the repetition seed (0–9). This corresponds to `repetition_seed` in `partition_map.csv`. The `seed_id` column in `partition_map.csv` instead identifies one of the 30 runs (0–29).

To join the files, match:

- The task identifier `rh_paper_ood_case<case_id>`, derived from the archive's `case_id`, to partition map `split_id`.
- Archive `partition_id` to partition map `partition_id`.
- Archive `seed_id` to partition map `repetition_seed`.

Do not join the two `seed_id` columns directly, and do not match `case_id` against `split_id` without deriving the task identifier first, as these columns hold different values.

The archive also records `split_family`, which is shared with the corresponding task records.

## Trajectory archive

`trajectories/acquisition_trajectories.csv.gz` contains the acquisition trajectories, including task and partition identifiers, repetition seeds, acquisition methods, acquisition steps, acquisition fractions, target set performance, model and representation labels, and selected reactions.

The selected reaction fields identify the row acquired from the Rh source dataset at each step and its acquisition score. The source dataset is publicly available from the 4TU.ResearchData record linked above.

## Limitation

The original split row assignments were not available in the source publication's released repository, notebook, result workbook, or result pickle.

Seeds 42, 43, and 44 define new partitions, with seed 42 matching the default of the public split function. These tasks therefore reproduce the published split protocol, but not the exact original row assignments.

The three partitions are retained separately in the released trajectories and examined in a sensitivity analysis (`../sensitivity/rh_partition/`). For the primary statistical analyses, their values are averaged within each repetition seed.
