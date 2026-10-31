# Wittig transfer task (Published Split family)

The Wittig literature-to-CAS transfer task is the Wittig member of the
Published Split family. Its task ID is
`wittig_f119_textbook_base_cas_random_85_15`.

## Task definition

All **734** Vedejs and Peterson literature reactions are fixed background
training data. Within each of ten paired repetitions the **233** CAS-derived
reactions are divided into a **198**-reaction acquisition pool and a
**35**-reaction target set. The task therefore measures how well an
acquisition function transfers from the Vedejs–Peterson literature set to a
distinct CAS-curated reaction domain.

This is the one Published Split task the Wittig dataset contributes to the
118-task evaluation; the other 28 Wittig tasks are cluster and random splits.

| File | Contents |
|---|---|
| `per_seed_window_scores.csv` | One row per seed × method × window (350 rows = 10 seeds × 5 methods × 7 windows), with the windowed AUC in the `auc` column. |
| `mean_curves_0_30.csv` | Mean R² on the target set across the ten repetitions on the 31-point 0–30 % acquisition grid, per method. |
| `transfer_curve.csv` | The higher-resolution 61-point mean transfer curve, per method, with standard errors. Source of Fig 3D. |

## Reaction data

The 734 literature and 233 CAS reactions are in `../../data/Wittig/`. These files
contain AUC values from individual repetitions and learning curves averaged
across repetitions.
