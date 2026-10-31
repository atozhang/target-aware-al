# Figures

Every figure here is drawn from a committed result table. Redraw them all with:

```bash
python scripts/make_figures.py
```

| Figure | What it shows | Source table |
|---|---|---|
| `wittig_transfer_curve.{pdf,png}` — Fig 3D | Mean R² on the target set against acquired fraction of the CAS pool for five methods in the Wittig literature-to-CAS transfer task. Shaded bands are ± 1 SEM over the ten paired repetitions | `results/wittig_transfer/transfer_curve.csv` |
| `bh_representative_curves.{pdf,png}` — Fig 4B | Mean R² on the target set against acquired pool percentage for one representative Buchwald–Hartwig task from each split family | `supporting_information/data/all_benchmark_learning_curves.csv` |
| `statistical_competitiveness_by_dataset_and_split.{pdf,png}` | Percentage of tasks per method that are statistically competitive (not detected as worse than the highest-mean method) at the primary window, by reaction dataset and split family. Multiple methods can be competitive on the same task, so bars within a group do not sum to 100 %. Highest-mean counts are in `supporting_information/tables/competitiveness/highest_mean_by_*.csv` | `results/benchmark/per_task_classification.csv`, at `window == "2-12"` |
| `learnability_random_splits.{pdf,png}` | The 60 Random Split tasks: task ceiling (best method mean AUC₂–₁₂) against the EPIG − BALD margin, marked by dataset | `results/benchmark_inputs/per_seed_window_auc.csv` (50 tasks) + `rh_random_split_per_seed_window_auc.csv` (10 tasks) |
| `efficiency_threshold_sensitivity.{pdf,png}` | Tasks reaching a fraction *q* of their achievable improvement in R² on the target set within the 30 % budget, using the first native integer-step crossing of the method-mean curve, for *q* = 0.70…0.95 | `results/sensitivity/target_level_counts.csv` |
| `learning_curves/` | The mean learning curve for all 118 tasks, nine pages grouped by dataset | `supporting_information/data/all_benchmark_learning_curves.csv` |

`python scripts/make_manuscript_figures.py` redraws Fig 3D and Fig 4C at manuscript
size and style into `manuscript/` (Arial when installed, otherwise the bundled TeX
Gyre Heros):

| Figure | What it shows | Source table |
|---|---|---|
| `manuscript/fig3d_wittig_published_split.{pdf,png}` — Fig 3D | The unchanged Wittig transfer-task learning curves (Published Split family) with the 2–12 % window shaded and a dotted red line at the native-step efficiency target, 80 % of the achievable improvement in R² (“80% of achievable improvement”) | `results/wittig_transfer/transfer_curve.csv`, `results/efficiency/efficiency_per_task.csv` (endpoint input: `results/benchmark_inputs/native_method_mean_curves_0_30.csv.gz`) |
| `manuscript/fig4c_benchmark_performance.{pdf,png}` — Fig 4C | Highest-mean (solid) and statistically competitive (pale) shares of tasks at AUC₂–₁₂, by split family and by dataset | `supporting_information/tables/competitiveness/{highest_mean,competitive}_by_{split_family,dataset}.csv` |

The nine SI learning-curve pages come from
`python scripts/build_si_learning_curves.py`, which `make_figures.py` invokes;
run it directly to rebuild only those pages.

## Reproduction scope

These files are programmatic visualizations of the released data underlying the
manuscript figures. Some manuscript figures are hand-composed multi-panel
graphics, so the standalone repository renderings may differ in layout, labels,
or layering while preserving the underlying numerical quantities.

`make_figures.py` registers the TeX Gyre Heros faces bundled in `fonts/`, so
its output is font-deterministic on any machine rather than depending on
locally installed typefaces.

## Figures that are not here

- **Fig 1, Fig 2 and Fig 4A** are schematics, drawn by hand rather than
  generated from data.
- **Fig 3A and Fig 3B** are hand-drawn reaction schemes.
- **Fig 3C and Fig 3E** are UMAP maps composed with reaction structures. The F119
  UMAP coordinates and the Fig 3E reactions are in `../results/wittig_qualitative_study/`.
- **Fig 3 and Fig 4** as printed are hand-composed multi-panel composites; the
  individual panels that can be released are the files listed above.

## Typeface

`fonts/` contains the four TeX Gyre Heros faces used by the figures, under the
GUST Font License (`fonts/GUST-FONT-LICENSE.txt`).
