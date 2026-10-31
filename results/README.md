# Results

This directory contains the evaluation results, summary tables, and additional analyses reported in the manuscript and supporting information.

The root `README.md` provides the main reproduction commands. The subdirectory READMEs provide further details about individual analyses.

## Directory contents

| Path | Contents |
|---|---|
| `main_results_table.csv` | Main results summary for the primary acquisition window, including highest-mean and statistical competitiveness results across all 118 tasks. |
| `benchmark_inputs/` | Learning curves, native integer-step method means, AUC values from individual runs, and task information used in the evaluation, including all 20 Rh random-target splits and the split membership of the 29 Wittig tasks. |
| `benchmark/` | Task classifications and summary tables for all seven acquisition windows. |
| `efficiency/` | Experimental efficiency (reactions needed to reach the efficiency target) and savings relative to Random acquisition, including bootstrap confidence intervals and a crossing-rule sensitivity analysis. |
| `sensitivity/` | Analyses examining the choice of acquisition window, Rh partition aggregation, and performance thresholds. See `sensitivity/README.md`. |
| `rh_published_split/` | Experimental design, partition information, and acquisition trajectories for the seven Rh Published Split tasks. See `rh_published_split/README.md`. |
| `wittig_transfer/` | Results for the Wittig transfer task (Published Split family). See `wittig_transfer/README.md`. |
| `wittig_qualitative_study/` | The qualitative Wittig study of Fig 3E, separate from the 118 tasks: split membership, the 225 acquisitions with their reaction chemistry, and the illustrated reactions. See `wittig_qualitative_study/README.md`. |

`benchmark_inputs/` contains the saved results and information used in the analyses, while `benchmark/` contains the resulting classifications and summary tables. The separate Rh and Wittig directories contain analyses specific to those evaluation settings.

## Wittig split membership

`benchmark_inputs/wittig_split_membership.csv` gives the exact reaction sets of the 29 Wittig tasks, by the `record_id` of `data/Wittig/` (columns `split_id`, `seed`, `role`, `record_id`). Only the rows that are not implied are listed:

- Random and Cluster tasks: the `target` rows, with `seed` empty because the target set is the same in all ten repetitions, and the 25 `initial` (warm-start) rows of each repetition (`seed` 0–9). The full non-target pool includes the initial set, but the 25 initially labeled reactions are excluded from subsequent acquisition candidates.
- Published Split: the 35 `target` rows of each repetition (`seed` 0–9). The other 198 CAS reactions form that repetition's pool, and the 734 literature reactions are the fixed training base.

Task `wittig_f119_cluster_k8_clN` targets cluster N of a Ward clustering (`sklearn.cluster.AgglomerativeClustering(n_clusters=8)`) of the standardized F119 descriptors of the 967 reactions, in released order.

## Treatment of repetitions

Each of the 118 active learning tasks contributes ten paired repetitions to the statistical analyses.

The seven Rh Published Split tasks each contain 30 trajectories: three fixed partitions (seeds 42, 43, and 44) crossed with ten matched repetition seeds. For statistical testing, results are averaged across the three partitions for each repetition seed. This yields ten paired values per task rather than treating all 30 trajectories as independent observations.

The other 111 tasks have ten matched repetitions and are analyzed directly.

Averaging the three partitions within each repetition seed preserves the overall mean, so the highest-mean classifications are identical under the primary and pooled analyses. The paired statistical tests can differ. Results for the pooled analysis and each individual partition are provided in `sensitivity/rh_partition/`.

## Reproducing released analyses

The following commands reproduce or verify the released analyses using saved results rather than rerunning the active learning campaigns. Run them from the repository root.

| Command | Analysis or output |
|---|---|
| `python scripts/verify_results.py` | Recomputes task classifications and summary tables across all seven acquisition windows. |
| `python scripts/efficiency_endpoint.py` | Recomputes the per-task experimental-efficiency table, its dataset and split-family summaries, the target-level counts, and the crossing-rule sensitivity from the released native integer-step method-mean curves. |
| `python scripts/efficiency_pool_spared.py` | Recomputes acquisition pool savings relative to Random, with 5,000-draw intervals (`efficiency/pool_spared_vs_random_summary.csv`). |
| `python scripts/efficiency_reactions_saved.py` | Recomputes mean reactions saved relative to Random, with 5,000-draw intervals (`efficiency/reactions_saved_vs_random_summary.csv`). |
| `python scripts/efficiency_savings_intervals.py` | Recomputes the published 100,000-draw confidence intervals and examines the effect of the bootstrap method. |
| `python scripts/build_si_learning_curves.py` | Rebuilds the SI learning curves for all 118 tasks. |
| `python scripts/build_si_efficiency.py` | Rebuilds the SI experimental efficiency table. |
| `python scripts/build_si_pairwise_tests.py` | Rebuilds the SI pairwise statistical test table. |
| `python scripts/build_benchmark_tables.py` | Rebuilds the tables derived from the per-task classification (`benchmark/`, `sensitivity/rh_partition/`, `efficiency/negative_ceiling_margin.csv`, `main_results_table.csv`, and the SI competitiveness tables). `--check` compares a fresh rebuild with the committed files. |
| `python scripts/make_manuscript_figures.py` | Redraws Fig 3D and Fig 4C at manuscript size and style into `figures/manuscript/`. |
| `python scripts/wittig_qualitative_study.py` | Reruns the qualitative Wittig study (split and 225 acquisitions) from `data/Wittig/` and compares it with `wittig_qualitative_study/`. Requires `xgboost`. |
| `python scripts/wittig_umap_embedding.py` | Regenerates the Wittig UMAP coordinates (Fig 3C and Fig 3E) from `data/Wittig/` and compares them with `wittig_qualitative_study/split_membership.csv`. Requires `umap-learn` (versions in `wittig_qualitative_study/README.md`). |

`verify_results.py` checks the released results without modifying files. The other scripts reproduce their corresponding analyses or generate the specified outputs. Run `efficiency_endpoint.py` before the savings scripts, which read its per-task table, and `build_benchmark_tables.py` after `efficiency_savings_intervals.py`, whose intervals the main results table reads.

### Experimental efficiency endpoint

For each task, R²₀ is the shared initial method-mean R² on the target set. With
B = ⌈0.30 N_pool⌉, R²max,30 is the highest method-level mean target-set R²
observed for any of the five methods at an integer acquisition step from 0
through B, inclusive. The target is R²₀ + 0.8 (R²max,30 − R²₀).

A method's endpoint is the **first native integer acquisition step** at which
its method-level mean curve reaches or exceeds the target. The curve is not
interpolated, smoothed, floored at R²₀, or made monotone. A method that does not
reach the target within the budget is assigned B reactions; the separate
`attained` flag distinguishes that case from reaching the target at exactly B.
Ordinary task means give equal weight to ten repetitions. For the seven Rh
Published-Split tasks, the three fixed partitions are first averaged within
each repetition and the ten repetition means are then averaged.

The independently reproducible endpoint input is
`benchmark_inputs/native_method_mean_curves_0_30.csv.gz`. Its metadata and
SHA-256 checksum are in the adjacent `.metadata.json` and `.sha256` files. The
artifact contains task and method identifiers, every integer step through B,
the method mean, pool size, budget, and weighting metadata; it contains neither
private paths nor reaction-level records. The loader validates the checksum,
schema, complete integer-step coverage, weighting protocol, and agreement with
the separately retained 1%-grid curves. Those grid curves remain the inputs to
the AUC and statistical-competitiveness analyses and are unchanged.

From the repository root, verify the packaged bytes and reproduce the endpoint
with:

```bash
(cd results/benchmark_inputs && sha256sum --check native_method_mean_curves_0_30.sha256)
python scripts/efficiency_endpoint.py
```

As a sensitivity analysis, `efficiency/crossing_rule_sensitivity.csv` repeats
the attainment counts, mean reactions, and savings relative to Random with a
two-consecutive rule. At native resolution this requires the target to remain
met for one additional acquired reaction; it is not the former test at adjacent
1%-of-pool checkpoints. `sensitivity/target_level_counts.csv` gives the number
of tasks reaching the target for q = 0.70–0.95 under the primary rule. The
estimators are implemented in `scripts/analysis/crossing.py`.

The native artifact was validated against 6,600 campaign trajectories. Of
their historical recorded byte hashes, 6,290 match the available normalized
source files and 310 do not. The affected files are limited to ten BH Random
Split tasks and four SAD tasks; the cause of those historical byte-hash
differences is unresolved. Reconstructed method means nevertheless reproduce
every approved 1%-grid value (maximum absolute difference
4.27 × 10⁻¹³), and nested and pooled Rh method means agree to numerical
precision. These facts, rather than a claim that all 6,600 historical hashes
were verified, are recorded in the artifact metadata.

### Bootstrap confidence intervals

The published intervals in `main_results_table.csv` and SI Table S8 are the
100,000-draw cluster intervals in
`efficiency/savings_intervals_100k_primary.csv`. Alternative task and
hierarchical resampling results are in
`efficiency/savings_intervals_100k_by_structure.csv` and summarized in
`efficiency/savings_structure_dependence.csv`. BALD's reaction-savings interval
includes zero under every reported resampling structure. Indiv. NN's interval
excludes zero under the primary cluster bootstrap but includes zero under the
task and hierarchical alternatives. EPIG's savings remain positive under all
reported structures. The
5,000-draw tables use the same procedure and seeds; their means match the
published values, but their intervals are not the published intervals.
