# Final Analysis Report

## What The Final Dataset Now Demonstrates

- The dataset stays exactly on the original 15 observed design points.
- The synthetic survival surface remains monotone in radiation and in observed thermal ordering.
- The generator is calibrated back to the observed matrix, so local deviation from the real design is small.
- Explainability is explicit: each active rule is linked to evidence, generator use, validation use, and design-point behavior.

## Main Quality Metrics

- `mean_wasserstein_normalized = 0.0045`
- `mean_ks_statistic = 0.0190`
- `tstr_mae = 0.0017`
- `tstr_r2 = 0.9995`
- `support_violation_rate_mean = 0.0000`
- `duplicate_rate_vs_real = 0.1280`
- `local_mean_abs_error = 0.0016`
- `local_max_abs_error = 0.0092`

## Uncertainty And Stability

- Bootstrap CI coverage for real survival values across design points: `0.8667`.
- Widest design-point CI width: `0.0287`.
- Highest CI upper bound occurred at `0 Gy, 42 C, 45 min`.
- Multi-seed runs analyzed: `8`.
- Worst design-point max error across seeds: `2 Gy, 42 C, 45 min -> 0.0121`.

CI misses:

- 0 Gy, 42 C, 45 min (real=0.53, CI=[0.518579, 0.52695])
- 8 Gy, 44 C, 30 min (real=0, CI=[5.72921e-09, 2.12041e-08])

## Interpretation

- For this project, the final dataset is best described as design-preserving synthetic augmentation rather than a free-form simulator.
- The strongest claims are about fidelity to the observed matrix and agreement with curated cell-level rules.
- The two remaining sensitive points are the maximum observed survival point and the zero-survival boundary point; this is a typical consequence of generating a smooth stochastic cloud around hard biological bounds.
- The weakest claim remains mechanistic generalization outside the observed domain, so the dataset should not be used to infer arbitrary unseen treatment regimes.
- Exact-boundary fidelity and lower duplicate rate are in tension here: preserving a small amount of variability near `0.53` and `0.0` keeps the dataset less degenerate, but it slightly weakens bootstrap CI coverage at those boundaries.

## Files To Cite In The Project

- `final_dataset_report.md`
- `final_analysis_report.md`
- `independent_cell_level_validation.json`
- `design_point_bootstrap_ci.csv`
- `robustness_multiseed_summary.md`
- `knowledge_base/cell_level_rule_traceability.md`
- `design_point_rule_explanations.md`
