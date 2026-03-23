# Final Cell-Level Article-Guided Synthetic Dataset

This dataset is the primary synthetic dataset for the 15-point cell survival experiment.

## Design Choice

- exact experimental support is preserved: synthetic rows stay on the same 15 design points as the observed experiment;
- variation is generated only in survival, not by inventing unsupported treatment conditions;
- each generated 5x3 survival block is projected onto a monotone surface across radiation and thermal intensity;
- the generator is calibrated back to the observed design-point matrix after projection, which reduces local bias while preserving the cell-level rules;
- only cell-level rules from the literature are active in the default pipeline; in vivo and clinical rules remain in the broader knowledge base but are not used here.

## Statistical Metrics

- `mean_wasserstein_normalized`: 0.0050
- `mean_ks_statistic`: 0.0203
- `pearson_correlation_mean_abs_diff`: 0.0015
- `spearman_correlation_mean_abs_diff`: 0.0090
- `tstr_mae`: 0.0022
- `tstr_r2`: 0.9991
- `support_violation_rate_mean`: 0.0000
- `duplicate_rate_vs_real`: 0.1230
- `separability_auc_mean`: 0.3798
- `separability_gini_abs_mean`: 0.2893

## Independent Cell-Level Validation

- `exact_design_support_rate`: 1.0000
- `local_mean_abs_error`: 0.0023
- `local_max_abs_error`: 0.0122
- `radiation_monotonicity_mean_rate`: 1.0000
- `thermal_monotonicity_mean_rate`: 1.0000
- `high_combined_dose_low_survival_rate`: 1.0000
- `independent_article_compliance_mean`: 1.0000

## Explainability Artifacts

- `knowledge_base/cell_level_rule_traceability.csv`
- `knowledge_base/cell_level_rule_traceability.md`
- `synthetic_data_cell_level_final/design_point_rule_explanations.csv`
- `synthetic_data_cell_level_final/design_point_rule_explanations.md`

## Files

- `final_synthetic_dataset.csv`
- `final_synthetic_dataset_full.csv`
- `real_design_points.csv`
- `generation_metadata.json`
- `evaluation_metrics.json`
- `independent_cell_level_validation.json`

The dataset is suitable for downstream ML/AI experiments that should remain faithful to the original experimental design and to cell-level rules extracted from the literature.
