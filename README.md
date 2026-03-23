# Thermoradiobiology Synthetic Data Project

This repository contains a literature-grounded expert knowledge base and a cleaned final pipeline for generating a **cell-level synthetic survival dataset** for the 15-point experiment on combined radiation and hyperthermia.

## Active Project Core

The files below are the active, recommended workflow:

- `knowledge_base/cell_level_rules.json` - machine-readable cell-level rules used by the final generator and validation layer
- `knowledge_base/cell_level_rules_ru.md` - human-readable Russian description of the active cell-level rules
- `knowledge_base/cell_level_rule_traceability.csv` - traceability matrix `rule -> evidence -> generator -> validation`
- `knowledge_base/cell_level_rule_traceability.md` - readable traceability report
- `scripts/common_synthetic_metrics.py` - shared statistical quality metrics for synthetic tabular data
- `scripts/generate_cell_level_article_guided_dataset.py` - design-preserving monotone generator for the 15 observed design points
- `scripts/build_cell_level_final_pipeline.py` - end-to-end production pipeline for the final dataset and explainability artifacts
- `scripts/analyze_cell_level_final_dataset.py` - uncertainty, bootstrap, and multi-seed robustness analysis for the final dataset
- `scripts/validate_cell_level_datasets.py` - independent audit using only observable columns and exact design support
- `synthetic_data_cell_level_final/final_synthetic_dataset.csv` - primary final synthetic dataset
- `synthetic_data_cell_level_final/final_dataset_report.md` - report for the active final dataset
- `synthetic_data_cell_level_final/final_analysis_report.md` - final interpretation and robustness report
- `synthetic_data_cell_level_final/design_point_rule_explanations.csv` - explanation per observed design point
- `synthetic_data_cell_level_final/design_point_rule_explanations.md` - readable design-point explanation report
- `comparison/cell_level_end_to_end_audit.md` - benchmark comparison focused on exact design support and independent rule compliance
- `comparison/implementation_review.md` - review of the whole implementation and why the final pipeline was selected

## Repository Layout

- `literature/` - literature slice and evidence table extracted from source papers
- `knowledge_base/` - expert rules, merged knowledge base, ontology, and traceability artifacts
- `scripts/` - active generation, validation, and metric code
- `synthetic_data_cell_level_final/` - active final outputs
- `comparison/` - active audit/review documents
- `benchmarks/` - minimal archive note; raw old benchmark outputs were removed during cleanup
- `ml/` - auxiliary training examples for ML/AI

## Recommended Workflow

Build the final dataset:

```bash
python scripts/build_cell_level_final_pipeline.py
```

Re-run the independent audit:

```bash
python scripts/validate_cell_level_datasets.py
```

Run the uncertainty and robustness analysis:

```bash
python scripts/analyze_cell_level_final_dataset.py
```

## Why This Pipeline Is The Default

The final pipeline was chosen because it is the most defensible for the very small `n=15` dataset:

- it preserves the **exact experimental support** instead of inventing unsupported treatment conditions;
- it uses only **cell-level rules** that are appropriate for the observed survival matrix;
- it enforces monotone structure across radiation and thermal intensity;
- it calibrates synthetic survival back to the observed design-point matrix after projection;
- it produces explicit explainability artifacts that connect each rule to the source literature and to the final dataset.

## Current Final Dataset Snapshot

The active final dataset in `synthetic_data_cell_level_final/` currently achieves:

- `mean_wasserstein_normalized = 0.0050`
- `mean_ks_statistic = 0.0203`
- `tstr_mae = 0.0023`
- `tstr_r2 = 0.9991`
- `support_violation_rate_mean = 0.0000`
- `exact_design_support_rate = 1.0000`
- `local_mean_abs_error = 0.0023`
- `radiation_monotonicity_mean_rate = 1.0000`
- `thermal_monotonicity_mean_rate = 1.0000`

## Rule Philosophy

The project now separates three layers clearly:

- `cell-level rules` - active in the final generator
- `tumor / in vivo rules` - kept in the broader knowledge base, but not used in the default cell-level generator
- `clinical rules` - preserved for the expert knowledge base, but excluded from synthetic generation for the 15-point cell dataset

This avoids mixing patient-level logic with a simple in vitro survival matrix.

## Historical Benchmarks

Older CVAE and continuous-generator code is still available in `scripts/legacy/`, but raw historical outputs were removed during cleanup because they no longer improve the active project workflow.

## Caution

This is a research repository, not a validated clinical treatment protocol. Human expert review is still required before biomedical interpretation or downstream publication claims.
