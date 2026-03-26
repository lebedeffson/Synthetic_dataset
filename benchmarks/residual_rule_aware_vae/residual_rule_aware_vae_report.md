# Residual Rule-Aware Block VAE

This experimental generator models residual stochasticity around the rule-guided matrix prior instead of generating the full block from scratch.

## Core Idea

- prior mean surface comes from the rule-guided matrix generator;
- the VAE models only residual variation around that surface;
- rule penalties are applied during training on raw outputs;
- projection/cap/calibration remain as a safety layer after sampling.

## Final Metrics

- `mean_wasserstein_normalized`: 0.0043
- `mean_ks_statistic`: 0.0198
- `tstr_mae`: 0.0008
- `tstr_r2`: 0.9999
- `local_mean_abs_error`: 0.0008
- `local_max_abs_error`: 0.0041
- `independent_article_compliance_mean`: 1.0000

## Explainability

- `mean_constraint_pressure`: 0.0025
- `mean_raw_radiation_violation`: 0.000016
- `mean_raw_thermal_violation`: 0.000000
- `mean_residual_budget_utilization`: 0.2874

## Interpretation

- This model is mathematically more honest than a free-form neural generator for `n=15` real points because it keeps the rule-guided surface as prior structure.
- The main question is not only fidelity, but how much safety-layer correction remains necessary after raw generation.
