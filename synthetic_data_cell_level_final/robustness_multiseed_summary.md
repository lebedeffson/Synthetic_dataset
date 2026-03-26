# Multi-Seed Robustness Summary

Seeds analyzed: `7, 11, 17, 23, 42, 84, 126, 256`

## Key Findings

- `exact_design_support_rate` stayed at `1.0000` to `1.0000` across all runs.
- `radiation_monotonicity_mean_rate` stayed at `1.0000` to `1.0000`.
- `thermal_monotonicity_mean_rate` stayed at `1.0000` to `1.0000`.
- `local_mean_abs_error` ranged from `0.0013` to `0.0018`.
- `mean_wasserstein_normalized` ranged from `0.0040` to `0.0050`.
- `x_mean_pressure` (Explainability) ranged from `0.0348` to `0.0348`.
- `x_pass_rate` (Enforcement) was always `1.0000` across seeds.
- Worst design-point max error across seeds: `2 Gy, 42 C, 45 min -> 0.0121`.

## Outputs

- `robustness_multiseed_metrics.csv`
- `robustness_multiseed_summary.csv`
- `robustness_multiseed_design_points.csv`
- `robustness_multiseed_summary.json`
