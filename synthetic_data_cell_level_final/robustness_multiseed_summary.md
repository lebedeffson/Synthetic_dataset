# Multi-Seed Robustness Summary

Seeds analyzed: `7, 11, 17, 23, 42, 84, 126, 256`

## Key Findings

- `exact_design_support_rate` stayed at `1.0000` to `1.0000` across all runs.
- `radiation_monotonicity_mean_rate` stayed at `1.0000` to `1.0000`.
- `thermal_monotonicity_mean_rate` stayed at `1.0000` to `1.0000`.
- `local_mean_abs_error` ranged from `0.0020` to `0.0024`.
- `mean_wasserstein_normalized` ranged from `0.0042` to `0.0053`.
- Worst design-point max error across seeds: `0 Gy, 42 C, 45 min -> 0.0132`.

## Outputs

- `robustness_multiseed_metrics.csv`
- `robustness_multiseed_summary.csv`
- `robustness_multiseed_design_points.csv`
- `robustness_multiseed_summary.json`
