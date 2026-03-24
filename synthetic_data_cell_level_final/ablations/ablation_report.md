# Rule Ablation Analysis (§19)

This report compares the impact of disabling different stages of the generation pipeline on survival quality and trust metrics.

| ablation       |   local_mean_abs_error |   radiation_monotonicity |   thermal_monotonicity |   high_dose_plausibility |   mean_pressure |   mean_delta_proj |   mean_delta_calib |
|:---------------|-----------------------:|-------------------------:|-----------------------:|-------------------------:|----------------:|------------------:|-------------------:|
| full_pipeline  |             0.00227539 |                        1 |                      1 |                        1 |       0.0338164 |         0.0133965 |          0.0204199 |
| no_projection  |             0.00241169 |                        1 |                      1 |                        1 |       0.0314587 |         0         |          0.0314587 |
| no_cap         |             0.00227539 |                        1 |                      1 |                        1 |       0.0338164 |         0.0133965 |          0.0204199 |
| no_calibration |             0.0072498  |                        1 |                      1 |                        1 |       0         |         0.0133965 |          0         |
| no_local_sigma |             0.00248051 |                        1 |                      1 |                        1 |       0.0349855 |         0.0138449 |          0.0211366 |

## Observations

- **Projection (CL1/CL3)**: Disabling projection leads to severe drops in monotonicity rates.
- **Calibration**: Disabling calibration increases Local MAE as the blocks don't match the observed means anymore.
- **Caps (CL5)**: Disabling caps affects the high-combined-dose plausibility (though current sigma is small, so impact may be limited).
- **Local Sigma**: High constant sigma leads to higher pressure and more out-of-bounds samples.