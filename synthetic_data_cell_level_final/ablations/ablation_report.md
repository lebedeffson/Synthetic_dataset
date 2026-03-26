# Rule Ablation Analysis (§19)

This report compares the impact of disabling different stages of the generation pipeline on survival quality and trust metrics.

| ablation | local_mean_abs_error | radiation_monotonicity | thermal_monotonicity | high_dose_plausibility | mean_pressure | mean_delta_proj | mean_delta_calib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_pipeline | 0.001598 | 1.000000 | 1.000000 | 1.000000 | 0.034788 | 0.003326 | 0.031462 |
| no_projection | 0.001608 | 1.000000 | 1.000000 | 1.000000 | 0.032588 | 0.000000 | 0.032588 |
| no_cap | 0.001598 | 1.000000 | 1.000000 | 1.000000 | 0.034788 | 0.003326 | 0.031462 |
| no_calibration | 0.005979 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.003326 | 0.000000 |
| no_local_sigma | 0.001598 | 1.000000 | 1.000000 | 1.000000 | 0.034788 | 0.003326 | 0.031462 |

## Observations

- **Projection (CL1/CL3)**: в этом датасете final monotonicity не просела, но pipeline потерял явную projection-stage коррекцию; Local MAE изменился на `+0.0000`, pressure на `-0.0022`.
- **Calibration**: отключение calibration меняет Local MAE на `+0.0044` и pressure на `-0.0348`; этот шаг нужен для возврата synthetic mean к наблюдаемой матрице.
- **Caps (CL5)**: high-dose plausibility меняется на `+0.0000`; Local MAE меняется на `+0.0000`.
- **Local sigma**: без локальной оценки шума меняются fidelity и burden (Local MAE `+0.0000`, pressure `+0.0000`).