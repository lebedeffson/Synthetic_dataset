# Cell-Level End-to-End Audit

| Dataset | Exact Design Support | Local Mean Abs Error | Radiation Mono | Thermal Mono | Independent Article Compliance |
|---|---:|---:|---:|---:|---:|
| notebook_cvae | 0.0000 | NA | NA | NA | 0.0000 |
| rule_guided_cvae | 0.0000 | NA | NA | NA | 0.0000 |
| continuous_kernel_final | 0.0390 | 0.0109 | NA | NA | 0.5195 |
| cell_level_article_guided_final | 1.0000 | 0.0023 | 1.0000 | 1.0000 | 1.0000 |

## Interpretation

- `Exact Design Support` equals the fraction of synthetic rows that stay on the 15 observed experimental conditions.
- `Local Mean Abs Error` compares synthetic mean survival against the observed value at each exact design point.
- `Radiation Mono` checks non-increasing mean survival with increasing radiation at fixed thermal condition.
- `Thermal Mono` checks non-increasing mean survival with stronger thermal condition at fixed radiation.
- `Independent Article Compliance` averages only checks based on observable columns or exact design support.
