# Comparison: Notebook Autoencoders vs Rule-Guided CVAE

## Compared Approaches

### A. Notebook method 1

File: `C:/Users/AIRC2/Desktop/Таня/Copy_of_Синтетические_данные.ipynb`

- Not an autoencoder.
- A `RandomForestRegressor` is trained on the 15 observed points.
- New `Radiation`, `Temperature`, and `Time` values are sampled uniformly.
- `Survival` is predicted by the regressor and perturbed with noise.

### B. Notebook method 2

File: `C:/Users/AIRC2/Desktop/Таня/Copy_of_Синтетические_данные.ipynb`

- A plain `VAE` is trained on the full 4D vector `[Radiation, Temperature, Time, Survival]`.
- New points are obtained by sampling latent vectors and decoding them into 4 fields.

### C. Notebook method 3

File: `C:/Users/AIRC2/Desktop/Таня/Copy_of_Синтетические_данные.ipynb`

- A simple `Conditional VAE` is trained as `p(Survival | Radiation, Temperature, Time)`.
- New `X = [Radiation, Temperature, Time]` are sampled uniformly.
- `Survival` is sampled from the decoder with one latent draw per condition.

### D. Current method

File: `C:/Users/AIRC2/Desktop/База знаний Таня/scripts/generate_rule_guided_cvae.py`

- A rule-guided `Conditional VAE` is trained on an augmented bootstrap frame.
- Literature-derived merged rules are encoded as soft guidance features.
- The model predicts survival in log-space.
- Synthetic output is filtered by rule consistency and evaluated quantitatively.

## Concrete Findings in the Old Notebook

### High severity

1. Wrong CSV variables are saved.

- In notebook cell 3, the code writes `df_new.to_csv("synthetic_data2.csv", ...)`, but method 2 actually generated `df_gen`.
- In notebook cell 5, the code writes `df_new.to_csv("synthetic_data3.csv", ...)`, but method 3 actually generated `df_syn`.
- This means the comparison among `synthetic_data.csv`, `synthetic_data2.csv`, and `synthetic_data3.csv` can be invalid if the cells were executed as shown.

2. The plain VAE does not condition on experimental inputs.

- Method 2 generates all 4 variables jointly from latent space.
- This can produce internally plausible-looking points, but it does not explicitly enforce `Survival` to be conditioned on `Radiation`, `Temperature`, and `Time`.
- For a dose-response setting this is a serious modeling weakness.

### Medium severity

3. No feature scaling or target transform.

- The notebook CVAE trains directly on raw `Radiation`, `Temperature`, `Time`, and strongly skewed `Survival`.
- `Survival` spans from `0.53` down to near zero, so raw-space training is unstable.
- The current method uses `log_survival`, which is much better for this target.

4. No domain-guided constraints during training.

- The notebook CVAE only applies a hard post-filter like `if Temperature > 44 and Radiation > 6 then Survival <= 0.01`.
- The current method injects domain structure before generation via rule-guided features and consistency filtering.

5. Uniform sampling ranges are much wider than the observed support.

- Notebook CVAE samples:
  - `Radiation` in `[0, 8]`
  - `Temperature` in `[40, 45]`
  - `Time` in `[20, 60]`
- Real data support is much narrower:
  - `Radiation` in `{0, 2, 4, 6, 8}`
  - `Temperature` in `{42, 43, 44}`
  - `Time` in `{30, 45}`
- This strongly increases support violations and synthetic drift.

6. No explicit quality evaluation in the notebook itself.

- The old notebook compares histograms visually.
- The current pipeline adds distribution, correlation, separability, utility, and coverage metrics.

### Low severity

7. Training is full-batch on only 15 observations.

- This is acceptable for a toy prototype, but too weak for a defendable synthetic data pipeline without additional safeguards.

## Quantitative Comparison

### Current rule-guided CVAE

Source:
- `C:/Users/AIRC2/Desktop/База знаний Таня/synthetic_data/evaluation_metrics.json`

Metrics:
- `mean_wasserstein_normalized = 0.0723`
- `mean_ks_statistic = 0.2175`
- `pearson_correlation_mean_abs_diff = 0.0556`
- `spearman_correlation_mean_abs_diff = 0.0856`
- `separability_auc_mean = 0.4396`
- `tstr_mae = 0.0153`
- `tstr_r2 = 0.9515`
- `support_violation_rate_mean = 0.1765`

### Notebook-style CVAE baseline

Source:
- `C:/Users/AIRC2/Desktop/База знаний Таня/synthetic_data_notebook_cvae/evaluation_metrics.json`

Metrics:
- `mean_wasserstein_normalized = 0.1421`
- `mean_ks_statistic = 0.3313`
- `pearson_correlation_mean_abs_diff = 0.2663`
- `spearman_correlation_mean_abs_diff = 0.2886`
- `separability_auc_mean = 0.6364`
- `tstr_mae = 0.0573`
- `tstr_r2 = 0.7255`
- `support_violation_rate_mean = 0.3743`

## Interpretation

The current rule-guided CVAE is better on all major axes:

- closer marginal distributions
- much better preservation of correlation structure
- much harder to distinguish from real data
- much better train-on-synthetic/test-on-real utility
- much lower support violation rate

The notebook-style CVAE is still useful as a baseline, but not as the main production generator for a strong evidence base.

## Recommendation

For a defendable synthetic dataset workflow:

1. Keep the current rule-guided CVAE as the primary generator.
2. Keep the notebook-style CVAE only as a baseline comparator.
3. Tighten support constraints in the current generator if strict physical plausibility is more important than diversity.
4. Document all assumptions explicitly, especially:
   - assumed interval
   - assumed hypoxia
   - rule-guided bootstrap augmentation
   - evaluation protocol

## Added Baseline Script

To preserve comparability, a baseline implementation of the notebook-style CVAE was created:

- `C:/Users/AIRC2/Desktop/База знаний Таня/scripts/baseline_notebook_cvae.py`

Its outputs are stored in:

- `C:/Users/AIRC2/Desktop/База знаний Таня/synthetic_data_notebook_cvae/`
