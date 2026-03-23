# Implementation Review

## Scope

Review target:

- old `notebook_cvae`
- `rule_guided_cvae`
- continuous `rule_guided_kernel`
- final `cell_level_article_guided` pipeline

## Main Findings

### 1. The old notebook-style CVAE is not suitable as the main production generator

Reasons:

- it generates unsupported treatment conditions from wide continuous ranges;
- it relies on weak post-hoc clipping instead of article-guided constraints;
- it shows poor fidelity and poor support compliance;
- it is not aligned with the exact 15-point design.

### 2. The first rule-guided CVAE is better, but still mixes cell-level and clinical logic

Strengths:

- it substantially improves fidelity and utility over the old notebook baseline;
- it embeds domain guidance in features, loss, and filtering.

Weaknesses:

- it uses assumptions such as tumor hypoxia and HT-RT interval even though the dataset is only a cell-level survival matrix;
- it generates many continuous treatment conditions outside the exact observed design;
- part of the previous article-based validation reused the same derived rule features, which can overstate apparent compliance.

### 3. The continuous kernel generator is conservative, but still not fully aligned with the exact design

Strengths:

- much better support behavior than the CVAE variants;
- strong fidelity and utility.

Weaknesses:

- only a small fraction of rows stay on the exact 15 observed design points;
- independent cell-level checks are only partially applicable because most rows are not exact design conditions.

### 4. The final cell-level pipeline is the most defensible for this dataset

Design choices:

- only cell-level rules are kept;
- the exact experimental support is preserved;
- synthetic variation is introduced only in survival;
- each generated 5x3 survival block is projected to preserve monotonicity across radiation and thermal condition;
- the projected blocks are calibrated back to the observed design-point matrix to reduce local bias without breaking the active rules;
- explicit traceability and design-point explanation files are generated alongside the final dataset.

This makes the final dataset much easier to justify scientifically than a free-form deep generator trained on only 15 rows.

## Why The Rule Split Was Necessary

The project originally mixed three kinds of rules:

- cell-level mechanistic rules;
- tumor/in vivo rules;
- clinical outcome rules.

For this dataset, only the first group is directly appropriate.

Examples of rules that were moved out of the default cell-level generator:

- perfusion / oxygenation gain rules for hypoxic tumors;
- patient interval rules;
- `CEM43 T90` superficial-tumor local-control rules;
- repeated-session clinical control rules.

## Final Quantitative Summary

Final primary dataset: `synthetic_data_cell_level_final/final_synthetic_dataset.csv`

- exact design support: `1.0000`
- local mean absolute error: `0.0023`
- radiation monotonicity: `1.0000`
- thermal monotonicity: `1.0000`
- independent article compliance: `1.0000`
- normalized Wasserstein: `0.0050`
- mean KS statistic: `0.0203`
- TSTR MAE: `0.0023`
- TSTR R2: `0.9991`
- support violation mean: `0.0000`
- duplicate rate vs real: `0.0680`

## Residual Limitations

These should still be stated explicitly in any report:

- the real dataset has only 15 observations;
- synthetic variability is model-based because there are no biological replicates per design point;
- the final dataset is best interpreted as a design-preserving synthetic augmentation, not as a simulator for arbitrary unseen treatment conditions;
- very strong metrics are expected partly because the final generator intentionally preserves the exact support of the observed experiment.

## Recommendation

Use the final `cell_level_article_guided` dataset as the primary synthetic dataset for downstream ML/AI.

Keep the older generators only as baselines in the comparison section, not as the production dataset.

Those historical baselines are now stored under `benchmarks/` and `scripts/legacy/` so the active project surface remains focused on the final cell-level pipeline.
