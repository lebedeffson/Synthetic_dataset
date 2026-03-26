# Benchmarks

This folder now contains the active benchmark artifacts for the current synthetic-generation project, not only legacy leftovers.

## Active Benchmark Outputs

- `generator_family_benchmark/`
  Single-run comparison of the current generator families:
  `Matrix`, `Residual VAE`, `VAE`, `Diffusion`, `GAN`.

- `generator_family_benchmark_multiseed/`
  Full family benchmark across multiple seeds. This is the main benchmark used in the presentation materials and method comparison tables.

- `residual_rule_aware_vae/`
  Dedicated single-run artifacts for the hybrid `Residual VAE` generator.

- `residual_rule_aware_vae_multiseed/`
  Dedicated multi-seed robustness run for `Residual VAE`.

## Main Scripts

- `../scripts/benchmark_generator_families.py`
  Runs the single-run family benchmark.

- `../scripts/benchmark_generator_families_multiseed.py`
  Runs the full multi-seed family benchmark.

- `../scripts/benchmark_residual_rule_aware_vae_multiseed.py`
  Runs the dedicated multi-seed benchmark for `Residual VAE`.

## Canonical Run Order

### 1. Production dataset

```bash
/home/lebedeffson/Code/venv/bin/python scripts/build_cell_level_final_pipeline.py
/home/lebedeffson/Code/venv/bin/python scripts/analyze_counterfactual_rule_interventions.py
/home/lebedeffson/Code/venv/bin/python scripts/explain_constraint_pressure_shap.py
/home/lebedeffson/Code/venv/bin/python scripts/analyze_cell_level_final_dataset.py
/home/lebedeffson/Code/venv/bin/python scripts/perform_rule_ablations.py
```

### 2. Generator benchmarks

```bash
/home/lebedeffson/Code/venv/bin/python scripts/benchmark_generator_families.py
/home/lebedeffson/Code/venv/bin/python scripts/benchmark_residual_rule_aware_vae_multiseed.py
/home/lebedeffson/Code/venv/bin/python scripts/benchmark_generator_families_multiseed.py
```

### 3. Presentation materials

```bash
/home/lebedeffson/Code/venv/bin/python scripts/build_presentation_materials.py
```

## Notes

- The repository uses the virtual environment at `/home/lebedeffson/Code/venv/bin/python`.
- The benchmark reports consumed by the current presentation bundle are generated from the folders listed above.
- Legacy scripts are still available under `../scripts/legacy/` if an old result needs to be reproduced separately.
