# Thermoradiobiology Expert KB

This workspace contains a literature-grounded expert knowledge base about the combined effect of temperature and ionizing radiation on cancer cells and tumors.

## Files

- `literature/literature_slice.md` - curated literature slice with extracted patterns
- `literature/literature_evidence.csv` - tabular evidence sheet for downstream analysis
- `knowledge_base/domain_ontology.yaml` - domain ontology, variables, and fuzzy partitions
- `knowledge_base/fuzzy_rules.fcl` - fuzzy `IF-THEN` rule base in FCL style
- `knowledge_base/rules_prolog.pl` - alternative crisp logical representation in Prolog-like facts
- `knowledge_base/expert_kb.json` - unified machine-readable expert KB
- `knowledge_base/merged_rules_ru.md` - human-readable merged rules in Russian
- `knowledge_base/merged_rules.yaml` - merged rule base in YAML
- `knowledge_base/merged_rules.json` - merged rule base in JSON
- `scripts/generate_rule_guided_cvae.py` - Python generator for a rule-guided Conditional VAE synthetic dataset
- `notebooks/rule_guided_cvae_workflow.ipynb` - standalone Jupyter notebook with the full CVAE workflow code inside
- `synthetic_data/cvae_synthetic_dataset.csv` - final generated synthetic dataset with 4 columns only
- `synthetic_data/cvae_synthetic_dataset_full.csv` - extended synthetic dataset with rule features and diagnostics
- `synthetic_data/rule_guided_training_frame.csv` - augmented training frame used for CVAE fitting
- `synthetic_data/cvae_run_metadata.json` - generator run metadata
- `synthetic_data/evaluation_distribution_metrics.csv` - per-column distribution metrics
- `synthetic_data/evaluation_summary_statistics.csv` - descriptive statistics comparison
- `synthetic_data/evaluation_metrics.json` - global quality metrics for the synthetic dataset
- `ml/train_rules.jsonl` - structured examples for ML/AI training

## Core Modeling Assumptions

- The main focus is locoregional hyperthermia combined with radiotherapy.
- The best-supported sensitizing window is roughly `39-43 C`, with many mechanistic data concentrated around `41-43 C`.
- Temperatures above `43 C` are modeled as a transition zone toward stronger direct cytotoxicity and higher normal-tissue risk.
- Conflicting interval findings are preserved through weighted rules rather than collapsed into a single hard statement.

## Suggested Usage

- Expert system: load `knowledge_base/domain_ontology.yaml` and `knowledge_base/fuzzy_rules.fcl`.
- Logic engine: load `knowledge_base/rules_prolog.pl` or parse `knowledge_base/expert_kb.json`.
- Retrieval / RAG: index `literature/literature_slice.md` and `knowledge_base/expert_kb.json`.
- ML/AI: use `literature/literature_evidence.csv` as an evidence table and `ml/train_rules.jsonl` as supervised examples.

## Synthetic Dataset Generation

Run:

```bash
python scripts/generate_rule_guided_cvae.py --epochs 250 --n-boot-per-row 80 --n-synthetic 1000 --device cpu
```

What the script does:

- computes derived thermoradiobiology features such as `CEM43`
- operationalizes merged rules as fuzzy guidance signals
- builds a rule-guided bootstrap training frame from the small raw dataset
- trains a Conditional VAE on log-transformed survival
- generates a synthetic dataset and filters it by rule consistency
- computes quality metrics for synthetic tabular data, including distribution similarity, correlation preservation, separability, utility, and coverage

## Caution

This is a research knowledge base, not a validated clinical treatment protocol. Human expert review is required before any biomedical or clinical use.
