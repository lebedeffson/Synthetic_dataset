from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from generate_cell_level_article_guided_dataset import CellLevelConfig, blocks_to_dataframe, save_outputs
from generate_residual_rule_aware_vae import (
    ResidualRuleAwareVAEConfig,
    apply_block_constraints,
    make_teacher_blocks,
    sample_blocks,
    set_seed,
    summarize_records,
    train_model,
)
from validate_cell_level_datasets import evaluate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "benchmarks" / "residual_rule_aware_vae_multiseed"
SEEDS = [7, 21, 42, 77, 123]


def run_seed(seed: int) -> Dict:
    cfg = ResidualRuleAwareVAEConfig(
        seed=seed,
        outdir=str(Path("benchmarks") / "residual_rule_aware_vae_multiseed" / f"seed_{seed}"),
    )
    set_seed(seed)
    real_df, target_matrix, prior_matrix, sigma_matrix, teacher_blocks = make_teacher_blocks(cfg)
    residual_targets = teacher_blocks.reshape(len(teacher_blocks), -1).astype(np.float32) - prior_matrix.reshape(1, -1).astype(np.float32)
    target_residual_mean = residual_targets.mean(axis=0)

    model, train_info, history_df = train_model(teacher_blocks, prior_matrix, sigma_matrix, cfg)
    n_blocks = int(math.ceil(cfg.n_synthetic / target_matrix.size))
    raw_blocks = sample_blocks(model, sigma_matrix, prior_matrix, target_residual_mean, n_blocks, cfg)
    clipped_blocks, constrained_blocks, records = apply_block_constraints(raw_blocks, prior_matrix, target_matrix, sigma_matrix, cfg)

    cell_cfg = CellLevelConfig(
        seed=cfg.seed,
        n_synthetic=cfg.n_synthetic,
        outdir=cfg.outdir,
        explainability_mode="log_only",
        save_explainability_plots=False,
    )
    raw_df = blocks_to_dataframe(clipped_blocks, cell_cfg)
    raw_validation = evaluate_dataset(
        raw_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(),
        f"residual_rule_aware_vae_seed_{seed}_raw",
    )
    synthetic_df = blocks_to_dataframe(constrained_blocks, cell_cfg)
    metadata = {
        "generator": "residual_rule_aware_block_vae",
        "seed": seed,
        "model_config": cfg.__dict__,
        "train_info": train_info,
        "rules_file": "knowledge_base/cell_level_rules.json",
        "teacher_source": "rule-guided matrix bootstrap blocks",
        "teacher_blocks": cfg.teacher_blocks,
        "raw_independent_validation": raw_validation,
        "explainability_records": records,
        "explainability_summary": summarize_records(records),
        "target_matrix_log10": target_matrix.tolist(),
        "prior_mu_matrix_log10": prior_matrix.tolist(),
        "prior_sigma_matrix_log10": sigma_matrix.tolist(),
    }

    outdir = save_outputs(real_df, synthetic_df, metadata, cell_cfg)
    history_df.to_csv(outdir / "training_history.csv", index=False, encoding="utf-8-sig")
    with (outdir / "raw_independent_cell_level_validation.json").open("w", encoding="utf-8") as fh:
        json.dump(raw_validation, fh, ensure_ascii=False, indent=2)

    independent = evaluate_dataset(
        synthetic_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(),
        f"residual_rule_aware_vae_seed_{seed}",
    )
    with (outdir / "independent_cell_level_validation.json").open("w", encoding="utf-8") as fh:
        json.dump(independent, fh, ensure_ascii=False, indent=2)

    with (outdir / "evaluation_metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    return {
        "seed": seed,
        "local_mean_abs_error": independent["local_mean_abs_error"],
        "local_max_abs_error": independent["local_max_abs_error"],
        "mean_wasserstein_normalized": metrics["mean_wasserstein_normalized"],
        "tstr_r2": metrics["tstr_r2"],
        "mean_constraint_pressure": metadata["explainability_summary"]["mean_constraint_pressure"],
        "raw_local_mean_abs_error": raw_validation["local_mean_abs_error"],
        "raw_article_compliance_mean": raw_validation["independent_article_compliance_mean"],
        "raw_radiation_monotonicity_mean_rate": raw_validation["radiation_monotonicity_mean_rate"],
        "raw_thermal_monotonicity_mean_rate": raw_validation["thermal_monotonicity_mean_rate"],
        "mean_raw_radiation_violation": metadata["explainability_summary"]["mean_raw_radiation_violation"],
        "mean_raw_thermal_violation": metadata["explainability_summary"]["mean_raw_thermal_violation"],
        "parameter_count": train_info["parameters"],
    }


def build_report(summary_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> None:
    lines = [
        "# Residual Rule-Aware VAE Multi-seed Benchmark",
        "",
        "Seeds: `7, 21, 42, 77, 123`",
        "",
        "## Summary",
        "",
        summary_df.to_csv(index=False),
        "",
        "## Aggregate",
        "",
        aggregate_df.to_csv(),
    ]
    (OUTDIR / "residual_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for seed in SEEDS:
        print(f"[residual_multiseed] seed={seed}")
        rows.append(run_seed(seed))

    summary_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    aggregate_df = summary_df[
        [
            "local_mean_abs_error",
            "local_max_abs_error",
            "mean_wasserstein_normalized",
            "tstr_r2",
            "mean_constraint_pressure",
            "raw_local_mean_abs_error",
            "raw_article_compliance_mean",
            "raw_radiation_monotonicity_mean_rate",
            "mean_raw_radiation_violation",
            "mean_raw_thermal_violation",
        ]
    ].agg(["mean", "std", "min", "max"])

    summary_df.to_csv(OUTDIR / "residual_multiseed_summary.csv", index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(OUTDIR / "residual_multiseed_aggregate.csv", encoding="utf-8-sig")
    build_report(summary_df, aggregate_df)
    print(f"[residual_multiseed] saved to {OUTDIR}")


if __name__ == "__main__":
    main()
