from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from generate_rule_guided_cvae import (
    EPS,
    GenerationConfig,
    build_bootstrap_training_frame,
    enrich_with_rule_features,
    evaluate_synthetic_quality,
    inverse_log_survival,
    kernel_predict,
    make_raw_frame,
    set_seed,
)


@dataclass
class KernelGeneratorConfig:
    seed: int = 42
    n_synthetic: int = 1000
    n_boot_per_row: int = 80
    assumed_interval_min: float = 0.0
    assumed_hypoxia: float = 0.70
    outdir: str = "benchmarks/results/synthetic_data_kernel"
    evaluation_rounds: int = 10


def sample_conditions_conservative(observed_df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    observed_df = observed_df.copy()

    for _ in range(n_samples):
        base = observed_df.sample(n=1, replace=True, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        radiation = float(np.clip(rng.normal(base["Радиация"], 0.28), 0.0, 8.0))
        temperature = float(np.clip(rng.normal(base["Температура"], 0.10), 42.0, 44.0))
        duration = float(np.clip(rng.normal(base["Время"], 1.8), 30.0, 45.0))

        rows.append(
            {
                "Радиация": radiation,
                "Температура": temperature,
                "Время": duration,
                "Выживаемость": 0.0,
                "source": "synthetic_condition_kernel",
            }
        )

    return pd.DataFrame(rows)


def generate_kernel_synthetic_dataset(observed_df: pd.DataFrame, cfg: KernelGeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_cfg = GenerationConfig(
        seed=cfg.seed,
        n_boot_per_row=cfg.n_boot_per_row,
        n_synthetic=cfg.n_synthetic,
        assumed_interval_min=cfg.assumed_interval_min,
        assumed_hypoxia=cfg.assumed_hypoxia,
        discriminator_rounds=cfg.evaluation_rounds,
        outdir=cfg.outdir,
    )
    train_df = build_bootstrap_training_frame(observed_df, train_cfg)

    base_x = train_df[["Радиация", "Температура", "Время"]].to_numpy(dtype=float)
    base_log_y = train_df["log_survival"].to_numpy(dtype=float)
    base_rule = train_df["Оценка_правил"].to_numpy(dtype=float)

    cond_df = sample_conditions_conservative(observed_df, cfg.n_synthetic, cfg.seed)
    cond_df = enrich_with_rule_features(cond_df, cfg.assumed_interval_min, cfg.assumed_hypoxia)

    query_x = cond_df[["Радиация", "Температура", "Время"]].to_numpy(dtype=float)
    kernel_log_surv = kernel_predict(base_x, base_log_y, query_x, bandwidth=(0.75, 0.12, 2.0))
    kernel_rule = kernel_predict(base_x, base_rule, query_x, bandwidth=(0.75, 0.12, 2.0))

    direct_rule = cond_df["Оценка_правил"].to_numpy(dtype=float)
    rule_delta = direct_rule - kernel_rule

    rng = np.random.default_rng(cfg.seed + 101)
    noise = rng.normal(0.0, 0.035, size=len(cond_df))
    log_surv = np.clip(kernel_log_surv - 0.55 * rule_delta + noise, math.log10(EPS), -0.001)
    survival = inverse_log_survival(log_surv)

    cond_df["Выживаемость"] = np.clip(survival, 0.0, 1.0)
    cond_df["Неопределенность_kernel"] = np.abs(noise)
    cond_df["Источник"] = "rule_guided_kernel"

    upper_bound = np.clip(1.0 - 0.90 * cond_df["Оценка_правил"].to_numpy(dtype=float), 0.0, 1.0)
    gap = np.clip(cond_df["Выживаемость"].to_numpy(dtype=float) - upper_bound, 0.0, None)
    cond_df["RuleConsistency"] = np.clip(1.0 - gap, 0.0, 1.0)

    filtered = cond_df[cond_df["RuleConsistency"] >= 0.90].copy()
    if len(filtered) < max(250, cfg.n_synthetic // 3):
        filtered = cond_df[cond_df["RuleConsistency"] >= 0.85].copy()

    filtered = filtered.sort_values(
        by=["Радиация", "Температура", "Время", "Выживаемость"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return train_df, filtered


def save_outputs(train_df: pd.DataFrame, synthetic_df: pd.DataFrame, cfg: KernelGeneratorConfig, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "rule_guided_training_frame.csv", index=False, encoding="utf-8-sig")

    minimal_cols = ["Радиация", "Температура", "Время", "Выживаемость"]
    synthetic_df[minimal_cols].to_csv(outdir / "kernel_synthetic_dataset.csv", index=False, encoding="utf-8-sig")
    synthetic_df.to_csv(outdir / "kernel_synthetic_dataset_full.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "generator": "rule_guided_kernel",
        "seed": cfg.seed,
        "n_boot_per_row": cfg.n_boot_per_row,
        "n_synthetic_requested": cfg.n_synthetic,
        "n_synthetic_saved": int(len(synthetic_df)),
        "assumed_interval_min": cfg.assumed_interval_min,
        "assumed_hypoxia": cfg.assumed_hypoxia,
        "rule_consistency_threshold_preferred": 0.90,
        "outdir": cfg.outdir,
    }
    with (outdir / "kernel_run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = KernelGeneratorConfig()
    set_seed(cfg.seed)
    real_df = make_raw_frame()[["Радиация", "Температура", "Время", "Выживаемость"]].copy()
    train_df, synthetic_df = generate_kernel_synthetic_dataset(real_df, cfg)

    project_root = Path(__file__).resolve().parents[1]
    outdir = project_root / cfg.outdir
    save_outputs(train_df, synthetic_df, cfg, outdir)

    eval_cfg = GenerationConfig(
        seed=cfg.seed,
        n_synthetic=cfg.n_synthetic,
        n_boot_per_row=cfg.n_boot_per_row,
        assumed_interval_min=cfg.assumed_interval_min,
        assumed_hypoxia=cfg.assumed_hypoxia,
        discriminator_rounds=cfg.evaluation_rounds,
        outdir=cfg.outdir,
    )
    evaluation_files = evaluate_synthetic_quality(real_df, synthetic_df, eval_cfg, outdir)
    print(f"saved synthetic dataset to: {outdir / 'kernel_synthetic_dataset.csv'}")
    for path in evaluation_files:
        print(f"saved evaluation output to: {path}")


if __name__ == "__main__":
    main()
