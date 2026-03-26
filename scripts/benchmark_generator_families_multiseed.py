from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from benchmark_generator_families import (
    BenchmarkConfig,
    build_matrix_family,
    make_teacher_blocks,
    set_seed,
    train_neural_family,
    train_residual_family,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "benchmarks" / "generator_family_benchmark_multiseed"
SEEDS = [7, 21, 42, 77, 123]


def run_seed(seed: int) -> List[Dict]:
    cfg = BenchmarkConfig(seed=seed, benchmark_subdir=f"generator_family_benchmark_multiseed/seed_{seed}")
    set_seed(seed)
    real_df, target_matrix, prior_matrix, sigma_matrix, teacher_blocks = make_teacher_blocks(cfg)

    rows: List[Dict] = []
    print(f"[family_multiseed] seed={seed} matrix")
    rows.append(build_matrix_family(cfg))

    for family in ("vae", "gan", "diffusion"):
        print(f"[family_multiseed] seed={seed} {family}")
        rows.append(
            train_neural_family(
                family=family,
                cfg=cfg,
                real_df=real_df,
                target_matrix=target_matrix,
                teacher_blocks=teacher_blocks,
            )
        )

    print(f"[family_multiseed] seed={seed} residual_vae")
    rows.append(
        train_residual_family(
            cfg=cfg,
            real_df=real_df,
            target_matrix=target_matrix,
            prior_matrix=prior_matrix,
            sigma_matrix=sigma_matrix,
            teacher_blocks=teacher_blocks,
        )
    )
    for row in rows:
        row["seed"] = seed
    return rows


def build_aggregate(summary_df: pd.DataFrame) -> pd.DataFrame:
    families = []
    for family, sub in summary_df.groupby("family", sort=False):
        record = {"family": family}
        for metric in [
            "local_mean_abs_error",
            "mean_wasserstein_normalized",
            "tstr_r2",
            "mean_constraint_pressure",
            "local_max_abs_error",
        ]:
            record[f"{metric}__mean"] = float(sub[metric].mean())
            record[f"{metric}__std"] = float(sub[metric].std(ddof=1))
            record[f"{metric}__min"] = float(sub[metric].min())
            record[f"{metric}__max"] = float(sub[metric].max())
        families.append(record)
    return pd.DataFrame(families)


def write_report(summary_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> None:
    lines = [
        "# Full Family Multi-seed Benchmark",
        "",
        "Seeds: `7, 21, 42, 77, 123`",
        "",
        "## Summary CSV",
        "",
        summary_df.to_csv(index=False),
        "",
        "## Aggregate CSV",
        "",
        aggregate_df.to_csv(index=False),
    ]
    (OUTDIR / "multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for seed in SEEDS:
        rows.extend(run_seed(seed))

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["seed", "family"]).reset_index(drop=True)
    aggregate_df = build_aggregate(summary_df)

    summary_df.to_csv(OUTDIR / "multiseed_summary.csv", index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(OUTDIR / "multiseed_aggregate.csv", index=False, encoding="utf-8-sig")
    with (OUTDIR / "multiseed_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    write_report(summary_df, aggregate_df)
    print(f"[family_multiseed] saved to {OUTDIR}")


if __name__ == "__main__":
    main()
