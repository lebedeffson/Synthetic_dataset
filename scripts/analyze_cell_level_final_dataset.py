from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common_synthetic_metrics import (
    COL_RADIATION,
    COL_SURVIVAL,
    COL_TEMPERATURE,
    COL_TIME,
    EvaluationConfig,
    compute_correlation_metrics,
    compute_coverage_metrics,
    compute_distribution_metrics,
    compute_separability_metrics,
    compute_utility_metrics,
)
from generate_cell_level_article_guided_dataset import CellLevelConfig, build_generation_artifacts, make_real_dataframe
from validate_cell_level_datasets import evaluate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "synthetic_data_cell_level_final"
SEEDS = [7, 11, 17, 23, 42, 84, 126, 256]
BOOTSTRAP_REPEATS = 2000


def bootstrap_design_point_ci(final_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260323)
    rows: List[Dict] = []

    grouped_real = real_df.groupby([COL_RADIATION, COL_TEMPERATURE, COL_TIME], as_index=False)[COL_SURVIVAL].first()

    for _, real_row in grouped_real.iterrows():
        mask = (
            (final_df[COL_RADIATION] == real_row[COL_RADIATION])
            & (final_df[COL_TEMPERATURE] == real_row[COL_TEMPERATURE])
            & (final_df[COL_TIME] == real_row[COL_TIME])
        )
        values = final_df.loc[mask, COL_SURVIVAL].to_numpy(dtype=float)
        boot_means = []
        for _ in range(BOOTSTRAP_REPEATS):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_means.append(float(np.mean(sample)))
        boot_means = np.asarray(boot_means, dtype=float)

        ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
        rows.append(
            {
                COL_RADIATION: float(real_row[COL_RADIATION]),
                COL_TEMPERATURE: float(real_row[COL_TEMPERATURE]),
                COL_TIME: float(real_row[COL_TIME]),
                "real_survival": float(real_row[COL_SURVIVAL]),
                "synthetic_mean": float(np.mean(values)),
                "bootstrap_mean": float(np.mean(boot_means)),
                "bootstrap_std": float(np.std(boot_means, ddof=0)),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "real_inside_ci95": bool(ci_low <= float(real_row[COL_SURVIVAL]) <= ci_high),
                "n_rows": int(len(values)),
            }
        )

    return pd.DataFrame(rows).sort_values([COL_RADIATION, COL_TEMPERATURE, COL_TIME]).reset_index(drop=True)


def evaluate_single_seed(seed: int) -> tuple[Dict, pd.DataFrame]:
    cfg = CellLevelConfig(seed=seed)
    real_df, synthetic_df, _ = build_generation_artifacts(cfg)
    real_min = real_df[[COL_RADIATION, COL_TEMPERATURE, COL_TIME, COL_SURVIVAL]].copy()
    synth_min = synthetic_df[[COL_RADIATION, COL_TEMPERATURE, COL_TIME, COL_SURVIVAL]].copy()
    eval_cfg = EvaluationConfig(seed=seed)

    distribution_df = compute_distribution_metrics(real_min, synth_min)
    metrics = {
        "seed": seed,
        "mean_wasserstein_normalized": float(distribution_df["wasserstein_normalized"].mean()),
        "mean_ks_statistic": float(distribution_df["ks_statistic"].mean()),
    }
    metrics.update(compute_correlation_metrics(real_min, synth_min))
    metrics.update(compute_separability_metrics(real_min, synth_min, eval_cfg))
    metrics.update(compute_utility_metrics(real_min, synth_min, eval_cfg))
    metrics.update(compute_coverage_metrics(real_min, synth_min))

    independent = evaluate_dataset(synth_min, f"seed_{seed}")
    metrics.update(
        {
            "exact_design_support_rate": independent["exact_design_support_rate"],
            "local_mean_abs_error": independent["local_mean_abs_error"],
            "local_max_abs_error": independent["local_max_abs_error"],
            "radiation_monotonicity_mean_rate": independent["radiation_monotonicity_mean_rate"],
            "thermal_monotonicity_mean_rate": independent["thermal_monotonicity_mean_rate"],
            "high_combined_dose_low_survival_rate": independent["high_combined_dose_low_survival_rate"],
            "independent_article_compliance_mean": independent["independent_article_compliance_mean"],
        }
    )

    design_df = pd.DataFrame(independent["grouped_design_summary"])[
        [COL_RADIATION, COL_TEMPERATURE, COL_TIME, "synthetic_mean", "abs_mean_error"]
    ].copy()
    design_df["seed"] = seed
    return metrics, design_df


def build_multiseed_robustness() -> Dict[str, Path]:
    metric_rows: List[Dict] = []
    design_rows: List[pd.DataFrame] = []
    for seed in SEEDS:
        metrics, design_df = evaluate_single_seed(seed)
        metric_rows.append(metrics)
        design_rows.append(design_df)

    metrics_df = pd.DataFrame(metric_rows).sort_values("seed").reset_index(drop=True)
    design_df = pd.concat(design_rows, ignore_index=True)

    summary_df = metrics_df.drop(columns=["seed"]).agg(["mean", "std", "min", "max"]).T.reset_index()
    summary_df = summary_df.rename(columns={"index": "metric"})

    design_summary = (
        design_df.groupby([COL_RADIATION, COL_TEMPERATURE, COL_TIME], as_index=False)
        .agg(
            synthetic_mean_mean=("synthetic_mean", "mean"),
            synthetic_mean_std=("synthetic_mean", "std"),
            synthetic_mean_min=("synthetic_mean", "min"),
            synthetic_mean_max=("synthetic_mean", "max"),
            abs_mean_error_mean=("abs_mean_error", "mean"),
            abs_mean_error_max=("abs_mean_error", "max"),
        )
        .sort_values([COL_RADIATION, COL_TEMPERATURE, COL_TIME])
        .reset_index(drop=True)
    )

    worst_point = design_summary.sort_values("abs_mean_error_max", ascending=False).iloc[0].to_dict()

    metrics_csv = OUTDIR / "robustness_multiseed_metrics.csv"
    summary_csv = OUTDIR / "robustness_multiseed_summary.csv"
    design_csv = OUTDIR / "robustness_multiseed_design_points.csv"
    json_path = OUTDIR / "robustness_multiseed_summary.json"
    md_path = OUTDIR / "robustness_multiseed_summary.md"

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    design_summary.to_csv(design_csv, index=False, encoding="utf-8-sig")

    payload = {
        "seeds": SEEDS,
        "n_runs": len(SEEDS),
        "worst_design_point_by_max_error": worst_point,
        "metric_summary": summary_df.to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    md_lines = [
        "# Multi-Seed Robustness Summary",
        "",
        f"Seeds analyzed: `{', '.join(map(str, SEEDS))}`",
        "",
        "## Key Findings",
        "",
        f"- `exact_design_support_rate` stayed at `{metrics_df['exact_design_support_rate'].min():.4f}` to `{metrics_df['exact_design_support_rate'].max():.4f}` across all runs.",
        f"- `radiation_monotonicity_mean_rate` stayed at `{metrics_df['radiation_monotonicity_mean_rate'].min():.4f}` to `{metrics_df['radiation_monotonicity_mean_rate'].max():.4f}`.",
        f"- `thermal_monotonicity_mean_rate` stayed at `{metrics_df['thermal_monotonicity_mean_rate'].min():.4f}` to `{metrics_df['thermal_monotonicity_mean_rate'].max():.4f}`.",
        f"- `local_mean_abs_error` ranged from `{metrics_df['local_mean_abs_error'].min():.4f}` to `{metrics_df['local_mean_abs_error'].max():.4f}`.",
        f"- `mean_wasserstein_normalized` ranged from `{metrics_df['mean_wasserstein_normalized'].min():.4f}` to `{metrics_df['mean_wasserstein_normalized'].max():.4f}`.",
        f"- Worst design-point max error across seeds: `{int(worst_point[COL_RADIATION])} Gy, {int(worst_point[COL_TEMPERATURE])} C, {int(worst_point[COL_TIME])} min -> {worst_point['abs_mean_error_max']:.4f}`.",
        "",
        "## Outputs",
        "",
        "- `robustness_multiseed_metrics.csv`",
        "- `robustness_multiseed_summary.csv`",
        "- `robustness_multiseed_design_points.csv`",
        "- `robustness_multiseed_summary.json`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "metrics_csv": metrics_csv,
        "summary_csv": summary_csv,
        "design_csv": design_csv,
        "json": json_path,
        "md": md_path,
    }


def build_final_analysis_report(ci_df: pd.DataFrame, robustness_paths: Dict[str, Path]) -> Path:
    with (OUTDIR / "evaluation_metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    with (OUTDIR / "independent_cell_level_validation.json").open("r", encoding="utf-8") as fh:
        independent = json.load(fh)
    with (OUTDIR / "robustness_multiseed_summary.json").open("r", encoding="utf-8") as fh:
        robustness = json.load(fh)

    ci_coverage = float(np.mean(ci_df["real_inside_ci95"].astype(float)))
    largest_ci = ci_df.sort_values("ci95_high", ascending=False).iloc[0]
    widest_ci = (ci_df["ci95_high"] - ci_df["ci95_low"]).max()
    ci_misses = ci_df[~ci_df["real_inside_ci95"]].copy()
    ci_miss_text = []
    for _, row in ci_misses.iterrows():
        ci_miss_text.append(
            f"{int(row[COL_RADIATION])} Gy, {int(row[COL_TEMPERATURE])} C, {int(row[COL_TIME])} min "
            f"(real={row['real_survival']:.6g}, CI=[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}])"
        )

    report_path = OUTDIR / "final_analysis_report.md"
    lines = [
        "# Final Analysis Report",
        "",
        "## What The Final Dataset Now Demonstrates",
        "",
        "- The dataset stays exactly on the original 15 observed design points.",
        "- The synthetic survival surface remains monotone in radiation and in observed thermal ordering.",
        "- The generator is calibrated back to the observed matrix, so local deviation from the real design is small.",
        "- Explainability is explicit: each active rule is linked to evidence, generator use, validation use, and design-point behavior.",
        "",
        "## Main Quality Metrics",
        "",
        f"- `mean_wasserstein_normalized = {metrics['mean_wasserstein_normalized']:.4f}`",
        f"- `mean_ks_statistic = {metrics['mean_ks_statistic']:.4f}`",
        f"- `tstr_mae = {metrics['tstr_mae']:.4f}`",
        f"- `tstr_r2 = {metrics['tstr_r2']:.4f}`",
        f"- `support_violation_rate_mean = {metrics['support_violation_rate_mean']:.4f}`",
        f"- `duplicate_rate_vs_real = {metrics['duplicate_rate_vs_real']:.4f}`",
        f"- `local_mean_abs_error = {independent['local_mean_abs_error']:.4f}`",
        f"- `local_max_abs_error = {independent['local_max_abs_error']:.4f}`",
        "",
        "## Uncertainty And Stability",
        "",
        f"- Bootstrap CI coverage for real survival values across design points: `{ci_coverage:.4f}`.",
        f"- Widest design-point CI width: `{widest_ci:.4f}`.",
        f"- Highest CI upper bound occurred at `{int(largest_ci[COL_RADIATION])} Gy, {int(largest_ci[COL_TEMPERATURE])} C, {int(largest_ci[COL_TIME])} min`.",
        f"- Multi-seed runs analyzed: `{robustness['n_runs']}`.",
        f"- Worst design-point max error across seeds: `{int(robustness['worst_design_point_by_max_error'][COL_RADIATION])} Gy, {int(robustness['worst_design_point_by_max_error'][COL_TEMPERATURE])} C, {int(robustness['worst_design_point_by_max_error'][COL_TIME])} min -> {robustness['worst_design_point_by_max_error']['abs_mean_error_max']:.4f}`.",
        "",
        "CI misses:",
        "",
        *[f"- {text}" for text in ci_miss_text],
        "",
        "## Interpretation",
        "",
        "- For this project, the final dataset is best described as design-preserving synthetic augmentation rather than a free-form simulator.",
        "- The strongest claims are about fidelity to the observed matrix and agreement with curated cell-level rules.",
        "- The two remaining sensitive points are the maximum observed survival point and the zero-survival boundary point; this is a typical consequence of generating a smooth stochastic cloud around hard biological bounds.",
        "- The weakest claim remains mechanistic generalization outside the observed domain, so the dataset should not be used to infer arbitrary unseen treatment regimes.",
        "- Exact-boundary fidelity and lower duplicate rate are in tension here: preserving a small amount of variability near `0.53` and `0.0` keeps the dataset less degenerate, but it slightly weakens bootstrap CI coverage at those boundaries.",
        "",
        "## Files To Cite In The Project",
        "",
        "- `final_dataset_report.md`",
        "- `final_analysis_report.md`",
        "- `independent_cell_level_validation.json`",
        "- `design_point_bootstrap_ci.csv`",
        "- `robustness_multiseed_summary.md`",
        "- `knowledge_base/cell_level_rule_traceability.md`",
        "- `design_point_rule_explanations.md`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    final_df = pd.read_csv(OUTDIR / "final_synthetic_dataset.csv")
    real_df = make_real_dataframe()[[COL_RADIATION, COL_TEMPERATURE, COL_TIME, COL_SURVIVAL]].copy()

    ci_df = bootstrap_design_point_ci(final_df, real_df)
    ci_csv = OUTDIR / "design_point_bootstrap_ci.csv"
    ci_df.to_csv(ci_csv, index=False, encoding="utf-8-sig")

    robustness_paths = build_multiseed_robustness()
    report_path = build_final_analysis_report(ci_df, robustness_paths)

    print("saved analysis files:")
    print(ci_csv)
    for path in robustness_paths.values():
        print(path)
    print(report_path)


if __name__ == "__main__":
    main()
