from __future__ import annotations

import json
from pathlib import Path

from compare_generators_article_compliance import evaluate_article_compliance
from generate_rule_guided_cvae import GenerationConfig, evaluate_synthetic_quality, make_raw_frame, set_seed
from rule_guided_kernel_generator import KernelGeneratorConfig, generate_kernel_synthetic_dataset


def main() -> None:
    cfg = KernelGeneratorConfig(outdir="benchmarks/results/synthetic_data_final")
    set_seed(cfg.seed)

    project_root = Path(__file__).resolve().parents[1]
    outdir = project_root / cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    real_df = make_raw_frame()[["Радиация", "Температура", "Время", "Выживаемость"]].copy()
    train_df, synthetic_df = generate_kernel_synthetic_dataset(real_df, cfg)

    train_df.to_csv(outdir / "final_training_frame.csv", index=False, encoding="utf-8-sig")
    synthetic_df[["Радиация", "Температура", "Время", "Выживаемость"]].to_csv(
        outdir / "final_synthetic_dataset.csv", index=False, encoding="utf-8-sig"
    )
    synthetic_df.to_csv(outdir / "final_synthetic_dataset_full.csv", index=False, encoding="utf-8-sig")

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
    article_compliance = evaluate_article_compliance(
        synthetic_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(),
        "final_article_guided_dataset",
    )

    article_json_path = outdir / "final_article_rule_compliance.json"
    with article_json_path.open("w", encoding="utf-8") as fh:
        json.dump(article_compliance, fh, ensure_ascii=False, indent=2)

    metrics_path = outdir / "evaluation_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    auc_mean = metrics["separability_auc_mean"]
    gini_abs = abs(2.0 * auc_mean - 1.0)

    metadata = {
        "selected_generator": "rule_guided_kernel",
        "reason": "best balance of statistical fidelity, article-rule compliance, and support compliance for n=15",
        "n_synthetic_saved": int(len(synthetic_df)),
        "evaluation_outputs": evaluation_files,
        "article_compliance_file": str(article_json_path),
    }
    with (outdir / "final_generation_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    report_lines = [
        "# Final Article-Guided Synthetic Dataset",
        "",
        "Selected generator: `rule_guided_kernel`.",
        "",
        "## Why this generator was selected",
        "",
        "- very low support violation on the 15-point experimental domain;",
        "- best distribution and correlation fidelity among tested generators;",
        "- highest article-based rule compliance among tested generators;",
        "- utility close to the best CVAE result, without the same degree of support drift.",
        "",
        "## Final metrics",
        "",
        f"- `n_synthetic`: {len(synthetic_df)}",
        f"- `mean_wasserstein_normalized`: {metrics['mean_wasserstein_normalized']:.4f}",
        f"- `mean_ks_statistic`: {metrics['mean_ks_statistic']:.4f}",
        f"- `tstr_mae`: {metrics['tstr_mae']:.4f}",
        f"- `tstr_r2`: {metrics['tstr_r2']:.4f}",
        f"- `support_violation_rate_mean`: {metrics['support_violation_rate_mean']:.4f}",
        f"- `support_violation_rate_max`: {metrics['support_violation_rate_max']:.4f}",
        f"- `separability_auc_mean`: {auc_mean:.4f}",
        f"- `separability_gini_abs`: {gini_abs:.4f}",
        f"- `article_rule_compliance_mean`: {article_compliance['article_rule_compliance_mean']:.4f}",
        f"- `radiation_monotonicity`: {article_compliance['process_checks']['monotonicity_by_radiation']['compliance_rate']:.4f}",
        f"- `cem43_monotonicity`: {article_compliance['process_checks']['monotonicity_by_cem43']['compliance_rate']:.4f}",
        "",
        "## Produced files",
        "",
        "- `final_synthetic_dataset.csv`",
        "- `final_synthetic_dataset_full.csv`",
        "- `final_training_frame.csv`",
        "- `evaluation_metrics.json`",
        "- `final_article_rule_compliance.json`",
        "- `final_generation_metadata.json`",
        "",
        "This dataset is intended as the primary synthetic dataset for downstream ML/AI experiments in this project.",
    ]
    (outdir / "final_dataset_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"saved final dataset to: {outdir / 'final_synthetic_dataset.csv'}")
    print(f"saved final report to: {outdir / 'final_dataset_report.md'}")


if __name__ == "__main__":
    main()
