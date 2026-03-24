from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common_synthetic_metrics import (
    COL_RADIATION,
    COL_SURVIVAL,
    COL_TEMPERATURE,
    COL_TIME,
    MINIMAL_COLUMNS,
)
from generate_cell_level_article_guided_dataset import (
    CellLevelConfig,
    build_generation_artifacts,
    cem43_from_temp_time,
    save_outputs,
)
from validate_cell_level_datasets import evaluate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cell_level_rules() -> Dict:
    rules_path = PROJECT_ROOT / "knowledge_base" / "cell_level_rules.json"
    with rules_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_evidence_map() -> Dict[str, Dict]:
    evidence_df = pd.read_csv(PROJECT_ROOT / "literature" / "literature_evidence.csv")
    return {str(row["evidence_id"]): row.to_dict() for _, row in evidence_df.iterrows()}


def build_rule_traceability_artifacts() -> List[Path]:
    rules_payload = load_cell_level_rules()
    evidence_map = load_evidence_map()
    out_csv = PROJECT_ROOT / "knowledge_base" / "cell_level_rule_traceability.csv"
    out_md = PROJECT_ROOT / "knowledge_base" / "cell_level_rule_traceability.md"

    generator_use = {
        "CL1": "Monotone projection forces non-increasing survival along the radiation axis at each thermal condition.",
        "CL2": "Used as a scope rule: the active generator keeps the 41-43 C / 30-60 min sensitizing interpretation but does not invent extra covariates.",
        "CL3": "Monotone projection also preserves thermal ordering in the observed domain: 42C_45min >= 43C_45min >= 44C_30min.",
        "CL4": "Supports treating 44C_30min as the strongest thermal condition in the observed domain and prevents biologically softer 44 C outcomes.",
        "CL5": "High-combined-dose cells are capped to very low survival for radiation >= 6 Gy and CEM43 >= 45.",
    }
    validation_use = {
        "CL1": "Checked by `radiation_monotonicity_mean_rate` on grouped exact-design means.",
        "CL2": "Not validated as an independent hard check because it is mechanistic and partly latent; documented as scope and interpretation.",
        "CL3": "Checked by `thermal_monotonicity_mean_rate` on grouped exact-design means.",
        "CL4": "Interpreted through the thermal ordering check and design-point explanations for 44C_30min.",
        "CL5": "Checked by `high_combined_dose_low_survival_rate` on observable columns only.",
    }
    scope_note = {
        "CL1": "Cell-level hard process rule.",
        "CL2": "Cell-level mechanistic rule retained from in vitro DNA-repair literature.",
        "CL3": "Observed-domain ordering rule for this exact 15-point matrix.",
        "CL4": "Mechanistic temperature-shift rule restricted to the observed domain.",
        "CL5": "Project-level hard plausibility rule supported by the most damaging combinations in the observed matrix.",
    }

    rows: List[Dict] = []
    md_lines = [
        "# Cell-Level Rule Traceability",
        "",
        "This matrix links each active cell-level rule to literature evidence, generator behavior, and independent validation.",
        "",
        "| Rule | Type | Support Basis | Evidence | Generator Use | Independent Validation | Scope Note |",
        "|---|---|---|---|---|---|---|",
    ]

    for rule in rules_payload["rules"]:
        evidence_items = []
        evidence_summaries = []
        evidence_urls = []
        for evidence_id in rule.get("evidence_ids", []):
            evidence = evidence_map.get(evidence_id, {})
            pmid_value = evidence.get("pmid", "NA")
            if pd.isna(pmid_value):
                pmid = "NA"
            else:
                try:
                    pmid = str(int(float(pmid_value)))
                except (TypeError, ValueError):
                    pmid = str(pmid_value)
            year = evidence.get("year", "NA")
            evidence_items.append(f"{evidence_id} (PMID {pmid}, {year})")
            if evidence.get("main_finding"):
                evidence_summaries.append(f"{evidence_id}: {evidence['main_finding']}")
            if evidence.get("source_url"):
                evidence_urls.append(str(evidence["source_url"]))

        rows.append(
            {
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "rule_type": rule["type"],
                "support_basis": "; ".join(rule.get("support_basis", [])),
                "evidence_ids": "; ".join(rule.get("evidence_ids", [])),
                "evidence_refs": "; ".join(evidence_items),
                "generator_use": generator_use.get(rule["id"], ""),
                "validation_use": validation_use.get(rule["id"], ""),
                "scope_note": scope_note.get(rule["id"], ""),
                "evidence_summary": " | ".join(evidence_summaries),
                "source_urls": "; ".join(evidence_urls),
            }
        )

        md_lines.append(
            f"| {rule['id']} | {rule['type']} | {'; '.join(rule.get('support_basis', []))} | {'; '.join(evidence_items)} | "
            f"{generator_use.get(rule['id'], '')} | {validation_use.get(rule['id'], '')} | {scope_note.get(rule['id'], '')} |"
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return [out_csv, out_md]


def applicable_rule_ids(radiation: float, temperature: float, duration: float, cem43: float) -> List[str]:
    rule_ids = ["CL1", "CL3"]
    if radiation > 0.0 and 41.0 <= temperature <= 43.0 and 30.0 <= duration <= 60.0:
        rule_ids.append("CL2")
    if temperature >= 43.0:
        rule_ids.append("CL4")
    if radiation >= 6.0 and cem43 >= 45.0:
        rule_ids.append("CL5")
    return sorted(set(rule_ids))


def design_point_explanation(radiation: float, temperature: float, duration: float, cem43: float) -> str:
    parts = [
        "CL1: при фиксированном терморежиме выживаемость не должна расти с дозой радиации.",
    ]

    if radiation > 0.0 and temperature == 42.0 and duration == 45.0:
        parts.append("CL2: 42 C, 45 мин остается внутри sensitizing window и может усиливать радиочувствительность без выхода в более жесткий тепловой режим.")

    if temperature == 42.0 and duration == 45.0:
        parts.append("CL3: это самый мягкий тепловой режим в наблюдаемом дизайне и он должен давать наибольшую выживаемость среди трех терморежимов.")
    if temperature == 43.0 and duration == 45.0:
        parts.append("CL2: режим попадает в окно 41-43 C и 30-60 мин, где по статьям ожидается подавление репарации ДНК и радиосенсибилизация.")
        parts.append("CL3/CL4: режим должен лежать между 42C_45min и 44C_30min по степени повреждения.")
    if temperature == 44.0 and duration == 30.0:
        parts.append("CL4: температура выше 43 C смещает механизм к более выраженному direct heat kill.")
        parts.append("CL3: в наблюдаемом домене этот режим не должен выглядеть мягче, чем 43C_45min.")
    if radiation >= 6.0 and cem43 >= 45.0:
        parts.append("CL5: высокая комбинированная доза требует очень низкой выживаемости.")
    if radiation == 0.0:
        parts.append("Без радиации различия между точками определяются главным образом тепловым порядком режимов.")

    return " ".join(parts)


def build_design_point_explainability(independent: Dict) -> List[Path]:
    rules_payload = load_cell_level_rules()
    evidence_by_rule = {rule["id"]: rule.get("evidence_ids", []) for rule in rules_payload["rules"]}

    grouped = pd.DataFrame(independent["grouped_design_summary"]).copy()
    if grouped.empty:
        return []

    grouped["CEM43"] = [cem43_from_temp_time(t, tm) for t, tm in zip(grouped[COL_TEMPERATURE], grouped[COL_TIME])]
    grouped["Примененные_правила"] = [
        "; ".join(applicable_rule_ids(r, t, tm, cem43))
        for r, t, tm, cem43 in zip(grouped[COL_RADIATION], grouped[COL_TEMPERATURE], grouped[COL_TIME], grouped["CEM43"])
    ]
    grouped["Evidence_IDs"] = [
        "; ".join(
            sorted(
                {
                    evidence_id
                    for rule_id in applicable_rule_ids(r, t, tm, cem43)
                    for evidence_id in evidence_by_rule.get(rule_id, [])
                }
            )
        )
        for r, t, tm, cem43 in zip(grouped[COL_RADIATION], grouped[COL_TEMPERATURE], grouped[COL_TIME], grouped["CEM43"])
    ]
    grouped["Объяснение"] = [
        design_point_explanation(r, t, tm, cem43)
        for r, t, tm, cem43 in zip(grouped[COL_RADIATION], grouped[COL_TEMPERATURE], grouped[COL_TIME], grouped["CEM43"])
    ]

    ordered_columns = [
        COL_RADIATION,
        COL_TEMPERATURE,
        COL_TIME,
        "CEM43",
        "real_survival",
        "synthetic_mean",
        "synthetic_median",
        "synthetic_count",
        "abs_mean_error",
        "abs_median_error",
        "Примененные_правила",
        "Evidence_IDs",
        "Объяснение",
    ]
    grouped = grouped[ordered_columns].sort_values([COL_RADIATION, COL_TEMPERATURE, COL_TIME]).reset_index(drop=True)

    out_csv = PROJECT_ROOT / "synthetic_data_cell_level_final" / "design_point_rule_explanations.csv"
    out_md = PROJECT_ROOT / "synthetic_data_cell_level_final" / "design_point_rule_explanations.md"
    grouped.to_csv(out_csv, index=False, encoding="utf-8-sig")

    md_lines = [
        "# Design-Point Rule Explanations",
        "",
        "Each row summarizes how the final synthetic dataset behaves at one of the 15 observed experimental design points and which cell-level rules justify that behavior.",
        "",
    ]

    for _, row in grouped.iterrows():
        md_lines.extend(
            [
                f"## {int(row[COL_RADIATION])} Gy, {int(row[COL_TEMPERATURE])} C, {int(row[COL_TIME])} min",
                "",
                f"- `real_survival`: {row['real_survival']:.6f}",
                f"- `synthetic_mean`: {row['synthetic_mean']:.6f}",
                f"- `synthetic_median`: {row['synthetic_median']:.6f}",
                f"- `abs_mean_error`: {row['abs_mean_error']:.6f}",
                f"- `rules`: {row['Примененные_правила']}",
                f"- `evidence_ids`: {row['Evidence_IDs']}",
                f"- explanation: {row['Объяснение']}",
                "",
            ]
        )

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return [out_csv, out_md]


def main() -> None:
    cfg = CellLevelConfig()
    real_df, synthetic_df, metadata = build_generation_artifacts(cfg)
    outdir = save_outputs(real_df, synthetic_df, metadata, cfg)

    independent = evaluate_dataset(synthetic_df[MINIMAL_COLUMNS].copy(), "cell_level_article_guided_final")
    with (outdir / "independent_cell_level_validation.json").open("w", encoding="utf-8") as fh:
        json.dump(independent, fh, ensure_ascii=False, indent=2)

    traceability_files = build_rule_traceability_artifacts()
    explainability_files = build_design_point_explainability(independent)

    with (outdir / "evaluation_metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    report_lines = [
        "# Final Cell-Level Article-Guided Synthetic Dataset",
        "",
        "This dataset is the primary synthetic dataset for the 15-point cell survival experiment.",
        "",
        "## Design Choice",
        "",
        "- exact experimental support is preserved: synthetic rows stay on the same 15 design points as the observed experiment;",
        "- variation is generated only in survival, not by inventing unsupported treatment conditions;",
        "- each generated 5x3 survival block is projected onto a monotone surface across radiation and thermal intensity;",
        "- the generator is calibrated back to the observed design-point matrix after projection, which reduces local bias while preserving the cell-level rules;",
        "- only cell-level rules from the literature are active in the default pipeline; in vivo and clinical rules remain in the broader knowledge base but are not used here.",
        "",
        "## Statistical Metrics",
        "",
        f"- `mean_wasserstein_normalized`: {metrics['mean_wasserstein_normalized']:.4f}",
        f"- `mean_ks_statistic`: {metrics['mean_ks_statistic']:.4f}",
        f"- `pearson_correlation_mean_abs_diff`: {metrics['pearson_correlation_mean_abs_diff']:.4f}",
        f"- `spearman_correlation_mean_abs_diff`: {metrics['spearman_correlation_mean_abs_diff']:.4f}",
        f"- `tstr_mae`: {metrics['tstr_mae']:.4f}",
        f"- `tstr_r2`: {metrics['tstr_r2']:.4f}",
        f"- `support_violation_rate_mean`: {metrics['support_violation_rate_mean']:.4f}",
        f"- `duplicate_rate_vs_real`: {metrics['duplicate_rate_vs_real']:.4f}",
        f"- `separability_auc_mean`: {metrics['separability_auc_mean']:.4f}",
        f"- `separability_gini_abs_mean`: {metrics['separability_gini_abs_mean']:.4f}",
        "",
        "## Independent Cell-Level Validation",
        "",
        f"- `exact_design_support_rate`: {independent['exact_design_support_rate']:.4f}",
        f"- `local_mean_abs_error`: {independent['local_mean_abs_error']:.4f}",
        f"- `local_max_abs_error`: {independent['local_max_abs_error']:.4f}",
        f"- `radiation_monotonicity_mean_rate`: {independent['radiation_monotonicity_mean_rate']:.4f}",
        f"- `thermal_monotonicity_mean_rate`: {independent['thermal_monotonicity_mean_rate']:.4f}",
        f"- `high_combined_dose_low_survival_rate`: {independent['high_combined_dose_low_survival_rate']:.4f}",
        f"- `independent_article_compliance_mean`: {independent['independent_article_compliance_mean']:.4f}",
        "",
        "## Explainability Artifacts",
        "",
        "- `knowledge_base/cell_level_rule_traceability.csv`",
        "- `knowledge_base/cell_level_rule_traceability.md`",
        "- `synthetic_data_cell_level_final/design_point_rule_explanations.csv`",
        "- `synthetic_data_cell_level_final/design_point_rule_explanations.md`",
        "- `synthetic_data_cell_level_final/block_explainability_log.csv` (§8, §12)",
        "- `synthetic_data_cell_level_final/block_explainability_summary.json` (§12)",
        "- `synthetic_data_cell_level_final/explainability_plots/` (§14)",
        "",
        "## Files",
        "",
        "- `final_synthetic_dataset.csv`",
        "- `final_synthetic_dataset_full.csv`",
        "- `real_design_points.csv`",
        "- `generation_metadata.json`",
        "- `evaluation_metrics.json`",
        "- `independent_cell_level_validation.json`",
        "",
        "The dataset is suitable for downstream ML/AI experiments that should remain faithful to the original experimental design and to cell-level rules extracted from the literature.",
    ]
    (outdir / "final_dataset_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    produced_files = [
        outdir / "final_synthetic_dataset.csv",
        outdir / "final_synthetic_dataset_full.csv",
        outdir / "real_design_points.csv",
        outdir / "generation_metadata.json",
        outdir / "evaluation_metrics.json",
        outdir / "independent_cell_level_validation.json",
        outdir / "final_dataset_report.md",
        outdir / "block_explainability_log.csv",
        outdir / "block_explainability_summary.json",
        *list((outdir / "explainability_plots").glob("*.png")),
        *traceability_files,
        *explainability_files,
    ]
    print("saved files:")
    for path in produced_files:
        print(path)


if __name__ == "__main__":
    main()
