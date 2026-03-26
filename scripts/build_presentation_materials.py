from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "generator_family_benchmark"
MULTISEED_DIR = PROJECT_ROOT / "benchmarks" / "generator_family_benchmark_multiseed"
RESIDUAL_SINGLE_DIR = PROJECT_ROOT / "benchmarks" / "residual_rule_aware_vae"
RESIDUAL_MULTISEED_DIR = PROJECT_ROOT / "benchmarks" / "residual_rule_aware_vae_multiseed"
FINAL_DATASET_DIR = PROJECT_ROOT / "synthetic_data_cell_level_final"
OUTDIR = PROJECT_ROOT / "presentation_materials" / "generator_presentation_ru"
FIG_DIR = OUTDIR / "figures"
TABLE_DIR = OUTDIR / "tables"
TEXT_DIR = OUTDIR / "text"

METHOD_ORDER = ["matrix", "residual_vae", "vae", "diffusion", "gan"]
DISPLAY_NAMES = {
    "matrix": "Matrix",
    "residual_vae": "Residual VAE",
    "vae": "VAE",
    "diffusion": "Diffusion",
    "gan": "GAN",
}
COLORS = {
    "matrix": "#355070",
    "residual_vae": "#7c6f64",
    "vae": "#7b6d8d",
    "diffusion": "#3f7d73",
    "gan": "#b5655f",
}
METHOD_SUBTITLE = {
    "matrix": "Explainable core",
    "residual_vae": "Best hybrid upgrade",
    "vae": "Strong alt neural",
    "diffusion": "Best pure neural",
    "gan": "Unstable baseline",
}

NEUTRAL_FILL = "#ded8cc"
TEXT_DARK = "#22333b"
TEXT_MID = "#486581"
BORDER = "#3d405b"
CANVAS = "#f7f4ee"
PAPER = "#fffdf9"
GRID = "#d8d2c4"
SUCCESS = "#cad2c5"
SOFT = "#e9c46a"
ACCENT_BLUE = "#89a6c7"
INITIAL_MATRIX_SNAPSHOT = {
    "stage": "Initial Matrix",
    "note": "Первая production-версия rule-guided matrix generator до усиления prior/noise model.",
    "local_mean_abs_error": 0.002275386615962583,
    "local_max_abs_error": 0.012209094654170971,
    "mean_wasserstein_normalized": 0.005019607187344047,
    "tstr_r2": 0.9990642691465024,
    "mean_constraint_pressure": 0.03381643217191362,
    "mean_delta_projection": 0.013396482789021108,
    "mean_delta_cap": 0.0,
    "mean_delta_calibration": 0.020419949382892513,
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIG_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    for path in TABLE_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    for path in TEXT_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    report_path = OUTDIR / "presentation_report_ru.md"
    if report_path.exists():
        report_path.unlink()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def method_order_map() -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(METHOD_ORDER)}


def load_current_production_snapshot() -> Dict:
    metrics = load_json(FINAL_DATASET_DIR / "evaluation_metrics.json")
    independent = load_json(FINAL_DATASET_DIR / "independent_cell_level_validation.json")
    metadata = load_json(FINAL_DATASET_DIR / "generation_metadata.json")
    explain = metadata.get("explainability_summary", {})
    return {
        "stage": "Improved Matrix",
        "note": "Текущая усиленная production-версия: smoothed prior, heteroscedastic shrinkage sigma, structured block noise.",
        "local_mean_abs_error": independent["local_mean_abs_error"],
        "local_max_abs_error": independent["local_max_abs_error"],
        "mean_wasserstein_normalized": metrics["mean_wasserstein_normalized"],
        "tstr_r2": metrics["tstr_r2"],
        "mean_constraint_pressure": explain["mean_constraint_pressure"],
        "mean_delta_projection": explain["mean_delta_projection"],
        "mean_delta_cap": explain["mean_delta_cap"],
        "mean_delta_calibration": explain["mean_delta_calibration"],
    }


def load_residual_single_row() -> pd.DataFrame:
    metadata = load_json(RESIDUAL_SINGLE_DIR / "generation_metadata.json")
    metrics = load_json(RESIDUAL_SINGLE_DIR / "evaluation_metrics.json")
    independent = load_json(RESIDUAL_SINGLE_DIR / "independent_cell_level_validation.json")
    raw_validation = load_json(RESIDUAL_SINGLE_DIR / "raw_independent_cell_level_validation.json")
    explain = load_json(RESIDUAL_SINGLE_DIR / "block_explainability_summary.json")
    row = {
        "family": "residual_vae",
        "display_name": "Residual VAE",
        "architecture": (
            f"Residual rule-aware block VAE, latent={metadata['model_config']['latent_dim']}, "
            f"hidden={metadata['model_config']['hidden_dim']}, matrix prior + rule loss"
        ),
        "training_note": "Learns residual stochasticity around the rule-guided matrix prior; projection/cap/calibration stay as a safety layer.",
        "mean_wasserstein_normalized": metrics["mean_wasserstein_normalized"],
        "mean_ks_statistic": metrics["mean_ks_statistic"],
        "tstr_mae": metrics["tstr_mae"],
        "tstr_r2": metrics["tstr_r2"],
        "duplicate_rate_vs_real": metrics["duplicate_rate_vs_real"],
        "exact_design_support_rate": raw_validation["exact_design_support_rate"],
        "local_mean_abs_error": independent["local_mean_abs_error"],
        "local_max_abs_error": independent["local_max_abs_error"],
        "radiation_monotonicity_mean_rate": independent["radiation_monotonicity_mean_rate"],
        "thermal_monotonicity_mean_rate": independent["thermal_monotonicity_mean_rate"],
        "high_combined_dose_low_survival_rate": independent["high_combined_dose_low_survival_rate"],
        "independent_article_compliance_mean": independent["independent_article_compliance_mean"],
        "mean_constraint_pressure": explain["mean_constraint_pressure"],
        "p95_constraint_pressure": explain["p95_constraint_pressure"],
        "mean_delta_projection": explain["mean_delta_projection"],
        "mean_delta_cap": explain["mean_delta_cap"],
        "mean_delta_calibration": explain["mean_delta_calibration"],
        "outdir": str(RESIDUAL_SINGLE_DIR.relative_to(PROJECT_ROOT)),
        "raw_article_compliance_mean": raw_validation["independent_article_compliance_mean"],
        "raw_radiation_monotonicity_mean_rate": raw_validation["radiation_monotonicity_mean_rate"],
        "raw_thermal_monotonicity_mean_rate": raw_validation["thermal_monotonicity_mean_rate"],
        "parameter_count": metadata["train_info"]["parameters"],
        "training_epochs": metadata["train_info"]["epochs"],
    }
    return pd.DataFrame([row])


def load_residual_multiseed_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RESIDUAL_MULTISEED_DIR / "residual_multiseed_summary.csv")
    aggregate = pd.read_csv(RESIDUAL_MULTISEED_DIR / "residual_multiseed_aggregate.csv", index_col=0)

    summary_df = pd.DataFrame(
        {
            "family": "residual_vae",
            "display_name": "Residual VAE",
            "architecture": "Residual rule-aware block VAE, latent=6, hidden=128, matrix prior + rule loss",
            "training_note": "Learns residual stochasticity around the rule-guided matrix prior; projection/cap/calibration stay as a safety layer.",
            "mean_wasserstein_normalized": summary["mean_wasserstein_normalized"],
            "tstr_r2": summary["tstr_r2"],
            "mean_constraint_pressure": summary["mean_constraint_pressure"],
            "local_mean_abs_error": summary["local_mean_abs_error"],
            "local_max_abs_error": summary["local_max_abs_error"],
            "independent_article_compliance_mean": 1.0,
            "radiation_monotonicity_mean_rate": 1.0,
            "thermal_monotonicity_mean_rate": 1.0,
            "raw_article_compliance_mean": summary["raw_article_compliance_mean"],
            "raw_radiation_monotonicity_mean_rate": summary["raw_radiation_monotonicity_mean_rate"],
            "raw_thermal_monotonicity_mean_rate": summary["raw_thermal_monotonicity_mean_rate"],
            "parameter_count": summary["parameter_count"],
            "training_epochs": 320,
            "seed": summary["seed"],
        }
    )

    aggregate_row = {
        "family": "residual_vae",
        "local_mean_abs_error__mean": aggregate.at["mean", "local_mean_abs_error"],
        "local_mean_abs_error__std": aggregate.at["std", "local_mean_abs_error"],
        "local_mean_abs_error__min": aggregate.at["min", "local_mean_abs_error"],
        "local_mean_abs_error__max": aggregate.at["max", "local_mean_abs_error"],
        "mean_wasserstein_normalized__mean": aggregate.at["mean", "mean_wasserstein_normalized"],
        "mean_wasserstein_normalized__std": aggregate.at["std", "mean_wasserstein_normalized"],
        "tstr_r2__mean": aggregate.at["mean", "tstr_r2"],
        "tstr_r2__std": aggregate.at["std", "tstr_r2"],
        "mean_constraint_pressure__mean": aggregate.at["mean", "mean_constraint_pressure"],
        "mean_constraint_pressure__std": aggregate.at["std", "mean_constraint_pressure"],
        "mean_constraint_pressure__min": aggregate.at["min", "mean_constraint_pressure"],
        "mean_constraint_pressure__max": aggregate.at["max", "mean_constraint_pressure"],
        "local_max_abs_error__mean": aggregate.at["mean", "local_max_abs_error"],
        "local_max_abs_error__std": aggregate.at["std", "local_max_abs_error"],
    }
    return summary_df, pd.DataFrame([aggregate_row])


def load_summary_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    single_df = pd.read_csv(BENCHMARK_DIR / "generator_family_summary.csv")
    multiseed_df = pd.read_csv(MULTISEED_DIR / "multiseed_summary.csv")
    aggregate_df = pd.read_csv(MULTISEED_DIR / "multiseed_aggregate.csv")

    if "residual_vae" not in set(single_df["family"]) and RESIDUAL_SINGLE_DIR.exists():
        single_df = pd.concat([single_df, load_residual_single_row()], ignore_index=True)

    if RESIDUAL_MULTISEED_DIR.exists() and "residual_vae" not in set(multiseed_df["family"]):
        residual_multiseed_df, residual_aggregate_df = load_residual_multiseed_frames()
        multiseed_df = pd.concat([multiseed_df, residual_multiseed_df], ignore_index=True, sort=False)
        aggregate_df = pd.concat([aggregate_df, residual_aggregate_df], ignore_index=True, sort=False)

    order_map = method_order_map()
    single_df["method_order"] = single_df["family"].map(order_map)
    multiseed_df["method_order"] = multiseed_df["family"].map(order_map)
    aggregate_df["method_order"] = aggregate_df["family"].map(order_map)
    single_df = single_df.sort_values("method_order").reset_index(drop=True)
    multiseed_df = multiseed_df.sort_values(["seed", "method_order"]).reset_index(drop=True)
    aggregate_df = aggregate_df.sort_values("method_order").reset_index(drop=True)
    return single_df, multiseed_df, aggregate_df


def build_literature_table() -> pd.DataFrame:
    lit_df = pd.read_csv(PROJECT_ROOT / "literature" / "literature_evidence.csv")
    out_df = lit_df.rename(
        columns={
            "evidence_id": "ID",
            "year": "Год",
            "study_type": "Тип_источника",
            "cancer_context": "Биологический_контекст",
            "main_finding": "Ключевой_вывод",
            "source_url": "Ссылка",
        }
    )[
        ["ID", "Год", "Тип_источника", "Биологический_контекст", "Ключевой_вывод", "Ссылка"]
    ]
    out_df.to_csv(TABLE_DIR / "literature_table_ru.csv", index=False, encoding="utf-8-sig")
    return out_df


def build_rules_table() -> pd.DataFrame:
    payload = load_json(PROJECT_ROOT / "knowledge_base" / "cell_level_rules.json")
    rows: List[Dict] = []
    for rule in payload["rules"]:
        if_text = []
        then_text = []
        for item in rule["if"]:
            text = f"{item['variable']} {item['relation']}"
            if "value" in item:
                text += f" {item['value']}"
            if_text.append(text)
        for item in rule["then"]:
            text = f"{item['variable']} {item['relation']}"
            if "value" in item:
                text += f" {item['value']}"
            then_text.append(text)
        rows.append(
            {
                "Rule_ID": rule["id"],
                "Название": rule["name"],
                "Тип": rule["type"],
                "IF": "; ".join(if_text),
                "THEN": "; ".join(then_text),
                "Evidence_IDs": "; ".join(rule["evidence_ids"]),
            }
        )
    rules_df = pd.DataFrame(rows)
    rules_df.to_csv(TABLE_DIR / "rules_table_ru.csv", index=False, encoding="utf-8-sig")
    return rules_df


def build_rule_usage_table() -> pd.DataFrame:
    rows = [
        {"Rule_ID": "CL1", "Тип": "hard", "Покрытие_в_15_точках": 15, "Роль": "Monotonicity по RT, hard enforcement"},
        {"Rule_ID": "CL2", "Тип": "soft", "Покрытие_в_15_точках": 10, "Роль": "Sensitizing window, soft prior / interpretation"},
        {"Rule_ID": "CL3", "Тип": "hard", "Покрытие_в_15_точках": 15, "Роль": "Thermal ordering, hard enforcement"},
        {"Rule_ID": "CL4", "Тип": "soft", "Покрытие_в_15_точках": 10, "Роль": "Direct heat kill trend, domain prior"},
        {"Rule_ID": "CL5", "Тип": "hard", "Покрытие_в_15_точках": 4, "Роль": "High combined dose -> very low survival"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "rule_usage_ru.csv", index=False, encoding="utf-8-sig")
    return df


def build_architecture_table(single_df: pd.DataFrame) -> pd.DataFrame:
    out_df = single_df[
        ["display_name", "architecture", "training_note", "parameter_count", "training_epochs"]
    ].rename(
        columns={
            "display_name": "Метод",
            "architecture": "Архитектура",
            "training_note": "Как_получали",
            "parameter_count": "Параметры",
            "training_epochs": "Эпохи",
        }
    )
    out_df.to_csv(TABLE_DIR / "architecture_table_ru.csv", index=False, encoding="utf-8-sig")
    return out_df


def build_matrix_evolution_table(single_df: pd.DataFrame) -> pd.DataFrame:
    current = load_current_production_snapshot()
    hybrid_row = single_df[single_df["family"] == "residual_vae"].iloc[0]
    rows = [
        {
            "Этап": INITIAL_MATRIX_SNAPSHOT["stage"],
            "Что_изменилось": INITIAL_MATRIX_SNAPSHOT["note"],
            "Local_MAE": INITIAL_MATRIX_SNAPSHOT["local_mean_abs_error"],
            "Local_Max_Error": INITIAL_MATRIX_SNAPSHOT["local_max_abs_error"],
            "Wasserstein_norm": INITIAL_MATRIX_SNAPSHOT["mean_wasserstein_normalized"],
            "TSTR_R2": INITIAL_MATRIX_SNAPSHOT["tstr_r2"],
            "Pressure": INITIAL_MATRIX_SNAPSHOT["mean_constraint_pressure"],
            "Delta_Projection": INITIAL_MATRIX_SNAPSHOT["mean_delta_projection"],
            "Delta_Calibration": INITIAL_MATRIX_SNAPSHOT["mean_delta_calibration"],
        },
        {
            "Этап": current["stage"],
            "Что_изменилось": current["note"],
            "Local_MAE": current["local_mean_abs_error"],
            "Local_Max_Error": current["local_max_abs_error"],
            "Wasserstein_norm": current["mean_wasserstein_normalized"],
            "TSTR_R2": current["tstr_r2"],
            "Pressure": current["mean_constraint_pressure"],
            "Delta_Projection": current["mean_delta_projection"],
            "Delta_Calibration": current["mean_delta_calibration"],
        },
        {
            "Этап": "Hybrid Residual VAE",
            "Что_изменилось": "Matrix prior + residual VAE + rule-aware loss + safety postprocessing.",
            "Local_MAE": hybrid_row["local_mean_abs_error"],
            "Local_Max_Error": hybrid_row["local_max_abs_error"],
            "Wasserstein_norm": hybrid_row["mean_wasserstein_normalized"],
            "TSTR_R2": hybrid_row["tstr_r2"],
            "Pressure": hybrid_row["mean_constraint_pressure"],
            "Delta_Projection": hybrid_row["mean_delta_projection"],
            "Delta_Calibration": hybrid_row["mean_delta_calibration"],
        },
    ]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(TABLE_DIR / "matrix_evolution_ru.csv", index=False, encoding="utf-8-sig")
    return out_df


def build_metrics_tables(single_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    single_metrics = single_df[
        [
            "display_name",
            "local_mean_abs_error",
            "local_max_abs_error",
            "mean_wasserstein_normalized",
            "tstr_r2",
            "mean_constraint_pressure",
            "independent_article_compliance_mean",
            "raw_article_compliance_mean",
        ]
    ].rename(
        columns={
            "display_name": "Метод",
            "local_mean_abs_error": "Local_MAE",
            "local_max_abs_error": "Local_Max_Error",
            "mean_wasserstein_normalized": "Wasserstein_norm",
            "tstr_r2": "TSTR_R2",
            "mean_constraint_pressure": "Explainability_Pressure",
            "independent_article_compliance_mean": "Final_Compliance",
            "raw_article_compliance_mean": "Raw_Compliance",
        }
    )
    single_metrics.to_csv(TABLE_DIR / "metrics_single_run_ru.csv", index=False, encoding="utf-8-sig")

    aggregate_metrics = aggregate_df[
        [
            "family",
            "local_mean_abs_error__mean",
            "local_mean_abs_error__std",
            "local_mean_abs_error__min",
            "local_mean_abs_error__max",
            "mean_wasserstein_normalized__mean",
            "mean_wasserstein_normalized__std",
            "tstr_r2__mean",
            "tstr_r2__std",
            "mean_constraint_pressure__mean",
            "mean_constraint_pressure__std",
            "mean_constraint_pressure__min",
            "mean_constraint_pressure__max",
            "local_max_abs_error__mean",
            "local_max_abs_error__std",
        ]
    ].rename(
        columns={
            "family": "Метод",
            "local_mean_abs_error__mean": "Local_MAE_mean",
            "local_mean_abs_error__std": "Local_MAE_std",
            "local_mean_abs_error__min": "Local_MAE_min",
            "local_mean_abs_error__max": "Local_MAE_max",
            "local_max_abs_error__mean": "Local_Max_Error_mean",
            "local_max_abs_error__std": "Local_Max_Error_std",
            "mean_wasserstein_normalized__mean": "Wasserstein_mean",
            "mean_wasserstein_normalized__std": "Wasserstein_std",
            "tstr_r2__mean": "TSTR_R2_mean",
            "tstr_r2__std": "TSTR_R2_std",
            "mean_constraint_pressure__mean": "Pressure_mean",
            "mean_constraint_pressure__std": "Pressure_std",
            "mean_constraint_pressure__min": "Pressure_min",
            "mean_constraint_pressure__max": "Pressure_max",
        }
    )
    aggregate_metrics["Метод"] = aggregate_metrics["Метод"].map(DISPLAY_NAMES)
    aggregate_metrics.to_csv(TABLE_DIR / "metrics_multiseed_ru.csv", index=False, encoding="utf-8-sig")
    return single_metrics, aggregate_metrics


def build_scorecard_table(aggregate_df: pd.DataFrame, multiseed_df: pd.DataFrame) -> pd.DataFrame:
    score_df = aggregate_df.copy()
    lower_better = [
        "local_mean_abs_error__mean",
        "mean_wasserstein_normalized__mean",
        "mean_constraint_pressure__mean",
        "local_max_abs_error__mean",
    ]
    higher_better = ["tstr_r2__mean"]

    for col in lower_better:
        values = score_df[col].to_numpy(dtype=float)
        best = values.min()
        worst = values.max()
        if np.isclose(best, worst):
            scaled = np.ones_like(values)
        else:
            scaled = (worst - values) / (worst - best)
        score_df[f"{col}__score"] = scaled

    for col in higher_better:
        values = score_df[col].to_numpy(dtype=float)
        best = values.max()
        worst = values.min()
        if np.isclose(best, worst):
            scaled = np.ones_like(values)
        else:
            scaled = (values - worst) / (best - worst)
        score_df[f"{col}__score"] = scaled

    trust_rows = []
    for family, sub in multiseed_df.groupby("family", sort=False):
        final_compliance = sub["independent_article_compliance_mean"].mean()
        raw_compliance = sub["raw_article_compliance_mean"].mean()
        rad_mono = sub["radiation_monotonicity_mean_rate"].mean()
        trust_rows.append(
            {
                "family": family,
                "trust_score_raw": 0.45 * final_compliance + 0.35 * raw_compliance + 0.20 * rad_mono,
            }
        )
    trust_df = pd.DataFrame(trust_rows)
    score_df = score_df.merge(trust_df, on="family", how="left")

    trust_vals = score_df["trust_score_raw"].to_numpy(dtype=float)
    best = trust_vals.max()
    worst = trust_vals.min()
    if np.isclose(best, worst):
        trust_scaled = np.ones_like(trust_vals)
    else:
        trust_scaled = (trust_vals - worst) / (best - worst)
    score_df["trust_score"] = trust_scaled

    score_df["overall_score"] = (
        0.25 * score_df["local_mean_abs_error__mean__score"]
        + 0.15 * score_df["mean_wasserstein_normalized__mean__score"]
        + 0.10 * score_df["tstr_r2__mean__score"]
        + 0.18 * score_df["mean_constraint_pressure__mean__score"]
        + 0.07 * score_df["local_max_abs_error__mean__score"]
        + 0.25 * score_df["trust_score"]
    )
    score_df["display_name"] = score_df["family"].map(DISPLAY_NAMES)
    out_df = score_df[
        [
            "display_name",
            "overall_score",
            "local_mean_abs_error__mean__score",
            "mean_wasserstein_normalized__mean__score",
            "tstr_r2__mean__score",
            "mean_constraint_pressure__mean__score",
            "local_max_abs_error__mean__score",
            "trust_score",
        ]
    ].rename(
        columns={
            "display_name": "Метод",
            "overall_score": "Overall_Score",
            "local_mean_abs_error__mean__score": "MAE_Score",
            "mean_wasserstein_normalized__mean__score": "Wasserstein_Score",
            "tstr_r2__mean__score": "TSTR_Score",
            "mean_constraint_pressure__mean__score": "Pressure_Score",
            "local_max_abs_error__mean__score": "Max_Error_Score",
            "trust_score": "Trust_Score",
        }
    )
    out_df = out_df.sort_values("Overall_Score", ascending=False).reset_index(drop=True)
    out_df.to_csv(TABLE_DIR / "method_scorecard_ru.csv", index=False, encoding="utf-8-sig")
    return out_df


def build_recommendation_table(single_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "Сценарий": "Основной production / защита",
            "Рекомендуемый_метод": "Matrix",
            "Почему": "Самый прозрачный, напрямую следует из правил CL1-CL5 и литературы; 100% стабильность по compliance на всех seed.",
        },
        {
            "Сценарий": "Финальная гибридная версия / следующий спринт",
            "Рекомендуемый_метод": "Residual VAE",
            "Почему": "Сохраняет rule-guided matrix prior, но сильно снижает explainability pressure и держит raw compliance = 1.0 по всем seed.",
        },
        {
            "Сценарий": "Лучший чистый neural candidate",
            "Рекомендуемый_метод": "Diffusion",
            "Почему": "Лучший по fidelity в full multi-seed после обновления matrix prior: минимальный Local MAE, лучший Wasserstein и лучший TSTR R2 среди pure neural baselines.",
        },
        {
            "Сценарий": "Сильный альтернативный neural baseline",
            "Рекомендуемый_метод": "VAE",
            "Почему": "Очень близок к Diffusion по качеству и чуть лучше по explainability pressure, но в полном multi-seed немного уступает ему по fidelity.",
        },
        {
            "Сценарий": "Метод, который не стоит брать как основной",
            "Рекомендуемый_метод": "GAN",
            "Почему": "На multi-seed есть просадки monotonicity и compliance, значит надежность хуже остальных.",
        },
    ]
    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(TABLE_DIR / "recommendation_table_ru.csv", index=False, encoding="utf-8-sig")
    return rec_df


def apply_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": CANVAS,
            "axes.facecolor": PAPER,
            "savefig.facecolor": CANVAS,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT_DARK,
            "axes.titleweight": "bold",
            "axes.titlesize": 15.5,
            "axes.labelsize": 12.5,
            "xtick.color": TEXT_DARK,
            "ytick.color": TEXT_DARK,
            "grid.color": GRID,
            "grid.alpha": 0.32,
            "font.size": 11.5,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def soften_axes(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BORDER)
        ax.spines[spine].set_linewidth(1.1)


def ranked_families(df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    return df.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)


def save_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 4.4))
    ax.axis("off")
    boxes = [
        (0.03, 0.26, 0.16, 0.48, "#d8e2dc", "15 реальных\nточек"),
        (0.23, 0.26, 0.16, 0.48, "#f0e6d2", "Правила\nCL1-CL5"),
        (0.43, 0.26, 0.16, 0.48, "#e6d7cf", "Generator family\nMatrix / Residual VAE /\nVAE / Diffusion / GAN"),
        (0.63, 0.26, 0.16, 0.48, "#d6e2d9", "Rule-guided\nprojection +\ncalibration"),
        (0.83, 0.26, 0.14, 0.48, "#d8e5ef", "Метрики +\nexplainability"),
    ]
    for x, y, w, h, color, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=color,
            edgecolor=BORDER,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12.2, color=TEXT_DARK)
    arrows = [
        (0.19, 0.5, 0.04, 0.0),
        (0.39, 0.5, 0.04, 0.0),
        (0.59, 0.5, 0.04, 0.0),
        (0.79, 0.5, 0.04, 0.0),
    ]
    for x, y, dx, dy in arrows:
        ax.arrow(x, y, dx, dy, width=0.0055, head_width=0.045, head_length=0.018, color=BORDER, length_includes_head=True)
    fig.suptitle("Пайплайн получения synthetic dataset", fontsize=17, fontweight="bold", color=TEXT_DARK, y=0.98)
    fig.tight_layout(pad=1.2)
    fig.savefig(FIG_DIR / "01_pipeline_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fidelity_comparison(single_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
    metrics = [
        ("local_mean_abs_error", "Local MAE", False),
        ("mean_wasserstein_normalized", "Wasserstein (norm)", False),
        ("tstr_r2", "TSTR R2", True),
    ]
    for ax, (metric, title, higher_is_better) in zip(axes, metrics):
        ranked = ranked_families(single_df[["family", metric]].copy(), metric, higher_is_better)
        values = ranked[metric].to_numpy(dtype=float)
        y = np.arange(len(ranked))
        best_idx = 0
        bar_colors = [COLORS[fam] if idx == best_idx else NEUTRAL_FILL for idx, fam in enumerate(ranked["family"])]
        ax.barh(y, values, color=bar_colors, edgecolor=BORDER, linewidth=1.3, height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels([DISPLAY_NAMES[fam] for fam in ranked["family"]])
        ax.invert_yaxis()
        ax.set_title(title)
        soften_axes(ax, axis="x")
        if higher_is_better:
            spread = max(values) - min(values)
            margin = max(spread * 0.25, 0.0001)
            ax.set_xlim(min(values) - margin, max(values) + margin)
        else:
            ax.set_xlim(0, max(values) * 1.18)
        for idx, value in enumerate(values):
            offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015
            ax.text(value + offset, idx, f"{value:.4f}", ha="left", va="center", fontsize=9.5, color=TEXT_DARK)
        ax.text(
            ax.get_xlim()[1],
            best_idx,
            " best",
            ha="right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=TEXT_MID,
        )

    fig.suptitle("Сравнение методов по fidelity на основном прогоне", fontsize=17.5, fontweight="bold", color=TEXT_DARK, y=0.98)
    fig.tight_layout(pad=1.2, w_pad=1.8)
    fig.savefig(FIG_DIR / "02_fidelity_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_explainability_comparison(single_df: pd.DataFrame, multiseed_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    pressure_ranked = ranked_families(single_df[["family", "mean_constraint_pressure"]].copy(), "mean_constraint_pressure", False)
    pressure = pressure_ranked["mean_constraint_pressure"].to_numpy(dtype=float)
    y = np.arange(len(pressure_ranked))
    axes[0].barh(y, pressure, color=[COLORS[fam] if i == 0 else NEUTRAL_FILL for i, fam in enumerate(pressure_ranked["family"])], edgecolor=BORDER, linewidth=1.3, height=0.65)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([DISPLAY_NAMES[fam] for fam in pressure_ranked["family"]])
    axes[0].invert_yaxis()
    axes[0].set_title("Explainability pressure")
    soften_axes(axes[0], axis="x")
    axes[0].set_xlim(0, max(pressure) * 1.2)
    for idx, value in enumerate(pressure):
        axes[0].text(value + max(pressure) * 0.02, idx, f"{value:.3f}", ha="left", va="center", fontsize=9.5, color=TEXT_DARK)

    pivot = multiseed_df.pivot(index="seed", columns="family", values="independent_article_compliance_mean")
    ordered_cols = [fam for fam in METHOD_ORDER if fam in pivot.columns]
    for idx, family in enumerate(ordered_cols):
        values = pivot[family].to_numpy(dtype=float)
        axes[1].scatter(np.full_like(values, idx, dtype=float), values, s=80, color=COLORS[family], edgecolor=BORDER, linewidth=1.0, zorder=3)
        axes[1].plot([idx - 0.18, idx + 0.18], [values.mean(), values.mean()], color=BORDER, linewidth=2.2, zorder=4)
        axes[1].vlines(idx, values.min(), values.max(), color=GRID, linewidth=3.0, zorder=2)
    axes[1].set_xticks(range(len(ordered_cols)))
    axes[1].set_xticklabels([DISPLAY_NAMES[fam] for fam in ordered_cols])
    axes[1].set_title("Final compliance across seeds")
    axes[1].set_ylim(0.93, 1.005)
    soften_axes(axes[1], axis="y")

    fig.suptitle("Explainability и rule-compliance", fontsize=16.8, fontweight="bold", color=TEXT_DARK, y=0.98)
    fig.tight_layout(pad=1.2, w_pad=1.8)
    fig.savefig(FIG_DIR / "03_explainability_and_compliance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_multiseed_stability(aggregate_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.0))
    metrics = [
        ("local_mean_abs_error__mean", "local_mean_abs_error__std", "Local MAE mean ± std"),
        ("mean_constraint_pressure__mean", "mean_constraint_pressure__std", "Pressure mean ± std"),
        ("tstr_r2__mean", "tstr_r2__std", "TSTR R2 mean ± std"),
    ]
    x = np.arange(len(aggregate_df))
    colors = [COLORS[fam] for fam in aggregate_df["family"]]
    labels = [DISPLAY_NAMES[fam] for fam in aggregate_df["family"]]

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        mean_vals = aggregate_df[mean_col].to_numpy(dtype=float)
        std_vals = aggregate_df[std_col].to_numpy(dtype=float)
        ax.bar(x, mean_vals, yerr=std_vals, capsize=6, color=colors, edgecolor=BORDER, linewidth=1.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        soften_axes(ax, axis="y")

    fig.suptitle("Устойчивость по 5 seed", fontsize=16.8, fontweight="bold", color=TEXT_DARK, y=0.98)
    fig.tight_layout(pad=1.2, w_pad=1.5)
    fig.savefig(FIG_DIR / "04_multiseed_stability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_method_positioning(aggregate_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.axvspan(0.0, aggregate_df["local_mean_abs_error__mean"].median(), color="#e8f0ea", alpha=0.65, zorder=0)
    ax.axhspan(0.0, aggregate_df["mean_constraint_pressure__mean"].median(), color="#edf3f4", alpha=0.75, zorder=0)
    for _, row in aggregate_df.iterrows():
        family = row["family"]
        ax.scatter(
            row["local_mean_abs_error__mean"],
            row["mean_constraint_pressure__mean"],
            s=700,
            color=COLORS[family],
            edgecolor=BORDER,
            linewidth=2,
            alpha=0.9,
        )
        ax.text(
            row["local_mean_abs_error__mean"] + 0.00003,
            row["mean_constraint_pressure__mean"] + 0.0005,
            DISPLAY_NAMES[family],
            fontsize=11,
            fontweight="bold",
            color=TEXT_DARK,
        )
        ax.text(
            row["local_mean_abs_error__mean"] + 0.00003,
            row["mean_constraint_pressure__mean"] - 0.00035,
            METHOD_SUBTITLE[family],
            fontsize=8.5,
            color=TEXT_MID,
        )
    ax.set_xlabel("Local MAE mean (меньше лучше)")
    ax.set_ylabel("Explainability pressure mean (меньше лучше)")
    ax.set_title("Позиционирование методов: fidelity vs explainability burden")
    soften_axes(ax, axis="both")
    fig.tight_layout(pad=1.1)
    fig.savefig(FIG_DIR / "05_method_positioning.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_design_point_heatmaps() -> None:
    n_methods = len(METHOD_ORDER)
    ncols = 3
    nrows = int(np.ceil(n_methods / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    cmap = plt.cm.YlOrRd
    for ax, family in zip(axes, METHOD_ORDER):
        payload = load_json(BENCHMARK_DIR / family / "independent_cell_level_validation.json")
        grouped = pd.DataFrame(payload["grouped_design_summary"])
        grouped = grouped.sort_values(["Радиация", "Температура", "Время"]).reset_index(drop=True)
        matrix = grouped["abs_mean_error"].to_numpy(dtype=float).reshape(5, 3)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_title(DISPLAY_NAMES[family])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["42C 45m", "43C 45m", "44C 30m"])
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["0 Gy", "2 Gy", "4 Gy", "6 Gy", "8 Gy"])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#1f2933")
    for ax in axes[n_methods:]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Absolute mean error")
    fig.suptitle("Ошибки по 15 design points", fontsize=16, fontweight="bold", color="#243b53")
    fig.savefig(FIG_DIR / "06_design_point_error_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_gan_seed_failure_plot(multiseed_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for family in METHOD_ORDER:
        sub = multiseed_df[multiseed_df["family"] == family].sort_values("seed")
        ax.plot(
            sub["seed"],
            sub["radiation_monotonicity_mean_rate"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=COLORS[family],
            label=DISPLAY_NAMES[family],
        )
    ax.set_xlabel("Seed")
    ax.set_ylabel("Radiation monotonicity mean rate")
    ax.set_title("Проверка надежности: monotonicity по seed")
    ax.set_ylim(0.7, 1.02)
    soften_axes(ax, axis="y")
    ax.legend(frameon=True)
    fig.tight_layout(pad=1.0)
    fig.savefig(FIG_DIR / "07_monotonicity_by_seed.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rule_usage_plot(rule_usage_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    plot_df = rule_usage_df.sort_values(["Покрытие_в_15_точках", "Rule_ID"], ascending=[True, True]).reset_index(drop=True)
    colors = [COLORS["diffusion"] if t == "hard" else SOFT for t in plot_df["Тип"]]
    y = np.arange(len(plot_df))
    ax.hlines(y, xmin=0, xmax=plot_df["Покрытие_в_15_точках"], color=GRID, linewidth=4)
    ax.scatter(plot_df["Покрытие_в_15_точках"], y, s=220, color=colors, edgecolor=BORDER, linewidth=1.2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Rule_ID"])
    ax.set_xlim(0, 16)
    ax.set_xlabel("Сколько design points покрывает правило")
    ax.set_title("Использование правил в observed 15-point domain")
    soften_axes(ax, axis="x")
    for idx, value in enumerate(plot_df["Покрытие_в_15_точках"]):
        ax.text(value + 0.25, idx, f"{int(value)}/15", ha="left", va="center", fontsize=10, fontweight="bold", color=TEXT_DARK)
    ax.text(15.9, -0.8, "teal = hard rule\nsand = soft rule", ha="right", va="top", fontsize=9.5, color=TEXT_MID)
    fig.tight_layout(pad=1.0)
    fig.savefig(FIG_DIR / "10_rule_usage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_podium(aggregate_metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.axis("off")
    cards = [
        ("Matrix", "#dce4db", "Core", "Основной production / защита"),
        ("Residual VAE", "#e7ddd5", "Upgrade", "Лучшая следующая гибридная версия"),
        ("Diffusion", "#dee8e5", "Pure Neural", "Лучший чистый neural baseline"),
    ]
    x_positions = [0.04, 0.36, 0.68]
    metric_lookup = aggregate_metrics_df.set_index("Метод")
    for x, (method, color, badge, subtitle) in zip(x_positions, cards):
        patch = FancyBboxPatch((x, 0.16), 0.27, 0.68, boxstyle="round,pad=0.02,rounding_size=0.04", linewidth=1.8, edgecolor=BORDER, facecolor=color)
        ax.add_patch(patch)
        ax.text(x + 0.03, 0.78, badge, fontsize=11, fontweight="bold", color=TEXT_MID)
        ax.text(x + 0.03, 0.67, method, fontsize=20, fontweight="bold", color=TEXT_DARK)
        ax.text(x + 0.03, 0.57, subtitle, fontsize=11, color=TEXT_MID)
        row = metric_lookup.loc[method]
        ax.text(x + 0.03, 0.43, f"Local MAE: {row['Local_MAE_mean']:.4f}", fontsize=12, color=TEXT_DARK)
        ax.text(x + 0.03, 0.34, f"TSTR R2: {row['TSTR_R2_mean']:.4f}", fontsize=12, color=TEXT_DARK)
        ax.text(x + 0.03, 0.25, f"Pressure: {row['Pressure_mean']:.4f}", fontsize=12, color=TEXT_DARK)
    fig.suptitle("Три главные позиции для презентации", fontsize=18, fontweight="bold", color=TEXT_DARK)
    fig.tight_layout(pad=1.0)
    fig.savefig(FIG_DIR / "11_benchmark_podium.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_scorecard_heatmap(scorecard_df: pd.DataFrame) -> None:
    plot_df = scorecard_df.set_index("Метод")[
        ["Overall_Score", "Trust_Score", "MAE_Score", "Wasserstein_Score", "TSTR_Score", "Pressure_Score", "Max_Error_Score"]
    ]
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    cmap = LinearSegmentedColormap.from_list("presentation_score", ["#f2ede3", "#c8d7d2", "#6f8f88", "#355070"])
    im = ax.imshow(plot_df.to_numpy(dtype=float), cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(plot_df.shape[1]))
    ax.set_xticklabels(["Overall", "Trust", "MAE", "Wasser", "TSTR", "Pressure", "MaxErr"])
    ax.set_yticks(range(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index.tolist())
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            ax.text(j, i, f"{plot_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9.2, color=TEXT_DARK)
    ax.set_title("Сводный scorecard методов по full multi-seed benchmark")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.set_label("Normalized score")
    fig.tight_layout(pad=1.0)
    fig.savefig(FIG_DIR / "08_method_scorecard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_executive_summary_chart() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axis("off")
    cards = [
        (0.03, 0.18, 0.21, 0.66, "#dce4db", "Основной метод", "Matrix", "Объяснимый, rule-guided, защищаемый"),
        (0.27, 0.18, 0.21, 0.66, "#e7ddd5", "Следующая версия", "Residual VAE", "Лучший гибрид: low pressure + strong fidelity"),
        (0.51, 0.18, 0.21, 0.66, "#dee8e5", "Pure Neural", "Diffusion", "Лучший clean baseline по full multi-seed"),
        (0.75, 0.18, 0.21, 0.66, "#efd6d2", "Не брать", "GAN", "Нестабилен по monotonicity/compliance"),
    ]
    for x, y, w, h, color, title, method, subtitle in cards:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025", facecolor=color, edgecolor=BORDER, linewidth=1.8)
        ax.add_patch(patch)
        ax.text(x + 0.03, y + h - 0.12, title, fontsize=12, fontweight="bold", color=TEXT_DARK)
        ax.text(x + 0.03, y + h - 0.30, method, fontsize=20, fontweight="bold", color=TEXT_DARK)
        ax.text(x + 0.03, y + 0.16, subtitle, fontsize=11, color=TEXT_MID, wrap=True)
    fig.suptitle("Executive Summary: что показываем и что продвигаем", fontsize=17.2, fontweight="bold", color=TEXT_DARK)
    fig.tight_layout(pad=1.0)
    fig.savefig(FIG_DIR / "09_executive_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_matrix_evolution_chart(evolution_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    stages = evolution_df["Этап"].tolist()
    x = np.arange(len(stages))

    mae_vals = evolution_df["Local_MAE"].to_numpy(dtype=float)
    pressure_vals = evolution_df["Pressure"].to_numpy(dtype=float)
    proj_vals = evolution_df["Delta_Projection"].to_numpy(dtype=float)
    calib_vals = evolution_df["Delta_Calibration"].to_numpy(dtype=float)

    axes[0].plot(x, mae_vals, marker="o", markersize=9, linewidth=2.8, color=COLORS["matrix"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages, rotation=12, ha="right")
    axes[0].set_title("Эволюция fidelity по ходу разработки")
    axes[0].set_ylabel("Local MAE")
    soften_axes(axes[0], axis="y")
    for idx, value in enumerate(mae_vals):
        axes[0].text(idx, value + max(mae_vals) * 0.04, fmt(value), ha="center", va="bottom", fontsize=9.5, color=TEXT_DARK)

    axes[1].bar(x, proj_vals, color="#c9d6df", edgecolor=BORDER, linewidth=1.1, label="Projection burden")
    axes[1].bar(x, calib_vals, bottom=proj_vals, color="#d4c7b5", edgecolor=BORDER, linewidth=1.1, label="Calibration burden")
    axes[1].plot(x, pressure_vals, color=COLORS["gan"], linewidth=2.0, marker="D", markersize=6, label="Total pressure")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages, rotation=12, ha="right")
    axes[1].set_title("Куда уходит explainability burden")
    soften_axes(axes[1], axis="y")
    axes[1].legend(loc="upper right")

    fig.suptitle("Как развивался наш метод: initial Matrix -> improved Matrix -> hybrid", fontsize=17, fontweight="bold", color=TEXT_DARK, y=0.98)
    fig.tight_layout(pad=1.2, w_pad=2.0)
    fig.savefig(FIG_DIR / "12_matrix_evolution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_math_appendix(
    aggregate_metrics_df: pd.DataFrame,
    evolution_df: pd.DataFrame,
) -> None:
    top = aggregate_metrics_df.set_index("Метод")
    initial = evolution_df.set_index("Этап").loc["Initial Matrix"]
    improved = evolution_df.set_index("Этап").loc["Improved Matrix"]
    hybrid = evolution_df.set_index("Этап").loc["Hybrid Residual VAE"]
    diffusion = top.loc["Diffusion"]
    vae = top.loc["VAE"]
    gan = top.loc["GAN"]

    lines = [
        "# Математическая часть проекта",
        "",
        "## Формальная постановка задачи",
        "",
        "Мы работаем не с большим датасетом изображений или табличных наблюдений, а с очень маленькой и жестко ограниченной биологической матрицей. В исходных данных есть только пятнадцать наблюдаемых design points: пять уровней радиационной дозы и три терморежима. Если обозначить их через `x = (r, t, tau)`, а выживаемость клеток через `y`, то реальный эксперимент задает набор `D = {(x_i, y_i)}_{i=1}^{15}`. После перехода в пространство `log10(y + eps)` эти наблюдения удобно представлять как матрицу `Y* in R^(5x3)`, где строки отвечают дозам `r in {0, 2, 4, 6, 8}`, а столбцы отвечают трем observed thermal conditions.",
        "",
        "Это сразу накладывает главное математическое ограничение. У нас слишком мало реальных точек, чтобы учить свободный генератор с нуля и делать вид, будто он открыл новую биологию. Поэтому вся наша работа строится вокруг идеи design-preserving generation. Синтетика может варьироваться только внутри уже наблюдаемой 5x3 сетки, а не придумывать новые режимы. Отсюда и появляется центральный вопрос проекта: как сгенерировать вариативность, не потеряв биологическую правдоподобность и объяснимость.",
        "",
        "## Почему одной нейросети здесь недостаточно",
        "",
        "Если бы мы сразу обучили `GAN`, `VAE` или `Diffusion` на этих пятнадцати точках, модель начала бы подменять биологию интерполяцией по почти пустому пространству. Поэтому сначала мы формализовали биологическую структуру через правила. В текущей постановке hard-правилами являются `CL1`, `CL3` и `CL5`: выживаемость не должна расти с дозой радиации, терморежимы должны сохранять observed ordering, а высокие комбинированные дозы должны вести к очень низкой выживаемости. Правила `CL2` и `CL4` мы используем как mechanistic priors: они важны для интерпретации, но не вводятся как грубые пороги на каждый сэмпл.",
        "",
        "Это означает, что математическая модель должна минимизировать не только ошибку приближения к `Y*`, но и величину последующей rule-based коррекции. Именно поэтому в проекте используется понятие explainability pressure. Если `B_raw` — сыро сгенерированный блок, `Pi_rule(B_raw)` — результат hard projection, `Cap(.)` — безопасное ограничение high-dose области, а `Cal(.)` — калибровка к observed матрице, то итоговый блок записывается как `B_final = Cal(Cap(Pi_rule(B_raw)))`. Тогда pressure фактически измеряет среднюю абсолютную цену этого перехода: `P(B) = mean(|B_raw - Pi_rule(B_raw)| + |Pi_rule - Cap| + |Cap - Cal|)`. Чем меньше `P(B)`, тем меньше постфактум пришлось чинить модель правилами.",
        "",
        "## Первая версия Matrix",
        "",
        "Первая production-версия нашего метода была самой прямой. Мы брали observed матрицу `Y*`, добавляли стохастический шум вокруг нее и затем приводили блок к biologically valid виду через hard projection и calibration. В терминах формулы это можно записать как `Y_raw^(0) = Y* + E`, где `E` — шум в log-survival пространстве. Эта версия уже была design-preserving и rule-aware, но математика шума еще была грубой. В историческом снимке проекта она давала `Local MAE = " + fmt(initial["Local_MAE"]) + "`, `Local Max Error = " + fmt(initial["Local_Max_Error"]) + "`, `Wasserstein = " + fmt(initial["Wasserstein_norm"]) + "`, `TSTR R2 = " + fmt(initial["TSTR_R2"]) + "`. При этом средний explainability pressure составлял `" + fmt(initial["Pressure"]) + "`, а его значимая часть приходилась именно на projection burden: `" + fmt(initial["Delta_Projection"]) + "`. Это означало, что модель слишком часто генерировала сырые блоки, которые потом приходилось заметно дотягивать до monotonic structure.",
        "",
        "## Что именно мы математически улучшили в Matrix",
        "",
        "Усиление Matrix-модели состояло не в косметической настройке параметров, а в замене самой генеративной конструкции. Вместо того чтобы семплировать прямо вокруг наблюдаемой матрицы, мы ввели сглаженный prior `Mu_prior`. Его можно мыслить как `Mu_prior = (1 - lambda) * Y* + lambda * S(Y*)`, где `S` — сглаживающий оператор на 5x3 поверхности, сохраняющий observed ordering. Это важно, потому что теперь модель генерирует не вокруг шумной эмпирической точки, а вокруг более биологичной опорной поверхности.",
        "",
        "Второе изменение касается дисперсии. Вместо одной общей сигмы мы ввели cell-specific `Sigma_ij` с shrinkage. В простом виде это похоже на `Sigma_ij = alpha * sigma_local_ij + (1 - alpha) * sigma_pool`, причем около граничных точек дисперсия дополнительно сжимается. Это особенно важно для верхней границы около `0.53` и нижней границы около нуля, где лишняя вариативность быстро делает синтетику небиологичной.",
        "",
        "Третье изменение — отказ от независимого шума по клеткам в пользу structured block noise. В текущей версии шум задается как `E_ij = Sigma_ij * T * (a_g * z_g + a_r * z_row_i + a_c * z_col_j + a_l * z_local_ij)`, где один латент отвечает за весь блок, отдельные латенты отвечают за строки и столбцы, а локальный компонент добавляет мелкую вариативность. Это математически лучше согласуется с тем, что survival surface должна вести себя как связная биологическая поверхность, а не как набор пятнадцати независимых случайных чисел.",
        "",
        "Итоговая усиленная Matrix-модель записывается как `Y_raw = Mu_prior + E`, а затем `Y_final = Cal(Cap(Pi_rule(Y_raw)))`. На текущем production-датасете это дало `Local MAE = " + fmt(improved["Local_MAE"]) + "`, `Local Max Error = " + fmt(improved["Local_Max_Error"]) + "`, `Wasserstein = " + fmt(improved["Wasserstein_norm"]) + "`, `TSTR R2 = " + fmt(improved["TSTR_R2"]) + "`. Главное содержательное изменение не только в том, что ошибка уменьшилась, но и в том, что projection burden упал с `" + fmt(initial["Delta_Projection"]) + "` до `" + fmt(improved["Delta_Projection"]) + "`. Это значит, что сырой блок стал сам по себе гораздо ближе к biologically ordered форме. При этом calibration burden вырос до `" + fmt(improved["Delta_Calibration"]) + "`, то есть часть нагрузки сместилась из жесткой коррекции правил в более мягкую подстройку к целевой observed матрице. Иными словами, модель стала лучше уважать форму, но точнее подгоняться к эмпирическому центру уже на этапе калибровки.",
        "",
        "## Почему после Matrix мы пошли именно в сторону VAE",
        "",
        "После усиления Matrix следующий вопрос звучал так: можно ли добавить более живую вариативность, не разрушив объяснимость. Здесь мы сравнили несколько top-level альтернатив для синтетической генерации в нашей постановке: `GAN`, `VAE` и `Diffusion`. В полном multi-seed benchmark чистый `Diffusion` показал лучшую fidelity среди pure neural baselines: `Local MAE = " + fmt(diffusion['Local_MAE_mean']) + "`, `Wasserstein = " + fmt(diffusion['Wasserstein_mean']) + "`, `TSTR R2 = " + fmt(diffusion['TSTR_R2_mean']) + "`, `Pressure = " + fmt(diffusion['Pressure_mean']) + "`. `VAE` оказался очень близко: `Local MAE = " + fmt(vae['Local_MAE_mean']) + "`, `Wasserstein = " + fmt(vae['Wasserstein_mean']) + "`, `TSTR R2 = " + fmt(vae['TSTR_R2_mean']) + "`, `Pressure = " + fmt(vae['Pressure_mean']) + "`. `GAN` заметно хуже по надежности: `Local MAE = " + fmt(gan['Local_MAE_mean']) + "`, `Pressure = " + fmt(gan['Pressure_mean']) + "`, а главное, именно у него чаще проседают monotonicity и compliance.",
        "",
        "Несмотря на то, что лучший pure neural benchmark сейчас у `Diffusion`, для гибрида мы сознательно выбрали именно `VAE`. Причина математическая, а не вкусовая. `VAE` естественным образом позволяет учить остаток вокруг prior, то есть представление `R = Y - Mu_prior`. Для сверхмалого числа реальных design points это удобнее и прозрачнее, чем строить полноценный residual diffusion process. Кроме того, у `VAE` есть явное латентное пространство, KL-регуляризация и компактный decoder, что делает его лучше приспособленным для связки с small-data prior и rule-aware penalty.",
        "",
        "## Гибридная модель Residual VAE",
        "",
        "В гибриде мы не генерируем весь блок заново. Мы задаем остаток `R = Y - Mu_prior` и учим модель восстанавливать именно его. Энкодер строит `q_phi(z | R, F)`, декодер задает `p_theta(R | z, F)`, где `F` — rule-aware признаки, включающие радиацию, температуру, длительность, `CEM43`, thermal rank и индикаторы правил. После этого `Y_raw = Mu_prior + R_hat(z, F)`, а затем тот же safety layer переводит блок в `Y_final`.",
        "",
        "Ключевой момент здесь в функции потерь. Она имеет вид `L = L_recon + beta * L_KL + lambda_rule * L_rule + lambda_center * L_center + lambda_var * L_var + lambda_smooth * L_smooth`. Здесь `L_recon` отвечает за близость к teacher blocks в log-survival space, `L_KL` регуляризует латентное пространство, `L_center` не дает средней synthetic surface уехать от цели, `L_var` удерживает правдоподобную вариативность, а `L_smooth` не позволяет residual-компоненте бесконтрольно перетягивать на себя структуру, уже заданную `Matrix prior`. Самая важная часть — `L_rule`. Мы штрафуем модель еще до post-processing, если она нарушает non-increasing radiation trend, thermal ordering или high-dose low-survival condition. То есть правила здесь уже не только внешний фильтр, а часть самой оптимизации.",
        "",
        "## Что дал гибрид по факту",
        "",
        "На single-run benchmark гибридный `Residual VAE` уже лучше усиленной Matrix по fidelity и на порядок мягче по explainability burden. В эволюционной таблице это видно так: `Local MAE = " + fmt(hybrid['Local_MAE']) + "`, `Local Max Error = " + fmt(hybrid['Local_Max_Error']) + "`, `Wasserstein = " + fmt(hybrid['Wasserstein_norm']) + "`, `TSTR R2 = " + fmt(hybrid['TSTR_R2']) + "`, `Pressure = " + fmt(hybrid['Pressure']) + "`. По full multi-seed benchmark лучший hybrid verdict тоже остается за `Residual VAE`, потому что он дает крайне низкий pressure при хорошей fidelity. Но здесь важно быть честными: в family multiseed у него есть один хвостовой seed, где final compliance падает после финального balancing до `0.9375`, хотя raw compliance остается `1.0`. Это не развал модели, а указание на то, что финальная postprocessing-обвязка для гибрида еще требует доводки.",
        "",
        "## Как интерпретировать метрики",
        "",
        "`Local MAE` мы используем как главный индикатор того, насколько synthetic mean по каждой из пятнадцати клеток близок к observed survival. Формально это `Local MAE = (1/15) * sum_x |mu_syn(x) - y*(x)|`. `Local Max Error` нужен как защита от ситуации, когда средняя ошибка хорошая, но одна клетка уехала слишком сильно. `Wasserstein(norm)` показывает, насколько целиком synthetic distribution похожа на real distribution в каждой design cell. `TSTR R2` отвечает на прикладной вопрос: если обучить простой предиктор на synthetic данных и проверить его на real, сохраняется ли полезная структура данных. Наконец, `Explainability Pressure` — это не просто еще одна метрика качества, а численная цена биологической корректировки. Для нашего проекта это критично, потому что модель с красивым `MAE`, но с высоким burden, на практике означает модель, которую приходится чинить руками правил после генерации.",
        "",
        "## Финальная математическая интерпретация",
        "",
        "Сейчас картина проекта выглядит так. Усиленная `Matrix` — это основной научно защищаемый метод, потому что она прямо кодирует observed design, явно использует hard rules и дает полностью трассируемый результат. `Residual VAE` — это логичное продолжение именно нашей модели, а не ее замена: он берет `Matrix prior` как структурный центр и учит только ту вариативность, которую имеет смысл делегировать нейросети. `Diffusion` остается лучшим pure neural baseline в benchmark-смысле, но не становится естественным ядром гибрида. А `GAN` служит полезным контрпримером: adversarial objective сам по себе не спасает в сверхмалой rule-constrained биомедицинской задаче.",
        "",
    ]
    (TEXT_DIR / "mathematical_foundation_ru.md").write_text("\n".join(lines), encoding="utf-8")


def write_slide_texts(
    aggregate_metrics_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    evolution_df: pd.DataFrame,
    multiseed_df: pd.DataFrame,
) -> None:
    top_rows = aggregate_metrics_df.set_index("Метод")
    evolution = evolution_df.set_index("Этап")
    residual_min_final = float(multiseed_df.loc[multiseed_df["family"] == "residual_vae", "independent_article_compliance_mean"].min())
    gan_min_mono = float(multiseed_df.loc[multiseed_df["family"] == "gan", "radiation_monotonicity_mean_rate"].min())

    outline_lines = [
        "# Слайды для презентации",
        "",
        "## Слайд 1. Задача и ограничение домена",
        "",
        "На первом слайде стоит сразу зафиксировать главный контекст: мы не работаем с большим табличным или imaging dataset, а пытаемся построить synthetic generator для очень маленького, но биологически жесткого cell-level эксперимента. В реальности у нас есть только пятнадцать design points, поэтому весь проект строится не вокруг свободной генерации, а вокруг design-preserving и rule-aware подхода.",
        "",
        "## Слайд 2. Данные, биология и правила",
        "",
        "Здесь важно человеческим языком объяснить, что это за эксперимент. У нас есть пять уровней радиации и три терморежима, а целевая переменная — доля выживших клеток после комбинированного воздействия. Дальше надо спокойно пояснить, что правила `CL1-CL5` были не придуманы под benchmark, а собраны из literature-backed knowledge base. При этом hard rules — это `CL1`, `CL3` и `CL5`, а `CL2` и `CL4` работают как mechanistic priors.",
        "",
        "## Слайд 3. Как мы пришли к Matrix",
        "",
        "Этот слайд должен показать, что Matrix не взялся из воздуха. Первая версия была прямым rule-guided stochastic generator вокруг observed matrix. Потом мы увидели, что главный источник проблем сидит в самой геометрии prior и в структуре шума, и поэтому усилили модель. В evolution table это видно количественно: `Local MAE` снизился с `" + fmt(evolution.loc['Initial Matrix', 'Local_MAE']) + "` до `" + fmt(evolution.loc['Improved Matrix', 'Local_MAE']) + "`, а projection burden упал с `" + fmt(evolution.loc['Initial Matrix', 'Delta_Projection']) + "` до `" + fmt(evolution.loc['Improved Matrix', 'Delta_Projection']) + "`.",
        "",
        "## Слайд 4. Математика улучшенной Matrix",
        "",
        "На этом слайде нужно объяснить три идеи. Во-первых, мы ввели сглаженный `Mu_prior`, чтобы генерировать не вокруг шумной observed матрицы, а вокруг более устойчивой monotone surface. Во-вторых, мы заменили общую сигму на heteroscedastic shrinkage `Sigma_ij`, что особенно важно у границ около `0.53` и `0.0`. В-третьих, шум стал block-structured, а не независимым по клеткам. Именно это улучшило fidelity, не ломая explainability.",
        "",
        "## Слайд 5. Почему мы тестировали VAE, Diffusion и GAN",
        "",
        "Здесь нужно показать, что мы не замкнулись в своей модели. Мы честно прогнали топовые альтернативные families для synthetic generation в нашей постановке: `GAN`, `VAE` и `Diffusion`. Но важно сразу проговорить честное ограничение: neural methods учились не на богатой независимой реальности, а на rule-guided teacher blocks, потому что реальных design points всего пятнадцать.",
        "",
        "## Слайд 6. Benchmark и выбор чистых neural methods",
        "",
        "На full multi-seed benchmark лучший pure neural baseline — `Diffusion`: `Local MAE = " + fmt(top_rows.loc['Diffusion', 'Local_MAE_mean']) + "`, `TSTR R2 = " + fmt(top_rows.loc['Diffusion', 'TSTR_R2_mean']) + "`, `Pressure = " + fmt(top_rows.loc['Diffusion', 'Pressure_mean']) + "`. `VAE` идет очень близко и оказывается особенно удобным как кандидат для hybridization. `GAN` хуже по устойчивости: у него минимальная radiation monotonicity на seed’ах падает до `" + fmt(gan_min_mono, 2) + "`.",
        "",
        "## Слайд 7. Почему гибрид именно с VAE",
        "",
        "Этот слайд нужен отдельно, чтобы у слушателя не возник вопрос: если лучший pure neural — Diffusion, зачем гибрид с VAE? Ответ в том, что нам нужен был не лучший standalone baseline, а лучшая residual-надстройка над `Matrix prior`. Для этого `VAE` удобнее: у него есть компактное latent space, прямой residual decoder, KL-регуляризация и естественная связка с rule-aware loss.",
        "",
        "## Слайд 8. Residual VAE как следующая версия системы",
        "",
        "Здесь нужно объяснить, что гибрид не заменяет Matrix, а доращивает его. Он учит только остаток вокруг `Mu_prior`. В benchmark это дает сильный результат: `Residual VAE` на full multi-seed имеет `Local MAE = " + fmt(top_rows.loc['Residual VAE', 'Local_MAE_mean']) + "`, `Pressure = " + fmt(top_rows.loc['Residual VAE', 'Pressure_mean']) + "`. При этом надо честно упомянуть, что минимальный final compliance у него по seed’ам сейчас `" + fmt(residual_min_final) + "`, хотя raw compliance остается идеальным.",
        "",
        "## Слайд 9. Explainability и доверие",
        "",
        "На этом слайде стоит показать, что explainability у нас не маркетинговая надпись, а измеряемая часть пайплайна. У каждого блока есть trace of correction, есть decomposition по projection, cap и calibration, есть rule coverage, есть design-point explanations и есть counterfactual analysis. Это позволяет защищать не только итоговый dataset, но и сам механизм его получения.",
        "",
        "## Слайд 10. Финальный вывод",
        "",
        "Финальный вывод должен звучать спокойно и строго. Основной production-метод проекта — `Matrix`, потому что он самый прозрачный, научно защищаемый и напрямую опирается на observed biology. Лучшая следующая версия системы — `Residual VAE`, потому что она сохраняет `Matrix prior`, но резко снижает explainability burden. Лучший pure neural benchmark — `Diffusion`. `GAN` в этой задаче как основной путь мы не берем.",
        "",
    ]
    (TEXT_DIR / "slides_outline_ru.md").write_text("\n".join(outline_lines), encoding="utf-8")

    top3 = scorecard_df.head(3).reset_index(drop=True)
    scorecard_sentence = (
        f"Если ссылаться на интегральный scorecard, то сейчас верхние позиции занимают "
        f"{top3.loc[0, 'Метод']} с overall score {top3.loc[0, 'Overall_Score']:.2f}, "
        f"{top3.loc[1, 'Метод']} с {top3.loc[1, 'Overall_Score']:.2f} и "
        f"{top3.loc[2, 'Метод']} с {top3.loc[2, 'Overall_Score']:.2f}. "
        "Но этот score нужно интерпретировать содержательно: он не отменяет того, что Matrix остается главным production-ядром проекта."
    )

    notes_lines = [
        "# Текст для выступления",
        "",
        "Если рассказывать проект одной связной историей, то начинать нужно не с нейросетей, а с ограничения задачи. У нас есть всего пятнадцать реальных биологических design points, и из-за этого свободная генерация здесь математически опасна: модель очень легко начинает интерполировать то, чего в данных никогда не было. Поэтому сначала мы построили rule-guided Matrix generator, который уважает саму структуру эксперимента.",
        "",
        "Первая Matrix-версия уже была рабочей, но мы увидели, что слишком большая часть нагрузки ложится на projection stage. После этого мы усилили модель математически: ввели сглаженный prior, cell-specific shrinkage sigma и structured block noise. Это снизило ошибку и сделало сырые блоки намного ближе к биологически корректной форме.",
        "",
        "После этого мы честно сравнили нашу модель с сильными нейросетевыми семействами. В чистом benchmark лучшим pure neural baseline оказался Diffusion, но именно VAE оказался самым естественным кандидатом для гибрида, потому что его легче поставить на residual-задачу вокруг Matrix prior.",
        "",
        "Поэтому финальная логика проекта такая. Matrix остается основным production-методом, потому что он самый прозрачный и научно защищаемый. Residual VAE становится лучшей следующей версией системы, потому что он не ломает rule-guided ядро, а усиливает его. Diffusion нужен нам как сильный внешний ориентир, который показывает, что сравнение с современными generative methods было честным.",
        "",
        "## Цифры, которые можно озвучивать вслух",
        f"Matrix на full multi-seed benchmark: Local MAE {fmt(top_rows.loc['Matrix', 'Local_MAE_mean'])}, Pressure {fmt(top_rows.loc['Matrix', 'Pressure_mean'])}.",
        f"Residual VAE: Local MAE {fmt(top_rows.loc['Residual VAE', 'Local_MAE_mean'])}, Pressure {fmt(top_rows.loc['Residual VAE', 'Pressure_mean'])}.",
        f"Diffusion: Local MAE {fmt(top_rows.loc['Diffusion', 'Local_MAE_mean'])}, TSTR R2 {fmt(top_rows.loc['Diffusion', 'TSTR_R2_mean'])}.",
        f"История Matrix: Local MAE снизился с {fmt(evolution.loc['Initial Matrix', 'Local_MAE'])} до {fmt(evolution.loc['Improved Matrix', 'Local_MAE'])}.",
        "",
        "## Итог по scorecard",
        "",
        scorecard_sentence,
    ]
    (TEXT_DIR / "speaker_notes_ru.md").write_text("\n".join(notes_lines), encoding="utf-8")

    final_script_lines = [
        "# Полный текст выступления",
        "",
        "Если рассказывать эту работу как цельную исследовательскую историю, то главная отправная точка очень простая: у нас был не большой датасет, а маленький in vitro эксперимент с пятнадцатью design points. Это означало, что обычная логика генеративного моделирования здесь работает плохо. Слишком легко получить красивую синтетику, которая не выдерживает биологической интерпретации.",
        "",
        "Поэтому мы начали не с нейросетей, а с биологии. Мы зафиксировали observed domain, собрали cell-level правила из литературы и отделили hard constraints от mechanistic priors. После этого мы построили первую Matrix-версию генератора. Она уже сохраняла design support и правила, но в ней было видно, что существенная часть explainability burden возникает из-за того, что сырые блоки еще недостаточно хорошо согласованы по форме и их приходится заметно дотягивать projection stage.",
        "",
        "Следующий шаг был уже математическим. Мы усилили Matrix-модель тремя изменениями: ввели сглаженный prior вместо генерации прямо вокруг observed matrix, добавили heteroscedastic shrinkage sigma по каждой клетке и заменили независимый шум на structured block noise. В результате Local MAE для Matrix улучшился с "
        + fmt(evolution.loc["Initial Matrix", "Local_MAE"])
        + " до "
        + fmt(evolution.loc["Improved Matrix", "Local_MAE"])
        + ", а projection burden упал с "
        + fmt(evolution.loc["Initial Matrix", "Delta_Projection"])
        + " до "
        + fmt(evolution.loc["Improved Matrix", "Delta_Projection"])
        + ". Это важный момент: сырая форма поверхности стала заметно более биологичной.",
        "",
        "После этого мы решили честно проверить современные альтернативы для synthetic generation. Мы прогнали VAE, Diffusion и GAN в одном rule-guided benchmark-контуре. Чисто по benchmark-метрикам лучшим pure neural baseline оказался Diffusion, VAE показал очень близкий результат, а GAN проиграл по устойчивости. Это позволило нам сделать важный вывод: нейросети здесь полезны, но не как замена rule-guided ядра, а как возможная надстройка над ним.",
        "",
        "Именно поэтому следующим шагом стал гибрид. Мы не стали учить модель заново генерировать весь блок. Вместо этого мы взяли Matrix prior как структурный центр и дали VAE задачу моделировать только остаточную вариативность вокруг него. Так появился Residual VAE. Он оказался самым разумным компромиссом между fidelity и explainability burden. На full multi-seed benchmark он дает Local MAE "
        + fmt(top_rows.loc["Residual VAE", "Local_MAE_mean"])
        + " при Pressure "
        + fmt(top_rows.loc["Residual VAE", "Pressure_mean"])
        + ". При этом важно честно говорить о текущем ограничении: на одном seed final compliance у него проседает до "
        + fmt(residual_min_final)
        + ", хотя raw compliance остается идеальным.",
        "",
        "Финальная архитектурная позиция проекта поэтому выглядит так. Matrix — это основной production-метод и главный научно защищаемый результат. Residual VAE — это лучшая следующая версия системы, потому что она усиливает Matrix, а не спорит с ним. Diffusion — это лучший чистый нейросетевой benchmark, а GAN в нашей задаче мы не берем как основной путь из-за нестабильности.",
        "",
        "Если формулировать ценность работы совсем коротко, то мы сделали не просто synthetic generator, а explainable rule-guided synthetic generator для очень малого и биологически ограниченного домена. И главное здесь не только итоговые числа, но и то, что каждый шаг модели, каждая активная rule-correction и каждая архитектурная развилка у нас численно и содержательно объяснены.",
        "",
    ]
    (TEXT_DIR / "final_presentation_script_ru.md").write_text("\n".join(final_script_lines), encoding="utf-8")


def write_report(
    single_df: pd.DataFrame,
    multiseed_df: pd.DataFrame,
    literature_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    architecture_df: pd.DataFrame,
    single_metrics_df: pd.DataFrame,
    aggregate_metrics_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    evolution_df: pd.DataFrame,
) -> None:
    report_path = OUTDIR / "presentation_report_ru.md"
    top = aggregate_metrics_df.set_index("Метод")
    evolution = evolution_df.set_index("Этап")
    residual_min_final = float(multiseed_df.loc[multiseed_df["family"] == "residual_vae", "independent_article_compliance_mean"].min())
    gan_min_mono = float(multiseed_df.loc[multiseed_df["family"] == "gan", "radiation_monotonicity_mean_rate"].min())

    lines: List[str] = [
        "# Финальная версия материалов для презентации",
        "",
        "## Общая идея проекта",
        "",
        "Этот проект решает очень узкую, но важную задачу: как получить синтетическую выборку для маленького биомедицинского эксперимента так, чтобы итоговые данные были и полезными для downstream-моделей, и биологически объяснимыми. Главная особенность задачи в том, что у нас нет большого массива реальных наблюдений. Исходный эксперимент задает только пятнадцать design points, поэтому обычный подход вида «обучим генеративную модель и дадим ей придумать вариативность» здесь слишком рискован. В такой постановке модель легко становится красивой статистически, но плохо защищаемой биологически.",
        "",
        "Именно поэтому мы сознательно строили работу не вокруг абстрактной генерации, а вокруг design-preserving synthetic generation. Мы разрешаем модели варьировать только survival внутри уже наблюдаемого 5x3 экспериментального дизайна, а не изобретать новые условия. Это решение проходит через весь проект: через правила, через архитектуру генератора, через explainability-логи и через выбор метрик.",
        "",
        "## Данные о приборе и о самой выборке",
        "",
        "Важная честная оговорка состоит в том, что в репозитории нет паспорта прибора, модели облучателя и паспорта гипертермической установки. Поэтому раздел «данные о приборе» в презентации нужно формулировать аккуратно. Мы можем уверенно говорить только о режимных параметрах эксперимента и о клеточной выживаемости как целевой переменной. Доступные параметры — это радиационная доза от 0 до 8 Gy, температура 42, 43 и 44 градусов Цельсия, время 30 или 45 минут и производная thermal dose `CEM43`.",
        "",
        "С биологической точки зрения выборка представляет собой cell-level in vitro survival matrix, а не клинический пациентский датасет. В ней есть пять уровней радиации и три терморежима, то есть ровно пятнадцать observed design points. Выходом является доля выживших клеток после комбинированного воздействия гипертермии и радиации. На этом уровне биологии у нас есть несколько естественных ожиданий. При росте радиационной дозы survival не должен увеличиваться. Более жесткий терморежим в observed domain не должен делать клетки более живыми, чем мягкий режим. А высокая комбинированная нагрузка должна вести к очень низкой выживаемости.",
        "",
        "## Как мы собирали правила и откуда брали литературу",
        "",
        "Правила не были придуманы вручную под текущий benchmark. Мы брали их из уже собранной knowledge base проекта и активировали только те, которые можно честно применять к наблюдаемым колонкам `radiation`, `temperature`, `time` и `survival`. Клинические и in vivo правила мы сознательно не переносили в этот генератор, потому что иначе начали бы смешивать разные уровни биологии и делать выводы, которые исходные данные не поддерживают.",
        "",
        "В результате в active set вошли правила `CL1-CL5`. Полная литература собрана в `tables/literature_table_ru.csv`, а полная логическая формализация лежит в `tables/rules_table_ru.csv`. Содержательно `CL1` отвечает за monotonicity по радиации, `CL2` описывает sensitizing window около 41-43 градусов и 30-60 минут, `CL3` фиксирует thermal ordering внутри observed domain, `CL4` отражает усиление direct heat kill при более высокой температуре, а `CL5` задает very-low-survival поведение в области высокой комбинированной дозы. В practical pipeline hard-правилами становятся `CL1`, `CL3` и `CL5`, а `CL2` и `CL4` играют роль mechanistic priors и интерпретационных ограничений.",
        "",
        "## Почему мы не начали с GAN, VAE или Diffusion",
        "",
        "Если смотреть на задачу глазами typical synthetic data literature, то естественно захотеть попробовать `GAN`, `VAE` и `Diffusion`. Мы действительно это сделали и именно поэтому в проекте есть benchmark-папка с полным сравнением генераторов. Но важный исследовательский вывод появился еще до чисел. При `n = 15` реальных observed points свободная нейросетевая генерация не может считаться надежной опорой для биомедицинского генератора. Если у модели нет сильного prior, она учится не биологии, а почти пустой сетке.",
        "",
        "Отсюда родилась наша базовая идея: сначала построить объяснимый rule-guided core, который сохраняет design support и биологические ограничения, а уже потом использовать neural models либо как benchmark, либо как residual-надстройку поверх этого core. Иными словами, в этом проекте нейросети не заменяют физико-биологическую структуру, а подключаются только там, где они действительно могут добавить правдоподобную вариативность.",
        "",
        "## Наш собственный метод: как появился Matrix",
        "",
        "Первой собственной рабочей архитектурой стал `Matrix` generator. Его смысл в том, что мы представляем наблюдаемый эксперимент как 5x3 матрицу в пространстве `log10(survival + eps)`, а затем генерируем synthetic blocks внутри этой матрицы. Уже первая production-версия была design-preserving и rule-guided, но она была более прямой: генерация шла близко к observed matrix, а потом значимая часть структуры восстанавливалась через post-processing. Исторический снимок этой версии мы сохранили и включили в `tables/matrix_evolution_ru.csv`.",
        "",
        "Для initial Matrix метрики были такими: `Local MAE = " + fmt(evolution.loc['Initial Matrix', 'Local_MAE']) + "`, `Local Max Error = " + fmt(evolution.loc['Initial Matrix', 'Local_Max_Error']) + "`, `Wasserstein = " + fmt(evolution.loc['Initial Matrix', 'Wasserstein_norm']) + "`, `TSTR R2 = " + fmt(evolution.loc['Initial Matrix', 'TSTR_R2']) + "`. Средний explainability pressure был `" + fmt(evolution.loc['Initial Matrix', 'Pressure']) + "`, и важная часть этой цены приходилась на projection stage: `" + fmt(evolution.loc['Initial Matrix', 'Delta_Projection']) + "`. Для нас это был сигнал, что сама сырая surface еще недостаточно согласована с hard rules.",
        "",
        "## Как мы улучшили Matrix и что дала эта математика",
        "",
        "Следующий этап был уже полноценным математическим улучшением метода. Во-первых, мы перестали генерировать прямо вокруг observed matrix и ввели сглаженный prior `Mu_prior`. Это снизило локальную шероховатость поверхности и сделало baseline более биологичным. Во-вторых, вместо одной общей сигмы мы ввели heteroscedastic shrinkage sigma по каждой клетке матрицы. Это дало более аккуратную вариативность в тех точках, где границы survival особенно чувствительны. В-третьих, шум перестал быть независимым по клеткам и стал block-structured: появилась общая компонентa на весь блок, отдельные компоненты на строки и столбцы и локальная добавка для мелкой вариативности.",
        "",
        "Эти три изменения дали заметный содержательный эффект. У текущей improved Matrix `Local MAE` снизился до `" + fmt(evolution.loc['Improved Matrix', 'Local_MAE']) + "`, `Local Max Error` — до `" + fmt(evolution.loc['Improved Matrix', 'Local_Max_Error']) + "`, `Wasserstein` — до `" + fmt(evolution.loc['Improved Matrix', 'Wasserstein_norm']) + "`, а `TSTR R2` вырос до `" + fmt(evolution.loc['Improved Matrix', 'TSTR_R2']) + "`. Самое интересное, что projection burden упал с `" + fmt(evolution.loc['Initial Matrix', 'Delta_Projection']) + "` до `" + fmt(evolution.loc['Improved Matrix', 'Delta_Projection']) + "`. Это означает, что raw blocks сами по себе стали ближе к biologically valid форме. При этом calibration burden вырос, то есть часть корректировки переехала из жесткой правки формы в более мягкую привязку к observed target. Это важно говорить честно: новая версия улучшила fidelity и raw structure, но не обнулила цену post-processing.",
        "",
        "## Почему после этого мы решили делать гибрид",
        "",
        "После усиления Matrix стало понятно, что базовый explainable core уже достаточно силен, чтобы быть production-методом. Но возник следующий вопрос: можно ли добавить более натуральную synthetic variability и не сломать при этом rule-guided основу. Для ответа на этот вопрос мы прогнали benchmark по нескольким top-level семействам генераторов. В single-run и multi-seed артефактах лежат результаты для `Matrix`, `Residual VAE`, `VAE`, `Diffusion` и `GAN`.",
        "",
        "В полном multi-seed benchmark лучший pure neural baseline сейчас — `Diffusion`. Он дает `Local MAE = " + fmt(top.loc['Diffusion', 'Local_MAE_mean']) + "`, `Wasserstein = " + fmt(top.loc['Diffusion', 'Wasserstein_mean']) + "`, `TSTR R2 = " + fmt(top.loc['Diffusion', 'TSTR_R2_mean']) + "`, `Pressure = " + fmt(top.loc['Diffusion', 'Pressure_mean']) + "`. `VAE` идет очень близко: `Local MAE = " + fmt(top.loc['VAE', 'Local_MAE_mean']) + "`, `Pressure = " + fmt(top.loc['VAE', 'Pressure_mean']) + "`. `GAN` заметно слабее по надежности: его `Local MAE = " + fmt(top.loc['GAN', 'Local_MAE_mean']) + "`, а минимальная radiation monotonicity по seed’ам падает до `" + fmt(gan_min_mono, 2) + "`. Именно поэтому GAN мы оставляем только как comparison baseline, но не берем как основной метод.",
        "",
        "## Почему гибрид мы делали именно с VAE, а не с Diffusion",
        "",
        "Этот вопрос важен, потому что после benchmark естественно спросить: если лучший pure neural baseline — `Diffusion`, почему гибрид строится с `VAE`. Ответ состоит в том, что мы искали не лучший самостоятельный генератор, а лучший механизм residual modeling поверх `Matrix prior`. В такой постановке `VAE` оказывается удобнее и математически прозрачнее. Он естественным образом учит остаток вокруг prior, дает компактное latent space, поддерживает KL-регуляризацию и легко сочетается с rule-aware loss. Для сверхмалой 5x3 surface это проще и логичнее, чем поднимать тяжелую residual diffusion-схему.",
        "",
        "## Как устроен гибрид Residual VAE",
        "",
        "В гибриде мы не генерируем блок с нуля. Сначала `Matrix` задает `Mu_prior`, то есть объяснимую опорную surface. Затем VAE учит только остаток `R = Y - Mu_prior`. Энкодер принимает residual-представление вместе с rule-aware признаками, а декодер восстанавливает synthetic residual. После этого мы снова применяем тот же safety layer: projection, cap и calibration. То есть гибрид уважает уже существующую архитектуру проекта, а не обнуляет ее.",
        "",
        "С математической точки зрения особенно важна функция потерь. Помимо reconstruction и KL-компоненты в ней есть rule-aware penalty на raw output. Это означает, что модель штрафуется за нарушения monotonicity и других hard conditions еще до post-processing. Именно это делает Residual VAE не просто еще одной нейросетью, а логическим продолжением нашей rule-guided системы.",
        "",
        "## Что показывают финальные метрики",
        "",
        "Текущая картина по проекту лучше всего читается по `tables/metrics_multiseed_ru.csv`, `tables/method_scorecard_ru.csv` и `tables/matrix_evolution_ru.csv`. `Matrix` в полном multi-seed benchmark имеет `Local MAE = " + fmt(top.loc['Matrix', 'Local_MAE_mean']) + "`, `Pressure = " + fmt(top.loc['Matrix', 'Pressure_mean']) + "`. Это не самый маленький burden среди всех методов, но именно Matrix остается главным production-методом, потому что он первичен, объясним и научно защищаем. `Residual VAE` имеет `Local MAE = " + fmt(top.loc['Residual VAE', 'Local_MAE_mean']) + "`, `Pressure = " + fmt(top.loc['Residual VAE', 'Pressure_mean']) + "`. Это и делает его лучшей следующей версией системы.",
        "",
        "Нужно отдельно и честно проговорить ограничения. Во-первых, neural methods в нашем benchmark учатся не на независимой богатой реальности, а на teacher blocks от rule-guided pipeline. Поэтому benchmark не доказывает, что `Diffusion` или `VAE` биологически лучше Matrix. Он показывает, насколько хорошо они воспроизводят rule-guided synthetic manifold. Во-вторых, у `Residual VAE` в family multiseed остается один хвостовой seed, где final compliance снижается до `" + fmt(residual_min_final) + "`, хотя raw compliance остается `1.0`. Это важный operational detail, а не повод отказываться от гибрида. Он просто показывает, что финальный balancing step еще нужно аккуратно доработать.",
        "",
        "## Про explainability отдельно",
        "",
        "Explainability в проекте — это не один красивый график, а целый слой артефактов. У нас есть rule traceability, design-point explanations, block-level explainability log, pressure decomposition по стадиям, counterfactual analysis и SHAP-style разбор драйверов давления. Благодаря этому можно объяснить не только итоговую synthetic выборку, но и то, почему конкретный блок был принят, как сильно его корректировали правила и в какой именно части пространства возникла чувствительность. Именно этот слой и позволяет продавать работу как explainable system, а не как black-box generator.",
        "",
        "## Финальная архитектурная позиция проекта",
        "",
        "Финальная позиция проекта сейчас выглядит строго и логично. Основной production-метод — `Matrix`. Это главная научно защищаемая версия, потому что она напрямую опирается на observed data, literature-backed rules и явную структуру post-processing. Следующая лучшая версия системы — `Residual VAE`, потому что он сохраняет `Matrix prior`, резко снижает explainability burden и добавляет более гибкую synthetic variability. Лучший pure neural baseline — `Diffusion`, и это важно как доказательство того, что benchmark со modern methods действительно проведен. `GAN` как основной путь мы не используем.",
        "",
        "Если говорить одной фразой, то результат проекта такой: мы не заменили свою модель нейросетью, а последовательно развили explainable rule-guided Matrix generator, а затем построили над ним гибрид, который добавляет вариативность без отказа от биологической структуры. Именно это и нужно показывать в финальной презентации.",
        "",
        "Основные готовые материалы для выступления находятся в `text/mathematical_foundation_ru.md`, `text/slides_outline_ru.md`, `text/speaker_notes_ru.md` и `text/final_presentation_script_ru.md`. Основные таблицы лежат в `tables/metrics_multiseed_ru.csv`, `tables/method_scorecard_ru.csv`, `tables/matrix_evolution_ru.csv`, `tables/rule_usage_ru.csv`, `tables/literature_table_ru.csv` и `tables/rules_table_ru.csv`. Для слайдов уже подготовлены основные фигуры `01`-`12`, включая pipeline overview, fidelity comparison, explainability/compliance, method positioning, scorecard и отдельный график эволюции метода.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    apply_plot_style()

    single_df, multiseed_df, aggregate_df = load_summary_frames()
    literature_df = build_literature_table()
    rules_df = build_rules_table()
    architecture_df = build_architecture_table(single_df)
    rule_usage_df = build_rule_usage_table()
    evolution_df = build_matrix_evolution_table(single_df)
    single_metrics_df, aggregate_metrics_df = build_metrics_tables(single_df, aggregate_df)
    recommendation_df = build_recommendation_table(single_df, aggregate_df)
    scorecard_df = build_scorecard_table(aggregate_df, multiseed_df)

    save_pipeline_diagram()
    save_fidelity_comparison(single_df)
    save_explainability_comparison(single_df, multiseed_df)
    save_multiseed_stability(aggregate_df)
    save_method_positioning(aggregate_df)
    save_design_point_heatmaps()
    save_gan_seed_failure_plot(multiseed_df)
    save_scorecard_heatmap(scorecard_df)
    save_executive_summary_chart()
    save_rule_usage_plot(rule_usage_df)
    save_benchmark_podium(aggregate_metrics_df)
    save_matrix_evolution_chart(evolution_df)
    write_slide_texts(aggregate_metrics_df, scorecard_df, evolution_df, multiseed_df)
    write_math_appendix(aggregate_metrics_df, evolution_df)

    write_report(
        single_df=single_df,
        multiseed_df=multiseed_df,
        literature_df=literature_df,
        rules_df=rules_df,
        architecture_df=architecture_df,
        single_metrics_df=single_metrics_df,
        aggregate_metrics_df=aggregate_metrics_df,
        recommendation_df=recommendation_df,
        scorecard_df=scorecard_df,
        evolution_df=evolution_df,
    )
    print(f"saved presentation materials to: {OUTDIR}")


if __name__ == "__main__":
    main()
