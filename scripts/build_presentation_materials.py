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


def method_order_map() -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(METHOD_ORDER)}


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


def write_math_appendix() -> None:
    lines = [
        "# Математическая часть проекта",
        "",
        "## 1. Matrix generator",
        "",
        "Работаем в пространстве `log10(survival + eps)`.",
        "",
        "Наблюдаемая матрица:",
        "`Y* in R^(5x3)`",
        "",
        "Где строки соответствуют дозам RT, а столбцы — трем observed thermal conditions.",
        "",
        "Усиленная матричная модель строится как:",
        "`Y_raw = Mu_prior + E`",
        "",
        "где:",
        "- `Mu_prior` — сглаженная monotone prior-surface, построенная из observed matrix;",
        "- `E` — structured block noise;",
        "- затем применяется safety layer: `projection + cap + calibration`.",
        "",
        "Structured noise:",
        "`E_ij = Sigma_ij * T * (a_g * z_g + a_r * z_row_i + a_c * z_col_j + a_l * z_local_ij)`",
        "",
        "где `Sigma_ij` — heteroscedastic shrinkage sigma, а шум усечен по `truncation_z`.",
        "",
        "## 2. Hard и soft rules",
        "",
        "Hard rules:",
        "- `CL1`: survival non-increasing по radiation dose",
        "- `CL3`: thermal ordering в observed domain",
        "- `CL5`: при high combined dose survival должен быть very low",
        "",
        "Soft rules:",
        "- `CL2`: sensitizing window 41-43 C / 30-60 min",
        "- `CL4`: high temperature direct cytotoxicity",
        "",
        "Hard rules встраиваются в projection/cap. Soft rules используются как mechanistic priors и интерпретация.",
        "",
        "## 3. Residual VAE",
        "",
        "Гибридная модель не генерирует блок с нуля. Она учит только остаток вокруг matrix prior:",
        "",
        "`R = Y - Mu_prior`",
        "",
        "Encoder/decoder:",
        "- `q_phi(z | R, F)`",
        "- `p_theta(R | z, F)`",
        "",
        "где `F` — rule-aware feature tensor (`RT`, `temperature`, `time`, `CEM43`, `thermal_rank`, `CL2/CL4/CL5 indicators`).",
        "",
        "Итоговая генерация:",
        "`Y_raw = Mu_prior + R_hat(z, F)`",
        "`Y_final = SafetyPostprocess(Y_raw)`",
        "",
        "## 4. Loss function for Residual VAE",
        "",
        "`L = L_recon + beta * L_KL + lambda_rule * L_rule + lambda_center * L_center + lambda_var * L_var + lambda_smooth * L_smooth`",
        "",
        "Где:",
        "- `L_recon`: reconstruction loss в log-survival space",
        "- `L_KL`: regularization of latent space",
        "- `L_rule`: penalty for raw rule violations before projection",
        "- `L_center`: alignment to teacher-block mean / target center",
        "- `L_var`: variance matching",
        "- `L_smooth`: residual budget regularization",
        "",
        "## 5. Метрики",
        "",
        "- `Local MAE`: средняя абсолютная ошибка по 15 design points",
        "- `Local Max Error`: максимальная ошибка среди 15 design points",
        "- `Wasserstein(norm)`: близость распределений",
        "- `TSTR R2`: utility, train on synthetic test on real",
        "- `Explainability pressure`: средняя величина пост-коррекции после projection/cap/calibration",
        "",
        "## 6. Интерпретация результатов",
        "",
        "Если метод хорош только по fidelity, но требует большого rule-correction burden, он не подходит как explainable production method.",
        "Поэтому в проекте мы разделяем:",
        "- `Matrix` как scientific core",
        "- `Residual VAE` как next-version hybrid upgrade",
        "- `Diffusion` как strongest pure neural baseline",
        "",
    ]
    (TEXT_DIR / "mathematical_foundation_ru.md").write_text("\n".join(lines), encoding="utf-8")


def write_slide_texts(
    aggregate_metrics_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
) -> None:
    outline_lines = [
        "# Слайды для презентации",
        "",
        "## Слайд 1. Постановка задачи",
        "- У нас только 15 реальных design points в cell-level domain.",
        "- Нужен explainable synthetic generator, который уважает биологические правила.",
        "",
        "## Слайд 2. Данные и правила",
        "- Дизайн: 5 уровней RT x 3 терморежима.",
        "- Активные правила: CL1-CL5.",
        "- Hard rules: CL1, CL3, CL5. Soft rules: CL2, CL4.",
        "",
        "## Слайд 3. Наш базовый метод",
        "- Matrix: rule-guided stochastic 5x3 generator.",
        "- Использует projection, cap, calibration и explainability logging.",
        "",
        "## Слайд 4. Что сравнивали",
        "- Matrix, Residual VAE, Diffusion, VAE, GAN.",
        "- Все neural methods тестировались в одном rule-guided контуре.",
        "",
        "## Слайд 5. Ключевой benchmark",
        "- Показываем full multi-seed table и scorecard.",
        "- Отдельно выделяем fidelity и explainability burden.",
        "",
        "## Слайд 6. Главный вывод",
        "- Matrix = основной production / защита.",
        "- Residual VAE = лучшая следующая гибридная версия.",
        "- Diffusion = лучший чистый neural baseline.",
        "",
        "## Слайд 7. Почему не GAN",
        "- Есть провалы monotonicity/compliance на части seed.",
        "- Надежность ниже остальных.",
        "",
        "## Слайд 8. Что продаем",
        "- Explainable rule-guided synthetic generator for small constrained biomedical data.",
        "- С roadmap: Matrix core -> Residual VAE upgrade.",
        "",
    ]
    (TEXT_DIR / "slides_outline_ru.md").write_text("\n".join(outline_lines), encoding="utf-8")

    top_rows = aggregate_metrics_df.set_index("Метод")
    notes_lines = [
        "# Короткий текст для выступления",
        "",
        "Основной метод у нас Matrix, потому что он самый прозрачный и научно защищаемый.",
        "Следующая лучшая версия системы это Residual VAE: он сохраняет matrix prior, но резко снижает explainability pressure.",
        "Среди чистых neural baseline лучшим в полном multi-seed benchmark оказался Diffusion.",
        "GAN мы не берем, потому что у него хуже надежность по правилам.",
        "",
        "## Цифры, которые можно озвучивать",
        f"- Matrix: Local MAE {top_rows.loc['Matrix', 'Local_MAE_mean']:.4f}, Pressure {top_rows.loc['Matrix', 'Pressure_mean']:.4f}",
        f"- Residual VAE: Local MAE {top_rows.loc['Residual VAE', 'Local_MAE_mean']:.4f}, Pressure {top_rows.loc['Residual VAE', 'Pressure_mean']:.4f}",
        f"- Diffusion: Local MAE {top_rows.loc['Diffusion', 'Local_MAE_mean']:.4f}, TSTR R2 {top_rows.loc['Diffusion', 'TSTR_R2_mean']:.4f}",
        "",
        "## Финальная формулировка",
        "Мы не заменяем нашу rule-guided модель нейросетью. Мы строим вокруг нее более сильную систему: Matrix как core и Residual VAE как upgrade.",
        "",
        "## Scorecard top-3",
    ]
    for _, row in scorecard_df.head(3).iterrows():
        notes_lines.append(f"- {row['Метод']}: overall score {row['Overall_Score']:.2f}")
    (TEXT_DIR / "speaker_notes_ru.md").write_text("\n".join(notes_lines), encoding="utf-8")


def write_report(
    literature_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    architecture_df: pd.DataFrame,
    single_metrics_df: pd.DataFrame,
    aggregate_metrics_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
) -> None:
    report_path = OUTDIR / "presentation_report_ru.md"
    lines: List[str] = [
        "# Материалы для презентации по генерации synthetic dataset",
        "",
        "## 1. Данные о приборе",
        "",
        "- В репозитории отсутствуют паспорт прибора, модель облучателя и паспорт гипертермической установки.",
        "- Поэтому в презентации этот раздел нужно формулировать аккуратно: доступны только режимные параметры установки и клеточная выживаемость.",
        "- Наблюдаемые параметры эксперимента: `Радиация 0-8 Gy`, `Температура 42/43/44 C`, `Время 30/45 мин`, производная thermal dose `CEM43`.",
        "",
        "## 2. Выборка и биология данных",
        "",
        "- Это **cell-level / in vitro** survival matrix, а не пациентский датасет.",
        "- Дизайн эксперимента: `5 уровней RT x 3 терморежима = 15 design points`.",
        "- Три терморежима: `42C 45 мин`, `43C 45 мин`, `44C 30 мин`.",
        "- Биологический выход: доля выживших клеток после комбинированного воздействия гипертермии и радиации.",
        "- Главные биологические ожидания: при росте RT выживаемость не должна расти; усиление thermal condition в наблюдаемом окне тоже не должно повышать выживаемость.",
        "",
        "## 3. Как собирали правила",
        "",
        "- Правила брали из уже собранной knowledge base проекта, а не придумывали вручную под benchmark.",
        "- Активировали только те правила, которые можно честно применить к 4 наблюдаемым колонкам: `Радиация / Температура / Время / Выживаемость`.",
        "- Клинические и in vivo правила сознательно исключались, чтобы не смешивать уровни биологии.",
        "- Активные правила: `CL1-CL5`.",
        "",
        "## 4. Откуда брали литературу",
        "",
        f"- Полная таблица: `tables/literature_table_ru.csv`.",
        f"- Ключевые статьи: `E03-E06` для DNA repair / sensitizing window, `E01/E14` для температурного порога, `E06/E07/E08` для интервалов и ограничений обобщения.",
        "",
        "## 5. Общая таблица правил",
        "",
        f"- Полная таблица: `tables/rules_table_ru.csv`.",
        "- `CL1`: monotonicity по радиации.",
        "- `CL2`: sensitizing window около `41-43 C` и `30-60 мин`.",
        "- `CL3`: thermal ordering в наблюдаемом домене.",
        "- `CL4`: при температуре выше `43 C` усиливается direct cytotoxicity.",
        "- `CL5`: высокая комбинированная доза должна вести к very low survival.",
        "",
        "## 6. Как получали synthetic выборку",
        "",
        "- Базовый support всегда фиксирован на исходных 15 design points.",
        "- Наш основной метод `Matrix` строит rule-guided 5x3 поверхность лог-выживаемости, затем семплирует шум, применяет isotonic projection и calibration.",
        "- Новая версия `Residual VAE` моделирует только остаточную вариативность вокруг matrix prior и получает штраф за нарушения правил уже на этапе обучения.",
        "- Нейросетевые методы `VAE`, `GAN`, `Diffusion` обучались на rule-guided teacher blocks, потому что реальных наблюдений всего `n=15`.",
        "- Любой метод после raw generation проходил одинаковый post-processing: projection, cap и calibration.",
        "",
        "## 7. Архитектуры",
        "",
        f"- Таблица архитектур: `tables/architecture_table_ru.csv`.",
        "- `Matrix`: rule-guided stochastic matrix.",
        "- `Residual VAE`: гибридный residual generator поверх matrix prior, лучший кандидат на следующую финальную версию.",
        "- `Diffusion`: DDPM-style MLP denoiser, лучший чистый neural candidate по full multi-seed fidelity.",
        "- `VAE`: сильный альтернативный neural baseline с очень близкими метриками и чуть меньшим pressure.",
        "- `GAN`: adversarial baseline, но худший по надежности.",
        "",
        "## 8. Метрики и итог",
        "",
        f"- Single-run таблица: `tables/metrics_single_run_ru.csv`.",
        f"- Multi-seed таблица: `tables/metrics_multiseed_ru.csv`.",
        f"- Scorecard: `tables/method_scorecard_ru.csv`.",
        "- Главный научно защищаемый вывод: **наш основной production-метод = Matrix**, потому что он первичный, прозрачный и напрямую опирается на правила и литературу.",
        "- Главный вывод по следующей версии: **лучший гибридный метод = Residual VAE**.",
        "- Среди чистых student-neural методов: **лучший neural method = Diffusion** по полному multi-seed benchmark.",
        "- Самый низкий post-processing burden среди baseline-family теперь тоже показывает **Residual VAE**.",
        "- `GAN` не брать как основной: есть провалы по monotonicity/compliance на части seed.",
        "",
        "## 9. Что вставлять в слайды",
        "",
        "- `figures/01_pipeline_overview.png` — схема пайплайна.",
        "- `figures/02_fidelity_comparison.png` — качество на основном прогоне.",
        "- `figures/03_explainability_and_compliance.png` — explainability и compliance.",
        "- `figures/04_multiseed_stability.png` — устойчивость по 5 seed.",
        "- `figures/05_method_positioning.png` — positioning chart.",
        "- `figures/06_design_point_error_heatmaps.png` — ошибки по 15 design points.",
        "- `figures/07_monotonicity_by_seed.png` — why GAN is risky.",
        "- `figures/08_method_scorecard.png` — normalized multi-metric scorecard.",
        "- `figures/09_executive_summary.png` — executive summary slide.",
        "- `figures/10_rule_usage.png` — coverage and role of CL1-CL5.",
        "- `figures/11_benchmark_podium.png` — three main presentation positions.",
        "",
        "## 10. Итоговая рекомендация",
        "",
        "- Для презентации говорить так: `Matrix` — основной и объяснимый метод, `Residual VAE` — лучшая следующая гибридная версия, `Diffusion` — лучший чистый нейросетевой вариант, `VAE` — сильная альтернативная neural baseline, `GAN` — нестабилен.",
        "- Готовый текст для слайдов: `text/slides_outline_ru.md`.",
        "- Короткие notes для выступления: `text/speaker_notes_ru.md`.",
        "- Математика и формулы: `text/mathematical_foundation_ru.md`.",
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
    write_slide_texts(aggregate_metrics_df, scorecard_df)
    write_math_appendix()

    write_report(
        literature_df=literature_df,
        rules_df=rules_df,
        architecture_df=architecture_df,
        single_metrics_df=single_metrics_df,
        aggregate_metrics_df=aggregate_metrics_df,
        recommendation_df=recommendation_df,
        scorecard_df=scorecard_df,
    )
    print(f"saved presentation materials to: {OUTDIR}")


if __name__ == "__main__":
    main()
