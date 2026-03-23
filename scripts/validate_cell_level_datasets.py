from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = PROJECT_ROOT / "comparison"

RADIATION_LEVELS = [0.0, 2.0, 4.0, 6.0, 8.0]
THERMAL_CONDITIONS = [
    (42.0, 45.0, "42C_45min", 11.25),
    (43.0, 45.0, "43C_45min", 45.0),
    (44.0, 30.0, "44C_30min", 60.0),
]
DESIGN_KEYS = {(r, t, tm) for r in RADIATION_LEVELS for t, tm, _, _ in THERMAL_CONDITIONS}

REAL_ROWS = [
    (0.0, 42.0, 45.0, 0.53),
    (0.0, 43.0, 45.0, 0.18),
    (0.0, 44.0, 30.0, 0.021),
    (2.0, 42.0, 45.0, 0.26),
    (2.0, 43.0, 45.0, 0.051),
    (2.0, 44.0, 30.0, 0.008),
    (4.0, 42.0, 45.0, 0.26),
    (4.0, 43.0, 45.0, 0.051),
    (4.0, 44.0, 30.0, 0.008),
    (6.0, 42.0, 45.0, 0.022),
    (6.0, 43.0, 45.0, 0.012),
    (6.0, 44.0, 30.0, 0.0003),
    (8.0, 42.0, 45.0, 0.0007),
    (8.0, 43.0, 45.0, 0.00004),
    (8.0, 44.0, 30.0, 0.0),
]


def make_real_df() -> pd.DataFrame:
    return pd.DataFrame(REAL_ROWS, columns=["Радиация", "Температура", "Время", "Выживаемость"])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:, :4].copy().rename(
        columns={
            df.columns[0]: "Радиация",
            df.columns[1]: "Температура",
            df.columns[2]: "Время",
            df.columns[3]: "Выживаемость",
        }
    )


def add_design_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    mapping = {(t, tm): rank for rank, (t, tm, _, _) in enumerate(THERMAL_CONDITIONS)}
    labels = {(t, tm): label for t, tm, label, _ in THERMAL_CONDITIONS}
    cem = {(t, tm): cem43 for t, tm, _, cem43 in THERMAL_CONDITIONS}
    out["thermal_rank"] = [mapping.get((float(t), float(tm)), np.nan) for t, tm in zip(out["Температура"], out["Время"])]
    out["thermal_label"] = [labels.get((float(t), float(tm)), "out_of_design") for t, tm in zip(out["Температура"], out["Время"])]
    out["CEM43"] = [cem.get((float(t), float(tm)), np.nan) for t, tm in zip(out["Температура"], out["Время"])]
    return out


def grouped_design_means(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Радиация", "Температура", "Время"], as_index=False)["Выживаемость"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "synthetic_mean", "median": "synthetic_median", "count": "synthetic_count"})
    )


def monotonicity_rate(values: List[float], tolerance: float = 1e-12) -> float | None:
    if len(values) < 2:
        return None
    comparisons = 0
    ok = 0
    for i in range(len(values) - 1):
        comparisons += 1
        if values[i + 1] <= values[i] + tolerance:
            ok += 1
    return float(ok / comparisons) if comparisons else None


def evaluate_dataset(df: pd.DataFrame, label: str) -> Dict:
    df = add_design_metadata(df)
    real_df = make_real_df()
    real_keyed = real_df.copy()
    real_keyed["key"] = list(zip(real_keyed["Радиация"], real_keyed["Температура"], real_keyed["Время"]))

    exact_support = df.apply(lambda row: (float(row["Радиация"]), float(row["Температура"]), float(row["Время"])) in DESIGN_KEYS, axis=1)
    design_rate = float(np.mean(exact_support))

    exact_df = df[exact_support].copy()
    grouped = grouped_design_means(exact_df)
    grouped["key"] = list(zip(grouped["Радиация"], grouped["Температура"], grouped["Время"]))
    grouped = grouped.merge(real_keyed[["key", "Выживаемость"]], on="key", how="left").rename(columns={"Выживаемость": "real_survival"})
    grouped["abs_mean_error"] = (grouped["synthetic_mean"] - grouped["real_survival"]).abs()
    grouped["abs_median_error"] = (grouped["synthetic_median"] - grouped["real_survival"]).abs()

    radiation_rates: List[float] = []
    for temperature, duration, _, _ in THERMAL_CONDITIONS:
        sub = grouped[(grouped["Температура"] == temperature) & (grouped["Время"] == duration)].sort_values("Радиация")
        if len(sub) == len(RADIATION_LEVELS):
            rate = monotonicity_rate(sub["synthetic_mean"].tolist())
            if rate is not None:
                radiation_rates.append(rate)
    radiation_monotonicity = float(np.mean(radiation_rates)) if radiation_rates else None

    thermal_rates: List[float] = []
    for radiation in RADIATION_LEVELS:
        sub = grouped[grouped["Радиация"] == radiation].copy()
        rank_map = {(t, tm): rank for rank, (t, tm, _, _) in enumerate(THERMAL_CONDITIONS)}
        if len(sub) == len(THERMAL_CONDITIONS):
            sub["rank"] = [rank_map[(float(t), float(tm))] for t, tm in zip(sub["Температура"], sub["Время"])]
            sub = sub.sort_values("rank")
            rate = monotonicity_rate(sub["synthetic_mean"].tolist())
            if rate is not None:
                thermal_rates.append(rate)
    thermal_monotonicity = float(np.mean(thermal_rates)) if thermal_rates else None

    high_combined_mask = (df["Радиация"] >= 6.0) & (df["CEM43"] >= 45.0)
    high_combined_rate = float(np.mean(df.loc[high_combined_mask, "Выживаемость"] <= 0.02)) if int(high_combined_mask.sum()) else None

    unique_design_points = int(grouped["key"].nunique())
    count_min = int(grouped["synthetic_count"].min()) if len(grouped) else 0
    count_max = int(grouped["synthetic_count"].max()) if len(grouped) else 0

    local_mean_abs_error = float(grouped["abs_mean_error"].mean()) if len(grouped) else None
    local_max_abs_error = float(grouped["abs_mean_error"].max()) if len(grouped) else None

    article_compliance_components = [
        design_rate,
    ]
    if radiation_monotonicity is not None:
        article_compliance_components.append(radiation_monotonicity)
    if thermal_monotonicity is not None:
        article_compliance_components.append(thermal_monotonicity)
    if high_combined_rate is not None:
        article_compliance_components.append(high_combined_rate)

    return {
        "dataset": label,
        "n_rows": int(len(df)),
        "exact_design_support_rate": design_rate,
        "unique_design_points_present": unique_design_points,
        "design_count_min": count_min,
        "design_count_max": count_max,
        "local_mean_abs_error": local_mean_abs_error,
        "local_max_abs_error": local_max_abs_error,
        "radiation_monotonicity_mean_rate": radiation_monotonicity,
        "thermal_monotonicity_mean_rate": thermal_monotonicity,
        "high_combined_dose_low_survival_rate": high_combined_rate,
        "independent_article_compliance_mean": float(np.mean(article_compliance_components)),
        "grouped_design_summary": grouped.drop(columns=["key"]).to_dict(orient="records"),
    }


def read_metrics_if_available(folder: Path) -> Dict | None:
    metrics_path = folder / "evaluation_metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    dataset_specs = [
        ("notebook_cvae", PROJECT_ROOT / "benchmarks" / "results" / "synthetic_data_notebook_cvae" / "cvae_synthetic_dataset.csv"),
        ("rule_guided_cvae", PROJECT_ROOT / "benchmarks" / "results" / "synthetic_data" / "cvae_synthetic_dataset.csv"),
        ("continuous_kernel_final", PROJECT_ROOT / "benchmarks" / "results" / "synthetic_data_final" / "final_synthetic_dataset.csv"),
        ("cell_level_article_guided_final", PROJECT_ROOT / "synthetic_data_cell_level_final" / "final_synthetic_dataset.csv"),
    ]

    results: List[Dict] = []
    for label, path in dataset_specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        result = evaluate_dataset(df, label)
        folder_metrics = read_metrics_if_available(path.parent)
        if folder_metrics is not None:
            result["statistical_metrics"] = folder_metrics
        results.append(result)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    json_path = COMPARISON_DIR / "cell_level_end_to_end_audit.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    lines = ["# Cell-Level End-to-End Audit", ""]
    lines.append("| Dataset | Exact Design Support | Local Mean Abs Error | Radiation Mono | Thermal Mono | Independent Article Compliance |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    fmt = lambda x: "NA" if x is None else f"{x:.4f}"
    for item in results:
        lines.append(
            f"| {item['dataset']} | {fmt(item['exact_design_support_rate'])} | {fmt(item['local_mean_abs_error'])} | "
            f"{fmt(item['radiation_monotonicity_mean_rate'])} | {fmt(item['thermal_monotonicity_mean_rate'])} | "
            f"{fmt(item['independent_article_compliance_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Exact Design Support` equals the fraction of synthetic rows that stay on the 15 observed experimental conditions.",
            "- `Local Mean Abs Error` compares synthetic mean survival against the observed value at each exact design point.",
            "- `Radiation Mono` checks non-increasing mean survival with increasing radiation at fixed thermal condition.",
            "- `Thermal Mono` checks non-increasing mean survival with stronger thermal condition at fixed radiation.",
            "- `Independent Article Compliance` averages only checks based on observable columns or exact design support.",
        ]
    )

    md_path = COMPARISON_DIR / "cell_level_end_to_end_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
