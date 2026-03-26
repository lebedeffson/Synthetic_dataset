
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from generate_cell_level_article_guided_dataset import CellLevelConfig, build_generation_artifacts
from common_synthetic_metrics import MINIMAL_COLUMNS
from validate_cell_level_datasets import evaluate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "synthetic_data_cell_level_final" / "ablations"
OUTDIR.mkdir(parents=True, exist_ok=True)

ABLATIONS = {
    "full_pipeline": {},
    "no_projection": {"disable_projection": True},
    "no_cap": {"disable_cap": True},
    "no_calibration": {"disable_calibration": True},
    "no_local_sigma": {"disable_local_sigma": True},
}


def format_delta(value: float, precision: int = 4) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.{precision}f}"


def build_observations(df: pd.DataFrame) -> List[str]:
    baseline = df.loc[df["ablation"] == "full_pipeline"].iloc[0]
    observations: List[str] = []

    for ablation_name, title in [
        ("no_projection", "Projection (CL1/CL3)"),
        ("no_calibration", "Calibration"),
        ("no_cap", "Caps (CL5)"),
        ("no_local_sigma", "Local sigma"),
    ]:
        row = df.loc[df["ablation"] == ablation_name].iloc[0]
        delta_mae = float(row["local_mean_abs_error"] - baseline["local_mean_abs_error"])
        delta_pressure = float(row["mean_pressure"] - baseline["mean_pressure"])

        if ablation_name == "no_projection":
            delta_rad = float(row["radiation_monotonicity"] - baseline["radiation_monotonicity"])
            delta_therm = float(row["thermal_monotonicity"] - baseline["thermal_monotonicity"])
            if delta_rad < -1e-6 or delta_therm < -1e-6:
                observations.append(
                    f"- **{title}**: без projection monotonicity падает (`RT {format_delta(delta_rad)}`, `HT {format_delta(delta_therm)}`), "
                    f"а Local MAE меняется на `{format_delta(delta_mae)}`."
                )
            else:
                observations.append(
                    f"- **{title}**: в этом датасете final monotonicity не просела, но pipeline потерял явную projection-stage коррекцию; "
                    f"Local MAE изменился на `{format_delta(delta_mae)}`, pressure на `{format_delta(delta_pressure)}`."
                )
        elif ablation_name == "no_calibration":
            observations.append(
                f"- **{title}**: отключение calibration меняет Local MAE на `{format_delta(delta_mae)}` и pressure на `{format_delta(delta_pressure)}`; "
                "этот шаг нужен для возврата synthetic mean к наблюдаемой матрице."
            )
        elif ablation_name == "no_cap":
            delta_hd = float(row["high_dose_plausibility"] - baseline["high_dose_plausibility"])
            observations.append(
                f"- **{title}**: high-dose plausibility меняется на `{format_delta(delta_hd)}`; "
                f"Local MAE меняется на `{format_delta(delta_mae)}`."
            )
        elif ablation_name == "no_local_sigma":
            observations.append(
                f"- **{title}**: без локальной оценки шума меняются fidelity и burden "
                f"(Local MAE `{format_delta(delta_mae)}`, pressure `{format_delta(delta_pressure)}`)."
            )

    return observations


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells: List[str] = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def perform_ablations():
    results = []
    
    for name, overrides in ABLATIONS.items():
        print(f"Running ablation: {name}...")
        cfg = CellLevelConfig(seed=42, explainability_mode="log_only", **overrides)
        _, synthetic_df, metadata = build_generation_artifacts(cfg)
        
        # Evaluate
        synth_min = synthetic_df[MINIMAL_COLUMNS].copy()
        independent = evaluate_dataset(synth_min, f"ablation_{name}")
        
        summary = metadata.get("explainability_summary", {})
        
        res = {
            "ablation": name,
            "local_mean_abs_error": independent["local_mean_abs_error"],
            "radiation_monotonicity": independent["radiation_monotonicity_mean_rate"],
            "thermal_monotonicity": independent["thermal_monotonicity_mean_rate"],
            "high_dose_plausibility": independent["high_combined_dose_low_survival_rate"],
            "mean_pressure": summary.get("mean_constraint_pressure", 0.0),
            "mean_delta_proj": summary.get("mean_delta_projection", 0.0),
            "mean_delta_calib": summary.get("mean_delta_calibration", 0.0),
        }
        results.append(res)

    df = pd.DataFrame(results)
    csv_path = OUTDIR / "ablation_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # Generate MD report
    md_path = OUTDIR / "ablation_report.md"
    lines = [
        "# Rule Ablation Analysis (§19)",
        "",
        "This report compares the impact of disabling different stages of the generation pipeline on survival quality and trust metrics.",
        "",
        dataframe_to_markdown(df),
        "",
        "## Observations",
        "",
        *build_observations(df),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"Ablation analysis saved to {md_path}")

if __name__ == "__main__":
    perform_ablations()
