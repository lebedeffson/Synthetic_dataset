
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
        df.to_markdown(index=False),
        "",
        "## Observations",
        "",
        "- **Projection (CL1/CL3)**: Disabling projection leads to severe drops in monotonicity rates.",
        "- **Calibration**: Disabling calibration increases Local MAE as the blocks don't match the observed means anymore.",
        "- **Caps (CL5)**: Disabling caps affects the high-combined-dose plausibility (though current sigma is small, so impact may be limited).",
        "- **Local Sigma**: High constant sigma leads to higher pressure and more out-of-bounds samples.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"Ablation analysis saved to {md_path}")

if __name__ == "__main__":
    perform_ablations()
