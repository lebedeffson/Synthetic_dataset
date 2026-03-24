
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "synthetic_data_cell_level_final" / "explainability_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

def run_shap_surrogate():
    meta_path = PROJECT_ROOT / "synthetic_data_cell_level_final" / "generation_metadata.json"
    if not meta_path.exists():
        print("Metadata not found.")
        return

    with meta_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    records = metadata.get("explainability_records", [])
    if not records:
        print("No explainability records.")
        return

    # Map design support
    design = metadata["design_support"]
    # Matrix was 5x3. design list should be 15 items in row-major order?
    # Actually generate_cell_level_article_guided_dataset.py logic:
    # [ (r, t) for r in RAD for t in THERMAL ]
    
    X_rows = []
    y_rows = []
    
    for rec in records:
        per_cell = np.array(rec["per_cell_delta_abs"]) # shape (5, 3)
        for i in range(5):
            for j in range(3):
                # Map to features
                # The design list in metadata is flat. 
                item = design[i * 3 + j]
                X_rows.append([
                    item["radiation"],
                    item["temperature"],
                    item["time"]
                ])
                y_rows.append(per_cell[i, j])

    X = pd.DataFrame(X_rows, columns=["Radiation", "Temperature", "Time"])
    y = np.array(y_rows)

    print(f"Training SHAP surrogate on {len(X)} cell-level observations...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Plot 1: Summary Bar
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Constraint Pressure Drivers (§18): SHAP Global Importance")
    plt.savefig(OUTDIR / "shap_pressure_global_bar.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Plot 2: Summary Dot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Pressure Detail: High values of features vs Attribution")
    plt.savefig(OUTDIR / "shap_pressure_detail_dots.png", bbox_inches="tight", dpi=150)
    plt.close()
    
    print(f"SHAP plots saved to {OUTDIR}")

if __name__ == "__main__":
    run_shap_surrogate()
