from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_ROOT / "synthetic_data_cell_level_final" / "explainability_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)


def _load_records() -> tuple[pd.DataFrame, np.ndarray]:
    meta_path = PROJECT_ROOT / "synthetic_data_cell_level_final" / "generation_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    records = metadata.get("explainability_records", [])
    if not records:
        raise ValueError("No explainability records found in generation metadata.")

    design = metadata["design_support"]
    x_rows = []
    y_rows = []
    for rec in records:
        per_cell = np.array(rec["per_cell_delta_abs"], dtype=float)
        for i in range(5):
            for j in range(3):
                item = design[i * 3 + j]
                x_rows.append([item["radiation"], item["temperature"], item["time"]])
                y_rows.append(per_cell[i, j])

    x = pd.DataFrame(x_rows, columns=["Radiation", "Temperature", "Time"])
    y = np.asarray(y_rows, dtype=float)
    return x, y


def _try_compute_shap(model: RandomForestRegressor, x: pd.DataFrame) -> np.ndarray | None:
    try:
        import shap  # type: ignore
    except ModuleNotFoundError:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    return np.asarray(shap_values, dtype=float)


def _save_global_importance(importances: np.ndarray, feature_names: list[str]) -> Path:
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(np.array(feature_names)[order], importances[order], color="#6b9080", edgecolor="#2f3e46")
    ax.set_title("Constraint Pressure Drivers: Global Importance")
    ax.set_xlabel("Mean absolute contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path = OUTDIR / "shap_pressure_global_bar.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def _save_detail_plot(values: np.ndarray, x: pd.DataFrame, feature_names: list[str]) -> Path:
    fig, axes = plt.subplots(1, len(feature_names), figsize=(15, 4.8), sharey=True)
    if len(feature_names) == 1:
        axes = [axes]
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        ax.scatter(x[feature], values[:, idx], s=18, alpha=0.45, color="#457b9d", edgecolors="none")
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Feature contribution to pressure")
    fig.suptitle("Constraint Pressure Detail by Feature", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = OUTDIR / "shap_pressure_detail_dots.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def run_shap_surrogate() -> None:
    x, y = _load_records()
    print(f"Training pressure surrogate on {len(x)} cell-level observations...")

    model = RandomForestRegressor(n_estimators=160, random_state=42)
    model.fit(x, y)

    feature_names = x.columns.tolist()
    shap_values = _try_compute_shap(model, x)

    if shap_values is not None and shap_values.ndim == 2:
        contrib = np.mean(np.abs(shap_values), axis=0)
        detail_values = shap_values
        mode = "SHAP"
    else:
        contrib = model.feature_importances_
        centered_x = x - x.mean(axis=0)
        detail_values = centered_x.to_numpy(dtype=float) * contrib[None, :]
        mode = "feature-importance fallback"

    global_path = _save_global_importance(np.asarray(contrib, dtype=float), feature_names)
    detail_path = _save_detail_plot(np.asarray(detail_values, dtype=float), x, feature_names)
    print(f"saved pressure explanation plots using {mode}:")
    print(global_path)
    print(detail_path)


if __name__ == "__main__":
    run_shap_surrogate()
