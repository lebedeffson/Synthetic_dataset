from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from common_synthetic_metrics import EvaluationConfig, evaluate_synthetic_quality


EPS = 1e-6

RADIATION_LEVELS = [0.0, 2.0, 4.0, 6.0, 8.0]
THERMAL_CONDITIONS = [
    {"thermal_rank": 0, "temperature": 42.0, "time": 45.0, "label": "42C_45min"},
    {"thermal_rank": 1, "temperature": 43.0, "time": 45.0, "label": "43C_45min"},
    {"thermal_rank": 2, "temperature": 44.0, "time": 30.0, "label": "44C_30min"},
]
THERMAL_BY_KEY = {(x["temperature"], x["time"]): x for x in THERMAL_CONDITIONS}


RAW_ROWS = [
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


@dataclass
class CellLevelConfig:
    seed: int = 42
    n_synthetic: int = 1000
    outdir: str = "synthetic_data_cell_level_final"
    loo_bandwidth: Tuple[float, float, float] = (1.5, 0.6, 10.0)
    sigma_floor: float = 0.035
    sigma_cap: float = 0.120
    projection_max_iter: int = 50
    calibration_iterations: int = 4
    # Explainability & Trust
    explainability_mode: str = "log_only"  # off | log_only | enforced
    constraint_pressure_threshold: float = 0.15
    max_resample_attempts: int = 3
    save_explainability_log: bool = True
    save_explainability_plots: bool = True

    # Rule Ablation (GEMINI.md §19)
    disable_projection: bool = False
    disable_cap: bool = False
    disable_calibration: bool = False
    disable_local_sigma: bool = False


def cem43_from_temp_time(temperature_c: float, duration_min: float) -> float:
    r_factor = 0.25 if temperature_c < 43.0 else 0.5
    return float(duration_min * (r_factor ** (43.0 - temperature_c)))


def log_survival_transform(survival: np.ndarray | float) -> np.ndarray:
    survival = np.asarray(survival, dtype=float)
    return np.log10(np.clip(survival, 0.0, None) + EPS)


def inverse_log_survival(log_survival: np.ndarray | float) -> np.ndarray:
    log_survival = np.asarray(log_survival, dtype=float)
    return np.clip(np.power(10.0, log_survival) - EPS, 0.0, 1.0)


def kernel_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, bandwidth: Tuple[float, float, float]) -> np.ndarray:
    bw = np.asarray(bandwidth, dtype=float)
    diffs = (query_x[:, None, :] - train_x[None, :, :]) / bw[None, None, :]
    sq_dist = np.sum(diffs * diffs, axis=2)
    weights = np.exp(-0.5 * sq_dist)
    weight_sum = np.clip(weights.sum(axis=1), 1e-12, None)
    return (weights @ train_y) / weight_sum


def make_real_dataframe() -> pd.DataFrame:
    rows: List[Dict] = []
    for radiation, temperature, duration, survival in RAW_ROWS:
        thermal_meta = THERMAL_BY_KEY[(temperature, duration)]
        rows.append(
            {
                "Радиация": radiation,
                "Температура": temperature,
                "Время": duration,
                "Выживаемость": survival,
                "thermal_rank": thermal_meta["thermal_rank"],
                "thermal_label": thermal_meta["label"],
                "CEM43": cem43_from_temp_time(temperature, duration),
            }
        )
    return pd.DataFrame(rows)


def build_log_survival_matrix(real_df: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(RADIATION_LEVELS), len(THERMAL_CONDITIONS)), dtype=float)
    for i, radiation in enumerate(RADIATION_LEVELS):
        sub = real_df[real_df["Радиация"] == radiation].sort_values("thermal_rank")
        matrix[i, :] = log_survival_transform(sub["Выживаемость"].to_numpy(dtype=float))
    return matrix


def project_monotone_matrix(matrix: np.ndarray, max_iter: int) -> np.ndarray:
    projected = np.asarray(matrix, dtype=float).copy()
    row_ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    col_ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    x_row = np.arange(projected.shape[0], dtype=float)
    x_col = np.arange(projected.shape[1], dtype=float)

    for _ in range(max_iter):
        previous = projected.copy()
        for col in range(projected.shape[1]):
            projected[:, col] = row_ir.fit_transform(x_row, projected[:, col])
        for row in range(projected.shape[0]):
            projected[row, :] = col_ir.fit_transform(x_col, projected[row, :])
        if float(np.max(np.abs(projected - previous))) < 1e-7:
            break
    return projected


def estimate_noise_matrix(real_df: pd.DataFrame, mu_matrix: np.ndarray, cfg: CellLevelConfig) -> np.ndarray:
    features = real_df[["Радиация", "Температура", "Время"]].to_numpy(dtype=float)
    y = log_survival_transform(real_df["Выживаемость"].to_numpy(dtype=float))

    loo_preds: List[float] = []
    for idx in range(len(real_df)):
        mask = np.ones(len(real_df), dtype=bool)
        mask[idx] = False
        pred = kernel_predict(features[mask], y[mask], features[idx : idx + 1], cfg.loo_bandwidth)[0]
        loo_preds.append(float(pred))

    residuals = y - np.asarray(loo_preds, dtype=float)
    global_sigma = float(np.std(residuals, ddof=0))
    global_sigma = float(np.clip(global_sigma, cfg.sigma_floor, cfg.sigma_cap))

    sigma = np.full_like(mu_matrix, fill_value=global_sigma, dtype=float)
    for i in range(mu_matrix.shape[0]):
        for j in range(mu_matrix.shape[1]):
            gaps: List[float] = []
            if i > 0:
                gaps.append(abs(mu_matrix[i - 1, j] - mu_matrix[i, j]))
            if i < mu_matrix.shape[0] - 1:
                gaps.append(abs(mu_matrix[i, j] - mu_matrix[i + 1, j]))
            if j > 0:
                gaps.append(abs(mu_matrix[i, j - 1] - mu_matrix[i, j]))
            if j < mu_matrix.shape[1] - 1:
                gaps.append(abs(mu_matrix[i, j] - mu_matrix[i, j + 1]))

            positive_gaps = [g for g in gaps if g > 1e-6]
            if positive_gaps and not cfg.disable_local_sigma:
                local_sigma = min(global_sigma, 0.35 * min(positive_gaps))
            else:
                local_sigma = global_sigma
            sigma[i, j] = float(np.clip(local_sigma, cfg.sigma_floor, cfg.sigma_cap))
    return sigma


def _single_pass_row_projection(matrix: np.ndarray) -> np.ndarray:
    """One-pass row-only isotonic projection for CL1 attribution."""
    result = matrix.copy()
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    x = np.arange(result.shape[0], dtype=float)
    for col in range(result.shape[1]):
        result[:, col] = ir.fit_transform(x, result[:, col])
    return result


def generate_block_samples(
    mu_matrix: np.ndarray, sigma_matrix: np.ndarray, cfg: CellLevelConfig,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Generate synthetic blocks with optional explainability tracking (§8)."""
    rng = np.random.default_rng(cfg.seed)
    n_blocks = int(math.ceil(cfg.n_synthetic / mu_matrix.size))
    max_log = float(log_survival_transform(np.array([0.53]))[0])
    min_log = float(log_survival_transform(np.array([0.0]))[0])
    high_combined_log_cap = float(log_survival_transform(np.array([0.02]))[0])
    track = cfg.explainability_mode != "off"

    blocks: List[np.ndarray] = []
    records: List[Dict] = []

    for block_id in range(n_blocks):
        sampled = mu_matrix + rng.normal(0.0, sigma_matrix)
        lower = mu_matrix - 2.5 * sigma_matrix
        upper = mu_matrix + 2.5 * sigma_matrix
        sampled = np.clip(sampled, lower, upper)
        sampled = np.clip(sampled, min_log, max_log)

        if track:
            before_constraints = sampled.copy()

        # Stage: monotone projection (CL1 + CL3)
        if not cfg.disable_projection:
            sampled = project_monotone_matrix(sampled, cfg.projection_max_iter)

        if track:
            after_projection = sampled.copy()

        # Stage: high-combined-dose caps (CL5)
        if not cfg.disable_cap:
            for i, radiation in enumerate(RADIATION_LEVELS):
                for j, thermal_meta in enumerate(THERMAL_CONDITIONS):
                    if radiation >= 6.0 and thermal_meta["label"] in {"43C_45min", "44C_30min"}:
                        sampled[i, j] = min(sampled[i, j], high_combined_log_cap)

        if not cfg.disable_projection:
            sampled = project_monotone_matrix(sampled, cfg.projection_max_iter)
        sampled = np.clip(sampled, min_log, max_log)

        if track:
            after_cap = sampled.copy()
            delta_proj = after_projection - before_constraints
            delta_cap_val = after_cap - after_projection
            # rule-aware split: radiation vs thermal
            row_only = _single_pass_row_projection(before_constraints)
            delta_proj_rad = row_only - before_constraints
            delta_proj_therm = delta_proj - delta_proj_rad

            records.append({
                "block_id": block_id,
                "sampled_before_constraints_mean": float(before_constraints.mean()),
                "after_projection_mean": float(after_projection.mean()),
                "after_cap_mean": float(after_cap.mean()),
                # calibration fields filled later
                "after_calibration_mean": None,
                "final_mean": None,
                "delta_projection_mean": float(np.abs(delta_proj).mean()),
                "delta_projection_radiation_mean": float(np.abs(delta_proj_rad).mean()),
                "delta_projection_thermal_mean": float(np.abs(delta_proj_therm).mean()),
                "delta_cap_mean": float(np.abs(delta_cap_val).mean()),
                "delta_calibration_mean": None,
                "total_adjustment_abs_mean": None,
                "constraint_pressure_score": None,
                "per_cell_delta_abs": np.abs(delta_proj) + np.abs(delta_cap_val), # will add calibration later
                "projection_active_rate": float(np.mean(np.abs(delta_proj) > 1e-8)),
                "cap_active_rate": float(np.mean(np.abs(delta_cap_val) > 1e-8)),
                "calibration_active_rate": None,
                "rule_CL1_active": bool(np.any(np.abs(delta_proj_rad) > 1e-8)),
                "rule_CL3_active": bool(np.any(np.abs(delta_proj_therm) > 1e-8)),
                "rule_CL5_active": bool(np.any(np.abs(delta_cap_val) > 1e-8)),
                "explainability_status": "pass",
                "reject_reason": "",
                # keep raw matrices for calibration stage
                "_after_cap_matrix": after_cap.copy(),
            })

        blocks.append(sampled)
    return blocks, records


def calibrate_blocks_to_targets(
    blocks: List[np.ndarray], target_matrix: np.ndarray, cfg: CellLevelConfig, records: List[Dict],
) -> tuple[List[np.ndarray], List[Dict]]:
    """Calibrate blocks and update explainability records (§8)."""
    if cfg.disable_calibration:
        for r in records:
            r.update({
                "after_calibration_mean": r["after_cap_mean"],
                "delta_calibration_mean": 0.0,
                "calibration_active_rate": 0.0,
                "rule_CL5_active": r.get("rule_CL5_active", False) # already set
            })
        return blocks, records

    if not blocks:
        return blocks, records

    min_log = float(log_survival_transform(np.array([0.0]))[0])
    max_log = float(log_survival_transform(np.array([0.53]))[0])
    high_combined_log_cap = float(log_survival_transform(np.array([0.02]))[0])
    stacked = np.stack(blocks, axis=0)

    for _ in range(cfg.calibration_iterations):
        current_mean = stacked.mean(axis=0)
        delta = target_matrix - current_mean
        stacked = np.clip(stacked + delta[None, :, :], min_log, max_log)

        for block_idx in range(stacked.shape[0]):
            for i, radiation in enumerate(RADIATION_LEVELS):
                for j, thermal_meta in enumerate(THERMAL_CONDITIONS):
                    if radiation >= 6.0 and thermal_meta["label"] in {"43C_45min", "44C_30min"}:
                        stacked[block_idx, i, j] = min(stacked[block_idx, i, j], high_combined_log_cap)

            stacked[block_idx] = project_monotone_matrix(stacked[block_idx], cfg.projection_max_iter)
            stacked[block_idx] = np.clip(stacked[block_idx], min_log, max_log)

    result_blocks = [stacked[idx] for idx in range(stacked.shape[0])]

    # Update explainability records with calibration stage
    for idx, rec in enumerate(records):
        if idx >= stacked.shape[0]:
            break
        after_cal = stacked[idx]
        before_cal = rec.pop("_after_cap_matrix", None)
        rec["after_calibration_mean"] = float(after_cal.mean())
        rec["final_mean"] = float(after_cal.mean())
        if before_cal is not None:
            delta_calib = after_cal - before_cal
            delta_abs = np.abs(delta_calib)
            rec["delta_calibration_mean"] = float(delta_abs.mean())
            rec["calibration_active_rate"] = float(np.mean(delta_abs > 1e-8))
            
            # Finalize total pressure
            if "per_cell_delta_abs" in rec:
                rec["per_cell_delta_abs"] += delta_abs
                rec["total_adjustment_abs_mean"] = float(rec["per_cell_delta_abs"].mean())
                rec["constraint_pressure_score"] = rec["total_adjustment_abs_mean"]
                # Convert to list for serializability
                rec["per_cell_delta_abs"] = rec["per_cell_delta_abs"].tolist()
            else:
                rec["total_adjustment_abs_mean"] = rec["delta_calibration_mean"] # fallback
                rec["constraint_pressure_score"] = rec["total_adjustment_abs_mean"]
        else:
            rec["delta_calibration_mean"] = 0.0
            rec["calibration_active_rate"] = 0.0
            # If before_cal was None, then per_cell_delta_abs would not have been initialized
            # or updated with calibration. We need to ensure total_adjustment_abs_mean and
            # constraint_pressure_score are set correctly based on previous stages.
            # This assumes previous stages already set these or they default to 0.
            if "per_cell_delta_abs" not in rec:
                # This case should ideally not happen if track is True and _after_cap_matrix is always present.
                # But as a fallback, ensure these are set.
                rec["total_adjustment_abs_mean"] = rec["delta_projection_mean"] + rec["delta_cap_mean"]
                rec["constraint_pressure_score"] = rec["total_adjustment_abs_mean"]
            # If per_cell_delta_abs was present, it means previous stages were tracked,
            # and calibration had no effect, so the existing total is correct.

    return result_blocks, records


def apply_admission_control(
    blocks: List[np.ndarray], records: List[Dict], cfg: CellLevelConfig,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """GEMINI.md §10-§11: gate enforcement and admission control."""
    if cfg.explainability_mode != "enforced" or not records:
        return blocks, records

    for rec in records:
        if rec["constraint_pressure_score"] > cfg.constraint_pressure_threshold:
            rec["explainability_status"] = "rejected"
            rec["reject_reason"] = (
                f"constraint_pressure_score {rec['constraint_pressure_score']:.6f} "
                f"> threshold {cfg.constraint_pressure_threshold}"
            )

    passed_blocks = [
        b for b, r in zip(blocks, records) if r["explainability_status"] == "pass"
    ]
    return passed_blocks, records


def compute_explainability_summary(records: List[Dict]) -> Dict:
    """GEMINI.md §13: aggregate summary for logging."""
    if not records:
        return {}
    total = len(records)
    passed = sum(1 for r in records if r["explainability_status"] == "pass")
    rejected = total - passed
    reject_reasons: Dict[str, int] = {}
    for r in records:
        if r["reject_reason"]:
            key = r["reject_reason"].split(">")[0].strip() if ">" in r["reject_reason"] else r["reject_reason"]
            reject_reasons[key] = reject_reasons.get(key, 0) + 1

    def _safe_mean(key: str) -> float:
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    pressures = [r["constraint_pressure_score"] for r in records if r.get("constraint_pressure_score") is not None]
    p95 = float(np.percentile(pressures, 95)) if pressures else 0.0

    return {
        "blocks_total": total,
        "pass_count": passed,
        "reject_count": rejected,
        "pass_rate": passed / total if total else 0.0,
        "reject_rate": rejected / total if total else 0.0,
        "mean_delta_projection": _safe_mean("delta_projection_mean"),
        "mean_delta_cap": _safe_mean("delta_cap_mean"),
        "mean_delta_calibration": _safe_mean("delta_calibration_mean"),
        "mean_total_adjustment_abs": _safe_mean("total_adjustment_abs_mean"),
        "mean_constraint_pressure": _safe_mean("constraint_pressure_score"),
        "p95_constraint_pressure": p95,
        "mean_projection_active_rate": _safe_mean("projection_active_rate"),
        "mean_cap_active_rate": _safe_mean("cap_active_rate"),
        "mean_calibration_active_rate": _safe_mean("calibration_active_rate"),
        "top_reject_reasons": reject_reasons,
    }


def print_generation_summary(summary: Dict) -> None:
    """GEMINI.md §13: print log summary."""
    if not summary:
        return
    print("=== Explainability Summary ===")
    for key in ["blocks_total", "pass_count", "reject_count", "pass_rate", "reject_rate",
                "mean_delta_projection", "mean_delta_cap", "mean_delta_calibration",
                "mean_constraint_pressure", "p95_constraint_pressure"]:
        val = summary.get(key)
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")
    if summary.get("top_reject_reasons"):
        print(f"  top_reject_reasons: {summary['top_reject_reasons']}")


def plot_explainability_artifacts(metadata: Dict, outdir: Path) -> List[Path]:
    """GEMINI.md §14: plot explainability histograms and charts."""
    records = metadata.get("explainability_records", [])
    if not records:
        return []

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_dir = outdir / "explainability_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    log_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    df = pd.DataFrame(log_rows)

    sns.set_theme(style="whitegrid", context="talk")

    # 1. Histogram of constraint_pressure_score
    plt.figure(figsize=(10, 6))
    sns.histplot(df["constraint_pressure_score"], bins=20, kde=True, color="skyblue")
    plt.title("Constraint Pressure Score Distribution (§8, §11, §14)")
    plt.xlabel("Pressure Score (sum of abs deltas)")
    pressure_path = plot_dir / "constraint_pressure_hist.png"
    plt.savefig(pressure_path, bbox_inches="tight", dpi=150)
    plt.close()
    paths.append(pressure_path)

    # 2. Distribution of stage deltas
    delta_cols = ["delta_projection_mean", "delta_cap_mean", "delta_calibration_mean"]
    melted = df.melt(value_vars=delta_cols, var_name="Stage", value_name="Mean Abs Delta")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="Stage", y="Mean Abs Delta", palette="viridis")
    plt.title("Correction Magnitude per Stage (§8)")
    delta_path = plot_dir / "stage_deltas_boxplot.png"
    plt.savefig(delta_path, bbox_inches="tight", dpi=150)
    plt.close()
    paths.append(delta_path)

    # 3. Activation rates
    rate_cols = ["projection_active_rate", "cap_active_rate", "calibration_active_rate"]
    rates_df = df[rate_cols].mean().reset_index()
    rates_df.columns = ["Stage", "Activation Rate"]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=rates_df, x="Stage", y="Activation Rate", palette="magma")
    plt.title("Rule Activation Rates per Stage (§13, §14)")
    plt.ylim(0, 1.05)
    rate_path = plot_dir / "activation_rates_bar.png"
    plt.savefig(rate_path, bbox_inches="tight", dpi=150)
    plt.close()
    paths.append(rate_path)

    # 4. Reject reasons
    if df["explainability_status"].eq("rejected").any():
        plt.figure(figsize=(10, 6))
        reasons = df[df["explainability_status"] == "rejected"]["reject_reason"].apply(
            lambda x: x.split(">")[0].strip() if ">" in x else x
        ).value_counts()
        reasons.plot(kind="bar", color="salmon")
        plt.title("Reject Reasons (§11, §14)")
        plt.ylabel("Count")
        reject_path = plot_dir / "reject_reasons_bar.png"
        plt.savefig(reject_path, bbox_inches="tight", dpi=150)
        plt.close()
        paths.append(reject_path)

    # 5. Per-design-point correction heatmap (§14)
    # Average the per_cell_delta_abs across pass/all? Let's take all that we have.
    all_abs = [np.array(r["per_cell_delta_abs"]) for r in records if "per_cell_delta_abs" in r]
    if all_abs:
        mean_abs = np.mean(all_abs, axis=0)
        plt.figure(figsize=(10, 8))
        xticklabels = [t["label"] for t in THERMAL_CONDITIONS]
        yticklabels = [f"{r} Gy" for r in RADIATION_LEVELS]
        sns.heatmap(mean_abs, annot=True, fmt=".3f", cmap="YlOrRd", 
                    xticklabels=xticklabels, yticklabels=yticklabels)
        plt.title("Per-Design-Point Correction Pressure (§14)")
        plt.xlabel("Thermal Condition")
        plt.ylabel("Radiation Dose")
        heat_path = plot_dir / "pressure_heatmap.png"
        plt.savefig(heat_path, bbox_inches="tight", dpi=150)
        plt.close()
        paths.append(heat_path)

    return paths


def save_explainability_artifacts(metadata: Dict, outdir: Path) -> List[Path]:
    """GEMINI.md §12: save explainability log and summary."""
    records = metadata.get("explainability_records", [])
    summary = metadata.get("explainability_summary", {})
    if not records:
        return []

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # Save block-level log CSV
    log_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    log_df = pd.DataFrame(log_rows)
    log_path = outdir / "block_explainability_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    paths.append(log_path)

    # Save summary JSON
    summary_path = outdir / "block_explainability_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    paths.append(summary_path)

    return paths


def blocks_to_dataframe(blocks: List[np.ndarray], cfg: CellLevelConfig) -> pd.DataFrame:
    rows: List[Dict] = []
    for block_idx, matrix in enumerate(blocks):
        for i, radiation in enumerate(RADIATION_LEVELS):
            for j, thermal_meta in enumerate(THERMAL_CONDITIONS):
                survival_value = float(inverse_log_survival(np.array([matrix[i, j]]))[0])
                survival_value = float(np.clip(survival_value, 0.0, 0.53))
                if survival_value <= 1e-15:
                    survival_value = 0.0
                rows.append(
                    {
                        "Радиация": radiation,
                        "Температура": thermal_meta["temperature"],
                        "Время": thermal_meta["time"],
                        "Выживаемость": survival_value,
                        "thermal_rank": thermal_meta["thermal_rank"],
                        "thermal_label": thermal_meta["label"],
                        "CEM43": cem43_from_temp_time(thermal_meta["temperature"], thermal_meta["time"]),
                        "generation_block": block_idx,
                    }
                )

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(cfg.seed + 17)
    excess = len(df) - cfg.n_synthetic
    if excess > 0:
        design_keys = list(range(len(RADIATION_LEVELS) * len(THERMAL_CONDITIONS)))
        drop_keys = set(rng.choice(design_keys, size=excess, replace=False).tolist())
        keep_mask = []
        for idx in range(len(df)):
            design_key = idx % (len(RADIATION_LEVELS) * len(THERMAL_CONDITIONS))
            keep_mask.append(design_key not in drop_keys or idx >= len(design_keys))
        df = df[np.array(keep_mask, dtype=bool)].copy()

    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    return df.iloc[: cfg.n_synthetic].copy()


def build_generation_artifacts(cfg: CellLevelConfig) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    real_df = make_real_dataframe()
    mu_matrix = build_log_survival_matrix(real_df)
    sigma_matrix = estimate_noise_matrix(real_df, mu_matrix, cfg)
    
    # §11, §24: Resampling loop
    final_blocks: List[np.ndarray] = []
    final_records: List[Dict] = []
    target_count_blocks = int(math.ceil(cfg.n_synthetic / mu_matrix.size))
    
    attempts = 0
    while len(final_blocks) < target_count_blocks and attempts < cfg.max_resample_attempts:
        needed = target_count_blocks - len(final_blocks)
        tmp_cfg = dataclass_replace(cfg, n_synthetic=needed * mu_matrix.size, seed=cfg.seed + attempts)
        blocks, records = generate_block_samples(mu_matrix, sigma_matrix, tmp_cfg)
        blocks, records = calibrate_blocks_to_targets(blocks, mu_matrix, tmp_cfg, records)
        
        if cfg.explainability_mode == "enforced":
            passed_b, _ = apply_admission_control(blocks, records, tmp_cfg)
            final_blocks.extend(passed_b)
            final_records.extend(records) # we keep rejected records for logging
        else:
            final_blocks.extend(blocks)
            final_records.extend(records)
        attempts += 1
    
    if len(final_blocks) < target_count_blocks:
        print(f"Warning: only {len(final_blocks)}/{target_count_blocks} blocks passed after {attempts} attempts.")

    synthetic_df = blocks_to_dataframe(final_blocks, cfg)

    metadata = {
        "generator": "cell_level_article_guided_design_preserving",
        "seed": cfg.seed,
        "n_synthetic_requested": cfg.n_synthetic,
        "n_synthetic_saved": int(len(synthetic_df)),
        "resample_attempts_used": attempts,
        "design_support": [
            {"radiation": r, "temperature": t["temperature"], "time": t["time"], "label": t["label"]}
            for r in RADIATION_LEVELS
            for t in THERMAL_CONDITIONS
        ],
        "sigma_matrix_log10": sigma_matrix.tolist(),
        "calibration_iterations": cfg.calibration_iterations,
        "rules_file": "knowledge_base/cell_level_rules.json",
    }

    # §8-§13: add explainability to metadata
    if cfg.explainability_mode != "off":
        metadata["explainability_records"] = final_records
        summary = compute_explainability_summary(final_records)
        metadata["explainability_summary"] = summary
        print_generation_summary(summary)

    return real_df, synthetic_df, metadata


def save_outputs(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, metadata: Dict, cfg: CellLevelConfig) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    outdir = project_root / cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    real_df.to_csv(outdir / "real_design_points.csv", index=False, encoding="utf-8-sig")
    synthetic_df[["Радиация", "Температура", "Время", "Выживаемость"]].to_csv(
        outdir / "final_synthetic_dataset.csv", index=False, encoding="utf-8-sig"
    )
    synthetic_df.to_csv(outdir / "final_synthetic_dataset_full.csv", index=False, encoding="utf-8-sig")

    with (outdir / "generation_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    if cfg.explainability_mode != "off":
        save_explainability_artifacts(metadata, outdir)
        if cfg.save_explainability_plots:
            plot_explainability_artifacts(metadata, outdir)

    eval_cfg = EvaluationConfig(seed=cfg.seed)
    evaluate_synthetic_quality(
        real_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(),
        synthetic_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(),
        eval_cfg,
        outdir,
    )
    return outdir


def main() -> None:
    cfg = CellLevelConfig()
    real_df, synthetic_df, metadata = build_generation_artifacts(cfg)
    outdir = save_outputs(real_df, synthetic_df, metadata, cfg)
    print(f"saved final dataset to: {outdir / 'final_synthetic_dataset.csv'}")


if __name__ == "__main__":
    main()
