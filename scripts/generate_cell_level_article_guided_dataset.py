from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
            if positive_gaps:
                local_sigma = min(global_sigma, 0.35 * min(positive_gaps))
            else:
                local_sigma = global_sigma
            sigma[i, j] = float(np.clip(local_sigma, cfg.sigma_floor, cfg.sigma_cap))
    return sigma


def generate_block_samples(mu_matrix: np.ndarray, sigma_matrix: np.ndarray, cfg: CellLevelConfig) -> List[np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n_blocks = int(math.ceil(cfg.n_synthetic / mu_matrix.size))
    max_log = float(log_survival_transform(np.array([0.53]))[0])
    min_log = float(log_survival_transform(np.array([0.0]))[0])
    high_combined_log_cap = float(log_survival_transform(np.array([0.02]))[0])

    blocks: List[np.ndarray] = []
    for _ in range(n_blocks):
        sampled = mu_matrix + rng.normal(0.0, sigma_matrix)
        lower = mu_matrix - 2.5 * sigma_matrix
        upper = mu_matrix + 2.5 * sigma_matrix
        sampled = np.clip(sampled, lower, upper)
        sampled = np.clip(sampled, min_log, max_log)
        sampled = project_monotone_matrix(sampled, cfg.projection_max_iter)

        for i, radiation in enumerate(RADIATION_LEVELS):
            for j, thermal_meta in enumerate(THERMAL_CONDITIONS):
                if radiation >= 6.0 and thermal_meta["label"] in {"43C_45min", "44C_30min"}:
                    sampled[i, j] = min(sampled[i, j], high_combined_log_cap)

        sampled = project_monotone_matrix(sampled, cfg.projection_max_iter)
        sampled = np.clip(sampled, min_log, max_log)
        blocks.append(sampled)
    return blocks


def calibrate_blocks_to_targets(blocks: List[np.ndarray], target_matrix: np.ndarray, cfg: CellLevelConfig) -> List[np.ndarray]:
    if not blocks:
        return blocks

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

    return [stacked[idx] for idx in range(stacked.shape[0])]


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
    blocks = generate_block_samples(mu_matrix, sigma_matrix, cfg)
    blocks = calibrate_blocks_to_targets(blocks, mu_matrix, cfg)
    synthetic_df = blocks_to_dataframe(blocks, cfg)

    metadata = {
        "generator": "cell_level_article_guided_design_preserving",
        "seed": cfg.seed,
        "n_synthetic_requested": cfg.n_synthetic,
        "n_synthetic_saved": int(len(synthetic_df)),
        "design_support": [
            {"radiation": r, "temperature": t["temperature"], "time": t["time"], "label": t["label"]}
            for r in RADIATION_LEVELS
            for t in THERMAL_CONDITIONS
        ],
        "sigma_matrix_log10": sigma_matrix.tolist(),
        "calibration_iterations": cfg.calibration_iterations,
        "rules_file": "knowledge_base/cell_level_rules.json",
    }
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
