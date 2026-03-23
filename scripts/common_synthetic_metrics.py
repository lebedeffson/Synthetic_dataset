from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COL_RADIATION = "\u0420\u0430\u0434\u0438\u0430\u0446\u0438\u044f"
COL_TEMPERATURE = "\u0422\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430"
COL_TIME = "\u0412\u0440\u0435\u043c\u044f"
COL_SURVIVAL = "\u0412\u044b\u0436\u0438\u0432\u0430\u0435\u043c\u043e\u0441\u0442\u044c"
MINIMAL_COLUMNS = [COL_RADIATION, COL_TEMPERATURE, COL_TIME, COL_SURVIVAL]


@dataclass
class EvaluationConfig:
    seed: int = 42
    discriminator_rounds: int = 10


def normalize_minimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns[:4]) == MINIMAL_COLUMNS:
        return df[MINIMAL_COLUMNS].copy()

    first_four = df.iloc[:, :4].copy()
    first_four.columns = MINIMAL_COLUMNS
    return first_four


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_distribution_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    real_df = normalize_minimal_columns(real_df)
    synthetic_df = normalize_minimal_columns(synthetic_df)

    rows: List[Dict[str, float | str]] = []
    for col in MINIMAL_COLUMNS:
        real = real_df[col].to_numpy(dtype=float)
        synth = synthetic_df[col].to_numpy(dtype=float)
        pooled_range = max(real.max(), synth.max()) - min(real.min(), synth.min())
        pooled_range = float(max(pooled_range, 1e-8))
        ks = ks_2samp(real, synth, method="auto")
        rows.append(
            {
                "feature": col,
                "real_mean": float(real.mean()),
                "synthetic_mean": float(synth.mean()),
                "real_std": float(real.std(ddof=0)),
                "synthetic_std": float(synth.std(ddof=0)),
                "wasserstein": float(wasserstein_distance(real, synth)),
                "wasserstein_normalized": float(wasserstein_distance(real, synth) / pooled_range),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return pd.DataFrame(rows)


def compute_correlation_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
    real_df = normalize_minimal_columns(real_df)
    synthetic_df = normalize_minimal_columns(synthetic_df)

    metrics: Dict[str, float] = {}
    for method in ("pearson", "spearman"):
        real_corr = real_df[MINIMAL_COLUMNS].corr(method=method).to_numpy(dtype=float)
        synth_corr = synthetic_df[MINIMAL_COLUMNS].corr(method=method).to_numpy(dtype=float)
        diff = real_corr - synth_corr
        metrics[f"{method}_correlation_frobenius"] = float(np.linalg.norm(diff, ord="fro"))
        metrics[f"{method}_correlation_mean_abs_diff"] = float(np.mean(np.abs(diff)))
    return metrics


def compute_separability_metrics(
    real_df: pd.DataFrame, synthetic_df: pd.DataFrame, cfg: EvaluationConfig
) -> Dict[str, float]:
    real_df = normalize_minimal_columns(real_df)
    synthetic_df = normalize_minimal_columns(synthetic_df)

    real = real_df[MINIMAL_COLUMNS].copy()
    n_real = len(real)
    aucs: List[float] = []

    for round_idx in range(cfg.discriminator_rounds):
        synth = synthetic_df[MINIMAL_COLUMNS].sample(
            n=n_real,
            replace=len(synthetic_df) < n_real,
            random_state=cfg.seed + round_idx,
        )
        x_all = pd.concat([real, synth], ignore_index=True)
        y = np.array([0] * len(real) + [1] * len(synth))
        cv = StratifiedKFold(n_splits=min(5, n_real), shuffle=True, random_state=cfg.seed + round_idx)
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=2000, random_state=cfg.seed + round_idx)),
            ]
        )
        probas = cross_val_predict(clf, x_all, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(float(roc_auc_score(y, probas)))

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    gini_values = [2.0 * auc - 1.0 for auc in aucs]
    return {
        "separability_auc_mean": auc_mean,
        "separability_auc_std": auc_std,
        "separability_auc_distance_from_0_5": float(abs(auc_mean - 0.5)),
        "separability_gini_mean": float(np.mean(gini_values)),
        "separability_gini_abs_mean": float(np.mean(np.abs(gini_values))),
    }


def compute_utility_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, cfg: EvaluationConfig) -> Dict[str, float]:
    real_df = normalize_minimal_columns(real_df)
    synthetic_df = normalize_minimal_columns(synthetic_df)

    feature_cols = [COL_RADIATION, COL_TEMPERATURE, COL_TIME]
    x_real = real_df[feature_cols].to_numpy(dtype=float)
    y_real = real_df[COL_SURVIVAL].to_numpy(dtype=float)
    x_synth = synthetic_df[feature_cols].to_numpy(dtype=float)
    y_synth = synthetic_df[COL_SURVIVAL].to_numpy(dtype=float)

    tstr_model = RandomForestRegressor(
        n_estimators=400,
        random_state=cfg.seed,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    tstr_model.fit(x_synth, y_synth)
    y_pred_tstr = tstr_model.predict(x_real)

    loo = LeaveOneOut()
    y_pred_trtr = np.zeros_like(y_real, dtype=float)
    for train_idx, test_idx in loo.split(x_real):
        model = RandomForestRegressor(
            n_estimators=400,
            random_state=cfg.seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(x_real[train_idx], y_real[train_idx])
        y_pred_trtr[test_idx] = model.predict(x_real[test_idx])

    tstr_mae = mean_absolute_error(y_real, y_pred_tstr)
    trtr_mae = mean_absolute_error(y_real, y_pred_trtr)

    return {
        "tstr_mae": float(tstr_mae),
        "tstr_rmse": rmse(y_real, y_pred_tstr),
        "tstr_r2": float(r2_score(y_real, y_pred_tstr)),
        "trtr_mae_leave_one_out": float(trtr_mae),
        "trtr_rmse_leave_one_out": rmse(y_real, y_pred_trtr),
        "trtr_r2_leave_one_out": float(r2_score(y_real, y_pred_trtr)),
        "tstr_to_trtr_mae_ratio": float(tstr_mae / max(trtr_mae, 1e-8)),
    }


def compute_coverage_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
    real_df = normalize_minimal_columns(real_df)
    synthetic_df = normalize_minimal_columns(synthetic_df)

    combined = pd.concat([real_df[MINIMAL_COLUMNS], synthetic_df[MINIMAL_COLUMNS]], ignore_index=True)
    scaler = StandardScaler().fit(combined)
    real_scaled = scaler.transform(real_df[MINIMAL_COLUMNS])
    synth_scaled = scaler.transform(synthetic_df[MINIMAL_COLUMNS])

    nn_real_to_synth = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    dist_real_to_synth = nn_real_to_synth.kneighbors(real_scaled, return_distance=True)[0].ravel()

    nn_synth_to_real = NearestNeighbors(n_neighbors=1).fit(real_scaled)
    dist_synth_to_real = nn_synth_to_real.kneighbors(synth_scaled, return_distance=True)[0].ravel()

    duplicate_real_rows = set(map(tuple, np.round(real_df[MINIMAL_COLUMNS].to_numpy(dtype=float), 6)))
    synth_rows = list(map(tuple, np.round(synthetic_df[MINIMAL_COLUMNS].to_numpy(dtype=float), 6)))
    duplicate_rate = float(np.mean([row in duplicate_real_rows for row in synth_rows]))

    support_violation = {}
    for col in MINIMAL_COLUMNS:
        real_min = float(real_df[col].min())
        real_max = float(real_df[col].max())
        values = synthetic_df[col].to_numpy(dtype=float)
        support_violation[col] = float(np.mean((values < real_min) | (values > real_max)))

    return {
        "real_to_synth_mean_nn_distance": float(dist_real_to_synth.mean()),
        "synth_to_real_mean_nn_distance": float(dist_synth_to_real.mean()),
        "duplicate_rate_vs_real": duplicate_rate,
        "support_violation_rate_mean": float(np.mean(list(support_violation.values()))),
        "support_violation_rate_max": float(max(support_violation.values())),
    }


def evaluate_synthetic_quality(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    cfg: EvaluationConfig,
    outdir: Path,
) -> List[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    real_min = normalize_minimal_columns(real_df)
    synth_min = normalize_minimal_columns(synthetic_df)

    distribution_df = compute_distribution_metrics(real_min, synth_min)
    distribution_path = outdir / "evaluation_distribution_metrics.csv"
    distribution_df.to_csv(distribution_path, index=False, encoding="utf-8-sig")

    summary_df = pd.concat(
        [
            real_min.describe().T.add_prefix("real_"),
            synth_min.describe().T.add_prefix("synthetic_"),
        ],
        axis=1,
    )
    summary_path = outdir / "evaluation_summary_statistics.csv"
    summary_df.to_csv(summary_path, encoding="utf-8-sig")

    metrics = {
        "n_real": int(len(real_min)),
        "n_synthetic": int(len(synth_min)),
        "mean_wasserstein_normalized": float(distribution_df["wasserstein_normalized"].mean()),
        "mean_ks_statistic": float(distribution_df["ks_statistic"].mean()),
    }
    metrics.update(compute_correlation_metrics(real_min, synth_min))
    metrics.update(compute_separability_metrics(real_min, synth_min, cfg))
    metrics.update(compute_utility_metrics(real_min, synth_min, cfg))
    metrics.update(compute_coverage_metrics(real_min, synth_min))

    metrics_path = outdir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    return [str(distribution_path), str(summary_path), str(metrics_path)]
