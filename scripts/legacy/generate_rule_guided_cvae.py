from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


RAW_DATA = {
    "Радиация": [0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8],
    "Температура": [42, 43, 44, 42, 43, 44, 42, 43, 44, 42, 43, 44, 42, 43, 44],
    "Время": [45, 45, 30, 45, 45, 30, 45, 45, 30, 45, 45, 30, 45, 45, 30],
    "Выживаемость": [
        0.53,
        0.18,
        0.021,
        0.26,
        0.051,
        0.008,
        0.26,
        0.051,
        0.008,
        0.022,
        0.012,
        0.0003,
        0.0007,
        0.00004,
        0.0,
    ],
}


EPS = 1e-6


@dataclass
class GenerationConfig:
    seed: int = 42
    epochs: int = 350
    batch_size: int = 64
    latent_dim: int = 4
    hidden_dim: int = 64
    lr: float = 1e-3
    beta_kl: float = 0.02
    rule_penalty_weight: float = 0.30
    n_boot_per_row: int = 80
    n_synthetic: int = 1000
    assumed_interval_min: float = 0.0
    assumed_hypoxia: float = 0.70
    run_evaluation: bool = True
    discriminator_rounds: int = 10
    outdir: str = "benchmarks/results/synthetic_data"
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def smooth_window(x: np.ndarray | float, left: float, right: float, sharpness: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    left_gate = sigmoid((x - left) * sharpness)
    right_gate = sigmoid((right - x) * sharpness)
    return np.clip(left_gate * right_gate * 4.0, 0.0, 1.0)


def cem43_from_temp_time(temperature_c: np.ndarray | float, duration_min: np.ndarray | float) -> np.ndarray:
    temperature_c = np.asarray(temperature_c, dtype=float)
    duration_min = np.asarray(duration_min, dtype=float)
    r_factor = np.where(temperature_c < 43.0, 0.25, 0.5)
    return duration_min * np.power(r_factor, 43.0 - temperature_c)


def log_survival_transform(survival: np.ndarray | float) -> np.ndarray:
    survival = np.asarray(survival, dtype=float)
    return np.log10(np.clip(survival, 0.0, None) + EPS)


def inverse_log_survival(log_survival: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if isinstance(log_survival, torch.Tensor):
        return torch.clamp(torch.pow(10.0, log_survival) - EPS, min=0.0, max=1.0)
    return np.clip(np.power(10.0, log_survival) - EPS, 0.0, 1.0)


def kernel_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    bandwidth: Tuple[float, float, float] = (1.25, 0.30, 4.0),
) -> np.ndarray:
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    query_x = np.asarray(query_x, dtype=float)
    bw = np.asarray(bandwidth, dtype=float)
    diffs = (query_x[:, None, :] - train_x[None, :, :]) / bw[None, None, :]
    sq_dist = np.sum(diffs * diffs, axis=2)
    weights = np.exp(-0.5 * sq_dist)
    weight_sum = np.clip(weights.sum(axis=1), 1e-12, None)
    return (weights @ train_y) / weight_sum


def load_merged_rules(rules_path: Path) -> Dict:
    with rules_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def make_raw_frame() -> pd.DataFrame:
    df = pd.DataFrame(RAW_DATA).copy()
    df["source"] = "observed"
    return df


def enrich_with_rule_features(df: pd.DataFrame, assumed_interval_min: float, assumed_hypoxia: float) -> pd.DataFrame:
    out = df.copy()
    out["Интервал_мин"] = assumed_interval_min
    out["Гипоксия_предположенная"] = assumed_hypoxia
    out["CEM43"] = cem43_from_temp_time(out["Температура"].to_numpy(), out["Время"].to_numpy())

    temp = out["Температура"].to_numpy(dtype=float)
    duration = out["Время"].to_numpy(dtype=float)
    radiation = out["Радиация"].to_numpy(dtype=float)
    interval = out["Интервал_мин"].to_numpy(dtype=float)
    cem43 = out["CEM43"].to_numpy(dtype=float)

    temp_mild = smooth_window(temp, 39.0, 42.1, 2.7)
    temp_sens = smooth_window(temp, 40.8, 43.1, 3.8)
    temp_high = sigmoid((temp - 43.0) * 5.0)

    duration_medium = smooth_window(duration, 30.0, 60.0, 0.20)
    duration_long = sigmoid((duration - 55.0) * 0.18)
    interval_short = sigmoid((90.0 - interval) * 0.10)
    thermal_adequate = smooth_window(cem43, 8.0, 60.0, 0.08)
    thermal_high = sigmoid((cem43 - 60.0) * 0.06)
    hypoxia_effect = np.full_like(temp_mild, fill_value=float(assumed_hypoxia))

    dna_repair_inhibition = np.clip(temp_sens * (0.60 + 0.40 * duration_medium), 0.0, 1.0)
    oxygenation_gain = np.clip(temp_mild * (0.35 + 0.65 * hypoxia_effect), 0.0, 1.0)
    direct_cytotoxicity = np.clip(
        0.50 * temp_high + 0.25 * thermal_high + 0.25 * (temp_high * duration_long), 0.0, 1.0
    )
    high_temp_risk = np.clip(temp_high * (0.55 + 0.45 * duration_long), 0.0, 1.0)

    radiation_on = (radiation > 0).astype(float)
    synergy_score = np.clip(
        radiation_on
        * (
            0.38 * dna_repair_inhibition
            + 0.20 * oxygenation_gain
            + 0.22 * interval_short
            + 0.20 * thermal_adequate
        ),
        0.0,
        1.0,
    )
    radiosensitization = np.clip(
        0.35 * dna_repair_inhibition
        + 0.15 * oxygenation_gain * radiation_on
        + 0.25 * synergy_score
        + 0.15 * thermal_adequate
        + 0.10 * radiation_on,
        0.0,
        1.0,
    )

    rad_norm = radiation / 8.0
    cem43_norm = np.clip(np.log1p(cem43) / np.log1p(120.0), 0.0, 1.0)
    rule_kill_prior = np.clip(
        0.28 * rad_norm
        + 0.17 * dna_repair_inhibition
        + 0.12 * oxygenation_gain * radiation_on
        + 0.18 * synergy_score
        + 0.15 * cem43_norm
        + 0.10 * high_temp_risk,
        0.0,
        0.995,
    )

    out["Подавление_репарации"] = dna_repair_inhibition
    out["Оксигенация_выигрыш"] = oxygenation_gain
    out["Синергия"] = synergy_score
    out["Радиосенсибилизация"] = radiosensitization
    out["Прямая_цитотоксичность"] = direct_cytotoxicity
    out["Риск_высокой_температуры"] = high_temp_risk
    out["Оценка_правил"] = rule_kill_prior
    out["log_survival"] = log_survival_transform(out["Выживаемость"].to_numpy())
    return out


def build_bootstrap_training_frame(df: pd.DataFrame, cfg: GenerationConfig) -> pd.DataFrame:
    observed = enrich_with_rule_features(df, cfg.assumed_interval_min, cfg.assumed_hypoxia)
    base_x = observed[["Радиация", "Температура", "Время"]].to_numpy(dtype=float)
    base_log_y = observed["log_survival"].to_numpy(dtype=float)
    base_rule = observed["Оценка_правил"].to_numpy(dtype=float)

    synthetic_rows: List[Dict] = []
    for _, row in observed.iterrows():
        for _ in range(cfg.n_boot_per_row):
            radiation = float(np.clip(np.random.normal(row["Радиация"], 0.35), 0.0, 8.0))
            temperature = float(np.clip(np.random.normal(row["Температура"], 0.14), 41.8, 44.2))
            duration = float(np.clip(np.random.normal(row["Время"], 2.5), 25.0, 50.0))

            query_x = np.array([[radiation, temperature, duration]], dtype=float)
            baseline_log_surv = kernel_predict(base_x, base_log_y, query_x)[0]
            baseline_rule = kernel_predict(base_x, base_rule, query_x)[0]

            tmp_df = pd.DataFrame(
                {
                    "Радиация": [radiation],
                    "Температура": [temperature],
                    "Время": [duration],
                    "Выживаемость": [max(float(inverse_log_survival(np.array([baseline_log_surv]))[0]), EPS)],
                    "source": ["bootstrap"],
                }
            )
            tmp_df = enrich_with_rule_features(tmp_df, cfg.assumed_interval_min, cfg.assumed_hypoxia)

            kill_delta = float(tmp_df.loc[0, "Оценка_правил"] - baseline_rule)
            log_adjustment = -0.80 * kill_delta
            stochastic_noise = np.random.normal(0.0, 0.08)
            log_surv = np.clip(baseline_log_surv + log_adjustment + stochastic_noise, math.log10(EPS), -0.001)
            survival = float(inverse_log_survival(np.array([log_surv]))[0])

            out_row = {
                "Радиация": radiation,
                "Температура": temperature,
                "Время": duration,
                "Выживаемость": survival,
                "source": "bootstrap",
            }
            synthetic_rows.append(out_row)

    boot_df = pd.DataFrame(synthetic_rows)
    boot_df = enrich_with_rule_features(boot_df, cfg.assumed_interval_min, cfg.assumed_hypoxia)
    train_df = pd.concat([observed, boot_df], ignore_index=True)
    return train_df


class ConditionalVAE(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        enc_in = x_dim + cond_dim
        dec_in = latent_dim + cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def encode(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(torch.cat([x, cond], dim=1))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        return self.decoder(torch.cat([z, cond], dim=1))

    def forward(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar


def prepare_model_inputs(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[str]]:
    cond_cols = [
        "Радиация",
        "Температура",
        "Время",
        "CEM43",
        "Подавление_репарации",
        "Оксигенация_выигрыш",
        "Синергия",
        "Радиосенсибилизация",
        "Риск_высокой_температуры",
        "Оценка_правил",
    ]
    cond = train_df[cond_cols].to_numpy(dtype=np.float32)
    cond_mean = cond.mean(axis=0)
    cond_std = cond.std(axis=0) + 1e-6
    cond_scaled = (cond - cond_mean) / cond_std

    x = train_df[["log_survival"]].to_numpy(dtype=np.float32)
    x_mean = float(x.mean())
    x_std = float(x.std() + 1e-6)
    x_scaled = (x - x_mean) / x_std

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "cond_mean": cond_mean.tolist(),
        "cond_std": cond_std.tolist(),
    }
    return x_scaled, cond_scaled, stats, cond_cols


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_cvae(
    train_df: pd.DataFrame,
    cfg: GenerationConfig,
    device: torch.device,
) -> Tuple[ConditionalVAE, Dict[str, float], List[str]]:
    x_scaled, cond_scaled, stats, cond_cols = prepare_model_inputs(train_df)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    cond_tensor = torch.tensor(cond_scaled, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, cond_tensor)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = ConditionalVAE(
        x_dim=x_tensor.shape[1],
        cond_dim=cond_tensor.shape[1],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    kill_idx = cond_cols.index("Оценка_правил")

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, cond_batch in loader:
            x_batch = x_batch.to(device)
            cond_batch = cond_batch.to(device)

            recon, mu, logvar = model(x_batch, cond_batch)
            recon_loss = F.mse_loss(recon, x_batch)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            pred_log_surv = recon * stats["x_std"] + stats["x_mean"]
            pred_surv = inverse_log_survival(pred_log_surv)
            pred_surv = torch.clamp(pred_surv, 0.0, 1.0)

            cond_mean = torch.tensor(stats["cond_mean"], dtype=torch.float32, device=device)
            cond_std = torch.tensor(stats["cond_std"], dtype=torch.float32, device=device)
            cond_unscaled = cond_batch * cond_std + cond_mean
            kill_prior = cond_unscaled[:, kill_idx : kill_idx + 1]
            upper_survival_bound = torch.clamp(1.0 - 0.90 * kill_prior, 0.0, 1.0)
            rule_penalty = torch.mean(F.relu(pred_surv - upper_survival_bound) ** 2)

            loss = recon_loss + cfg.beta_kl * kl + cfg.rule_penalty_weight * rule_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if epoch % 50 == 0 or epoch == cfg.epochs - 1:
            avg_loss = epoch_loss / max(len(loader), 1)
            print(f"epoch={epoch:03d} loss={avg_loss:.5f}")

    return model, stats, cond_cols


def sample_conditions(
    observed_df: pd.DataFrame,
    n_samples: int,
    cfg: GenerationConfig,
) -> pd.DataFrame:
    rows: List[Dict] = []
    observed_df = observed_df.copy()
    for _ in range(n_samples):
        base = observed_df.sample(n=1, replace=True).iloc[0]
        mix_mode = np.random.rand()
        if mix_mode < 0.75:
            radiation = float(np.clip(np.random.normal(base["Радиация"], 0.45), 0.0, 8.0))
            temperature = float(np.clip(np.random.normal(base["Температура"], 0.20), 41.8, 44.2))
            duration = float(np.clip(np.random.normal(base["Время"], 3.0), 25.0, 50.0))
        else:
            radiation = float(np.clip(np.random.uniform(0.0, 8.0), 0.0, 8.0))
            temperature = float(np.clip(np.random.uniform(41.9, 44.1), 41.8, 44.2))
            duration = float(np.clip(np.random.uniform(28.0, 48.0), 25.0, 50.0))

        rows.append(
            {
                "Радиация": radiation,
                "Температура": temperature,
                "Время": duration,
                "Выживаемость": 0.0,
                "source": "synthetic_condition",
            }
        )
    return pd.DataFrame(rows)


def build_condition_tensor(df: pd.DataFrame, cond_cols: List[str], stats: Dict[str, float]) -> Tensor:
    cond = df[cond_cols].to_numpy(dtype=np.float32)
    cond_mean = np.asarray(stats["cond_mean"], dtype=np.float32)
    cond_std = np.asarray(stats["cond_std"], dtype=np.float32)
    cond_scaled = (cond - cond_mean) / cond_std
    return torch.tensor(cond_scaled, dtype=torch.float32)


def generate_synthetic_dataset(
    model: ConditionalVAE,
    observed_df: pd.DataFrame,
    stats: Dict[str, float],
    cond_cols: List[str],
    cfg: GenerationConfig,
    device: torch.device,
) -> pd.DataFrame:
    cond_df = sample_conditions(observed_df, cfg.n_synthetic, cfg)
    cond_df = enrich_with_rule_features(cond_df, cfg.assumed_interval_min, cfg.assumed_hypoxia)

    cond_tensor = build_condition_tensor(cond_df, cond_cols, stats).to(device)
    model.eval()
    with torch.no_grad():
        mc_samples = 32
        preds: List[np.ndarray] = []
        for _ in range(mc_samples):
            z = torch.randn((cond_tensor.shape[0], cfg.latent_dim), device=device)
            recon = model.decode(z, cond_tensor)
            log_surv = recon.cpu().numpy() * stats["x_std"] + stats["x_mean"]
            surv = inverse_log_survival(log_surv).reshape(-1)
            preds.append(surv)

    pred_matrix = np.stack(preds, axis=1)
    median_surv = np.median(pred_matrix, axis=1)
    uncertainty = np.std(pred_matrix, axis=1)

    cond_df["Выживаемость"] = np.clip(median_surv, 0.0, 1.0)
    cond_df["Неопределенность_CVAE"] = uncertainty
    cond_df["Источник"] = "CVAE+merged_rules"

    upper_bound = np.clip(1.0 - 0.90 * cond_df["Оценка_правил"].to_numpy(dtype=float), 0.0, 1.0)
    gap = np.clip(cond_df["Выживаемость"].to_numpy(dtype=float) - upper_bound, 0.0, None)
    cond_df["RuleConsistency"] = np.clip(1.0 - gap, 0.0, 1.0)

    cond_df = cond_df[cond_df["RuleConsistency"] >= 0.85].copy()
    cond_df = cond_df.sort_values(
        by=["Радиация", "Температура", "Время", "Выживаемость"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return cond_df


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_distribution_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Радиация", "Температура", "Время", "Выживаемость"]
    rows: List[Dict[str, float | str]] = []
    for col in cols:
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
    cols = ["Радиация", "Температура", "Время", "Выживаемость"]
    metrics: Dict[str, float] = {}
    for method in ("pearson", "spearman"):
        real_corr = real_df[cols].corr(method=method).to_numpy(dtype=float)
        synth_corr = synthetic_df[cols].corr(method=method).to_numpy(dtype=float)
        diff = real_corr - synth_corr
        metrics[f"{method}_correlation_frobenius"] = float(np.linalg.norm(diff, ord="fro"))
        metrics[f"{method}_correlation_mean_abs_diff"] = float(np.mean(np.abs(diff)))
    return metrics


def compute_separability_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, cfg: GenerationConfig) -> Dict[str, float]:
    cols = ["Радиация", "Температура", "Время", "Выживаемость"]
    real = real_df[cols].copy()
    n_real = len(real)
    aucs: List[float] = []

    for round_idx in range(cfg.discriminator_rounds):
        synth = synthetic_df[cols].sample(n=n_real, replace=len(synthetic_df) < n_real, random_state=cfg.seed + round_idx)
        X = pd.concat([real, synth], ignore_index=True)
        y = np.array([0] * len(real) + [1] * len(synth))
        cv = StratifiedKFold(n_splits=min(5, n_real), shuffle=True, random_state=cfg.seed + round_idx)
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=2000, random_state=cfg.seed + round_idx)),
            ]
        )
        probas = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
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


def compute_utility_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, cfg: GenerationConfig) -> Dict[str, float]:
    feature_cols = ["Радиация", "Температура", "Время"]
    X_real = real_df[feature_cols].to_numpy(dtype=float)
    y_real = real_df["Выживаемость"].to_numpy(dtype=float)
    X_synth = synthetic_df[feature_cols].to_numpy(dtype=float)
    y_synth = synthetic_df["Выживаемость"].to_numpy(dtype=float)

    tstr_model = RandomForestRegressor(
        n_estimators=400,
        random_state=cfg.seed,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    tstr_model.fit(X_synth, y_synth)
    y_pred_tstr = tstr_model.predict(X_real)

    loo = LeaveOneOut()
    y_pred_trtr = np.zeros_like(y_real, dtype=float)
    for train_idx, test_idx in loo.split(X_real):
        model = RandomForestRegressor(
            n_estimators=400,
            random_state=cfg.seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(X_real[train_idx], y_real[train_idx])
        y_pred_trtr[test_idx] = model.predict(X_real[test_idx])

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
    cols = ["Радиация", "Температура", "Время", "Выживаемость"]
    combined = pd.concat([real_df[cols], synthetic_df[cols]], ignore_index=True)
    scaler = StandardScaler().fit(combined)
    real_scaled = scaler.transform(real_df[cols])
    synth_scaled = scaler.transform(synthetic_df[cols])

    nn_real_to_synth = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    dist_real_to_synth = nn_real_to_synth.kneighbors(real_scaled, return_distance=True)[0].ravel()

    nn_synth_to_real = NearestNeighbors(n_neighbors=1).fit(real_scaled)
    dist_synth_to_real = nn_synth_to_real.kneighbors(synth_scaled, return_distance=True)[0].ravel()

    duplicate_real_rows = set(map(tuple, np.round(real_df[cols].to_numpy(dtype=float), 6)))
    synth_rows = list(map(tuple, np.round(synthetic_df[cols].to_numpy(dtype=float), 6)))
    duplicate_rate = float(np.mean([row in duplicate_real_rows for row in synth_rows]))

    support_violation = {}
    for col in cols:
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
    cfg: GenerationConfig,
    outdir: Path,
) -> List[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    minimal_cols = ["Радиация", "Температура", "Время", "Выживаемость"]
    real_min = real_df[minimal_cols].copy()
    synth_min = synthetic_df[minimal_cols].copy()

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


def save_outputs(
    train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    cfg: GenerationConfig,
    rules_meta: Dict,
    outdir: Path,
    evaluation_files: List[str] | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "rule_guided_training_frame.csv", index=False, encoding="utf-8-sig")
    minimal_cols = ["Радиация", "Температура", "Время", "Выживаемость"]
    synthetic_minimal = synthetic_df[minimal_cols].copy()
    synthetic_minimal.to_csv(outdir / "cvae_synthetic_dataset.csv", index=False, encoding="utf-8-sig")
    synthetic_df.to_csv(outdir / "cvae_synthetic_dataset_full.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "generator": "rule_guided_conditional_vae",
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "n_boot_per_row": cfg.n_boot_per_row,
        "n_synthetic_requested": cfg.n_synthetic,
        "n_synthetic_saved": int(len(synthetic_df)),
        "assumed_interval_min": cfg.assumed_interval_min,
        "assumed_hypoxia": cfg.assumed_hypoxia,
        "rules_file": "knowledge_base/merged_rules.json",
        "used_merged_rules": [rule["id"] for rule in rules_meta.get("merged_rules", [])],
        "evaluation_outputs": evaluation_files or [],
        "columns_minimal": minimal_cols,
        "columns_full": list(synthetic_df.columns),
    }
    with (outdir / "cvae_run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate a rule-guided synthetic dataset using Conditional VAE.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-kl", type=float, default=0.02)
    parser.add_argument("--rule-penalty-weight", type=float, default=0.30)
    parser.add_argument("--n-boot-per-row", type=int, default=80)
    parser.add_argument("--n-synthetic", type=int, default=1000)
    parser.add_argument("--assumed-interval-min", type=float, default=0.0)
    parser.add_argument("--assumed-hypoxia", type=float, default=0.70)
    parser.add_argument("--discriminator-rounds", type=int, default=10)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--outdir", type=str, default="benchmarks/results/synthetic_data")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    return GenerationConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        beta_kl=args.beta_kl,
        rule_penalty_weight=args.rule_penalty_weight,
        n_boot_per_row=args.n_boot_per_row,
        n_synthetic=args.n_synthetic,
        assumed_interval_min=args.assumed_interval_min,
        assumed_hypoxia=args.assumed_hypoxia,
        run_evaluation=not args.skip_evaluation,
        discriminator_rounds=args.discriminator_rounds,
        outdir=args.outdir,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = choose_device(cfg.device)

    project_root = Path(__file__).resolve().parents[1]
    rules_path = project_root / "knowledge_base" / "merged_rules.json"
    rules_meta = load_merged_rules(rules_path)

    raw_df = make_raw_frame()
    train_df = build_bootstrap_training_frame(raw_df, cfg)
    model, stats, cond_cols = train_cvae(train_df, cfg, device)
    synthetic_df = generate_synthetic_dataset(model, raw_df, stats, cond_cols, cfg, device)

    outdir = project_root / cfg.outdir
    evaluation_files = evaluate_synthetic_quality(raw_df, synthetic_df, cfg, outdir) if cfg.run_evaluation else []
    save_outputs(train_df, synthetic_df, cfg, rules_meta, outdir, evaluation_files=evaluation_files)
    print(f"saved training frame to: {outdir / 'rule_guided_training_frame.csv'}")
    print(f"saved synthetic dataset to: {outdir / 'cvae_synthetic_dataset.csv'}")
    print(f"saved metadata to: {outdir / 'cvae_run_metadata.json'}")
    for eval_path in evaluation_files:
        print(f"saved evaluation output to: {eval_path}")


if __name__ == "__main__":
    main()
