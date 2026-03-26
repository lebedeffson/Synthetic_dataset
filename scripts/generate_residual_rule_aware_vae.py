from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from generate_cell_level_article_guided_dataset import (
    CellLevelConfig,
    RADIATION_LEVELS,
    THERMAL_CONDITIONS,
    _single_pass_row_projection,
    blocks_to_dataframe,
    build_prior_mean_matrix,
    build_log_survival_matrix,
    calibrate_blocks_to_targets,
    estimate_noise_matrix,
    generate_block_samples,
    log_survival_transform,
    make_real_dataframe,
    project_monotone_matrix,
    save_outputs,
)
from validate_cell_level_datasets import evaluate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = "benchmarks/residual_rule_aware_vae"
HIGH_COMBINED_LABELS = {"43C_45min", "44C_30min"}


@dataclass
class ResidualRuleAwareVAEConfig:
    seed: int = 42
    n_synthetic: int = 1000
    teacher_blocks: int = 768
    batch_size: int = 64
    epochs: int = 320
    latent_dim: int = 6
    hidden_dim: int = 128
    lr: float = 1e-3
    beta_kl: float = 0.02
    kl_warmup_epochs: int = 60
    lambda_rule: float = 2.0
    lambda_center: float = 1.0
    lambda_var: float = 0.25
    lambda_smooth: float = 0.02
    sigma_scale: float = 2.5
    device: str = "cpu"
    outdir: str = DEFAULT_OUTDIR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_teacher_blocks(cfg: ResidualRuleAwareVAEConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    real_df = make_real_dataframe()
    target_matrix = build_log_survival_matrix(real_df)
    prior_matrix = build_prior_mean_matrix(real_df, CellLevelConfig(seed=cfg.seed))
    sigma_matrix = estimate_noise_matrix(real_df, prior_matrix, CellLevelConfig(seed=cfg.seed))
    teacher_cfg = CellLevelConfig(
        seed=cfg.seed + 19,
        n_synthetic=cfg.teacher_blocks * target_matrix.size,
        explainability_mode="off",
        save_explainability_plots=False,
    )
    blocks, _ = generate_block_samples(prior_matrix, sigma_matrix, teacher_cfg)
    blocks, _ = calibrate_blocks_to_targets(blocks, target_matrix, teacher_cfg, [])
    return real_df, target_matrix, prior_matrix, sigma_matrix, np.stack(blocks, axis=0)


def build_rule_feature_vector() -> np.ndarray:
    rows: List[List[float]] = []
    for radiation in RADIATION_LEVELS:
        for thermal_meta in THERMAL_CONDITIONS:
            temperature = float(thermal_meta["temperature"])
            duration = float(thermal_meta["time"])
            thermal_rank = float(thermal_meta["thermal_rank"])
            cem43 = float(thermal_meta["time"] * ((0.25 if temperature < 43.0 else 0.5) ** (43.0 - temperature)))
            cl2 = 1.0 if 41.0 <= temperature <= 43.0 and 30.0 <= duration <= 60.0 else 0.0
            cl4 = 1.0 if temperature >= 43.0 else 0.0
            cl5 = 1.0 if radiation >= 6.0 and cem43 >= 45.0 else 0.0
            rows.append(
                [
                    radiation / 8.0,
                    (temperature - 42.0) / 2.0,
                    (duration - 30.0) / 15.0,
                    np.log1p(cem43) / np.log1p(60.0),
                    thermal_rank / 2.0,
                    cl2,
                    cl4,
                    cl5,
                ]
            )
    return np.asarray(rows, dtype=np.float32).reshape(-1)


def build_cell_weights() -> np.ndarray:
    weights = np.ones((len(RADIATION_LEVELS), len(THERMAL_CONDITIONS)), dtype=np.float32)
    weights[0, 0] = 3.0
    weights[0, 1] = 2.0
    weights[-1, -1] = 3.0
    weights[-2:, 1:] = np.maximum(weights[-2:, 1:], 2.5)
    weights[1, 0] = 2.0
    weights[0, 2] = 2.0
    return weights.reshape(-1)


def build_high_dose_mask() -> np.ndarray:
    mask_rows: List[bool] = []
    for radiation in RADIATION_LEVELS:
        for thermal_meta in THERMAL_CONDITIONS:
            temperature = float(thermal_meta["temperature"])
            duration = float(thermal_meta["time"])
            cem43 = float(duration * ((0.25 if temperature < 43.0 else 0.5) ** (43.0 - temperature)))
            mask_rows.append(bool(radiation >= 6.0 and cem43 >= 45.0))
    return np.asarray(mask_rows, dtype=bool)


class ResidualRuleAwareBlockVAE(nn.Module):
    def __init__(self, residual_dim: int, cond_dim: int, latent_dim: int, hidden_dim: int, sigma_scale: float) -> None:
        super().__init__()
        self.sigma_scale = float(sigma_scale)
        encoder_in = residual_dim + cond_dim
        decoder_in = latent_dim + cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(encoder_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, residual_dim),
        )

    def encode(self, residual: Tensor, cond: Tensor, sigma: Tensor) -> Tuple[Tensor, Tensor]:
        residual_norm = residual / sigma
        h = self.encoder(torch.cat([residual_norm, cond], dim=1))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, cond: Tensor, sigma: Tensor) -> Tensor:
        raw = self.decoder(torch.cat([z, cond], dim=1))
        return self.sigma_scale * sigma * torch.tanh(raw)

    def forward(self, residual: Tensor, cond: Tensor, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(residual, cond, sigma)
        z = self.reparameterize(mu, logvar)
        pred_residual = self.decode(z, cond, sigma)
        return pred_residual, mu, logvar


def rule_penalty(pred_log_blocks: Tensor, high_dose_mask: Tensor, high_cap_log: float) -> Tuple[Tensor, Dict[str, float]]:
    blocks = pred_log_blocks.view(-1, len(RADIATION_LEVELS), len(THERMAL_CONDITIONS))
    rad_diff = blocks[:, 1:, :] - blocks[:, :-1, :]
    therm_diff = blocks[:, :, 1:] - blocks[:, :, :-1]
    rad_penalty = torch.mean(F.relu(rad_diff) ** 2)
    therm_penalty = torch.mean(F.relu(therm_diff) ** 2)

    high_vals = blocks.view(blocks.shape[0], -1)[:, high_dose_mask]
    if high_vals.numel():
        high_penalty = torch.mean(F.relu(high_vals - high_cap_log) ** 2)
    else:
        high_penalty = torch.tensor(0.0, device=pred_log_blocks.device)

    total = rad_penalty + therm_penalty + high_penalty
    return total, {
        "rule_rad_penalty": float(rad_penalty.detach().cpu()),
        "rule_therm_penalty": float(therm_penalty.detach().cpu()),
        "rule_high_penalty": float(high_penalty.detach().cpu()),
    }


def train_model(
    teacher_blocks: np.ndarray,
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    cfg: ResidualRuleAwareVAEConfig,
) -> Tuple[ResidualRuleAwareBlockVAE, Dict, pd.DataFrame]:
    device = torch.device(cfg.device)
    mu_flat_np = mu_matrix.reshape(-1).astype(np.float32)
    sigma_flat_np = np.clip(sigma_matrix.reshape(-1).astype(np.float32), 1e-3, None)
    teacher_mean_np = teacher_blocks.reshape(len(teacher_blocks), -1).astype(np.float32).mean(axis=0)
    residual_targets_np = teacher_blocks.reshape(len(teacher_blocks), -1).astype(np.float32) - mu_flat_np[None, :]
    cond_vector_np = build_rule_feature_vector()
    weights_np = build_cell_weights().astype(np.float32)
    target_std_np = np.clip(residual_targets_np.std(axis=0), 0.003, None).astype(np.float32)
    high_dose_mask_np = build_high_dose_mask()

    dataset = TensorDataset(torch.tensor(residual_targets_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = ResidualRuleAwareBlockVAE(
        residual_dim=residual_targets_np.shape[1],
        cond_dim=cond_vector_np.shape[0],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        sigma_scale=cfg.sigma_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    mu_flat = torch.tensor(mu_flat_np, dtype=torch.float32, device=device).unsqueeze(0)
    sigma_flat = torch.tensor(sigma_flat_np, dtype=torch.float32, device=device).unsqueeze(0)
    teacher_mean = torch.tensor(teacher_mean_np, dtype=torch.float32, device=device)
    cond_vector = torch.tensor(cond_vector_np, dtype=torch.float32, device=device).unsqueeze(0)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device).unsqueeze(0)
    target_std = torch.tensor(target_std_np, dtype=torch.float32, device=device)
    high_dose_mask = torch.tensor(high_dose_mask_np, dtype=torch.bool, device=device)
    high_cap_log = float(log_survival_transform(np.array([0.02]))[0])

    history_rows: List[Dict] = []
    for epoch in range(cfg.epochs):
        beta = cfg.beta_kl * min(1.0, float(epoch + 1) / max(cfg.kl_warmup_epochs, 1))
        epoch_stats = {
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "rule_loss": 0.0,
            "center_loss": 0.0,
            "var_loss": 0.0,
            "smooth_loss": 0.0,
            "total_loss": 0.0,
            "rule_rad_penalty": 0.0,
            "rule_therm_penalty": 0.0,
            "rule_high_penalty": 0.0,
            "batches": 0,
        }

        for (residual_target,) in loader:
            residual_target = residual_target.to(device)
            batch_size = residual_target.shape[0]
            cond_batch = cond_vector.repeat(batch_size, 1)
            sigma_batch = sigma_flat.repeat(batch_size, 1)
            mu_batch = mu_flat.repeat(batch_size, 1)

            pred_residual, mu_latent, logvar = model(residual_target, cond_batch, sigma_batch)
            pred_log = mu_batch + pred_residual
            target_log = mu_batch + residual_target

            recon_loss = torch.mean(weights * (pred_log - target_log) ** 2)
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mu_latent.pow(2) - logvar.exp())
            rule_loss, rule_parts = rule_penalty(pred_log, high_dose_mask, high_cap_log)
            center_loss = torch.mean(weights.squeeze(0) * (pred_log.mean(dim=0) - teacher_mean) ** 2)
            var_loss = torch.mean(torch.abs(torch.std(pred_residual, dim=0, unbiased=False) - target_std))
            smooth_loss = torch.mean((pred_residual / (cfg.sigma_scale * sigma_batch)) ** 2)

            loss = (
                recon_loss
                + beta * kl_loss
                + cfg.lambda_rule * rule_loss
                + cfg.lambda_center * center_loss
                + cfg.lambda_var * var_loss
                + cfg.lambda_smooth * smooth_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_stats["recon_loss"] += float(recon_loss.detach().cpu())
            epoch_stats["kl_loss"] += float(kl_loss.detach().cpu())
            epoch_stats["rule_loss"] += float(rule_loss.detach().cpu())
            epoch_stats["center_loss"] += float(center_loss.detach().cpu())
            epoch_stats["var_loss"] += float(var_loss.detach().cpu())
            epoch_stats["smooth_loss"] += float(smooth_loss.detach().cpu())
            epoch_stats["total_loss"] += float(loss.detach().cpu())
            epoch_stats["rule_rad_penalty"] += rule_parts["rule_rad_penalty"]
            epoch_stats["rule_therm_penalty"] += rule_parts["rule_therm_penalty"]
            epoch_stats["rule_high_penalty"] += rule_parts["rule_high_penalty"]
            epoch_stats["batches"] += 1

        denom = max(epoch_stats.pop("batches"), 1)
        epoch_record = {"epoch": epoch, "beta_kl": beta}
        for key, value in epoch_stats.items():
            epoch_record[key] = value / denom
        history_rows.append(epoch_record)
        if epoch % 80 == 0 or epoch == cfg.epochs - 1:
            print(
                f"[residual_vae] epoch={epoch:03d} total={epoch_record['total_loss']:.6f} "
                f"recon={epoch_record['recon_loss']:.6f} rule={epoch_record['rule_loss']:.6f} "
                f"center={epoch_record['center_loss']:.6f}"
            )

    train_info = {
        "parameters": count_parameters(model),
        "epochs": cfg.epochs,
        "cond_dim": int(cond_vector_np.shape[0]),
        "residual_dim": int(residual_targets_np.shape[1]),
        "loss_weights": {
            "beta_kl": cfg.beta_kl,
            "lambda_rule": cfg.lambda_rule,
            "lambda_center": cfg.lambda_center,
            "lambda_var": cfg.lambda_var,
            "lambda_smooth": cfg.lambda_smooth,
        },
        "teacher_mean_anchor": "teacher_block_mean",
    }
    history_df = pd.DataFrame(history_rows)
    return model, train_info, history_df


def sample_blocks(
    model: ResidualRuleAwareBlockVAE,
    sigma_matrix: np.ndarray,
    mu_matrix: np.ndarray,
    target_residual_mean: np.ndarray,
    n_blocks: int,
    cfg: ResidualRuleAwareVAEConfig,
) -> List[np.ndarray]:
    device = torch.device(cfg.device)
    model.eval()
    sigma_flat = torch.tensor(np.clip(sigma_matrix.reshape(-1), 1e-3, None), dtype=torch.float32, device=device).unsqueeze(0)
    cond_vector = torch.tensor(build_rule_feature_vector(), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        z = torch.randn((n_blocks, cfg.latent_dim), device=device)
        residuals = model.decode(z, cond_vector.repeat(n_blocks, 1), sigma_flat.repeat(n_blocks, 1)).cpu().numpy()

    residuals = residuals - residuals.mean(axis=0, keepdims=True) + target_residual_mean.reshape(1, -1)
    raw_flat = residuals + mu_matrix.reshape(-1)[None, :]
    return [row.reshape(mu_matrix.shape).astype(float) for row in raw_flat]


def apply_block_constraints(
    raw_blocks: List[np.ndarray],
    prior_matrix: np.ndarray,
    target_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    cfg: ResidualRuleAwareVAEConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    min_log = float(log_survival_transform(np.array([0.0]))[0])
    max_log = float(log_survival_transform(np.array([0.53]))[0])
    high_combined_log_cap = float(log_survival_transform(np.array([0.02]))[0])
    sigma_scale_matrix = cfg.sigma_scale * sigma_matrix

    clipped_blocks: List[np.ndarray] = []
    constrained_blocks: List[np.ndarray] = []
    records: List[Dict] = []

    for block_id, matrix in enumerate(raw_blocks):
        before = np.clip(np.asarray(matrix, dtype=float), min_log, max_log)
        clipped_blocks.append(before.copy())

        after_projection = project_monotone_matrix(before, CellLevelConfig().projection_max_iter)
        after_cap = after_projection.copy()
        for i, radiation in enumerate(RADIATION_LEVELS):
            for j, thermal_meta in enumerate(THERMAL_CONDITIONS):
                if radiation >= 6.0 and thermal_meta["label"] in HIGH_COMBINED_LABELS:
                    after_cap[i, j] = min(after_cap[i, j], high_combined_log_cap)
        after_cap = project_monotone_matrix(after_cap, CellLevelConfig().projection_max_iter)
        after_cap = np.clip(after_cap, min_log, max_log)

        delta_proj = after_projection - before
        delta_cap_val = after_cap - after_projection
        row_only = _single_pass_row_projection(before)
        delta_proj_rad = row_only - before
        delta_proj_therm = delta_proj - delta_proj_rad
        residual = before - prior_matrix

        rad_diff = before[1:, :] - before[:-1, :]
        therm_diff = before[:, 1:] - before[:, :-1]
        high_mask = np.zeros_like(before, dtype=bool)
        for i, radiation in enumerate(RADIATION_LEVELS):
            for j, thermal_meta in enumerate(THERMAL_CONDITIONS):
                temperature = float(thermal_meta["temperature"])
                duration = float(thermal_meta["time"])
                cem43 = float(duration * ((0.25 if temperature < 43.0 else 0.5) ** (43.0 - temperature)))
                high_mask[i, j] = bool(radiation >= 6.0 and cem43 >= 45.0)

        high_violation = np.maximum(before[high_mask] - high_combined_log_cap, 0.0) if np.any(high_mask) else np.array([0.0])

        records.append(
            {
                "block_id": block_id,
                "generator_family": "residual_rule_aware_vae",
                "sampled_before_constraints_mean": float(before.mean()),
                "after_projection_mean": float(after_projection.mean()),
                "after_cap_mean": float(after_cap.mean()),
                "after_calibration_mean": None,
                "final_mean": None,
                "delta_projection_mean": float(np.abs(delta_proj).mean()),
                "delta_projection_radiation_mean": float(np.abs(delta_proj_rad).mean()),
                "delta_projection_thermal_mean": float(np.abs(delta_proj_therm).mean()),
                "delta_cap_mean": float(np.abs(delta_cap_val).mean()),
                "delta_calibration_mean": None,
                "total_adjustment_abs_mean": None,
                "constraint_pressure_score": None,
                "per_cell_delta_abs": np.abs(delta_proj) + np.abs(delta_cap_val),
                "projection_active_rate": float(np.mean(np.abs(delta_proj) > 1e-8)),
                "cap_active_rate": float(np.mean(np.abs(delta_cap_val) > 1e-8)),
                "calibration_active_rate": None,
                "rule_CL1_active": bool(np.any(np.abs(delta_proj_rad) > 1e-8)),
                "rule_CL3_active": bool(np.any(np.abs(delta_proj_therm) > 1e-8)),
                "rule_CL5_active": bool(np.any(np.abs(delta_cap_val) > 1e-8)),
                "raw_radiation_violation_mean": float(np.mean(np.maximum(rad_diff, 0.0))),
                "raw_thermal_violation_mean": float(np.mean(np.maximum(therm_diff, 0.0))),
                "raw_highdose_violation_mean": float(np.mean(high_violation)),
                "residual_budget_utilization": float(np.mean(np.abs(residual) / np.clip(sigma_scale_matrix, 1e-6, None))),
                "explainability_status": "pass",
                "reject_reason": "",
                "_after_cap_matrix": after_cap.copy(),
            }
        )
        constrained_blocks.append(after_cap)

    cell_cfg = CellLevelConfig(
        seed=cfg.seed,
        n_synthetic=cfg.n_synthetic,
        outdir=cfg.outdir,
        explainability_mode="log_only",
        save_explainability_plots=False,
    )
    constrained_blocks, records = calibrate_blocks_to_targets(constrained_blocks, target_matrix, cell_cfg, records)
    return clipped_blocks, constrained_blocks, records


def summarize_records(records: List[Dict]) -> Dict:
    if not records:
        return {}
    pressures = [r["constraint_pressure_score"] for r in records if r.get("constraint_pressure_score") is not None]
    return {
        "blocks_total": len(records),
        "pass_count": len(records),
        "reject_count": 0,
        "pass_rate": 1.0,
        "reject_rate": 0.0,
        "mean_delta_projection": float(np.mean([r["delta_projection_mean"] for r in records])),
        "mean_delta_cap": float(np.mean([r["delta_cap_mean"] for r in records])),
        "mean_delta_calibration": float(np.mean([r["delta_calibration_mean"] for r in records if r["delta_calibration_mean"] is not None])),
        "mean_total_adjustment_abs": float(np.mean([r["total_adjustment_abs_mean"] for r in records if r["total_adjustment_abs_mean"] is not None])),
        "mean_constraint_pressure": float(np.mean(pressures)) if pressures else 0.0,
        "p95_constraint_pressure": float(np.percentile(pressures, 95)) if pressures else 0.0,
        "mean_projection_active_rate": float(np.mean([r["projection_active_rate"] for r in records])),
        "mean_cap_active_rate": float(np.mean([r["cap_active_rate"] for r in records])),
        "mean_calibration_active_rate": float(np.mean([r["calibration_active_rate"] for r in records if r["calibration_active_rate"] is not None])),
        "mean_raw_radiation_violation": float(np.mean([r["raw_radiation_violation_mean"] for r in records])),
        "mean_raw_thermal_violation": float(np.mean([r["raw_thermal_violation_mean"] for r in records])),
        "mean_raw_highdose_violation": float(np.mean([r["raw_highdose_violation_mean"] for r in records])),
        "mean_residual_budget_utilization": float(np.mean([r["residual_budget_utilization"] for r in records])),
    }


def write_report(outdir: Path, metrics: Dict, independent: Dict, metadata: Dict) -> None:
    report_lines = [
        "# Residual Rule-Aware Block VAE",
        "",
        "This experimental generator models residual stochasticity around the rule-guided matrix prior instead of generating the full block from scratch.",
        "",
        "## Core Idea",
        "",
        "- prior mean surface comes from the rule-guided matrix generator;",
        "- the VAE models only residual variation around that surface;",
        "- rule penalties are applied during training on raw outputs;",
        "- projection/cap/calibration remain as a safety layer after sampling.",
        "",
        "## Final Metrics",
        "",
        f"- `mean_wasserstein_normalized`: {metrics['mean_wasserstein_normalized']:.4f}",
        f"- `mean_ks_statistic`: {metrics['mean_ks_statistic']:.4f}",
        f"- `tstr_mae`: {metrics['tstr_mae']:.4f}",
        f"- `tstr_r2`: {metrics['tstr_r2']:.4f}",
        f"- `local_mean_abs_error`: {independent['local_mean_abs_error']:.4f}",
        f"- `local_max_abs_error`: {independent['local_max_abs_error']:.4f}",
        f"- `independent_article_compliance_mean`: {independent['independent_article_compliance_mean']:.4f}",
        "",
        "## Explainability",
        "",
        f"- `mean_constraint_pressure`: {metadata['explainability_summary']['mean_constraint_pressure']:.4f}",
        f"- `mean_raw_radiation_violation`: {metadata['explainability_summary']['mean_raw_radiation_violation']:.6f}",
        f"- `mean_raw_thermal_violation`: {metadata['explainability_summary']['mean_raw_thermal_violation']:.6f}",
        f"- `mean_residual_budget_utilization`: {metadata['explainability_summary']['mean_residual_budget_utilization']:.4f}",
        "",
        "## Interpretation",
        "",
        "- This model is mathematically more honest than a free-form neural generator for `n=15` real points because it keeps the rule-guided surface as prior structure.",
        "- The main question is not only fidelity, but how much safety-layer correction remains necessary after raw generation.",
        "",
    ]
    (outdir / "residual_rule_aware_vae_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    cfg = ResidualRuleAwareVAEConfig()
    set_seed(cfg.seed)

    real_df, target_matrix, prior_matrix, sigma_matrix, teacher_blocks = make_teacher_blocks(cfg)
    residual_targets = teacher_blocks.reshape(len(teacher_blocks), -1).astype(np.float32) - prior_matrix.reshape(1, -1).astype(np.float32)
    target_residual_mean = residual_targets.mean(axis=0)
    model, train_info, history_df = train_model(teacher_blocks, prior_matrix, sigma_matrix, cfg)

    n_blocks = int(math.ceil(cfg.n_synthetic / target_matrix.size))
    raw_blocks = sample_blocks(model, sigma_matrix, prior_matrix, target_residual_mean, n_blocks, cfg)
    clipped_blocks, constrained_blocks, records = apply_block_constraints(raw_blocks, prior_matrix, target_matrix, sigma_matrix, cfg)

    cell_cfg = CellLevelConfig(
        seed=cfg.seed,
        n_synthetic=cfg.n_synthetic,
        outdir=cfg.outdir,
        explainability_mode="log_only",
        save_explainability_plots=False,
    )
    raw_df = blocks_to_dataframe(clipped_blocks, cell_cfg)
    raw_validation = evaluate_dataset(raw_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(), "residual_rule_aware_vae_raw")
    synthetic_df = blocks_to_dataframe(constrained_blocks, cell_cfg)

    metadata = {
        "generator": "residual_rule_aware_block_vae",
        "seed": cfg.seed,
        "model_config": asdict(cfg),
        "train_info": train_info,
        "rules_file": "knowledge_base/cell_level_rules.json",
        "teacher_source": "rule-guided matrix bootstrap blocks",
        "teacher_blocks": cfg.teacher_blocks,
        "raw_independent_validation": raw_validation,
        "explainability_records": records,
        "explainability_summary": summarize_records(records),
        "target_matrix_log10": target_matrix.tolist(),
        "prior_mu_matrix_log10": prior_matrix.tolist(),
        "prior_sigma_matrix_log10": sigma_matrix.tolist(),
    }

    outdir = save_outputs(real_df, synthetic_df, metadata, cell_cfg)
    history_df.to_csv(outdir / "training_history.csv", index=False, encoding="utf-8-sig")
    with (outdir / "raw_independent_cell_level_validation.json").open("w", encoding="utf-8") as fh:
        json.dump(raw_validation, fh, ensure_ascii=False, indent=2)

    independent = evaluate_dataset(synthetic_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(), "residual_rule_aware_vae")
    with (outdir / "independent_cell_level_validation.json").open("w", encoding="utf-8") as fh:
        json.dump(independent, fh, ensure_ascii=False, indent=2)

    with (outdir / "evaluation_metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    write_report(outdir, metrics, independent, metadata)
    print(f"saved residual rule-aware VAE outputs to: {outdir}")


if __name__ == "__main__":
    main()
