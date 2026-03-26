from __future__ import annotations

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
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from generate_cell_level_article_guided_dataset import (
    CellLevelConfig,
    RADIATION_LEVELS,
    THERMAL_CONDITIONS,
    _single_pass_row_projection,
    blocks_to_dataframe,
    build_generation_artifacts,
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
from generate_residual_rule_aware_vae import (
    ResidualRuleAwareVAEConfig,
    apply_block_constraints as apply_residual_constraints,
    sample_blocks as sample_residual_blocks,
    summarize_records as summarize_residual_records,
    train_model as train_residual_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "generator_family_benchmark"

HIGH_COMBINED_LABELS = {"43C_45min", "44C_30min"}


@dataclass
class BenchmarkConfig:
    seed: int = 42
    n_synthetic: int = 1000
    teacher_blocks: int = 768
    batch_size: int = 64
    vae_epochs: int = 260
    gan_epochs: int = 420
    diffusion_epochs: int = 280
    latent_dim: int = 8
    hidden_dim: int = 96
    diffusion_steps: int = 64
    residual_vae_epochs: int = 320
    residual_latent_dim: int = 6
    residual_hidden_dim: int = 128
    benchmark_subdir: str = "generator_family_benchmark"
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def family_cfg(base: BenchmarkConfig, family: str) -> CellLevelConfig:
    return CellLevelConfig(
        seed=base.seed,
        n_synthetic=base.n_synthetic,
        outdir=str(Path("benchmarks") / base.benchmark_subdir / family),
        explainability_mode="log_only",
        save_explainability_plots=False,
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_teacher_blocks(cfg: BenchmarkConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    real_df = make_real_dataframe()
    target_matrix = build_log_survival_matrix(real_df)
    prior_matrix = build_prior_mean_matrix(real_df, CellLevelConfig(seed=cfg.seed))
    sigma_matrix = estimate_noise_matrix(real_df, prior_matrix, CellLevelConfig(seed=cfg.seed))
    teacher_cfg = CellLevelConfig(
        seed=cfg.seed + 11,
        n_synthetic=cfg.teacher_blocks * target_matrix.size,
        explainability_mode="off",
        save_explainability_plots=False,
    )
    blocks, _ = generate_block_samples(prior_matrix, sigma_matrix, teacher_cfg)
    blocks, _ = calibrate_blocks_to_targets(blocks, target_matrix, teacher_cfg, [])
    block_array = np.stack(blocks, axis=0)
    return real_df, target_matrix, prior_matrix, sigma_matrix, block_array


def standardize_blocks(block_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = block_array.reshape(block_array.shape[0], -1).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.clip(std, 0.02, None)
    scaled = (flat - mean) / std
    return scaled, mean.astype(np.float32), std.astype(np.float32)


def unscale_samples(samples: np.ndarray, mean: np.ndarray, std: np.ndarray, matrix_shape: Tuple[int, int]) -> List[np.ndarray]:
    flat = samples * std[None, :] + mean[None, :]
    return [row.reshape(matrix_shape).astype(float) for row in flat]


class BlockVAE(nn.Module):
    def __init__(self, dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class BlockGenerator(nn.Module):
    def __init__(self, latent_dim: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class BlockDiscriminator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def sinusoidal_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(10_000.0),
            half_dim,
            device=timesteps.device,
        )
        * (-1.0)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiffusionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, time_dim: int = 32) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        return self.net(torch.cat([x, t_emb], dim=1))


def train_vae(train_x: np.ndarray, cfg: BenchmarkConfig) -> Tuple[BlockVAE, Dict]:
    device = torch.device(cfg.device)
    dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = BlockVAE(train_x.shape[1], cfg.latent_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(cfg.vae_epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.03 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if epoch % 80 == 0 or epoch == cfg.vae_epochs - 1:
            avg_loss = total_loss / max(len(loader), 1)
            print(f"[vae] epoch={epoch:03d} loss={avg_loss:.6f}")

    return model, {"parameters": count_parameters(model), "epochs": cfg.vae_epochs}


def sample_vae(model: BlockVAE, n_blocks: int, cfg: BenchmarkConfig) -> np.ndarray:
    device = torch.device(cfg.device)
    model.eval()
    with torch.no_grad():
        z = torch.randn((n_blocks, cfg.latent_dim), device=device)
        samples = model.decode(z).cpu().numpy()
    return samples


def train_gan(train_x: np.ndarray, cfg: BenchmarkConfig) -> Tuple[BlockGenerator, Dict]:
    device = torch.device(cfg.device)
    dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    generator = BlockGenerator(cfg.latent_dim, train_x.shape[1], cfg.hidden_dim).to(device)
    discriminator = BlockDiscriminator(train_x.shape[1], cfg.hidden_dim).to(device)
    g_opt = torch.optim.Adam(generator.parameters(), lr=8e-4, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=8e-4, betas=(0.5, 0.9))

    for epoch in range(cfg.gan_epochs):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.shape[0]
            real_target = torch.ones((batch_size, 1), device=device)
            fake_target = torch.zeros((batch_size, 1), device=device)

            z = torch.randn((batch_size, cfg.latent_dim), device=device)
            fake_batch = generator(z).detach()
            real_score = discriminator(real_batch)
            fake_score = discriminator(fake_batch)
            d_loss = 0.5 * (
                F.mse_loss(real_score, real_target) +
                F.mse_loss(fake_score, fake_target)
            )
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()
            d_loss_epoch += float(d_loss.item())

            z = torch.randn((batch_size, cfg.latent_dim), device=device)
            fake_batch = generator(z)
            fake_score = discriminator(fake_batch)
            g_loss = F.mse_loss(fake_score, real_target)
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()
            g_loss_epoch += float(g_loss.item())

        if epoch % 120 == 0 or epoch == cfg.gan_epochs - 1:
            d_avg = d_loss_epoch / max(len(loader), 1)
            g_avg = g_loss_epoch / max(len(loader), 1)
            print(f"[gan] epoch={epoch:03d} d_loss={d_avg:.6f} g_loss={g_avg:.6f}")

    return generator, {"parameters": count_parameters(generator), "epochs": cfg.gan_epochs}


def sample_gan(generator: BlockGenerator, n_blocks: int, cfg: BenchmarkConfig) -> np.ndarray:
    device = torch.device(cfg.device)
    generator.eval()
    with torch.no_grad():
        z = torch.randn((n_blocks, cfg.latent_dim), device=device)
        return generator(z).cpu().numpy()


def train_diffusion(train_x: np.ndarray, cfg: BenchmarkConfig) -> Tuple[DiffusionMLP, Dict]:
    device = torch.device(cfg.device)
    data = torch.tensor(train_x, dtype=torch.float32)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = DiffusionMLP(train_x.shape[1], cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    betas = torch.linspace(1e-4, 0.02, cfg.diffusion_steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    for epoch in range(cfg.diffusion_epochs):
        total_loss = 0.0
        for (x0,) in loader:
            x0 = x0.to(device)
            batch_size = x0.shape[0]
            t = torch.randint(0, cfg.diffusion_steps, (batch_size,), device=device)
            noise = torch.randn_like(x0)
            a_bar_t = alpha_bar[t].unsqueeze(1)
            x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * noise
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if epoch % 90 == 0 or epoch == cfg.diffusion_epochs - 1:
            avg_loss = total_loss / max(len(loader), 1)
            print(f"[diffusion] epoch={epoch:03d} loss={avg_loss:.6f}")

    return model, {"parameters": count_parameters(model), "epochs": cfg.diffusion_epochs}


def sample_diffusion(model: DiffusionMLP, n_blocks: int, cfg: BenchmarkConfig) -> np.ndarray:
    device = torch.device(cfg.device)
    betas = torch.linspace(1e-4, 0.02, cfg.diffusion_steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)
    posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    model.eval()
    x = torch.randn((n_blocks, 15), device=device)
    with torch.no_grad():
        for step in reversed(range(cfg.diffusion_steps)):
            t = torch.full((n_blocks,), step, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            alpha_t = alphas[step]
            alpha_bar_t = alpha_bar[step]
            coef = betas[step] / torch.sqrt(1.0 - alpha_bar_t)
            x = (x - coef * pred_noise) / torch.sqrt(alpha_t)
            if step > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(torch.clamp(posterior_var[step], min=1e-8)) * noise
    return x.cpu().numpy()


def apply_block_constraints(raw_blocks: List[np.ndarray], target_matrix: np.ndarray, cfg: CellLevelConfig, family: str) -> Tuple[List[np.ndarray], List[Dict], List[np.ndarray]]:
    min_log = float(log_survival_transform(np.array([0.0]))[0])
    max_log = float(log_survival_transform(np.array([0.53]))[0])
    high_combined_log_cap = float(log_survival_transform(np.array([0.02]))[0])

    clipped_blocks: List[np.ndarray] = []
    constrained_blocks: List[np.ndarray] = []
    records: List[Dict] = []

    for block_id, matrix in enumerate(raw_blocks):
        before = np.clip(np.asarray(matrix, dtype=float), min_log, max_log)
        clipped_blocks.append(before.copy())

        after_projection = project_monotone_matrix(before, cfg.projection_max_iter)
        after_cap = after_projection.copy()
        for i, radiation in enumerate(RADIATION_LEVELS):
            for j, thermal_meta in enumerate(THERMAL_CONDITIONS):
                if radiation >= 6.0 and thermal_meta["label"] in HIGH_COMBINED_LABELS:
                    after_cap[i, j] = min(after_cap[i, j], high_combined_log_cap)
        after_cap = project_monotone_matrix(after_cap, cfg.projection_max_iter)
        after_cap = np.clip(after_cap, min_log, max_log)

        delta_proj = after_projection - before
        delta_cap_val = after_cap - after_projection
        row_only = _single_pass_row_projection(before)
        delta_proj_rad = row_only - before
        delta_proj_therm = delta_proj - delta_proj_rad

        records.append(
            {
                "block_id": block_id,
                "generator_family": family,
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
                "per_cell_delta_abs": (np.abs(delta_proj) + np.abs(delta_cap_val)),
                "projection_active_rate": float(np.mean(np.abs(delta_proj) > 1e-8)),
                "cap_active_rate": float(np.mean(np.abs(delta_cap_val) > 1e-8)),
                "calibration_active_rate": None,
                "rule_CL1_active": bool(np.any(np.abs(delta_proj_rad) > 1e-8)),
                "rule_CL3_active": bool(np.any(np.abs(delta_proj_therm) > 1e-8)),
                "rule_CL5_active": bool(np.any(np.abs(delta_cap_val) > 1e-8)),
                "explainability_status": "pass",
                "reject_reason": "",
                "_after_cap_matrix": after_cap.copy(),
            }
        )
        constrained_blocks.append(after_cap)

    constrained_blocks, records = calibrate_blocks_to_targets(constrained_blocks, target_matrix, cfg, records)
    return clipped_blocks, constrained_blocks, records


def write_independent_validation(outdir: Path, df: pd.DataFrame, label: str) -> Dict:
    independent = evaluate_dataset(df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(), label)
    with (outdir / "independent_cell_level_validation.json").open("w", encoding="utf-8") as fh:
        json.dump(independent, fh, ensure_ascii=False, indent=2)
    return independent


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_family_bundle(
    family: str,
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metadata: Dict,
    cfg: CellLevelConfig,
) -> Tuple[Path, Dict, Dict]:
    outdir = save_outputs(real_df, synthetic_df, metadata, cfg)
    metrics = load_json(outdir / "evaluation_metrics.json")
    independent = write_independent_validation(outdir, synthetic_df, family)
    return outdir, metrics, independent


def summarize_family(
    family: str,
    display_name: str,
    architecture: str,
    training_note: str,
    outdir: Path,
    metrics: Dict,
    independent: Dict,
    metadata: Dict,
    raw_validation: Dict | None = None,
    extra: Dict | None = None,
) -> Dict:
    explain = metadata.get("explainability_summary", {})
    row = {
        "family": family,
        "display_name": display_name,
        "architecture": architecture,
        "training_note": training_note,
        "mean_wasserstein_normalized": metrics.get("mean_wasserstein_normalized"),
        "mean_ks_statistic": metrics.get("mean_ks_statistic"),
        "tstr_mae": metrics.get("tstr_mae"),
        "tstr_r2": metrics.get("tstr_r2"),
        "duplicate_rate_vs_real": metrics.get("duplicate_rate_vs_real"),
        "exact_design_support_rate": independent.get("exact_design_support_rate"),
        "local_mean_abs_error": independent.get("local_mean_abs_error"),
        "local_max_abs_error": independent.get("local_max_abs_error"),
        "radiation_monotonicity_mean_rate": independent.get("radiation_monotonicity_mean_rate"),
        "thermal_monotonicity_mean_rate": independent.get("thermal_monotonicity_mean_rate"),
        "high_combined_dose_low_survival_rate": independent.get("high_combined_dose_low_survival_rate"),
        "independent_article_compliance_mean": independent.get("independent_article_compliance_mean"),
        "mean_constraint_pressure": explain.get("mean_constraint_pressure"),
        "p95_constraint_pressure": explain.get("p95_constraint_pressure"),
        "mean_delta_projection": explain.get("mean_delta_projection"),
        "mean_delta_cap": explain.get("mean_delta_cap"),
        "mean_delta_calibration": explain.get("mean_delta_calibration"),
        "outdir": str(outdir.relative_to(PROJECT_ROOT)),
    }
    if raw_validation is not None:
        row["raw_article_compliance_mean"] = raw_validation.get("independent_article_compliance_mean")
        row["raw_radiation_monotonicity_mean_rate"] = raw_validation.get("radiation_monotonicity_mean_rate")
        row["raw_thermal_monotonicity_mean_rate"] = raw_validation.get("thermal_monotonicity_mean_rate")
    if extra:
        row.update(extra)
    return row


def train_neural_family(
    family: str,
    cfg: BenchmarkConfig,
    real_df: pd.DataFrame,
    target_matrix: np.ndarray,
    teacher_blocks: np.ndarray,
) -> Dict:
    scaled_train, mean, std = standardize_blocks(teacher_blocks)
    n_blocks = int(math.ceil(cfg.n_synthetic / target_matrix.size))
    cell_cfg = family_cfg(cfg, family)

    if family == "vae":
        model, train_info = train_vae(scaled_train, cfg)
        raw_samples = sample_vae(model, n_blocks, cfg)
        architecture = f"MLP VAE, latent={cfg.latent_dim}, hidden={cfg.hidden_dim}"
    elif family == "gan":
        model, train_info = train_gan(scaled_train, cfg)
        raw_samples = sample_gan(model, n_blocks, cfg)
        architecture = f"MLP GAN (LSGAN), latent={cfg.latent_dim}, hidden={cfg.hidden_dim}"
    elif family == "diffusion":
        model, train_info = train_diffusion(scaled_train, cfg)
        raw_samples = sample_diffusion(model, n_blocks, cfg)
        architecture = f"DDPM-style MLP denoiser, steps={cfg.diffusion_steps}, hidden={cfg.hidden_dim}"
    else:
        raise ValueError(f"Unsupported family: {family}")

    raw_blocks = unscale_samples(raw_samples, mean, std, target_matrix.shape)
    clipped_blocks, constrained_blocks, records = apply_block_constraints(raw_blocks, target_matrix, cell_cfg, family)

    raw_df = blocks_to_dataframe(clipped_blocks, cell_cfg)
    raw_validation = evaluate_dataset(raw_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(), f"{family}_raw")

    synthetic_df = blocks_to_dataframe(constrained_blocks, cell_cfg)
    metadata = {
        "generator": family,
        "seed": cfg.seed,
        "n_synthetic_requested": cfg.n_synthetic,
        "n_synthetic_saved": int(len(synthetic_df)),
        "rules_file": "knowledge_base/cell_level_rules.json",
        "teacher_source": "rule-guided matrix bootstrap blocks",
        "teacher_blocks": cfg.teacher_blocks,
        "scaling": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "architecture": architecture,
        "train_info": train_info,
        "raw_independent_validation": raw_validation,
        "explainability_records": records,
    }

    explainability_summary = {
        "blocks_total": len(records),
        "pass_count": len(records),
        "reject_count": 0,
        "pass_rate": 1.0 if records else 0.0,
        "reject_rate": 0.0,
        "mean_delta_projection": float(np.mean([r["delta_projection_mean"] for r in records])) if records else 0.0,
        "mean_delta_cap": float(np.mean([r["delta_cap_mean"] for r in records])) if records else 0.0,
        "mean_delta_calibration": float(np.mean([r["delta_calibration_mean"] for r in records if r["delta_calibration_mean"] is not None])) if records else 0.0,
        "mean_total_adjustment_abs": float(np.mean([r["total_adjustment_abs_mean"] for r in records if r["total_adjustment_abs_mean"] is not None])) if records else 0.0,
        "mean_constraint_pressure": float(np.mean([r["constraint_pressure_score"] for r in records if r["constraint_pressure_score"] is not None])) if records else 0.0,
        "p95_constraint_pressure": float(np.percentile([r["constraint_pressure_score"] for r in records if r["constraint_pressure_score"] is not None], 95)) if records else 0.0,
        "mean_projection_active_rate": float(np.mean([r["projection_active_rate"] for r in records])) if records else 0.0,
        "mean_cap_active_rate": float(np.mean([r["cap_active_rate"] for r in records])) if records else 0.0,
        "mean_calibration_active_rate": float(np.mean([r["calibration_active_rate"] for r in records if r["calibration_active_rate"] is not None])) if records else 0.0,
    }
    metadata["explainability_summary"] = explainability_summary

    outdir, metrics, independent = save_family_bundle(family, real_df, synthetic_df, metadata, cell_cfg)
    return summarize_family(
        family=family,
        display_name=family.upper() if family != "diffusion" else "Diffusion",
        architecture=architecture,
        training_note="Trained on rule-guided bootstrap blocks distilled from the observed 15-point matrix.",
        outdir=outdir,
        metrics=metrics,
        independent=independent,
        metadata=metadata,
        raw_validation=raw_validation,
        extra={
            "parameter_count": train_info["parameters"],
            "training_epochs": train_info["epochs"],
        },
    )


def train_residual_family(
    cfg: BenchmarkConfig,
    real_df: pd.DataFrame,
    target_matrix: np.ndarray,
    prior_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    teacher_blocks: np.ndarray,
) -> Dict:
    residual_cfg = ResidualRuleAwareVAEConfig(
        seed=cfg.seed,
        n_synthetic=cfg.n_synthetic,
        teacher_blocks=cfg.teacher_blocks,
        batch_size=cfg.batch_size,
        epochs=cfg.residual_vae_epochs,
        latent_dim=cfg.residual_latent_dim,
        hidden_dim=cfg.residual_hidden_dim,
        device=cfg.device,
        outdir=str(Path("benchmarks") / cfg.benchmark_subdir / "residual_vae"),
    )
    residual_targets = teacher_blocks.reshape(len(teacher_blocks), -1).astype(np.float32) - prior_matrix.reshape(1, -1).astype(np.float32)
    target_residual_mean = residual_targets.mean(axis=0)

    model, train_info, history_df = train_residual_model(teacher_blocks, prior_matrix, sigma_matrix, residual_cfg)
    n_blocks = int(math.ceil(cfg.n_synthetic / target_matrix.size))
    raw_blocks = sample_residual_blocks(model, sigma_matrix, prior_matrix, target_residual_mean, n_blocks, residual_cfg)
    clipped_blocks, constrained_blocks, records = apply_residual_constraints(raw_blocks, prior_matrix, target_matrix, sigma_matrix, residual_cfg)

    cell_cfg = family_cfg(cfg, "residual_vae")
    raw_df = blocks_to_dataframe(clipped_blocks, cell_cfg)
    raw_validation = evaluate_dataset(
        raw_df[["Радиация", "Температура", "Время", "Выживаемость"]].copy(),
        "residual_vae_raw",
    )
    synthetic_df = blocks_to_dataframe(constrained_blocks, cell_cfg)
    metadata = {
        "generator": "residual_vae",
        "seed": cfg.seed,
        "n_synthetic_requested": cfg.n_synthetic,
        "n_synthetic_saved": int(len(synthetic_df)),
        "rules_file": "knowledge_base/cell_level_rules.json",
        "teacher_source": "rule-guided matrix bootstrap blocks",
        "teacher_blocks": cfg.teacher_blocks,
        "architecture": (
            f"Residual rule-aware block VAE, latent={residual_cfg.latent_dim}, "
            f"hidden={residual_cfg.hidden_dim}, matrix prior + rule loss"
        ),
        "train_info": train_info,
        "model_config": residual_cfg.__dict__,
        "raw_independent_validation": raw_validation,
        "explainability_records": records,
        "explainability_summary": summarize_residual_records(records),
        "target_matrix_log10": target_matrix.tolist(),
        "prior_mu_matrix_log10": prior_matrix.tolist(),
        "prior_sigma_matrix_log10": sigma_matrix.tolist(),
    }

    outdir, metrics, independent = save_family_bundle("residual_vae", real_df, synthetic_df, metadata, cell_cfg)
    history_df.to_csv(outdir / "training_history.csv", index=False, encoding="utf-8-sig")
    with (outdir / "raw_independent_cell_level_validation.json").open("w", encoding="utf-8") as fh:
        json.dump(raw_validation, fh, ensure_ascii=False, indent=2)

    return summarize_family(
        family="residual_vae",
        display_name="Residual VAE",
        architecture=metadata["architecture"],
        training_note=(
            "Learns residual stochasticity around the rule-guided matrix prior; "
            "projection/cap/calibration stay as a safety layer."
        ),
        outdir=outdir,
        metrics=metrics,
        independent=independent,
        metadata=metadata,
        raw_validation=raw_validation,
        extra={
            "parameter_count": train_info["parameters"],
            "training_epochs": train_info["epochs"],
            "mean_raw_radiation_violation": metadata["explainability_summary"].get("mean_raw_radiation_violation"),
            "mean_raw_thermal_violation": metadata["explainability_summary"].get("mean_raw_thermal_violation"),
            "mean_residual_budget_utilization": metadata["explainability_summary"].get("mean_residual_budget_utilization"),
        },
    )


def build_matrix_family(cfg: BenchmarkConfig) -> Dict:
    cell_cfg = family_cfg(cfg, "matrix")
    real_df, synthetic_df, metadata = build_generation_artifacts(cell_cfg)
    metadata["architecture"] = "Rule-guided stochastic 5x3 matrix with isotonic projection and calibration"
    outdir, metrics, independent = save_family_bundle("matrix", real_df, synthetic_df, metadata, cell_cfg)
    return summarize_family(
        family="matrix",
        display_name="Matrix",
        architecture=metadata["architecture"],
        training_note="No neural training. Direct rule-guided generator on the exact 15-point design support.",
        outdir=outdir,
        metrics=metrics,
        independent=independent,
        metadata=metadata,
        raw_validation=independent,
        extra={
            "parameter_count": 0,
            "training_epochs": 0,
        },
    )


def rules_markdown_table() -> List[str]:
    rules = load_json(PROJECT_ROOT / "knowledge_base" / "cell_level_rules.json")["rules"]
    lines = [
        "| Rule | Тип | IF | THEN | Evidence |",
        "|---|---|---|---|---|",
    ]
    for rule in rules:
        if_parts = []
        for item in rule["if"]:
            relation = item["relation"]
            value = item.get("value")
            text = f"{item['variable']} {relation}"
            if value is not None:
                text += f" {value}"
            if_parts.append(text)
        then_parts = []
        for item in rule["then"]:
            relation = item["relation"]
            value = item.get("value")
            text = f"{item['variable']} {relation}"
            if value is not None:
                text += f" {value}"
            then_parts.append(text)
        lines.append(
            f"| {rule['id']} | {rule['type']} | {'; '.join(if_parts)} | {'; '.join(then_parts)} | {'; '.join(rule['evidence_ids'])} |"
        )
    return lines


def literature_markdown_table() -> List[str]:
    literature_df = pd.read_csv(PROJECT_ROOT / "literature" / "literature_evidence.csv")
    lines = [
        "| ID | Year | Type | Context | Main finding | Link |",
        "|---|---:|---|---|---|---|",
    ]
    for _, row in literature_df.iterrows():
        link = str(row["source_url"])
        lines.append(
            f"| {row['evidence_id']} | {int(row['year']) if pd.notna(row['year']) else 'NA'} | {row['study_type']} | "
            f"{row['cancer_context']} | {row['main_finding']} | {link} |"
        )
    return lines


def summary_markdown_table(summary_df: pd.DataFrame) -> List[str]:
    order = [
        "display_name",
        "independent_article_compliance_mean",
        "local_mean_abs_error",
        "tstr_r2",
        "mean_wasserstein_normalized",
        "mean_constraint_pressure",
        "raw_article_compliance_mean",
    ]
    lines = [
        "| Generator | Final compliance | Local MAE | TSTR R2 | Mean Wasserstein(norm) | Explainability pressure | Raw compliance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.sort_values(["independent_article_compliance_mean", "local_mean_abs_error"], ascending=[False, True]).iterrows():
        vals = [row[col] for col in order]
        lines.append(
            f"| {vals[0]} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} | {vals[4]:.4f} | {vals[5]:.4f} | {vals[6]:.4f} |"
        )
    return lines


def architecture_markdown(summary_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"### {row['display_name']}",
                "",
                f"- Архитектура: {row['architecture']}",
                f"- Обучение/получение: {row['training_note']}",
                f"- Параметров: {int(row['parameter_count'])}",
                f"- Эпох: {int(row['training_epochs'])}",
                "",
            ]
        )
    return lines


def data_description_lines() -> List[str]:
    rows: List[str] = [
        "- Экспериментальный дизайн: 5 уровней радиации (`0/2/4/6/8 Gy`) x 3 терморежима (`42C 45 мин`, `43C 45 мин`, `44C 30 мин`) = 15 наблюдаемых design points.",
        "- Биологический выход: доля выживших клеток (`Выживаемость`) после комбинированного воздействия ионизирующего излучения и гипертермии.",
        "- В проекте это cell-level / in vitro survival matrix, а не клинический датасет пациентов.",
        "- Производные признаки для интерпретации: `thermal_rank`, `thermal_label`, `CEM43`.",
        "- Наблюдаемая биология: при усилении радиации выживаемость не должна расти; усиление терморежима внутри наблюдаемого окна тоже не должно повышать выживаемость.",
    ]
    return rows


def instrument_lines() -> List[str]:
    return [
        "- В репозитории нет паспорта прибора, производителя или модели облучателя/гипертермической установки, поэтому эти данные нельзя добросовестно восстановить.",
        "- Что точно есть в данных: номинальная доза радиации (`0-8 Gy`), температура (`42/43/44 C`), длительность нагрева (`30/45 мин`) и вычисляемая thermal dose `CEM43`.",
        "- Для презентации это лучше формулировать как: `экспериментальная установка задавала фиксированные режимы RT + HT, а в анализе доступны только режимные параметры и клеточная выживаемость`.",
    ]


def generation_method_lines() -> List[str]:
    return [
        "- Сначала фиксировался исходный 15-точечный экспериментальный support: синтетика не изобретает новые режимы лечения вне наблюдаемого дизайна.",
        "- Затем строилась rule-guided матрица среднего лог-выживания и локальных шумов на базе наблюдаемой survival-матрицы.",
        "- Для нейросетевых семейств (`VAE`, `GAN`, `diffusion`) обучение шло не на произвольных новых условиях, а на rule-guided bootstrap blocks, дистиллированных из исходной матрицы. Это нужно из-за крайне малого объема реальных наблюдений (`n=15`).",
        "- Новая гибридная версия `Residual VAE` не генерирует весь блок с нуля: она моделирует только остаточную вариативность вокруг matrix prior и получает штраф за raw-rule violations уже в обучении.",
        "- После семплирования любой нейросетевой блок проходил одинаковый post-processing: isotonic projection по радиации и термическому порядку, cap для высокой комбинированной дозы и глобальную калибровку назад к реальной матрице.",
        "- Поэтому benchmark сравнивает не только качество генерации, но и то, насколько сильно каждую модель приходится `дотягивать правилами` до биологически объяснимого результата.",
    ]


def report_conclusion_lines(summary_df: pd.DataFrame) -> List[str]:
    best_fidelity = summary_df.sort_values(["independent_article_compliance_mean", "local_mean_abs_error"], ascending=[False, True]).iloc[0]
    best_explain = summary_df.sort_values(["mean_constraint_pressure", "local_mean_abs_error"], ascending=[True, True]).iloc[0]
    neural_df = summary_df[summary_df["family"] != "matrix"].copy()
    best_neural = neural_df.sort_values(["independent_article_compliance_mean", "mean_constraint_pressure", "local_mean_abs_error"], ascending=[False, True, True]).iloc[0]
    hybrid_row = summary_df[summary_df["family"] == "residual_vae"]

    lines = [
        f"- Лучший итоговый баланс по fidelity и article-compliance в этом прогоне показал `{best_fidelity['display_name']}`.",
        f"- Наиболее объяснимым по минимальному rule-correction pressure оказался `{best_explain['display_name']}`.",
        f"- Среди нейросетевых семейств самым практичным компромиссом вышел `{best_neural['display_name']}`.",
        "- Для этого проекта матричный генератор остается основным кандидатом на демонстрацию, потому что его логика напрямую читается через правила CL1-CL5 и литературу.",
        "- Нейросетевые варианты полезны как sanity-check и как демонстрация, что even modern generative families на сверхмалой биомедицинской матрице все равно нуждаются в жестком rule-guided post-processing.",
    ]
    if not hybrid_row.empty:
        row = hybrid_row.iloc[0]
        lines.insert(
            3,
            (
                f"- `Residual VAE` стоит рассматривать как next-version кандидата: он сохраняет matrix prior, "
                f"но снижает explainability pressure до `{row['mean_constraint_pressure']:.4f}` при `Local MAE = {row['local_mean_abs_error']:.4f}`."
            ),
        )
    return lines


def write_report(summary_df: pd.DataFrame) -> None:
    report_path = BENCHMARK_DIR / "generator_family_benchmark_report_ru.md"
    lines: List[str] = [
        "# Сравнение 5 генераторов для cell-level synthetic dataset",
        "",
        "## Данные о приборе",
        "",
        *instrument_lines(),
        "",
        "## Выборка и биология данных",
        "",
        *data_description_lines(),
        "",
        "## Как собирали правила",
        "",
        "- Правила брались не `из головы`, а из уже собранной локальной knowledge base проекта: `knowledge_base/cell_level_rules.json`, `knowledge_base/cell_level_rules_ru.md`, `literature/literature_evidence.csv`.",
        "- В активную cell-level часть вошли только правила, которые можно честно применить к 4 колонкам `Радиация / Температура / Время / Выживаемость`.",
        "- Клинические и in vivo закономерности (перфузия, глубина очага, repeated sessions, patient toxicity) сознательно исключались, чтобы не смешивать уровни биологии.",
        "- Итоговые активные правила: `CL1-CL5`. Они задают монотонность по RT, термический порядок, sensitizing window и very-low-survival regime для high combined dose.",
        "",
        "## Откуда брали литературу",
        "",
        *literature_markdown_table(),
        "",
        "## Общая таблица правил",
        "",
        *rules_markdown_table(),
        "",
        "## Каким образом получали синтетическую выборку",
        "",
        *generation_method_lines(),
        "",
        "## Архитектуры",
        "",
        *architecture_markdown(summary_df),
        "## Метрики",
        "",
        "- `Final compliance`: независимая article/rule compliance по наблюдаемым колонкам.",
        "- `Local MAE`: средняя ошибка между synthetic mean и реальным значением на каждом design point.",
        "- `TSTR R2`: utility-метрика train-on-synthetic test-on-real.",
        "- `Mean Wasserstein(norm)`: сходство распределений.",
        "- `Explainability pressure`: насколько сильно post-processing правил должен был сдвигать raw samples модели.",
        "",
        *summary_markdown_table(summary_df),
        "",
        "## Ванино объяснение",
        "",
        *report_conclusion_lines(summary_df),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(42)
    cfg = BenchmarkConfig()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    real_df, target_matrix, prior_matrix, sigma_matrix, teacher_blocks = make_teacher_blocks(cfg)
    summary_rows: List[Dict] = []

    print("[benchmark] running matrix baseline")
    summary_rows.append(build_matrix_family(cfg))

    for family in ("vae", "gan", "diffusion"):
        print(f"[benchmark] running {family}")
        summary_rows.append(
            train_neural_family(
                family=family,
                cfg=cfg,
                real_df=real_df,
                target_matrix=target_matrix,
                teacher_blocks=teacher_blocks,
            )
        )

    print("[benchmark] running residual_vae")
    summary_rows.append(
        train_residual_family(
            cfg=cfg,
            real_df=real_df,
            target_matrix=target_matrix,
            prior_matrix=prior_matrix,
            sigma_matrix=sigma_matrix,
            teacher_blocks=teacher_blocks,
        )
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(BENCHMARK_DIR / "generator_family_summary.csv", index=False, encoding="utf-8-sig")
    with (BENCHMARK_DIR / "generator_family_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, ensure_ascii=False, indent=2)

    write_report(summary_df)
    print(f"[benchmark] saved summary to {BENCHMARK_DIR}")


if __name__ == "__main__":
    main()
