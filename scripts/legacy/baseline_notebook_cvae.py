from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from generate_rule_guided_cvae import (
    GenerationConfig,
    evaluate_synthetic_quality,
    make_raw_frame,
    save_outputs,
    set_seed,
)


@dataclass
class NotebookCVAEConfig:
    seed: int = 42
    epochs: int = 2000
    x_dim: int = 3
    y_dim: int = 1
    h_dim: int = 16
    z_dim: int = 2
    lr: float = 1e-3
    n_gen: int = 1000
    rad_min: float = 0.0
    rad_max: float = 8.0
    temp_min: float = 40.0
    temp_max: float = 45.0
    time_min: float = 20.0
    time_max: float = 60.0
    outdir: str = "benchmarks/results/synthetic_data_notebook_cvae"


class NotebookCVAE(nn.Module):
    def __init__(self, x_dim: int = 3, y_dim: int = 1, h_dim: int = 16, z_dim: int = 2):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(y_dim + x_dim, h_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, y_dim),
            nn.Sigmoid(),
        )

    def encode(self, y: torch.Tensor, x: torch.Tensor):
        inp = torch.cat([y, x], dim=1)
        h = self.encoder(inp)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, x: torch.Tensor):
        inp = torch.cat([z, x], dim=1)
        return self.decoder(inp)

    def forward(self, y: torch.Tensor, x: torch.Tensor):
        mu, logvar = self.encode(y, x)
        z = self.reparameterize(mu, logvar)
        y_recon = self.decode(z, x)
        return y_recon, mu, logvar


def loss_function(y_recon: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    recon_loss = nn.MSELoss(reduction="sum")(y_recon, y)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


def train_notebook_cvae(df: pd.DataFrame, cfg: NotebookCVAEConfig) -> NotebookCVAE:
    X = df[["Радиация", "Температура", "Время"]].to_numpy(dtype=np.float32, copy=True)
    y = df[["Выживаемость"]].to_numpy(dtype=np.float32, copy=True)

    X_torch = torch.from_numpy(X)
    y_torch = torch.from_numpy(y)

    model = NotebookCVAE(x_dim=cfg.x_dim, y_dim=cfg.y_dim, h_dim=cfg.h_dim, z_dim=cfg.z_dim)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        y_recon, mu, logvar = model(y_torch, X_torch)
        loss = loss_function(y_recon, y_torch, mu, logvar)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch={epoch:04d} loss={loss.item():.4f}")

    return model


def generate_notebook_style_synthetic(model: NotebookCVAE, cfg: NotebookCVAEConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    radiation = rng.uniform(cfg.rad_min, cfg.rad_max, cfg.n_gen).astype(np.float32)
    temperature = rng.uniform(cfg.temp_min, cfg.temp_max, cfg.n_gen).astype(np.float32)
    duration = rng.uniform(cfg.time_min, cfg.time_max, cfg.n_gen).astype(np.float32)

    X_gen = np.stack([radiation, temperature, duration], axis=1).astype(np.float32)
    X_t = torch.from_numpy(X_gen)

    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(cfg.n_gen):
            z = torch.randn(1, model.z_dim)
            x_rep = X_t[i].unsqueeze(0)
            y_out = model.decode(z, x_rep)
            rows.append(
                {
                    "Радиация": float(X_gen[i, 0]),
                    "Температура": float(X_gen[i, 1]),
                    "Время": float(X_gen[i, 2]),
                    "Выживаемость": float(y_out[0, 0]),
                }
            )

    df_syn = pd.DataFrame(rows)
    df_syn["Выживаемость"] = df_syn["Выживаемость"].clip(lower=0, upper=1)

    mask_extreme = (df_syn["Температура"] > 44) & (df_syn["Радиация"] > 6)
    df_syn.loc[mask_extreme, "Выживаемость"] = df_syn.loc[mask_extreme, "Выживаемость"].clip(upper=0.01)
    return df_syn


def main() -> None:
    cfg = NotebookCVAEConfig()
    set_seed(cfg.seed)
    real_df = make_raw_frame()[["Радиация", "Температура", "Время", "Выживаемость"]].copy()

    model = train_notebook_cvae(real_df, cfg)
    synthetic_df = generate_notebook_style_synthetic(model, cfg)

    project_root = Path(__file__).resolve().parents[1]
    outdir = project_root / cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    synthetic_df.to_csv(outdir / "cvae_synthetic_dataset.csv", index=False, encoding="utf-8-sig")
    synthetic_df.to_csv(outdir / "cvae_synthetic_dataset_full.csv", index=False, encoding="utf-8-sig")

    eval_cfg = GenerationConfig(run_evaluation=True, discriminator_rounds=10, outdir=cfg.outdir)
    evaluation_files = evaluate_synthetic_quality(real_df, synthetic_df, eval_cfg, outdir)

    metadata = {
        "generator": "baseline_notebook_cvae",
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "n_synthetic_saved": int(len(synthetic_df)),
        "sampling_ranges": {
            "radiation": [cfg.rad_min, cfg.rad_max],
            "temperature": [cfg.temp_min, cfg.temp_max],
            "time": [cfg.time_min, cfg.time_max],
        },
        "evaluation_outputs": evaluation_files,
    }
    with (outdir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(f"saved synthetic dataset to: {outdir / 'cvae_synthetic_dataset.csv'}")
    for path in evaluation_files:
        print(f"saved evaluation output to: {path}")


if __name__ == "__main__":
    main()
