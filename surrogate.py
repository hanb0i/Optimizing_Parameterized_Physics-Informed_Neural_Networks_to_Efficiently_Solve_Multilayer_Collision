"""
Phase 5 — Differentiable surrogate model: mu -> y_hat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Bounds, PipelineConfig, SurrogateConfig
from dataset import SupervisedDataset, denormalize_minmax, normalize_minmax


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, features_dim: int, sigma: float, seed: int = 0):
        super().__init__()
        g = torch.Generator(device=torch.device("cpu"))
        g.manual_seed(int(seed))
        B = torch.randn((features_dim, input_dim), generator=g) * float(sigma)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * torch.pi * (x @ self.B.t())
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_layers: int,
        hidden_units: int,
        activation: str,
        fourier: Optional[FourierFeatures] = None,
    ):
        super().__init__()
        self.fourier = fourier
        act = _activation(activation)

        feat_dim = input_dim if fourier is None else 2 * fourier.B.shape[0]
        layers: List[nn.Module] = []
        layers.append(nn.Linear(feat_dim, hidden_units))
        layers.append(act)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(act)
        layers.append(nn.Linear(hidden_units, output_dim))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        feats = x_norm if self.fourier is None else self.fourier(x_norm)
        return self.net(feats)


@dataclass(frozen=True)
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int


@dataclass(frozen=True)
class SurrogateBundle:
    model: nn.Module
    bounds: Bounds
    param_names: List[str]
    target_names: List[str]
    x_min: torch.Tensor
    x_max: torch.Tensor
    y_min: torch.Tensor
    y_max: torch.Tensor

    def to(self, device: torch.device) -> "SurrogateBundle":
        self.model.to(device)
        return self

    def predict_raw(self, mu_raw: Union[torch.Tensor, Dict[str, float]], *, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or next(self.model.parameters()).device

        if isinstance(mu_raw, dict):
            mu_raw_t = torch.tensor([mu_raw[k] for k in self.param_names], dtype=torch.float32, device=device).view(1, -1)
        else:
            mu_raw_t = mu_raw.to(device=device, dtype=torch.float32).view(1, -1)

        x_norm = normalize_minmax(mu_raw_t, self.x_min.to(device), self.x_max.to(device)).clamp(0.0, 1.0)
        self.model.eval()
        with torch.no_grad():
            y_norm = self.model(x_norm)
        y_raw = denormalize_minmax(y_norm, self.y_min.to(device), self.y_max.to(device))
        return y_raw.view(-1)

    def predict_dict(self, mu_raw: Union[torch.Tensor, Dict[str, float]], *, device: Optional[torch.device] = None) -> Dict[str, float]:
        y = self.predict_raw(mu_raw, device=device)
        return {name: float(y[i].item()) for i, name in enumerate(self.target_names)}


def _regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
    err = y_pred - y_true
    mae = err.abs().mean(dim=0)
    rmse = torch.sqrt((err.pow(2).mean(dim=0)) + 1e-12)
    ss_res = (err.pow(2)).sum(dim=0)
    y_mean = y_true.mean(dim=0)
    ss_tot = ((y_true - y_mean).pow(2)).sum(dim=0).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_surrogate(dataset: SupervisedDataset, bounds: Bounds, cfg: PipelineConfig) -> Tuple[SurrogateBundle, TrainHistory, Dict[str, torch.Tensor]]:
    scfg: SurrogateConfig = cfg.surrogate
    device = cfg.device

    x = dataset.x_norm().to(device)
    y = dataset.y_norm().to(device)

    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    fourier = None
    if scfg.use_fourier_features:
        fourier = FourierFeatures(bounds.dim, scfg.fourier_features_dim, scfg.fourier_sigma, seed=scfg.seed).to(device)

    model = MLPRegressor(
        input_dim=bounds.dim,
        output_dim=len(dataset.target_names),
        hidden_layers=scfg.hidden_layers,
        hidden_units=scfg.hidden_units,
        activation=scfg.activation,
        fourier=fourier,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=scfg.lr)
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = int(scfg.patience)
    best_epoch = -1
    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []

    out_w = scfg.output_weights
    if out_w is None:
        out_w = torch.ones((len(dataset.target_names),), dtype=torch.float32, device=device)
    out_w = out_w.to(device).view(1, -1)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(scfg.max_epochs)):
        model.train()
        perm = torch.randperm(x_train.shape[0], device=device)
        x_shuf = x_train[perm]
        y_shuf = y_train[perm]

        batch_losses = []
        for start in range(0, x_shuf.shape[0], int(scfg.batch_size)):
            xb = x_shuf[start : start + int(scfg.batch_size)]
            yb = y_shuf[start : start + int(scfg.batch_size)]
            pred = model(xb)
            mse = (pred - yb).pow(2)
            loss = (mse * out_w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch_losses.append(loss.detach())
        train_loss = torch.stack(batch_losses).mean().item() if batch_losses else 0.0

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
            val_mse = (pred_val - y_val).pow(2)
            val_loss = (val_mse * out_w).mean().item()

        train_loss_hist.append(float(train_loss))
        val_loss_hist.append(float(val_loss))

        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_dir / "last.pt")
        if val_loss + float(scfg.min_delta) < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(scfg.patience)
            best_epoch = epoch
            torch.save({"epoch": epoch, "state_dict": best_state}, ckpt_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_all = model(x)
    metrics = _regression_metrics(y, pred_all)

    bundle = SurrogateBundle(
        model=model,
        bounds=bounds,
        param_names=list(dataset.param_names),
        target_names=list(dataset.target_names),
        x_min=dataset.x_min.clone(),
        x_max=dataset.x_max.clone(),
        y_min=dataset.y_min.clone(),
        y_max=dataset.y_max.clone(),
    )

    hist = TrainHistory(train_loss=train_loss_hist, val_loss=val_loss_hist, best_epoch=best_epoch)
    return bundle, hist, metrics

