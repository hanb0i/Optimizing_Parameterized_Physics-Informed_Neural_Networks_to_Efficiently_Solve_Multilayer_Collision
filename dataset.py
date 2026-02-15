"""
Phase 5 — Dataset generation for surrogate training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch

from config import Bounds, PipelineConfig
from metrics import compute_metrics


MuType = Union[torch.Tensor, Mapping[str, float]]
PhysicsRunner = Callable[[MuType], Mapping[str, Any]]


@dataclass(frozen=True)
class SupervisedDataset:
    param_names: List[str]
    target_names: List[str]
    x_raw: torch.Tensor
    y_raw: torch.Tensor
    x_min: torch.Tensor
    x_max: torch.Tensor
    y_min: torch.Tensor
    y_max: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor

    def x_norm(self) -> torch.Tensor:
        return normalize_minmax(self.x_raw, self.x_min, self.x_max)

    def y_norm(self) -> torch.Tensor:
        return normalize_minmax(self.y_raw, self.y_min, self.y_max)


def normalize_minmax(x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
    denom = (x_max - x_min).clamp_min(1e-12)
    return (x - x_min) / denom


def denormalize_minmax(x_norm: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
    return x_norm * (x_max - x_min) + x_min


def sample_mu(n: int, bounds: Bounds, *, seed: int = 0) -> torch.Tensor:
    """
    Uniform random sampling in raw parameter space (bounds.low/high).
    """

    g = torch.Generator(device=bounds.low.device)
    g.manual_seed(int(seed))
    u = torch.rand((n, bounds.dim), generator=g, device=bounds.low.device)
    return bounds.low.view(1, -1) + u * (bounds.high - bounds.low).view(1, -1)


def mu_tensor_to_dict(mu: torch.Tensor, param_names: List[str]) -> Dict[str, float]:
    mu = mu.detach().cpu().view(-1)
    return {k: float(mu[i].item()) for i, k in enumerate(param_names)}


def mu_dict_to_tensor(mu: Mapping[str, float], bounds: Bounds) -> torch.Tensor:
    return torch.tensor([mu[k] for k in bounds.names], dtype=torch.float32, device=bounds.low.device)


def split_indices(n: int, train_frac: float, val_frac: float, *, seed: int = 0, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device or torch.device("cpu"))
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g, device=device or torch.device("cpu"))
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def generate_dataset(
    n: int,
    bounds: Bounds,
    physics_runner: PhysicsRunner,
    cfg: PipelineConfig,
    *,
    seed: int = 0,
    target_names: Optional[List[str]] = None,
) -> SupervisedDataset:
    """
    Generate (X_mu, Y_metrics) by sampling mu, running physics, and computing metrics.

    Targets are the 3 required recommendation metrics by default:
      - y_strain_energy
      - y_accel_peak
      - y_disp_peak
    """

    target_names = target_names or ["y_strain_energy", "y_accel_peak", "y_disp_peak"]

    x_raw = sample_mu(n, bounds, seed=seed).to(cfg.device)
    y_rows: List[torch.Tensor] = []

    for i in range(n):
        mu_i = x_raw[i]
        sim_out = physics_runner(mu_i)
        metrics = compute_metrics(sim_out, cfg.metrics)
        y_rows.append(torch.stack([metrics[name].to(cfg.device) for name in target_names], dim=0))

    y_raw = torch.stack(y_rows, dim=0)

    x_min = x_raw.min(dim=0).values
    x_max = x_raw.max(dim=0).values
    y_min = y_raw.min(dim=0).values
    y_max = y_raw.max(dim=0).values

    train_idx, val_idx, test_idx = split_indices(n, cfg.surrogate.train_frac, cfg.surrogate.val_frac, seed=seed, device=torch.device("cpu"))
    return SupervisedDataset(
        param_names=list(bounds.names),
        target_names=target_names,
        x_raw=x_raw.detach().cpu(),
        y_raw=y_raw.detach().cpu(),
        x_min=x_min.detach().cpu(),
        x_max=x_max.detach().cpu(),
        y_min=y_min.detach().cpu(),
        y_max=y_max.detach().cpu(),
        train_idx=train_idx.detach().cpu(),
        val_idx=val_idx.detach().cpu(),
        test_idx=test_idx.detach().cpu(),
    )

