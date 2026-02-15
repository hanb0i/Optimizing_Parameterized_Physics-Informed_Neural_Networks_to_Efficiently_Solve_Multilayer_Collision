"""
Phase 6 — Gradient-based design optimization through the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from config import Bounds, OptimizeConfig, PipelineConfig
from dataset import denormalize_minmax, normalize_minmax
from surrogate import SurrogateBundle


@dataclass(frozen=True)
class OptStep:
    step: int
    mu_norm: torch.Tensor
    mu_raw: torch.Tensor
    y_pred: torch.Tensor
    objective: float


def _objective(y_pred: torch.Tensor, ocfg: OptimizeConfig) -> torch.Tensor:
    """
    y_pred order must be: [y_strain_energy, y_accel_peak, y_disp_peak].
    """

    y_strain = y_pred[..., 0]
    y_accel = y_pred[..., 1]
    y_disp = y_pred[..., 2]

    beta = float(ocfg.penalty_beta)
    penalty_accel = F.softplus((y_accel / float(ocfg.a_cap) - 1.0) * beta) / beta
    penalty_disp = F.softplus((y_disp / float(ocfg.u_cap) - 1.0) * beta) / beta

    return y_strain + float(ocfg.lambda_accel) * penalty_accel + float(ocfg.beta_disp) * penalty_disp


def optimize_design(
    surrogate: SurrogateBundle,
    bounds: Bounds,
    cfg: PipelineConfig,
    *,
    init_mu_norm: Optional[torch.Tensor] = None,
) -> Tuple[List[Dict[str, float]], List[OptStep]]:
    """
    Returns (top_k_candidates, trajectory).

    Each candidate is a dict mapping param_name -> raw value.
    """

    device = cfg.device
    ocfg = cfg.optimize

    b_low = bounds.low.to(device).view(1, -1)
    b_high = bounds.high.to(device).view(1, -1)
    x_min = surrogate.x_min.to(device).view(1, -1)
    x_max = surrogate.x_max.to(device).view(1, -1)

    def norm_to_raw(mu_norm_box: torch.Tensor) -> torch.Tensor:
        return b_low + mu_norm_box * (b_high - b_low)

    top_steps: List[OptStep] = []
    all_traj: List[OptStep] = []

    g = torch.Generator(device=device)
    g.manual_seed(int(cfg.surrogate.seed))

    for r in range(int(ocfg.restarts)):
        if init_mu_norm is not None and r == 0:
            mu0 = init_mu_norm.to(device=device, dtype=torch.float32).view(1, -1)
        else:
            mu0 = torch.rand((1, bounds.dim), generator=g, device=device)

        mu = torch.nn.Parameter(mu0.clone())
        opt = torch.optim.Adam([mu], lr=float(ocfg.lr))

        for step in range(int(ocfg.steps)):
            if ocfg.project_box:
                with torch.no_grad():
                    mu.clamp_(0.0, 1.0)

            mu_raw = norm_to_raw(mu)
            x_norm = normalize_minmax(mu_raw, x_min, x_max).clamp(0.0, 1.0)
            y_norm = surrogate.model(x_norm)
            y_pred = denormalize_minmax(y_norm, surrogate.y_min.to(device), surrogate.y_max.to(device)).view(-1)

            J = _objective(y_pred, ocfg)
            opt.zero_grad(set_to_none=True)
            J.backward()
            opt.step()

            rec = OptStep(
                step=step + r * int(ocfg.steps),
                mu_norm=mu.detach().view(-1).cpu().clone(),
                mu_raw=mu_raw.detach().view(-1).cpu().clone(),
                y_pred=y_pred.detach().cpu().clone(),
                objective=float(J.detach().cpu().item()),
            )
            all_traj.append(rec)

        top_steps.extend(sorted(all_traj[-int(ocfg.steps):], key=lambda s: s.objective)[: max(1, int(ocfg.top_k))])

    top_steps = sorted(top_steps, key=lambda s: s.objective)[: max(1, int(ocfg.top_k))]
    candidates: List[Dict[str, float]] = []
    for s in top_steps:
        mu_vec = s.mu_raw.view(-1)
        candidates.append({name: float(mu_vec[i].item()) for i, name in enumerate(bounds.names)})
    return candidates, all_traj
