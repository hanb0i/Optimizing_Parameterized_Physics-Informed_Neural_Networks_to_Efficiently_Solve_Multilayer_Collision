"""
Phase 7 — Physics validation, repeated impacts with damage, and active-learning loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from config import Bounds, DamageConfig, PipelineConfig
from dataset import generate_dataset, mu_dict_to_tensor
from metrics import MetricsConfig, compute_metrics
from surrogate import SurrogateBundle, train_surrogate


MuType = Union[torch.Tensor, Mapping[str, float]]
PhysicsRunner = Callable[[MuType], Mapping[str, Any]]


def evaluate_mu(mu: MuType, physics_runner: PhysicsRunner, metrics_cfg: MetricsConfig) -> Dict[str, float]:
    sim_out = physics_runner(mu)
    metrics = compute_metrics(sim_out, metrics_cfg)
    return {k: float(v.detach().cpu().item()) for k, v in metrics.items()}


def compare_designs(
    baseline_mu: MuType,
    candidate_mus: Sequence[MuType],
    physics_runner: PhysicsRunner,
    metrics_cfg: MetricsConfig,
    *,
    keys: Sequence[str] = ("y_strain_energy", "y_accel_peak", "y_disp_peak"),
) -> List[Dict[str, Any]]:
    base = evaluate_mu(baseline_mu, physics_runner, metrics_cfg)
    rows: List[Dict[str, Any]] = []
    for i, mu in enumerate(candidate_mus):
        m = evaluate_mu(mu, physics_runner, metrics_cfg)
        row: Dict[str, Any] = {"candidate": i}
        for k in keys:
            row[f"baseline_{k}"] = base.get(k)
            row[f"candidate_{k}"] = m.get(k)
        rows.append(row)
    return rows


@dataclass
class DamageState:
    D: float = 0.0


def apply_damage_to_mu(mu: MuType, bounds: Bounds, dcfg: DamageConfig, state: DamageState) -> MuType:
    """
    Default damage model: scale any configured "modulus" parameters by (1 - clamp(D)).

    This is intentionally modular because mu semantics are application-specific.
    """

    if not dcfg.modulus_param_names:
        return mu

    scale = 1.0 - float(min(max(state.D, 0.0), float(dcfg.d_max))) * float(dcfg.modulus_scale_max)
    scale = float(max(scale, 0.0))

    if isinstance(mu, dict):
        mu2 = dict(mu)
        for k in dcfg.modulus_param_names:
            if k in mu2:
                mu2[k] = float(mu2[k]) * scale
        return mu2

    mu_t = mu.clone()
    for k in dcfg.modulus_param_names:
        if k in bounds.names:
            idx = bounds.names.index(k)
            mu_t[idx] = mu_t[idx] * scale
    return mu_t


def run_repeated_impacts(
    mu0: MuType,
    bounds: Bounds,
    physics_runner: PhysicsRunner,
    metrics_cfg: MetricsConfig,
    dcfg: DamageConfig,
) -> List[Dict[str, Any]]:
    """
    Sequential impacts with accumulated smooth damage:
      D_{k+1} = D_k + ∫∫ softplus(ε-ε_crit) dx dt
    """

    if not dcfg.enabled:
        raise ValueError("DamageConfig.enabled is False.")

    state = DamageState(D=0.0)
    records: List[Dict[str, Any]] = []

    for k in range(int(dcfg.n_impacts)):
        mu_k = apply_damage_to_mu(mu0, bounds, dcfg, state)

        sim_out = physics_runner(mu_k)
        mcfg = MetricsConfig(
            absorb_mask=metrics_cfg.absorb_mask,
            t_max=metrics_cfg.t_max,
            accel_smooth_window=metrics_cfg.accel_smooth_window,
            include_optional=True,
            include_damage=True,
            damage_eps_crit=float(dcfg.eps_crit),
            damage_softplus_beta=float(dcfg.softplus_beta),
            strain_fn=metrics_cfg.strain_fn,
            protected_index=metrics_cfg.protected_index,
        )
        metrics = compute_metrics(sim_out, mcfg)

        D_inc = float(metrics.get("D_damage", torch.tensor(0.0)).detach().cpu().item())
        state.D += D_inc
        records.append(
            {
                "impact": k,
                "D_inc": D_inc,
                "D_total": state.D,
                "y_strain_energy": float(metrics["y_strain_energy"].detach().cpu().item()),
                "y_accel_peak": float(metrics["y_accel_peak"].detach().cpu().item()),
                "y_disp_peak": float(metrics["y_disp_peak"].detach().cpu().item()),
            }
        )

    return records


def active_learning_refine(
    dataset_n: int,
    bounds: Bounds,
    physics_runner: PhysicsRunner,
    cfg: PipelineConfig,
    candidate_mus: Sequence[Mapping[str, float]],
    surrogate: SurrogateBundle,
) -> Optional[SurrogateBundle]:
    """
    If surrogate discrepancies are large on evaluated candidates, add new points and retrain.
    """

    if not cfg.active_learning.enabled:
        return None

    target_names = ["y_strain_energy", "y_accel_peak", "y_disp_peak"]

    bad_points: List[torch.Tensor] = []
    bad_targets: List[torch.Tensor] = []

    for mu_d in candidate_mus:
        true_metrics = evaluate_mu(mu_d, physics_runner, cfg.metrics)
        y_true = torch.tensor([true_metrics[k] for k in target_names], dtype=torch.float32)
        y_pred = surrogate.predict_raw(mu_d).cpu()

        rel = (y_pred - y_true).abs() / (y_true.abs().clamp_min(1e-6))
        if float(rel.max().item()) > float(cfg.active_learning.discrepancy_tol):
            bad_points.append(mu_dict_to_tensor(mu_d, bounds).cpu())
            bad_targets.append(y_true.cpu())

    if not bad_points:
        return None

    n_add = min(int(cfg.active_learning.add_points_per_iter), len(bad_points))
    x_add = torch.stack(bad_points[:n_add], dim=0)
    y_add = torch.stack(bad_targets[:n_add], dim=0)

    base_ds = generate_dataset(dataset_n, bounds, physics_runner, cfg, seed=cfg.surrogate.seed, target_names=target_names)
    x_raw = torch.cat([base_ds.x_raw, x_add], dim=0)
    y_raw = torch.cat([base_ds.y_raw, y_add], dim=0)

    x_min = x_raw.min(dim=0).values
    x_max = x_raw.max(dim=0).values
    y_min = y_raw.min(dim=0).values
    y_max = y_raw.max(dim=0).values

    # Reuse split sizes from the newly created base dataset
    from dataset import SupervisedDataset

    ds2 = SupervisedDataset(
        param_names=base_ds.param_names,
        target_names=base_ds.target_names,
        x_raw=x_raw,
        y_raw=y_raw,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        train_idx=base_ds.train_idx,
        val_idx=base_ds.val_idx,
        test_idx=base_ds.test_idx,
    )

    refined, _, _ = train_surrogate(ds2, bounds, cfg)
    return refined

