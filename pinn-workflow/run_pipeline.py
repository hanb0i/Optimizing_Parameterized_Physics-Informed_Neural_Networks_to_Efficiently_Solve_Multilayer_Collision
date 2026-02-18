"""
End-to-end runnable example for Phases 4–7 using a synthetic dummy physics_runner.

Swap in your real PiNN/FEA runner by replacing `dummy_physics_runner(mu)`.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Mapping, Union

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import torch

from config import Bounds, PipelineConfig
from dataset import generate_dataset, mu_tensor_to_dict, sample_mu
from diagnostics import plot_discrepancy_history, plot_optimization_trajectory, plot_parity_metrics, plot_training_history
from evaluate import compare_designs, run_repeated_impacts, active_learning_refine, discrepancy_report
from metrics import MetricsConfig
from optimize import optimize_design
from surrogate import train_surrogate


MuType = Union[torch.Tensor, Mapping[str, float]]


def dummy_physics_runner(mu: MuType) -> Dict[str, Any]:
    """
    Synthetic impact-like response generator.

    - Treats mu as a generic vector; uses a smooth mapping to a damped oscillator.
    - Produces:
        u_prot[t]    protected displacement
        eps[t, i]    synthetic "strain" field at Npts
        dt           timestep
        weights      integration weights for eps (uniform)
    """

    if isinstance(mu, dict):
        mu_vec = torch.tensor(list(mu.values()), dtype=torch.float32)
    else:
        mu_vec = mu.to(dtype=torch.float32).view(-1).detach()

    Nt = 400
    dt = 5e-4
    t = torch.arange(Nt, dtype=torch.float32) * dt

    # Smooth map from mu -> effective parameters (positive, bounded)
    s = torch.sigmoid(mu_vec)
    k = 50.0 + 250.0 * s.mean()
    c = 0.5 + 3.0 * s.std().clamp_min(0.0)
    m = 0.5 + 2.0 * s[0] if s.numel() > 0 else torch.tensor(1.0)

    w0 = torch.sqrt(k / m)
    zeta = (c / (2.0 * torch.sqrt(k * m))).clamp(0.02, 1.0)
    wd = w0 * torch.sqrt((1.0 - zeta**2).clamp_min(1e-6))

    v0 = 1.0 + 2.0 * (s[1] if s.numel() > 1 else torch.tensor(0.0))
    u_prot = (v0 / wd) * torch.exp(-zeta * w0 * t) * torch.sin(wd * t)

    # Synthetic strain field: spatial profile * displacement with mild time variation
    Npts = 80
    x = torch.linspace(0.0, 1.0, Npts, dtype=torch.float32)
    profile = (1.0 - x).pow(2) + 0.1
    time_gain = 1.0 + 0.2 * torch.sin(2.0 * torch.pi * 40.0 * t)
    eps = (u_prot.view(-1, 1) * profile.view(1, -1)) * time_gain.view(-1, 1)

    weights = torch.ones((Npts,), dtype=torch.float32) / float(Npts)

    return {"u_prot": u_prot, "eps": eps, "dt": dt, "weights": weights}


def main() -> None:
    device = torch.device("cpu")
    cfg = PipelineConfig(device=device)

    bounds_dict = {f"p{i}": (0.0, 1.0) for i in range(6)}
    bounds = Bounds.from_dict(bounds_dict, device=device)
    cfg = PipelineConfig(
        device=device,
        out_dir=cfg.out_dir,
        bounds=bounds,
        metrics=MetricsConfig(
            absorb_mask=(torch.linspace(0.0, 1.0, 80) > 0.3),
            accel_smooth_window=9,
            include_optional=False,
        ),
        surrogate=cfg.surrogate,
        optimize=cfg.optimize,
        damage=cfg.damage,
        active_learning=cfg.active_learning,
        dataset_size=cfg.dataset_size,
    )

    dataset = generate_dataset(cfg.dataset_size, bounds, dummy_physics_runner, cfg, seed=cfg.surrogate.seed)
    surrogate, hist, fit_metrics = train_surrogate(dataset, bounds, cfg)
    plots_dir = cfg.out_dir / "plots"
    plot_training_history(hist, plots_dir)

    mu0 = {k: (bounds_dict[k][0] + bounds_dict[k][1]) * 0.5 for k in bounds.names}
    candidates, traj = optimize_design(surrogate, bounds, cfg)
    plot_optimization_trajectory(traj, bounds.names, plots_dir)

    if cfg.active_learning.enabled:
        reports = []
        for it in range(int(cfg.active_learning.iters)):
            candidates, _ = optimize_design(surrogate, bounds, cfg)
            eval_mus = list(candidates)
            n_rand = int(cfg.active_learning.eval_random_points)
            if n_rand > 0:
                x_rand = sample_mu(n_rand, bounds, seed=int(cfg.surrogate.seed + 1000 + it)).cpu()
                eval_mus.extend([mu_tensor_to_dict(x_rand[i], list(bounds.names)) for i in range(x_rand.shape[0])])

            rep = discrepancy_report(
                surrogate,
                eval_mus,
                dummy_physics_runner,
                cfg.metrics,
                relerr_floor=float(cfg.active_learning.relerr_floor),
            )
            rep["iter"] = it
            reports.append(rep)
            print(f"Surrogate discrepancy iter {it}: max_rel={rep['max_rel']:.4f}, mean_rel={rep['mean_rel']:.4f}")
            print(f"  per_target_max_rel={rep['per_target_max_rel']}")
            if rep["max_rel"] <= float(cfg.active_learning.discrepancy_tol):
                break
            refined = active_learning_refine(cfg.dataset_size, bounds, dummy_physics_runner, cfg, candidates, surrogate)
            if refined is None:
                break
            surrogate = refined
        plot_discrepancy_history(reports, plots_dir)

    candidates, _ = optimize_design(surrogate, bounds, cfg)
    table = compare_designs(mu0, candidates, dummy_physics_runner, cfg.metrics)
    print("Validation vs baseline (true physics):")
    for row in table:
        print(row)
    plot_parity_metrics(surrogate, candidates, dummy_physics_runner, cfg.metrics, plots_dir)

    if cfg.damage.enabled:
        print("Repeated impacts with damage:")
        records = run_repeated_impacts(mu0, bounds, dummy_physics_runner, cfg.metrics, cfg.damage)
        for r in records:
            print(r)

    print(f"Surrogate best epoch: {hist.best_epoch}")
    print("Fit metrics on full normalized dataset:")
    for k, v in fit_metrics.items():
        print(k, v.detach().cpu().numpy())


if __name__ == "__main__":
    main()
