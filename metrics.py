"""
Phase 4 — Performance mapping: simulation outputs -> scalar metrics.

This module is written to work with either a PiNN or an FEA wrapper as long as
the physics runner returns a dict-like `sim_outputs` with (some of) the keys:
  - "eps": strain field, shape [Nt, Npts] or [Nt, Nelem]
  - "u": displacement field, shape [Nt, Npts] or [Nt, Npts, dim]
  - "u_prot": protected displacement time series, shape [Nt]
  - "dt": scalar timestep (float or 0-d tensor)
  - "coords": spatial coordinates (optional, for strain-from-u hook)
  - "connectivity": mesh connectivity (optional, for strain-from-u hook)
  - "weights"/"volumes": spatial integration weights, shape [Npts] or [Nelem]

All computations are torch-first and differentiable when inputs are torch
tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F


TensorLike = Union[torch.Tensor, float, int]


def _as_tensor(x: Any, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device is not None else x
    return torch.as_tensor(x, dtype=dtype, device=device)


def compute_strain(u: torch.Tensor, coords: Optional[torch.Tensor] = None, connectivity: Any = None) -> torch.Tensor:
    """
    Placeholder hook to compute strain from displacement.

    For an FEA mesh you might compute element strains from shape function
    gradients. For a PiNN you might compute strain via autograd w.r.t. coords.

    Expected output shape: [Nt, Npts] or [Nt, Nelem].
    """

    raise NotImplementedError(
        "Strain was not provided in sim_outputs['eps'] and the default "
        "compute_strain hook is not implemented. Provide sim_outputs['eps'] or "
        "pass a custom strain_fn in MetricsConfig."
    )


@dataclass(frozen=True)
class MetricsConfig:
    absorb_mask: Optional[torch.Tensor] = None
    t_max: Optional[float] = None

    accel_smooth_window: int = 0
    include_optional: bool = False

    include_strain_max: bool = False
    include_force_peak: bool = False
    include_damage: bool = False

    damage_eps_crit: float = 0.02
    damage_softplus_beta: float = 50.0

    strain_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor], Any], torch.Tensor]] = None

    protected_index: Optional[int] = None
    weights_key_candidates: Sequence[str] = ("weights", "volumes")


def _slice_time_window(x: torch.Tensor, dt: torch.Tensor, t_max: Optional[float]) -> torch.Tensor:
    if t_max is None:
        return x
    n_keep = int(float(torch.clamp(torch.floor(_as_tensor(t_max) / dt) + 1, min=1.0)))
    return x[:n_keep]


def second_derivative_central(u_t: torch.Tensor, dt: TensorLike) -> torch.Tensor:
    """
    Stable discrete 2nd derivative (central difference on interior points).

    Returns a tensor with the same shape as `u_t` (time dimension first).
    """

    u_t = _as_tensor(u_t)
    dt_t = _as_tensor(dt, device=u_t.device, dtype=u_t.dtype)
    if u_t.ndim != 1:
        raise ValueError(f"Expected u_t shape [Nt], got {tuple(u_t.shape)}")
    if u_t.numel() < 3:
        return torch.zeros_like(u_t)

    a = torch.empty_like(u_t)
    a[1:-1] = (u_t[2:] - 2.0 * u_t[1:-1] + u_t[:-2]) / (dt_t * dt_t)
    a[0] = a[1]
    a[-1] = a[-2]
    return a


def moving_average_1d(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    if x.ndim != 1:
        raise ValueError(f"Expected x shape [N], got {tuple(x.shape)}")
    w = min(int(window), int(x.numel()))
    kernel = torch.ones((1, 1, w), device=x.device, dtype=x.dtype) / float(w)
    x3 = x.view(1, 1, -1)
    pad = w // 2
    x3p = F.pad(x3, (pad, w - 1 - pad), mode="replicate")
    y = F.conv1d(x3p, kernel)
    return y.view(-1)


def _get_u_prot(sim_outputs: Mapping[str, Any], cfg: MetricsConfig, device: torch.device) -> torch.Tensor:
    if "u_prot" in sim_outputs and sim_outputs["u_prot"] is not None:
        return _as_tensor(sim_outputs["u_prot"], device=device).view(-1)

    if "u" in sim_outputs and sim_outputs["u"] is not None and cfg.protected_index is not None:
        u = _as_tensor(sim_outputs["u"], device=device)
        if u.ndim == 2:
            return u[:, cfg.protected_index].contiguous()
        if u.ndim == 3:
            return u[:, cfg.protected_index, 0].contiguous()
        raise ValueError(f"Unsupported sim_outputs['u'] shape: {tuple(u.shape)}")

    raise KeyError("Missing protected displacement: provide sim_outputs['u_prot'] or (sim_outputs['u'] and MetricsConfig.protected_index).")


def _get_eps(sim_outputs: Mapping[str, Any], cfg: MetricsConfig, device: torch.device) -> torch.Tensor:
    if "eps" in sim_outputs and sim_outputs["eps"] is not None:
        return _as_tensor(sim_outputs["eps"], device=device)
    if "u" in sim_outputs and sim_outputs["u"] is not None:
        strain_fn = cfg.strain_fn or compute_strain
        u = _as_tensor(sim_outputs["u"], device=device)
        coords = _as_tensor(sim_outputs.get("coords"), device=device) if sim_outputs.get("coords") is not None else None
        conn = sim_outputs.get("connectivity")
        return strain_fn(u, coords, conn)
    raise KeyError("Missing strain field: provide sim_outputs['eps'] or sim_outputs['u'] with a strain_fn hook.")


def compute_metrics(sim_outputs: Mapping[str, Any], cfg: MetricsConfig) -> Dict[str, torch.Tensor]:
    """
    Compute scalar metrics from physics outputs.

    Required outputs:
      - strain energy: needs eps or (u + strain_fn)
      - accel/displacement: needs u_prot or (u + protected_index)
      - dt: sim_outputs['dt'] (float or tensor)
    """

    device = None
    for key in ("eps", "u", "u_prot"):
        val = sim_outputs.get(key)
        if isinstance(val, torch.Tensor):
            device = val.device
            break
    device = device or torch.device("cpu")

    dt = sim_outputs.get("dt", None)
    if dt is None:
        raise KeyError("sim_outputs must include 'dt' (timestep).")
    dt_t = _as_tensor(dt, device=device)

    eps = _get_eps(sim_outputs, cfg, device)
    eps = _slice_time_window(eps, dt_t, cfg.t_max)
    if eps.ndim != 2:
        raise ValueError(f"Expected eps shape [Nt, Nspace], got {tuple(eps.shape)}")

    absorb_mask = cfg.absorb_mask
    if absorb_mask is not None:
        absorb_mask = _as_tensor(absorb_mask, device=device).bool().view(-1)
        if absorb_mask.numel() != eps.shape[1]:
            raise ValueError(f"absorb_mask length {absorb_mask.numel()} != eps spatial dim {eps.shape[1]}")
        eps_abs = eps[:, absorb_mask]
    else:
        eps_abs = eps

    weights = None
    for wkey in cfg.weights_key_candidates:
        if wkey in sim_outputs and sim_outputs[wkey] is not None:
            weights = _as_tensor(sim_outputs[wkey], device=device).view(-1)
            break

    if weights is not None:
        if absorb_mask is not None:
            weights = weights[absorb_mask]
        if weights.numel() != eps_abs.shape[1]:
            raise ValueError(f"weights length {weights.numel()} != eps_abs spatial dim {eps_abs.shape[1]}")
        spatial_integral = (eps_abs.pow(2) * weights.view(1, -1)).sum(dim=1)
    else:
        spatial_integral = eps_abs.pow(2).sum(dim=1)

    y_strain_energy = spatial_integral.sum() * dt_t

    u_prot = _get_u_prot(sim_outputs, cfg, device)
    u_prot = _slice_time_window(u_prot, dt_t, cfg.t_max)

    a_prot = second_derivative_central(u_prot, dt_t)
    a_mag = a_prot.abs()
    y_accel_peak = a_mag.max()

    if cfg.accel_smooth_window and cfg.accel_smooth_window > 1:
        a_smooth = moving_average_1d(a_mag, int(cfg.accel_smooth_window))
        y_accel_peak_smooth = a_smooth.max()
    else:
        y_accel_peak_smooth = y_accel_peak

    y_accel_rms = torch.sqrt(torch.mean(a_prot.pow(2)) + 1e-12)
    y_disp_peak = u_prot.abs().max()

    metrics: Dict[str, torch.Tensor] = {
        "y_strain_energy": y_strain_energy,
        "y_accel_peak": y_accel_peak,
        "y_accel_peak_smooth": y_accel_peak_smooth,
        "y_accel_rms": y_accel_rms,
        "y_disp_peak": y_disp_peak,
    }

    if cfg.include_optional and cfg.include_strain_max:
        metrics["y_strain_max"] = eps.abs().max()

    if cfg.include_optional and cfg.include_force_peak:
        f = sim_outputs.get("F_interface", None)
        if f is not None:
            f_t = _as_tensor(f, device=device).view(-1)
            f_t = _slice_time_window(f_t, dt_t, cfg.t_max)
            metrics["y_force_peak"] = f_t.abs().max()

    if cfg.include_optional and cfg.include_damage:
        beta = float(cfg.damage_softplus_beta)
        eps_crit = float(cfg.damage_eps_crit)
        damage_density = F.softplus((eps - eps_crit) * beta) / beta
        if absorb_mask is not None:
            damage_density = damage_density[:, absorb_mask]
        if weights is not None:
            D = (damage_density * weights.view(1, -1)).sum(dim=1).sum() * dt_t
        else:
            D = damage_density.sum(dim=1).sum() * dt_t
        metrics["D_damage"] = D

    return metrics
