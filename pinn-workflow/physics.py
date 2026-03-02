from __future__ import annotations


import torch
import torch.autograd as autograd
import pinn_config as config

def _n_layers() -> int:
    n = int(getattr(config, "NUM_LAYERS", 2))
    if n < 1:
        raise ValueError(f"config.NUM_LAYERS must be >= 1, got {n}")
    return n

def _col_E(layer: int) -> int:
    return 3 + 2 * int(layer)

def _col_t(layer: int) -> int:
    return 3 + 2 * int(layer) + 1

def _col_r() -> int:
    return 3 + 2 * _n_layers()

def _col_mu() -> int:
    return _col_r() + 1

def _col_v0() -> int:
    return _col_r() + 2

def _layer_idx(x: torch.Tensor) -> torch.Tensor:
    """
    x layout: [x,y,z,E1,t1,...,EL,tL,r,mu,v0]
    Returns (N,) long tensor in {0,...,L-1}.
    """
    n_layers = _n_layers()
    if n_layers <= 1:
        return torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
    z = x[:, 2:3]
    t_cols = [torch.clamp(x[:, _col_t(i) : _col_t(i) + 1], min=1e-4) for i in range(n_layers)]
    cum = torch.cumsum(torch.cat(t_cols, dim=1), dim=1)  # (N, L)
    interfaces = [cum[:, i : i + 1] for i in range(n_layers - 1)]
    out = torch.full((x.shape[0],), n_layers - 1, device=x.device, dtype=torch.long)
    for li, zi in enumerate(interfaces):
        m = z < zi
        out[m[:, 0]] = li
    return out


def _select_E_local(x: torch.Tensor) -> torch.Tensor:
    n_layers = _n_layers()
    Es = torch.cat([x[:, _col_E(i) : _col_E(i) + 1] for i in range(n_layers)], dim=1)  # (N, L)
    idx = _layer_idx(x).unsqueeze(1)  # (N, 1)
    E = torch.gather(Es, dim=1, index=idx)  # (N, 1)
    return torch.clamp(E, min=1e-8)

def _E_eff(x: torch.Tensor) -> torch.Tensor:
    """
    Thickness-weighted effective modulus (N,1) for global decode.
    x layout: [x,y,z,E1,t1,...,EL,tL,r,mu,v0]
    """
    n_layers = _n_layers()
    Es = [torch.clamp(x[:, _col_E(i) : _col_E(i) + 1], min=1e-8) for i in range(n_layers)]
    Ts = [torch.clamp(x[:, _col_t(i) : _col_t(i) + 1], min=1e-8) for i in range(n_layers)]
    T = torch.clamp(sum(Ts), min=1e-8)
    num = sum(e * t for e, t in zip(Es, Ts))
    return num / T

def decode_u(v: torch.Tensor, x: torch.Tensor, *, E_override: torch.Tensor | None = None) -> torch.Tensor:
    """
    Convert network output v to displacement u according to config.DISPLACEMENT_DECODE_MODE.
    - none: u=v
    - local: u=v/E_local (or E_override)
    - global: u=v/E_eff (or E_override)
    """
    # Mixed-form networks output [u(3), sigma_voigt(6)].
    if v.shape[1] > 3:
        v = v[:, 0:3]
    mode = str(getattr(config, "DISPLACEMENT_DECODE_MODE", "none")).lower().strip()
    if mode in {"none", "identity", "u"}:
        u = v
    elif mode in {"global", "eff", "e_eff", "avg"}:
        E = E_override if E_override is not None else _E_eff(x)
        u = v / torch.clamp(E, min=1e-8)
    else:
        E = E_override if E_override is not None else _select_E_local(x)
        u = v / torch.clamp(E, min=1e-8)

    # Optional: enforce box-mode clamped sides (x/y min/max) by construction to eliminate
    # rigid-body drift that can satisfy traction BCs with near-zero strain energy.
    if bool(getattr(config, "HARD_CLAMP_SIDES", False)):
        if str(getattr(config, "GEOMETRY_MODE", "box")).lower() == "box" and bool(getattr(config, "BOX_CLAMP_SIDES", False)):
            Lx = float(getattr(config, "Lx", 1.0))
            Ly = float(getattr(config, "Ly", 1.0))
            x0 = torch.clamp(x[:, 0:1], 0.0, Lx)
            y0 = torch.clamp(x[:, 1:2], 0.0, Ly)
            bx = (x0 * (Lx - x0)) / max(1e-12, 0.25 * Lx * Lx)
            by = (y0 * (Ly - y0)) / max(1e-12, 0.25 * Ly * Ly)
            b = bx * by
            u = u * b
    return u

def encode_v(u: torch.Tensor, x: torch.Tensor, *, E_override: torch.Tensor | None = None) -> torch.Tensor:
    """
    Inverse of `decode_u` for displacement-only networks:
      v = u              (mode="none")
      v = u * E_eff      (mode="global")
      v = u * E_local    (mode="local")
    """
    if u.shape[1] > 3:
        u = u[:, 0:3]
    mode = str(getattr(config, "DISPLACEMENT_DECODE_MODE", "none")).lower().strip()
    if mode in {"none", "identity", "u"}:
        return u
    if mode in {"global", "eff", "e_eff", "avg"}:
        E = E_override if E_override is not None else _E_eff(x)
        return u * torch.clamp(E, min=1e-8)
    E = E_override if E_override is not None else _select_E_local(x)
    return u * torch.clamp(E, min=1e-8)


def _total_thickness(x: torch.Tensor) -> torch.Tensor:
    n_layers = _n_layers()
    Ts = [torch.clamp(x[:, _col_t(i) : _col_t(i) + 1], min=1e-4) for i in range(n_layers)]
    return torch.clamp(sum(Ts), min=1e-4)


def _traction_from_stress(sig, normals):
    # sig: (N,3,3), normals: (N,3) -> traction: (N,3)
    n = normals.unsqueeze(2)
    return torch.bmm(sig, n).squeeze(2)

def compliance_scale(E, t):
    e_safe = torch.clamp(E, min=1e-8)
    t_safe = torch.clamp(t, min=1e-8)
    h_ref = float(getattr(config, "H", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    if alpha == 0.0:
        return 1.0 / (e_safe ** e_pow)
    return (1.0 / (e_safe ** e_pow)) * (h_ref / t_safe) ** alpha

def v_to_u(v, E, t):
    return v * compliance_scale(E, t)

def load_mask(x):
    x_coord = x[:, 0]
    y_coord = x[:, 1]
    
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    
    in_patch = (
        (x_coord >= x_min)
        & (x_coord <= x_max)
        & (y_coord >= y_min)
        & (y_coord <= y_max)
    )
    if not bool(getattr(config, "USE_SOFT_LOAD_MASK", True)):
        return torch.where(in_patch, torch.ones_like(x_coord), torch.zeros_like(x_coord))

    # Smooth quadratic falloff (matches fea-workflow/solver/fem_solver.py default).
    # Normalize to [0,1] inside the patch, then apply 16*x(1-x)*y(1-y).
    dx = float(x_max - x_min) if float(x_max - x_min) != 0.0 else 1.0
    dy = float(y_max - y_min) if float(y_max - y_min) != 0.0 else 1.0
    x_norm = (x_coord - float(x_min)) / dx
    y_norm = (y_coord - float(y_min)) / dy
    soft = 16.0 * x_norm * (1.0 - x_norm) * y_norm * (1.0 - y_norm)
    soft = torch.clamp(soft, min=0.0)
    return soft * in_patch.to(dtype=soft.dtype)

def gradient(u, x):
    # u: (N, 3), x: (N, 3)
    # Returns du/dx: (N, 3, 3)
    # [ [dux/dx, dux/dy, dux/dz],
    #   [duy/dx, duy/dy, duy/dz],
    #   [duz/dx, duz/dy, duz/dz] ]
    
    grad_u = torch.zeros(x.shape[0], 3, 3, device=x.device)
    
    for i in range(3): # u_x, u_y, u_z
        u_i = u[:, i].unsqueeze(1)
        grad_i = autograd.grad(
            u_i, x, 
            grad_outputs=torch.ones_like(u_i),
            create_graph=True, 
            retain_graph=True
        )[0]
        # Extract only spatial gradients (first 3 columns: dx, dy, dz)
        grad_u[:, i, :] = grad_i[:, :3]
        
    return grad_u

def strain(grad_u):
    # epsilon = 0.5 * (grad_u + grad_u^T)
    return 0.5 * (grad_u + grad_u.transpose(1, 2))

def stress(eps, lm, mu):
    # sigma = lambda * tr(eps) * I + 2 * mu * eps
    trace_eps = torch.einsum('bii->b', eps).unsqueeze(1).unsqueeze(2) # (N, 1, 1)
    eye = torch.eye(3, device=eps.device).unsqueeze(0).repeat(eps.shape[0], 1, 1)
    
    sigma = lm * trace_eps * eye + 2 * mu * eps
    return sigma

def divergence(sigma, x):
    # sigma: (N, 3, 3), x: (N, 3)
    # div_sigma: (N, 3) vector
    # We need d(sigma_ij)/dx_j
    
    div = torch.zeros(x.shape[0], 3, device=x.device)
    
    # Row 0: d(sig_xx)/dx + d(sig_xy)/dy + d(sig_xz)/dz
    # etc.
    
    for i in range(3): # For each component of force equilibrium
        # We need d(sigma_i0)/dx + d(sigma_i1)/dy + d(sigma_i2)/dz
        div_i = 0
        for j in range(3):
            sig_ij = sigma[:, i, j].unsqueeze(1)
            grad_sig_ij = autograd.grad(
                sig_ij, x,
                grad_outputs=torch.ones_like(sig_ij),
                create_graph=True,
                retain_graph=True
            )[0]
            div_i += grad_sig_ij[:, j]
        div[:, i] = div_i
        
    return div

def compute_loss(model, data, device, weights=None):
    total_loss = 0
    losses = {}
    if weights is None:
        weights = config.WEIGHTS
    
    # --- 1. PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device).detach().clone().requires_grad_(True)
    
    # Dynamic material properties (layered E, fixed nu)
    E_local = _select_E_local(x_int)
    t_local = _total_thickness(x_int)
    nu = float(getattr(config, "NU_FIXED", config.nu_vals[0] if hasattr(config, "nu_vals") else 0.3))
    lm = (E_local * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_int = model(x_int)

    if getattr(config, "ENFORCE_IMPACT_INVARIANCE", False):
        # Neutral-parameter mode: keep restitution/friction effect suppressed.
        x_int_variant = x_int.clone()
        r_min, r_max = getattr(config, "RESTITUTION_RANGE", (0.5, 0.5))
        mu_min, mu_max = getattr(config, "FRICTION_RANGE", (0.3, 0.3))
        v0_min, v0_max = getattr(config, "IMPACT_VELOCITY_RANGE", (1.0, 1.0))
        if r_max > r_min:
            x_int_variant[:, _col_r() : _col_r() + 1] = torch.rand_like(x_int_variant[:, _col_r() : _col_r() + 1]) * (r_max - r_min) + r_min
        if mu_max > mu_min:
            x_int_variant[:, _col_mu() : _col_mu() + 1] = torch.rand_like(x_int_variant[:, _col_mu() : _col_mu() + 1]) * (mu_max - mu_min) + mu_min
        if v0_max > v0_min:
            x_int_variant[:, _col_v0() : _col_v0() + 1] = torch.rand_like(x_int_variant[:, _col_v0() : _col_v0() + 1]) * (v0_max - v0_min) + v0_min
        v_int_variant = model(x_int_variant)
        impact_invariance_loss = torch.mean((v_int - v_int_variant) ** 2)
    else:
        impact_invariance_loss = torch.zeros((), device=x_int.device)
    losses['impact_invariance'] = impact_invariance_loss
    total_loss += weights.get('impact_invariance', 0.0) * impact_invariance_loss

    u = decode_u(v_int, x_int)
    
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    w_pde = float(weights.get("pde", 0.0))
    if w_pde > 0.0:
        div_sigma = divergence(sig, x_int)
        # Equilibrium: -div(sigma) = 0 (scale to stress units)
        residual = -div_sigma * getattr(config, "PDE_LENGTH_SCALE", 1.0)
        pde_loss = torch.mean(residual**2)
        losses["pde"] = pde_loss
        total_loss += w_pde * pde_loss
    else:
        losses["pde"] = torch.zeros((), device=x_int.device)
    
    # Internal strain energy (volume integral, approximated by mean * volume).
    # Prefer unbiased sampling if provided.
    x_energy = data.get("interior_energy", None)
    if x_energy is not None and x_energy.shape[0] > 0:
        x_e = x_energy.to(device).detach().clone().requires_grad_(True)
        E_local_e = _select_E_local(x_e)
        t_local_e = _total_thickness(x_e)
        lm_e = (E_local_e * nu) / ((1 + nu) * (1 - 2 * nu))
        mu_e = E_local_e / (2 * (1 + nu))
        lm_e = lm_e.unsqueeze(2)
        mu_e = mu_e.unsqueeze(2)
        v_e = model(x_e)
        u_e = decode_u(v_e, x_e)
        grad_u_e = gradient(u_e, x_e)
        eps_e = strain(grad_u_e)
        sig_e = stress(eps_e, lm_e, mu_e)
        energy_density = 0.5 * torch.einsum("bij,bij->b", eps_e, sig_e)
        domain_volume = data.get("domain_volume", None)
        if domain_volume is not None:
            vol_t = torch.tensor(float(domain_volume), device=device, dtype=energy_density.dtype)
            internal_energy = energy_density.mean() * vol_t
        else:
            internal_energy = energy_density.mean() * (config.Lx * config.Ly * t_local_e.mean())
    else:
        energy_density = 0.5 * torch.einsum('bij,bij->b', eps, sig)
        domain_volume = data.get("domain_volume", None)
        if domain_volume is not None:
            vol_t = torch.tensor(float(domain_volume), device=device, dtype=energy_density.dtype)
            internal_energy = energy_density.mean() * vol_t
        else:
            internal_energy = energy_density.mean() * (config.Lx * config.Ly * t_local.mean())
    
    # --- 2. Dirichlet BCs (Clamp) ---
    w_clamp = weights.get('clamp', weights.get('bc', 0.0))
    # Box mode (FEA parity): clamp side walls via data["sides"].
    # CAD mode: clamp bottom cap via data["bottom_clamp"].
    if "sides" in data:
        x_clamp = data["sides"][0].to(device)
    else:
        x_clamp = data["bottom_clamp"].to(device)
    v_clamp = model(x_clamp)
    u_clamp = decode_u(v_clamp, x_clamp)
    bc_loss = torch.mean(u_clamp**2)
    losses['bc_sides'] = bc_loss  # kept key name for compatibility with existing logs
    total_loss += w_clamp * bc_loss
    
    # --- 3. Traction BCs (Top & Bottom) ---
    # Top Loaded
    x_top_load = data['top_load'].to(device).detach().clone().requires_grad_(True)
    
    E_local_load = _select_E_local(x_top_load)
    lm = (E_local_load * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local_load / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_top = model(x_top_load)
    u_top = decode_u(v_top, x_top_load)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)

    n_top = data.get("top_load_normal", None)
    if n_top is not None:
        n_top = n_top.to(device)
        T = _traction_from_stress(sig_top, n_top)
        load_dir = str(getattr(config, "CAD_LOAD_DIRECTION", "normal")).lower()
        mask = load_mask(x_top_load).unsqueeze(1)
        if load_dir in {"global_z", "vertical", "z"}:
            target = torch.zeros_like(n_top)
            target[:, 2] = -float(config.p0) * mask[:, 0]
        else:
            target = -float(config.p0) * mask * n_top
        target_load = None
    else:
        T = sig_top[:, :, 2]
        mask = load_mask(x_top_load).unsqueeze(1)  # (N, 1)
        target_load = -config.p0 * mask
        target = torch.cat(
            [
                torch.zeros_like(target_load),
                torch.zeros_like(target_load),
                target_load,
            ],
            dim=1,
        )
    impact_contact_loss = torch.zeros((), device=x_top_load.device)
    friction_coulomb_loss = torch.zeros((), device=x_top_load.device)
    friction_stick_loss = torch.zeros((), device=x_top_load.device)
    if n_top is None and getattr(config, "USE_EXPLICIT_IMPACT_PHYSICS", False):
        # Restitution-aware normal traction:
        # lower restitution -> stronger dissipative/impact contact response.
        restitution_local = torch.clamp(x_top_load[:, _col_r() : _col_r() + 1], 0.0, 1.0)
        impact_velocity_local = torch.clamp(x_top_load[:, _col_v0() : _col_v0() + 1], min=0.0)
        thickness_local = torch.clamp(_total_thickness(x_top_load), min=1e-8)
        compression = torch.relu(-u_top[:, 2:3]) / thickness_local
        gain = float(getattr(config, "IMPACT_RESTITUTION_GAIN", 0.75))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
        v_gain = float(getattr(config, "IMPACT_VELOCITY_GAIN", 0.20))
        v_ratio = impact_velocity_local / max(v0_ref, 1e-8)
        dynamic_scale = 1.0 + v_gain * (v_ratio ** 2)
        restitution_scale = 1.0 + gain * (1.0 - restitution_local) * compression
        target_load_contact = -config.p0 * mask * restitution_scale * dynamic_scale
        impact_contact_loss = torch.mean((T[:, 2:3] - target_load_contact) ** 2)
        losses['impact_contact'] = impact_contact_loss
        total_loss += weights.get('impact_contact', 0.0) * impact_contact_loss

        # Coulomb limit: ||T_t|| <= mu * |T_n| on loaded patch.
        mu_local = torch.clamp(x_top_load[:, _col_mu() : _col_mu() + 1], min=0.0)
        tangential_mag = torch.norm(T[:, :2], dim=1, keepdim=True)
        normal_mag = torch.abs(T[:, 2:3])
        friction_limit = mu_local * normal_mag
        # Slightly stricter friction limit at larger impact velocity.
        friction_limit = friction_limit / torch.sqrt(1.0 + v_gain * (v_ratio ** 2))
        friction_violation = torch.relu(tangential_mag - friction_limit) * mask
        friction_coulomb_loss = torch.mean(friction_violation ** 2)
        losses['friction_coulomb'] = friction_coulomb_loss
        total_loss += weights.get('friction_coulomb', 0.0) * friction_coulomb_loss

        # Stick-style regularization: higher mu discourages tangential slip.
        friction_stick_loss = torch.mean((mu_local * mask * u_top[:, :2]) ** 2)
        losses['friction_stick'] = friction_stick_loss
        total_loss += weights.get('friction_stick', 0.0) * friction_stick_loss

    # Weight by mask so high-pressure regions dominate (helps match the localized indentation seen in FEA).
    mask_power = float(getattr(config, "LOAD_MASK_LOSS_POWER", 1.0))
    w = torch.clamp(mask, min=0.0) ** mask_power
    w = w / (w.mean().clamp_min(1e-8))
    err2 = torch.sum((T - target) ** 2, dim=1, keepdim=True)
    loss_load = torch.mean(w * err2)
    losses['load'] = loss_load
    total_loss += weights['load'] * loss_load
    
    patch_area = (config.LOAD_PATCH_X[1] - config.LOAD_PATCH_X[0]) * (
        config.LOAD_PATCH_Y[1] - config.LOAD_PATCH_Y[0]
    )
    # External work: prefer uniformly-sampled top-load points if provided, to avoid bias from mask-focused sampling.
    x_top_work = data.get("top_load_energy", None)
    if x_top_work is not None and x_top_work.shape[0] > 0:
        x_tw = x_top_work.to(device)
        # NOTE: external work must remain differentiable w.r.t. model parameters;
        # do not wrap in no_grad(), otherwise the energy objective cannot drive
        # the displacement magnitude under the applied load.
        v_tw = model(x_tw)
        u_tw = decode_u(v_tw, x_tw)
        mask_tw = load_mask(x_tw).unsqueeze(1)
        if n_top is None:
            external_work = (-config.p0 * u_tw[:, 2:3] * mask_tw).mean() * patch_area
        else:
            load_area = float(data.get("top_load_area", patch_area))
            load_area_t = torch.tensor(load_area, device=device, dtype=u_tw.dtype)
            target_tw = -float(config.p0) * mask_tw * n_top[:1].to(device)  # approximate with global normal
            external_work = (target_tw * u_tw).sum(dim=1, keepdim=True).mean() * load_area_t
    else:
        if n_top is None:
            external_work = (-config.p0 * u_top[:, 2:3] * mask).mean() * patch_area
        else:
            # Approximate pressure work via Monte Carlo on the loaded surface samples.
            load_area = float(data.get("top_load_area", patch_area))
            load_area_t = torch.tensor(load_area, device=device, dtype=u_top.dtype)
            external_work = (target * u_top).sum(dim=1, keepdim=True).mean() * load_area_t
    # Energy objective (physics-based, no supervision).
    #
    # For static linear elasticity with prescribed tractions, the solution minimizes
    # total potential energy Π(u) = U(u) - W_ext(u), where:
    #   U = ∫ 0.5 * ε:σ dV  (strain energy)
    #   W_ext = ∫ t·u dA    (external work of applied tractions)
    #
    # Alternatively, the energy theorem gives 2U = W_ext at equilibrium; we can penalize
    # (2U - W_ext)^2 as a soft constraint.
    energy_obj = str(getattr(config, "ENERGY_OBJECTIVE", "potential")).lower().strip()
    if energy_obj in {"balance", "theorem", "2u"}:
        energy_resid = (2.0 * internal_energy) - external_work
        losses["energy"] = energy_resid
        total_loss += weights["energy"] * (energy_resid ** 2)
    else:
        potential_energy = internal_energy - external_work
        losses["energy"] = potential_energy
        total_loss += weights["energy"] * potential_energy
    
    # Top Free (traction-free)
    w_free = weights.get('free', weights.get('bc', 0.0))
    x_top_free = data['top_free'].to(device).detach().clone().requires_grad_(True)
    if x_top_free.shape[0] == 0:
        loss_free = torch.zeros((), device=device)
        losses['free_top'] = loss_free
        total_loss += w_free * loss_free
    else:
        E_local_free = _select_E_local(x_top_free)
        lm_free = (E_local_free * nu) / ((1 + nu) * (1 - 2 * nu))
        mu_free = E_local_free / (2 * (1 + nu))
        lm_free = lm_free.unsqueeze(2)
        mu_free = mu_free.unsqueeze(2)
        
        v_top_free = model(x_top_free)
        u_top_free = decode_u(v_top_free, x_top_free)
        grad_u_free = gradient(u_top_free, x_top_free)
        sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
        n_free = data.get("top_free_normal", None)
        if n_free is None:
            # Backward-compatible fallback: assume top plane normal is +z.
            n_free = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=x_top_free.dtype).repeat(x_top_free.shape[0], 1)
        else:
            n_free = n_free.to(device)
        T_free = _traction_from_stress(sig_top_free, n_free)
        loss_free = torch.mean(T_free ** 2)
        losses['free_top'] = loss_free
        total_loss += w_free * loss_free
    
    # Side walls (traction-free)
    x_side_free = data['side_free'].to(device).detach().clone().requires_grad_(True)
    if x_side_free.shape[0] == 0:
        loss_side = torch.zeros((), device=device)
    else:
        n_side = data.get("side_free_normal", None)
        if n_side is None:
            n_side = torch.zeros((x_side_free.shape[0], 3), device=device, dtype=x_side_free.dtype)
        else:
            n_side = n_side.to(device)
        E_side = _select_E_local(x_side_free)
        lm_side = (E_side * nu) / ((1 + nu) * (1 - 2 * nu))
        mu_side = E_side / (2 * (1 + nu))
        lm_side = lm_side.unsqueeze(2)
        mu_side = mu_side.unsqueeze(2)
        v_side = model(x_side_free)
        u_side = decode_u(v_side, x_side_free)
        grad_u_side = gradient(u_side, x_side_free)
        sig_side = stress(strain(grad_u_side), lm_side, mu_side)
        T_side = _traction_from_stress(sig_side, n_side)
        loss_side = torch.mean(T_side ** 2)
    losses["free_side"] = loss_side
    total_loss += w_free * loss_side

    # Bottom (traction-free) in box mode.
    if "bottom" in data:
        x_bot = data["bottom"].to(device).detach().clone().requires_grad_(True)
        if x_bot.shape[0] == 0:
            loss_bot = torch.zeros((), device=device)
        else:
            E_bot = _select_E_local(x_bot)
            lm_bot = (E_bot * nu) / ((1 + nu) * (1 - 2 * nu))
            mu_bot = E_bot / (2 * (1 + nu))
            lm_bot = lm_bot.unsqueeze(2)
            mu_bot = mu_bot.unsqueeze(2)
            v_bot = model(x_bot)
            u_bot = decode_u(v_bot, x_bot)
            grad_u_bot = gradient(u_bot, x_bot)
            sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
            # outward normal for z=0 plane is -z_hat
            n_bot = torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=x_bot.dtype).repeat(x_bot.shape[0], 1)
            T_bot = _traction_from_stress(sig_bot, n_bot)
            loss_bot = torch.mean(T_bot ** 2)
        losses["free_bot"] = loss_bot
        total_loss += w_free * loss_bot
    else:
        # CAD mode: bottom is clamped, so traction-free is skipped.
        loss_bot = torch.zeros((), device=device)
        losses['free_bot'] = loss_bot

    # --- 3.5 Interface continuity losses (bonded layered stack) ---
    interfaces = data.get("interfaces", None)
    n_layers = _n_layers()
    if interfaces is not None and n_layers >= 2 and len(interfaces) >= (n_layers - 1):
        interface_u_losses = []
        interface_t_losses = []
        n_intf = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=x_int.dtype)
        for k, (layer_a, layer_b) in enumerate([(i, i + 1) for i in range(n_layers - 1)]):
            x_intf = interfaces[k].to(device).detach().clone().requires_grad_(True)
            Ea = torch.clamp(x_intf[:, _col_E(layer_a) : _col_E(layer_a) + 1], min=1e-8)
            Eb = torch.clamp(x_intf[:, _col_E(layer_b) : _col_E(layer_b) + 1], min=1e-8)

            va = model(x_intf, layer_idx=layer_a)
            vb = model(x_intf, layer_idx=layer_b)
            ua = decode_u(va, x_intf, E_override=Ea)
            ub = decode_u(vb, x_intf, E_override=Eb)
            interface_u_losses.append(torch.mean((ua - ub) ** 2))

            # traction continuity
            nua = nu
            lm_a = (Ea * nua) / ((1 + nua) * (1 - 2 * nua))
            mu_a = Ea / (2 * (1 + nua))
            lm_a = lm_a.unsqueeze(2)
            mu_a = mu_a.unsqueeze(2)
            grad_ua = gradient(ua, x_intf)
            sig_a = stress(strain(grad_ua), lm_a, mu_a)

            lm_b = (Eb * nua) / ((1 + nua) * (1 - 2 * nua))
            mu_b = Eb / (2 * (1 + nua))
            lm_b = lm_b.unsqueeze(2)
            mu_b = mu_b.unsqueeze(2)
            grad_ub = gradient(ub, x_intf)
            sig_b = stress(strain(grad_ub), lm_b, mu_b)

            n_loc = n_intf.repeat(x_intf.shape[0], 1)
            ta = _traction_from_stress(sig_a, n_loc)
            tb = _traction_from_stress(sig_b, n_loc)
            interface_t_losses.append(torch.mean((ta - tb) ** 2))

        interface_u = torch.stack(interface_u_losses).mean()
        interface_t = torch.stack(interface_t_losses).mean()
    else:
        interface_u = torch.zeros((), device=device)
        interface_t = torch.zeros((), device=device)
    losses["interface_u"] = interface_u
    losses["interface_t"] = interface_t
    total_loss += weights.get("interface_u", 0.0) * interface_u
    total_loss += weights.get("interface_t", 0.0) * interface_t

    # Optional: near-interface displacement continuity band (smooths kinks from hard routing).
    interfaces_band = data.get("interfaces_band", None)
    if interfaces_band is not None and n_layers >= 2 and len(interfaces_band) >= (n_layers - 1):
        band_losses = []
        band_grad_losses = []
        for k, (layer_a, layer_b) in enumerate([(i, i + 1) for i in range(n_layers - 1)]):
            x_band = interfaces_band[k].to(device).detach().clone().requires_grad_(True)
            va = model(x_band, layer_idx=layer_a)
            vb = model(x_band, layer_idx=layer_b)
            band_losses.append(torch.mean((va - vb) ** 2))

            # Match through-thickness gradients near the interface (reduces visible kinks in XZ slices).
            ga = gradient(va, x_band)  # (N,3,3)
            gb = gradient(vb, x_band)
            da_dz = ga[:, :, 2]
            db_dz = gb[:, :, 2]
            band_grad_losses.append(torch.mean((da_dz - db_dz) ** 2))
        interface_band_u = torch.stack(band_losses).mean()
        interface_band_grad = torch.stack(band_grad_losses).mean()
    else:
        interface_band_u = torch.zeros((), device=device)
        interface_band_grad = torch.zeros((), device=device)
    losses["interface_band_u"] = interface_band_u
    total_loss += weights.get("interface_band_u", 0.0) * interface_band_u
    losses["interface_band_grad"] = interface_band_grad
    total_loss += weights.get("interface_band_grad", 0.0) * interface_band_grad
    
    # --- 4. Supervised Data Loss (Hybrid/Parametric) ---
    if 'x_data' in data and 'u_data' in data:
        x_data = data['x_data'].to(device)
        u_data = data['u_data'].to(device)
        
        # Predict v directly (network output space).
        v_pred = model(x_data)
        v_target = encode_v(u_data, x_data)
        
        loss_data = torch.mean((v_pred - v_target)**2)
        losses['data'] = loss_data
        
        # specific weight for data or default high weight
        w_data = weights.get('data', 1.0) 
        total_loss += w_data * loss_data

    # Optional: top-patch u_z anchor (depth calibration).
    # Uses MAE (matches evaluation metric style).
    w_patch = float(weights.get("patch_uz_sup", 0.0))
    x_patch = data.get("patch_sup_x", None)
    uz_patch = data.get("patch_sup_uz", None)
    if w_patch > 0.0 and x_patch is not None and uz_patch is not None:
        xps = x_patch.to(device)
        target = uz_patch.to(device)
        vps = model(xps)
        ups = decode_u(vps, xps)
        patch_sup_loss = torch.mean(torch.abs(ups[:, 2:3] - target))
        losses["patch_uz_sup"] = patch_sup_loss
        total_loss += w_patch * patch_sup_loss
    else:
        losses["patch_uz_sup"] = torch.zeros((), device=device)
    
    losses['total'] = total_loss
    return total_loss, losses

def compute_residuals(model, data, device):
    residuals = {}

    # --- PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device).detach().clone().requires_grad_(True)
    
    E_local = _select_E_local(x_int)
    t_local = _total_thickness(x_int)
    nu = float(getattr(config, "NU_FIXED", config.nu_vals[0] if hasattr(config, "nu_vals") else 0.3))
    lm = (E_local * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_int = model(x_int)
    u = decode_u(v_int, x_int)
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    residual = -div_sigma * getattr(config, "PDE_LENGTH_SCALE", 1.0)
    residual_mag = torch.sqrt(torch.sum(residual**2, dim=1))
    residuals['interior'] = residual_mag.cpu()
    
    # --- Clamp residuals ---
    if "sides" in data:
        x_clamp = data["sides"][0].to(device)
        key = "sides"
    else:
        x_clamp = data["bottom_clamp"].to(device)
        key = "bottom_clamp"
    v_clamp = model(x_clamp)
    u_clamp = decode_u(v_clamp, x_clamp)
    bc_residual = torch.sqrt(torch.sum(u_clamp**2, dim=1))
    residuals[key] = bc_residual.cpu()
    
    # --- Top Load Residuals ---
    x_top_load = data['top_load'].to(device).detach().clone().requires_grad_(True)
    
    E_local_load = _select_E_local(x_top_load)
    lm = (E_local_load * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_local_load / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_top = model(x_top_load)
    u_top = decode_u(v_top, x_top_load)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)
    n_top = data.get("top_load_normal", None)
    if n_top is not None:
        n_top = n_top.to(device)
        T = _traction_from_stress(sig_top, n_top)
        load_dir = str(getattr(config, "CAD_LOAD_DIRECTION", "normal")).lower()
        mask = load_mask(x_top_load).unsqueeze(1)
        if load_dir in {"global_z", "vertical", "z"}:
            target = torch.zeros_like(n_top)
            target[:, 2] = -float(config.p0) * mask[:, 0]
        else:
            target = -float(config.p0) * mask * n_top
    else:
        T = sig_top[:, :, 2]
        mask = load_mask(x_top_load).unsqueeze(1)
        target_load = -config.p0 * mask
        target = torch.cat([
            torch.zeros_like(target_load),
            torch.zeros_like(target_load),
            target_load,
        ], dim=1)
    load_residual = torch.sqrt(torch.sum((T - target) ** 2, dim=1))
    residuals['top_load'] = load_residual.cpu()
    
    # --- Top Free Residuals ---
    x_top_free = data['top_free'].to(device).detach().clone().requires_grad_(True)
    if x_top_free.shape[0] == 0:
        residuals['top_free'] = torch.zeros((0,), device=device).cpu()
    else:
        E_local_free = _select_E_local(x_top_free)
        lm_free = (E_local_free * nu) / ((1 + nu) * (1 - 2 * nu))
        mu_free = E_local_free / (2 * (1 + nu))
        lm_free = lm_free.unsqueeze(2)
        mu_free = mu_free.unsqueeze(2)
        
        v_top_free = model(x_top_free)
        u_top_free = decode_u(v_top_free, x_top_free)
        grad_u_free = gradient(u_top_free, x_top_free)
        sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
        n_free = data.get("top_free_normal", None)
        if n_free is None:
            n_free = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=x_top_free.dtype).repeat(x_top_free.shape[0], 1)
        else:
            n_free = n_free.to(device)
        T_free = _traction_from_stress(sig_top_free, n_free)
        free_residual = torch.sqrt(torch.sum(T_free**2, dim=1))
        residuals['top_free'] = free_residual.cpu()
    
    # --- Side walls residuals (traction-free) ---
    x_side = data['side_free'].to(device).detach().clone().requires_grad_(True)
    if x_side.shape[0] == 0:
        residuals['side_free'] = torch.zeros((0,), device=device).cpu()
    else:
        n_side = data.get("side_free_normal", None)
        if n_side is None:
            n_side = torch.zeros((x_side.shape[0], 3), device=device, dtype=x_side.dtype)
        else:
            n_side = n_side.to(device)
        E_side = _select_E_local(x_side)
        lm_side = (E_side * nu) / ((1 + nu) * (1 - 2 * nu))
        mu_side = E_side / (2 * (1 + nu))
        lm_side = lm_side.unsqueeze(2)
        mu_side = mu_side.unsqueeze(2)
        v_side = model(x_side)
        u_side = decode_u(v_side, x_side)
        grad_u_side = gradient(u_side, x_side)
        sig_side = stress(strain(grad_u_side), lm_side, mu_side)
        T_side = _traction_from_stress(sig_side, n_side)
        side_residual = torch.sqrt(torch.sum(T_side**2, dim=1))
        residuals['side_free'] = side_residual.cpu()
    
    # Bottom traction-free residuals (box mode)
    if "bottom" in data:
        x_bot = data["bottom"].to(device).detach().clone().requires_grad_(True)
        if x_bot.shape[0] == 0:
            residuals["bottom"] = torch.zeros((0,), device=device).cpu()
        else:
            E_bot = _select_E_local(x_bot)
            lm_bot = (E_bot * nu) / ((1 + nu) * (1 - 2 * nu))
            mu_bot = E_bot / (2 * (1 + nu))
            lm_bot = lm_bot.unsqueeze(2)
            mu_bot = mu_bot.unsqueeze(2)
            v_bot = model(x_bot)
            u_bot = decode_u(v_bot, x_bot)
            grad_u_bot = gradient(u_bot, x_bot)
            sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
            n_bot = torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=x_bot.dtype).repeat(x_bot.shape[0], 1)
            T_bot = _traction_from_stress(sig_bot, n_bot)
            bot_residual = torch.sqrt(torch.sum(T_bot**2, dim=1))
            residuals["bottom"] = bot_residual.cpu()

    return residuals
