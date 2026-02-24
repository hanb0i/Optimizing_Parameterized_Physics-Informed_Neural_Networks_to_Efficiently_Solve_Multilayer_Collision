import torch
import torch.optim as optim
import numpy as np
import time
import os

import soap

import pinn_config as config
import data
import model
import physics
import matplotlib.pyplot as plt

def _apply_env_overrides():
    """
    Lightweight runtime overrides for faster smoke training without editing pinn_config.py.
    """
    def _maybe_int(env_key: str, cfg_key: str):
        val = os.getenv(env_key)
        if val is None or val == "":
            return
        try:
            setattr(config, cfg_key, int(val))
            print(f"Override: config.{cfg_key} = {getattr(config, cfg_key)} (from {env_key})")
        except ValueError:
            print(f"Warning: ignored invalid int {env_key}={val!r}")

    def _maybe_float(env_key: str, cfg_key: str):
        val = os.getenv(env_key)
        if val is None or val == "":
            return
        try:
            setattr(config, cfg_key, float(val))
            print(f"Override: config.{cfg_key} = {getattr(config, cfg_key)} (from {env_key})")
        except ValueError:
            print(f"Warning: ignored invalid float {env_key}={val!r}")

    def _maybe_str(env_key: str, cfg_key: str):
        val = os.getenv(env_key)
        if val is None or val == "":
            return
        setattr(config, cfg_key, str(val))
        print(f"Override: config.{cfg_key} = {getattr(config, cfg_key)!r} (from {env_key})")

    _maybe_str("PINN_GEOMETRY_MODE", "GEOMETRY_MODE")

    _maybe_int("PINN_N_INTERIOR", "N_INTERIOR")
    _maybe_int("PINN_N_SIDES", "N_SIDES")
    _maybe_int("PINN_N_TOP_LOAD", "N_TOP_LOAD")
    _maybe_int("PINN_N_TOP_FREE", "N_TOP_FREE")
    _maybe_int("PINN_N_BOTTOM", "N_BOTTOM")
    _maybe_int("PINN_N_INTERFACES", "N_INTERFACES")
    _maybe_int("PINN_LAYERS", "LAYERS")
    _maybe_int("PINN_NEURONS", "NEURONS")

    _maybe_float("PINN_P0", "p0")
    # Weight overrides
    w_map = {
        "PINN_W_PDE": "pde",
        "PINN_W_BC": "bc",
        "PINN_W_LOAD": "load",
        "PINN_W_ENERGY": "energy",
        "PINN_W_INTERFACE_U": "interface_u",
        "PINN_W_INTERFACE_T": "interface_t",
    }
    for env_key, w_key in w_map.items():
        val = os.getenv(env_key)
        if val is None or val == "":
            continue
        try:
            f = float(val)
        except ValueError:
            print(f"Warning: ignored invalid float {env_key}={val!r}")
            continue
        if not hasattr(config, "WEIGHTS") or not isinstance(config.WEIGHTS, dict):
            continue
        if w_key in config.WEIGHTS:
            config.WEIGHTS[w_key] = f
        else:
            config.WEIGHTS[w_key] = f
        print(f"Override: config.WEIGHTS[{w_key!r}] = {config.WEIGHTS[w_key]} (from {env_key})")

    # Boolean overrides
    def _maybe_bool(env_key: str, cfg_key: str):
        val = os.getenv(env_key)
        if val is None or val == "":
            return
        v = val.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            b = True
        elif v in {"0", "false", "no", "n", "off"}:
            b = False
        else:
            print(f"Warning: ignored invalid bool {env_key}={val!r}")
            return
        setattr(config, cfg_key, b)
        print(f"Override: config.{cfg_key} = {getattr(config, cfg_key)!r} (from {env_key})")

    _maybe_bool("PINN_BOX_CLAMP_SIDES", "BOX_CLAMP_SIDES")
    _maybe_bool("PINN_USE_SOFT_LOAD_MASK", "USE_SOFT_LOAD_MASK")
    _maybe_bool("PINN_TRAIN_FIXED_PARAMS", "TRAIN_FIXED_PARAMS")

def _load_compatible_state_dict(pinn, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)

    # If loading an older single-net checkpoint, replicate weights into each layer subnetwork.
    if any(k.startswith("layer.") for k in sd.keys()):
        replicated = {}
        for k, v in sd.items():
            if k.startswith("layer."):
                suffix = k[len("layer.") :]
                for li in range(3):
                    replicated[f"layers.{li}.{suffix}"] = v
            else:
                replicated[k] = v
        sd = replicated

    target_sd = pinn.state_dict()

    def _adapt_first_layer(src_w: torch.Tensor, tgt_w: torch.Tensor) -> torch.Tensor:
        # Map old feature layouts into the new 15-dim feature layout used by LayerNet.
        # Old 11-dim layout (common):
        #   [x,y,z_hat,E_norm,t_norm,r_norm,mu_norm,v0_norm,inv1,inv2,inv3]
        # New 15-dim layout:
        #   [x,y,z_hat,E1_norm,t1_scaled,E2_norm,t2_scaled,E3_norm,t3_scaled,r_norm,mu_norm,v0_norm,inv1,inv2,inv3]
        if src_w.shape[1] == tgt_w.shape[1]:
            return src_w
        if src_w.shape[0] != tgt_w.shape[0]:
            return src_w
        adapted = torch.zeros_like(tgt_w)
        if src_w.shape[1] == 11 and tgt_w.shape[1] == 15:
            adapted[:, 0:3] = src_w[:, 0:3]
            adapted[:, 3] = src_w[:, 3]
            adapted[:, 5] = src_w[:, 3]
            adapted[:, 7] = src_w[:, 3]
            adapted[:, 4] = src_w[:, 4]
            adapted[:, 6] = src_w[:, 4]
            adapted[:, 8] = src_w[:, 4]
            adapted[:, 9:12] = src_w[:, 5:8]
            adapted[:, 12:15] = src_w[:, 8:11]
            return adapted
        if src_w.shape[1] == 10 and tgt_w.shape[1] == 15:
            adapted[:, 0:3] = src_w[:, 0:3]
            adapted[:, 3] = src_w[:, 3]
            adapted[:, 5] = src_w[:, 3]
            adapted[:, 7] = src_w[:, 3]
            adapted[:, 4] = src_w[:, 4]
            adapted[:, 6] = src_w[:, 4]
            adapted[:, 8] = src_w[:, 4]
            adapted[:, 9:11] = src_w[:, 5:7]
            adapted[:, 12:15] = src_w[:, 7:10]
            return adapted
        if src_w.shape[1] == 8 and tgt_w.shape[1] == 15:
            adapted[:, 0:3] = src_w[:, 0:3]
            adapted[:, 3] = src_w[:, 3]
            adapted[:, 5] = src_w[:, 3]
            adapted[:, 7] = src_w[:, 3]
            adapted[:, 4] = src_w[:, 4]
            adapted[:, 6] = src_w[:, 4]
            adapted[:, 8] = src_w[:, 4]
            adapted[:, 12:15] = src_w[:, 5:8]
            return adapted
        # Generic: copy as many leading columns as possible.
        n = min(src_w.shape[1], tgt_w.shape[1])
        adapted[:, :n] = src_w[:, :n]
        return adapted

    for li in range(3):
        w_key = f"layers.{li}.net.0.weight"
        if w_key in sd and w_key in target_sd:
            src_w = sd[w_key]
            tgt_w = target_sd[w_key]
            if src_w.shape != tgt_w.shape:
                sd[w_key] = _adapt_first_layer(src_w, tgt_w)

    # Drop incompatible tensors (e.g., when NEURONS/LAYERS changed) to allow partial warm-start.
    filtered = {}
    dropped = 0
    for k, v in sd.items():
        tv = target_sd.get(k, None)
        if tv is None:
            continue
        if hasattr(v, "shape") and hasattr(tv, "shape") and tuple(v.shape) == tuple(tv.shape):
            filtered[k] = v
        else:
            dropped += 1

    missing, unexpected = pinn.load_state_dict(filtered, strict=False)
    print(f"Warm-start loaded from {ckpt_path}")
    if dropped:
        print(f"  Dropped incompatible tensors: {dropped}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

def get_loss_weights(epoch, use_hard_bc=True):
    weights = dict(config.WEIGHTS)
    if config.WEIGHT_RAMP_EPOCHS > 0 and epoch < config.WEIGHT_RAMP_EPOCHS:
        ramp = max(1, config.WEIGHT_RAMP_EPOCHS)
        t = epoch / ramp
        weights['load'] = config.LOAD_WEIGHT_START + t * (config.WEIGHTS['load'] - config.LOAD_WEIGHT_START)
        weights['pde'] = config.PDE_WEIGHT_START + t * (config.WEIGHTS['pde'] - config.PDE_WEIGHT_START)
        weights['energy'] = config.ENERGY_WEIGHT_START + t * (config.WEIGHTS['energy'] - config.ENERGY_WEIGHT_START)
    if not use_hard_bc:
        weights['pde'] *= config.SOFT_MODE_PDE_WEIGHT_SCALE
        weights['load'] *= config.SOFT_MODE_LOAD_WEIGHT_SCALE
    return weights

def train():
    _apply_env_overrides()

    requested_device = os.getenv("PINN_DEVICE")
    if requested_device:
        device = torch.device(requested_device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Using p0 = {config.p0}")

    epochs_adam = int(os.getenv("PINN_EPOCHS_ADAM", str(config.EPOCHS_ADAM)))
    epochs_lbfgs = int(os.getenv("PINN_EPOCHS_LBFGS", str(config.EPOCHS_LBFGS)))
    log_every = int(os.getenv("PINN_LOG_EVERY", "100"))
    resample_every = int(os.getenv("PINN_RESAMPLE_EVERY", "500"))
    
    # Initialize Model
    pinn = model.MultiLayerPINN().to(device)
    print(pinn)
    if os.getenv("PINN_WARM_START", "1") == "1":
        ckpt_candidates = ["pinn_model.pth", os.path.join("..", "pinn_model.pth")]
        for ckpt in ckpt_candidates:
            if os.path.exists(ckpt):
                _load_compatible_state_dict(pinn, ckpt, device)
                break
    if config.FORCE_SOFT_SIDE_BC_FROM_START:
        config.USE_HARD_SIDE_BC = False
        pinn.set_hard_bc(False)
    if str(getattr(config, "GEOMETRY_MODE", "box")).lower() == "cad":
        # Hard side masks assume a unit box boundary; disable for arbitrary CAD.
        config.USE_HARD_SIDE_BC = False
        pinn.set_hard_bc(False)
    
    # Initialize Optimizers
    # SOAP improves conditioning for stiff, multi-term PINN losses; prefer it when Adam/AdamW stagnates.
    # precondition_frequency controls how often curvature stats are refreshed: lower is more stable, higher is cheaper.
    optimizer_adam = soap.SOAP(
        pinn.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.95, 0.95),
        weight_decay=0.0, #was 1e-2
        precondition_frequency=config.SOAP_PRECONDITION_FREQUENCY,
    )
    
    # Learning rate scheduler: reduce by 0.3 every epochs_adam//5 steps
    scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=config.EPOCHS_ADAM//5, gamma=0.3)
    
    # Load FEM data for comparison
    print("Loading FEM solution for comparison...")
    try:
        fem_data = np.load("fea_solution.npy", allow_pickle=True).item()
        X_fea = fem_data['x']
        Y_fea = fem_data['y']
        Z_fea = fem_data['z']
        U_fea = fem_data['u']
        
        # Prepare FEM evaluation grid for the 3-layer laminate PINN:
        # input layout: [x,y,z,E1,t1,E2,t2,E3,t3,r,mu,v0]
        pts_fea = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1)
        e_ones = np.ones((pts_fea.shape[0], 1)) * config.E_vals[0]
        t_total = float(getattr(config, "H", 0.1))
        t1_ones = np.ones((pts_fea.shape[0], 1)) * (t_total / 3.0)
        t2_ones = np.ones((pts_fea.shape[0], 1)) * (t_total / 3.0)
        t3_ones = np.ones((pts_fea.shape[0], 1)) * (t_total / 3.0)
        r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
        mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
        r_ones = np.ones((pts_fea.shape[0], 1)) * r_ref
        mu_ones = np.ones((pts_fea.shape[0], 1)) * mu_ref
        v0_ones = np.ones((pts_fea.shape[0], 1)) * v0_ref
        pts_fea = np.hstack([pts_fea, e_ones, t1_ones, e_ones, t2_ones, e_ones, t3_ones, r_ones, mu_ones, v0_ones])
        pts_fea_tensor = torch.tensor(pts_fea, dtype=torch.float32).to(device)
        u_fea_flat = U_fea.reshape(-1, 3)
        
        fem_available = True
        print(f"FEM data loaded: {X_fea.shape}")
    except FileNotFoundError:
        print("FEM solution not found. Training without FEM comparison.")
        fem_available = False
    
    # Data Container
    training_data = data.get_data()

    # Load and attach Parametric/Hybrid Supervision Data
    if getattr(config, "USE_SUPERVISION_DATA", True) and hasattr(config, "N_DATA_POINTS") and hasattr(config, "DATA_E_VALUES"):
        print(f"Loading hybrid supervision data (N={config.N_DATA_POINTS}, E={config.DATA_E_VALUES})...")
        x_data, u_data = data.load_fem_supervision_data()
        training_data['x_data'] = x_data
        training_data['u_data'] = u_data
        print(f"Attached {len(x_data)} supervision points to training data.")
    else:
        print("Supervision data disabled or not configured; training physics-only.")
 
    # History - store all loss components separately for each optimizer
    adam_history = {
        'total': [],
        'pde': [],
        'bc_sides': [],
        'free_top': [],
        'free_side': [],
        'free_bot': [],
        'load': [],
        'energy': [],
        'interface_u': [],
        'interface_t': [],
        'impact_invariance': [],
        'impact_contact': [],
        'friction_coulomb': [],
        'friction_stick': [],
        'fem_mae': [],
        'fem_max_err': [],
        'epochs': []
    }
    
    lbfgs_history = {
        'total': [],
        'pde': [],
        'bc_sides': [],
        'free_top': [],
        'free_side': [],
        'free_bot': [],
        'load': [],
        'energy': [],
        'interface_u': [],
        'interface_t': [],
        'impact_invariance': [],
        'impact_contact': [],
        'friction_coulomb': [],
        'friction_stick': [],
        'fem_mae': [],
        'fem_max_err': [],
        'steps': []
    }
    
    print("Starting SOAP Pretraining...")
    start_time = time.time()
    last_time = start_time
    best_fem_mae = float("inf")
    best_fem_epoch = None
    
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()

        if config.FORCE_SOFT_SIDE_BC_FROM_START:
            use_hard_bc = False
        else:
            use_hard_bc = epoch < config.HARD_BC_EPOCHS
            if config.USE_HARD_SIDE_BC != use_hard_bc:
                config.USE_HARD_SIDE_BC = use_hard_bc
                pinn.set_hard_bc(use_hard_bc)
                if not use_hard_bc:
                    print("Switching to soft side BCs (mask off) to lift deflection.")
        
        if resample_every > 0 and epoch % resample_every == 0 and epoch > 0:
            # Compute residuals for adaptive sampling
            residuals = physics.compute_residuals(pinn, training_data, device)
            training_data = data.get_data(prev_data=training_data, residuals=residuals)
            print(f"  Resampled with residual-based adaptive sampling at epoch {epoch}")
            
        weights = get_loss_weights(epoch, use_hard_bc)
        loss_val, losses = physics.compute_loss(pinn, training_data, device, weights=weights)
        loss_val.backward()
        optimizer_adam.step()
        scheduler.step()  # Update learning rate
        
        adam_history['total'].append(loss_val.item())
        adam_history['pde'].append(losses['pde'].item())
        adam_history['bc_sides'].append(losses['bc_sides'].item())
        adam_history['free_top'].append(losses['free_top'].item())
        adam_history['free_side'].append(losses.get('free_side', torch.tensor(0.0)).item())
        adam_history['free_bot'].append(losses['free_bot'].item())
        adam_history['load'].append(losses['load'].item())
        adam_history['energy'].append(losses['energy'].item())
        adam_history['interface_u'].append(losses.get('interface_u', torch.tensor(0.0)).item())
        adam_history['interface_t'].append(losses.get('interface_t', torch.tensor(0.0)).item())
        adam_history['impact_invariance'].append(losses.get('impact_invariance', torch.tensor(0.0)).item())
        adam_history['impact_contact'].append(losses.get('impact_contact', torch.tensor(0.0)).item())
        adam_history['friction_coulomb'].append(losses.get('friction_coulomb', torch.tensor(0.0)).item())
        adam_history['friction_stick'].append(losses.get('friction_stick', torch.tensor(0.0)).item())
        
        if log_every > 0 and epoch % log_every == 0:
            current_time = time.time()
            step_duration = current_time - last_time
            last_time = current_time
            current_lr = scheduler.get_last_lr()[0]
            
            # Compute FEM error every 100 epochs
            if fem_available:
                with torch.no_grad():
                    v_pinn_flat = pinn(pts_fea_tensor).cpu().numpy()
                    # Decode u from v using the same convention as the physics loss for FEA parity: u = v / E_local.
                    z = pts_fea[:, 2:3]
                    t1 = np.clip(pts_fea[:, 4:5], 1e-8, None)
                    t2 = np.clip(pts_fea[:, 6:7], 1e-8, None)
                    z1 = t1
                    z2 = t1 + t2
                    E1 = np.clip(pts_fea[:, 3:4], 1e-8, None)
                    E2 = np.clip(pts_fea[:, 5:6], 1e-8, None)
                    E3 = np.clip(pts_fea[:, 7:8], 1e-8, None)
                    E_local = np.where(z < z1, E1, np.where(z < z2, E2, E3))
                    u_pinn_flat = v_pinn_flat / E_local
                     
                    diff = np.abs(u_pinn_flat - u_fea_flat)
                    mae = np.mean(diff)
                    max_err = np.max(diff)
                    adam_history['fem_mae'].append(mae)
                    adam_history['fem_max_err'].append(max_err)
                    adam_history['epochs'].append(epoch)

                    if mae < best_fem_mae:
                        best_fem_mae = float(mae)
                        best_fem_epoch = int(epoch)
                        torch.save(pinn.state_dict(), "pinn_model_best_fem.pth")
                        print(f"  New best FEM MAE: {best_fem_mae:.6f} at epoch {best_fem_epoch} (saved pinn_model_best_fem.pth)")
                    
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | BC_sides: {losses['bc_sides']:.6f} | "
                      f"Free_top: {losses['free_top']:.6f} | Free_side: {losses.get('free_side', torch.tensor(0.0)).item():.6f} | Free_bot: {losses['free_bot']:.6f} | "
                      f"Load: {losses['load']:.6f} | Energy: {losses['energy']:.6f} | "
                      f"IntU: {losses.get('interface_u', torch.tensor(0.0)).item():.6f} | IntT: {losses.get('interface_t', torch.tensor(0.0)).item():.6f} | "
                      f"ImpactC: {losses.get('impact_contact', torch.tensor(0.0)).item():.6f} | "
                      f"FricC: {losses.get('friction_coulomb', torch.tensor(0.0)).item():.6f} | "
                      f"FricS: {losses.get('friction_stick', torch.tensor(0.0)).item():.6f} | "
                      f"ImpactInv: {losses.get('impact_invariance', torch.tensor(0.0)).item():.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"FEM MAE: {mae:.6f} | Time: {step_duration:.4f}s")
            else:
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | BC_sides: {losses['bc_sides']:.6f} | "
                      f"Free_top: {losses['free_top']:.6f} | Free_side: {losses.get('free_side', torch.tensor(0.0)).item():.6f} | Free_bot: {losses['free_bot']:.6f} | "
                      f"Load: {losses['load']:.6f} | Energy: {losses['energy']:.6f} | "
                      f"IntU: {losses.get('interface_u', torch.tensor(0.0)).item():.6f} | IntT: {losses.get('interface_t', torch.tensor(0.0)).item():.6f} | "
                      f"ImpactC: {losses.get('impact_contact', torch.tensor(0.0)).item():.6f} | "
                      f"FricC: {losses.get('friction_coulomb', torch.tensor(0.0)).item():.6f} | "
                      f"FricS: {losses.get('friction_stick', torch.tensor(0.0)).item():.6f} | "
                      f"ImpactInv: {losses.get('impact_invariance', torch.tensor(0.0)).item():.6f} | "
                      f"LR: {current_lr:.2e} | Time: {step_duration:.4f}s")
            
    print(f"SOAP Pretraining Complete. Total Time: {time.time() - start_time:.2f}s")
    if fem_available and best_fem_epoch is not None:
        print(f"Best FEM MAE during Adam: {best_fem_mae:.6f} at epoch {best_fem_epoch} (pinn_model_best_fem.pth)")
    
    # L-BFGS Fine-Tuning
    print("Starting L-BFGS Fine-Tuning...")
    config.USE_HARD_SIDE_BC = False
    pinn.set_hard_bc(False)
    optimizer_lbfgs = optim.LBFGS(
        pinn.parameters(),
        lr=1.0,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )
        
    num_lbfgs_steps = epochs_lbfgs
    print(f"Running {num_lbfgs_steps} L-BFGS outer steps.")
    print("Using fixed collocation set during L-BFGS for stability.")
    
    for i in range(num_lbfgs_steps):
        # Keep collocation points fixed during L-BFGS for stability
        
        step_start = time.time()
        def closure():
            optimizer_lbfgs.zero_grad()
            loss_val, _ = physics.compute_loss(pinn, training_data, device, weights=config.WEIGHTS)
            loss_val.backward()
            return loss_val

        loss_val = optimizer_lbfgs.step(closure)
        step_end = time.time()
        
        # Compute losses for logging
        _, losses = physics.compute_loss(pinn, training_data, device, weights=config.WEIGHTS)
        lbfgs_history['total'].append(loss_val.item())
        lbfgs_history['pde'].append(losses['pde'].item())
        lbfgs_history['bc_sides'].append(losses['bc_sides'].item())
        lbfgs_history['free_top'].append(losses['free_top'].item())
        lbfgs_history['free_side'].append(losses.get('free_side', torch.tensor(0.0)).item())
        lbfgs_history['free_bot'].append(losses['free_bot'].item())
        lbfgs_history['load'].append(losses['load'].item())
        lbfgs_history['energy'].append(losses['energy'].item())
        lbfgs_history['interface_u'].append(losses.get('interface_u', torch.tensor(0.0)).item())
        lbfgs_history['interface_t'].append(losses.get('interface_t', torch.tensor(0.0)).item())
        lbfgs_history['impact_invariance'].append(losses.get('impact_invariance', torch.tensor(0.0)).item())
        lbfgs_history['impact_contact'].append(losses.get('impact_contact', torch.tensor(0.0)).item())
        lbfgs_history['friction_coulomb'].append(losses.get('friction_coulomb', torch.tensor(0.0)).item())
        lbfgs_history['friction_stick'].append(losses.get('friction_stick', torch.tensor(0.0)).item())
        
        # Compute FEM error and print
        if fem_available:
            with torch.no_grad():
                v_pinn_flat = pinn(pts_fea_tensor).cpu().numpy()
                # Decode u from v using u = v / E_local (matches physics/verify).
                z = pts_fea[:, 2:3]
                t1 = np.clip(pts_fea[:, 4:5], 1e-8, None)
                t2 = np.clip(pts_fea[:, 6:7], 1e-8, None)
                z1 = t1
                z2 = t1 + t2
                E1 = np.clip(pts_fea[:, 3:4], 1e-8, None)
                E2 = np.clip(pts_fea[:, 5:6], 1e-8, None)
                E3 = np.clip(pts_fea[:, 7:8], 1e-8, None)
                E_local = np.where(z < z1, E1, np.where(z < z2, E2, E3))
                u_pinn_flat = v_pinn_flat / E_local
                diff = np.abs(u_pinn_flat - u_fea_flat)
                mae = np.mean(diff)
                max_err = np.max(diff)
                lbfgs_history['fem_mae'].append(mae)
                lbfgs_history['fem_max_err'].append(max_err)
                lbfgs_history['steps'].append(i)

                if mae < best_fem_mae:
                    best_fem_mae = float(mae)
                    best_fem_epoch = f"lbfgs_step_{i}"
                    torch.save(pinn.state_dict(), "pinn_model_best_fem.pth")
                    print(f"  New best FEM MAE: {best_fem_mae:.6e} at {best_fem_epoch} (saved pinn_model_best_fem.pth)")
            print(f"L-BFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_side: {losses.get('free_side', torch.tensor(0.0)).item():.6e} | Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"Energy: {losses['energy'].item():.6e} | "
                  f"IntU: {losses.get('interface_u', torch.tensor(0.0)).item():.6e} | IntT: {losses.get('interface_t', torch.tensor(0.0)).item():.6e} | "
                  f"ImpactC: {losses.get('impact_contact', torch.tensor(0.0)).item():.6e} | "
                  f"FricC: {losses.get('friction_coulomb', torch.tensor(0.0)).item():.6e} | "
                  f"FricS: {losses.get('friction_stick', torch.tensor(0.0)).item():.6e} | "
                  f"ImpactInv: {losses.get('impact_invariance', torch.tensor(0.0)).item():.6e} | "
                  f"FEM MAE: {mae:.6e} | Time: {step_end - step_start:.4f}s")
        else:
            print(f"L-BFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_side: {losses.get('free_side', torch.tensor(0.0)).item():.6e} | Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"Energy: {losses['energy'].item():.6e} | "
                  f"IntU: {losses.get('interface_u', torch.tensor(0.0)).item():.6e} | IntT: {losses.get('interface_t', torch.tensor(0.0)).item():.6e} | "
                  f"ImpactC: {losses.get('impact_contact', torch.tensor(0.0)).item():.6e} | "
                  f"FricC: {losses.get('friction_coulomb', torch.tensor(0.0)).item():.6e} | "
                  f"FricS: {losses.get('friction_stick', torch.tensor(0.0)).item():.6e} | "
                  f"ImpactInv: {losses.get('impact_invariance', torch.tensor(0.0)).item():.6e} | "
                  f"Time: {step_end - step_start:.4f}s")
        
        # Save model at every L-BFGS step
        torch.save(pinn.state_dict(), "pinn_model.pth")
            
    # Save Model and Loss Histories
    torch.save(pinn.state_dict(), "pinn_model.pth")
    loss_history = {'adam': adam_history, 'lbfgs': lbfgs_history}
    np.save("loss_history.npy", loss_history)
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
