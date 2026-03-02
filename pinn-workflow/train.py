from __future__ import annotations

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
    _maybe_int("PINN_N_INTERFACE_BAND", "N_INTERFACE_BAND")
    _maybe_int("PINN_N_INTERIOR_ENERGY", "N_INTERIOR_ENERGY")
    _maybe_int("PINN_N_TOP_LOAD_ENERGY", "N_TOP_LOAD_ENERGY")
    _maybe_int("PINN_LAYERS", "LAYERS")
    _maybe_int("PINN_NEURONS", "NEURONS")
    _maybe_int("PINN_EPOCHS_ADAM", "EPOCHS_ADAM")
    _maybe_int("PINN_EPOCHS_LBFGS", "EPOCHS_LBFGS")

    _maybe_float("PINN_P0", "p0")
    _maybe_float("PINN_PDE_LENGTH_SCALE", "PDE_LENGTH_SCALE")
    # Weight overrides
    w_map = {
        "PINN_W_PDE": "pde",
        "PINN_W_CONSTITUTIVE": "constitutive",
        "PINN_W_BC": "bc",
        "PINN_W_CLAMP": "clamp",
        "PINN_W_FREE": "free",
        "PINN_W_LOAD": "load",
        "PINN_W_ENERGY": "energy",
        "PINN_W_INTERFACE_U": "interface_u",
        "PINN_W_INTERFACE_T": "interface_t",
        "PINN_W_INTERFACE_BAND_U": "interface_band_u",
        "PINN_W_INTERFACE_BAND_GRAD": "interface_band_grad",
        "PINN_W_PATCH_UZ_SUP": "patch_uz_sup",
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
    _maybe_bool("PINN_HARD_CLAMP_SIDES", "HARD_CLAMP_SIDES")
    _maybe_bool("PINN_USE_SOFT_LOAD_MASK", "USE_SOFT_LOAD_MASK")
    _maybe_bool("PINN_TRAIN_FIXED_PARAMS", "TRAIN_FIXED_PARAMS")
    _maybe_bool("PINN_FORCE_SOFT_SIDE_BC_FROM_START", "FORCE_SOFT_SIDE_BC_FROM_START")
    _maybe_bool("PINN_USE_HARD_SIDE_BC", "USE_HARD_SIDE_BC")
    _maybe_bool("PINN_USE_EXPLICIT_IMPACT_PHYSICS", "USE_EXPLICIT_IMPACT_PHYSICS")
    _maybe_bool("PINN_USE_MIXED_FORMULATION", "USE_MIXED_FORMULATION")

    _maybe_str("PINN_HARD_BC_SCHEDULE", "HARD_BC_SCHEDULE")
    _maybe_str("PINN_ENERGY_OBJECTIVE", "ENERGY_OBJECTIVE")
    _maybe_float("PINN_HARD_SIDE_BC_POWER", "HARD_SIDE_BC_POWER")
    _maybe_float("PINN_HARD_SIDE_BC_POWER_START", "HARD_SIDE_BC_POWER_START")
    _maybe_float("PINN_HARD_SIDE_BC_POWER_END", "HARD_SIDE_BC_POWER_END")

    _maybe_str("PINN_DECODE_MODE", "DISPLACEMENT_DECODE_MODE")
    _maybe_str("PINN_FEA_NPY", "FEA_NPY_PATH")
    _maybe_float("PINN_TRAIN_FIXED_E", "TRAIN_FIXED_E")
    _maybe_float("PINN_TRAIN_FIXED_TOTAL_THICKNESS", "TRAIN_FIXED_TOTAL_THICKNESS")
    _maybe_float("PINN_TRAIN_FIXED_E1", "TRAIN_FIXED_E1")
    _maybe_float("PINN_TRAIN_FIXED_E2", "TRAIN_FIXED_E2")
    _maybe_float("PINN_TRAIN_FIXED_E3", "TRAIN_FIXED_E3")
    _maybe_float("PINN_TRAIN_FIXED_T1", "TRAIN_FIXED_T1")
    _maybe_float("PINN_TRAIN_FIXED_T2", "TRAIN_FIXED_T2")
    _maybe_float("PINN_TRAIN_FIXED_T3", "TRAIN_FIXED_T3")
    _maybe_int("PINN_HARD_BC_EPOCHS", "HARD_BC_EPOCHS")
    _maybe_float("PINN_LOAD_MASK_SAMPLING_POWER", "LOAD_MASK_SAMPLING_POWER")
    _maybe_float("PINN_LOAD_MASK_LOSS_POWER", "LOAD_MASK_LOSS_POWER")
    _maybe_float("PINN_LOAD_MASK_SAMPLING_BIASED_FRACTION", "LOAD_MASK_SAMPLING_BIASED_FRACTION")
    _maybe_float("PINN_TOP_FREE_RING_FRACTION", "TOP_FREE_RING_FRACTION")
    _maybe_float("PINN_TOP_FREE_RING_WIDTH_FRAC", "TOP_FREE_RING_WIDTH_FRAC")
    _maybe_str("PINN_LAYER_GATING", "LAYER_GATING")
    _maybe_float("PINN_LAYER_GATE_BETA", "LAYER_GATE_BETA")
    _maybe_int("PINN_WEIGHT_RAMP_EPOCHS", "WEIGHT_RAMP_EPOCHS")
    _maybe_float("PINN_LOAD_WEIGHT_START", "LOAD_WEIGHT_START")
    _maybe_float("PINN_PDE_WEIGHT_START", "PDE_WEIGHT_START")
    _maybe_float("PINN_ENERGY_WEIGHT_START", "ENERGY_WEIGHT_START")
    _maybe_float("PINN_SOFT_MODE_PDE_WEIGHT_SCALE", "SOFT_MODE_PDE_WEIGHT_SCALE")
    _maybe_float("PINN_SOFT_MODE_LOAD_WEIGHT_SCALE", "SOFT_MODE_LOAD_WEIGHT_SCALE")
    _maybe_float("PINN_LR", "LEARNING_RATE")
    _maybe_int("PINN_PATCH_FOCUS_EPOCH", "PATCH_FOCUS_EPOCH")
    _maybe_float("PINN_PATCH_FOCUS_MASK_SAMPLING_POWER", "PATCH_FOCUS_MASK_SAMPLING_POWER")
    _maybe_float("PINN_PATCH_FOCUS_MASK_LOSS_POWER", "PATCH_FOCUS_MASK_LOSS_POWER")
    _maybe_int("PINN_PATCH_UZ_SUP_DECAY_EPOCH", "PATCH_UZ_SUP_DECAY_EPOCH")
    _maybe_float("PINN_PATCH_UZ_SUP_DECAY_FACTOR", "PATCH_UZ_SUP_DECAY_FACTOR")

def _load_compatible_state_dict(pinn, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    n_layers = len(getattr(pinn, "layers", [])) or int(getattr(config, "NUM_LAYERS", 2))

    # If loading an older single-net checkpoint, replicate weights into each layer subnetwork.
    if any(k.startswith("layer.") for k in sd.keys()):
        replicated = {}
        for k, v in sd.items():
            if k.startswith("layer."):
                suffix = k[len("layer.") :]
                for li in range(n_layers):
                    replicated[f"layers.{li}.{suffix}"] = v
            else:
                replicated[k] = v
        sd = replicated

    target_sd = pinn.state_dict()

    def _adapt_first_layer(src_w: torch.Tensor, tgt_w: torch.Tensor) -> torch.Tensor:
        # Map old feature layouts into the current LayerNet feature layout.
        # Expected feature layout (dim = 9 + 2*L):
        #   [x,y,z_hat, E1_norm,t1_scaled,...,EL_norm,tL_scaled, r_norm,mu_norm,v0_norm, inv1,inv2,inv3]
        if src_w.shape[1] == tgt_w.shape[1] or src_w.shape[0] != tgt_w.shape[0]:
            return src_w

        def _infer_L(d: int) -> int | None:
            if d < 9:
                return None
            if (d - 9) % 2 != 0:
                return None
            return (d - 9) // 2

        Ls = _infer_L(int(src_w.shape[1]))
        Lt = _infer_L(int(tgt_w.shape[1]))
        if Ls is None or Lt is None:
            adapted = torch.zeros_like(tgt_w)
            n = min(int(src_w.shape[1]), int(tgt_w.shape[1]))
            adapted[:, :n] = src_w[:, :n]
            return adapted

        adapted = torch.zeros_like(tgt_w)
        adapted[:, 0:3] = src_w[:, 0:3]  # x,y,z_hat

        # Per-layer (E_norm, t_scaled): copy min(Ls,Lt), then repeat last available.
        for i in range(Lt):
            si = min(i, max(0, Ls - 1))
            adapted[:, 3 + 2 * i] = src_w[:, 3 + 2 * si]
            adapted[:, 4 + 2 * i] = src_w[:, 4 + 2 * si]

        # Impact params: old checkpoints may omit v0 or all impact params.
        src_imp_start = 3 + 2 * Ls
        tgt_imp_start = 3 + 2 * Lt
        if int(src_w.shape[1]) >= src_imp_start + 3:
            adapted[:, tgt_imp_start : tgt_imp_start + 3] = src_w[:, src_imp_start : src_imp_start + 3]
        elif int(src_w.shape[1]) >= src_imp_start + 2:
            adapted[:, tgt_imp_start : tgt_imp_start + 2] = src_w[:, src_imp_start : src_imp_start + 2]
            adapted[:, tgt_imp_start + 2] = 0.5  # v0_norm default
        else:
            adapted[:, tgt_imp_start : tgt_imp_start + 3] = 0.5

        # Invariants: last 3 columns (inv1,inv2,inv3) if present, else zeros.
        if int(src_w.shape[1]) >= 3:
            adapted[:, -3:] = src_w[:, -3:]
        return adapted

    for li in range(n_layers):
        w_key = f"layers.{li}.net.0.weight"
        if w_key in sd and w_key in target_sd:
            src_w = sd[w_key]
            tgt_w = target_sd[w_key]
            if src_w.shape != tgt_w.shape:
                sd[w_key] = _adapt_first_layer(src_w, tgt_w)

    # Adapt output layer when switching between displacement-only (3) and mixed (9) outputs.
    # Preserve the learned displacement rows and initialize extra stress rows to 0 for stability.
    for li in range(n_layers):
        # Find the last Linear layer index for this layer net.
        prefix = f"layers.{li}.net."
        idxs = []
        for k in target_sd.keys():
            if not (k.startswith(prefix) and k.endswith(".weight")):
                continue
            try:
                idxs.append(int(k.split(".")[3]))
            except Exception:
                continue
        if not idxs:
            continue
        out_idx = max(idxs)
        w_key = f"layers.{li}.net.{out_idx}.weight"
        b_key = f"layers.{li}.net.{out_idx}.bias"
        if w_key in sd and w_key in target_sd:
            src_w = sd[w_key]
            tgt_w = target_sd[w_key]
            if src_w.shape != tgt_w.shape and src_w.shape[1] == tgt_w.shape[1]:
                out_src, out_tgt = int(src_w.shape[0]), int(tgt_w.shape[0])
                if out_src == 3 and out_tgt == 9:
                    w_new = torch.zeros_like(tgt_w)
                    w_new[0:3, :] = src_w
                    sd[w_key] = w_new
                elif out_src == 9 and out_tgt == 3:
                    sd[w_key] = src_w[0:3, :].clone()
        if b_key in sd and b_key in target_sd:
            src_b = sd[b_key]
            tgt_b = target_sd[b_key]
            if src_b.shape != tgt_b.shape:
                out_src, out_tgt = int(src_b.shape[0]), int(tgt_b.shape[0])
                if out_src == 3 and out_tgt == 9:
                    b_new = torch.zeros_like(tgt_b)
                    b_new[0:3] = src_b
                    sd[b_key] = b_new
                elif out_src == 9 and out_tgt == 3:
                    sd[b_key] = src_b[0:3].clone()

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
        load0 = float(getattr(config, "LOAD_WEIGHT_START", weights.get("load", 1.0)))
        pde0 = float(getattr(config, "PDE_WEIGHT_START", weights.get("pde", 1.0)))
        en0 = float(getattr(config, "ENERGY_WEIGHT_START", weights.get("energy", 0.0)))
        weights['load'] = load0 + t * (config.WEIGHTS.get('load', weights['load']) - load0)
        weights['pde'] = pde0 + t * (config.WEIGHTS.get('pde', weights['pde']) - pde0)
        weights['energy'] = en0 + t * (config.WEIGHTS.get('energy', weights['energy']) - en0)
    if not use_hard_bc:
        weights['pde'] *= config.SOFT_MODE_PDE_WEIGHT_SCALE
        weights['load'] *= config.SOFT_MODE_LOAD_WEIGHT_SCALE

    # Optional patch u_z anchor decay schedule.
    decay_epoch = getattr(config, "PATCH_UZ_SUP_DECAY_EPOCH", None)
    if decay_epoch is not None and int(decay_epoch) >= 0 and epoch >= int(decay_epoch):
        factor = float(getattr(config, "PATCH_UZ_SUP_DECAY_FACTOR", 0.1))
        if "patch_uz_sup" in weights:
            weights["patch_uz_sup"] = float(weights["patch_uz_sup"]) * factor
    return weights

def train():
    _apply_env_overrides()
    run_tag = os.getenv("PINN_RUN_TAG", "").strip()
    warm_start_path = os.getenv("PINN_WARM_START_PATH", "").strip()

    def _tag_path(path: str) -> str:
        if not run_tag:
            return path
        root, ext = os.path.splitext(path)
        return f"{root}_{run_tag}{ext}"

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
    stop_global_mae_pct_env = os.getenv("PINN_STOP_GLOBAL_MAE_PCT", "").strip()
    stop_patch_top_uz_pct_env = os.getenv("PINN_STOP_PATCH_TOP_UZ_PCT", "").strip()
    try:
        stop_global_mae_pct = float(stop_global_mae_pct_env) if stop_global_mae_pct_env else None
    except ValueError:
        stop_global_mae_pct = None
    try:
        stop_patch_top_uz_mae_pct = float(stop_patch_top_uz_pct_env) if stop_patch_top_uz_pct_env else None
    except ValueError:
        stop_patch_top_uz_mae_pct = None
    
    # Initialize Model
    pinn = model.MultiLayerPINN().to(device)
    print(pinn)
    if warm_start_path:
        if os.path.exists(warm_start_path):
            _load_compatible_state_dict(pinn, warm_start_path, device)
        else:
            print(f"Warm-start path not found: {warm_start_path!r} (continuing without warm-start)")
    elif os.getenv("PINN_WARM_START", "1") == "1":
        ckpt_candidates = [
            "pinn_model_best_fem.pth",
            "pinn_model.pth",
            os.path.join("..", "pinn_model_best_fem.pth"),
            os.path.join("..", "pinn_model.pth"),
        ]
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
    
    # Learning rate scheduler: reduce by 0.3 every ~epochs_adam//5 steps (min 1).
    scheduler = optim.lr_scheduler.StepLR(
        optimizer_adam,
        step_size=max(1, int(config.EPOCHS_ADAM // 5)),
        gamma=0.3,
    )
    
    # Load FEM data for comparison
    print("Loading FEM solution for comparison...")
    try:
        fea_path = getattr(config, "FEA_NPY_PATH", "fea_solution.npy")
        fem_data = np.load(fea_path, allow_pickle=True).item()
        X_fea = fem_data['x']
        Y_fea = fem_data['y']
        Z_fea = fem_data['z']
        U_fea = fem_data['u']
        
        # Prepare FEM evaluation grid for the layered PINN:
        # input layout: [x,y,z,E1,t1,...,EL,tL,r,mu,v0]
        pts_xyz = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1).astype(np.float32, copy=False)
        n_layers = int(getattr(config, "NUM_LAYERS", 2))
        default_E = float(getattr(config, "TRAIN_FIXED_E", getattr(config, "E_vals", [1.0])[0]))

        E_list = []
        for i in range(n_layers):
            e_cfg = getattr(config, f"TRAIN_FIXED_E{i+1}", None)
            E_list.append(float(default_E if e_cfg is None else e_cfg))

        t_total = float(getattr(config, "TRAIN_FIXED_TOTAL_THICKNESS", getattr(config, "H", 0.1)))
        t_cfgs = [getattr(config, f"TRAIN_FIXED_T{i+1}", None) for i in range(n_layers)]
        if all(v is None for v in t_cfgs):
            t_list = [t_total / float(n_layers)] * n_layers
        else:
            t_list = [float((t_total / n_layers) if v is None else v) for v in t_cfgs]
            tsum = max(1e-8, float(sum(t_list)))
            scale = t_total / tsum
            t_list = [tv * scale for tv in t_list]

        params_cols = []
        for Ei, ti in zip(E_list, t_list):
            params_cols.append(np.ones((pts_xyz.shape[0], 1), dtype=np.float32) * float(Ei))
            params_cols.append(np.ones((pts_xyz.shape[0], 1), dtype=np.float32) * float(ti))
        r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
        mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
        r_ones = np.ones((pts_fea.shape[0], 1)) * r_ref
        mu_ones = np.ones((pts_fea.shape[0], 1)) * mu_ref
        v0_ones = np.ones((pts_fea.shape[0], 1)) * v0_ref
        pts_fea = np.hstack([pts_xyz, *params_cols, r_ones.astype(np.float32), mu_ones.astype(np.float32), v0_ones.astype(np.float32)])
        pts_fea_tensor = torch.tensor(pts_fea, dtype=torch.float32).to(device)
        u_fea_flat = U_fea.reshape(-1, 3)
        u_fea_abs_max = float(np.max(np.abs(u_fea_flat)))
        u_fea_top_uz = U_fea[:, :, -1, 2].astype(np.float32, copy=False)
        top_uz_denom = float(np.max(np.abs(u_fea_top_uz)))
        
        fem_available = True
        print(f"FEM data loaded: {X_fea.shape} (path={fea_path!r})")
    except FileNotFoundError:
        print("FEM solution not found. Training without FEM comparison.")
        fem_available = False
        u_fea_top_uz = None
        top_uz_denom = 0.0
    
    # Data Container
    training_data = data.get_data()

    # Optional: light supervision anchor on FEM top-patch u_z (depth calibration only).
    # Enabled by setting config.WEIGHTS['patch_uz_sup'] > 0 via env override PINN_W_PATCH_UZ_SUP.
    patch_sup_x = None
    patch_sup_uz = None
    if fem_available and float(config.WEIGHTS.get("patch_uz_sup", 0.0)) > 0.0:
        z = pts_fea[:, 2]
        H_fea = float(getattr(config, "H", np.max(z)))
        top = np.isclose(z, H_fea)
        x = pts_fea[:, 0]
        y = pts_fea[:, 1]
        x0, x1 = map(float, config.LOAD_PATCH_X)
        y0, y1 = map(float, config.LOAD_PATCH_Y)
        patch = top & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if np.any(patch):
            patch_sup_x = torch.tensor(pts_fea[patch], dtype=torch.float32)
            patch_sup_uz = torch.tensor(u_fea_flat[patch, 2], dtype=torch.float32).unsqueeze(1)
            training_data["patch_sup_x"] = patch_sup_x
            training_data["patch_sup_uz"] = patch_sup_uz
            print(
                f"Patch u_z anchor enabled: N={int(patch_sup_x.shape[0])} "
                f"(weight={float(config.WEIGHTS.get('patch_uz_sup', 0.0))})"
            )

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
    best_fem_patch_mae = float("inf")
    best_fem_patch_epoch = None
    best_patch_top_uz_mae_pct = float("inf")
    best_patch_top_uz_epoch = None
    best_top_uz_mae_pct = float("inf")
    best_top_uz_epoch = None
    
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()

        # Optional patch-focus schedule (adjusts soft-mask emphasis mid-training).
        patch_focus_epoch = getattr(config, "PATCH_FOCUS_EPOCH", None)
        if patch_focus_epoch is not None and int(patch_focus_epoch) == int(epoch):
            config.LOAD_MASK_SAMPLING_POWER = float(getattr(config, "PATCH_FOCUS_MASK_SAMPLING_POWER", config.LOAD_MASK_SAMPLING_POWER))
            config.LOAD_MASK_LOSS_POWER = float(getattr(config, "PATCH_FOCUS_MASK_LOSS_POWER", config.LOAD_MASK_LOSS_POWER))
            training_data = data.get_data()  # ensure new sampling takes effect immediately
            print(
                f"Patch-focus enabled at epoch {epoch}: LOAD_MASK_SAMPLING_POWER={config.LOAD_MASK_SAMPLING_POWER}, "
                f"LOAD_MASK_LOSS_POWER={config.LOAD_MASK_LOSS_POWER}"
            )

        # impact_params-style soft->hard (or hard->soft) clamp schedule.
        # We implement this as an exponent ramp on the hard-mask factor:
        #   u <- u * mask**power, power∈[0, HARD_SIDE_BC_POWER]
        # power=0 => effectively soft; power=HARD_SIDE_BC_POWER => fully hard.
        p_start = float(getattr(config, "HARD_SIDE_BC_POWER_START", 0.0))
        p_end = float(getattr(config, "HARD_SIDE_BC_POWER_END", float(getattr(config, "HARD_SIDE_BC_POWER", 1.0))))
        # Backward compatibility: if a user only overrides HARD_SIDE_BC_POWER, use it as the end power.
        if os.getenv("PINN_HARD_SIDE_BC_POWER") is not None and os.getenv("PINN_HARD_SIDE_BC_POWER") != "":
            p_end = float(getattr(config, "HARD_SIDE_BC_POWER", p_end))
        base_power = max(p_start, p_end)
        if config.FORCE_SOFT_SIDE_BC_FROM_START:
            target_power = 0.0
            config.USE_HARD_SIDE_BC = False
            pinn.set_hard_bc(False)
        else:
            sched = str(getattr(config, "HARD_BC_SCHEDULE", "hard_to_soft")).lower().strip()
            denom = float(max(1, int(getattr(config, "HARD_BC_EPOCHS", 1))))
            t = float(epoch) / denom
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            if sched == "soft_to_hard":
                target_power = p_start + t * (p_end - p_start)
            elif sched == "hard_to_soft":
                target_power = p_start + t * (p_end - p_start)
            else:
                # Fallback to legacy behavior: step toggle.
                target_power = p_end if epoch < int(getattr(config, "HARD_BC_EPOCHS", 0)) else p_start

            if target_power <= 0.0:
                if config.USE_HARD_SIDE_BC:
                    config.USE_HARD_SIDE_BC = False
                    pinn.set_hard_bc(False)
            else:
                if not config.USE_HARD_SIDE_BC:
                    config.USE_HARD_SIDE_BC = True
                    pinn.set_hard_bc(True)
        # keep the current power on config for LayerNet.forward()
        config.HARD_SIDE_BC_POWER = float(max(0.0, target_power))
        # Consider the mask "hard" only when the ramp is essentially fully on.
        use_hard_bc = bool(
            config.USE_HARD_SIDE_BC and base_power > 0.0 and (config.HARD_SIDE_BC_POWER >= 0.999 * base_power)
        )
        if log_every > 0 and epoch % log_every == 0:
            if config.USE_HARD_SIDE_BC:
                print(f"Side clamp mask: ON (power={config.HARD_SIDE_BC_POWER:.3f}/{base_power:.3f}, schedule={getattr(config,'HARD_BC_SCHEDULE','')})")
            else:
                print("Side clamp mask: OFF")
        
        if resample_every > 0 and epoch % resample_every == 0 and epoch > 0:
            if bool(getattr(config, "TRAIN_FIXED_PARAMS", False)):
                # Compute residuals for adaptive sampling (best for single-case parity)
                residuals = physics.compute_residuals(pinn, training_data, device)
                training_data = data.get_data(prev_data=training_data, residuals=residuals)
                print(f"  Resampled with residual-based adaptive sampling at epoch {epoch}")
            else:
                # For parametric training, keep broad coverage of the parameter domain.
                training_data = data.get_data()
                print(f"  Resampled uniformly (parametric mode) at epoch {epoch}")
            if patch_sup_x is not None and patch_sup_uz is not None:
                training_data["patch_sup_x"] = patch_sup_x
                training_data["patch_sup_uz"] = patch_sup_uz
            
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
                    v_pinn = pinn(pts_fea_tensor)
                    u_pinn_flat = physics.decode_u(v_pinn, pts_fea_tensor).cpu().numpy()
                     
                    diff = np.abs(u_pinn_flat - u_fea_flat)
                    mae = np.mean(diff)
                    max_err = np.max(diff)
                    if u_fea_abs_max > 0.0:
                        mae_pct = (float(mae) / float(u_fea_abs_max)) * 100.0
                    else:
                        mae_pct = float("nan")

                    # Top surface u_z metric (MAE as % of max |FEA u_z| on the top surface).
                    if u_fea_top_uz is not None and top_uz_denom > 0.0:
                        u_pred_grid = u_pinn_flat.reshape(U_fea.shape)
                        top_abs = np.abs(u_pred_grid[:, :, -1, 2] - u_fea_top_uz)
                        top_uz_mae = float(np.mean(top_abs))
                        top_uz_mae_pct = (top_uz_mae / float(top_uz_denom)) * 100.0
                    else:
                        top_uz_mae_pct = float("nan")

                    # Patch-focused metric (top surface + load patch).
                    z = pts_fea[:, 2]
                    H_fea = float(getattr(config, "H", np.max(z)))
                    top = np.isclose(z, H_fea)
                    x = pts_fea[:, 0]
                    y = pts_fea[:, 1]
                    x0, x1 = map(float, config.LOAD_PATCH_X)
                    y0, y1 = map(float, config.LOAD_PATCH_Y)
                    patch = top & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                    if np.any(patch):
                        patch_mae = float(np.mean(np.abs(u_pinn_flat[patch] - u_fea_flat[patch])))
                        if u_fea_top_uz is not None and top_uz_denom > 0.0:
                            patch_uz_mae = float(np.mean(np.abs(u_pinn_flat[patch, 2] - u_fea_flat[patch, 2])))
                            patch_uz_mae_pct = (patch_uz_mae / float(top_uz_denom)) * 100.0
                        else:
                            patch_uz_mae_pct = float("nan")
                    else:
                        patch_mae = float("nan")
                        patch_uz_mae_pct = float("nan")

                    adam_history['fem_mae'].append(mae)
                    adam_history['fem_max_err'].append(max_err)
                    adam_history['epochs'].append(epoch)

                    if mae < best_fem_mae:
                        best_fem_mae = float(mae)
                        best_fem_epoch = int(epoch)
                        best_path = _tag_path("pinn_model_best_fem.pth")
                        torch.save(pinn.state_dict(), best_path)
                        print(f"  New best FEM MAE: {best_fem_mae:.6f} at epoch {best_fem_epoch} (saved {best_path})")

                    if np.isfinite(patch_mae) and patch_mae < best_fem_patch_mae:
                        best_fem_patch_mae = float(patch_mae)
                        best_fem_patch_epoch = int(epoch)
                        best_patch_path = _tag_path("pinn_model_best_fem_patch.pth")
                        torch.save(pinn.state_dict(), best_patch_path)
                        print(f"  New best FEM PATCH MAE: {best_fem_patch_mae:.6f} at epoch {best_fem_patch_epoch} (saved {best_patch_path})")

                    if np.isfinite(top_uz_mae_pct) and top_uz_mae_pct < best_top_uz_mae_pct:
                        best_top_uz_mae_pct = float(top_uz_mae_pct)
                        best_top_uz_epoch = int(epoch)
                        best_top_path = _tag_path("pinn_model_best_top_uz.pth")
                        torch.save(pinn.state_dict(), best_top_path)
                        print(
                            f"  New best TOP u_z MAE%: {best_top_uz_mae_pct:.2f}% at epoch {best_top_uz_epoch} (saved {best_top_path})"
                        )

                    if np.isfinite(patch_uz_mae_pct) and patch_uz_mae_pct < best_patch_top_uz_mae_pct:
                        best_patch_top_uz_mae_pct = float(patch_uz_mae_pct)
                        best_patch_top_uz_epoch = int(epoch)
                        best_patch_uz_path = _tag_path("pinn_model_best_patch_top_uz.pth")
                        torch.save(pinn.state_dict(), best_patch_uz_path)
                        print(
                            f"  New best PATCH-top u_z MAE%: {best_patch_top_uz_mae_pct:.2f}% at epoch {best_patch_top_uz_epoch} (saved {best_patch_uz_path})"
                        )
                    
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | BC_sides: {losses['bc_sides']:.6f} | "
                      f"Free_top: {losses['free_top']:.6f} | Free_side: {losses.get('free_side', torch.tensor(0.0)).item():.6f} | Free_bot: {losses['free_bot']:.6f} | "
                      f"Load: {losses['load']:.6f} | Energy: {losses['energy']:.6f} | "
                      f"PatchSup: {losses.get('patch_uz_sup', torch.tensor(0.0)).item():.6f} | "
                      f"IntU: {losses.get('interface_u', torch.tensor(0.0)).item():.6f} | IntT: {losses.get('interface_t', torch.tensor(0.0)).item():.6f} | "
                      f"ImpactC: {losses.get('impact_contact', torch.tensor(0.0)).item():.6f} | "
                      f"FricC: {losses.get('friction_coulomb', torch.tensor(0.0)).item():.6f} | "
                      f"FricS: {losses.get('friction_stick', torch.tensor(0.0)).item():.6f} | "
                      f"ImpactInv: {losses.get('impact_invariance', torch.tensor(0.0)).item():.6f} | "
                      f"LR: {current_lr:.2e} | "
                      f"FEM MAE: {mae:.6f} ({mae_pct:.2f}%) | FEM PATCH MAE: {patch_mae:.6f} | TOP u_z MAE%: {top_uz_mae_pct:.2f}% | PATCH-top u_z MAE%: {patch_uz_mae_pct:.2f}% | Time: {step_duration:.4f}s")

                if stop_global_mae_pct is not None or stop_patch_top_uz_mae_pct is not None:
                    ok_global = True if stop_global_mae_pct is None else (np.isfinite(mae_pct) and mae_pct <= stop_global_mae_pct)
                    ok_patch = True if stop_patch_top_uz_mae_pct is None else (np.isfinite(patch_uz_mae_pct) and patch_uz_mae_pct <= stop_patch_top_uz_mae_pct)
                    if ok_global and ok_patch:
                        final_path = _tag_path("pinn_model_target_reached.pth")
                        torch.save(pinn.state_dict(), final_path)
                        print(f"Target reached at epoch {epoch}: saved {final_path}")
                        return pinn
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
        print(f"Best FEM MAE during Adam: {best_fem_mae:.6f} at epoch {best_fem_epoch} ({_tag_path('pinn_model_best_fem.pth')})")
    if fem_available and best_fem_patch_epoch is not None:
        print(f"Best FEM PATCH MAE during Adam: {best_fem_patch_mae:.6f} at epoch {best_fem_patch_epoch} ({_tag_path('pinn_model_best_fem_patch.pth')})")
    
    # L-BFGS Fine-Tuning
    print("Starting L-BFGS Fine-Tuning...")
    # Keep the current hard/soft side-BC setting for L-BFGS unless the user explicitly
    # overrides it via env/config. For many FEM-parity cases, the hard mask is required
    # to preserve boundary shape during second-order optimization.
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
                v_pinn = pinn(pts_fea_tensor)
                u_pinn_flat = physics.decode_u(v_pinn, pts_fea_tensor).cpu().numpy()
                diff = np.abs(u_pinn_flat - u_fea_flat)
                mae = np.mean(diff)
                max_err = np.max(diff)
                if u_fea_abs_max > 0.0:
                    mae_pct = (float(mae) / float(u_fea_abs_max)) * 100.0
                else:
                    mae_pct = float("nan")

                if u_fea_top_uz is not None and top_uz_denom > 0.0:
                    u_pred_grid = u_pinn_flat.reshape(U_fea.shape)
                    top_abs = np.abs(u_pred_grid[:, :, -1, 2] - u_fea_top_uz)
                    top_uz_mae = float(np.mean(top_abs))
                    top_uz_mae_pct = (top_uz_mae / float(top_uz_denom)) * 100.0
                else:
                    top_uz_mae_pct = float("nan")

                z = pts_fea[:, 2]
                H_fea = float(getattr(config, "H", np.max(z)))
                top = np.isclose(z, H_fea)
                x = pts_fea[:, 0]
                y = pts_fea[:, 1]
                x0, x1 = map(float, config.LOAD_PATCH_X)
                y0, y1 = map(float, config.LOAD_PATCH_Y)
                patch = top & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                if np.any(patch):
                    patch_mae = float(np.mean(np.abs(u_pinn_flat[patch] - u_fea_flat[patch])))
                else:
                    patch_mae = float("nan")
                lbfgs_history['fem_mae'].append(mae)
                lbfgs_history['fem_max_err'].append(max_err)
                lbfgs_history['steps'].append(i)

                if mae < best_fem_mae:
                    best_fem_mae = float(mae)
                    best_fem_epoch = f"lbfgs_step_{i}"
                    best_path = _tag_path("pinn_model_best_fem.pth")
                    torch.save(pinn.state_dict(), best_path)
                    print(f"  New best FEM MAE: {best_fem_mae:.6e} at {best_fem_epoch} (saved {best_path})")
                if np.isfinite(patch_mae) and patch_mae < best_fem_patch_mae:
                    best_fem_patch_mae = float(patch_mae)
                    best_fem_patch_epoch = f"lbfgs_step_{i}"
                    best_patch_path = _tag_path("pinn_model_best_fem_patch.pth")
                    torch.save(pinn.state_dict(), best_patch_path)
                    print(f"  New best FEM PATCH MAE: {best_fem_patch_mae:.6e} at {best_fem_patch_epoch} (saved {best_patch_path})")
                if np.isfinite(top_uz_mae_pct) and top_uz_mae_pct < best_top_uz_mae_pct:
                    best_top_uz_mae_pct = float(top_uz_mae_pct)
                    best_top_uz_epoch = f"lbfgs_step_{i}"
                    best_top_path = _tag_path("pinn_model_best_top_uz.pth")
                    torch.save(pinn.state_dict(), best_top_path)
                    print(f"  New best TOP u_z MAE%: {best_top_uz_mae_pct:.2f}% at {best_top_uz_epoch} (saved {best_top_path})")
            print(f"L-BFGS Step {i}: Total Loss: {loss_val.item():.6e} | PDE: {losses['pde'].item():.6e} | "
                  f"BC_sides: {losses['bc_sides'].item():.6e} | Free_top: {losses['free_top'].item():.6e} | "
                  f"Free_side: {losses.get('free_side', torch.tensor(0.0)).item():.6e} | Free_bot: {losses['free_bot'].item():.6e} | Load: {losses['load'].item():.6e} | "
                  f"Energy: {losses['energy'].item():.6e} | "
                  f"IntU: {losses.get('interface_u', torch.tensor(0.0)).item():.6e} | IntT: {losses.get('interface_t', torch.tensor(0.0)).item():.6e} | "
                  f"ImpactC: {losses.get('impact_contact', torch.tensor(0.0)).item():.6e} | "
                  f"FricC: {losses.get('friction_coulomb', torch.tensor(0.0)).item():.6e} | "
                  f"FricS: {losses.get('friction_stick', torch.tensor(0.0)).item():.6e} | "
                  f"ImpactInv: {losses.get('impact_invariance', torch.tensor(0.0)).item():.6e} | "
                  f"FEM MAE: {mae:.6e} ({mae_pct:.2f}%) | FEM PATCH MAE: {patch_mae:.6e} | TOP u_z MAE%: {top_uz_mae_pct:.2f}% | Time: {step_end - step_start:.4f}s")

            if stop_global_mae_pct is not None or stop_patch_top_uz_mae_pct is not None:
                ok_global = True if stop_global_mae_pct is None else (np.isfinite(mae_pct) and mae_pct <= stop_global_mae_pct)
                ok_patch = True if stop_patch_top_uz_mae_pct is None else (np.isfinite(top_uz_mae_pct) and top_uz_mae_pct <= stop_patch_top_uz_mae_pct)
                if ok_global and ok_patch:
                    final_path = _tag_path("pinn_model_target_reached.pth")
                    torch.save(pinn.state_dict(), final_path)
                    print(f"Target reached at LBFGS step {i}: saved {final_path}")
                    return pinn
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
        torch.save(pinn.state_dict(), _tag_path("pinn_model.pth"))
            
    # Save Model and Loss Histories
    torch.save(pinn.state_dict(), _tag_path("pinn_model.pth"))
    loss_history = {'adam': adam_history, 'lbfgs': lbfgs_history}
    np.save(_tag_path("loss_history.npy"), loss_history)
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
