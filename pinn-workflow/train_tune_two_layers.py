from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config  # noqa: E402
import model  # noqa: E402
import physics  # noqa: E402
import fem_solver  # noqa: E402


@dataclass(frozen=True)
class Case2L:
    H: float
    t1: float
    t2: float
    E1: float
    E2: float

    @property
    def frac(self) -> float:
        return float(self.t1 / max(self.H, 1e-12))


def _select_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / max(den, eps)


def _case_key(case: Case2L, *, ne: int, use_soft_mask: bool) -> str:
    return (
        f"2l_ne{int(ne)}_soft{int(bool(use_soft_mask))}"
        f"_H{case.H:.6f}_t1{case.t1:.6f}_t2{case.t2:.6f}_E1{case.E1:.6f}_E2{case.E2:.6f}"
    )


def _solve_fea_case(
    case: Case2L,
    *,
    ne: int,
    nu: float,
    p0: float,
    use_soft_mask: bool,
    cache_dir: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, _case_key(case, ne=ne, use_soft_mask=use_soft_mask) + ".npz")
        if os.path.exists(cache_path):
            d = np.load(cache_path, allow_pickle=False)
            return d["x"], d["y"], d["z"], d["u"]

    x0, x1 = map(float, getattr(config, "LOAD_PATCH_X", (1.0 / 3.0, 2.0 / 3.0)))
    y0, y1 = map(float, getattr(config, "LOAD_PATCH_Y", (1.0 / 3.0, 2.0 / 3.0)))
    Lx = float(getattr(config, "Lx", 1.0))
    Ly = float(getattr(config, "Ly", 1.0))

    cfg = {
        "geometry": {"Lx": Lx, "Ly": Ly, "H": float(case.H)},
        "mesh": {"ne_x": int(ne), "ne_y": int(ne), "ne_z": int(ne)},
        "layers": [
            {"t": float(case.t1), "E": float(case.E1), "nu": float(nu)},
            {"t": float(case.t2), "E": float(case.E2), "nu": float(nu)},
        ],
        "load_patch": {
            "pressure": float(p0),
            "x_start": x0 / Lx,
            "x_end": x1 / Lx,
            "y_start": y0 / Ly,
            "y_end": y1 / Ly,
        },
        "use_soft_mask": bool(use_soft_mask),
    }
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(cfg)
    x_arr = np.asarray(x_nodes, dtype=np.float32)
    y_arr = np.asarray(y_nodes, dtype=np.float32)
    z_arr = np.asarray(z_nodes, dtype=np.float32)
    # NumPy 2.x may require a copy here depending on the underlying object type.
    u_arr = np.asarray(u_grid, dtype=np.float32)

    if cache_dir:
        np.savez_compressed(cache_path, x=x_arr, y=y_arr, z=z_arr, u=u_arr)
    return x_arr, y_arr, z_arr, u_arr


def _build_case_dataset(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    z_nodes: np.ndarray,
    u_grid: np.ndarray,
    case: Case2L,
    *,
    impact_seed: int,
    randomize_impact_params: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32, copy=False)

    if randomize_impact_params:
        rng = np.random.default_rng(int(impact_seed) & 0xFFFFFFFF)
        r_min, r_max = map(float, getattr(config, "RESTITUTION_RANGE", (0.5, 0.5)))
        mu_min, mu_max = map(float, getattr(config, "FRICTION_RANGE", (0.3, 0.3)))
        v0_min, v0_max = map(float, getattr(config, "IMPACT_VELOCITY_RANGE", (1.0, 1.0)))
        # Use one (r,mu,v0) triplet per FEA case (not per point) to keep supervision stable
        # across rounds while still teaching invariance over the parameter ranges.
        r0 = float(rng.uniform(r_min, r_max))
        mu0 = float(rng.uniform(mu_min, mu_max))
        v00 = float(rng.uniform(v0_min, v0_max))
        r = np.full((pts.shape[0], 1), r0, dtype=np.float32)
        mu = np.full((pts.shape[0], 1), mu0, dtype=np.float32)
        v0 = np.full((pts.shape[0], 1), v00, dtype=np.float32)
    else:
        r = np.full((pts.shape[0], 1), float(getattr(config, "RESTITUTION_REF", 0.5)), dtype=np.float32)
        mu = np.full((pts.shape[0], 1), float(getattr(config, "FRICTION_REF", 0.3)), dtype=np.float32)
        v0 = np.full((pts.shape[0], 1), float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)), dtype=np.float32)

    params = np.array([case.E1, case.t1, case.E2, case.t2], dtype=np.float32)[None, :]
    params_rep = np.repeat(params, pts.shape[0], axis=0)
    x_in = np.concatenate([pts, params_rep, r, mu, v0], axis=1).astype(np.float32, copy=False)

    u_out = np.asarray(u_grid, dtype=np.float32).reshape(-1, 3)
    return x_in, u_out


def _sample_cases(
    rng: np.random.Generator,
    n: int,
    *,
    include_extremes: bool,
) -> list[Case2L]:
    e_min, e_max = map(float, getattr(config, "E_RANGE", (1.0, 10.0)))
    t_min, t_max = map(float, getattr(config, "THICKNESS_RANGE", (float(getattr(config, "H", 0.1)), float(getattr(config, "H", 0.1)))))
    frac_min = float(getattr(config, "LAYER_THICKNESS_FRACTION_MIN", 0.05))
    frac_min = max(1e-4, min(frac_min, 0.49))

    out: list[Case2L] = []
    if include_extremes:
        for H in (t_min, t_max):
            for f in (frac_min, 0.5, 1.0 - frac_min):
                for E1 in (e_min, e_max):
                    for E2 in (e_min, e_max):
                        t1 = float(H * f)
                        t2 = float(H - t1)
                        out.append(Case2L(H=float(H), t1=t1, t2=t2, E1=float(E1), E2=float(E2)))

    while len(out) < int(n):
        H = float(rng.uniform(t_min, t_max))
        f = float(rng.uniform(frac_min, 1.0 - frac_min))
        t1 = float(H * f)
        t2 = float(H - t1)
        E1 = float(rng.uniform(e_min, e_max))
        E2 = float(rng.uniform(e_min, e_max))
        out.append(Case2L(H=H, t1=t1, t2=t2, E1=E1, E2=E2))
    return out[: int(n)]

def _load_cases_json(path: str) -> list[Case2L]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("cases json must be a list of objects")
    out: list[Case2L] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"case[{i}] must be an object")
        H = float(item.get("H"))
        if "t1" in item and "t2" in item:
            t1 = float(item["t1"])
            t2 = float(item["t2"])
        elif "frac" in item:
            f = float(item["frac"])
            t1 = float(H * f)
            t2 = float(H - t1)
        else:
            raise ValueError(f"case[{i}] must include (t1,t2) or frac")
        E1 = float(item.get("E1"))
        E2 = float(item.get("E2"))
        out.append(Case2L(H=H, t1=t1, t2=t2, E1=E1, E2=E2))
    return out


def _eval_case(
    pinn: torch.nn.Module,
    device: torch.device,
    case: Case2L,
    *,
    ne: int,
    nu: float,
    p0: float,
    use_soft_mask: bool,
    cache_dir: str | None,
) -> tuple[float, float]:
    x_nodes, y_nodes, z_nodes, u_fea = _solve_fea_case(
        case,
        ne=ne,
        nu=nu,
        p0=p0,
        use_soft_mask=use_soft_mask,
        cache_dir=cache_dir,
    )
    X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32, copy=False)
    params = np.array([case.E1, case.t1, case.E2, case.t2], dtype=np.float32)[None, :]
    params_rep = np.repeat(params, pts.shape[0], axis=0)
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    r = np.full((pts.shape[0], 1), r_ref, dtype=np.float32)
    mu = np.full((pts.shape[0], 1), mu_ref, dtype=np.float32)
    v0 = np.full((pts.shape[0], 1), v0_ref, dtype=np.float32)
    x_in = np.concatenate([pts, params_rep, r, mu, v0], axis=1).astype(np.float32, copy=False)

    with torch.no_grad():
        x_t = torch.tensor(x_in, dtype=torch.float32, device=device)
        v = pinn(x_t)
        u_pred = physics.decode_u(v, x_t).cpu().numpy().astype(np.float32).reshape(u_fea.shape)

    u_true = np.asarray(u_fea, dtype=np.float32)
    uz_fea_top = u_true[:, :, -1, 2]
    uz_pinn_top = u_pred[:, :, -1, 2]
    peak_fea = float(np.min(uz_fea_top))
    peak_pinn = float(np.min(uz_pinn_top))
    peak_rel = abs(peak_pinn - peak_fea) / max(abs(peak_fea), 1e-12)
    l2_rel = _relative_l2(uz_pinn_top, uz_fea_top)
    return peak_rel, l2_rel


def main() -> None:
    ap = argparse.ArgumentParser(description="Train+tune a 2-layer surrogate to match layered FEA (<5% top u_z errors).")
    ap.add_argument("--device", default=None, help="cpu|cuda|mps (auto if omitted).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ne", type=int, default=12, help="FEA mesh elements per axis (ne_x=ne_y=ne_z).")
    ap.add_argument("--nu", type=float, default=float(getattr(config, "NU_FIXED", 0.3)))
    ap.add_argument("--p0", type=float, default=float(getattr(config, "p0", 1.0)))
    ap.add_argument("--use_soft_mask", type=int, default=int(getattr(config, "USE_SOFT_LOAD_MASK", True)))
    ap.add_argument("--cache_dir", default="pinn-workflow/fea_cache", help="FEA solution cache directory (empty disables).")

    ap.add_argument("--train_cases", type=int, default=60)
    ap.add_argument("--val_cases", type=int, default=20)
    ap.add_argument("--train_cases_json", default=None, help="Optional JSON list of training cases.")
    ap.add_argument("--val_cases_json", default=None, help="Optional JSON list of validation cases.")
    ap.add_argument("--include_extremes", type=int, default=1, help="Include min/max cases in the training set.")
    ap.add_argument("--randomize_impact_params", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_decay_per_round", type=float, default=0.7)
    ap.add_argument("--weight_top", type=float, default=3.0)
    ap.add_argument("--weight_patch", type=float, default=6.0)
    ap.add_argument("--weight_uz", type=float, default=3.0)

    ap.add_argument("--target_pct", type=float, default=5.0)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--add_worst", type=int, default=5, help="How many worst validation cases to add each round.")
    ap.add_argument("--save", default="pinn_model_two_layers_tuned.pth")
    args = ap.parse_args()

    if int(getattr(config, "NUM_LAYERS", 2)) != 2:
        raise ValueError(f"Expected config.NUM_LAYERS=2 (got {getattr(config, 'NUM_LAYERS', None)}).")
    if int(getattr(config, "LAYERS", 4)) != 4 or int(getattr(config, "NEURONS", 64)) != 64:
        print(f"Warning: config LAYERS/NEURONS is {getattr(config, 'LAYERS', None)}/{getattr(config, 'NEURONS', None)}; expected 4/64.")

    device = _select_device(args.device)
    rng = np.random.default_rng(int(args.seed))
    use_soft_mask = bool(int(args.use_soft_mask))
    cache_dir = str(args.cache_dir).strip() or None

    pinn = model.MultiLayerPINN().to(device)
    pinn.train()

    thresh = float(args.target_pct) / 100.0
    if args.train_cases_json:
        train_set = _load_cases_json(str(args.train_cases_json))
    else:
        train_set = _sample_cases(rng, int(args.train_cases), include_extremes=bool(int(args.include_extremes)))
    if args.val_cases_json:
        val_set = _load_cases_json(str(args.val_cases_json))
    else:
        val_set = _sample_cases(rng, int(args.val_cases), include_extremes=False)

    case_data_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def build_dataset(cases: list[Case2L]) -> tuple[torch.Tensor, torch.Tensor]:
        x_list: list[torch.Tensor] = []
        u_list: list[torch.Tensor] = []
        for c in cases:
            key = _case_key(c, ne=int(args.ne), use_soft_mask=use_soft_mask)
            cached = case_data_cache.get(key)
            if cached is None:
                x_nodes, y_nodes, z_nodes, u_grid = _solve_fea_case(
                    c,
                    ne=int(args.ne),
                    nu=float(args.nu),
                    p0=float(args.p0),
                    use_soft_mask=use_soft_mask,
                    cache_dir=cache_dir,
                )
                # Stable seed per case (independent of Python's randomized hash()).
                seed = int.from_bytes(key.encode("utf-8"), "little", signed=False) % (2**32)
                x_in, u_out = _build_case_dataset(
                    x_nodes,
                    y_nodes,
                    z_nodes,
                    u_grid,
                    c,
                    impact_seed=seed,
                    randomize_impact_params=bool(int(args.randomize_impact_params)),
                )
                cached = (torch.tensor(x_in, dtype=torch.float32), torch.tensor(u_out, dtype=torch.float32))
                case_data_cache[key] = cached
            x_list.append(cached[0])
            u_list.append(cached[1])
        if not x_list:
            return torch.zeros((0, 10), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.float32)
        return torch.cat(x_list, dim=0), torch.cat(u_list, dim=0)

    opt = torch.optim.AdamW(pinn.parameters(), lr=float(args.lr))
    best_score = float("inf")
    best_sd: dict[str, torch.Tensor] | None = None

    for round_idx in range(int(args.max_rounds)):
        print(f"\n=== Round {round_idx+1}/{int(args.max_rounds)} ===")
        t0 = time.time()
        x_train, u_train = build_dataset(train_set)
        ds = TensorDataset(x_train, u_train)
        dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
        print(f"Dataset: {len(ds)} points from {len(train_set)} cases (build {time.time()-t0:.1f}s)")

        # Decay learning rate each round for stability as the training set grows.
        if round_idx > 0:
            decay = float(args.lr_decay_per_round)
            if decay > 0.0 and decay < 1.0:
                for pg in opt.param_groups:
                    pg["lr"] = float(pg["lr"]) * decay
        current_lr = float(opt.param_groups[0]["lr"])
        print(f"Optimizer lr={current_lr:.3e}")

        x0, x1 = map(float, getattr(config, "LOAD_PATCH_X", (1.0 / 3.0, 2.0 / 3.0)))
        y0, y1 = map(float, getattr(config, "LOAD_PATCH_Y", (1.0 / 3.0, 2.0 / 3.0)))
        w_top = float(args.weight_top)
        w_patch = float(args.weight_patch)
        w_uz = float(args.weight_uz)

        for epoch in range(int(args.epochs)):
            pinn.train()
            loss_sum = 0.0
            n_seen = 0
            for xb, ub in dl:
                xb = xb.to(device)
                ub = ub.to(device)
                opt.zero_grad(set_to_none=True)
                v = pinn(xb)
                u_pred = physics.decode_u(v, xb)
                err = u_pred - ub

                # Sample weights: emphasize top surface and the load patch on top.
                t_total = torch.clamp(xb[:, 4:5] + xb[:, 6:7], min=1e-8)
                top = torch.isclose(xb[:, 2:3], t_total, rtol=0.0, atol=1e-7)
                in_patch = (xb[:, 0:1] >= x0) & (xb[:, 0:1] <= x1) & (xb[:, 1:2] >= y0) & (xb[:, 1:2] <= y1)
                patch_top = top & in_patch
                w = torch.ones_like(t_total)
                if w_top > 0.0:
                    w = w + w_top * top.to(dtype=xb.dtype)
                if w_patch > 0.0:
                    w = w + w_patch * patch_top.to(dtype=xb.dtype)

                se = (err[:, 0:1] ** 2) + (err[:, 1:2] ** 2) + (w_uz * (err[:, 2:3] ** 2))
                loss = torch.mean(w * se)
                loss.backward()
                opt.step()

                loss_sum += float(loss.item()) * int(xb.shape[0])
                n_seen += int(xb.shape[0])

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"epoch {epoch+1:5d}/{int(args.epochs)} | loss {loss_sum/max(1,n_seen):.6e}")

        # Validation against fresh random cases (top-surface u_z metrics).
        pinn.eval()
        worst: list[tuple[float, float, float, Case2L]] = []
        peak_ok = True
        l2_ok = True
        for i, c in enumerate(val_set):
            peak_rel, l2_rel = _eval_case(
                pinn,
                device,
                c,
                ne=int(args.ne),
                nu=float(args.nu),
                p0=float(args.p0),
                use_soft_mask=use_soft_mask,
                cache_dir=cache_dir,
            )
            worst.append((max(peak_rel, l2_rel), peak_rel, l2_rel, c))
            peak_ok = peak_ok and (peak_rel <= thresh)
            l2_ok = l2_ok and (l2_rel <= thresh)
            print(
                f"val{i:02d}: H={c.H:.4f} t1/T={c.frac:.2f} E1={c.E1:.2f} E2={c.E2:.2f} | "
                f"peak_rel={peak_rel*100:.2f}% l2_rel={l2_rel*100:.2f}%"
            )

        worst.sort(key=lambda t: t[0], reverse=True)
        max_peak = max(w[1] for w in worst) if worst else float("nan")
        max_l2 = max(w[2] for w in worst) if worst else float("nan")
        print(
            f"\nValidation worst peak_rel={max_peak*100:.2f}% l2_rel={max_l2*100:.2f}% "
            f"(target {args.target_pct:.2f}%)"
        )

        score = max(max_peak, max_l2)
        if score < best_score:
            best_score = float(score)
            best_sd = {k: v.detach().cpu().clone() for k, v in pinn.state_dict().items()}
            torch.save(pinn.state_dict(), str(args.save))
            print(f"New best: saved {args.save} (worst={best_score*100:.2f}%)")
        else:
            # Revert to the best-seen checkpoint to prevent drift.
            if best_sd is not None:
                pinn.load_state_dict(best_sd, strict=False)
                pinn.to(device)
                print(f"Reverted to best checkpoint (worst={best_score*100:.2f}%)")

        if peak_ok and l2_ok:
            print(f"Target reached: {args.target_pct:.2f}% on all {len(val_set)} validation cases.")
            return

        # Active learning: add the worst failing offenders to training set.
        add_n = max(0, int(args.add_worst))
        added = 0
        existing = {_case_key(c, ne=int(args.ne), use_soft_mask=use_soft_mask) for c in train_set}
        for _, peak_rel, l2_rel, c in worst:
            if added >= add_n:
                break
            if max(peak_rel, l2_rel) <= thresh:
                continue
            k = _case_key(c, ne=int(args.ne), use_soft_mask=use_soft_mask)
            if k in existing:
                continue
            existing.add(k)
            train_set.append(c)
            added += 1
            print(
                f"Added hard case: H={c.H:.4f} t1/T={c.frac:.2f} E1={c.E1:.2f} E2={c.E2:.2f} "
                f"(peak_rel={peak_rel*100:.2f}% l2_rel={l2_rel*100:.2f}%)"
            )

        # Keep the on-disk checkpoint as the best-seen model.
        print(f"Round complete: added {added} cases (best worst={best_score*100:.2f}%)")

    # Ensure the saved checkpoint is the best one we found.
    if best_sd is not None:
        pinn.load_state_dict(best_sd, strict=False)
        torch.save(pinn.state_dict(), str(args.save))
    print(f"Max rounds reached: best saved to {args.save} (best worst={best_score*100:.2f}%)")


if __name__ == "__main__":
    main()
