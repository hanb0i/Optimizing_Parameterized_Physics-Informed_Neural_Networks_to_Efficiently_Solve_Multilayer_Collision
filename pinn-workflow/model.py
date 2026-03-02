from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import pinn_config as config


class LayerNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_units=64, activation=nn.Tanh()):
        super().__init__()
        layers = []
        n_layers = int(getattr(config, "NUM_LAYERS", 2))
        if n_layers < 1:
            raise ValueError(f"config.NUM_LAYERS must be >= 1, got {n_layers}")

        # Feature layout (for NUM_LAYERS = L):
        #   [x, y, z_hat,
        #    E1_norm, t1_scaled, ..., EL_norm, tL_scaled,
        #    r_norm, mu_norm, v0_norm,
        #    inv1, inv2, inv3]
        current_dim = 3 + 2 * n_layers + 3 + 3
        
        layers.append(nn.Linear(current_dim, hidden_units))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)
            
        out_dim = 9 if bool(getattr(config, "USE_MIXED_FORMULATION", False)) else 3
        layers.append(nn.Linear(hidden_units, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Input layout:
        #   [x, y, z, E1, t1, ..., EL, tL, r, mu, v0]
        n_layers = int(getattr(config, "NUM_LAYERS", 2))
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]

        # Layer parameters start at column 3: (E1,t1,...,EL,tL)
        base = 3
        e_cols = [base + 2 * i for i in range(n_layers)]
        t_cols = [base + 2 * i + 1 for i in range(n_layers)]

        Es = [x[:, c : c + 1] for c in e_cols]
        Ts = [torch.clamp(x[:, c : c + 1], min=1e-4) for c in t_cols]
        t_total = torch.clamp(sum(Ts), min=1e-4)

        r_col = base + 2 * n_layers
        mu_col = r_col + 1
        v0_col = r_col + 2
        r_param = x[:, r_col : r_col + 1]
        mu_param = x[:, mu_col : mu_col + 1]
        v0_param = x[:, v0_col : v0_col + 1]
        
        e_min, e_max = config.E_RANGE
        e_span = float(e_max - e_min) if float(e_max - e_min) != 0.0 else 1.0
        E_norm = [(e - float(e_min)) / e_span for e in Es]

        # Thicknesses are layer-wise and can be smaller than the total thickness lower bound;
        # scale by the total-thickness upper bound for a stable [0,~1] range.
        _, t_max = config.THICKNESS_RANGE
        t_scale = float(t_max) if float(t_max) > 0.0 else 1.0
        T_scaled = [t / t_scale for t in Ts]

        r_min, r_max = config.RESTITUTION_RANGE
        r_span = float(r_max - r_min) if float(r_max - r_min) != 0.0 else 1.0
        r_norm = (r_param - float(r_min)) / r_span

        mu_min, mu_max = config.FRICTION_RANGE
        mu_span = float(mu_max - mu_min) if float(mu_max - mu_min) != 0.0 else 1.0
        mu_norm = (mu_param - float(mu_min)) / mu_span

        v0_min, v0_max = config.IMPACT_VELOCITY_RANGE
        v0_span = float(v0_max - v0_min) if float(v0_max - v0_min) != 0.0 else 1.0
        v0_norm = (v0_param - float(v0_min)) / v0_span

        z_hat = z_coord / t_total
        
        h_ref = getattr(config, 'H', 0.1)
        ratio = float(h_ref) / t_total
        feat_inv1 = ratio
        feat_inv2 = ratio ** 2
        feat_inv3 = ratio ** 3
        extra_feats = torch.cat([feat_inv1, feat_inv2, feat_inv3], dim=1)
        
        feats = [x_coord, y_coord, z_hat]
        for e_n, t_s in zip(E_norm, T_scaled):
            feats.extend([e_n, t_s])
        feats.extend([r_norm, mu_norm, v0_norm, extra_feats])
        x_scaled = torch.cat(feats, dim=1)
        y_raw = self.net(x_scaled)

        # impact_params-style hard clamp-by-construction (box mode only):
        # multiply the displacement by a smooth mask that is 0 on x/y side faces.
        if bool(getattr(config, "USE_HARD_SIDE_BC", False)):
            if str(getattr(config, "GEOMETRY_MODE", "box")).lower() == "box" and bool(getattr(config, "BOX_CLAMP_SIDES", False)):
                Lx = float(getattr(config, "Lx", 1.0))
                Ly = float(getattr(config, "Ly", 1.0))
                x0 = torch.clamp(x_coord, 0.0, Lx)
                y0 = torch.clamp(y_coord, 0.0, Ly)
                bx = (x0 * (Lx - x0)) / max(1e-12, 0.25 * Lx * Lx)
                by = (y0 * (Ly - y0)) / max(1e-12, 0.25 * Ly * Ly)
                mask = bx * by
                power = float(getattr(config, "HARD_SIDE_BC_POWER", 1.0))
                if power <= 0.0:
                    mask_eff = torch.ones_like(mask)
                else:
                    mask_eff = mask.clamp_min(1e-12) ** power
                if y_raw.shape[1] == 3:
                    y_raw = y_raw * mask_eff
                else:
                    u_part = y_raw[:, 0:3] * mask_eff
                    y_raw = torch.cat([u_part, y_raw[:, 3:]], dim=1)

        return y_raw

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        n_layers = int(getattr(config, "NUM_LAYERS", 2))
        self.layers = nn.ModuleList([LayerNet(hidden_layers=config.LAYERS, hidden_units=config.NEURONS) for _ in range(n_layers)])

    @staticmethod
    def _interfaces(x: torch.Tensor) -> list[torch.Tensor]:
        n_layers = int(getattr(config, "NUM_LAYERS", 2))
        if n_layers <= 1:
            return []
        base = 3
        t_cols = [base + 2 * i + 1 for i in range(n_layers)]
        Ts = [torch.clamp(x[:, c : c + 1], min=1e-4) for c in t_cols]
        cum = torch.cumsum(torch.cat(Ts, dim=1), dim=1)
        # interfaces are at cumulative thicknesses excluding the top surface
        return [cum[:, i : i + 1] for i in range(n_layers - 1)]

    @classmethod
    def _layer_idx(cls, x: torch.Tensor) -> torch.Tensor:
        n_layers = int(getattr(config, "NUM_LAYERS", 2))
        if n_layers <= 1:
            return torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        z = x[:, 2:3]
        interfaces = cls._interfaces(x)
        # Start as last layer, then overwrite earlier ones.
        out = torch.full((x.shape[0],), n_layers - 1, device=x.device, dtype=torch.long)
        for li, z_intf in enumerate(interfaces):
            m = z < z_intf
            out[m[:, 0]] = li
        return out
        
    def forward(self, x: torch.Tensor, layer_idx: int | None = None) -> torch.Tensor:
        if layer_idx is not None:
            return self.layers[int(layer_idx)](x)

        gating = str(getattr(config, "LAYER_GATING", "hard")).lower().strip()
        if gating == "hard":
            idx = self._layer_idx(x)
            out_dim = 9 if bool(getattr(config, "USE_MIXED_FORMULATION", False)) else 3
            out = torch.empty((x.shape[0], out_dim), device=x.device, dtype=x.dtype)
            for li in range(len(self.layers)):
                m = idx == li
                if torch.any(m):
                    out[m] = self.layers[li](x[m])
            return out

        # Soft gating: differentiable stick-breaking blend across interfaces.
        # For L layers and interfaces s_k = sigmoid(beta*(z - z_k)):
        #   w0 = 1 - s0
        #   wi = s_{i-1} * (1 - s_i) for i=1..L-2
        #   w_{L-1} = s_{L-2}
        n_layers = len(self.layers)
        if n_layers <= 1:
            return self.layers[0](x)
        z = x[:, 2:3]
        interfaces = self._interfaces(x)
        beta = float(getattr(config, "LAYER_GATE_BETA", 200.0))
        s = [torch.sigmoid(beta * (z - zi)) for zi in interfaces]  # length L-1

        w = []
        w.append(1.0 - s[0])
        for i in range(1, n_layers - 1):
            w.append(s[i - 1] * (1.0 - s[i]))
        w.append(s[-1])
        ws = torch.stack(w, dim=0).sum(dim=0).clamp_min(1e-8)
        w = [wi / ws for wi in w]

        ys = [layer(x) for layer in self.layers]
        out = w[0] * ys[0]
        for wi, yi in zip(w[1:], ys[1:]):
            out = out + wi * yi
        return out

    def predict_all(self, x):
        return self.forward(x)

    def set_hard_bc(self, use_hard):
        config.USE_HARD_SIDE_BC = bool(use_hard)
