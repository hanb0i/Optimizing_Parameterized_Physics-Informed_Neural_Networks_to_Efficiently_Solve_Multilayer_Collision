import torch
import torch.nn as nn
import numpy as np
import pinn_config as config


class LayerNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_units=64, activation=nn.Tanh()):
        super().__init__()
        layers = []
        # Feature layout:
        #   [x, y, z_hat,
        #    E1_norm, t1_scaled, E2_norm, t2_scaled, E3_norm, t3_scaled,
        #    r_norm, mu_norm, v0_norm,
        #    inv1, inv2, inv3]
        current_dim = 15
        
        layers.append(nn.Linear(current_dim, hidden_units))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)
            
        layers.append(nn.Linear(hidden_units, 3))
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Input layout:
        #   [x, y, z, E1, t1, E2, t2, E3, t3, r, mu, v0]
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        e1 = x[:, 3:4]
        t1 = x[:, 4:5]
        e2 = x[:, 5:6]
        t2 = x[:, 6:7]
        e3 = x[:, 7:8]
        t3 = x[:, 8:9]
        r_param = x[:, 9:10]
        mu_param = x[:, 10:11]
        v0_param = x[:, 11:12]

        t1 = torch.clamp(t1, min=1e-4)
        t2 = torch.clamp(t2, min=1e-4)
        t3 = torch.clamp(t3, min=1e-4)
        t_total = torch.clamp(t1 + t2 + t3, min=1e-4)
        
        e_min, e_max = config.E_RANGE
        e_span = float(e_max - e_min) if float(e_max - e_min) != 0.0 else 1.0
        e1_norm = (e1 - float(e_min)) / e_span
        e2_norm = (e2 - float(e_min)) / e_span
        e3_norm = (e3 - float(e_min)) / e_span

        # Thicknesses are layer-wise and can be smaller than the total thickness lower bound;
        # scale by the total-thickness upper bound for a stable [0,~1] range.
        _, t_max = config.THICKNESS_RANGE
        t_scale = float(t_max) if float(t_max) > 0.0 else 1.0
        t1_scaled = t1 / t_scale
        t2_scaled = t2 / t_scale
        t3_scaled = t3 / t_scale

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
        
        x_scaled = torch.cat(
            [
                x_coord,
                y_coord,
                z_hat,
                e1_norm,
                t1_scaled,
                e2_norm,
                t2_scaled,
                e3_norm,
                t3_scaled,
                r_norm,
                mu_norm,
                v0_norm,
                extra_feats,
            ],
            dim=1,
        )
        u_raw = self.net(x_scaled)

        return u_raw

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LayerNet(hidden_layers=config.LAYERS, hidden_units=config.NEURONS),
                LayerNet(hidden_layers=config.LAYERS, hidden_units=config.NEURONS),
                LayerNet(hidden_layers=config.LAYERS, hidden_units=config.NEURONS),
            ]
        )

    @staticmethod
    def _interfaces(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t1 = torch.clamp(x[:, 4:5], min=1e-4)
        t2 = torch.clamp(x[:, 6:7], min=1e-4)
        z1 = t1
        z2 = t1 + t2
        return z1, z2

    @classmethod
    def _layer_idx(cls, x: torch.Tensor) -> torch.Tensor:
        z = x[:, 2:3]
        z1, z2 = cls._interfaces(x)
        idx0 = z < z1
        idx1 = (z >= z1) & (z < z2)
        # default to layer 2
        out = torch.full((x.shape[0],), 2, device=x.device, dtype=torch.long)
        out[idx0[:, 0]] = 0
        out[idx1[:, 0]] = 1
        return out
        
    def forward(self, x: torch.Tensor, layer_idx: int | None = None) -> torch.Tensor:
        if layer_idx is not None:
            return self.layers[int(layer_idx)](x)

        idx = self._layer_idx(x)
        out = torch.empty((x.shape[0], 3), device=x.device, dtype=x.dtype)
        for li in (0, 1, 2):
            m = idx == li
            if torch.any(m):
                out[m] = self.layers[li](x[m])
        return out

    def predict_all(self, x):
        return self.forward(x)

    def set_hard_bc(self, use_hard):
        # Hard BC masking is intentionally disabled for the 3-layer laminate model.
        _ = use_hard
