import torch
import torch.nn as nn
import numpy as np
import pinn_config as config


class LayerNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_units=64, activation=nn.Tanh()):
        super().__init__()
        layers = []
        # Input: x, y, z_hat + 5 normalized params (E, t, r, mu, v0) + 3 physics features
        current_dim = 8 + 3
        
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
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        e_param = x[:, 3:4]
        t_param = x[:, 4:5]
        r_param = x[:, 5:6]
        mu_param = x[:, 6:7]
        v0_param = x[:, 7:8]
        
        e_min, e_max = config.E_RANGE
        e_span = (e_max - e_min) if (e_max - e_min) != 0 else 1.0
        e_norm = (e_param - e_min) / e_span

        t_min, t_max = config.THICKNESS_RANGE
        t_span = (t_max - t_min) if (t_max - t_min) != 0 else 1.0
        t_norm = (t_param - t_min) / t_span

        r_min, r_max = config.RESTITUTION_RANGE
        r_span = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
        r_norm = (r_param - r_min) / r_span

        mu_min, mu_max = config.FRICTION_RANGE
        mu_span = (mu_max - mu_min) if (mu_max - mu_min) != 0 else 1.0
        mu_norm = (mu_param - mu_min) / mu_span

        v0_min, v0_max = config.IMPACT_VELOCITY_RANGE
        v0_span = (v0_max - v0_min) if (v0_max - v0_min) != 0 else 1.0
        v0_norm = (v0_param - v0_min) / v0_span

        t_safe = torch.clamp(t_param, min=1e-6)
        z_hat = z_coord / t_safe
        
        h_ref = getattr(config, 'H', 0.1)
        ratio = h_ref / t_safe 
        feat_inv1 = ratio
        feat_inv2 = ratio ** 2
        feat_inv3 = ratio ** 3
        extra_feats = torch.cat([feat_inv1, feat_inv2, feat_inv3], dim=1)
        
        x_scaled = torch.cat([x_coord, y_coord, z_hat, e_norm, t_norm, r_norm, mu_norm, v0_norm, extra_feats], dim=1)
        u_raw = self.net(x_scaled)
        
        if config.USE_HARD_SIDE_BC:
            x_c = x[:, 0:1]
            y_c = x[:, 1:2]
            mask = x_c * (1.0 - x_c) * y_c * (1.0 - y_c) * 16.0
            u_raw = u_raw * mask
        
        return u_raw

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LayerNet(
            hidden_layers=config.LAYERS,
            hidden_units=config.NEURONS,
        )
        
    def forward(self, x, layer_idx=0):
        return self.layer(x)

    def predict_all(self, x):
        return self.forward(x)

    def set_hard_bc(self, use_hard):
        config.USE_HARD_SIDE_BC = bool(use_hard)
