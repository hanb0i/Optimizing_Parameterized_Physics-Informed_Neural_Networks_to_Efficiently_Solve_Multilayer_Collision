
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure paths
sys.path.append(os.path.join(os.getcwd(), 'pinn-workflow'))
import pinn_config as config
import model

def diagnostic_e_sweep():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = model.MultiLayerPINN().to(device)
    
    path = "pinn_model.pth"
    if os.path.exists(path):
        pinn.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        pinn.eval()
        print(f"Loaded model from {path}")
    else:
        print("Model not found.")
        return

    # Sweep E from 1 to 10
    e_vals = np.linspace(1.0, 10.0, 10)
    thickness = 0.1
    
    u_peaks = []
    v_peaks = []
    
    for E in e_vals:
        # Sample center point
        pts = torch.tensor([[0.5, 0.5, thickness, E, thickness, 0.5, 0.3, 1.0]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Get v (raw network output scaled by scale_pinn)
            # We need to peek inside LayerNet.forward or re-implement
            x_coord = pts[:, 0:1]
            y_coord = pts[:, 1:2]
            z_coord = pts[:, 2:3]
            e_param = pts[:, 3:4]
            t_param = pts[:, 4:5]
            
            # Reconstruct the forward logic to see 'v' vs 'u'
            # We use the internal layer to get raw output
            net_out = pinn.layer.net(pinn.layer.forward_features(pts))
            raw_v = net_out[0, 2].item() * config.OUTPUT_SCALE_Z
            
            # Final anchored u
            final_u = pinn(pts)[0, 2].item()
            
            u_peaks.append(abs(final_u))
            v_peaks.append(abs(raw_v))
            
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(e_vals, u_peaks, 'o-', label='Final Displacement |u|')
    plt.xlabel('Young\'s Modulus E')
    plt.ylabel('|u|')
    plt.title('Final Compliance Scaling')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(e_vals, v_peaks, 's-', color='orange', label='Raw Network |v|')
    plt.xlabel('Young\'s Modulus E')
    plt.ylabel('|v|')
    plt.title('Network Signal (Target O(1))')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('e_sweep_diagnostic.png')
    print("Saved diagnostic plot to e_sweep_diagnostic.png")

# Helper to access internal features
def patch_model():
    def forward_features(self, x):
        t_param = x[:, 4:5]
        z_coord = x[:, 2:3]
        z_hat = z_coord / torch.clamp(t_param, min=1e-6)
        
        e_min, e_max = config.E_RANGE
        e_norm = (x[:, 3:4] - e_min) / (e_max - e_min)
        t_min, t_max = config.THICKNESS_RANGE
        t_norm = (x[:, 4:5] - t_min) / (t_max - t_min)
        r_norm = (x[:, 5:6] - config.RESTITUTION_RANGE[0]) / (config.RESTITUTION_RANGE[1] - config.RESTITUTION_RANGE[0])
        mu_norm = (x[:, 6:7] - config.FRICTION_RANGE[0]) / (config.FRICTION_RANGE[1] - config.FRICTION_RANGE[0])
        v0_norm = (x[:, 7:8] - config.IMPACT_VELOCITY_RANGE[0]) / (config.IMPACT_VELOCITY_RANGE[1] - config.IMPACT_VELOCITY_RANGE[0])
        
        feat_inv1 = config.H / torch.clamp(t_param, min=1e-6)
        feat_inv2 = feat_inv1 ** 2
        feat_inv3 = feat_inv1 ** 3
        extra_feats = torch.cat([feat_inv1, feat_inv2, feat_inv3], dim=1)
        
        return torch.cat([x[:, 0:1], x[:, 1:2], z_hat, e_norm, t_norm, r_norm, mu_norm, v0_norm, extra_feats], dim=1)
    
    model.LayerNet.forward_features = forward_features

if __name__ == "__main__":
    patch_model()
    diagnostic_e_sweep()
