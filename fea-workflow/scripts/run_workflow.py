
import sys
import os
import torch
import numpy as np

# Add project root to path to access both workflows
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)
# Also add pinn-workflow specifically so 'import config' works inside train.py
sys.path.append(os.path.join(root_dir, 'pinn-workflow'))

# Import from pinn-workflow
import pinn_config # Now this works directly
import train as pinn_train

# Import from fea-workflow
# Need to add fea-workflow to path or rel import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver.fem_solver import solve_fem
from postprocessing.visualization import plot_pinn_results, plot_comparison, plot_loss_history

def get_cfg_from_pinn_config():
    """Convert pinn-workflow config module variables to the dictionary required by FEM solver"""
    return {
        'geometry': {
            'Lx': pinn_config.Lx, 
            'Ly': pinn_config.Ly, 
            'H': pinn_config.H
        },
        'load_patch': {
            'x_start': 0.33, 'x_end': 0.67, 
            'y_start': 0.33, 'y_end': 0.67,
            'pressure': pinn_config.p0
        },
        'material': {
            'E': pinn_config.E_vals[0], 
            'nu': pinn_config.nu_vals[0]
        },
        'layers': {
            'interfaces': pinn_config.Layer_Interfaces,
            'E': pinn_config.E_vals,
            'nu': pinn_config.nu_vals
        },
    }

def main():
    print("=== Phase 1: Training PINN via pinn-workflow ===")
    
    # Define Callback for Visualization
    cfg = get_cfg_from_pinn_config()
    def visualization_callback(tag, model, device):
        print(f"Creating visualization for {tag}...")
        save_path = os.path.dirname(__file__)
        plot_pinn_results(model, cfg, device, save_path=save_path)
        # Rename so we don't overwrite
        os.rename(os.path.join(save_path, "pinn_top.png"), 
                  os.path.join(save_path, f"pinn_top_{tag}.png"))
        print(f"Saved pinn_top_{tag}.png")

    # Call the external training script
    # This will use pinn-workflow/config.py
    pinn_model = pinn_train.train(callback=visualization_callback)
    
    # Plot Loss History
    try:
        loss_hist = np.load("loss_history.npy")
        plot_loss_history(loss_hist, adam_epochs=pinn_config.EPOCHS_ADAM, save_path=os.path.dirname(__file__))
    except FileNotFoundError:
        print("Could not find loss_history.npy to plot.")
    
    print("\n=== Phase 2: Running FEA Benchmark ===")
    cfg = get_cfg_from_pinn_config()
    x, y, z, u_fea = solve_fem(cfg)
    
    print("\n=== Phase 3: Comparison ===")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    plot_pinn_results(pinn_model, cfg, device, save_path=os.path.dirname(__file__))
    plot_comparison(u_fea, (x, y, z), pinn_model, cfg, device, save_path=os.path.dirname(__file__))
    uz_fea_top = u_fea[:, :, -1, 2]
    X_fea, Y_fea = np.meshgrid(x, y, indexing='ij')
    pts_top = np.stack([X_fea.ravel(), Y_fea.ravel(), np.ones_like(X_fea.ravel())*cfg['geometry']['H']], axis=1)
    pts_top_t = torch.tensor(pts_top, dtype=torch.float32).to(device)
    with torch.no_grad():
        u_pinn = (pinn_model(pts_top_t, 2) * 100.0).cpu().numpy()
        uz_pinn_top = u_pinn[:, 2].reshape(X_fea.shape)
    diff = np.abs(uz_fea_top - uz_pinn_top)
    denom = np.maximum(np.abs(uz_fea_top), 1e-8)
    percent_error = np.mean(diff / denom) * 100.0
    print(f"Mean % Error (Top u_z): {percent_error:.2f}% (lower is better)")
    
    print("Workflow Complete. Comparison plots generated.")

if __name__ == "__main__":
    main()
