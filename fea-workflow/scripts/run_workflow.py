
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
    }

def main():
    print("=== Phase 1: Training PINN via pinn-workflow ===")
    # Call the external training script
    # This will use pinn-workflow/config.py
    pinn_model = pinn_train.train()
    
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
    
    print("Workflow Complete. Comparison plots generated.")

if __name__ == "__main__":
    main()
