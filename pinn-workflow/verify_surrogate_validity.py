import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add current directory and fea-workflow to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_DIR = os.path.join(os.path.dirname(ROOT_DIR), "fea-workflow")
FEA_SOLVER_DIR = os.path.join(FEA_DIR, "solver")
sys.path.append(ROOT_DIR)
sys.path.append(FEA_DIR)
sys.path.append(FEA_SOLVER_DIR)

from surrogate_api import ParametricSurrogate
from surrogate_workflow import config as scfg
import fem_solver
solve_fem = fem_solver.solve_fem

def get_fea_response(mu_vec):
    """
    Computes FEA ground truth response for a given parameter set.
    """
    E_val, t_val, r_val, mu_fric, v0_val = mu_vec
    
    # Build FEA config matching pinn_config.py logic (Hard rectangular mask)
    fea_cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": t_val},
        "material": {"E": E_val, "nu": 0.3},
        "load_patch": {
            "x_start": 0.333, "x_end": 0.667,
            "y_start": 0.333, "y_end": 0.667,
            "pressure": 1.0
        },
        "use_soft_mask": False 
    }
    
    _, _, _, u_grid = solve_fem(fea_cfg)
    uz_top = u_grid[:, :, -1, 2]
    return float(np.abs(np.min(uz_top)))

def run_physical_sweeps(ps):
    """
    Generates plots for sensitivity sweeps across all 5 parameters.
    """
    print("\n--- Physical Trend Sweeps ---")
    os.makedirs(scfg.PLOTS_DIR, exist_ok=True)
    sweep_dir = os.path.join(scfg.PLOTS_DIR, "validity_sweeps")
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Baseline for sweeps
    baseline_params = {
        "E": 5.0, "thickness": 0.1, "restitution": 0.5, "friction": 0.3, "impact_velocity": 1.0
    }
    
    for param in scfg.DESIGN_PARAMS:
        sweep_vals = np.linspace(scfg.DESIGN_RANGES[param][0], scfg.DESIGN_RANGES[param][1], 50)
        preds = []
        for v in sweep_vals:
            p = baseline_params.copy()
            p[param] = v
            preds.append(ps.predict(p))
            
        plt.figure(figsize=(8, 5))
        plt.plot(sweep_vals, preds, 'b-', lw=2)
        plt.title(f"Surrogate Sensitivity: {param}")
        plt.xlabel(param)
        plt.ylabel("Peak Displacement (Magnitude)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(sweep_dir, f"sweep_{param}.png"))
        plt.close()
        print(f"Generated sweep plot for {param}")

def run_fea_cross_verification(ps, n_points=5):
    """
    Compares surrogate predictions against direct FEA ground truth on random samples.
    """
    print(f"\n--- FEA Cross-Verification ({n_points} points) ---")
    print(f" {'#':>2} | {'Params (E, t, r, mu, v0)':<40} | {'Surrogate':>10} | {'FEA':>10} | {'Error %':>8}")
    print("-" * 85)
    
    errors = []
    for i in range(n_points):
        # Sample random point in 5D space
        mu_vec = []
        p_dict = {}
        for p in scfg.DESIGN_PARAMS:
            v = np.random.uniform(scfg.DESIGN_RANGES[p][0], scfg.DESIGN_RANGES[p][1])
            mu_vec.append(v)
            p_dict[p] = v
            
        surr_pred = ps.predict(p_dict)
        fea_truth = get_fea_response(mu_vec)
        
        rel_err = abs(surr_pred - fea_truth) / fea_truth * 100
        errors.append(rel_err)
        
        param_str = "[" + ", ".join([f"{v:.2f}" for v in mu_vec]) + "]"
        print(f" {i+1:>2} | {param_str:<40} | {surr_pred:>10.4f} | {fea_truth:>10.4f} | {rel_err:>8.2f}%")
        
    avg_err = np.mean(errors)
    print("-" * 85)
    print(f"Average Relative Error: {avg_err:.2f}%")
    print(f"Maximum Relative Error: {max(errors):.2f}%")
    
    if avg_err > 8.0:
        print("\nVALIDATION WARNING: Surrogate error against FEA exceeds 8% threshold.")
    else:
        print("\nVALIDATION SUCCESS: Surrogate maintains high fidelity to FEA ground truth.")

if __name__ == "__main__":
    ps = ParametricSurrogate()
    run_physical_sweeps(ps)
    run_fea_cross_verification(ps, n_points=5)
