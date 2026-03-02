
import sys
import os
import torch
import numpy as np
import pandas as pd

# Add project paths
sys.path.append(os.path.join(os.getcwd(), 'pinn-workflow'))
sys.path.append(os.path.join(os.getcwd(), 'fea-workflow/solver'))
import fem_solver
from cascaded_surrogate import CascadedSandwichSolver

def run_benchmark():
    solver = CascadedSandwichSolver()
    
    # Define Test Cases
    test_cases = [
        {
            "name": "Standard Sandwich (Stiff-Soft-Stiff)",
            "layers": [
                {'E': 10.0, 't': 0.02}, # Top Face
                {'E': 1.0,  't': 0.06}, # Core
                {'E': 10.0, 't': 0.02}  # Bot Face
            ]
        },
        {
            "name": "Soft-Stiff-Soft (Inverted)",
            "layers": [
                {'E': 1.0,  't': 0.03},
                {'E': 10.0, 't': 0.04},
                {'E': 1.0,  't': 0.03}
            ]
        },
        {
            "name": "Thickness Gradient",
            "layers": [
                {'E': 10.0, 't': 0.01},
                {'E': 5.0,  't': 0.04},
                {'E': 2.0,  't': 0.05}
            ]
        },
        {
            "name": "Homogeneous Soft (E=1)",
            "layers": [
                {'E': 1.0, 't': 0.03},
                {'E': 1.0, 't': 0.04},
                {'E': 1.0, 't': 0.03}
            ]
        },
        {
            "name": "Homogeneous Stiff (E=10)",
            "layers": [
                {'E': 10.0, 't': 0.03},
                {'E': 10.0, 't': 0.04},
                {'E': 10.0, 't': 0.03}
            ]
        }
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("3-LAYER SANDWICH BENCHMARK SUITE")
    print("="*60)
    
    for case in test_cases:
        print(f"\nEvaluating: {case['name']}")
        
        # 1. PINN Cascaded Prediction
        total_pinn, _ = solver.solve_3_layer(case['layers'], p_impact=1.0)
        
        # 2. FEA Ground Truth
        layers = case['layers']
        total_thickness = sum(l['t'] for l in layers)
        
        fea_cfg = {
            'geometry': {'Lx': 1.0, 'Ly': 1.0, 'H': total_thickness},
            'material': [{'E': l['E'], 'nu': 0.3} for l in layers], 
            'load_patch': {
                'x_start': 0.333, 'x_end': 0.667,
                'y_start': 0.333, 'y_end': 0.667,
                'pressure': 1.0
            },
            'use_soft_mask': True,
            'mesh': {'ne_x': 15, 'ne_y': 15, 'ne_z': 15}
        }
        
        _, _, _, u_grid = fem_solver.solve_fem(fea_cfg)
        u_z = u_grid[:, :, :, 2]
        total_fea = np.abs(np.min(u_z))
        
        error = np.abs(total_pinn - total_fea) / total_fea * 100
        
        print(f"  PINN: {total_pinn:.4f}")
        print(f"  FEA:  {total_fea:.4f}")
        print(f"  Error: {error:.2f}%")
        
        results.append({
            "Case": case['name'],
            "PINN": total_pinn,
            "FEA": total_fea,
            "Error (%)": error
        })
        
    # Summarize in Table
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Save to file
    df.to_csv("benchmark_3layer_results.csv", index=False)
    print("\nResults saved to benchmark_3layer_results.csv")

if __name__ == "__main__":
    run_benchmark()
