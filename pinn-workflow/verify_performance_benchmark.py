import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT_DIR)
FEA_DIR = os.path.join(REPO_ROOT, "fea-workflow")
FEA_SOLVER_DIR = os.path.join(FEA_DIR, "solver")

sys.path.append(ROOT_DIR)
sys.path.append(FEA_SOLVER_DIR)

import pinn_config as config
import fem_solver_complex
from surrogate_api import ParametricSurrogate

def run_benchmark():
    print("=== Performance Benchmark: FEA vs. PINN Surrogate ===\n")
    
    # --- Global Font Styling ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'Roboto', 'Arial', 'sans-serif']
    plt.rcParams['font.size'] = 16

    # 1. Benchmark Surrogate
    print("Benchmarking Surrogate Prediction...")
    surrogate_api = ParametricSurrogate()
    params = {
        "E": 1.0,
        "thickness": config.H,
        "restitution": config.RESTITUTION_REF,
        "friction": config.FRICTION_REF,
        "impact_velocity": config.IMPACT_VELOCITY_REF
    }
    
    # Warmup
    for _ in range(10):
        _ = surrogate_api.predict(params)
        
    start_time = time.time()
    n_iters = 100
    for _ in range(n_iters):
        _ = surrogate_api.predict(params)
    surrogate_time = (time.time() - start_time) / n_iters
    print(f"Surrogate Avg Time: {surrogate_time:.6f} seconds ({surrogate_time*1000:.3f} ms)")

    # 2. Benchmark FEA (Voxel Benchmark)
    print("\nBenchmarking FEA Solve (40x40x20 mesh)...")
    ne_x = 40
    ne_y = 40
    ne_z = 20
    Lx, Ly, H = config.Lx, config.Ly, config.H
    
    materials = [
        {'E': config.LAYER_E_VALS[0], 'nu': config.LAYER_NU_VALS[0]},
        {'E': config.LAYER_E_VALS[1], 'nu': config.LAYER_NU_VALS[1]},
        {'E': config.LAYER_E_VALS[2], 'nu': config.LAYER_NU_VALS[2]}
    ]
    
    material_grid = np.full((ne_x, ne_y, ne_z), -1, dtype=int)
    dx = Lx / ne_x
    dy = Ly / ne_y
    dz = H / ne_z
    
    def get_z_top(x, y):
        cx, cy = config.Lx/2, config.Ly/2
        r2 = (x - cx)**2 + (y - cy)**2
        dent = config.DENT_DEPTH * np.exp(-r2 / (2 * config.DENT_WIDTH**2))
        return config.H - dent

    for k in range(ne_z):
        z_center = (k + 0.5) * dz
        for j in range(ne_y):
            y_center = (j + 0.5) * dy
            for i in range(ne_x):
                x_center = (i + 0.5) * dx
                z_limit = get_z_top(x_center, y_center)
                if z_center > z_limit:
                    material_grid[i,j,k] = -1
                else:
                    z_rel = z_center / z_limit
                    if z_rel <= config.LAYER_Z_RATIOS[0]: material_grid[i,j,k] = 0
                    elif z_rel <= config.LAYER_Z_RATIOS[1]: material_grid[i,j,k] = 1
                    else: material_grid[i,j,k] = 2

    fea_cfg = {
        'geometry': {'Lx': Lx, 'Ly': Ly, 'H': H},
        'mesh': {'ne_x': ne_x, 'ne_y': ne_y, 'ne_z': ne_z},
        'material': materials,
        'load_patch': {
            'x_start': config.LOAD_PATCH_X[0], 'x_end': config.LOAD_PATCH_X[1],
            'y_start': config.LOAD_PATCH_Y[0], 'y_end': config.LOAD_PATCH_Y[1],
            'pressure': config.p0
        },
        'use_soft_mask': True
    }
    
    start_time = time.time()
    _, _, _, _ = fem_solver_complex.solve_fem_complex(fea_cfg, material_grid)
    fea_time = time.time() - start_time
    print(f"FEA Solve Time: {fea_time:.2f} seconds")

    # 3. Create Visualization
    print("\nCreating Comparison Plot...")
    labels = ['FEA Solver (40x40x20)', 'PINN Surrogate']
    times = [fea_time, surrogate_time]
    
    # Log scale is better for such massive differences
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, times, color=['#e74c3c', '#2ecc71'])
    
    ax.set_yscale('log')
    ax.set_ylabel('Execution Time (seconds, log scale)', fontsize=12)
    ax.set_title('Inference Speed Benchmark: FEA vs. Surrogate Model', fontsize=14, fontweight='bold', pad=20)
    
    # Annotate bars with actual times
    for bar in bars:
        height = bar.get_height()
        if height >= 1.0:
            label_str = f'{height:.2f} s'
        else:
            label_str = f'{height*1000:.2f} ms'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label_str, ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(surrogate_time / 10, fea_time * 10)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Performance ratio
    speedup = fea_time / surrogate_time
    plt.figtext(0.5, 0.01, f'Surrogate Speedup: ~{int(speedup):,}x faster', 
                ha='center', fontsize=12, style='italic', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    outfile = os.path.join(ROOT_DIR, "visualization", "performance_comparison.png")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=150)
    print(f"Benchmark plot saved to {outfile}")

if __name__ == "__main__":
    run_benchmark()
