#!/usr/bin/env python3
"""Verify all claims about the FEM supervision data configuration."""

import sys
import os
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[1]
THREE_LAYER_DIR = REPO_ROOT / "three-layer-workflow"
sys.path.insert(0, str(THREE_LAYER_DIR))

import pinn_config as config

print("=" * 70)
print("SUPERVISION DATA CONFIGURATION VERIFICATION")
print("=" * 70)

# 1. Parameter extreme values
print("\n--- 1. Parameter Values Used for Supervision ---")
e_vals = config.DATA_E_VALUES
t1_vals = config.DATA_T1_VALUES
t2_vals = config.DATA_T2_VALUES
t3_vals = config.DATA_T3_VALUES
print(f"  E values:  {e_vals}  (count: {len(e_vals)})")
print(f"  t1 values: {t1_vals}  (count: {len(t1_vals)})")
print(f"  t2 values: {t2_vals}  (count: {len(t2_vals)})")
print(f"  t3 values: {t3_vals}  (count: {len(t3_vals)})")

e_range = config.E_RANGE
t1_range = config.T1_RANGE
t2_range = config.T2_RANGE
t3_range = config.T3_RANGE
print(f"\n  E range:  {e_range}")
print(f"  t1 range: {t1_range}")
print(f"  t2 range: {t2_range}")
print(f"  t3 range: {t3_range}")

only_extremes_E = set(e_vals) == {e_range[0], e_range[1]}
only_extremes_t1 = set(t1_vals) == {t1_range[0], t1_range[1]}
only_extremes_t2 = set(t2_vals) == {t2_range[0], t2_range[1]}
only_extremes_t3 = set(t3_vals) == {t3_range[0], t3_range[1]}
print(f"\n  E only at extremes?  {only_extremes_E}")
print(f"  t1 only at extremes? {only_extremes_t1}")
print(f"  t2 only at extremes? {only_extremes_t2}")
print(f"  t3 only at extremes? {only_extremes_t3}")

# 2. Number of configurations
print("\n--- 2. Number of Configurations ---")
n_E_combos = len(e_vals) ** 3  # E1 x E2 x E3
n_t_combos = len(t1_vals) * len(t2_vals) * len(t3_vals)
n_total_configs = n_E_combos * n_t_combos
print(f"  E combos (E1 x E2 x E3):    {len(e_vals)}^3 = {n_E_combos}")
print(f"  t combos (t1 x t2 x t3):    {len(t1_vals)} x {len(t2_vals)} x {len(t3_vals)} = {n_t_combos}")
print(f"  Total configurations:         {n_E_combos} x {n_t_combos} = {n_total_configs}")

# 3. Total supervision points and FEM mesh
print("\n--- 3. Supervision Budget and FEM Mesh ---")
print(f"  N_DATA_POINTS:     {config.N_DATA_POINTS}")
print(f"  FEM mesh:          {config.FEM_NE_X} x {config.FEM_NE_Y} x {config.FEM_NE_Z}")
nodes_per_mesh = (config.FEM_NE_X + 1) * (config.FEM_NE_Y + 1) * (config.FEM_NE_Z + 1)
print(f"  Nodes per mesh:    {nodes_per_mesh}")
print(f"  Avg pts/config:    {config.N_DATA_POINTS / n_total_configs:.1f}")
print(f"  Sampling fraction: {config.N_DATA_POINTS / (n_total_configs * nodes_per_mesh) * 100:.1f}%")

# 4. Thickness weighting
print("\n--- 4. Thickness Weighting ---")
print(f"  SUPERVISION_THICKNESS_POWER: {config.SUPERVISION_THICKNESS_POWER}")

# 5. USE_SUPERVISION_DATA flag
print("\n--- 5. Supervision Enabled? ---")
print(f"  USE_SUPERVISION_DATA: {config.USE_SUPERVISION_DATA}")

# 6. Loss weights
print("\n--- 6. Loss Weights ---")
for k, v in config.WEIGHTS.items():
    print(f"  {k:20s}: {v}")

physics_weights = config.WEIGHTS['pde'] + config.WEIGHTS['bc'] + config.WEIGHTS['load'] + config.WEIGHTS['energy'] + config.WEIGHTS.get('interface_u', 0)
data_weight = config.WEIGHTS.get('data', 0)
print(f"\n  Sum of physics weights: {physics_weights:.2f}")
print(f"  Data weight:            {data_weight:.2f}")
print(f"  Data / (physics+data):  {data_weight / (physics_weights + data_weight) * 100:.1f}%")

# 7. Verify supervision cache exists and inspect it
print("\n--- 7. Supervision Cache Inspection ---")
cache_dir = THREE_LAYER_DIR / "supervision_cache"
if cache_dir.exists():
    cache_files = list(cache_dir.glob("*.pt"))
    print(f"  Cache directory: {cache_dir}")
    print(f"  Cache files found: {len(cache_files)}")
    
    if cache_files:
        import torch
        blob = torch.load(cache_files[0], map_location="cpu", weights_only=False)
        x_data = blob["x"]
        u_data = blob["u"]
        meta = blob.get("meta", {})
        
        print(f"\n  Cached data shape: x={x_data.shape}, u={u_data.shape}")
        print(f"  Total supervision points: {len(x_data)}")
        
        # Extract unique parameter configurations
        # Columns: x, y, z, E1, t1, E2, t2, E3, t3, r, mu, v0
        e1_unique = sorted(set(x_data[:, 3].numpy().round(6).tolist()))
        t1_unique = sorted(set(x_data[:, 4].numpy().round(6).tolist()))
        e2_unique = sorted(set(x_data[:, 5].numpy().round(6).tolist()))
        t2_unique = sorted(set(x_data[:, 6].numpy().round(6).tolist()))
        e3_unique = sorted(set(x_data[:, 7].numpy().round(6).tolist()))
        t3_unique = sorted(set(x_data[:, 8].numpy().round(6).tolist()))
        
        print(f"\n  Unique E1 values in cache: {e1_unique}")
        print(f"  Unique t1 values in cache: {t1_unique}")
        print(f"  Unique E2 values in cache: {e2_unique}")
        print(f"  Unique t2 values in cache: {t2_unique}")
        print(f"  Unique E3 values in cache: {e3_unique}")
        print(f"  Unique t3 values in cache: {t3_unique}")
        
        # Count points per configuration
        configs = []
        for i in range(len(x_data)):
            key = (round(x_data[i, 3].item(), 4),
                   round(x_data[i, 4].item(), 4),
                   round(x_data[i, 5].item(), 4),
                   round(x_data[i, 6].item(), 4),
                   round(x_data[i, 7].item(), 4),
                   round(x_data[i, 8].item(), 4))
            configs.append(key)
        
        config_counts = Counter(configs)
        print(f"\n  Unique configurations in cache: {len(config_counts)}")
        print(f"\n  Points per configuration:")
        for cfg, count in sorted(config_counts.items(), key=lambda x: -x[1])[:10]:
            e1, t1, e2, t2, e3, t3 = cfg
            thickness = t1 + t2 + t3
            print(f"    E=[{e1},{e2},{e3}] t=[{t1},{t2},{t3}] (H={thickness:.2f}): {count} pts")
        if len(config_counts) > 10:
            print(f"    ... and {len(config_counts) - 10} more")
        
        counts = list(config_counts.values())
        print(f"\n  Min points/config: {min(counts)}")
        print(f"  Max points/config: {max(counts)}")
        print(f"  Mean points/config: {sum(counts)/len(counts):.1f}")
        
        # Check spatial distribution within one config
        first_cfg = list(config_counts.keys())[0]
        mask = [configs[i] == first_cfg for i in range(len(configs))]
        x_sub = x_data[mask]
        z_vals = x_sub[:, 2].numpy()
        t1_val = first_cfg[1]
        t2_val = first_cfg[3]
        t3_val = first_cfg[5]
        H_val = t1_val + t2_val + t3_val
        
        n_top = (z_vals >= H_val - 1e-6).sum()
        n_if1 = ((z_vals > t1_val - 1e-3) & (z_vals < t1_val + 1e-3)).sum()
        n_if2 = ((z_vals > t1_val + t2_val - 1e-3) & (z_vals < t1_val + t2_val + 1e-3)).sum()
        n_interior = len(z_vals) - n_top - n_if1 - n_if2
        
        print(f"\n  Spatial distribution for config E=[{first_cfg[0]},{first_cfg[2]},{first_cfg[4]}] t=[{first_cfg[1]},{first_cfg[3]},{first_cfg[5]}]:")
        print(f"    Top surface (z≈{H_val}):       {n_top} ({100*n_top/len(z_vals):.1f}%)")
        print(f"    Interface 1 (z≈{t1_val}):     {n_if1} ({100*n_if1/len(z_vals):.1f}%)")
        print(f"    Interface 2 (z≈{t1_val+t2_val}):     {n_if2} ({100*n_if2/len(z_vals):.1f}%)")
        print(f"    Bulk interior:              {n_interior} ({100*n_interior/len(z_vals):.1f}%)")
        
        if meta:
            print(f"\n  Cache metadata:")
            for k, v in meta.items():
                print(f"    {k}: {v}")
else:
    print(f"  No supervision cache found at {cache_dir}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
