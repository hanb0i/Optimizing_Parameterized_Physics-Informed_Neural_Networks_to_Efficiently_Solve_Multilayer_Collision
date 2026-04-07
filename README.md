# Three-Layer PINN for Impact-Attenuation Design

Physics-Informed Neural Network (PINN) for predicting displacement fields in three-layered structures under impact loading, with surrogate-based design optimization.

## Overview

This repository implements a complete pipeline for:
1. **Physics Simulation**: PINN predicts 3D displacement fields for multi-layered materials
2. **Surrogate Modeling**: Fast MLP approximates physics for design optimization
3. **Design Optimization**: Gradient-based optimization finds optimal material configurations
4. **Validation**: Comparison tools verify PINN accuracy against FEA

## Repository Structure

```
├── config.py              # Pipeline configuration classes
├── dataset.py             # Dataset generation for surrogate training
├── surrogate.py           # Surrogate model training
├── optimize.py            # Design optimization
├── metrics.py             # Performance metrics computation
├── evaluate.py            # Validation and active learning
├── run_pipeline.py        # End-to-end pipeline demo
├── compare_three_layer_pinn_fem.py  # 3-layer validation
├── compare_two_layer_pinn_fem.py    # 2-layer validation
├── pinn-workflow/         # PINN training code
│   ├── train.py           # Main training script
│   ├── model.py           # Network architecture
│   ├── physics.py         # Physics loss functions
│   ├── data.py            # Data sampling
│   └── pinn_config.py     # Training configuration
├── fea-workflow/          # FEA solver
│   └── solver/
│       └── fem_solver.py  # Hex8 FEM implementation
└── pinn-workflow-2layer/  # 2-layer PINN (legacy)
```

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Train the PINN

```bash
cd pinn-workflow
python train.py
```

The trained model is saved as `pinn_model.pth`.

### Validate Against FEA

```bash
# 3-layer validation
python compare_three_layer_pinn_fem.py

# 2-layer validation (requires 2-layer model)
python compare_two_layer_pinn_fem.py
```

### Run Design Optimization Pipeline

```bash
python run_pipeline.py
```

## PINN Architecture

The PINN (`MultiLayerPINN`) predicts 3D displacement fields:

**Inputs** (12 features):
- Spatial: `x, y, z`
- Material: `E1, t1, E2, t2, E3, t3` (Young's modulus and thickness per layer)
- Impact: `restitution, friction, impact_velocity`

**Outputs** (3 components):
- Displacement: `ux, uy, uz`

**Network**: Fully-connected MLP with:
- Input normalization using configured parameter ranges
- Derived features (interface indicators, bending terms)
- Optional hard boundary condition enforcement

### Compliance Scaling

The network outputs `v` which is converted to displacement `u` via:
```
u = scale * v / E^p * (H/t)^alpha
```

where `p` (E_COMPLIANCE_POWER) and `alpha` (THICKNESS_COMPLIANCE_ALPHA) are configurable.

## Training

### Configuration

Key parameters in `pinn_config.py`:

```python
# Loss weights
WEIGHTS = {
    'pde': 10.0,         # Equilibrium equation
    'bc': 0.7,           # Boundary conditions
    'load': 5.0,         # Load patch traction
    'energy': 0.63,      # Energy consistency
    'interface_u': 300.0,# Interface continuity
    'data': 400.0,       # FEA supervision
}

# Sampling
N_INTERIOR = 15000     # Interior collocation points
N_SIDES = 2000         # Side boundary points
N_TOP_LOAD = 6000      # Load patch points
N_TOP_FREE = 2000      # Free surface points
N_INTERFACE = 16000    # Interface points

# Training
EPOCHS_SOAP = 400      # SOAP optimizer steps
EPOCHS_LBFGS = 0       # L-BFGS fine-tuning steps
LEARNING_RATE = 1e-3
```

### Environment Variables

Override config without editing files:

```bash
# Device selection
export PINN_DEVICE=cuda

# Loss weights
export PINN_W_PDE=20.0
export PINN_W_INTERFACE_U=500.0
export PINN_W_LOAD=10.0

# Supervision data
export PINN_DATA_E_VALUES="1.0,5.0,10.0"
export PINN_DATA_T1_VALUES="0.02,0.06,0.10"

# Warm start
export PINN_WARM_START=1

# Output directory
export PINN_OUT_DIR=/path/to/outputs
```

### Training Process

1. **SOAP Optimization**: Main training using second-order preconditioning
2. **Adaptive Resampling**: Every 500 epochs, resample based on PDE residuals
3. **L-BFGS Fine-tuning**: Optional second-stage optimization
4. **Checkpointing**: Model saved after each L-BFGS step

## Surrogate Model & Design Optimization

### Pipeline Overview

```
Physics Runner (PINN/FEA)
    ↓
Metrics Computation (strain energy, acceleration, displacement)
    ↓
Surrogate Training (MLP: params → metrics)
    ↓
Gradient-Based Optimization
    ↓
Optimal Design
```

### Metrics

Three primary metrics for optimization:
- **Strain Energy**: To be minimized (impact absorption)
- **Peak Acceleration**: Constrained (protection requirement)
- **Peak Displacement**: Constrained (space requirement)

### Optimization Objective

```
minimize:   strain_energy
subject to: acceleration ≤ a_cap
            displacement ≤ u_cap
```

Implemented via softplus penalties on constraint violations.

### Active Learning

Iteratively improves surrogate by:
1. Optimizing design candidates through current surrogate
2. Evaluating true physics on candidates
3. Adding points with large prediction errors to training set
4. Retraining surrogate

## FEA Solver

The FEA solver (`fem_solver.py`) implements:
- Hex8 elements with 2x2x2 Gauss quadrature
- Layer-wise material properties
- Penalty method for clamped boundaries
- Load patch with smooth distribution

Functions:
- `solve_fem()`: Single-layer baseline
- `solve_two_layer_fem()`: Two-layer structure
- `solve_three_layer_fem()`: Three-layer structure

## Validation

### Comparison Scripts

Generate side-by-side contour plots:

```bash
python compare_three_layer_pinn_fem.py
```

Creates visualizations in `pinn-workflow/visualization_three_layer/`:
- `{case}_top.png`: Top surface displacement and error
- `{case}_cross_section.png`: Cross-section displacement and error
- `three_layer_sweep_tmin_E2*.png`: MAE heatmaps across E1-E3 space

### Expected Accuracy

With proper training:
- Mean MAE: < 3% of max displacement
- Worst-case MAE: < 5-10% (typically at extreme parameter combinations)

## Configuration Reference

### Geometry (pinn_config.py)

```python
Lx = 1.0          # Plate length (x)
Ly = 1.0          # Plate width (y)
H = 0.1           # Total thickness (z)
NUM_LAYERS = 3    # Number of material layers
```

### Material Parameters

```python
E_RANGE = [1.0, 10.0]           # Young's modulus range
T1_RANGE = [0.02, 0.10]         # Layer 1 thickness
T2_RANGE = [0.02, 0.10]         # Layer 2 thickness
T3_RANGE = [0.02, 0.10]         # Layer 3 thickness
nu_vals = [0.3]                 # Poisson's ratio
```

### Loading

```python
p0 = 1.0                        # Load magnitude
LOAD_PATCH_X = [Lx/3, 2*Lx/3]  # Load patch x-range
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]  # Load patch y-range
```

## Troubleshooting

### Low Displacement Magnitude
- Increase `WEIGHTS['load']` or `WEIGHTS['energy']`
- Check compliance scaling parameters
- Verify FEA supervision data is loading (if enabled)

### High MAE on Validation
- Increase training epochs
- Adjust loss weights (increase `WEIGHTS['data']` if using supervision)
- Enable adaptive resampling
- Check interface continuity loss weight

### Convergence Issues
- Reduce learning rate
- Adjust SOAP precondition frequency
- Check for NaN in losses (may indicate sampling issues)

## Citation

If using this code, please cite relevant PINN literature and acknowledge the physics-informed neural network approach for multi-layered structures.

## License

This project is provided as-is for research and educational purposes.
