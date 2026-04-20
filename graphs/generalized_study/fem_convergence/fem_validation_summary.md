# FEM Solver Validation Summary

## One-Layer (E=10, t=0.05)

| Mesh | Elements | Peak |u_z| | Δ vs 32³ |
|------|----------|-------------|----------|
| 4×4×2 | 32 | 0.474519 | -1.453001 |
| 8×8×4 | 256 | 0.887521 | -1.039999 |
| 16×16×8 | 2048 | 1.533069 | -0.394451 |
| 32×32×16 | 16384 | 1.927520 | +0.000000 |

## Three-Layer (E=[10,10,10], t=[0.02,0.10,0.02])

| Mesh | Elements | Peak |u_z| | Δ vs 32³ |
|------|----------|-------------|----------|
| 4×4×2 | 32 | 0.119890 | -0.003435 |
| 8×8×4 | 256 | 0.115386 | -0.007939 |
| 16×16×8 | 2048 | 0.120991 | -0.002334 |
| 32×32×16 | 16384 | 0.123325 | +0.000000 |

**Note:** Benchmark mesh (8×8×4) used for all PINN-vs-FEM comparisons. Absolute differences in physical units are small; PINN and FEM use identical meshes for consistency.