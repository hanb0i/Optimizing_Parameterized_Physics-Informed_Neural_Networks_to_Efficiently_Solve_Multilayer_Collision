# CAD Geometry → PiNN Workflow (PhysicsNeMo-style)

```mermaid
flowchart TD
  A[CAD model exported as STL] --> B[Load STL triangles]
  B --> C[PhysicsNeMo-style tessellation sampling]
  C --> C1[sample_boundary: (x,y,z)+normals]
  C --> C2[sample_interior: reject via point-in-mesh + compute SDF (optional)]
  C1 --> D[Classify boundary points]
  D --> D1[Top surface (normal_z > thresh)]
  D --> D2[Bottom surface (normal_z < -thresh)]
  D --> D3[Sides (rest)]
  D1 --> E1[Split top into load patch vs free via LOAD_PATCH_X/Y]
  D2 --> E2[Bottom traction-free points]
  D3 --> E3[Clamped side BC points]
  C2 --> F[Interior PDE points]
  E1 --> G[Assemble training/inference tensors: (x,y,z,E,t,r,mu,v0)]
  E2 --> G
  E3 --> G
  F --> G
  G --> H[PiNN forward pass v(x,params)]
  H --> I[Convert to displacement u = v / E^p · (H/t)^α]
  I --> J[Losses / inference / visualization]
```

Outputs for the included demo STL are written by `pinn-workflow/visualize_cad_pinn.py` into `impact_pipeline_outputs/cad_viz/`.

