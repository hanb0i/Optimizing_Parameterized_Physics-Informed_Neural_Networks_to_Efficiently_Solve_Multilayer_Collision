"""
Central configuration for the Phase 4–7 impact-attenuation pipeline.

All values are intentionally generic and safe defaults for running the end-to-end
synthetic example in `run_pipeline.py`. When swapping in a real PiNN/FEA runner,
update bounds, caps, and any metric-region definitions as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from metrics import MetricsConfig


@dataclass(frozen=True)
class Bounds:
    names: List[str]
    low: torch.Tensor
    high: torch.Tensor

    @staticmethod
    def from_dict(bounds: Mapping[str, Tuple[float, float]], *, device: Optional[torch.device] = None) -> "Bounds":
        names = list(bounds.keys())
        low = torch.tensor([bounds[k][0] for k in names], dtype=torch.float32, device=device)
        high = torch.tensor([bounds[k][1] for k in names], dtype=torch.float32, device=device)
        return Bounds(names=names, low=low, high=high)

    @property
    def dim(self) -> int:
        return int(self.low.numel())


@dataclass(frozen=True)
class SurrogateConfig:
    hidden_layers: int = 3
    hidden_units: int = 128
    activation: str = "gelu"  # "tanh" | "gelu" | "relu"

    use_fourier_features: bool = False
    fourier_features_dim: int = 64
    fourier_sigma: float = 3.0
    seed: int = 7

    batch_size: int = 128
    lr: float = 1e-3
    max_epochs: int = 3000
    patience: int = 200
    min_delta: float = 1e-6

    train_frac: float = 0.7
    val_frac: float = 0.15

    output_weights: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class OptimizeConfig:
    a_cap: float = 50.0
    u_cap: float = 0.02
    lambda_accel: float = 10.0
    beta_disp: float = 10.0
    penalty_beta: float = 10.0

    lr: float = 5e-2
    steps: int = 300
    project_box: bool = True
    restarts: int = 5
    top_k: int = 3


@dataclass(frozen=True)
class DamageConfig:
    enabled: bool = False
    n_impacts: int = 3
    eps_crit: float = 0.02
    softplus_beta: float = 50.0
    d_max: float = 1.0
    modulus_param_names: Sequence[str] = field(default_factory=tuple)
    modulus_scale_max: float = 1.0


@dataclass(frozen=True)
class ActiveLearningConfig:
    enabled: bool = True
    iters: int = 2
    discrepancy_tol: float = 0.25
    add_points_per_iter: int = 5


@dataclass(frozen=True)
class PipelineConfig:
    device: torch.device = torch.device("cpu")
    out_dir: Path = Path("impact_pipeline_outputs")

    bounds: Optional[Bounds] = None
    metrics: MetricsConfig = MetricsConfig()
    surrogate: SurrogateConfig = SurrogateConfig()
    optimize: OptimizeConfig = OptimizeConfig()
    damage: DamageConfig = DamageConfig()
    active_learning: ActiveLearningConfig = ActiveLearningConfig()

    dataset_size: int = 256

