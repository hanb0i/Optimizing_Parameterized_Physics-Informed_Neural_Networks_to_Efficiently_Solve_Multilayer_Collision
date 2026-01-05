import os
import sys
import numpy as np

from surrogate_workflow import config

FEA_DIR = os.path.join(config.ROOT_DIR, "fea-workflow")
if FEA_DIR not in sys.path:
    sys.path.append(FEA_DIR)

from solver.fem_solver import solve_fem  # noqa: E402


def _layer_moduli_from_design(mu):
    e_layers = [config.BASE_LAYER_E] * 3
    for name, value in zip(config.DESIGN_PARAMS, mu):
        if name == "E1":
            e_layers[0] = value
        elif name == "E2":
            e_layers[1] = value
        elif name == "E3":
            e_layers[2] = value
        else:
            raise ValueError(f"Unsupported design parameter: {name}")
    return e_layers


def build_cfg(mu):
    e_layers = _layer_moduli_from_design(mu)
    nu_layers = [config.BASE_NU] * 3
    return {
        "geometry": dict(config.GEOMETRY),
        "load_patch": dict(config.LOAD_PATCH),
        "material": {
            "E": e_layers[0],
            "nu": config.BASE_NU,
        },
        "layers": {
            "interfaces": list(config.LAYER_INTERFACES),
            "E": e_layers,
            "nu": nu_layers,
        },
    }


def compute_response(mu):
    cfg = build_cfg(mu)
    _, _, _, u_grid = solve_fem(cfg)
    uz_top = u_grid[:, :, -1, 2]
    return float(-np.min(uz_top))
