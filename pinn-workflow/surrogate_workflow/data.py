import numpy as np

from surrogate_workflow import config
from surrogate_workflow import baseline


def latin_hypercube(n_samples: int, n_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.random((n_samples, n_dim))
    points = np.zeros_like(u)
    for j in range(n_dim):
        a = cut[:n_samples]
        b = cut[1:]
        points[:, j] = u[:, j] * (b - a) + a
        rng.shuffle(points[:, j])
    return points


def sample_designs(n_samples: int, ranges: dict, seed: int) -> np.ndarray:
    params = list(ranges.keys())
    lhs = latin_hypercube(n_samples, len(params), seed)
    x_raw = np.zeros_like(lhs)
    for i, name in enumerate(params):
        low, high = ranges[name]
        x_raw[:, i] = low + lhs[:, i] * (high - low)
    return x_raw


def normalize_inputs(x_raw: np.ndarray, ranges: dict):
    params = list(ranges.keys())
    x_min = np.array([ranges[name][0] for name in params], dtype=float)
    x_max = np.array([ranges[name][1] for name in params], dtype=float)
    denom = np.clip(x_max - x_min, 1e-12, None)
    x_norm = (x_raw - x_min) / denom
    return x_norm, x_min, x_max


def normalize_outputs(y_raw: np.ndarray):
    y_min = float(np.min(y_raw))
    y_max = float(np.max(y_raw))
    y_norm = (y_raw - y_min) / (y_max - y_min + 1e-12)
    return y_norm, y_min, y_max


def split_indices(n_samples: int, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    # Ensure non-empty splits when possible (helps small smoke runs).
    n_train = max(1, n_train)
    remaining = max(0, n_samples - n_train)
    if remaining >= 2:
        n_val = max(1, n_val)
    else:
        n_val = 0
    if n_train + n_val >= n_samples:
        n_val = max(0, n_samples - n_train - 1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def generate_dataset():
    x_raw = sample_designs(config.N_SAMPLES, config.DESIGN_RANGES, config.SEED)
    if getattr(config, "TREND_ANCHOR_POINTS", 0) > 0:
        param_names = list(config.DESIGN_RANGES.keys())
        if config.TREND_SWEEP_PARAM in param_names:
            ranges = config.DESIGN_RANGES
            sweep = np.linspace(
                ranges[config.TREND_SWEEP_PARAM][0],
                ranges[config.TREND_SWEEP_PARAM][1],
                config.TREND_ANCHOR_POINTS,
            )
            mu_mid = config.mid_design()
            idx = param_names.index(config.TREND_SWEEP_PARAM)
            anchors = np.tile(mu_mid, (config.TREND_ANCHOR_POINTS, 1))
            anchors[:, idx] = sweep
            x_raw = np.vstack([x_raw, anchors])
    y_raw = np.zeros(x_raw.shape[0], dtype=float)
    for i, mu in enumerate(x_raw):
        y_raw[i] = baseline.compute_response(mu)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"Baseline {i + 1}/{x_raw.shape[0]}: y={y_raw[i]:.6f}")
    x_norm, x_min, x_max = normalize_inputs(x_raw, config.DESIGN_RANGES)
    y_norm, y_min, y_max = normalize_outputs(y_raw)
    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "param_names": list(config.DESIGN_RANGES.keys()),
    }


def save_dataset(path: str, dataset: dict):
    np.savez(
        path,
        x_raw=dataset["x_raw"],
        y_raw=dataset["y_raw"],
        x_norm=dataset["x_norm"],
        y_norm=dataset["y_norm"],
        x_min=dataset["x_min"],
        x_max=dataset["x_max"],
        y_min=dataset["y_min"],
        y_max=dataset["y_max"],
        param_names=np.array(dataset["param_names"], dtype=object),
    )


def load_dataset(path: str) -> dict:
    blob = np.load(path, allow_pickle=True)
    return {
        "x_raw": blob["x_raw"],
        "y_raw": blob["y_raw"],
        "x_norm": blob["x_norm"],
        "y_norm": blob["y_norm"],
        "x_min": blob["x_min"],
        "x_max": blob["x_max"],
        "y_min": float(blob["y_min"]),
        "y_max": float(blob["y_max"]),
        "param_names": list(blob["param_names"]),
    }
