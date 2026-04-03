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
    mode = str(getattr(config, "Y_TRANSFORM", "identity")).strip().lower()
    eps = float(getattr(config, "Y_EPS", 1e-12))
    if mode == "log":
        y_trans = np.log(np.clip(y_raw, 0.0, None) + eps)
    else:
        y_trans = y_raw
        mode = "identity"
    y_min = float(np.min(y_trans))
    y_max = float(np.max(y_trans))
    y_norm = (y_trans - y_min) / (y_max - y_min + 1e-12)
    return y_norm, y_min, y_max, mode, eps


def split_indices(n_samples: int, train_frac: float, val_frac: float, seed: int, n_anchors: int = 0):
    rng = np.random.default_rng(seed)
    n_anchors = int(max(0, min(n_anchors, n_samples)))
    anchors = np.arange(n_anchors, dtype=int)
    remainder = np.arange(n_anchors, n_samples, dtype=int)
    perm = rng.permutation(remainder) if remainder.size else remainder
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
    n_train_rest = max(0, n_train - n_anchors)
    train_idx = np.concatenate([anchors, perm[:n_train_rest]]) if n_anchors else perm[:n_train]
    val_idx = perm[n_train_rest:n_train_rest + n_val]
    test_idx = perm[n_train_rest + n_val:]
    return train_idx, val_idx, test_idx


def generate_dataset():
    x_lhs = sample_designs(config.N_SAMPLES, config.DESIGN_RANGES, config.SEED)

    corner_anchors = []
    if bool(getattr(config, "CORNER_ANCHORS", False)):
        params = list(config.DESIGN_RANGES.keys())
        lows = [config.DESIGN_RANGES[p][0] for p in params]
        highs = [config.DESIGN_RANGES[p][1] for p in params]
        for mask in range(1 << len(params)):
            corner = []
            for i in range(len(params)):
                corner.append(highs[i] if (mask & (1 << i)) else lows[i])
            corner_anchors.append(corner)

    trend_anchors = []
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
            trend_anchors = anchors.tolist()

    anchor_blocks = []
    if corner_anchors:
        anchor_blocks.append(np.asarray(corner_anchors, dtype=float))
    if trend_anchors:
        anchor_blocks.append(np.asarray(trend_anchors, dtype=float))
    x_anchor = np.vstack(anchor_blocks) if anchor_blocks else np.zeros((0, x_lhs.shape[1]), dtype=float)
    n_anchors = int(x_anchor.shape[0])

    x_raw = np.vstack([x_anchor, x_lhs]) if n_anchors > 0 else x_lhs
    y_raw = np.zeros(x_raw.shape[0], dtype=float)
    for i, mu in enumerate(x_raw):
        y_raw[i] = baseline.compute_response(mu)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"Baseline {i + 1}/{x_raw.shape[0]}: y={y_raw[i]:.6f}")
    x_norm, x_min, x_max = normalize_inputs(x_raw, config.DESIGN_RANGES)
    y_norm, y_min, y_max, y_transform, y_eps = normalize_outputs(y_raw)
    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "y_transform": y_transform,
        "y_eps": y_eps,
        "param_names": list(config.DESIGN_RANGES.keys()),
        "n_anchors": int(n_anchors),
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
        y_transform=np.array(dataset.get("y_transform", "identity"), dtype=object),
        y_eps=float(dataset.get("y_eps", 1e-12)),
        param_names=np.array(dataset["param_names"], dtype=object),
        n_anchors=int(dataset.get("n_anchors", 0)),
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
        "y_transform": str(blob.get("y_transform", "identity")).strip(),
        "y_eps": float(blob.get("y_eps", 1e-12)),
        "param_names": list(blob["param_names"]),
        "n_anchors": int(blob.get("n_anchors", 0)),
    }
