from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from cad_geometry import _read_stl_triangles


@dataclass(frozen=True)
class TessellatedSurface:
    triangles: np.ndarray  # (T, 3, 3) float64
    tri_normals: np.ndarray  # (T, 3) float64, unit normals
    tri_areas: np.ndarray  # (T,) float64
    bounds_min: np.ndarray  # (3,) float64
    bounds_max: np.ndarray  # (3,) float64

    @property
    def bounds_size(self) -> np.ndarray:
        return self.bounds_max - self.bounds_min


def load_stl_surface(stl_path: str | Path) -> TessellatedSurface:
    tri = _read_stl_triangles(stl_path).astype(np.float64, copy=False)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"Expected triangles shaped (T,3,3), got {tri.shape}")

    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_unit = n / np.maximum(n_norm, 1e-18)
    areas = 0.5 * n_norm[:, 0]

    bounds_min = tri.reshape(-1, 3).min(axis=0)
    bounds_max = tri.reshape(-1, 3).max(axis=0)
    return TessellatedSurface(
        triangles=tri,
        tri_normals=n_unit,
        tri_areas=areas,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def affine_map_surface_to_bounds(
    surface: TessellatedSurface,
    dst_min_xyz: Tuple[float, float, float],
    dst_max_xyz: Tuple[float, float, float],
) -> TessellatedSurface:
    src_min = surface.bounds_min
    src_size = np.maximum(surface.bounds_size, 1e-12)
    dst_min = np.asarray(dst_min_xyz, dtype=np.float64)
    dst_max = np.asarray(dst_max_xyz, dtype=np.float64)
    dst_size = np.maximum(dst_max - dst_min, 1e-12)

    tri = surface.triangles
    tri_mapped = (tri - src_min[None, None, :]) / src_size[None, None, :] * dst_size[None, None, :] + dst_min[None, None, :]

    # Recompute normals/areas after affine map
    v0 = tri_mapped[:, 0, :]
    v1 = tri_mapped[:, 1, :]
    v2 = tri_mapped[:, 2, :]
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_unit = n / np.maximum(n_norm, 1e-18)
    areas = 0.5 * n_norm[:, 0]

    bounds_min = tri_mapped.reshape(-1, 3).min(axis=0)
    bounds_max = tri_mapped.reshape(-1, 3).max(axis=0)
    return TessellatedSurface(
        triangles=tri_mapped,
        tri_normals=n_unit,
        tri_areas=areas,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def _sample_triangle_points(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, n: int) -> np.ndarray:
    # Uniform points on a triangle using barycentric coordinates.
    r1 = np.random.rand(n, 1)
    r2 = np.random.rand(n, 1)
    s1 = np.sqrt(r1)
    return v0 * (1.0 - s1) + v1 * (1.0 - r2) * s1 + v2 * r2 * s1


def sample_boundary(surface: TessellatedSurface, nr_points: int) -> Dict[str, np.ndarray]:
    if nr_points <= 0:
        raise ValueError("nr_points must be > 0")

    areas = surface.tri_areas
    if not np.all(np.isfinite(areas)) or float(np.sum(areas)) <= 0.0:
        raise ValueError("Mesh triangle areas are invalid; check STL.")

    prob = areas / np.sum(areas)
    tri_idx = np.random.choice(np.arange(len(prob)), size=nr_points, p=prob)

    points = np.empty((nr_points, 3), dtype=np.float64)
    normals = np.empty((nr_points, 3), dtype=np.float64)
    for t in np.unique(tri_idx):
        mask = tri_idx == t
        n_local = int(np.sum(mask))
        v0, v1, v2 = surface.triangles[t]
        points[mask] = _sample_triangle_points(v0, v1, v2, n_local)
        normals[mask] = surface.tri_normals[t]

    area_per_point = float(np.sum(areas)) / float(nr_points)
    out = {
        "x": points[:, 0:1],
        "y": points[:, 1:2],
        "z": points[:, 2:3],
        "normal_x": normals[:, 0:1],
        "normal_y": normals[:, 1:2],
        "normal_z": normals[:, 2:3],
        "area": np.full((nr_points, 1), area_per_point, dtype=np.float64),
    }
    return out


def _ray_intersects_triangle_moller_trumbore(
    ray_o: np.ndarray, ray_d: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    # Vectorized per-point ray/triangle intersection; returns boolean mask of hits (t > 0).
    # ray_o, ray_d: (N,3), v0/v1/v2: (3,)
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(ray_d, e2[None, :])
    det = np.einsum("ij,j->i", pvec, e1)
    det_mask = np.abs(det) > eps

    inv_det = np.zeros_like(det)
    inv_det[det_mask] = 1.0 / det[det_mask]

    tvec = ray_o - v0[None, :]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    u_mask = (u >= 0.0) & (u <= 1.0) & det_mask

    qvec = np.cross(tvec, e1[None, :])
    v = np.einsum("ij,j->i", qvec, ray_d[0]) * inv_det  # ray_d constant across points
    v_mask = (v >= 0.0) & ((u + v) <= 1.0) & u_mask

    t = np.einsum("ij,j->i", qvec, e2) * inv_det
    return v_mask & (t > eps)


def points_inside_mesh(surface: TessellatedSurface, points: np.ndarray) -> np.ndarray:
    """
    Odd-even rule using a +x ray cast. Works best for watertight meshes.
    For very complex meshes you may want a BVH; this is intended for "simple CAD" first.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    # Small jitter reduces edge/vertex degeneracy.
    jitter = (np.random.rand(len(pts), 3) - 0.5) * 1e-9
    ray_o = pts + jitter
    ray_d = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float64), (len(pts), 1))

    hits = np.zeros(len(pts), dtype=np.int32)
    for tri in surface.triangles:
        v0, v1, v2 = tri
        hit = _ray_intersects_triangle_moller_trumbore(ray_o, ray_d, v0, v1, v2)
        hits += hit.astype(np.int32)
    return (hits % 2) == 1


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # From "Real-Time Collision Detection" (Christer Ericson).
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1e-18)
        return a + v * ab

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1e-18)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-18)
        return b + w * (c - b)

    denom = max(va + vb + vc, 1e-18)
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


def unsigned_distance_and_dir(surface: TessellatedSurface, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    min_d2 = np.full(len(pts), np.inf, dtype=np.float64)
    closest = np.zeros((len(pts), 3), dtype=np.float64)

    for tri in surface.triangles:
        a, b, c = tri
        for i, p in enumerate(pts):
            q = _closest_point_on_triangle(p, a, b, c)
            d2 = float(np.dot(p - q, p - q))
            if d2 < min_d2[i]:
                min_d2[i] = d2
                closest[i] = q

    dist = np.sqrt(np.maximum(min_d2, 0.0))
    direction = pts - closest
    direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-18)
    return dist.reshape(-1, 1), direction


def sample_interior(
    surface: TessellatedSurface,
    nr_points: int,
    bounds_min: Tuple[float, float, float] | None = None,
    bounds_max: Tuple[float, float, float] | None = None,
    compute_sdf_derivatives: bool = True,
    max_batches: int = 200,
) -> Dict[str, np.ndarray]:
    if bounds_min is None:
        bounds_min = tuple(map(float, surface.bounds_min))
    if bounds_max is None:
        bounds_max = tuple(map(float, surface.bounds_max))

    bmin = np.asarray(bounds_min, dtype=np.float64)
    bmax = np.asarray(bounds_max, dtype=np.float64)
    if np.any(bmax <= bmin):
        raise ValueError("Invalid bounds for interior sampling.")

    pts_out = []
    sdf_out = []
    sdf_dir_out = []
    tries = 0
    # Oversample and reject points outside the mesh.
    while sum(p.shape[0] for p in pts_out) < nr_points:
        if tries >= max_batches:
            raise RuntimeError("Could not sample enough interior points; check STL watertightness/scale.")
        tries += 1
        n_batch = max(1024, nr_points)
        u = np.random.rand(n_batch, 3)
        pts = bmin[None, :] + u * (bmax - bmin)[None, :]
        inside = points_inside_mesh(surface, pts)
        pts = pts[inside]
        if pts.shape[0] == 0:
            continue

        dist, direction = unsigned_distance_and_dir(surface, pts)
        # PhysicsNeMo convention: positive inside for CSG; tessellation returns "sdf" used with >0 interior.
        # We follow that here: sdf = +distance for inside points.
        sdf = dist

        pts_out.append(pts)
        sdf_out.append(sdf)
        sdf_dir_out.append(direction)

    pts_all = np.concatenate(pts_out, axis=0)[:nr_points]
    sdf_all = np.concatenate(sdf_out, axis=0)[:nr_points]
    dir_all = np.concatenate(sdf_dir_out, axis=0)[:nr_points]

    out = {
        "x": pts_all[:, 0:1],
        "y": pts_all[:, 1:2],
        "z": pts_all[:, 2:3],
        "sdf": sdf_all,
        "area": np.full((nr_points, 1), float(np.prod(bmax - bmin)) / float(nr_points), dtype=np.float64),
    }
    if compute_sdf_derivatives:
        out["sdf__x"] = dir_all[:, 0:1]
        out["sdf__y"] = dir_all[:, 1:2]
        out["sdf__z"] = dir_all[:, 2:3]
    return out
