"""Small explicit physical brush edits; source annotations remain immutable."""

import numpy as np


def edit_sphere(mask, affine_ras_mm, center_ras_mm, radius_mm, *, add):
    """Return a new mask edited in a sphere measured in physical millimeters.

    This is a voxel-center annotation brush, not a tissue removal primitive.
    The bounded crop supports oblique and sheared source grids.
    """
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("Brush radius must be finite and positive")
    center = np.asarray(center_ras_mm, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Brush center must contain three finite RAS coordinates")
    affine = np.asarray(affine_ras_mm, dtype=float)
    inverse = np.linalg.inv(affine)
    voxel_center = inverse[:3, :3] @ center + inverse[:3, 3]
    extent = radius_mm * np.linalg.norm(inverse[:3, :3], axis=1)
    lower = np.maximum(np.floor(voxel_center - extent).astype(int), 0)
    upper = np.minimum(np.ceil(voxel_center + extent).astype(int) + 1, mask.shape)
    result = np.array(mask, dtype=bool, copy=True)
    if np.any(upper <= lower):
        return result
    axes = [np.arange(lo, hi) for lo, hi in zip(lower, upper)]
    voxel = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    world = voxel @ affine[:3, :3].T + affine[:3, 3]
    selected = np.sum((world - center) ** 2, axis=-1) <= radius_mm ** 2 + 1e-10
    region = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
    result[region][selected] = bool(add)
    return result
