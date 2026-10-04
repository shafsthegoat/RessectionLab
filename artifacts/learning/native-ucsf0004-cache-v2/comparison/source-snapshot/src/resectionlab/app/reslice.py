"""Physical RAS multiplanar sampling, independent of Qt and VTK.

No voxel reorientation is assumed. These routines sample the source affine
directly, including oblique and anisotropic acquisitions. Views use neurological
convention: L at the left of axial/coronal views, with A/S at the top.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.ndimage import map_coordinates


PLANE_AXES = {"axial": (0, 1, 2), "coronal": (0, 2, 1), "sagittal": (1, 2, 0)}
PLANE_LABELS = {
    "axial": ("L", "R", "A", "P"),
    "coronal": ("L", "R", "S", "I"),
    "sagittal": ("P", "A", "S", "I"),
}


def ras_affine(affine: np.ndarray, frame: str = "RAS+") -> np.ndarray:
    """Return the voxel-to-RAS-mm affine for an explicitly declared frame."""
    affine = np.asarray(affine, dtype=float)
    if frame in ("RAS", "RAS+"):
        return affine
    if frame in ("LPS", "LPS+"):
        return np.diag([-1., -1., 1., 1.]) @ affine
    raise ValueError(f"Unsupported coordinate frame: {frame}")


def world_bounds(shape: tuple[int, ...], affine: np.ndarray) -> np.ndarray:
    """Return RAS bounds of the extreme voxel centers (the sampling domain)."""
    corners = np.array(list(product(*[(0, n - 1) for n in shape[:3]])))
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    return np.stack([world.min(axis=0), world.max(axis=0)])


@dataclass(frozen=True)
class SliceGeometry:
    plane: str
    bounds: np.ndarray
    position_mm: float
    width: int
    height: int

    @property
    def axes(self) -> tuple[int, int, int]:
        return PLANE_AXES[self.plane]

    @property
    def physical_size(self) -> tuple[float, float]:
        a, b, _ = self.axes
        return (float(np.ptp(self.bounds[:, a])), float(np.ptp(self.bounds[:, b])))

    def world_at(self, horizontal: float, vertical: float) -> np.ndarray:
        """Coordinates for normalized widget position (top left is 0,0)."""
        a, b, c = self.axes
        point = np.zeros(3)
        point[a] = self.bounds[0, a] + horizontal * np.ptp(self.bounds[:, a])
        point[b] = self.bounds[1, b] - vertical * np.ptp(self.bounds[:, b])
        point[c] = self.position_mm
        return point


def slice_geometry(shape, affine, cursor_mm, plane, max_pixels=384) -> SliceGeometry:
    bounds = world_bounds(shape, affine)
    a, b, c = PLANE_AXES[plane]
    extents = np.maximum(bounds[1] - bounds[0], 1e-3)
    scale = (max_pixels - 1) / max(extents[a], extents[b])
    return SliceGeometry(plane, bounds, float(cursor_mm[c]),
                         max(2, int(round(extents[a] * scale)) + 1),
                         max(2, int(round(extents[b] * scale)) + 1))


def sample_slice(data: np.ndarray, affine: np.ndarray, geometry: SliceGeometry,
                 *, order: int = 1) -> np.ndarray:
    a, b, c = geometry.axes
    horizontal = np.linspace(geometry.bounds[0, a], geometry.bounds[1, a], geometry.width)
    vertical = np.linspace(geometry.bounds[1, b], geometry.bounds[0, b], geometry.height)
    xx, yy = np.meshgrid(horizontal, vertical)
    world = np.empty((3, geometry.height * geometry.width))
    world[a], world[b], world[c] = xx.ravel(), yy.ravel(), geometry.position_mm
    inverse = np.linalg.inv(affine)
    voxel = inverse[:3, :3] @ world + inverse[:3, 3, None]
    # Rounding noise at exact volume boundaries must not turn valid pixels black.
    for axis, length in enumerate(data.shape):
        voxel[axis, np.isclose(voxel[axis], 0, atol=1e-7)] = 0
        voxel[axis, np.isclose(voxel[axis], length - 1, atol=1e-7)] = length - 1
    return map_coordinates(data, voxel, order=order, mode="constant", cval=0,
                           prefilter=False).reshape(geometry.height, geometry.width)


def window_to_rgb(values: np.ndarray, low: float, high: float) -> np.ndarray:
    gray = np.clip((values - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)
