"""Tiny generated-only mathematical controls; no patient source or task execution."""
from itertools import product
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from resectionlab.geometry import GENERIC_TOOLS, GeometryScene, point_segment_distances
from resectionlab.native_proposals import DEFAULT_COLUMN_OFFSETS
from resectionlab.native_resection import contained_capsule_cells

OUT = Path(__file__).resolve().parent
CUSHION = 1e-6


def scene(basis, translation=(0., 0., 0.)):
    affine = np.eye(4)
    affine[:3, :3] = basis
    affine[:3, 3] = translation
    return GeometryScene(np.zeros((4, 4, 4), bool), affine)


def endpoint(geometry, value):
    return geometry.affine[:3, :3] @ np.asarray(value) + geometry.affine[:3, 3]


def corners(geometry, cell):
    return endpoint(geometry, cell) + np.asarray(tuple(product((-.5, .5), repeat=3))) @ geometry.affine[:3, :3].T


def inside(geometry, cell, start, end, radius):
    return bool(len(contained_capsule_cells(geometry, np.asarray([cell]), start, end, radius)))


def main():
    records = []
    rotation = np.array([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., -1.]])
    # A cell between four columns is center-covered by the wider tip, yet its
    # farthest corners do not fit any of the 13 columns with either generic tip.
    for name, basis, translation in (
        ("identity", np.eye(3), (0., 0., 0.)),
        ("oblique_reflected_anisotropic", rotation @ np.diag([1.1, .8, 1.2]), (20., -5., 8.)),
    ):
        geometry = scene(basis, translation)
        for tool in GENERIC_TOOLS:
            outcomes = []
            for cell in ([1, 0, 0], [1, 1, 1]):
                maxima = []
                for a, b in DEFAULT_COLUMN_OFFSETS:
                    start, end = endpoint(geometry, [-10, a, b]), endpoint(geometry, [10, a, b])
                    maxima.append(float(point_segment_distances(corners(geometry, cell), start, end).max()))
                nominal = min(maxima) <= tool.tip_radius_mm - 1e-10
                enlarged = min(maxima) <= tool.tip_radius_mm + CUSHION
                outcomes.append({"cell": cell, "minimum_over_13_farthest_corner_distance_mm": min(maxima),
                    "native_capacity": nominal, "enlarged_capacity": enlarged})
            assert outcomes[0]["native_capacity"] and not outcomes[1]["enlarged_capacity"]
            records.append({"control": "all_13_axis_cell_corners", "frame": name, "tool": tool.tool_id, "outcomes": outcomes})

    # Actual generic radii are retained; generated transverse cell sizes put
    # the nearest whole-cell radial boundary at r, r+.5um, or r+2um.
    for tool in GENERIC_TOOLS:
        for delta in (0., .5e-6, 2e-6):
            spacing = (tool.tip_radius_mm + delta) / np.sqrt(4.5)
            geometry = scene(rotation @ np.diag([1., spacing, spacing]), (20., -5., 8.))
            cell = [1, 1, 1]
            start, end = endpoint(geometry, [-10, 0, 0]), endpoint(geometry, [10, 0, 0])
            nominal = inside(geometry, cell, start, end, tool.tip_radius_mm)
            enlarged = inside(geometry, cell, start, end, tool.tip_radius_mm + CUSHION + 1e-10)
            assert not nominal and enlarged == (delta < CUSHION)
            records.append({"control": "tangent_boundary", "tool": tool.tool_id, "radial_excess_mm": delta,
                "native_capacity": nominal, "enlarged_capacity": enlarged})

    geometry = scene(np.eye(3))
    cell, start, end = [3, 0, 0], np.array([-3., 0., 0.]), np.array([1., 0., 0.])
    finite = inside(geometry, cell, start, end, GENERIC_TOOLS[1].tip_radius_mm)
    radial = float(np.linalg.norm(corners(geometry, cell)[:, 1:], axis=1).max())
    assert not finite and radial < GENERIC_TOOLS[1].tip_radius_mm
    records.append({"control": "infinite_cylinder_cannot_prove_finite_capsule", "infinite_capacity": True, "finite_capacity": finite})

    geometry = scene(np.diag([8., .5, .5]))
    cell, radius = [1, 0, 0], GENERIC_TOOLS[0].tip_radius_mm
    full = inside(geometry, cell, np.array([-3., 0., 0.]), np.array([16., 0., 0.]), radius)
    micro = []
    for step in range(65):
        current, previous = step / 4., max(0., (step - 1) / 4.)
        micro.append(inside(geometry, cell, np.array([previous - 3., 0., 0.]), np.array([current, 0., 0.]), radius))
    assert full and not any(micro)
    records.append({"control": "full_sweep_pass_does_not_prove_microstep_containment", "full_capacity": full,
        "microstep_capsules_checked": len(micro), "microstep_capacity": any(micro)})

    sources = ["src/resectionlab/native_resection.py", "src/resectionlab/geometry.py",
        "src/resectionlab/native_proposals.py", "src/resectionlab/post_exposure.py",
        "build/select013-dead-frontier-witness-v1/blocker_coverage.py"]
    result = {"scope": "generated numerical controls only; no patient arrays/images/checkpoints or science run",
        "passed": True, "records": records,
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}}
    (OUT / "generated-control-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "controls": len(records), "output": str(OUT / "generated-control-results.json")}))


if __name__ == "__main__":
    main()
