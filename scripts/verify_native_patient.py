"""Real local collection integrity and geometric handoff receipt; no accuracy scores."""

import argparse
import json
from pathlib import Path
import numpy as np
from resectionlab.patient_anatomy import (
    PatientImagingCollection,
    PatientAnatomySnapshot,
    file_hash,
    world,
)
from resectionlab.anatomy_geometry import mesh
from resectionlab.anatomy_planning import removal_preview
from resectionlab.imaging import load_case

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("collection", type=Path)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
c = PatientImagingCollection(args.collection)
snapshot = PatientAnatomySnapshot.capture(c)
sources = []
for s in c.document["series"]:
    actual = file_hash(c.path(s["original"]["path"]))
    assert actual == s["original"]["sha256"]
    sources.append(
        {
            "id": s["id"],
            "sha256": actual,
            "shape": s["data"]["shape"],
            "grid_mm": s["grid_spacing_mm"],
        }
    )
vessel = c.layer("vessels")
indices = np.argwhere(c.mask(vessel))
point = world(c.patient_affine(vessel), indices[len(indices) // 2])
clearance = snapshot.clearance(point - [2, 0, 0], point + [2, 0, 0], 0.5)
assert clearance["constraints"]["vessels"]["intersects_or_unresolved"]
tumor = c.layer("tumor")
indices = np.argwhere(c.mask(tumor))
centre = world(c.patient_affine(tumor), indices[len(indices) // 2])
removed = removal_preview(snapshot, centre, 2.0)
before = snapshot.simulation_grid()
after = snapshot.simulation_grid(removed=removed)
assert after["targets"]["tumor"].sum() < before["targets"]["tumor"].sum()
assert np.array_equal(after["critical"]["vessels"], before["critical"]["vessels"])
geometry = {}
for name in ["tumor", "vessels", "brain"]:
    row = c.layer(name)
    h = row["data"]["sha256"]
    g = mesh(c, row)
    assert c.layer(name)["data"]["sha256"] == h
    geometry[name] = {
        "vertices": len(g["positions"]),
        "faces": len(g["indices"]),
        "max_displacement_mm": float(g["max_displacement_mm"]),
        "source_degenerate_faces_removed": int(
            g.get("source_degenerate_faces_removed", 0)
        ),
        "smoothing_rejected": bool(g.get("smoothing_rejected", False)),
    }
baseline = load_case(args.baseline)
receipt = {
    "snapshot_id": snapshot.snapshot_id,
    "sources": sources,
    "geometry": geometry,
    "vessel_crossing_query": clearance,
    "removal_preview": {
        "before_target_voxels": int(before["targets"]["tumor"].sum()),
        "preview_target_voxels": int(after["targets"]["tumor"].sum()),
        "vessels_unchanged": True,
        "source_snapshot_unchanged": c.identity() == snapshot.snapshot_id,
    },
    "old_bundle": {
        "path": str(args.baseline),
        "sha256": file_hash(args.baseline),
        "loaded_case": baseline.case_id,
        "shape": baseline.mri.shape,
    },
    "accuracy": None,
    "accuracy_reason": "no source tumor/vessel reference annotations available; independent visual review remains required",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
print(
    json.dumps(
        {
            "output": str(args.output),
            "snapshot_id": snapshot.snapshot_id,
            "passed": True,
        }
    )
)
