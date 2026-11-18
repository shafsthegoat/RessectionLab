#!/usr/bin/env python3
"""Build a generated, exposed surface control for the ordinary Cases → Rehearse workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_sequence_outcomes import sequence_outcome

SPACING = (0.8, 0.9, 1.0)
SHAPE = (64, 56, 42)
ORIGIN = (-25.6, -25.2, 0.0)
TISSUE, VESSEL, MEMBRANE, TARGET = (
    "Folded tissue control",
    "Bifurcating vessel control",
    "Membrane patch",
    "Two-component target",
)
SCOPE = (
    "Generated exposed field for display and contact checks. No patient scan, "
    "skull, scalp, operative access qualification, or functional evidence."
)


def near_surface(x, y):
    """Inferior ellipsoid with one explicit, smooth analytic groove in mm."""
    cap = np.sqrt(np.maximum(0.0, 1.0 - (x / 19.0) ** 2 - (y / 17.0) ** 2))
    groove = 3.8 * np.exp(-(((x + 2.0 - 1.6 * np.sin(y / 6.0)) / 1.9) ** 2) - (y / 11.0) ** 4)
    return 21.0 - 12.0 * cap + groove


def branch_point(xy):
    x, y = xy
    return (x, y, float(near_surface(x, y)) - 0.55)


def build_case():
    affine = np.diag([*SPACING, 1.0])
    affine[:3, 3] = ORIGIN
    x, y, z = np.meshgrid(*[np.arange(n) * s + o for n, s, o in zip(SHAPE, SPACING, ORIGIN)], indexing="ij")
    radial = (x / 19.0) ** 2 + (y / 17.0) ** 2
    tissue = (
        (radial <= 1.0)
        & (z >= near_surface(x, y))
        & (z <= 21.0 + 12.0 * np.sqrt(np.maximum(0.0, 1.0 - radial)))
    )
    # Distances are measured in physical millimetres, including through-plane distance.
    junction = (-8.0, 0.0)
    tips = ((-7.0, -12.0), (-13.0, 7.0), (-6.0, 10.0))
    paths = []
    vessel = np.zeros(SHAPE, bool)
    xyz = np.stack([x, y, z], axis=-1)
    radii = (1.35, 1.05, 0.95)
    for tip, radius in zip(tips, radii):
        xy = np.linspace(junction, tip, 161)
        curve = np.array([branch_point(p) for p in xy])
        vessel |= cKDTree(curve).query(xyz, distance_upper_bound=radius)[0] <= radius
        paths.append(
            {
                "tipRasMm": list(curve[-1]),
                "radiusMm": radius,
                "maximumCenterlineSampleSpacingMm": float(
                    np.linalg.norm(np.diff(curve, axis=0), axis=1).max()
                ),
            }
        )
    # Exactly one occupied z cell in every supported vertical column of the membrane.
    membrane = (((x - 2.0) / 5.0) ** 2 + ((y + 8.0) / 3.2) ** 2 <= 1.0) & (z == 11.0)
    centers = [(6.4, 1.8, 10.0), (10.4, 7.2, 12.0)]
    axes = [(4.1, 3.3, 3.5), (1.65, 1.6, 1.7)]
    target = np.zeros(SHAPE, bool)
    for center, radius in zip(centers, axes):
        target |= sum(((coord - c) / r) ** 2 for coord, c, r in zip((x, y, z), center, radius)) <= 1.0
    tissue &= ~(target | membrane | vessel)
    masks = {TISSUE: tissue, VESSEL: vessel, MEMBRANE: membrane, TARGET: target}
    declarations = [
        (TISSUE, "brain", "#bf8883"),
        (VESSEL, "vessel", "#b44951"),
        (MEMBRANE, "membrane", "#a894bc"),
        (TARGET, "tumor", "#edb443"),
    ]
    signal = np.zeros(SHAPE, np.float32)
    for i, mask in enumerate(masks.values(), 1):
        signal[mask] = 0.2 * i
    case = CaseData(
        case_id="generated-curved-field-v1",
        mri=signal,
        compartments=masks,
        affine=affine,
        source_refs=(SourceRef("generated", "synthetic://curved-field-v1", provenance="simulated"),),
        unknowns=(SCOPE,),
        metadata={
            "generated": True,
            "is_synthetic": True,
            "surgical_adapter": {"type": "generated-engineering-control", "ownershipPrecedence": list(masks)},
            "anatomy_layers": {
                name: {"kind": kind, "color": color, "review_status": "generated-engineering"}
                for name, kind, color in declarations
            },
        },
    )
    topology = {}
    for name, mask in masks.items():
        cc, count = label(mask)
        sizes = np.bincount(cc[mask], minlength=count + 1)[1:]
        topology[name] = dict(
            hash=array_digest(mask),
            faceConnectedComponents=int(count),
            componentCells=sorted(map(int, sizes), reverse=True),
            volumeMm3=float(mask.sum() * np.prod(SPACING)),
        )
    truth = dict(
        scope=SCOPE,
        sourceSpacingMm=SPACING,
        sourceShape=SHAPE,
        affine=affine.tolist(),
        layers=topology,
        vessel=dict(junctionRasMm=branch_point(junction), branches=paths),
        membraneMaximumCellsPerColumn=int(membrane.sum(axis=2).max()),
        targetEllipsoids=[
            dict(centerRasMm=c, radiiMm=r, analyticVolumeMm3=float(4 * np.pi * np.prod(r) / 3))
            for c, r in zip(centers, axes)
        ],
        groove=dict(centerlineAtY0Mm=-2.0, amplitudeMm=3.8, transverseScaleMm=1.9, longitudinalScaleMm=11.0),
    )
    return case, truth


def generate(output):
    case, truth = build_case()
    scene = from_case(case)
    engine = SurgicalEngine(scene)
    actions = [
        dict(
            operation="place",
            pose=dict(position=[6.4, 1.8, 2.0], quaternion=[0.0, 0.0, 0.0, 1.0]),
            parameters=dict(radiusMm=0.9),
        ),
        dict(operation="apply", pose=dict(position=[6.4, 1.8, 8.0], quaternion=[0.0, 0.0, 0.0, 1.0])),
    ]
    for action in actions:
        result = engine.step(dict(slot="right", duration=0.2, **action))
        if result["status"] != "accepted":
            raise RuntimeError(result)
    before_rejection = engine.state_id
    rejection = engine.step(
        dict(slot="right", operation="exchange", instrument="cranial_drill", duration=0.2)
    )
    if rejection["status"] != "rejected" or engine.state_id != before_rejection:
        raise RuntimeError("Inserted exchange must refuse without changing the state")
    finish = engine.step(dict(slot="right", operation="withdraw", duration=0.2))
    if finish["status"] != "accepted":
        raise RuntimeError(finish)
    branch = "curved-contact"
    record = dict(
        mapping=scene.regions,
        branches={branch: engine.snapshot()},
        branchId=branch,
        position=0,
        parents={branch: None},
        timeline={branch: engine.history},
        researchSetup=dict(kind="generated-curved-optical-control", description=SCOPE),
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "curved-field.ressectionlab"
    if path.exists():
        if (
            load_case(path).semantic_hash != case.semantic_hash
            or read_case_artifacts(path)["surgical_toolbox"] != record
        ):
            raise ValueError(f"Different case already exists at {path}")
    else:
        save_case(case, path, artifacts={"surgical_toolbox": record})
    truth.update(
        caseHash=case.semantic_hash,
        endpointStateId=engine.state_id,
        accepted=engine.history,
        rejection=dict(
            status=rejection["status"], reasonCode=rejection["reasonCode"], stateId=before_rejection
        ),
        endpointLedger=sequence_outcome(engine),
    )
    (output / "geometry-and-actions.json").write_text(json.dumps(truth, indent=2, allow_nan=False) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(generate(parser.parse_args().output))
