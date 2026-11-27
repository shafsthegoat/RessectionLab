"""Build an independent, layered generated convexity case for operative rehearsal.

Coordinates are RAS millimetres. The 1 mm working grid and compact ellipsoid
describe a teaching phantom. No patient image, contour or property is input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resectionlab.core import CaseData, SourceRef, thaw_json
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.surgical_scene import from_case

CASE_ID = "generated-convexity-glioma-v1"
SHAPE = (108, 106, 127)
ORIGIN = (-30, -28, -16)
CENTER_XY = (32, 32)
SCOPE = ("Independent generated convexity phantom with synthetic image contrast, vessel identity, "
         "folds and mechanical assignments. Physiological mapping and patient tissue properties are unknown.")
# Source label IDs follow this declaration order.
LAYERS = (
    ("Scalp skin", "soft_tissue", "scalp", "#bd887d", 0.64),
    ("Galea", "soft_tissue", "galea", "#baaaa0", 0.57),
    ("Outer cortical table", "bone", "skull", "#d4c6ab", 0.17),
    ("Diploe", "bone", "diploe", "#a58b72", 0.31),
    ("Inner cortical table", "bone", "skull", "#d3c2ab", 0.17),
    ("Dura", "membrane", "dura", "#ccbac0", 0.43),
    ("Cortical gray matter", "soft_tissue", "grayMatter", "#bf8f8c", 0.52),
    ("Subcortical white matter", "soft_tissue", "whiteMatter", "#ccc5ae", 0.67),
    ("Glioma margin", "tumor", "tumor", "#b28e86", 0.60),
    ("Glioma core", "tumor", "tumor", "#a88a82", 0.57),
    ("Cortical artery", "vessel", "artery", "#aa5151", 0.73),
    ("Cortical vein", "vessel", "vein", "#785f80", 0.35),
    ("Scalp arterial branch", "vessel", "artery", "#a25854", 0.71),
)


def world(ijk):
    return [int(i + o) for i, o in zip(ijk, ORIGIN)]


def voxel(xyz):
    return tuple(int(v-o) for v, o in zip(xyz, ORIGIN))


def digital_line(points):
    """Connect successive sampled centers by whole face-neighbour steps."""
    out = [tuple(points[0])]
    for point in points[1:]:
        here = np.array(out[-1])
        # The shortest monotone stair-step joins a sample without skipped cells.
        while np.any(here != point):
            gap = np.asarray(point) - here
            axis = int(np.argmax(abs(gap)))
            here[axis] += np.sign(gap[axis])
            out.append(tuple(map(int, here)))
    return out


def _raster():
    if __package__:
        from .generate_curved_render_case import near_surface
    else:
        from generate_curved_render_case import near_surface

    x, y, z = np.ogrid[tuple(slice(o, o+n) for n, o in zip(SHAPE, ORIGIN))]
    x, y, z = (t.astype(np.float32) for t in (x, y, z))
    r = np.hypot(x-32, y-32)
    ellipsoid = np.sqrt(((x-17)/44)**2 + ((y-24)/49)**2 + ((z-28)/42)**2)
    depth = (1-ellipsoid)*34
    # A shallow crown at the operative site transitions continuously into
    # the ellipsoid. The compound roof can be resolved with three bone cells.
    t = np.clip((27-r)/11, 0, 1)
    crown_blend = (t*t*(3-2*t)) * np.clip((z-40)/7, 0, 1)
    depth = depth*(1-crown_blend) + (66.30-z-0.0007*r*r)*crown_blend
    labels = np.zeros(SHAPE, dtype=np.uint16)
    for number, low, high in [(1, 0, 1.45), (2, 1.45, 2.6), (3, 2.6, 3.6),
                              (4, 3.6, 4.6), (5, 4.6, 5.6), (6, 6.75, 7.78)]:
        labels[(depth >= low) & (depth < high)] = number

    # Reuse the smooth inferior-surface groove from the existing curved
    # rendering control, reflected onto a superior convexity. The original
    # ellipsoid term is removed so only its millimetre-valued sulcus remains.
    gx, gy = x-28, y-32
    cap = np.sqrt(np.maximum(0, 1-(gx/19)**2-(gy/17)**2))
    primary_groove = near_surface(gx, gy) - (21 - 12*cap)
    folds = np.exp(-((np.sin((x+0.35*y+2.5*np.sin(y/8))/3.5))/0.39)**2)
    fissure = 6*np.exp(-((x-16-0.7*np.sin(y/13))/1.5)**2)
    cortex_depth = 9.1 + 2.5*folds + primary_groove + fissure
    labels[depth >= cortex_depth] = 7
    labels[depth >= cortex_depth+3.0] = 8

    # Vary the envelope continuously and retain its mixed boundary with
    # healthy tissue. Both categorical target zones belong to one lesion.
    dx, dy, dz = x-32, y-32, z-52
    shape = ((dx-0.23*dy)/6.4)**2 + ((dy+0.18*dz)/5.2)**2 + (dz/4.0)**2
    ripple = 0.15*np.sin(dx*0.7 + dy*0.3) + 0.12*np.cos(dy*0.8 - dz*0.5)
    target = (shape < 1+ripple) & np.isin(labels, [7, 8])
    core = ((dx-0.8-0.25*dy)/3.4)**2 + (dy/2.8)**2 + ((dz+0.4)/2.3)**2 < 1
    labels[target] = 9
    labels[target & core] = 10

    # Both centerlines follow the sampled pial surface. Keep their source
    # identities separate. A short branch joins each trunk by a shared cell.
    top = SHAPE[2]-1-np.argmax(np.isin(labels, [7, 8, 9, 10])[:, :, ::-1], axis=2)
    def at_surface(wx, wy):
        ix, iy, _ = voxel((wx, wy, 0))
        return (ix, iy, int(top[ix, iy]))
    ys = np.arange(12, 52)
    artery = digital_line([at_surface(round(38+1.5*np.sin((yi-31)/8)), yi) for yi in ys])
    vein = digital_line([at_surface(round(25+1.2*np.sin((yi-29)/7)), yi) for yi in ys])
    for rid, trunk, branch_sign in [(11, artery, 1), (12, vein, -1)]:
        start = next(p for p in trunk if world(p)[1] == 34)
        wx, wy, _ = world(start)
        branch = digital_line([at_surface(wx+round(branch_sign*i*.8), wy+i) for i in range(12)])
        centers = np.zeros(SHAPE, dtype=bool)
        centers[tuple(np.array(trunk+branch).T)] = True
        # Keep the thin vessel in contact with the native cortex, including at
        # its turns. Only brain or free sulcal space can be assigned a vessel.
        vessel = distance_transform_edt(~centers) <= (1.05 if rid == 11 else 1.0)
        # Seat the tubular trunk on the pia. Reserve the cell directly
        # beneath its axis for its source tissue attachment wherever the
        # subjacent cell belongs to cortex and is outside the centerline.
        below = tuple((np.array(trunk+branch) + [0, 0, -1]).T)
        vessel[below] &= centers[below]
        labels[vessel & np.isin(labels, [0, 7, 8, 9, 10])] = rid
    scalp_line = [voxel((32, y, 65)) for y in range(12, 17)]
    cells = tuple(np.asarray(scalp_line).T)
    if not np.isin(labels[cells], [1, 2]).all():
        raise ValueError("Scalp branch must lie on the retained, generated scalp")
    labels[cells] = 13
    # The smooth crown transition can straddle an entire skin interval on
    # an oblique digital face. An exposed scalp face must be skin, while inner
    # galea and the source-mapped operative incision remain distinct.
    outside = ~binary_fill_holes(labels != 0)
    labels[(labels == 2) & binary_dilation(outside)] = 1
    # A prospectively authored exophytic tongue provides a tissue sample.
    # The distal two cells remain face-connected to the main tumor through
    # their single native neck bond. The biopsy must sever that connection.
    tongue = [voxel(p) for p in [(34, 32, 56), (34, 32, 57), (34, 31, 57), (34, 30, 57)]]
    tongue_cells = tuple(np.array(tongue).T)
    if labels[tongue_cells].any() or labels[voxel((34, 32, 55))] != 9:
        raise ValueError("The superficial tumor protrusion requires a free sulcal pocket and a marginal stalk")
    labels[tongue_cells] = 9
    return labels, artery, vein


def _source_members(labels, ids, radius, hinge=False, *, center_y=32, major_y=None, hinge_y=None):
    xs, ys, _ = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    wx = xs+ORIGIN[0]-32
    wy = ys+ORIGIN[1]-center_y
    ry = radius if major_y is None else major_y
    crown = (wx*wx*ry*ry+wy*wy*radius*radius <= radius*radius*ry*ry)
    if hinge:
        crown &= wy <= (int(radius*.62) if hinge_y is None else hinge_y-center_y)
    mask = np.isin(labels, ids)
    # Only the superior roof belongs to the local mobile envelope.
    visible = np.arange(SHAPE[2])[None, None, :]+ORIGIN[2] > 45
    return {"cells": np.flatnonzero(mask & crown & visible).tolist(),
            "layerCells": np.flatnonzero(mask).tolist()}


def _pial_point(labels, x, y):
    i, j, _ = voxel((x, y, 0))
    cells = np.flatnonzero(np.isin(labels[i, j], [7, 8, 9, 10, 11, 12]))
    p = (i, j, int(cells[-1]))
    return world(p)


def generated_anatomy():
    """Sample the physical teaching field independently of its engine settings."""
    labels, artery, vein = _raster()
    affine = np.eye(4)
    affine[:3, 3] = ORIGIN
    fabric = {name: labels == i for i, (name, *_) in enumerate(LAYERS, 1)}
    index = np.array([0, *(row[4] for row in LAYERS)], dtype=np.float32)
    x, y, z = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    signal = index[labels]
    signal += (labels != 0) * (0.028*np.sin(.47*x+.19*z)*np.cos(.24*y)
                              + 0.019*np.sin(.25*x+.91*y+.33*z))
    regions = [dict(id=i, name=name, material=mat, color=color, target=mat=="tumor")
               for i, (name, mat, _, color, _) in enumerate(LAYERS, 1)]
    case = CaseData(case_id=CASE_ID, revision=2, affine=affine, mri=signal, compartments=fabric,
                    source_refs=(SourceRef("generated", "synthetic://convexity-glioma/v2", provenance="simulated"),),
                    unknowns=(SCOPE, "No physiological mapping evidence or calibrated tissue properties."),
                    metadata={"generated": True, "is_synthetic": True,
                              "surgical_adapter": {"type": "generated-engineering-control",
                                                   "ownershipPrecedence": list(fabric),
                                                   "materialByRegion": {r["name"]: r["material"] for r in regions}},
                              "anatomy_layers": {name: {"kind": kind, "color": color, "review_status": "generated-engineering"}
                                                 for name, _, kind, color, _ in LAYERS}})
    return case, labels, regions, artery, vein


def build_glioma_case():
    from dataclasses import replace
    from resectionlab.surgical_cusa_v3 import default_profile as cusa_profile
    from resectionlab.surgical_physics_v2 import default_profile as physics_profile
    from resectionlab.surgical_glioma_materials import generated_vascular_properties
    from resectionlab.surgical_exposure_geometry import exposure_declaration

    case, labels, mapping, artery, vein = generated_anatomy()
    scalp = _source_members(labels, [1, 2], 17, hinge=True, center_y=35, major_y=18, hinge_y=50)
    bone = _source_members(labels, [3, 4, 5], 10)
    dura = _source_members(labels, [6], 9, hinge=True)
    support = {}
    for p in artery+vein:
        for off in [(0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
            q = tuple(a+b for a, b in zip(p, off))
            if labels[q] in (7, 8):
                support[p] = q
                break
    faces = {}
    for name, line in [("artery", artery), ("vein", vein)]:
        i = next(i for i, p in enumerate(line) if world(p)[1] == 32)
        faces[name] = (line[i], line[i+1])
    faces["scalp"] = (voxel((32, 16, 65)), voxel((32, 15, 65)))
    cottonoid, hemostat = _pial_point(labels, 28, 28), _pial_point(labels, 35, 28)
    park = [58, 32, 74]
    exposure = exposure_declaration(SHAPE, case.affine, scalp=scalp, bone=bone, dura=dura,
                                    bone_material=bone["layerCells"], park_center=park,
                                    cottonoid_site=cottonoid, hemostat_site=hemostat)
    # The pilot permits a half-cell lateral finishing pass around the center
    # line. Removal still requires local contact and spent work. A second
    # finite annulus admits only the guarded cutter beside the mobile flap.
    from resectionlab.geometry import GeometryScene, capsule_voxel_indices
    pilot = np.array([44.75, 32., 61.])
    nominal_tip = np.array([pilot[0], pilot[1], 62.])
    cover = capsule_voxel_indices(GeometryScene(np.zeros(SHAPE, bool), case.affine),
                                  [pilot[0], pilot[1], 67.], nominal_tip, 2.5)
    bone_mask = np.isin(labels, [3, 4, 5])
    port = np.ravel_multi_index(cover[bone_mask[tuple(cover.T)]].T, SHAPE).tolist()
    mobile = set(bone["cells"])
    xyz = np.array(np.unravel_index(bone["layerCells"], SHAPE)).T + ORIGIN
    radial = np.hypot(xyz[:, 0]-32, xyz[:, 1]-32)
    stock = (radial <= 13.5) & (xyz[:, 2] >= 60) & (xyz[:, 2] <= 64)
    candidates = np.array(bone["layerCells"])[stock]
    from scipy.spatial import cKDTree
    seam_ids = exposure["bodies"]["bone"]["kerfCells"]
    seam = np.array(np.unravel_index(seam_ids, SHAPE)).T + ORIGIN
    near_seam = cKDTree(seam).query(xyz[stock])[0] <= 2.5
    collar = candidates[near_seam]
    exposure["bodies"]["bone"].update(pilotCenterMm=pilot.tolist(), portCells=sorted(set(port)-mobile),
                                      bladeCorridorCells=sorted(set(map(int, collar))-mobile))
    from resectionlab.surgical_bone_rest import rest_declaration
    exposure["boneRest"] = rest_declaration(SHAPE, case.affine, exposure["bodies"]["bone"])
    exposure["instrumentGeometryRevision"] = 2
    exposure["exterior"] = {"normalRas": [0, 0, 1], "pointRasMm": [32, 32, 70]}
    sample_site = [34, 30, 57]
    attachment = {"id": "surface-sampling-attachment", "kind": "soft", "breakable": False,
                  "cells": [int(np.ravel_multi_index(voxel(sample_site), SHAPE))]}
    props = {"attachments": [attachment], "tissuePhysicsV2": physics_profile(), "cusaV3": cusa_profile(), "operativeExposureV1": exposure,
             **generated_vascular_properties(
                 SHAPE, artery, vein, support, CASE_ID,
                 faces=faces, biopsy_neck_ijk=(voxel((34, 31, 57)), voxel((34, 32, 57))),
                 scalp_ijk=[voxel((32, y, 65)) for y in range(12, 17)],
                 scalp_seam_ijk=voxel((32, 17, 65)))}
    metadata = thaw_json(case.metadata)
    metadata["surgical_properties"] = props
    case = replace(case, revision=3, metadata=metadata)
    scene = from_case(case, mapping)
    landmarks = {"caseHash": case.semantic_hash, "sceneId": scene.identity, "sourceCaseId": CASE_ID,
                 "entryCenterRasMm": [32, 32, 66], "outwardRasUnit": [0, 0, 1],
                 "outsideRasMm": {"left": [12, 23, 78], "right": [51, 23, 78]},
                 "targetRasMm": [32, 32, 52], "corticalContactRasMm": _pial_point(labels, 29, 35),
                 "cottonoidSiteRasMm": cottonoid, "hemostatSiteRasMm": hemostat, "boneParkRasMm": park,
                 "sampleTipRasMm": sample_site, "sampleNeckRasMm": [34, 31.5, 57],
                 "resectionEntryRasMm": [33, 33, 56], "mappingSiteRasMm": [30, 35, 56],
                 "retractorSiteRasMm": [29, 35, 56], "corticalBleedFaceRasMm": [world(p) for p in faces["artery"]],
                 "vascularFacesRasMm": {k: [world(p) for p in pair] for k, pair in faces.items()},
                 "shape": list(SHAPE), "affine": case.affine.tolist(), "exposure": exposure}
    return case, scene, landmarks


def main():
    from resectionlab.surgical_engine import SurgicalEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory for generated-glioma.ressectionlab and landmarks.json")
    args = parser.parse_args()
    folder = args.directory
    folder.mkdir(parents=True, exist_ok=True)
    case, scene, points = build_glioma_case()
    dest = folder/"generated-glioma.ressectionlab"
    native = SurgicalEngine(scene)
    record = {"mapping": scene.regions, "branches": {"main": native.snapshot()}, "branchId": "main",
              "position": 0, "parents": {"main": None}, "timeline": {"main": []}}
    if dest.exists():
        if load_case(dest).semantic_hash != case.semantic_hash or read_case_artifacts(dest).get("surgical_toolbox") != record:
            raise ValueError("Existing case differs. Select a new output directory.")
    else:
        save_case(case, dest, artifacts={"surgical_toolbox": record})
    receipt = folder/"landmarks.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != points:
            raise ValueError("Existing landmark receipt differs. Select a new output directory.")
    else:
        with receipt.open("x") as out:
            json.dump(points, out, separators=(",", ":"), allow_nan=False)
            out.write("\n")
    print(json.dumps({"case": str(dest), "caseHash": case.semantic_hash, "sceneId": scene.identity}))


if __name__ == "__main__":
    main()
