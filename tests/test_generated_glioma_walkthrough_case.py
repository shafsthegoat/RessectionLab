"""Source invariants for the reusable, generated layered convexity case."""
import numpy as np
import pytest
from scipy.ndimage import binary_dilation, binary_fill_holes, label

from scripts.generated_glioma_walkthrough_case import build_glioma_case, voxel
from resectionlab.surgical_scene import original_target_memberships


@pytest.fixture(scope="module")
def phantom():
    return build_glioma_case()


def test_normal_transverse_access_reaches_two_tissue_types_before_target(phantom):
    case, scene, points = phantom
    assert case.metadata["generated"] and case.metadata["is_synthetic"]
    assert case.revision == 3 and scene.properties["operativeExposureV1"]["instrumentGeometryRevision"] == 2
    assert all(s.provenance == "simulated" and s.uri.startswith("synthetic://") for s in case.source_refs)
    x, y, _ = points["entryCenterRasMm"]
    # Source sequence from exterior down to tumour. A true cortical layer
    # remains between the incised dura and the target on the central path.
    owners = [scene.labels[voxel((x, y, z))] for z in range(67, 51, -1)]
    assert owners == [0, 1, 1, 2, 3, 4, 5, 0, 6, 0, 7, 7, 9, 9, 10, 10]
    masks = original_target_memberships(scene)
    assert set(masks) == {"Glioma margin", "Glioma core"}
    assert not np.intersect1d(*masks.values()).size
    assert [len(masks[k]) for k in ("Glioma margin", "Glioma core")] == [434, 89]
    assert label(np.isin(scene.labels, [9, 10]))[1] == 1
    assert case.functional_evidence is None
    assert points["sceneId"] == scene.identity and points["caseHash"] == case.semantic_hash


def test_compound_exposure_and_vascular_landmarks_share_the_simulation_grid(phantom):
    _, scene, points = phantom
    exposure = scene.properties["operativeExposureV1"]
    assert set(exposure["bodies"]) == {"scalp", "bone", "dura"}
    for name, expected in [("scalp", [1, 2]), ("bone", [3, 4, 5]), ("dura", [6])]:
        body = exposure["bodies"][name]
        assert sorted(np.unique(scene.labels.ravel()[body["cells"]])) == expected
        mask = np.zeros(scene.shape, dtype=bool)
        mask.ravel()[body["cells"]] = True
        assert label(mask)[1] == 1
        assert bool(body["hingeEdges"]) == (name != "bone")
    for name in ("corticalContactRasMm", "cottonoidSiteRasMm", "hemostatSiteRasMm"):
        assert scene.labels[voxel(points[name])] == 7
    edges = {e["id"]: e for e in scene.properties["vascularGraphV1"]["edges"]}
    for name, rid in [("artery", 11), ("vein", 12), ("scalp", 13)]:
        assert label(scene.labels == rid)[1] == 1
        assert np.all(scene.labels.ravel()[edges[name]["sourceCells"]] == rid)
        a, b = np.array(np.unravel_index(edges[name]["sourceFace"], scene.shape)).T
        assert np.abs(a-b).sum() == 1
        assert scene.points(edges[name]["sourceFace"]).tolist() == points["vascularFacesRasMm"][name]


def test_biopsy_target_requires_a_real_neck_cut_and_release(phantom):
    _, scene, points = phantom
    specimen = {voxel(points["sampleTipRasMm"]), voxel((34, 31, 57))}
    bonds = set()
    for cell in specimen:
        assert scene.labels[cell] == 9
        for delta in np.vstack((np.eye(3, dtype=int), -np.eye(3, dtype=int))):
            other = tuple(np.asarray(cell)+delta)
            if other not in specimen and scene.labels[other] in (9, 10):
                bonds.add((cell, other))
    assert bonds == {(voxel((34, 31, 57)), voxel((34, 32, 57)))}
    graph = scene.properties["vascularGraphV1"]
    capillary = next(e for e in graph["edges"] if e["id"] == "biopsy-bed")
    assert capillary["sourceFace"] == scene.properties["gliomaMaterialsV1"]["biopsyNeckFace"]
    assert scene.points(capillary["sourceFace"]).tolist() == [[34, 32, 57], [34, 31, 57]]
    source_id = np.ravel_multi_index(voxel(points["sampleTipRasMm"]), scene.shape)
    assert scene.properties["attachments"] == [
        {"id": "surface-sampling-attachment", "kind": "soft", "cells": [source_id], "breakable": False}
    ]
    # Every overhead source solid in the specimen's vertical line of travel
    # is accounted for by the three actual mobile bodies.
    carried = set.union(*(set(body["cells"]) for body in scene.properties["operativeExposureV1"]["bodies"].values()))
    for i, j, z in specimen:
        for upper in range(z+1, scene.shape[2]):
            if scene.labels[i, j, upper]:
                assert np.ravel_multi_index((i, j, upper), scene.shape) in carried


def test_exterior_skin_and_finite_bone_work_are_bound_to_mask_solids(phantom):
    _, scene, _ = phantom
    exterior = binary_dilation(~binary_fill_holes(scene.labels != 0))
    assert set(np.unique(scene.labels[exterior])) == {0, 1}
    config = scene.properties["operativeExposureV1"]
    bone = config["bodies"]["bone"]
    assert set(bone["kerfCells"]) <= set(bone["bladeCorridorCells"])
    assert set(bone["kerfCells"]) == {right for _, right in bone["seamEdges"]}
    assert not set(bone["cells"]).intersection(set(bone["portCells"]) | set(bone["bladeCorridorCells"]))
    assert len(config["boneRest"]["pads"]) == 3
    assert bone["pilotCenterMm"] == [44.75, 32., 61.]
