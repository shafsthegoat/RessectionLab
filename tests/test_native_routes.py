"""Explicit native research alternatives preserve geometry and source evidence."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.geometry import GENERIC_TOOLS
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionEngine, native_config_from_case
from resectionlab.native_routes import generate_native_axis_routes, propose_native_axis_access
from resectionlab.planning import SearchConfig, generate_candidate_routes


def case_fixture(affine=None):
    shape = (25, 25, 25)
    brain = np.zeros(shape, dtype=bool)
    brain[2:23, 2:23, 2:23] = True
    grid = np.indices(shape)
    target = sum((grid[axis] - (17, 12, 12)[axis]) ** 2 for axis in range(3)) <= 9
    return CaseData("native-axis-fixture", brain.astype(np.float32), {"target": target},
                    np.eye(4) if affine is None else affine,
                    (SourceRef("analytic", "synthetic:native-axis", provenance="simulated"),), brain_mask=brain)


def test_named_alternatives_have_exact_native_ray_and_never_change_generic_tools():
    case = case_fixture()
    original_tools = tuple((tool.tool_id, tool.tip_radius_mm, tool.shaft_radius_mm) for tool in GENERIC_TOOLS)
    source_hash = case.semantic_hash
    alternatives = generate_native_axis_routes(case)
    assert len(alternatives.candidates) == 2
    assert alternatives.requested_candidates == 2
    assert alternatives.access_support["cortical_access_permitted"] is False
    for route, tool in zip(alternatives.candidates, NATIVE_GENERIC_TOOLS, strict=True):
        assert route.tool == tool
        assert route.entry_mm == (22.5, 12., 12.)
        assert route.target_mm == (17., 12., 12.)
        assert route.window.radius_mm == 6.
        assert route.feasible
        assert route.simulated_removed_target_volume_mm3 is None
        assert route.clinical_deficit_probability is None
        engine = NativeResectionEngine(native_config_from_case(case, access=route.window, tools=(tool,)))
        before = engine.state_hash
        preview = engine.preview_stroke(tool.tool_id, route.target_mm, entry_mm=route.entry_mm)
        assert preview.feasible and preview.removed_volume_mm3 > 0
        assert engine.state_hash == before
        assert engine.revision == 0
    assert case.semantic_hash == source_hash
    assert original_tools == tuple((tool.tool_id, tool.tip_radius_mm, tool.shaft_radius_mm) for tool in GENERIC_TOOLS)
    with pytest.raises(ValueError, match="unchanged native research"):
        generate_native_axis_routes(case, tools=(GENERIC_TOOLS[0],))
    with pytest.raises(ValueError, match="unchanged native research"):
        generate_native_axis_routes(case, tools=(replace(NATIVE_GENERIC_TOOLS[0], shaft_radius_mm=.1),))


def test_native_ray_is_invariant_under_oblique_patient_frame_and_lps_conversion():
    angle = .7
    rotation = np.array(((np.cos(angle), -np.sin(angle), 0.),
                         (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag((2., 1., 1.5))
    affine[:3, 3] = (10., -30., 12.)
    case = case_fixture(affine)
    ras = propose_native_axis_access(case)
    flip = np.diag((-1., -1., 1., 1.))
    lps = propose_native_axis_access(case.revised(affine=flip @ affine, frame="LPS+"))
    assert np.allclose(lps.window.center_mm, flip[:3, :3] @ ras.window.center_mm)
    assert np.allclose(lps.window.normal_inward, flip[:3, :3] @ ras.window.normal_inward)
    assert np.allclose(lps.target_mm, flip[:3, :3] @ ras.target_mm)
    entry_index = case.world_to_voxel(ras.window.center_mm)
    assert np.count_nonzero(~np.isclose(entry_index, np.rint(entry_index))) == 1
    assert ras.source_voxel == lps.source_voxel == (17, 12, 12)


def test_disconnected_target_centroid_uses_an_actual_annotated_cell():
    case = case_fixture()
    target = np.zeros(case.mri.shape, dtype=bool)
    target[5, 12, 12] = target[19, 12, 12] = True
    case = case.revised(compartments={"target": target})
    proposal = propose_native_axis_access(case)
    assert target[proposal.source_voxel]
    assert proposal.source_voxel == (5, 12, 12)
    assert proposal.target_mm == (5., 12., 12.)


@pytest.mark.parametrize("exclusion", [
    {"skull_stripped": False}, {"structural_coverage": "full_head"},
    {"allow_nonzero_mri_access_support": False},
])
def test_explicit_source_prohibitions_override_collection_prior(exclusion):
    case = case_fixture().revised(brain_mask=None,
        metadata={"source_collection": {"name": "UCSF-PDGM"}, **exclusion})
    with pytest.raises(ValueError, match="BRAIN_ENVELOPE_REQUIRED"):
        generate_native_axis_routes(case)


def test_unknown_mri_and_conflicting_or_sheared_support_do_not_become_access():
    case = case_fixture()
    with pytest.raises(ValueError, match="BRAIN_ENVELOPE_REQUIRED"):
        propose_native_axis_access(case.revised(brain_mask=None))
    conflict = np.array(case.brain_mask)
    conflict[case.compartments["target"]] = False
    with pytest.raises(ValueError, match="outside the supplied brain"):
        propose_native_axis_access(case.revised(brain_mask=conflict))
    affine = np.eye(4)
    affine[0, 1] = .2
    with pytest.raises(ValueError, match="orthogonal"):
        propose_native_axis_access(case.revised(affine=affine))
    with pytest.raises(ValueError, match="nonoverlapping"):
        propose_native_axis_access(case.revised(compartments={"a": case.compartments["target"],
                                                            "b": case.compartments["target"]}))


def test_explicit_target_validation_hash_and_original_search_are_separate():
    case = case_fixture()
    config = SearchConfig(targets_per_compartment=1)
    original = generate_candidate_routes(case, config=config)
    alternative = generate_native_axis_routes(case)
    assert generate_candidate_routes(case, config=config).to_dict()["candidates"] == original.to_dict()["candidates"]
    assert {route.route_id for route in original.candidates}.isdisjoint(route.route_id for route in alternative.candidates)
    assert original.planning_model_hash != alternative.planning_model_hash
    window = alternative.candidates[0].window
    first = generate_candidate_routes(case, windows=(window,), explicit_targets=(("target", (17., 12., 12.)),))
    changed = generate_candidate_routes(case, windows=(window,), explicit_targets=(("target", (18., 12., 12.)),))
    assert first.planning_model_hash != changed.planning_model_hash
    for bad in (("target", (17.1, 12., 12.)), ("target", (1., 1., 1.)), ("missing", (17., 12., 12.)),
                ("target", (np.nan, 12., 12.))):
        with pytest.raises(ValueError, match="Explicit targets"):
            generate_candidate_routes(case, windows=(window,), explicit_targets=(bad,))
