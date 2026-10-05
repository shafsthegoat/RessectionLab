"""Analytical ingress controls only: no patients, optimization or training."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import array_digest
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_ingress import IngressCandidate, screen_axis_accesses
from resectionlab.native_proposals import NominalCavityProposalConfig, PreparedNominalCavityProposer
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine

TOOLS = (ToolGeometry("fine", 1.25, .1, 30, 35, 2), ToolGeometry("wide", 2.25, 1.1, 30, 35, 3))


def candidates(*, tissue=None, nominal=None, affine=None, hard=None, tools=TOOLS,
               offsets=((0, 0),), radius=6., distances=None, entries=None, rule=None):
    if tissue is None:
        tissue = np.zeros((11, 11, 11), bool); tissue[3:8, 3:8, 3:8] = True
    if nominal is None:
        nominal = tissue.astype(np.float32)
    affine = np.eye(4) if affine is None else affine
    result = []
    for axis in range(3):
        for sign in (-1, 1):
            voxel = np.array([5., 5., 5.]); voxel[axis] = 2.5 if sign == -1 else 7.5
            if entries and (axis, sign) in entries:
                voxel = np.array(entries[(axis, sign)], dtype=float)
            normal = -sign * affine[:3, axis]/np.linalg.norm(affine[:3, axis])
            access = AccessWindow(affine[:3,:3]@voxel + affine[:3,3], normal, radius, f"axis{axis}:{sign}")
            config = NativeResectionConfig(tissue, np.zeros(tissue.shape, np.int16), affine,
                access, tools, "analytic-source", "declared analytic support", hard_exclusion=hard)
            engine = NativeResectionEngine(config)
            proposer = PreparedNominalCavityProposer(config, nominal,
                nominal_provenance={"source_hash": config.source_hash, "nominal_target_hash": array_digest(np.asarray(nominal,np.float32)),
                    "source_kind": "supplied_annotation", "derivation": "analytic declared nominal"},
                config=rule or NominalCavityProposalConfig(offsets))
            distance = 2.5 if distances is None else distances[(axis, sign)]
            result.append(IngressCandidate(axis, sign, distance, engine, proposer))
    return result


def row(result, key):
    return next(r for r in result.to_dict()["exits"] if (r["axis"],r["outward_sign"]) == key)


def test_six_exit_tie_break_and_no_native_preview_or_commit(monkeypatch):
    sources = candidates()
    before = [c.engine.state_hash for c in sources]
    def forbidden(*args, **kwargs):
        pytest.fail("Ingress must not preview or commit strokes")
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    monkeypatch.setattr(NativeResectionEngine, "commit_preview", forbidden)
    result = screen_axis_accesses(reversed(sources))
    assert result.complete and result.selected_exit == (0,-1), result.error
    assert result.screen_count == 12
    assert before == [c.engine.state_hash for c in sources]
    assert all(not c.engine.removed_mask.any() and not c.engine.contact_mask.any() for c in sources)
    assert not result.to_dict()["complete_stroke_certified"] and not result.to_dict()["removal_authorized"]
    assert all(p["geometry"]["unknowns"] for r in result.to_dict()["exits"] for p in r["poses"])


def test_external_neighbor_does_not_clear_backward_shaft():
    tissue = np.zeros((11,11,11),bool); tissue[3:8,3:8,3:8]=True; tissue[5,5,0]=True
    sources = candidates(tissue=tissue)
    assert sources[4].engine.connected_free_mask[5,5,2]
    result = screen_axis_accesses(sources)
    bad = row(result,(2,-1))
    assert result.complete and not bad["eligible"]
    assert all(p["first_blocked_cell"] == [5,5,0] for p in bad["poses"])
    for proposal in sources[4].proposer.propose(sources[4].engine).proposals:
        actual = sources[4].engine.preview_stroke(proposal.tool_id, proposal.tip_mm, entry_mm=proposal.entry_mm)
        assert actual.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"
        assert actual.failure_tip_mm == proposal.entry_mm and not actual.microsteps


def test_any_single_tool_entry_is_sufficient_and_distance_precedes_axis():
    tissue = np.zeros((11,11,11),bool); tissue[3:8,3:8,3:8]=True; tissue[6,5,0]=True
    distances = {(a,s):10. for a in range(3) for s in (-1,1)}; distances[2,-1]=1.
    result = screen_axis_accesses(candidates(tissue=tissue,distances=distances))
    poses = row(result,(2,-1))["poses"]
    assert result.selected_exit == (2,-1)
    assert all(p["admissible"] for p in poses if p["tool_id"] == "fine")
    assert all(not p["admissible"] for p in poses if p["tool_id"] == "wide")


def test_hard_active_tip_contact_is_rejected_without_turning_ordinary_tip_contact_into_removal():
    sources = candidates()
    assert screen_axis_accesses(sources).complete
    hard = np.zeros((11,11,11),bool); hard[5,5,3]=True
    result = screen_axis_accesses(candidates(hard=hard))
    rejected = row(result,(2,-1))
    assert all("HARD_GEOMETRY:FORBIDDEN_COLLISION" in p["reasons"] for p in rejected["poses"])
    assert all(p["blocked_cell_count"] == 0 for p in rejected["poses"])


def test_aperture_tangency_rejects_all_and_preserves_no_selection():
    result = screen_axis_accesses(candidates(radius=1.25,tools=(TOOLS[0],)))
    assert result.complete and result.status == "no_ingress_admissible_access" and result.selected_exit is None
    assert all("HARD_GEOMETRY:ACCESS_APERTURE" in p["reasons"] for r in result.to_dict()["exits"] for p in r["poses"])


def test_initial_ingress_does_not_certify_later_short_tip_stroke():
    tool = ToolGeometry("short",.9,.1,30,35,.1)
    entries = {}
    for axis in range(3):
        for sign in (-1,1):
            point=[5.,5.,5.]; point[axis]=2.49 if sign==-1 else 7.51; entries[axis,sign]=point
    sources = candidates(tools=(tool,),entries=entries)
    result = screen_axis_accesses(sources)
    assert result.selected_exit == (0,-1)
    candidate = sources[0]
    ray = next(p for p in candidate.proposer.propose(candidate.engine).proposals if p.family == "distal_nominal")
    actual = candidate.engine.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm)
    assert not actual.feasible and actual.reason == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"
    assert result.selected_exit == (0,-1)  # No reward/stroke-driven access fallback.


def test_exact_duplicate_entries_reuse_checks_but_one_ulp_entries_do_not(monkeypatch):
    sources = candidates()
    original = sources[0].proposer.propose
    def changed(engine, **kwargs):
        batch = original(engine,**kwargs)
        rays = list(batch.proposals)
        target = next(i for i,p in enumerate(rays) if p.family=="distal_nominal" and p.tool_id=="fine")
        ray=rays[target]; entry=list(ray.entry_mm); entry[1]=float(np.nextafter(entry[1],np.inf))
        rays[target]=replace(ray,entry_mm=tuple(entry))
        return replace(batch,proposals=tuple(rays))
    monkeypatch.setattr(sources[0].proposer,"propose",changed)
    result=screen_axis_accesses(sources)
    assert result.complete and result.screen_count==13, result.error
    assert len({p["screen_index"] for p in row(result,(0,-1))["poses"]})==3


def test_reflected_oblique_anisotropic_frame_matches_source_cell_witness():
    tissue=np.zeros((11,11,11),bool); tissue[3:8,3:8,3:8]=True; tissue[5,5,0]=True
    reference=screen_axis_accesses(candidates(tissue=tissue))
    angle=.4; affine=np.eye(4)
    affine[:3,:3]=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,-1]])@np.diag([1.2,1.,1.])
    affine[:3,3]=[20.,-10.,7.]
    result=screen_axis_accesses(candidates(tissue=tissue,affine=affine))
    assert result.complete, result.error
    first=row(result,(2,-1))["poses"][0]
    assert first["first_blocked_cell"] == row(reference,(2,-1))["poses"][0]["first_blocked_cell"]
    np.testing.assert_allclose(first["first_blocked_cell_mm"],affine[:3,:3]@[5,5,0]+affine[:3,3],rtol=0,atol=1e-13)


def test_capped_inventory_is_incomplete_and_cannot_select_other_exit():
    result=screen_axis_accesses(candidates(rule=NominalCavityProposalConfig(((0,0),),max_candidates=1)))
    assert not result.complete and result.selected_exit is None and result.status=="failed"
    assert "Incomplete" in result.error and result.screen_count==0


def test_cancellation_retains_completed_rows_but_never_partial_selection():
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        return calls==12
    result=screen_axis_accesses(candidates(),cancelled=cancel)
    assert result.status=="interrupted" and not result.complete and result.selected_exit is None
    assert any(r["status"]=="screened" for r in result.exit_records)


def test_final_callback_source_mutation_invalidates_prior_clear_exits(monkeypatch):
    import resectionlab.native_ingress as ingress
    sources=candidates(); completed=0
    original=ingress.check_pose
    def track(*args,**kwargs):
        nonlocal completed
        result=original(*args,**kwargs); completed+=1
        return result
    monkeypatch.setattr(ingress,"check_pose",track)
    def mutate():
        if completed==12: sources[0].engine.remaining_mask[3,3,3]=False
        return False
    result=screen_axis_accesses(sources,cancelled=mutate)
    assert result.status=="failed" and result.selected_exit is None and "remaining_mask" in result.error


def test_noninitial_cavity_is_not_a_new_access_selection():
    sources=candidates(); sources[0].engine.revision=1
    result=screen_axis_accesses(sources)
    assert not result.complete and result.selected_exit is None and result.screen_count==0


def test_other_source_at_one_exit_refuses_comparison():
    sources=candidates(); tissue=np.zeros((11,11,11),bool); tissue[4:8,3:8,3:8]=True
    sources[1]=candidates(tissue=tissue)[1]
    result=screen_axis_accesses(sources)
    assert not result.complete and result.selected_exit is None and "same physical source" in result.error


def test_receipt_immutable_and_export_is_detached():
    result=screen_axis_accesses(candidates())
    fingerprint=result.fingerprint
    with pytest.raises(TypeError): result.exit_records[0]["eligible"]=False
    exported=result.to_dict(); exported["exits"][0]["poses"][0]["admissible"]=False
    assert result.fingerprint==fingerprint


@pytest.mark.parametrize("items", [[],[None]*6])
def test_invalid_exit_set_returns_no_selection(items):
    result=screen_axis_accesses(items)
    assert not result.complete and result.selected_exit is None


@pytest.mark.parametrize("changes", [{"axis":True},{"outward_sign":0},{"distance_mm":float("nan")},{"distance_mm":True},{"distance_mm":0}])
def test_invalid_candidate_metadata_rejected(changes):
    with pytest.raises(ValueError): replace(candidates()[0],**changes)


def test_full_thirteen_column_ledger_remains_complete_with_unavailable_columns():
    result=screen_axis_accesses(candidates(rule=NominalCavityProposalConfig()))
    assert result.complete, result.error
    assert sum(r["proposal_dispositions"]["slot_count"] for r in result.to_dict()["exits"])==468
    assert result.screen_count<=468
    reasons={slot["reason"] for r in result.to_dict()["exits"] for slot in r["proposal_dispositions"]["ledger"]}
    assert "NO_EXPOSED_REMAINING_TISSUE" in reasons


def test_oversized_declared_catalog_refuses_before_geometry():
    offsets=tuple((a,b) for a in (-2,-1,0,1,2) for b in (-1,0,1))
    result=screen_axis_accesses(candidates(rule=NominalCavityProposalConfig(offsets)))
    assert not result.complete and result.selected_exit is None and result.screen_count==0
    assert "oversized" in result.error
