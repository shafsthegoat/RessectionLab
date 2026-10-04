"""Proposal inventories bind source and cavity without granting removal."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow, _immutable
from resectionlab.native_proposals import AxisColumnProposalConfig, PreparedAxisColumnProposer
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig, NativeResectionEngine


def fixture(*, center=(8., 8., -.5), shape=(17, 17, 20), radius=6., normal=(0, 0, 1), affine=None):
    tissue = np.ones(shape, bool)
    labels = np.zeros(shape, np.int16)
    labels[:, :, 5:19] = 1
    affine = np.eye(4) if affine is None else affine
    center_mm = affine[:3, :3] @ center + affine[:3, 3]
    normal_mm = affine[:3, :3] @ normal
    config = NativeResectionConfig(tissue, labels, affine, AccessWindow(center_mm, normal_mm, radius),
        tuple(replace(tool) for tool in NATIVE_GENERIC_TOOLS), "source:test-native-proposals", "explicit synthetic support")
    return NativeResectionEngine(config)


def prototype_rays(engine):
    path = Path(__file__).parents[1] / "artifacts/native-frontier-expansion-v1/experiment-2/experiment-script.py"
    spec = importlib.util.spec_from_file_location("frozen_frontier_prototype", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.axis_rays(engine)[0]


def comparable(batch):
    return [(p.tool_id, p.entry_mm, (p.primary_target_mm,) + (() if p.fallback_target_mm is None else (p.fallback_target_mm,)))
            for p in batch.proposals]


def legacy_comparable(engine):
    return [(p["tool_id"], tuple(p["entry_mm"]), tuple(tuple(point) for point in p["endpoints_mm"]))
            for p in prototype_rays(engine)]


def test_matches_reviewed_prototype_initial_and_after_paid_native_stroke():
    engine = fixture()
    provider = PreparedAxisColumnProposer(engine.config)
    before = engine.state_hash
    remaining = engine.remaining_mask.copy()
    initial = provider.propose(engine)
    assert comparable(initial) == legacy_comparable(engine)
    assert initial.counts == {"PROPOSED_UNCERTIFIED": 26}
    assert len(initial.ledger) == 26
    assert provider.propose(engine) == initial == provider.propose(engine.clone())
    assert engine.state_hash == before
    np.testing.assert_array_equal(engine.remaining_mask, remaining)
    assert not initial.to_dict()["geometry_certified"]
    assert not initial.to_dict()["removal_authorized"]
    chosen = next(p for p in initial.proposals if p.column_index == 0 and p.tool_id == "native-wide-aspiration")
    cut = engine.execute_stroke(chosen.tool_id, chosen.primary_target_mm, entry_mm=chosen.entry_mm)
    assert cut.feasible
    after = provider.propose(engine)
    assert comparable(after) == legacy_comparable(engine)
    assert after.proposal_model_hash == initial.proposal_model_hash
    assert after.cavity_state_hash != initial.cavity_state_hash
    assert {p.proposal_id for p in initial.proposals}.isdisjoint(p.proposal_id for p in after.proposals)
    assert after.counts["NO_REMAINING_TARGET_IN_COLUMN"] >= 2
    assert sum(after.counts.values()) == len(after.ledger) == 26


def test_candidate_cap_aperture_and_out_of_image_have_complete_slot_denominators():
    engine = fixture()
    capped = PreparedAxisColumnProposer(engine.config, AxisColumnProposalConfig(max_primary_rays=3)).propose(engine)
    assert capped.counts == {"PROPOSED_UNCERTIFIED": 3, "PRIMARY_CANDIDATE_CAP": 23}
    assert [p.column_index for p in capped.proposals] == [0, 0, 1]
    narrow = fixture(radius=2.)
    filtered = PreparedAxisColumnProposer(narrow.config).propose(narrow)
    assert filtered.counts == {"PROPOSED_UNCERTIFIED": 1, "FULL_TOOL_APERTURE_PREFILTER": 25}
    edge = fixture(center=(0., 0., -.5))
    original = edge.remaining_mask.copy()
    border = PreparedAxisColumnProposer(edge.config).propose(edge)
    assert border.counts == {"PROPOSED_UNCERTIFIED": 12, "COLUMN_OUT_OF_IMAGE": 14}
    assert len({(slot.column_index, slot.tool_id) for slot in border.ledger}) == 26
    np.testing.assert_array_equal(edge.remaining_mask, original)
    assert not edge.removed_mask.any()
    assert all(p.entry_mm[0] >= 0 and p.entry_mm[1] >= 0 for p in border.proposals)


@pytest.mark.parametrize("angle", [.001, 1e-6])
def test_near_axial_rotation_abstains_by_off_axis_components(angle):
    engine = fixture(normal=(np.sin(angle), 0., np.cos(angle)))
    assert prototype_rays(engine)  # Preserved prototype admitted these inputs.
    batch = PreparedAxisColumnProposer(engine.config).propose(engine)
    assert not batch.proposals
    assert batch.unsupported_reason == "UNSUPPORTED_NONAXIAL_ACCESS"
    assert batch.counts == {"UNSUPPORTED_NONAXIAL_ACCESS": 26}


def test_fractional_source_origin_uses_zero_relative_tolerance():
    engine = fixture(center=(154.0005, 8., -.5), shape=(200, 17, 20))
    assert prototype_rays(engine)
    batch = PreparedAxisColumnProposer(engine.config).propose(engine)
    assert batch.counts == {"UNSUPPORTED_FRACTIONAL_TRANSVERSE_ORIGIN": 26}
    assert not batch.proposals


def test_oblique_orthogonal_mirrored_and_anisotropic_grids_keep_physical_contract():
    angle = .4
    rotation = np.array(((np.cos(angle), 0, np.sin(angle)), (0, 1, 0), (-np.sin(angle), 0, np.cos(angle))))
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag((-1., 1., 1.))
    affine[:3, 3] = (12., -30., 2.)
    original = fixture()
    transformed = fixture(affine=affine)
    before = PreparedAxisColumnProposer(original.config).propose(original)
    after = PreparedAxisColumnProposer(transformed.config).propose(transformed)
    assert after.unsupported_reason is None
    assert len(after.proposals) == len(before.proposals)
    for one, two in zip(before.proposals, after.proposals, strict=True):
        np.testing.assert_allclose(affine[:3, :3] @ one.entry_mm + affine[:3, 3], two.entry_mm)
        np.testing.assert_allclose(affine[:3, :3] @ one.primary_target_mm + affine[:3, 3], two.primary_target_mm)
    anisotropic = fixture(affine=np.diag((.8, 1.4, 2., 1.)))
    assert comparable(PreparedAxisColumnProposer(anisotropic.config).propose(anisotropic)) == legacy_comparable(anisotropic)


@pytest.mark.parametrize("kind", ["nonfinite", "singular", "nonhomogeneous"])
def test_invalid_source_transforms_raise_before_any_candidate(kind):
    engine = fixture()
    affine = np.eye(4)
    if kind == "nonfinite":
        affine[0, 0] = np.nan
    elif kind == "singular":
        affine[0, 0] = 0.
    else:
        affine[3, 0] = .1
    # NativeConfig already rejects these at construction. Defend the new API
    # against a forcibly altered object rather than relying on its class alone.
    object.__setattr__(engine.config, "affine", _immutable(affine))
    with pytest.raises(ValueError, match="affine"):
        PreparedAxisColumnProposer(engine.config)


def test_shear_abstains_without_reorienting_source_grid():
    engine = fixture()
    affine = np.eye(4)
    affine[0, 1] = .000001
    object.__setattr__(engine.config, "affine", _immutable(affine))
    with pytest.raises(ValueError, match="orthogonal affine"):
        PreparedAxisColumnProposer(engine.config)


def test_model_identity_changes_with_rule_source_and_cavity_and_rejects_foreign_config():
    engine = fixture()
    rule = AxisColumnProposalConfig()
    original = PreparedAxisColumnProposer(engine.config, rule)
    changed_rule = PreparedAxisColumnProposer(engine.config, replace(rule, max_primary_rays=25))
    changed_source = replace(engine.config, source_hash="different-source")
    other = PreparedAxisColumnProposer(changed_source, rule)
    assert len({original.model_hash, changed_rule.model_hash, other.model_hash}) == 3
    with pytest.raises(ValueError, match="exact source"):
        original.propose(NativeResectionEngine(replace(engine.config)))
    object.__setattr__(rule, "max_primary_rays", 1)
    with pytest.raises(RuntimeError, match="changed"):
        original.propose(engine)
    provider = PreparedAxisColumnProposer(engine.config)
    object.__setattr__(engine.config.tools[0], "tip_radius_mm", 2.)
    with pytest.raises(RuntimeError, match="changed"):
        provider.propose(engine)


def test_changed_committed_contact_history_cannot_retain_cavity_identity():
    engine = fixture()
    provider = PreparedAxisColumnProposer(engine.config)
    chosen = provider.propose(engine).proposals[0]
    result = engine.execute_stroke(chosen.tool_id, chosen.primary_target_mm, entry_mm=chosen.entry_mm)
    assert result.feasible
    current = provider.propose(engine)
    engine.history[0]["microsteps"][1]["active_radius_mm"] += .1
    assert engine.state_hash == current.cavity_state_hash
    with pytest.raises(RuntimeError, match="ancestry"):
        provider.propose(engine)


def test_unreviewed_native_history_schema_cannot_reuse_ancestry_validation(monkeypatch):
    import resectionlab.native_proposals as module
    engine = fixture()
    monkeypatch.setattr(module, "NATIVE_RESECTION_VERSION", "unreviewed-future-native-schema")
    with pytest.raises(ValueError, match="reviewed native-engine history schema"):
        PreparedAxisColumnProposer(engine.config)


@pytest.mark.parametrize("changes", [
    {"offsets_source_voxels": ((0, 0), (0, 0))},
    {"offsets_source_voxels": ((0.1, 0),)},
    {"offsets_source_voxels": ((False, 0),)},
    {"offsets_source_voxels": ()}, {"max_primary_rays": 0}, {"max_primary_rays": True},
])
def test_rule_rejects_ambiguous_or_unbounded_inventory(changes):
    with pytest.raises(ValueError):
        AxisColumnProposalConfig(**changes)
