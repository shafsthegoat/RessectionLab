"""Complete native accounting, reference parity and source-cache invalidation."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_modeled_outcomes import summarize_engine_outcomes
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_sequence_outcomes import sequence_outcome


def source():
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8:13, 8:13, 8:11] = 1
    labels[8:13, 8:13, 11:15] = 2
    return SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'access tissue', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'synthetic target', 'material': 'tumor', 'target': True}],
        {'provenance': 'GENERATED accounting regression; not patient evidence'}, {})


def pose(z):
    return {'position': [10., 10., z], 'quaternion': [0., 0., 0., 1.]}


def execute(engine, action):
    result = engine.step(action)
    assert result['status'] == 'accepted', result


def prepared():
    engine = SurgicalEngine(source())
    execute(engine, {'slot': 'right', 'operation': 'place', 'pose': pose(4.), 'duration': .1})
    execute(engine, {'slot': 'right', 'operation': 'apply', 'pose': pose(10.), 'duration': 1.})
    return engine


def test_complete_ledger_counts_access_once_across_preview_stop_exchange_and_cold_replay():
    engine = prepared()
    baseline = engine.clone()
    before = sequence_outcome(engine)
    action = {'slot': 'right', 'operation': 'apply', 'pose': pose(14.), 'duration': 1.}
    preview = engine.preview(action)
    assert preview['status'] == 'accepted'
    assert sequence_outcome(engine) == before
    execute(engine, action)
    # Repeating removal cannot credit the same source identities twice.
    target = engine.metrics()['targetRemovedMm3']
    execute(engine, action)
    assert engine.metrics()['targetRemovedMm3'] == target
    execute(engine, {'slot': 'right', 'operation': 'withdraw', 'pose': pose(-4.), 'duration': 1.})
    execute(engine, {'slot': 'right', 'operation': 'exchange', 'instrument': 'dissector', 'duration': .2})
    result = sequence_outcome(engine, baseline=baseline)
    dims = result['dimensions']
    assert dims['nonTargetRemovedMm3']['cumulative'] == before['dimensions']['nonTargetRemovedMm3']['cumulative'] > 0
    assert dims['nonTargetRemovedMm3']['incremental'] == 0
    assert dims['targetRemovedMm3']['cumulative'] == target > 0
    assert dims['toolExchanges']['cumulative'] == dims['toolExchanges']['incremental'] == 1
    assert dims['elapsedSeconds']['cumulative'] == pytest.approx(engine.state['elapsedSeconds'])
    assert result['withdrawalComplete']
    assert result['target']['residualMm3'] + target == result['target']['sourceVolumeMm3']
    assert result['dimensions']['motorEncounterMm3']['cumulative'] is None
    assert result['dimensions']['thermalPeakC']['status'] == 'unavailable'
    restored = SurgicalEngine.restore(engine.scene, copy.deepcopy(engine.snapshot()))
    assert sequence_outcome(restored, baseline=restored.branch(len(baseline.actions))) == result
    continued = restored.clone()
    execute(continued, {'slot': 'right', 'operation': 'inspect', 'duration': .3})
    extended = sequence_outcome(continued, baseline=restored)
    assert extended['dimensions']['targetRemovedMm3']['incremental'] == 0
    assert extended['dimensions']['nonTargetRemovedMm3']['cumulative'] == dims['nonTargetRemovedMm3']['cumulative']
    assert extended['dimensions']['elapsedSeconds']['incremental'] == pytest.approx(.3)


def test_cached_and_uncached_source_outcomes_are_exact_after_each_cavity_edit(monkeypatch):
    engine = prepared()
    baseline = engine.clone()
    cached = summarize_engine_outcomes(engine, baseline=baseline)
    assert cached == summarize_engine_outcomes(engine, baseline=baseline, use_geometry_cache=False)
    # Warm evaluation reuses the immutable source geometry, including octants.
    monkeypatch.setattr(engine.scene, 'points', lambda *_: pytest.fail('Source geometry was rescanned'))
    assert summarize_engine_outcomes(engine, baseline=baseline) == cached
    execute(engine, {'slot': 'right', 'operation': 'apply', 'pose': pose(14.), 'duration': 1.})
    changed = summarize_engine_outcomes(engine, baseline=baseline)
    assert changed['target']['removedMm3'] > cached['target']['removedMm3']
    assert changed['target']['residualMm3'] < cached['target']['residualMm3']


@pytest.mark.parametrize('field', ['regions', 'affine', 'identity', 'properties'])
def test_unversioned_source_edits_cannot_reuse_outcome_geometry(field):
    engine = prepared()
    sequence_outcome(engine)
    if field == 'regions': engine.scene.regions[0]['target'] = True
    if field == 'affine': engine.scene.affine = engine.scene.affine.copy()
    if field == 'identity': engine.scene.identity = 'changed'
    if field == 'properties': engine.scene.properties['unversioned'] = True
    with pytest.raises(ValueError, match='Source scene changed'):
        sequence_outcome(engine)


def test_incompatible_baseline_and_rejected_material_proposal_have_no_credit():
    engine = prepared()
    before = sequence_outcome(engine)
    # A drill cannot purchase soft-tissue progress regardless of preferences.
    rejected = engine.step({'slot': 'left', 'operation': 'apply', 'pose': pose(13.), 'duration': 1.})
    # Partial free-space advance can be retained, but no incompatible effects.
    assert rejected['status'] in ('partial', 'rejected')
    after = sequence_outcome(engine)
    assert after['dimensions']['targetRemovedMm3']['cumulative'] == before['dimensions']['targetRemovedMm3']['cumulative']
    other = engine.fresh()
    execute(other, {'slot': 'right', 'operation': 'inspect', 'duration': .4})
    with pytest.raises(ValueError, match='exact native action prefix'):
        sequence_outcome(engine, baseline=other)


def test_original_target_overlap_remains_residual_even_after_all_owned_target_is_removed():
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_environment import standardize
    shape = (32, 32, 32)
    target = np.zeros(shape, bool); target[12, 12, 6] = target[12, 12, 12] = True
    exclusion = np.zeros(shape, bool); exclusion[12, 12, 12] = True
    case = CaseData(case_id='GENERATED-target-overlap-accounting', mri=np.zeros(shape, np.float32),
        compartments={'target': target, 'exclusion': exclusion}, affine=np.eye(4),
        source_refs=(SourceRef('generated', 'synthetic://target-overlap-accounting', provenance='simulated'),),
        metadata={'anatomy_layers': {'target': {'kind': 'tumor'},
                                   'exclusion': {'kind': 'brain', 'protected': True}},
                  'surgical_adapter': {'ownershipPrecedence': ['target', 'exclusion']}})
    scene = from_case(case)
    # Standardization keeps source membership while changing the physics profile.
    assert standardize(scene)._original_target_memberships is scene._original_target_memberships
    engine = SurgicalEngine(scene)
    execute(engine, {'slot': 'right', 'operation': 'place', 'pose': {'position': [12., 12., -5.], 'quaternion': [0., 0., 0., 1.]}, 'duration': .1})
    execute(engine, {'slot': 'right', 'operation': 'apply', 'pose': {'position': [12., 12., 6.], 'quaternion': [0., 0., 0., 1.]}, 'duration': .1})
    result = sequence_outcome(engine)
    assert result['target']['fractionRemoved'] == 1
    assert result['originalTarget']['sourceVolumeMm3'] == 2
    assert result['originalTarget']['removedMm3'] == 1
    assert result['originalTarget']['residualMm3'] == 1
    assert result['originalTarget']['fractionRemoved'] == .5
    assert result['originalTarget']['residualCentroidRasMm'] == [12., 12., 12.]
    assert result['dimensions']['targetRemovedMm3']['cumulative'] == 1
    restored = SurgicalEngine.restore(scene, engine.snapshot())
    assert sequence_outcome(restored) == result


def test_categorical_only_scene_does_not_claim_original_target_coverage():
    result = sequence_outcome(prepared())
    assert result['targetScope'] == 'owned categorical target labels'
    assert result['originalTarget']['status'] == 'unavailable'
    assert result['originalTarget']['residualMm3'] is None


def test_empty_coverage_contract_keeps_existing_baseline_snapshot_identity():
    from resectionlab.surgical_rl_protection import GuardConfig
    engine = SurgicalEngine(source(), guard_config=GuardConfig(require_critical_coverage=False))
    snapshot = engine.snapshot()
    assert snapshot['protectionContract']['annotations'] == []
    assert 'hardConstraintPolicy' not in snapshot
    assert SurgicalEngine.restore(engine.scene, snapshot).snapshot() == snapshot
