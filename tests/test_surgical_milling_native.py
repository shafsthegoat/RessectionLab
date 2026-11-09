"""Actual finite-work milling/suction controls; generated, never patient passes."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_milling import PROPERTY, SCHEMA, binding, contract_hash, milling_readiness
from resectionlab.surgical_transition_meter import TransitionMeter
from test_surgical_bone_transport import shell, pose, request


def milling_shell():
    old, _, _, _ = shell()
    labels, regions, source = old.scene.labels, old.scene.regions, old.scene.source
    anchor = int(np.ravel_multi_index((20, 20, 75), labels.shape))
    aperture = sorted(int(np.ravel_multi_index((x, y, 60), labels.shape))
                      for x in range(28, 37) for y in range(28, 37))
    bone = set(map(int, np.flatnonzero(labels == 1)))
    props = {'tissuePhysicsV2': default_profile(), 'boneAnchors': [anchor], 'attachments': []}
    base = SurgicalScene(labels, np.eye(4), regions, source, props)
    proof = {'status': 'reviewed', 'provenance': 'Explicit idealized generated shell; not patient evidence',
             'evidenceSha256': '2'*64}
    raw = {'schema': SCHEMA, 'method': 'milling', **binding(base),
        'apertureCells': aperture, 'retainedSkullCells': sorted(bone-set(aperture)), 'boneAnchorCells': [anchor],
        'entry': {'centerRasMm': [32., 32., 59.5], 'normalRasUnit': [0., 0., 1.],
                  'uRasUnit': [1., 0., 0.], 'halfExtentsMm': [6., 6.],
                  'maximumInwardMm': 20., 'maximumAngleDegrees': 30.},
        'exit': {'axis': 2, 'side': -1, 'planeVoxelBoundary': 59.,
                 'minimumVoxelBoundary': [-.5]*3, 'maximumVoxelBoundary': [63.5, 63.5, 95.5]},
        'evidence': {'scope': 'idealized-engineering', **{name: copy.deepcopy(proof)
                     for name in ('window', 'boneScope', 'innerBoundary', 'exit')}}}
    raw['contentHash'] = contract_hash(raw)
    props[PROPERTY] = raw
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4), regions, source, props))
    return engine, set(aperture)


def checked(engine, operation, *, allow_capacity_partial=False, **kwargs):
    action = request(operation, **kwargs)
    before = engine.state_id
    preview = engine.preview(action, include_view=False)
    assert before == engine.state_id
    result = engine.step(action)
    assert (result['status'] == 'accepted' or allow_capacity_partial and
            result['status'] == 'partial' and result['reasonCode'] == 'BONE_CUTTING_IN_PROGRESS'), result
    assert preview['stateId'] == result['stateId']
    return result


def mill_aperture(engine):
    checked(engine, 'configure', mechanicsRevision=2, clockRevision=1)
    checked(engine, 'place', pose=pose(29.5, 29.5, 55.))
    for x, y in ((29.5, 29.5), (34.5, 29.5), (34.5, 32.),
                 (29.5, 32.), (29.5, 34.5), (34.5, 34.5)):
        for _ in range(6):
            result = checked(engine, 'apply', pose=pose(x, y, 58.3), duration=30., allow_capacity_partial=True)
            if result['status'] == 'accepted':
                break
        else:
            pytest.fail('Frozen six-command finite-work continuation budget exhausted')
    checked(engine, 'withdraw', pose=pose(34.5, 34.5, 55.))


def suction_aperture(engine):
    checked(engine, 'exchange', instrument='frazier_suction')
    checked(engine, 'place', pose=pose(29., 29., 55.))
    for y, xs in ((29., (29., 32., 35.)), (32., (35., 32., 29.)), (35., (29., 32., 35.))):
        for x in xs:
            checked(engine, 'apply', pose=pose(x, y, 58.7), duration=2.)
    checked(engine, 'withdraw', pose=pose(35., 35., 55.))


def test_finite_work_whole_aperture_requires_actual_debris_collection_and_exit():
    source, aperture = milling_shell()
    original = source.snapshot()
    meter = TransitionMeter(160)
    engine = meter.wrap(source).clone()
    mill_aperture(engine)
    assert set(engine.state['removed']) == aperture
    status = milling_readiness(engine)
    assert status['historyVerified'] and status['apertureRemoved']
    assert status['generatedBoneDebrisMm3'] == 81.
    assert status['debrisPresentMm3'] == 81. and status['debrisRemovedMm3'] == 0.
    assert not status['openingComplete'] and not status['apertureCleared']
    assert engine.metrics()['drillCuttingWorkJ'] > 0.
    suction_aperture(engine)
    status = milling_readiness(engine)
    assert status['historyVerified'] and status['debrisConserved'] and status['openingComplete'], status
    assert status['debrisRemovedMm3'] == pytest.approx(81.)
    assert status['debrisPresentMm3'] == 0. and status['debrisEscapedMm3'] == 0.
    assert not status['treatmentEntrySupported']
    assert engine.metrics()['nonTargetRemovedMm3'] == 81. and engine.metrics()['targetRemovedMm3'] == 0.
    assert source.snapshot() == original
    assert SurgicalEngine.restore(source.scene, engine.snapshot()).state_id == engine.state_id
    replay = meter.wrap(source.fresh())
    for action, expected in zip(engine.actions, engine.history):
        result = replay.step(action)
        assert result['status'] == expected['status']
        assert result['reason'] == expected['reason']
        assert result['stateId'] == expected['stateAfter']
    assert replay.snapshot() == engine.snapshot()


def test_suction_full_body_cannot_enter_retained_solid_before_milling():
    engine, _ = milling_shell()
    checked(engine, 'configure', mechanicsRevision=2, clockRevision=1)
    checked(engine, 'exchange', instrument='frazier_suction')
    checked(engine, 'place', pose=pose(32., 32., 55.))
    before = engine.snapshot()
    result = engine.step(request('apply', pose=pose(z=58.7)))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'CRANIAL_ACCESS_MILLING_SUCTION_REQUIRES_CLEARED_VOID'
    assert engine.snapshot() == before


@pytest.fixture(scope='module')
def cleared_milling():
    engine, _ = milling_shell()
    mill_aperture(engine)
    suction_aperture(engine)
    assert milling_readiness(engine)['openingComplete']
    return engine


def test_bone_clearance_never_supplies_dural_or_tumor_entry_permission(cleared_milling):
    engine = cleared_milling.clone()
    checked(engine, 'exchange', instrument='cusa')
    checked(engine, 'place', pose=pose(z=55.))
    before = engine.snapshot()
    result = engine.step(request('place', pose=pose(z=64.)))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'CRANIAL_ACCESS_MILLING_TREATMENT_UNQUALIFIED'
    assert engine.snapshot() == before


def test_withdrawn_flags_cannot_complete_a_holder_still_at_inner_table(cleared_milling):
    engine = cleared_milling.clone()
    checked(engine, 'move', pose=pose(z=58.5))
    assert not milling_readiness(engine)['openingComplete']
    before = engine.snapshot()
    result = engine.step(request('withdraw', pose=pose(z=58.5)))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'CRANIAL_ACCESS_WITHDRAWAL_NOT_AT_EXTERIOR_EXIT'
    assert engine.snapshot() == before


def test_off_aperture_stock_drill_refuses_without_cutting_retained_skull():
    engine, _ = milling_shell()
    checked(engine, 'configure', mechanicsRevision=2, clockRevision=1)
    checked(engine, 'place', pose=pose(29., 32., 55.))
    before = engine.snapshot()
    result = engine.step(request('apply', pose=pose(29., 32., 58.3), duration=30.))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'CRANIAL_ACCESS_BONE_WORK_OUTSIDE_DECLARED_SCOPE'
    assert engine.snapshot() == before


@pytest.mark.parametrize('parameters', [{'radiusMm': 1.}, {'depthMm': 1.}])
def test_milling_stock_geometry_is_not_shrunk_by_actions(parameters):
    engine, _ = milling_shell()
    checked(engine, 'configure', mechanicsRevision=2, clockRevision=1)
    before = engine.snapshot()
    result = engine.step(request('place', pose=pose(29.5, 29.5, 55.), parameters=parameters))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'CRANIAL_ACCESS_MILLING_STOCK_GEOMETRY_REQUIRED'
    assert engine.snapshot() == before


def test_milling_requires_explicit_stock_revision_and_shared_clock():
    engine, _ = milling_shell()
    before = engine.snapshot()
    result = engine.step(request('place', pose=pose(29.5, 29.5, 55.)))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'CRANIAL_ACCESS_MILLING_STOCK_AND_CLOCK_REQUIRED'
    assert engine.snapshot() == before


def milling_case():
    from resectionlab.core import CaseData, SourceRef, thaw_json
    from resectionlab.surgical_scene import from_case
    engine, _ = milling_shell()
    props = copy.deepcopy(engine.scene.properties)
    raw = props.pop(PROPERTY)
    regions = copy.deepcopy(engine.scene.regions)
    for row in regions:
        row['color'] = '#dfd6bf' if row['material'] == 'bone' else '#d56089'
    metadata = {'generated_signal': True, 'surgical_properties': props,
                'anatomy_layers': {row['name']: {'kind': 'skull' if row['material'] == 'bone' else 'tumor',
                                                'color': row['color']} for row in regions}}
    base = CaseData(case_id='generated-finite-milling-control', mri=engine.scene.labels.astype(np.float32),
        affine=engine.scene.affine, compartments={row['name']: engine.scene.labels == row['id'] for row in regions},
        metadata=metadata, source_refs=(SourceRef('generated', 'generated:finite-milling', provenance='simulated'),))
    raw.update(binding(from_case(base, regions)))
    raw['contentHash'] = contract_hash(raw)
    metadata = thaw_json(base.metadata)
    metadata['surgical_properties'][PROPERTY] = raw
    case = replace(base, metadata=metadata)
    return case, SurgicalEngine(from_case(case, regions))


def test_milling_native_case_binding_cold_save_and_collected_mass(tmp_path):
    from resectionlab.imaging import load_case, read_case_artifacts, save_case
    from resectionlab.surgical_scene import from_case
    case, engine = milling_case()
    original = engine.snapshot()
    mill_aperture(engine)
    uncollected = engine.snapshot()
    suction_aperture(engine)
    complete = engine.snapshot()
    artifacts = {'surgical_toolbox': {'mapping': engine.scene.regions,
        'branches': {'main': original, 'debris-retained': uncollected, 'milled-cleared': complete},
        'branchId': 'milled-cleared', 'position': len(engine.actions),
        'parents': {'main': None, 'debris-retained': {'branchId': 'main', 'position': 0},
                    'milled-cleared': {'branchId': 'main', 'position': 0}}}}
    path = save_case(case, tmp_path/'milled.ressectionlab', artifacts=artifacts)
    reopened = load_case(path)
    saved = read_case_artifacts(path)
    assert reopened.semantic_hash == case.semantic_hash
    for name, snapshot in artifacts['surgical_toolbox']['branches'].items():
        restored = SurgicalEngine.restore(from_case(reopened, engine.scene.regions), saved['surgical_toolbox']['branches'][name])
        assert restored.snapshot() == snapshot
        assert milling_readiness(restored)['openingComplete'] == (name == 'milled-cleared')
    for changed in (replace(case, mri=np.zeros(case.mri.shape, np.float32)),
                    replace(case, affine=np.diag([1.1, 1., 1., 1.])),
                    replace(case, metadata={**dict(case.metadata), 'extra_constraint': True})):
        with pytest.raises(ValueError, match='MILLING_CONTRACT_INVALID'):
            from_case(changed, engine.scene.regions)
