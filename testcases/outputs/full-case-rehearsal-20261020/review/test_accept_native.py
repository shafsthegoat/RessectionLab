"""Acceptance auditor tests use actual public engineering transitions only."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from resectionlab.core import CaseData, SourceRef
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import OriginalFile, save_workspace

spec = importlib.util.spec_from_file_location('independent_accept_native', Path(__file__).with_name('accept_native.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(tmp_path):
    target = np.zeros((16, 16, 16), bool)
    target[6:10, 6:10, 5:9] = True
    target[12, 12, 12] = True
    case = CaseData(case_id='acceptance-engineering', mri=np.arange(target.size, dtype=np.float32).reshape(target.shape),
                    affine=np.eye(4), compartments={'target': target},
                    source_refs=(SourceRef('explicit-fixture', 'synthetic://acceptance', provenance='simulated'),),
                    metadata={'anatomy_layers': {'target': {'kind': 'tumor'}},
                              'surgical_properties': {'tissuePhysicsV2': default_profile()}})
    engine = SurgicalEngine(from_case(case))
    initial = engine.snapshot()
    originals = [OriginalFile('fixture.bin', 'synthetic://acquired-fixture', audit.file_bytes_hash(b'original bytes'),
                              b'original bytes', 'metadata')]
    base = {'mapping': engine.scene.regions, 'branchId': 'main', 'position': 0,
            'branches': {'main': initial}, 'parents': {'main': None}}
    source = tmp_path / 'source.ressectionlab'
    save_workspace(case, source, sources=[], imaging=None, episode=None, selection=None,
                   originals=originals, artifacts={'surgical_toolbox': base})
    for action in [
        {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': {'position': [7., 7., 1.], 'quaternion': [0., 0., 0., 1.]},
         'parameters': {'radiusMm': 1.5, 'forceN': 0.}},
        {'operation': 'apply', 'pose': {'position': [7., 7., 7.], 'quaternion': [0., 0., 0., 1.]}, 'duration': 1.},
    ]:
        result = engine.step({'slot': 'left', 'duration': .2, **action})
        assert result['status'] in ('accepted', 'partial'), result
    assert engine.metrics()['targetRemovedMm3'] > 0
    refused = engine.preview({'slot': 'left', 'operation': 'not-a-tool-operation', 'duration': .2}, include_view=False)
    assert refused['status'] == 'rejected'
    record = copy.deepcopy(base)
    record.update(branchId='checked', position=len(engine.actions))
    record['branches']['checked'] = engine.snapshot()
    record['parents']['checked'] = {'branchId': 'main', 'position': 0}
    record['fullCase'] = {'checked': {'lastStateId': engine.state_id, 'status': 'stopped',
                                      'rejections': [refused], 'phases': []}}
    native = tmp_path / 'episode.ressectionlab'
    def save():
        save_workspace(case, native, sources=[], imaging=None, episode=None, selection=None,
                       originals=originals, artifacts={'surgical_toolbox': record})
    save()
    return source, native, engine, record, save


def test_actual_public_engineering_replay_source_residual_and_proposal_report(tmp_path):
    source, native, engine, _, _ = fixture(tmp_path)
    source_bytes, native_bytes = source.read_bytes(), native.read_bytes()
    report = audit.verify(native, 'checked', source=source, engineering=True, report=tmp_path / 'receipt.json',
                          cache_directory=tmp_path / 'cache')
    assert report['auditStatus'] == 'passed' and report['episodeCompletionStatus'] == 'stopped'
    assert report['stateId'] == engine.state_id
    assert report['canonicalReplay']['perActionStateAssertions'] == len(engine.actions)
    assert report['canonicalReplay']['cacheUsed'] is False
    assert report['canonicalReplay']['verifiedCachePublication']['publications'] == 1
    assert report['accounting']['target']['removedOwnedCells'] == len(engine.state['removed'])
    assert report['accounting']['target']['retainedOwnedCells'] > 0
    assert len(report['accounting']['target']['residualConnectedRegions']) >= 2
    assert report['recordedUncommittedProposals'][0]['status'] == 'rejected'
    assert report['vascular']['severedPatentCount'] == 0
    assert report['guiReview'].startswith('pending')
    assert source.read_bytes() == source_bytes and native.read_bytes() == native_bytes


def test_tampered_checkpoint_is_not_admitted(tmp_path):
    source, native, _, record, save = fixture(tmp_path)
    record['branches']['checked']['checkpoint']['state']['fluidAddedMm3'] += 1.
    save()
    with pytest.raises(ValueError, match='Checkpoint differs'):
        audit.verify(native, 'checked', source=source, engineering=True, report=tmp_path / 'receipt.json',
                     cache_directory=tmp_path / 'cache')
    assert not list((tmp_path / 'cache').glob('entry-*'))


def test_mismatched_source_hash_refuses_before_replay(tmp_path):
    source, native, _, _, _ = fixture(tmp_path)
    with pytest.raises(AssertionError, match='source file hash'):
        audit.verify(native, 'checked', source=source, expected_source_sha256='0' * 64,
                     engineering=True, report=tmp_path / 'receipt.json')


def test_direct_protection_membership_counts_union_without_changing_target_denominator(tmp_path):
    _, _, engine, _, _ = fixture(tmp_path)
    targets = set(map(int, np.flatnonzero(engine.scene.labels == 1)))
    removed = set(engine.state['removed'])
    one_removed, one_retained = next(iter(removed)), next(iter(targets - removed))
    contract = {'contentHash': 'explicit-derived-membership-fixture', 'annotations': [
        {'structureId': 'first', 'cells': [one_removed, one_retained, 0]},
        {'structureId': 'second', 'cells': [one_retained]},
    ]}
    result = audit.protected_target_membership(engine.scene, engine.state, contract)
    assert result['originalOwnedTargetDenominatorCells'] == len(targets)
    assert result['originalOwnedTargetDirectCells'] == 2
    assert result['retainedOwnedTargetDirectCells'] == result['removedOwnedTargetDirectCells'] == 1
    assert sum(row['originalOwnedTargetDirectCells'] for row in result['perStructureNonexclusive']) == 3


def test_wound_and_fluid_checks_use_actual_injury_sealing_and_source_face_geometry():
    from resectionlab.surgical_scene import SurgicalScene
    labels = np.zeros((24, 24, 20), np.uint16)
    labels[8, 7:10, 8] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'vessel', 'material': 'vessel'}],
                           {'kind': 'explicit-audit-vessel-fixture'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    assert audit.wound_audit(engine.state, scene)['woundCount'] == 0
    samples = [audit.vascular_sample(engine, 0)]
    pose = lambda z: {'position': [8., 8., z], 'quaternion': [0., 0., 0., 1.]}
    for action in [
        {'operation': 'exchange', 'instrument': 'micro_scissors'},
        {'operation': 'place', 'pose': pose(4.), 'parameters': {'radiusMm': 1.5, 'jawOpeningMm': 0.}},
        {'operation': 'apply', 'pose': pose(7.), 'duration': 1.},
        {'operation': 'deactivate'},
        {'operation': 'inspect', 'duration': 1.},
        {'operation': 'withdraw'},
        {'operation': 'exchange', 'instrument': 'bipolar_forceps'},
        {'operation': 'place', 'pose': pose(4.), 'parameters': {'radiusMm': 1.5, 'jawOpeningMm': 0.}},
        {'operation': 'apply', 'pose': pose(7.), 'parameters': {'power': 10.}, 'duration': 1.},
        {'operation': 'deactivate'},
        {'operation': 'inspect', 'duration': 1.},
        {'operation': 'withdraw'},
        {'operation': 'exchange', 'instrument': 'irrigation'},
        {'operation': 'place', 'pose': pose(4.), 'parameters': {'radiusMm': 3., 'depthMm': .1, 'flowMm3PerSecond': 2., 'forceN': 0.}},
        {'operation': 'apply', 'pose': pose(7.), 'duration': 1.},
        {'operation': 'deactivate'},
        {'operation': 'withdraw'},
        {'operation': 'exchange', 'instrument': 'frazier_suction'},
        {'operation': 'place', 'pose': pose(4.), 'parameters': {'radiusMm': 3., 'depthMm': .1, 'flowMm3PerSecond': 2., 'forceN': 0.}},
        {'operation': 'apply', 'pose': pose(7.), 'duration': 1.},
        {'operation': 'deactivate'},
        {'operation': 'inspect', 'duration': 1.},
    ]:
        outcome = engine.step({'slot': 'left', 'duration': .2, **action})
        assert outcome['status'] in ('accepted', 'partial'), (action, outcome)
        audit.fluid_audit(engine.state)
        audit.wound_audit(engine.state, scene)
        samples.append(audit.vascular_sample(engine, len(engine.actions)))
    result = audit.wound_audit(engine.state, scene)
    assert result['woundCount'] > 0 and result['severedFaceCount'] > 0
    assert result['activeWounds'] == result['severedPatentCount'] == 0
    chain = audit.causal_vascular_audit(samples, engine.actions, scene.properties['tissuePhysicsV2']['vessel']['coagulationDose'], required=True)
    assert chain['completeChain'] and chain['actualBipolarDose'] and chain['suctionWithoutVesselRepair']
    corrupt_samples = copy.deepcopy(samples)
    corrupt_samples[-1]['faces'][next(iter(corrupt_samples[-1]['faces']))]['coagulationDose'] = 0.
    with pytest.raises(AssertionError, match='actual bipolar dose'):
        audit.causal_vascular_audit(corrupt_samples, engine.actions, scene.properties['tissuePhysicsV2']['vessel']['coagulationDose'], required=True)
    corrupt = copy.deepcopy(engine.state)
    next(iter(corrupt['vesselsV2']['faces'].values()))['patent'] = True
    with pytest.raises(AssertionError, match='Severed vessel became patent'):
        audit.wound_audit(corrupt, scene)


def test_raw_brain_membership_overlap_does_not_change_owned_target_accounting():
    from resectionlab.surgical_scene import SurgicalScene
    target = np.zeros((16, 16, 16), bool)
    target[6:10, 6:10, 5:9] = True
    brain = target.copy()
    brain[4, 4, 4] = True
    case = CaseData(case_id='explicit-overlapping-annotation-fixture', mri=np.zeros(target.shape, np.float32),
                    affine=np.eye(4), brain_mask=brain, compartments={'brain': brain, 'target': target},
                    source_refs=(SourceRef('overlap-fixture', 'synthetic://overlap-fixture', provenance='simulated'),),
                    metadata={'anatomy_layers': {'brain': {'kind': 'brain'}, 'target': {'kind': 'tumor'}}})
    # Explicit ordering makes the owned target take precedence over the raw
    # brain annotation, exactly as declared in the patient adapter contract.
    scene = from_case(case, [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}])
    engine = SurgicalEngine(scene)
    for action in [
        {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': {'position': [7., 7., 1.], 'quaternion': [0., 0., 0., 1.]}},
        {'operation': 'apply', 'pose': {'position': [7., 7., 7.], 'quaternion': [0., 0., 0., 1.]}, 'duration': 1.},
    ]:
        assert engine.step({'slot': 'left', 'duration': .2, **action})['status'] == 'accepted'
    owned = audit.source_accounting(scene, engine.state)
    raw = audit.annotation_memberships(case, scene, engine.state['removed'])
    removed = len(engine.state['removed'])
    assert removed > 0
    assert owned['ownedHealthyOrUnknownNonTargetRemoval']['cells'] == 0
    assert owned['materials']['soft_tissue']['removedCells'] == 0
    assert owned['target']['removedOwnedCells'] == removed
    assert owned['target']['originalOwnedCells'] == int(target.sum())
    assert raw['groups']['brain_mask']['removedSourceCellsWithinAnnotation'] == removed
    assert raw['groups']['compartments']['brain']['removedSourceCellsWithinAnnotation'] == removed
    assert raw['groups']['source_compartments']['brain']['removedSourceCellsWithinAnnotation'] == removed
    assert raw['groups']['brain_mask']['overlapWithRawTargetAnnotationCells'] == int(target.sum())
    assert raw['rawTargetAnnotationUnionCells'] == int(target.sum())
    assert raw['removedSourceCellsWithinRawTargetAnnotation'] == removed
    assert raw['rawTargetAnnotationMembershipRemovedPercent'] == 100. * removed / int(target.sum())
