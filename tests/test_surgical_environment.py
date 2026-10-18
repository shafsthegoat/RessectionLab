"""Cross-source environment admission, historical replay, and regression coverage."""
import copy
import numpy as np
from resectionlab.core import CaseData, SourceRef
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_environment import SCHEMA, default_target_names, standardize
from resectionlab.surgical_scene import from_case


def call(session, operation, args):
    return session.execute(operation, args, _Request(operation), lambda *_: None)


def test_contract_preserves_oblique_anisotropic_source_and_reports_missing_anatomy():
    affine = np.array([[0, -2, 0, 30], [.7, 0, .2, -10], [0, 0, 3, 80], [0, 0, 0, 1.]])
    mask = np.zeros((5, 6, 7), bool)
    mask[2, 3, 4] = True
    case = CaseData(case_id='different-reconstruction', mri=np.indices(mask.shape).sum(axis=0).astype('float32'),
                    compartments={'supplied lesion': mask}, affine=affine, source_refs=(SourceRef('fixture','synthetic://environment-test',provenance='simulated'),),
                    metadata={'anatomy_layers': {'supplied lesion': {'kind': 'tumor'}}})
    old = from_case(case)
    before = old.identity
    scene = standardize(old)
    env = scene.descriptor()['environment']
    assert env['schema'] == SCHEMA and env['physicsSchema'] == 'resectionlab.tissue-physics.v2'
    assert env['anatomy'] == {'brain': 'unknown', 'tumor': 'supplied', 'bone': 'unknown', 'vessels': 'unknown'}
    assert env['target']['initialVolumeMm3'] == scene.voxel_mm3
    np.testing.assert_array_equal(scene.affine, affine)
    np.testing.assert_array_equal(scene.labels, old.labels)
    assert old.identity == before and not old.properties
    assert env['clinicalValidation'] is False


def test_standard_target_selector_does_not_enlarge_primary_task_to_flair():
    masks = {name: np.ones((2, 2, 2), bool) for name in ('enhancing_target', 'FLAIR_abnormality', 'nonenhancing_or_necrotic_core')}
    case = CaseData(case_id='source-labels', mri=np.zeros((2, 2, 2), np.float32), compartments=masks, affine=np.eye(4), source_refs=(SourceRef('fixture','synthetic://environment-test',provenance='simulated'),),
                    metadata={'anatomy_layers': {name: {'kind': 'tumor'} for name in masks}})
    assert default_target_names(case) == ['enhancing_target']
    scene = standardize(from_case(case), default_target_names(case))
    assert [r['name'] for r in scene.regions if r['target']] == ['enhancing_target']


def test_noop_exchange_and_unchanged_mapping_preserve_history_and_complete_timeline(tmp_path):
    session = BridgeSession(tmp_path / 'transfers')
    try:
        f = call(session, 'createSurgicalFixture', {})
        def cmd(command, previous=None, **extra):
            return call(session, 'surgicalRehearsal', {'caseHash': f['caseHash'], 'command': command,
                **({'expectedStateId': previous['view']['stateId']} if previous else {}), **extra})
        created = cmd('create')
        noop = cmd('step', created, action={'slot':'right','operation':'exchange','instrument':'cusa','duration':1})
        assert noop['position'] == 0 and noop['view']['stateId'] == created['view']['stateId'] and not noop['changed']
        first = cmd('step', noop, action={'slot':'right','operation':'exchange','instrument':'irrigation','duration':.1})
        branch = cmd('branch', first)
        same = cmd('create', branch, mapping=branch['view']['scene']['regions'])
        assert same['branchId'] == branch['branchId'] and same['position'] == 1
        assert same['view']['stateId'] == branch['view']['stateId'] and not same['changed']
        undo = cmd('undo', same)
        assert undo['position'] == 0 and len(undo['timeline']) == 1 and not undo['view']['history']
        mapping = copy.deepcopy(same['view']['scene']['regions'])
        mapping[0]['target'] = True
        changed = cmd('create', undo, mapping=mapping)
        assert changed['position'] == 0 and changed['previousScenes']
        recovered = cmd('selectScene', changed, sceneIndex=0)
        assert recovered['branchId'] == branch['branchId'] and len(recovered['timeline']) == 1
        redo = cmd('redo', recovered)
        assert redo['view']['stateId'] == branch['view']['stateId']
    finally:
        session.close()


def test_native_preparation_uses_same_profile_without_inventing_missing_layers(tmp_path):
    from test_surgical_patient import fixture
    collection = fixture(tmp_path, status='unreviewed')
    manifest_before = (collection.root / 'patient-anatomy.json').read_bytes()
    session = BridgeSession(tmp_path / 'transfers')
    try:
        binding = call(session, 'anatomyOpen', {'path': str(collection.root / 'patient-anatomy.json')})
        prepared = call(session, 'prepareSurgicalEnvironment', {'collectionId': binding['collectionId'],
            'snapshotId': binding['document']['snapshot_id'], 'layers': {'brain': 'brain', 'tumor': 'tumor'},
            'allowUnreviewedRegistration': False})
        current = call(session, 'surgicalRehearsal', {'caseHash': prepared['caseHash'], 'command': 'inspect'})
        env = current['view']['scene']['environment']
        assert env['standardized'] and env['physicsSchema'] == 'resectionlab.tissue-physics.v2'
        assert env['anatomy']['vessels'] == 'unknown' and env['anatomy']['bone'] == 'unknown'
        assert current['position'] == 0
    finally:
        session.close()
    assert (collection.root / 'patient-anatomy.json').read_bytes() == manifest_before


def test_rl_loader_uses_saved_desktop_mapping_and_profile_without_saved_removal(tmp_path):
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
    session = BridgeSession(tmp_path / 'transfers')
    try:
        f = call(session, 'createSurgicalFixture', {})
        created = call(session, 'surgicalRehearsal', {'caseHash':f['caseHash'],'command':'create'})
        path = tmp_path / 'standard.ressectionlab'
        call(session, 'saveCase', {'caseHash':f['caseHash'],'path':str(path),'workspace':{},'overwrite':False})
        scene = load_environment(path)
        assert scene.identity == created['view']['scene']['sceneId']
        env = SurgicalRLEnv.from_patient(path, config=RLConfig(require_critical_coverage=False))
        _, info = env.reset(seed=3)
        assert env.scene.identity == scene.identity
        assert info['protection']['status'] == 'unknown-coverage'
        assert not env.engine.state['removed']
    finally:
        session.close()


def test_source_annotation_and_review_evidence_survive_standard_export(tmp_path):
    from test_surgical_patient import fixture
    from resectionlab.surgical_patient import prepare_surgical_environment
    from resectionlab.imaging import load_case
    collection = fixture(tmp_path)
    tumor = collection.layer('tumor')
    tumor['provenance'] = {'kind': 'observed_source_annotation', 'method': 'source label 4',
                           'source_sha256': 'explicit-annotation-binding'}
    tumor['review_status'] = 'reviewed'
    tumor['coverage'] = 'Source enhancement compartment; other compartments separate.'
    collection.save()
    before = (collection.root / 'patient-anatomy.json').read_bytes()
    report = prepare_surgical_environment(collection.root, tmp_path / 'standard',
                                           layer_ids={'brain': 'brain', 'tumor': 'tumor'})
    case = load_case(report['casePath'])
    evidence = case.metadata['anatomy_layers']['tumor']
    assert evidence['provenance'] == 'observed'
    assert evidence['review_status'] == 'reviewed'
    assert evidence['coverage'] == tumor['coverage']
    assert evidence['source_provenance']['source_sha256'] == 'explicit-annotation-binding'
    assert case.metadata['anatomy_layers']['brain']['provenance'] == 'estimated'
    assert report['environment']['anatomy']['vessels'] == 'unknown'
    assert (collection.root / 'patient-anatomy.json').read_bytes() == before
