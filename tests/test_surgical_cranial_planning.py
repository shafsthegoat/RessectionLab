"""Pure desktop adapter controls; generated native integration is separate."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import surgical_cranial_planning as planning


def declaration():
    return {'limits': {'max_bone_mm3': 10., 'max_preparation_nonbone_mm3': 4.,
        'max_work_nonbone_mm3': 2., 'max_native_actions': 80, 'max_public_calls': 200,
        'max_wall_seconds': 30., 'max_bone_contact_replans': 6,
        'max_tissue_contact_replans': 8, 'export_replay_reserve_calls': 40},
        'openingSkill': {'kind': 'opening', 'actions': [{'operation': 'inspect'}]},
        'preparationActions': [{'operation': 'deactivate', 'slot': 'left'}],
        'outsidePoses': {}, 'qualification': {'status': 'reviewed', 'sourceBinding': {'source': 'exact'},
            'evidenceSha256': '1'*64}, 'workRoutes': [{'id': 'near', 'motion': {'operation': 'apply',
                'slot': 'left', 'duration': .2, 'pose': {'position': [1., 0., 0.], 'quaternion': [0., 0., 0., 1.]}}}]}


def guard():
    return {'schema': 'guard', 'sceneId': 'base', 'sourceBinding': 'same-grid',
        'contentHash': 'base-seal', 'config': {'require_critical_coverage': True},
        'required': ['brainstem'], 'clearanceMm': 2., 'annotations': [{'structureId': 'brainstem',
            'cells': [3, 4], 'provenance': 'source-review', 'reviewStatus': 'reviewed'}]}


def planning_root(value):
    proof = {'status': 'reviewed', 'provenance': 'Explicit generated source geometry only',
        'evidenceSha256': '2'*64}
    return {'planning': value, 'openingContract': {'evidence': {'window': copy.deepcopy(proof)}},
        'sourceSnapshot': {'cranialStagesContract': {'evidence': {'work': copy.deepcopy(proof)}}}}


def test_root_guard_can_only_add_membership_without_replacing_review_or_configuration():
    base = guard(); extra = copy.deepcopy(base);extra['sceneId'] = 'other';extra['contentHash'] = 'new'
    extra['annotations'][0]['cells'].append(5)
    assert planning._protection_contains(base, extra)
    for mutation in ('missing', 'cells', 'provenance', 'review', 'clearance', 'config'):
        bad = copy.deepcopy(extra)
        if mutation == 'missing': bad = None
        if mutation == 'cells': bad['annotations'][0]['cells'] = [3]
        if mutation == 'provenance': bad['annotations'][0]['provenance'] = 'invented'
        if mutation == 'review': bad['annotations'][0]['reviewStatus'] = 'model-only'
        if mutation == 'clearance': bad['clearanceMm'] = 1.
        if mutation == 'config': bad['config'] = {}
        assert not planning._protection_contains(base, bad)


@pytest.mark.parametrize('mutation,code', [('model', 'REVIEWED_SOURCE_QUALIFICATION_REQUIRED'),
    ('source', 'REVIEWED_SOURCE_QUALIFICATION_REQUIRED'), ('evidence', 'REVIEWED_SOURCE_QUALIFICATION_REQUIRED'),
    ('reserve', 'EXPLICIT_SHARED_NATIVE_AND_EXPORT_BUDGET_REQUIRED'),
    ('cap', 'EXPLICIT_SHARED_NATIVE_AND_EXPORT_BUDGET_REQUIRED'),
    ('opening', 'SOURCE_OPENING_SKILL_REQUIRED'), ('prep', 'SOURCE_PREPARATION_PROGRAM_REQUIRED'),
    ('routes', 'SOURCE_CUSA_WORK_ROUTES_REQUIRED')])
def test_missing_qualification_inventory_or_explicit_budgets_block(monkeypatch, mutation, code):
    monkeypatch.setattr(planning, 'source_binding', lambda _: {'source': 'exact'})
    value = declaration();root = planning_root(value)
    assert planning._planning_errors(None, root, {}) == []
    if mutation == 'model': value['qualification']['status'] = 'model-only'
    if mutation == 'source': value['qualification']['sourceBinding'] = {'source': 'different'}
    if mutation == 'evidence': value['qualification']['evidenceSha256'] = 'unreviewed'
    if mutation == 'reserve': del value['limits']['export_replay_reserve_calls']
    if mutation == 'cap': value['limits']['max_wall_seconds'] = float('inf')
    if mutation == 'opening': value['openingSkill']['kind'] = 'new-geometry'
    if mutation == 'prep': value['preparationActions'] = []
    if mutation == 'routes': value['workRoutes'][0]['motion']['operation'] = 'move'
    assert code in [x['code'] for x in planning._planning_errors(None, root, {})]


def test_explicit_unknown_blocker_is_never_erased_by_reviewed_label(monkeypatch):
    monkeypatch.setattr(planning, 'source_binding', lambda _: {'source': 'exact'})
    p = declaration(); p['qualification']['blockers'] = [{'code': 'HARD_UNKNOWN', 'message': 'Unknown remains protected.'}]
    assert planning._planning_errors(None, planning_root(p), {}) == p['qualification']['blockers']


@pytest.mark.parametrize('where', ['opening', 'stage'])
def test_one_unreviewed_source_proof_blocks_only_that_sibling_root(monkeypatch, where):
    monkeypatch.setattr(planning, 'validate_registry', lambda *args: {})
    monkeypatch.setattr(planning, 'source_binding', lambda _: {'source': 'exact'})
    roots = []
    for i in range(2):
        r = planning_root(declaration());r.update(id='root-' + str(i), label='Source root ' + str(i),
            openingProperty='cranialAccessV1', sceneId='root-scene-' + str(i))
        r['sourceSnapshot']['stateId'] = 'root-source-' + str(i)
        roots.append(r)
    evidence = (roots[1]['openingContract']['evidence'] if where == 'opening' else
        roots[1]['sourceSnapshot']['cranialStagesContract']['evidence'])
    next(iter(evidence.values()))['status'] = 'unreviewed'
    scene = SimpleNamespace(identity='base', regions=[], properties={})
    record = {'branchId': 'source', 'position': 0, 'branches': {
        'source': {'sceneId': 'base', 'stateId': 'common', 'actions': []}},
        planning.FIELD: {'roots': roots, 'branchRoots': {}}}
    dto = planning.cranial_planning_availability(scene, record)
    assert dto['available'] and dto['roots'][0]['available'] and not dto['roots'][1]['available']
    assert dto['roots'][1]['blockers'][0]['code'] == 'SOURCE_STAGE_OR_OPENING_EVIDENCE_REQUIRED'


def test_actual_zero_transition_contract_and_stage_evidence_shapes_are_accepted(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    from test_surgical_membrane_transport import joint_source
    def forbidden(*args, **kwargs):
        pytest.fail('Zero-transition source-schema control attempted native work')
    monkeypatch.setattr(SurgicalEngine, 'preview', forbidden)
    monkeypatch.setattr(SurgicalEngine, 'step', forbidden)
    monkeypatch.setattr(planning, 'source_binding', lambda _: {'source': 'exact'})
    engine, stage, *_ = joint_source()
    root = planning_root(declaration())
    root['openingContract'] = engine._cranial_access.as_dict()
    root['sourceSnapshot'] = engine.snapshot()
    assert root['openingContract']['evidence']['scope'] == 'idealized-engineering'
    assert planning._reviewed_evidence(root['openingContract']['evidence'])
    assert planning._reviewed_evidence(stage.as_dict()['evidence'])
    assert planning._planning_errors(engine.scene, root, engine.snapshot()) == []
    assert not engine.actions and not engine.history
    for invalid in ({'scope': 'idealized-engineering'}, {'scope': ''}, {'scope': 1},
                    {'unexpected-metadata': 'value'}):
        assert not planning._reviewed_evidence(invalid)


def test_source_owned_proposer_changes_stage_and_ranks_only_existing_routes(monkeypatch):
    from resectionlab import surgical_joint_cranial_skills as skills, surgical_scene
    ready = {}; monkeypatch.setattr(skills, '_readiness', lambda _: ready)
    monkeypatch.setattr(surgical_scene, 'original_target_memberships', lambda _: {'original': np.array([1, 3])})
    scene = SimpleNamespace(regions=[{'id': 2, 'target': True}], labels=np.array([0, 2, 0, 2]),
        points=lambda ids: np.array([[float(i), 0., 0.] for i in ids]))
    current = SimpleNamespace(scene=scene, state_id='actual-prefix', state={'removed': []})
    p = declaration()
    for i in range(8):
        route = copy.deepcopy(p['workRoutes'][0]);route['id'] = 'declared-' + str(i)
        route['motion']['pose']['position'][0] = float(i)
        p['workRoutes'].append(route)
    before = copy.deepcopy(p); propose = planning._proposer(p)
    opening = propose(current)
    assert opening[0]['kind'] == 'opening' and opening[0]['sourceStateId'] == 'actual-prefix'
    ready.update(bonePrefixVerified=True)
    assert propose(current)[0]['kind'] == 'preparation'
    ready['duraCleared'] = True
    work = propose(current)
    assert len(work) == 6 and all(x['kind'] == 'cusa' for x in work)
    assert all(x['motion'] in [r['motion'] for r in p['workRoutes']] for x in work)
    current.state['removed'] = [1, 3]
    assert propose(current) == [] and p == before


def test_noncranial_none_but_skull_without_registry_is_precisely_blocked():
    scene = SimpleNamespace(identity='base', properties={}, regions=[])
    record = {'branchId': 'source', 'branches': {'source': {'stateId': 's'}}}
    assert planning.cranial_planning_availability(scene, record) is None
    scene.regions = [{'material': 'bone'}]
    dto = planning.cranial_planning_availability(scene, record)
    assert dto['present'] and not dto['available']
    assert dto['blockers'][0]['code'] == 'QUALIFIED_SOURCE_OPENING_ROOT_REQUIRED'


def test_availability_uses_common_source_even_after_selecting_generated_root(monkeypatch):
    monkeypatch.setattr(planning, 'validate_registry', lambda *args: {})
    monkeypatch.setattr(planning, '_planning_errors', lambda *args: [])
    scene = SimpleNamespace(identity='base', properties={}, regions=[])
    root = {'id': 'opening', 'label': 'Declared', 'openingProperty': 'cranialAccessV1',
        'sceneId': 'root', 'sourceSnapshot': {'stateId': 'root-source'}}
    record = {'branchId': 'source', 'position': 0, 'branches': {
        'source': {'sceneId': 'base', 'stateId': 'common', 'actions': []},
        'generated': {'sceneId': 'root', 'stateId': 'later', 'actions': [{}]}},
        planning.FIELD: {'branchRoots': {'generated': 'opening'}, 'roots': [root]}}
    assert planning.cranial_planning_availability(scene, record)['available']
    record['branchId'] = 'generated';record['position'] = 1
    dto = planning.cranial_planning_availability(scene, record)
    assert dto['sourceStateId'] == 'common' and not dto['available'] and dto['roots'][0]['available']


def test_late_install_check_is_atomic_and_keeps_source_cursor(monkeypatch):
    monkeypatch.setattr(planning, 'validate_cranial_report', lambda *args: None)
    monkeypatch.setattr(planning, 'bind_branch_root', lambda base, staged, branch, root:
        staged[planning.FIELD]['branchRoots'].update({branch: root}))
    class Budget:
        calls = 0
        fail = False
        def check(self):
            self.calls += 1
            if self.fail and self.calls == 3: raise TimeoutError('late publication')
    record = {'branchId': 'source', 'position': 0, 'branches': {'source': {'actions': []}},
        'parents': {'source': None}, planning.FIELD: {'branchRoots': {}}}
    before = copy.deepcopy(record);budget = Budget()
    result = {'budget': budget, 'report': {}, 'branches': [(SimpleNamespace(
        snapshot=lambda: {'actual': 'snapshot'}, history=[{'actual': 'row'}]),
        {'branchId': 'new', 'rootId': 'opening'})]}
    result['report']['alternatives'] = [result['branches'][0][1]]
    budget.fail = True
    with pytest.raises(TimeoutError): planning.install_cranial_comparison(record, result, None)
    assert record == before
    budget.fail = False;budget.calls = 0
    planning.install_cranial_comparison(record, result, None)
    assert record['branchId'] == 'source' and record['position'] == 0
    assert record['parents']['new'] is None and record[planning.FIELD]['branchRoots']['new'] == 'opening'
    assert record['timeline']['new'] == [{'actual': 'row'}]


def test_snapshot_counts_preserve_full_original_membership_and_bpw(monkeypatch):
    from resectionlab import surgical_cranial_stages as stages, surgical_scene, surgical_joint_cranial_exit as exits
    target = np.array([0, 1, 6, 7])
    monkeypatch.setattr(surgical_scene, 'original_target_memberships', lambda _: {'original': target})
    monkeypatch.setattr(stages, 'source_partition_sets', lambda _: {'bone': {0}, 'preparation': {2}, 'work': {1, 3, 6, 7}})
    monkeypatch.setattr(exits, 'joint_exit_complete', lambda _: False)
    labels = np.array([1, 2, 3, 4, 0, 0, 2, 2])
    scene = SimpleNamespace(labels=labels, regions=[{'id': 2, 'target': True}], voxel_mm3=.125,
        _original_target_proof=True, points=lambda ids: np.array([np.unravel_index(i, (2, 2, 2)) for i in ids]))
    source = SimpleNamespace(_cranial_stages='actual-contract', _cranial_access='actual-opening')
    snapshot = {'actions': [], 'checkpoint': {'state': {'removed': [0, 1, 2, 3], 'elapsedSeconds': 4.}}}
    goal, ledger, outside = planning._snapshot_quantities(scene, snapshot, source, [])
    assert goal['originalTargetCells'] == 4 and goal['removedOriginalTargetCells'] == 2
    assert goal['creditedOriginalTargetCells'] == 1 and goal['originalTargetVolumeMm3'] == .5
    assert goal['residualOriginalTargetMm3'] == .25 and goal['originalTargetTreatmentFraction'] == .25
    assert ledger['boneRemovedMm3'] == ledger['preparationNonboneRemovedMm3'] == ledger['workNonTargetRemovedMm3'] == .125
    assert ledger['wholeNonTargetRemovedMm3'] == .375 and not outside


@pytest.mark.parametrize('mutation', ['target', 'volume', 'bone', 'state', 'complete', 'calls'])
def test_cold_report_rejects_resealed_invented_counts_or_endpoint(monkeypatch, mutation):
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab import surgical_membrane_transport as membrane
    scene = SimpleNamespace(identity='root')
    base = SimpleNamespace(identity='base')
    source = SimpleNamespace(state_id='root-source', _cranial_stages=SimpleNamespace(content_hash='stage'))
    monkeypatch.setattr(SurgicalEngine, 'restore', lambda *args, **kwargs: source)
    monkeypatch.setattr(planning, 'source_binding', lambda _: {'source': 'exact'})
    monkeypatch.setattr(planning, 'validate_registry', lambda *args: {'opening': scene})
    monkeypatch.setattr(membrane, 'clearance', lambda *args: {'cleared': False})
    state = {'removed': [1], 'holds': {}, 'slots': {}}
    state_id = semantic_digest({'scene': 'root', 'state': state, 'cranialStagesContract': 'stage'})
    action = {'slot': 'left', 'operation': 'inspect'}
    snapshot = {'sceneId': 'root', 'stateId': state_id, 'actions': [action],
        'checkpoint': {'position': 1, 'stateId': state_id, 'state': state}}
    goal = {'originalTargetCells': 4, 'creditedOriginalTargetCells': 1, 'removedOriginalTargetCells': 1,
        'residualOriginalTargetCells': 3, 'originalTargetVolumeMm3': 4., 'residualOriginalTargetMm3': 3.,
        'originalTargetTreatmentFraction': .25, 'zonesReachingRequiredOriginalFraction': 1}
    quantities = {'boneRemovedMm3': 0., 'preparationNonboneRemovedMm3': 0.,
        'workNonTargetRemovedMm3': 0., 'wholeNonTargetRemovedMm3': 0., 'targetRemovedMm3': 1.,
        'modeledElapsedSeconds': 2., 'uniqueRemovedSourceCells': 1, 'unauthorizedRemovedSourceCells': 0}
    monkeypatch.setattr(planning, '_snapshot_quantities', lambda *args: (goal, quantities, False))
    row = {'rootId': 'opening', 'branchId': 'result', 'sceneId': 'root', 'finalStateId': state_id,
        'snapshotHash': semantic_digest(snapshot), 'nativeActions': 1, 'nativeHistory': [
            {'action': action, 'stateBefore': 'root-source', 'stateAfter': state_id, 'status': 'accepted'}],
        'goalAssessment': {**goal, 'completedFrozenTask': False, 'reasons': [
            'ORIGINAL_TARGET_FRACTION_BELOW_GOAL', 'ORIGINAL_SPATIAL_PROGRESS_BELOW_GOAL',
            'NATIVE_WORK_AND_WHOLE_EXIT_REQUIRED']},
        'ledger': {**quantities, 'withinLimits': True, 'spentPublicCalls': 1, 'wallSecondsSinceMeterCreation': 1.},
        'wholeExitComplete': False, 'completedFrozenTask': False, 'admittedPatientTask': False,
        'sequenceOutcome': {'sceneId': 'root', 'stateId': state_id, 'baselineStateId': 'root-source',
            'withdrawalComplete': True, 'dimensions': {'targetRemovedMm3': {'cumulative': 1.},
                'nonTargetRemovedMm3': {'cumulative': 0.}}}}
    report = {'schema': planning.SCHEMA, 'sourceSceneId': 'base', 'sourceStateId': 'common', 'sourceBranchId': 'source',
        'sourceBinding': {'source': 'exact'}, 'rootSeals': {'opening': 'sealed'}, 'admittedPatientTask': False,
        'decision': 'CHECKED_PROGRESS_ONLY',
        'alternatives': [row], 'selectedBranchId': None,
        'cost': {'physicalCalls': 1, 'previewCalls': 0, 'directStepCalls': 1, 'maxPhysicalCalls': 200,
            'roots': {'opening': {'physicalCalls': 1, 'previewCalls': 0, 'directStepCalls': 1,
                'trace': [{'kind': 'step'}]}}}}
    report['comparisonSeal'] = semantic_digest(report)
    record = {'branchId': 'source', 'position': 0, 'branches': {
        'source': {'sceneId': 'base', 'stateId': 'common', 'actions': []}, 'result': snapshot},
        planning.FIELD: {'branchRoots': {'result': 'opening'}, 'roots': [{'id': 'opening',
            'rootSeal': 'sealed', 'sourceSnapshot': {}, 'planning': declaration()}]}}
    assert planning.validate_cranial_report(base, record, report)['verified']
    if mutation == 'target': row['goalAssessment']['originalTargetCells'] = 2
    if mutation == 'volume': row['goalAssessment']['residualOriginalTargetMm3'] = 1.
    if mutation == 'bone': row['ledger']['boneRemovedMm3'] = 1.
    if mutation == 'state': snapshot['checkpoint']['state']['removed'] = [];row['snapshotHash'] = semantic_digest(snapshot)
    if mutation == 'complete': row['completedFrozenTask'] = row['goalAssessment']['completedFrozenTask'] = True
    if mutation == 'calls': report['cost']['roots']['opening']['trace'] = []
    report['comparisonSeal'] = semantic_digest({k: v for k, v in report.items() if k != 'comparisonSeal'})
    with pytest.raises(ValueError): planning.validate_cranial_report(base, record, report)


def test_public_summary_retains_alternatives_and_original_target_dto():
    row = {'branchId': 'checked', 'rootId': 'opening', 'goalAssessment': {'originalTargetVolumeMm3': 40.},
        'ledger': {'boneRemovedMm3': 12.}, 'nativeHistory': [{'private': 'saved'}], 'snapshotHash': 'bound'}
    report = {'schema': planning.SCHEMA, 'comparisonSeal': 'sealed', 'sourceSceneId': 'base',
        'sourceStateId': 'common', 'sourceBranchId': 'source', 'decision': 'CHECKED_PROGRESS_ONLY',
        'selectedBranchId': None, 'failures': [], 'limitations': ['Research only'], 'alternatives': [row],
        'cost': {'physicalCalls': 12, 'previewCalls': 6, 'directStepCalls': 6, 'wallSeconds': 1.}}
    dto = planning.cranial_summary(report)
    assert isinstance(dto['alternatives'], list) and dto['alternatives'][0]['rootId'] == 'opening'
    assert dto['alternatives'][0]['goalAssessment']['originalTargetVolumeMm3'] == 40.
    assert 'nativeHistory' not in dto['alternatives'][0] and dto['sourceBranchId'] == 'source'
    dto['alternatives'][0]['ledger']['boneRemovedMm3'] = 0.
    assert row['ledger']['boneRemovedMm3'] == 12.
