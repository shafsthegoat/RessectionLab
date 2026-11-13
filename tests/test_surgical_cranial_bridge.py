"""Cranial bridge dispatch/persistence controls, not an opening benchmark."""
import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from resectionlab.core import SourceRef, freeze_json, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_cranial_planning_roots import (
    bind_branch_root, create_registry, scene_for_branch, source_binding)
from resectionlab.surgical_engine import SurgicalEngine
from test_surgical_cranial_planning_roots import roots


@pytest.fixture
def adapter(monkeypatch):
    """Stub only the outer planner; all branches/edits/replay remain native.

    The one model-configuration action is deliberately not tumor treatment. Actual
    opening/search correctness belongs to the independent adapter/forest tests.
    """
    module = ModuleType('resectionlab.surgical_cranial_planning')
    calls = []
    class Budget:
        def check(self): calls.append('budget')
    def available(base, record):
        return {'present': True, 'available': True, 'sourceSceneId': base.identity,
            'roots': [{'rootId': row['id'], 'label': row['label']} for row in record['cranialPlanningRoots']['roots']]}
    def compare(base, record, **kwargs):
        kwargs['check']()
        calls.append(('compare', base.identity, kwargs['planning_horizon'], kwargs['beam_width'], kwargs['started_at']))
        kwargs['progress'](.5, 'Generated native configuration transport control')
        scene = scene_for_branch(base, record, 'first')
        engine = SurgicalEngine.restore(scene, record['branches']['first'], use_cache=False)
        configured = engine.step({'slot': 'left', 'operation': 'configure', 'duration': .1,
            'mechanicsRevision': 2, 'clockRevision': 1})
        assert configured['status'] == 'accepted', configured
        row = {'branchId': 'cranial-control', 'rootId': 'opening-0', 'sceneId': scene.identity,
            'finalStateId': engine.state_id, 'nativeActions': len(engine.actions)}
        report = {'schema': 'generated-cranial-bridge-dispatch-control',
            'sourceBranchId': record['branchId'], 'alternatives': [row]}
        return {'report': report, 'branches': [(engine, row)], 'budget': Budget()}
    def install(record, comparison, base):
        for engine, row in comparison['branches']:
            branch = row['branchId']
            record['branches'][branch] = engine.snapshot()
            record.setdefault('timeline', {})[branch] = copy.deepcopy(engine.history)
            record.setdefault('parents', {})[branch] = {'branchId': 'first', 'position': 0}
            bind_branch_root(base, record, branch, row['rootId'])
        record['cranialPlanning'] = copy.deepcopy(comparison['report'])
    module.cranial_planning_availability = available
    module.compare_cranial_strategies = compare
    module.install_cranial_comparison = install
    module.validate_cranial_report = lambda *_: True
    module.cranial_summary = copy.deepcopy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.fixture
def session(roots, adapter, tmp_path):
    case, base, _, saved = roots
    record = copy.deepcopy(saved)
    record['branches']['main'] = SurgicalEngine(base).snapshot()
    record['branchId'], record['position'] = 'main', 0
    session = BridgeSession(tmp_path/'transfers')
    session._install_case(case, {'surgical_toolbox': record}, _Request('install-generated'))
    try: yield session, case.semantic_hash, record
    finally: session.close()


def call(owner, command, previous=None, **kwargs):
    session, key, _ = owner
    return session.execute('surgicalRehearsal', {'caseHash': key, 'command': command,
        **({'expectedStateId': previous['view']['stateId']} if previous else {}), **kwargs},
        _Request('generated-cranial-bridge'), lambda *_: None)


def generate(owner):
    initial = call(owner, 'inspect')
    return call(owner, 'plan', initial, opponent='anatomy-cranial-strategy',
        acknowledgeResearchUnknowns=True, planningHorizon=5, beamWidth=3)


def test_existing_plan_command_uses_real_adapter_and_preserves_source_on_stop(roots, tmp_path):
    # A deliberately nontherapeutic declared inventory exercises the supported
    # real adapter/forest/bridge. Horizon one refuses its zero-target setup.
    case, base, values, original = roots
    rows = copy.deepcopy(values)
    for row in rows:
        row['planning'] = {'limits': {'max_bone_mm3': 285., 'max_preparation_nonbone_mm3': 49.,
            'max_work_nonbone_mm3': 20., 'max_native_actions': 80, 'max_public_calls': 200,
            'max_wall_seconds': 30., 'max_bone_contact_replans': 6,
            'max_tissue_contact_replans': 8, 'export_replay_reserve_calls': 40},
            'openingSkill': {'kind': 'opening', 'actions': [{'slot': 'left', 'operation': 'configure',
                'duration': .1, 'mechanicsRevision': 2, 'clockRevision': 1}]},
            'preparationActions': [{'slot': 'left', 'operation': 'deactivate', 'duration': .1}],
            'workRoutes': [{'id': 'unexecuted-work', 'motion': {'slot': 'left', 'operation': 'apply',
                'pose': {'position': [32., 32., 66.5], 'quaternion': [0., 0., 0., 1.]}, 'duration': .1}}],
            'outsidePoses': {}, 'qualification': {'status': 'reviewed', 'sourceBinding': source_binding(base),
                'evidenceSha256': '1'*64, 'blockers': []}}
    record = copy.deepcopy(original)
    record['cranialPlanningRoots'] = create_registry(base, rows, record['cranialPlanningRoots']['branchRoots'])
    source = SurgicalEngine(base).snapshot()
    record['branches']['main'] = copy.deepcopy(source)
    record['branchId'], record['position'] = 'main', 0
    owner = BridgeSession(tmp_path/'real-adapter')
    try:
        owner._install_case(case, {'surgical_toolbox': record}, _Request('install-real-control'))
        context = (owner, case.semantic_hash, record)
        initial = call(context, 'inspect')
        result = call(context, 'plan', initial, opponent='anatomy-cranial-strategy',
            acknowledgeResearchUnknowns=True, planningHorizon=1, beamWidth=3)
        assert result['branchId'] == 'main' and result['position'] == 0
        assert result['cranialPlanning']['decision'] == 'STOP'
        assert result['cranialPlanning']['alternatives'] == []
        assert result['cranialPlanning']['admittedPatientTask'] is False
        assert result['cranialPlanning']['cost']['physicalCalls'] > 0
        saved = thaw_json(owner.cases[case.semantic_hash].artifacts)['surgical_toolbox']
        assert saved['branches']['main'] == source
    finally: owner.close()


def test_cross_root_selection_manual_edit_and_explicit_branch_keep_bindings(session):
    planned = generate(session)
    selected = call(session, 'selectBranch', planned, branchId='cranial-control')
    assert selected['view']['scene']['sceneId'] != planned['view']['scene']['sceneId']
    original = thaw_json(session[0].cases[session[1]].artifacts)['surgical_toolbox']['branches']['cranial-control']
    edited = call(session, 'step', selected, action={'slot': 'left', 'operation': 'configure',
        'mechanicsRevision': 2, 'clockRevision': 1, 'duration': .1})
    assert edited['branchId'] != 'cranial-control'
    record = thaw_json(session[0].cases[session[1]].artifacts)['surgical_toolbox']
    assert record['branches']['cranial-control'] == original
    assert record['cranialPlanningRoots']['branchRoots'][edited['branchId']] == 'opening-0'
    branched = call(session, 'branch', edited)
    record = thaw_json(session[0].cases[session[1]].artifacts)['surgical_toolbox']
    assert record['cranialPlanningRoots']['branchRoots'][branched['branchId']] == 'opening-0'
    cold = BridgeSession(session[0].transfers.root.parent/'cold')
    try:
        cold._install_case(session[0].cases[session[1]].case, {'surgical_toolbox': record}, _Request('cold'))
        reopened = call((cold, session[1], record), 'inspect')
        assert reopened['view']['stateId'] == branched['view']['stateId']
    finally: cold.close()


def test_cross_root_hot_frames_reuse_admitted_scene_owner(session, monkeypatch):
    import resectionlab.surgical_bridge as bridge
    planned = generate(session)
    body = {'branchId': 'cranial-control', 'position': 0, 'frameIndex': 0}
    first = call(session, 'seekFrame', planned, **body)
    def forbidden(*_args, **_kwargs): pytest.fail('A hot frame rebuilt the source scene')
    monkeypatch.setattr(bridge, 'from_case', forbidden)
    second = call(session, 'seekFrame', planned, **body, timelineToken=first['timelineToken'])
    assert second['view']['stateId'] == first['view']['stateId']
    assert second['playback']['readOnly'] and second['timelineOmitted']


def test_argument_acknowledgement_and_source_cursor_refusals_precede_planning(session, adapter):
    initial = call(session, 'inspect')
    with pytest.raises(ValueError, match='horizon'):
        call(session, 'plan', initial, opponent='anatomy-cranial-strategy', planningHorizon=True)
    with pytest.raises(ValueError, match='Acknowledge'):
        call(session, 'plan', initial, opponent='anatomy-cranial-strategy')
    selected = call(session, 'selectBranch', initial, branchId='first')
    with pytest.raises(ValueError, match='common patient source'):
        call(session, 'plan', selected, opponent='anatomy-cranial-strategy', acknowledgeResearchUnknowns=True)
    assert not any(isinstance(row, tuple) and row[0] == 'compare' for row in adapter)


def test_stage_bound_archive_display_keeps_hash_and_remains_read_only(roots):
    from resectionlab.surgical_archive_view import inspect_archive_checkpoint
    _, base, _, original = roots
    record = copy.deepcopy(original)
    scene = scene_for_branch(base, record, 'first')
    shown = inspect_archive_checkpoint(scene, record, branch_id='first')
    assert shown['view']['stateId'] == record['branches']['first']['stateId']
    assert shown['verification']['readOnly'] and not shown['verification']['causalReplayVerified']
    assert shown['verification']['physicsActionsExecuted'] == 0
    bad = copy.deepcopy(record)
    bad['branches']['first']['checkpoint']['state']['elapsedSeconds'] = 1.
    with pytest.raises(ValueError, match='hash mismatch'):
        inspect_archive_checkpoint(scene, bad, branch_id='first')


def test_generated_flags_cannot_promote_a_reserved_or_patient_descendant_role():
    from resectionlab.surgical_bridge import _cranial_source_role
    case = SimpleNamespace(metadata={'generated': True, 'is_synthetic': True},
        source_refs=(SourceRef('control', 'synthetic://bridge-role', provenance='simulated'),))
    assert _cranial_source_role(SimpleNamespace(), case) == 'GENERATED_CONTROL'
    for declaration in ({'role': 'EVAL', 'split': 'EVAL'},
            {'role': 'TRAIN', 'split': 'TRAIN', 'family_id': 'patient-descendant'},
            {'role': None, 'split': 'TRAIN'}):
        changed = SimpleNamespace(metadata={**case.metadata, 'data_case': declaration},
            source_refs=case.source_refs)
        with pytest.raises(ValueError, match='SOURCE_ROLE'):
            _cranial_source_role(SimpleNamespace(), changed)


@pytest.mark.parametrize('status,cells,expected', [
    ('reviewed', [], 'MISSING_CRITICAL_ANNOTATIONS'),
    ('unreviewed', [1], 'CRITICAL_ANNOTATION_REVIEW_REQUIRED'),
    ('engineering-reviewed', [1], None)])
def test_acquired_critical_floor_cannot_be_acknowledged_away(status, cells, expected):
    from resectionlab.surgical_bridge import _cranial_critical_blockers, _cranial_context_availability
    record = {'cranialPlanningRoots': {'roots': [{'id': 'source-opening', 'sourceSnapshot': {
        'protectionContract': {'required': [], 'annotations': [{'structureId': 'brainstem',
            'reviewStatus': status, 'cells': cells}]}}}]}}
    available = {'present': True, 'available': True, 'blockers': [],
        'roots': [{'rootId': 'source-opening', 'available': True, 'blockers': []}]}
    acquired = SimpleNamespace(metadata={}, source_refs=())
    result = _cranial_context_availability(acquired, record, available)
    errors = _cranial_critical_blockers(record, {'source-opening'})
    assert bool(errors) == bool(expected)
    assert result['available'] == (expected is None)
    assert available['available'] is True  # Projection never mutates saved qualification.
    if expected:
        assert errors[0]['code'] == result['roots'][0]['blockers'][0]['code'] == expected
