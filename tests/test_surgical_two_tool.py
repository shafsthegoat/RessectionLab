"""Generated two-tool execution/decision controls; no biological validation."""
import copy

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_task_admission import prepare_admission, bind_admission, task_spec as geometry_task
from resectionlab.surgical_planning_observation import project_observation
from resectionlab.surgical_two_tool import (TwoToolObservation, task_spec, defaults, plan_observed_strategies,
    generate_comparison, validate_report, materialize)
from test_surgical_task_admission import prepared


@pytest.fixture(scope='module')
def qualified(prepared):
    case, original, record, native = prepared[0]
    # Distinct generated software case: a replayed wider/deeper exposure gives
    # the paired working parts room. The original narrow baseline is retained.
    engine = original.fresh()
    for action in original.actions:
        action = copy.deepcopy(action)
        if action.get('parameters', {}).get('radiusMm') == 5.: action['parameters']['radiusMm'] = 8.
        result = engine.step(action); assert result['status'] != 'rejected', result
    task = copy.deepcopy(record['taskAdmission']['task']); task['entry']['radiusMm'] = 4.
    contract = prepare_admission(engine, native_collection=native.root, task=task)
    bind_admission(engine, contract)
    record = copy.deepcopy(record); record['taskAdmission'] = contract
    record['branches'][record['branchId']] = engine.snapshot()
    record['position'] = len(engine.actions); record['timeline'][record['branchId']] = copy.deepcopy(engine.history)
    controls = defaults(engine)['defaults']; controls['geometry'].update(candidateCount=1, maxCandidates=8)
    return case, engine, record, controls


@pytest.fixture(scope='module')
def comparison(qualified):
    _, engine, _, controls = qualified
    report = generate_comparison(engine, controls)
    assert len(report['publicPlan']['alternatives']) == 3, report['publicPlan']['rejections']
    return report


def test_actual_two_tool_sequences_retain_progress_and_pending_work(qualified, comparison):
    _, engine, _, _ = qualified; before = engine.snapshot(); rows = comparison['publicPlan']['alternatives']
    short, long, stop = rows
    assert [r['id'] for r in rows] == ['short', 'long', 'stop']
    assert short['summary']['additionalTargetMm3'] < long['summary']['additionalTargetMm3']
    assert short['summary']['cusaPendingMm3'] > 0.
    assert comparison['publicPlan']['decision']['selectedId'] == 'long'
    for row in rows:
        assert row['summary']['additionalNonTargetMm3'] == 0.
        assert row['summary']['fluidRemovedMm3'] == 0.  # No invented blood/fluid benefit.
        native, checkpoints, receipt = materialize(engine, comparison, row['id'])
        assert receipt['nativeMatchesPublicPrediction'] and receipt['withdrawalComplete']
        assert native.state['removed'] == sorted(set(engine.state['removed']) | set(row['summary']['removedSourceCells']))
        assert SurgicalEngine.restore(native.scene, native.snapshot()).snapshot() == native.snapshot()
        if row['id'] != 'stop':
            assert checkpoints['operate'].state['slots']['right']['active']
            assert checkpoints['operate'].state['slots']['right']['instrument'] == 'frazier_suction'
            assert checkpoints['operate'].state['slots']['left']['inserted']
            assert [p['id'] for p in row['phases']] == ['inspect', 'access', 'position-secondary', 'operate', 'reassess', 'withdraw']
            assert any(r['status'] == 'partial' and r['reason'] == 'TISSUE_ASPIRATION_IN_PROGRESS' for r in row['actionReceipts'])
            assert row['reassessment']['probeAction']['pose'] == short['reassessment']['probeAction']['pose']
            assert row['visibility'] is None and row['clinicalRisk'] is None
    assert engine.snapshot() == before and not stop['actionProgram']


def test_public_private_reference_swap_preserves_programs_and_stop(qualified):
    _, e, _, controls = qualified
    value = e._planning_task_admission
    from resectionlab.surgical_task_admission import admission
    task = admission(e)['task']; obs = project_observation(e, task, ['brain', 'target', 'tumor'])
    first = plan_observed_strategies(TwoToolObservation(obs, task_spec()), controls)
    changed = e.clone(); changed._private_evaluation_truth = np.ones(e.scene.shape, bool)
    changed._private_evaluator = lambda: pytest.fail('Private evaluator was used')
    changed.history = [{'private': 'history metadata'}]
    changed.state['observations'] = [{'private': 'historical mapping output'}]
    changed.actions = [{}]*len(e.actions)
    second_obs = project_observation(changed, task, ['brain', 'target', 'tumor'])
    assert first == plan_observed_strategies(TwoToolObservation(second_obs, task_spec()), controls)
    required = copy.deepcopy(controls); required['geometry']['requiredEvidence'].append('functional')
    a = plan_observed_strategies(TwoToolObservation(obs, task_spec()), required)
    b = plan_observed_strategies(TwoToolObservation(second_obs, task_spec()), required)
    assert a == b and a['decision']['operation'] == 'STOP' and len(a['alternatives']) == 1
    assert a['decision']['reason'] == 'REQUIRED_EVIDENCE_UNAVAILABLE'
    assert e._planning_task_admission is value


def test_zero_progress_requirement_prefers_stop_and_large_requirement_refuses(qualified):
    _, e, _, controls = qualified
    for minimum in (0., 8.):
        changed = copy.deepcopy(controls); changed['minimumTargetMm3'] = minimum
        report = generate_comparison(e, changed)
        assert report['publicPlan']['decision']['operation'] == 'STOP'
        assert report['publicPlan']['decision']['selectedId'] == 'stop'
        assert validate_report(e, report)


@pytest.mark.parametrize('kind', ['program', 'state', 'progress', 'probe', 'choice', 'geometry', 'capability'])
def test_self_resealed_report_cannot_change_execution_or_decision(qualified, comparison, kind):
    _, engine, _, _ = qualified; changed = copy.deepcopy(comparison); public = changed['publicPlan']; long = public['alternatives'][1]
    if kind == 'program': long['actionProgram'][-1]['duration'] = .3
    elif kind == 'state': long['physicalStateHashes'][-1] = 'private-canary'
    elif kind == 'progress': long['summary']['additionalTargetMm3'] = 999.
    elif kind == 'probe': long['reassessment']['probeStatus'] = 'accepted' if long['reassessment']['probeStatus'] == 'rejected' else 'rejected'
    elif kind == 'choice': public['decision']['selectedId'] = 'short'
    elif kind == 'geometry': public['chosenPair']['secondary']['tip']['position'][0] += 1.
    else: public['task']['maxAdditionalNonTargetMm3'] = 100.
    public['publicPlanSeal'] = semantic_digest({k: v for k, v in public.items() if k != 'publicPlanSeal'})
    changed['reportSeal'] = semantic_digest({k: v for k, v in changed.items() if k != 'reportSeal'})
    with pytest.raises(ValueError, match='TWO_TOOL_TASK'):
        validate_report(engine, changed)


def test_task_packet_cannot_expand_capabilities_and_model_cannot_reset(qualified):
    from resectionlab.surgical_task_admission import admission
    _, e, _, _ = qualified; obs = project_observation(e, admission(e)['task'], ['brain', 'target'])
    altered = task_spec(); altered['allowedTools'].append('brain_retractor')
    with pytest.raises(ValueError, match='capability'): TwoToolObservation(obs, altered)
    model = TwoToolObservation(obs, task_spec()).planning_engine()
    with pytest.raises(ValueError, match='cannot be reset'): model.fresh()
    assert model.step({'slot': 'left', 'operation': 'configure', 'duration': .1})['status'] == 'rejected'


def test_controls_and_native_budget_are_enforced(qualified):
    from resectionlab.surgical_task_admission import admission
    _, e, _, controls = qualified
    for key, value in [('shortSeconds', True), ('longSeconds', 3.), ('minimumTargetMm3', -1.)]:
        invalid = copy.deepcopy(controls); invalid[key] = value
        with pytest.raises(ValueError, match='TWO_TOOL_TASK'): generate_comparison(e, invalid)
    obs = project_observation(e, admission(e)['task'], ['brain', 'target'])
    from resectionlab.core import thaw_json
    from resectionlab.surgical_planning_observation import PlanningObservation
    fields = thaw_json(obs.fields); fields['remainingActions'] = 2
    limited = TwoToolObservation(PlanningObservation(obs.labels, obs.affine, fields), task_spec())
    report = plan_observed_strategies(limited, controls)
    assert report['decision']['operation'] == 'STOP' and report['alternatives'][0]['id'] == 'stop'


def test_native_workspace_bridge_preview_install_save_and_reopen(qualified, tmp_path):
    from resectionlab.imaging import save_case, read_case_artifacts
    from resectionlab.desktop_bridge import BridgeSession, _Request
    case, e, record, controls = qualified
    path = tmp_path/'two-tool.ressectionlab'; save_case(case, path, artifacts={'surgical_toolbox': record})
    session = BridgeSession(tmp_path/'transfers')
    def call(command, state=None, **values):
        return session.execute('surgicalRehearsal', {'caseHash': case.semantic_hash, 'command': command,
            **({'expectedStateId': state} if state else {}), **values}, _Request('two-tool'), lambda *_: None)
    try:
        session.execute('loadCase', {'path': str(path)}, _Request('load'), lambda *_: None)
        initial = call('inspect'); start = initial['view']['stateId']
        assert initial['strategyPlanningAvailability']['available']
        generated = call('generateStrategies', start, constraints=controls, acknowledgeResearchUnknowns=True)
        report = generated['strategyPlanning']; assert report['publicPlan']['decision']['selectedId'] == 'long'
        preview = call('previewStrategy', start, candidateId='long', expectedPlanSeal=report['reportSeal'], phase='operate')
        assert preview['changed'] is False and preview['view']['stateId'] == start
        assert preview['approachPreview']['view']['stateId'] != start
        assert preview['approachPreview']['kind'] == 'strategy'
        assert preview['approachPreview']['receipt']['withdrawalComplete']
        committed = call('applyStrategy', start, candidateId='long', expectedPlanSeal=report['reportSeal'])
        assert committed['view']['stateId'] != start and committed['strategyPlanning']['current'] is False
        branch = committed['branchId']; endpoint = committed['view']['stateId']
        assert all(not slot['active'] and not slot['inserted'] for slot in committed['view']['slots'].values())
        phase = next(p for p in committed['strategyPlanning']['applications'][0]['receipt']['phases'] if p['id'] == 'reassess')
        intermediate = call('selectBranch', endpoint, branchId=branch, position=phase['position'])
        edited = call('step', intermediate['view']['stateId'], action={'slot': 'right', 'operation': 'inspect', 'duration': .1})
        assert edited['branchId'] != branch  # The checked complete branch stays immutable.
        restored = call('selectBranch', edited['view']['stateId'], branchId=report['sourceBranchId'], position=report['sourcePosition'])
        assert restored['view']['stateId'] == start
        refusal_controls = copy.deepcopy(controls); refusal_controls['geometry']['requiredEvidence'].append('functional')
        stopped = call('generateStrategies', start, constraints=refusal_controls, acknowledgeResearchUnknowns=True)
        assert stopped['strategyPlanning']['publicPlan']['decision']['operation'] == 'STOP'
        saved = tmp_path/'saved.ressectionlab'; session.execute('saveCase', {'caseHash': case.semantic_hash, 'path': str(saved)}, _Request('save'), lambda *_: None)
        assert branch in read_case_artifacts(saved)['surgical_toolbox']['branches']
    finally: session.close()
    fresh = BridgeSession(tmp_path/'fresh-transfers')
    try:
        fresh.execute('loadCase', {'path': str(saved)}, _Request('load'), lambda *_: None)
        reopened = fresh.execute('surgicalRehearsal', {'caseHash': case.semantic_hash, 'command': 'inspect'}, _Request('inspect'), lambda *_: None)
        assert reopened['view']['stateId'] == start and reopened['strategyPlanning']['publicPlan']['decision']['operation'] == 'STOP'
    finally: fresh.close()


@pytest.mark.parametrize('field', ['decision', 'execution', 'route-extra', 'metric-type'])
def test_nested_public_evidence_is_not_authorized_by_resealing(qualified, comparison, field):
    _, e, _, _ = qualified; report = copy.deepcopy(comparison); public = report['publicPlan']; approaches = public['approaches']
    if field == 'decision': approaches['decision']['reason'] = 'private-canary'
    elif field == 'execution':
        approaches['routes'][0]['publicExecution']['approachPhysicalStateHash'] = 'private-canary'
        public['chosenPair']['route'] = copy.deepcopy(approaches['routes'][0])
    elif field == 'route-extra':
        approaches['routes'][0]['privateMetric'] = 'private-canary'
        public['chosenPair']['route'] = copy.deepcopy(approaches['routes'][0])
    else: public['alternatives'][1]['summary']['additionalNonTargetMm3'] = False
    approaches['publicPlanSeal'] = semantic_digest({k: v for k, v in approaches.items() if k != 'publicPlanSeal'})
    public['publicPlanSeal'] = semantic_digest({k: v for k, v in public.items() if k != 'publicPlanSeal'})
    report['reportSeal'] = semantic_digest({k: v for k, v in report.items() if k != 'reportSeal'})
    with pytest.raises(ValueError): validate_report(e, report)


def test_rich_evaluator_attributes_never_enter_native_generation(qualified, comparison):
    _, e, _, controls = qualified
    changed = e.clone()
    class PrivateOwner:
        def __deepcopy__(self, memo): pytest.fail('Cloned private evaluator owner')
    changed._private_truth = np.ones(e.scene.shape, bool)
    changed._private_evaluator = PrivateOwner()
    assert generate_comparison(changed, controls) == comparison


def test_preproposal_computation_budget_is_not_reported_as_geometric_infeasibility(qualified, monkeypatch):
    import itertools
    from resectionlab import surgical_two_tool as planning
    _, e, _, controls = qualified
    ticks = itertools.chain([0.], itertools.repeat(100.))
    monkeypatch.setattr(planning.time, 'monotonic', lambda: next(ticks))
    report = generate_comparison(e, controls)
    assert report['publicPlan']['decision']['operation'] == 'STOP'
    assert report['publicPlan']['decision']['reason'] == 'COMPUTATION_BUDGET'


def test_completed_validation_is_reused_but_native_execution_is_not(qualified, comparison, monkeypatch):
    from resectionlab import surgical_two_tool as planning
    _, engine, _, _ = qualified
    planning._VALIDATED_REPORTS.clear()
    compile_calls = []; native_actions = []
    compile_recipe, step = planning._compile, SurgicalEngine.step
    def compile_counted(*args, **kwargs):
        compile_calls.append(1)
        return compile_recipe(*args, **kwargs)
    def step_counted(self, *args, **kwargs):
        if type(self) is SurgicalEngine: native_actions.append(1)
        return step(self, *args, **kwargs)
    monkeypatch.setattr(planning, '_compile', compile_counted)
    monkeypatch.setattr(SurgicalEngine, 'step', step_counted)
    before = engine.snapshot()
    assert validate_report(engine, comparison)
    assert len(compile_calls) == 2
    assert validate_report(engine.clone(), copy.deepcopy(comparison))
    cold, phases_a, receipt_a = materialize(engine, comparison, 'long')
    first_actions = len(native_actions)
    warm, phases_b, receipt_b = materialize(engine, copy.deepcopy(comparison), 'long')
    assert len(compile_calls) == 2  # No repeated public recipes on complete-content hits.
    assert first_actions > 0 and len(native_actions) == 2*first_actions
    assert cold.snapshot() == warm.snapshot() and receipt_a == receipt_b
    assert {k: v.snapshot() for k, v in phases_a.items()} == {k: v.snapshot() for k, v in phases_b.items()}
    assert engine.snapshot() == before
    info = planning.validation_cache_stats()
    assert info['entries'] == 1 and not info['reportsRetained'] and not info['enginesRetained']


@pytest.mark.parametrize('field', ['state', 'actions', 'history', 'catalog', 'source-bytes'])
def test_validation_hit_rechecks_native_and_source_dependencies(qualified, comparison, monkeypatch, field):
    from resectionlab import surgical_planning_observation as public
    from resectionlab import surgical_two_tool as planning
    _, engine, _, _ = qualified
    planning._VALIDATED_REPORTS.clear(); assert validate_report(engine, comparison)
    hits = planning.validation_cache_stats()['hits']; changed = engine.clone()
    path = original = None
    try:
        if field == 'state': changed.state['elapsedSeconds'] += .1
        elif field == 'actions': changed.actions[0]['duration'] += .01
        elif field == 'history': changed.history[0]['private-canary'] = True
        elif field == 'catalog':
            altered = public.catalog(); altered[0]['private-canary'] = True
            monkeypatch.setattr(public, 'catalog', lambda: copy.deepcopy(altered))
        else:
            from resectionlab.patient_anatomy import PatientImagingCollection
            from resectionlab.surgical_task_admission import admission
            collection = PatientImagingCollection(admission(engine)['verification']['nativeCollectionPath'])
            path = collection.path(collection.series('T1')['original']['path']); original = path.read_bytes()
            path.write_bytes(original+b'changed after validation')
        with pytest.raises(ValueError): validate_report(changed, comparison)
        assert planning.validation_cache_stats()['hits'] == hits
    finally:
        if path is not None: path.write_bytes(original)


@pytest.mark.parametrize('field', ['mask', 'config'])
def test_validation_hit_projects_actual_guards_even_with_unchanged_serialized_contract(qualified, field):
    from dataclasses import replace
    from resectionlab import surgical_two_tool as planning
    from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet, GuardConfig
    from resectionlab.surgical_task_admission import admission
    _, original, _, controls = qualified
    def protection(cell):
        mask = np.zeros(original.scene.shape, bool); mask[cell] = True
        return ProtectionSet(original.scene, [ProtectionAnnotation.from_mask(original.scene, 'brainstem', mask,
            provenance='Generated invalidation control')], clearance_mm=0.)
    engine = SurgicalEngine(original.scene, protection=protection((0, 0, 0)),
        guard_config=GuardConfig(require_critical_coverage=False))
    engine.state = copy.deepcopy(original.state); engine.actions = copy.deepcopy(original.actions)
    engine.history = copy.deepcopy(original.history)
    contract = prepare_admission(engine, native_collection=admission(original)['verification']['nativeCollectionPath'],
        task=admission(original)['task']); bind_admission(engine, contract)
    required = copy.deepcopy(controls); required['geometry']['requiredEvidence'].append('functional')
    report = generate_comparison(engine, required)
    planning._VALIDATED_REPORTS.clear(); assert validate_report(engine, report)
    changed = engine.clone()
    if field == 'mask': changed.protection = protection((18, 18, 20))
    else: changed.guard_config = replace(changed.guard_config, critical_max_displacement_mm=.123)
    assert planning._binding(engine) == planning._binding(changed)  # Serialized contract is deliberately stale.
    hits = planning.validation_cache_stats()['hits']
    with pytest.raises(ValueError, match='public observation differs'): validate_report(changed, report)
    assert planning.validation_cache_stats()['hits'] == hits


def test_validation_cache_is_bounded_and_resealed_failures_are_never_admitted(qualified, monkeypatch):
    from resectionlab import surgical_two_tool as planning
    _, engine, _, controls = qualified
    planning._VALIDATED_REPORTS.clear(); monkeypatch.setattr(planning, '_VALIDATION_CACHE_ENTRIES', 3)
    required = copy.deepcopy(controls); required['geometry']['requiredEvidence'].append('functional')
    reports = []
    for seconds in (.2, .3, .4, .5, .6):
        required['shortSeconds'] = seconds
        report = generate_comparison(engine, required); assert validate_report(engine, report); reports.append(report)
        assert planning.validation_cache_stats()['entries'] <= 3
    info = planning.validation_cache_stats()
    assert info['entries'] == 3 and info['estimatedResidentBytes'] < 4096
    misses = info['misses']; assert validate_report(engine, reports[0])
    assert planning.validation_cache_stats()['misses'] == misses+1  # Oldest content really was evicted.
    forged = copy.deepcopy(reports[0]); public = forged['publicPlan']
    public['alternatives'][0]['summary']['additionalNonTargetMm3'] = False
    public['publicPlanSeal'] = semantic_digest({k: v for k, v in public.items() if k != 'publicPlanSeal'})
    forged['reportSeal'] = semantic_digest({k: v for k, v in forged.items() if k != 'reportSeal'})
    for _ in range(2):
        with pytest.raises(ValueError, match='canonical programs'): validate_report(engine, forged)
    after = planning.validation_cache_stats()
    assert after['entries'] == 3 and after['hits'] == info['hits']


def test_requested_entry_patch_restricts_both_instruments(qualified):
    _, e, _, controls = qualified
    restricted = copy.deepcopy(controls); restricted['geometry']['entry']['radiusMm'] = .5
    report = generate_comparison(e, restricted)
    assert report['publicPlan']['decision']['operation'] == 'STOP'
    assert report['publicPlan']['chosenPair'] is None
    assert report['publicPlan']['rejections']
    assert validate_report(e, report)
