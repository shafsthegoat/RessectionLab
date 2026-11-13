"""Whole-comparison accounting across immutable opening policies."""
import pytest

from resectionlab.surgical_joint_cranial_forest import OpeningComparisonBudget
from resectionlab.surgical_transition_meter import TransitionBudgetExceeded, TransitionReserveReached


def charge(meter, kind='preview'):
    return meter._charge(kind, 0, 'unacted-source', 0)


def test_discarded_and_later_root_calls_share_one_hard_limit():
    budget = OpeningComparisonBudget(3, 10., export_reserve=0)
    first, second = budget.meter('first'), budget.meter('second')
    charge(first)
    charge(first, 'step')
    row = charge(second)
    assert row['comparisonCallIndex'] == 3
    assert budget.report()['physicalCalls'] == 3
    assert budget.report()['previewCalls'] == 2
    with pytest.raises(TransitionBudgetExceeded):
        charge(second)
    assert second.calls == 1


def test_export_reserve_survives_root_changes_and_is_not_refunded():
    budget = OpeningComparisonBudget(4, 10., export_reserve=2)
    first, second = budget.meter('first'), budget.meter('second')
    charge(first)
    charge(second)
    with pytest.raises(TransitionReserveReached):
        charge(second)
    with budget.export():
        charge(first, 'step')
        charge(second, 'step')
        with pytest.raises(TransitionBudgetExceeded):
            charge(second)
    assert budget.calls == 4
    with pytest.raises(TransitionReserveReached):
        charge(first)


def test_deadline_includes_preparation_and_export_and_blocks_late_result():
    now = [20.]
    budget = OpeningComparisonBudget(8, 4., export_reserve=2,
                                     started_at=18., clock=lambda: now[0])
    first = budget.meter('first')
    charge(first)
    with pytest.raises(TimeoutError):
        with budget.export():
            charge(first, 'step')
            now[0] = 22.
    assert budget.calls == 2
    with pytest.raises(TimeoutError):
        budget.meter('second')


def test_cancel_before_next_call_does_not_execute_or_reset_existing_cost():
    cancelled = [False]
    def check():
        if cancelled[0]:
            raise InterruptedError('user cancelled')
    budget = OpeningComparisonBudget(8, 10., export_reserve=1, check=check)
    first = budget.meter('first')
    charge(first)
    cancelled[0] = True
    with pytest.raises(InterruptedError):
        charge(first, 'step')
    assert first.calls == budget.calls == 1


def test_root_policies_cannot_recreate_a_meter_to_reset_cost():
    budget = OpeningComparisonBudget(8, 10., export_reserve=1)
    first = budget.meter('first')
    charge(first)
    with pytest.raises(ValueError, match='DISTINCT_ROOTS'):
        budget.meter('first')
    budget.meter('second')
    budget.meter('third')
    with pytest.raises(ValueError, match='MAX3'):
        budget.meter('fourth')
    assert budget.calls == 1


def test_failed_export_restores_search_reserve_without_refunding_calls():
    budget = OpeningComparisonBudget(4, 10., export_reserve=2)
    first = budget.meter('first')
    with pytest.raises(RuntimeError):
        with budget.export():
            charge(first, 'step')
            charge(first, 'step')
            raise RuntimeError('failed literal row')
    with pytest.raises(TransitionReserveReached):
        charge(first)
    assert budget.calls == 2


def test_standardized_flap_capacity_prefix_requires_real_finite_work_reason():
    from test_surgical_cranial_planning_roots import roots
    from test_surgical_bone_transport import request, pose
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_cranial_planning_roots import validate_registry, FIELD
    from resectionlab.surgical_joint_cranial_skills import _joint_bone_capacity_partial
    _, base, values, record = roots.__wrapped__()
    source = SurgicalEngine.restore(validate_registry(base, record[FIELD])['opening-0'],
                                    values[0]['sourceSnapshot'], use_cache=False)
    original = source.snapshot()
    action = request('apply', pose=pose(), duration=15.)
    finite = {'status': 'partial', 'reason': 'BONE_CUTTING_IN_PROGRESS',
              'completedFraction': 0., 'elapsedSeconds': .1}
    assert _joint_bone_capacity_partial(source, action, finite)
    assert not _joint_bone_capacity_partial(source, action,
        {**finite, 'reason': 'PROTECTED_STRUCTURE'})
    assert not _joint_bone_capacity_partial(source, action, {**finite, 'elapsedSeconds': 0.})
    assert not _joint_bone_capacity_partial(source, action, {**finite, 'completedFraction': float('nan')})
    assert not _joint_bone_capacity_partial(source, {**action, 'operation': 'move'}, finite)
    assert not _joint_bone_capacity_partial(source, {**action, 'path': [pose()]}, finite)
    assert source.snapshot() == original


@pytest.mark.parametrize('physics', ['standardized', 'declared_legacy'])
def test_native_forest_complete_legacy_sequence_and_standardized_inner_stop(physics):
    # This verifies the Generate integration mechanism on the existing fixture;
    # it is never a patient result or another milling optimization benchmark.
    from test_surgical_cranial_planning_roots import roots
    from test_surgical_membrane_transport import joint_source
    from test_surgical_joint_cranial_search import inventory
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_joint_cranial_forest import search_opening_forest
    from resectionlab.surgical_cranial_planning_roots import validate_registry, FIELD
    _, base, values, record = roots.__wrapped__()
    scenes = validate_registry(base, record[FIELD])
    source = SurgicalEngine.restore(scenes['opening-0'], values[0]['sourceSnapshot'], use_cache=False)
    if physics == 'declared_legacy':
        # Legacy desktop archives retain their declared source physics. Build
        # the original qualified control with full CaseData target evidence;
        # do not remove a profile from the standardized control or a patient.
        import copy
        from resectionlab.surgical_scene import from_case
        from resectionlab.surgical_craniotomy import binding, contract_hash
        from resectionlab.surgical_cranial_planning_roots import candidate_scene
        from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
        case, _, _, _ = roots.__wrapped__()
        template, stage, _, _, _ = joint_source()
        legacy = from_case(case, template.scene.regions)
        assert 'tissuePhysicsV2' not in legacy.properties
        opening = copy.deepcopy(template.scene.properties['cranialAccessV1'])
        opening.update(binding(legacy)); opening['contentHash'] = contract_hash(opening)
        scene = candidate_scene(legacy, 'cranialAccessV1', opening)
        bound = freeze_cranial_stage_contract(SurgicalEngine(scene),
            membrane_cells=stage.membrane_cells, retained_membrane_cells=stage.retained_membrane_cells,
            membrane_anchor_cells=stage.membrane_anchor_cells, work_cells=stage.work_cells,
            evidence=stage.as_dict()['evidence'])
        source = SurgicalEngine(scene, cranial_stages=bound)
    before = source.snapshot()
    _, _, contour, _, _ = joint_source()
    budget = OpeningComparisonBudget(1400, 180., export_reserve=300)
    def propose(current):
        values = inventory(current, current._cranial_stages, contour)
        # The source is the real desktop's standardized finite-work profile.
        # The older instantaneous fixture's default .1s kerf commands are not
        # a constructive program for that different, unchanged physics model.
        for value in values:
            if value['kind'] == 'opening' and physics == 'standardized':
                for action in value['actions']:
                    if action['operation'] == 'apply':
                        action['duration'] = 15.
        return values
    result = search_opening_forest([{
        'id': 'opening-0', 'label': 'Generated complete joint sequence', 'engine': source,
        'limits': {'max_bone_mm3': 285., 'max_preparation_nonbone_mm3': 49.,
                   'max_work_nonbone_mm3': 20., 'max_native_actions': 300,
                   'max_public_calls': 1400, 'max_wall_seconds': 180.},
        'propose': propose}],
        budget=budget, horizon=3, max_expansions=6)
    import json
    print('FOREST_GENERATED_CONTROL', json.dumps({
        'physics': physics, 'decision': result['decision'], 'failures': result['failures'],
        'cost': {k: result['cost'][k] for k in
                 ('physicalCalls', 'previewCalls', 'directStepCalls', 'wallSeconds')},
        'exportedActions': [len(item['engine'].actions) for item in result['branches']],
        'admittedPatientTask': result['admittedPatientTask']}))
    assert source.snapshot() == before
    if physics == 'standardized':
        assert result['selected'] is None and not result['branches']
        assert any(row['message'] == 'CRANIAL_ACCESS_BEYOND_INNER_TABLE_BEFORE_EVACUATION'
                   for row in result['failures'])
        assert result['cost']['directStepCalls'] > 0
        assert result['admittedPatientTask'] is False
        return
    assert result['selected'] is not None, result['failures']
    selected = result['selected']
    assert selected['node'].goal_assessment['completedFrozenTask']
    assert [p['kind'] for p in selected['node'].path] == ['opening', 'preparation', 'cusa']
    assert selected['node'].sequence_outcome['withdrawalComplete']
    assert selected['node'].ledger['wholeNonTargetRemovedMm3'] == 334.
    assert selected['node'].goal_assessment['originalTargetTreatmentFraction'] >= .5
    assert result['admittedPatientTask'] is False
    assert result['cost']['physicalCalls'] == budget.calls
    export_calls = sum(v['directStepCalls'] for name, v in
                       result['cost']['roots']['opening-0']['byPhase'].items()
                       if name == 'cranial.desktop.export-literal')
    assert export_calls == sum(len(item['engine'].actions) for item in result['branches'])
    assert result['cost']['physicalCalls'] > export_calls > 0
    assert all(item['engine'].snapshot() == item['node'].exited.snapshot()
               for item in result['branches'])
