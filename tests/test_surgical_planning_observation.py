"""Generated noninterference controls, not acquired or biological validation."""
import copy

import numpy as np
import pytest

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.surgical_engine import MAX_STEPS, SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_task_admission import task_spec
from resectionlab.surgical_planning_observation import project_observation, physical_state_hash
from resectionlab.surgical_interactive_planning import plan_observed_approaches, evaluate_sealed_contact_references


@pytest.fixture(scope='module')
def baseline():
    labels = np.zeros((36, 36, 36), np.uint16)
    labels[7:29, 7:29, 10:34] = 1; labels[14:23, 14:23, 20:32] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'private original brain name', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'private original tumor name', 'material': 'tumor', 'target': True}],
        {'kind': 'generated', 'patientId': 'private-patient-A', 'path': '/private/A'}, {'tissuePhysicsV2': default_profile()})
    e = SurgicalEngine(scene)
    actions = [
        {'slot': 'left', 'operation': 'place', 'pose': {'position': [-20., -20., -20.], 'quaternion': [0., 0., 0., 1.]}},
        {'slot': 'right', 'operation': 'place', 'pose': {'position': [18., 18., 0.], 'quaternion': [0., 0., 0., 1.]}, 'parameters': {'radiusMm': 5.}},
        {'slot': 'right', 'operation': 'apply', 'pose': {'position': [18., 18., 20.], 'quaternion': [0., 0., 0., 1.]}},
        {'slot': 'right', 'operation': 'withdraw'},
        {'slot': 'right', 'operation': 'inspect', 'parameters': {'radiusMm': 1.5, 'power': 1., 'duty': 1.}},
        {'slot': 'right', 'operation': 'place', 'pose': {'position': [45., 45., -10.], 'quaternion': [0., 0., 0., 1.]}},
        {'slot': 'right', 'operation': 'configure', 'mechanicsRevision': 2, 'clockRevision': 1}]
    for action in actions:
        result = e.step({'duration': .1, **action}); assert result['status'] == 'accepted', result
    task = task_spec(entry={'centerMm': [18., 18., 12.], 'radiusMm': 2.}, axis=[0., 0., 1.],
                     target={'centerMm': [18., 18., 28.], 'radiusMm': 8.}, allow_unreviewed=True)
    constraints = {key: copy.deepcopy(task[key]) for key in ('entry', 'target', 'requiredEvidence')}
    constraints.update(tool={'instrument': 'cusa', 'radiusMm': 1.5}, candidateCount=1, maxCandidates=8,
                       maxWallSeconds=30., minEntrySeparationMm=2., minAngleDeg=0., nonTargetClearanceMm=0.)
    return e, task, constraints


def observe(e, task): return project_observation(e, task, ['target', 'brain', 'tumor'])


@pytest.fixture(scope='module')
def planned(baseline):
    e, task, constraints = baseline
    observation = observe(e, task)
    plan = plan_observed_approaches(observation, constraints)
    assert plan['routes'] and plan['decision']['operation'] == 'COMPARE', plan['rejections']
    return observation, plan


class NoClone:
    def __deepcopy__(self, memo): raise AssertionError('Privileged owner was cloned')


def private_variant(e):
    source = {'kind': 'generated', 'patientId': 'private-patient-B', 'path': '/private/B', 'evaluationHash': 'private-hash'}
    props = copy.deepcopy(e.scene.properties)
    props['tissuePhysicsV2']['provenance']['description'] = 'Private provenance changed'
    props['tissuePhysicsV2']['provenance']['extra_private_metadata'] = 'not an input'
    props['tissuePhysicsV2']['brain']['model'] = 'private display string'
    regions = copy.deepcopy(e.scene.regions)
    for region in regions: region.update(name='private alternate name', color='#123456')
    changed = SurgicalEngine(SurgicalScene(e.scene.labels, e.scene.affine, regions, source, props))
    changed.state = copy.deepcopy(e.state)
    changed.state['observations'] = [{'private-reference-result': 'changed'}]
    changed.actions = [{'private-old-action': i} for i in range(len(e.actions))]
    changed.history = [{'private-history': 'changed'}]
    changed._observer = NoClone(); changed._evaluation_truth = np.ones(e.scene.shape, bool)
    changed._privileged_context = NoClone()
    return changed


def test_private_identity_history_callbacks_and_reference_truth_do_not_cross_boundary(baseline, planned):
    e, task, constraints = baseline; observation, plan = planned
    changed = private_variant(e)
    assert changed.scene.identity != e.scene.identity and changed.state_id != e.state_id
    other = observe(changed, task)
    assert other.manifest() == observation.manifest()
    assert plan_observed_approaches(other, constraints) == plan
    model = other.planning_engine()
    assert not hasattr(model, '_observer') and not hasattr(model, '_evaluation_truth')
    assert not hasattr(model, '_planning_task_admission') and model.actions == model.history == []
    assert 'private' not in str(thaw_json(other.fields))
    assert 'private' not in str(model.scene.source)


def test_private_swap_preserves_required_evidence_stop(baseline):
    e, task, constraints = baseline
    constraints = copy.deepcopy(constraints); constraints['requiredEvidence'].append('functional')
    first = plan_observed_approaches(observe(e, task), constraints)
    second = plan_observed_approaches(observe(private_variant(e), task), constraints)
    assert first == second and first['decision']['operation'] == 'STOP'
    assert first['decision']['reason'] == 'required_evidence_unavailable' and not first['routes']


@pytest.mark.parametrize('change', ['label', 'affine', 'removed', 'pose', 'parameter', 'physics', 'fluid', 'thermal', 'capacity', 'budget', 'task', 'coverage'])
def test_actual_public_inputs_change_input_hash(baseline, planned, change):
    e, task, _ = baseline; changed = e.clone(); task = copy.deepcopy(task)
    coverage = ['target', 'brain', 'tumor']
    if change in ('label', 'affine', 'physics'):
        labels, affine, props = e.scene.labels.copy(), e.scene.affine.copy(), copy.deepcopy(e.scene.properties)
        if change == 'label': labels[0, 0, 0] = 1
        elif change == 'affine': affine[0, 3] += .1
        else: props['tissuePhysicsV2']['brain']['brainStiffnessNPerMm'] *= 2.
        changed.scene = SurgicalScene(labels, affine, e.scene.regions, e.scene.source, props)
    elif change == 'removed': changed.state['removed'] = changed.state['removed'][1:]
    elif change == 'pose': changed.state['slots']['right']['position'][0] += 1.
    elif change == 'parameter': changed.state['slots']['right']['parameters']['flowMm3PerSecond'] = 1.
    elif change == 'fluid': changed.state['fluid']['10'] = .2
    elif change == 'thermal': changed.state['drillingV2']['temperatureC']['10'] = 40.
    elif change == 'capacity': changed.state['cusaV3']['progressMm3']['10'] = .1
    elif change == 'budget': changed.actions.append({'operation': 'private prefix contents irrelevant'})
    elif change == 'task': task['axisRasUnit'] = [0., 1., 0.]
    else: coverage.append('functional')
    observation = project_observation(changed, task, coverage)
    assert observation.manifest()['inputHash'] != planned[0].manifest()['inputHash']


def test_projection_preserves_supported_passive_burdens_without_copying_native_ancestry(baseline):
    e, task, _ = baseline; changed = e.clone()
    changed.state['fluid']['10'] = .3; changed.state['thermal']['10'] = .2
    changed.state['debris']['11'] = .4; changed.state['contact']['12'] = .7
    changed.state['drillingV2']['temperatureC']['13'] = 40.
    changed.state['cusaV3']['progressMm3']['14'] = .5
    changed.state['cusaV3']['progressExact']['14'] = [1, 2]
    changed.state['cusaV3']['contactSeconds']['14'] = .25
    changed.state['cusaV3']['inputCapacityMm3'] = .5
    changed.state['deformationModel']['remainderSeconds'] = .001
    public = observe(changed, task).planning_engine()
    assert physical_state_hash(public) == physical_state_hash(changed)
    assert public.scene is not changed.scene and public._cell_cover_cache is not changed._cell_cover_cache
    assert public.actions == [] and public.history == []
    for operation in (public.fresh, public.reset, lambda: public.replay([])):
        with pytest.raises(ValueError, match='cannot reset'): operation()


@pytest.mark.parametrize('field,value', [
    ('unknownPhysicalModel', {}), ('patientPosition', {}), ('sources', [{'id': 'bleed'}]),
    ('cutEdges', [[1, 2]]), ('holds', {'left': {}}), ('deformation', {'1': [0., 0., 0.]}),
    ('detachedFragments', []), ('flapComponents', [{'private': 'summary'}])])
def test_unsupported_physical_state_is_refused_instead_of_dropped(baseline, field, value):
    e, task, _ = baseline; changed = e.clone(); changed.state[field] = value
    with pytest.raises(ValueError, match='PLANNING_OBSERVATION'):
        observe(changed, task)


def test_pending_deformation_or_vascular_state_is_refused(baseline):
    e, task, _ = baseline
    for name, key, value in [('deformationModel', 'velocities', {'1': [1., 0., 0.]}), ('vesselsV2', 'wounds', {'1': {}})]:
        changed = e.clone(); changed.state[name][key] = value
        with pytest.raises(ValueError, match='outside this task track'): observe(changed, task)


def test_remaining_action_budget_covers_access_and_withdrawal_and_forces_stop(baseline):
    e, task, constraints = baseline; changed = e.clone()
    changed.actions = [{}]* (MAX_STEPS-3)  # Access needs 3 and withdrawal needs 1.
    observation = observe(changed, task)
    assert observation.manifest()['remainingActions'] == 3
    plan = plan_observed_approaches(observation, constraints)
    assert not plan['routes'] and plan['decision']['operation'] == 'STOP'
    assert any(row['reason'] == 'PUBLIC_EXECUTION_REFUSED' and 'budget' in row['detail'] for row in plan['rejections'])


def test_public_protection_changes_feasibility_and_metadata_does_not(baseline, planned):
    from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet, GuardConfig
    e, task, constraints = baseline; mask = np.zeros(e.scene.shape, bool)
    mask[18, 18, 20] = True
    def variant(text):
        p = ProtectionSet(e.scene, [ProtectionAnnotation.from_mask(e.scene, 'brainstem', mask, provenance=text,
            display_name=text)], clearance_mm=0.)
        changed = SurgicalEngine(e.scene, protection=p, guard_config=GuardConfig(require_critical_coverage=False))
        changed.state = copy.deepcopy(e.state); changed.actions = copy.deepcopy(e.actions)
        return changed
    first, second = observe(variant('private-A'), task), observe(variant('private-B'), task)
    assert first.manifest() == second.manifest() and first.manifest() != planned[0].manifest()
    a = plan_observed_approaches(first, constraints); b = plan_observed_approaches(second, constraints)
    assert a == b and a['decision']['operation'] == 'STOP'
    assert any('PROTECTED_STRUCTURE' in r['reason'] for r in a['rejections'])


def test_references_are_loaded_after_sealing_and_can_only_change_evaluation(planned):
    observation, plan = planned; before = copy.deepcopy(plan); events = []
    def loader(value):
        def load():
            assert plan['publicPlanSeal'] == semantic_digest({k: v for k, v in plan.items() if k != 'publicPlanSeal'})
            events.append('reference-loaded-after-seal')
            return {'private-endpoint': np.full(observation.labels.shape, value, bool)}
        return load
    a = evaluate_sealed_contact_references(plan, loader(False))
    b = evaluate_sealed_contact_references(plan, loader(True))
    assert plan == before and a['publicPlanSeal'] == b['publicPlanSeal']
    assert a['referenceContactCells'] != b['referenceContactCells']
    assert len(events) == 2
    altered = copy.deepcopy(plan); altered['decision']['operation'] = 'STOP'
    with pytest.raises(ValueError, match='sealed before evaluation'):
        evaluate_sealed_contact_references(altered, lambda: pytest.fail('Unsealed plan loaded private truth'))


def test_observation_arrays_and_metadata_are_immutable(planned):
    observation, _ = planned
    with pytest.raises(ValueError): observation.labels[0, 0, 0] = 1
    with pytest.raises(TypeError): observation.fields['remainingActions'] = 0
    with pytest.raises(ValueError, match='typed public'):
        plan_observed_approaches(object(), {})


@pytest.mark.parametrize('location', ['root', 'state', 'task', 'region', 'guard', 'profile'])
def test_direct_typed_construction_cannot_bypass_the_allowlist(planned, location):
    from resectionlab.surgical_planning_observation import PlanningObservation
    observation, _ = planned; fields = thaw_json(observation.fields)
    if location == 'root': fields['evaluationOnlyReferenceHash'] = 'private'
    elif location == 'state': fields['state']['evaluationOnlyReferenceHash'] = 'private'
    elif location == 'task': fields['task']['evaluationOnlyReferenceHash'] = 'private'
    elif location == 'region': fields['regions'][0]['name'] = 'private narrative'
    elif location == 'guard': fields['guard'] = {'evaluationOnlyReferenceHash': 'private'}
    else: fields['properties']['tissuePhysicsV2']['provenance']['description'] = 'private narrative'
    with pytest.raises(ValueError, match='PLANNING_OBSERVATION'):
        PlanningObservation(observation.labels, observation.affine, fields)


def test_evaluator_loader_cannot_relabel_or_mutate_the_frozen_plan(planned):
    observation, plan = planned; changed = copy.deepcopy(plan)
    def loader():
        changed['publicPlanSeal'] = 'private loader mutated seal'
        changed['routes'].clear()
        return {'reference': np.ones(observation.labels.shape, bool)}
    evaluation = evaluate_sealed_contact_references(changed, loader)
    assert evaluation['publicPlanSeal'] == plan['publicPlanSeal']
    assert set(evaluation['referenceContactCells']['reference']) == set(plan['decision']['candidateIds'])


def test_native_suffix_matches_projected_prediction_with_passive_burdens(baseline):
    e, task, constraints = baseline; changed = e.clone()
    changed.state['fluid']['1000'] = .3; changed.state['fluidAddedMm3'] = .3
    changed.state['debris']['1001'] = .4; changed.state['debrisMm3'] = .4
    changed.state['drillingV2']['temperatureC']['1002'] = 40.
    changed.state['cusaV3']['progressMm3']['1003'] = .5
    changed.state['cusaV3']['progressExact']['1003'] = [1, 2]
    changed.state['cusaV3']['contactSeconds']['1003'] = .25
    changed.state['cusaV3']['inputCapacityMm3'] = .5
    plan = plan_observed_approaches(observe(changed, task), constraints)
    assert plan['routes'], plan['rejections']
    route = plan['routes'][0]
    for actions, key in [(route['actionProgram'], 'approachPhysicalStateHash'),
                         (route['withdrawalProgram'], 'withdrawalPhysicalStateHash')]:
        for action in actions: assert changed.step(action)['status'] == 'accepted'
        assert physical_state_hash(changed) == route['publicExecution'][key]
    assert changed.state['cusaV3']['progressMm3']['1003'] == .5
    assert changed.state['drillingV2']['temperatureC']['1002'] < 40.
    assert changed.state['fluidAddedMm3'] == .3 and changed.state['debrisMm3'] > 0.


def test_preproposal_wall_budget_returns_a_sealed_stop(baseline, monkeypatch):
    from resectionlab import surgical_interactive_planning as planning
    e, task, constraints = baseline
    ticks = iter([0., 100.]); monkeypatch.setattr(planning.time, 'monotonic', lambda: next(ticks))
    plan = plan_observed_approaches(observe(e, task), constraints)
    assert plan['decision']['operation'] == 'STOP' and plan['status'] == 'no_geometric_route'
    assert plan['rejections'] == [{'reason': 'GEOMETRY_WALL_BUDGET'}]
    assert plan['publicPlanSeal'] == semantic_digest({key: value for key, value in plan.items() if key != 'publicPlanSeal'})


def test_direct_typed_guard_requires_valid_configuration(planned):
    from resectionlab.surgical_planning_observation import PlanningObservation
    observation, _ = planned; fields = thaw_json(observation.fields)
    fields['guard'] = {'clearanceMm': 0., 'required': [], 'annotations': [],
                       'config': {'require_critical_coverage': 'private canary', 'critical_max_displacement_mm': 0.}}
    with pytest.raises(ValueError, match='boolean'):
        PlanningObservation(observation.labels, observation.affine, fields)


def test_constructor_validates_the_same_metadata_snapshot_it_retains(planned):
    from collections.abc import Mapping
    from resectionlab.surgical_planning_observation import PlanningObservation
    observation, _ = planned; fields = thaw_json(observation.fields)
    class ChangesOnSecondRead(Mapping):
        reads = 0
        def __iter__(self): return iter(fields)
        def __len__(self): return len(fields)
        def __getitem__(self, key): return fields[key]
        def items(self):
            self.reads += 1
            return (fields if self.reads == 1 else {**fields, 'privateSecondSnapshot': 'canary'}).items()
    source = ChangesOnSecondRead()
    copied = PlanningObservation(observation.labels, observation.affine, source)
    assert copied.manifest() == observation.manifest() and source.reads == 1
