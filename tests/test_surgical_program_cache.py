"""Bounded exact public-prefix reuse; generated software cases only."""
import copy
import gc
import weakref

import numpy as np
import pytest
from scipy.spatial import cKDTree

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.surgical_program_cache import ProgramPrefixCache
from resectionlab.surgical_planning_observation import PlanningObservation, project_observation
from resectionlab.surgical_anatomical_encounter import AnatomicalObservation, project_context
from resectionlab.surgical_task_admission import admission
from resectionlab import surgical_two_tool as planning
from test_surgical_anatomical_encounter import context_case, context_report
from test_surgical_two_tool import qualified, comparison, prepared


def checked():
    pass


class Cancelled(RuntimeError):
    pass


def cancel_on(call_number):
    calls = []
    def check():
        calls.append(None)
        if len(calls) == call_number:
            raise Cancelled('cancelled at a cache boundary')
    return check


def prefix(number=1):
    return [{'slot': 'left', 'operation': 'inspect', 'duration': number}]


def payload():
    return {'state': {'removed': [1, 3], 'slots': {'left': {'parameters': {'power': 1.}}}},
            'actions': prefix(), 'history': [{'events': [{'sourceCells': [1]}]}],
            'receipt': {'status': 'accepted', 'elapsedSeconds': 1.},
            'contextSeen': {'shaft': [4], 'working': [5]}, 'physicalHash': 'physical'}


def test_put_and_get_detach_entire_checkpoint_graph_and_action_key():
    cache = ProgramPrefixCache()
    actions, value = prefix(), payload()
    original = copy.deepcopy(value)
    cache.put(actions, value, check=checked)
    actions[0]['duration'] = 99
    value['state']['removed'].append(99)
    value['history'][0]['events'][0]['sourceCells'].append(99)
    value['contextSeen']['shaft'].append(99)
    found = cache.get(prefix(), check=checked)
    assert found == original
    found['state']['slots']['left']['parameters']['power'] = 99.
    found['actions'].clear(); found['receipt']['status'] = 'rejected'
    found['contextSeen']['working'].clear()
    assert cache.get(prefix(), check=checked) == original
    assert cache.get(actions, check=checked) is None


def test_full_prefix_order_duration_parameters_and_json_types_are_distinct():
    cache = ProgramPrefixCache()
    first = {'slot': 'left', 'operation': 'apply', 'duration': .25,
             'parameters': {'power': 1}, 'pose': {'position': [1, 2, 3]}}
    last = {'slot': 'right', 'operation': 'inspect', 'duration': .01}
    key = [first, last]
    cache.put(key, {'qualified': True}, check=checked)
    # The last action alone is not an adequate key.
    variants = [copy.deepcopy(key) for _ in range(5)]
    variants[0][0]['duration'] = 1.
    variants[1][0]['parameters']['power'] = True
    variants[2][0]['parameters']['power'] = 1.
    variants[3][0]['pose']['position'][0] = 2
    variants[4].reverse()
    assert all(cache.get(k, check=checked) is None for k in variants)
    reordered = [{k: v for k, v in reversed(list(first.items()))}, last]
    assert cache.get(reordered, check=checked) == {'qualified': True}


def test_entry_lru_and_byte_eviction_do_not_discard_more_recent_authority():
    cache = ProgramPrefixCache(max_entries=2)
    for n in (1, 2): cache.put(prefix(n), {'n': n}, check=checked)
    assert cache.get(prefix(1), check=checked) == {'n': 1}
    cache.put(prefix(3), {'n': 3}, check=checked)
    assert cache.get(prefix(2), check=checked) is None
    assert cache.get(prefix(1), check=checked) == {'n': 1}
    assert cache.stats()['entries'] == 2 and cache.stats()['evictions'] == 1
    sizing = ProgramPrefixCache(); data = {'blob': 'x' * 2000}
    sizing.put(prefix(), data, check=checked)
    one = sizing.stats()['accountedPayloadBytes']
    bounded = ProgramPrefixCache(max_entries=20, max_bytes=2 * one - 1)
    bounded.put(prefix(1), data, check=checked)
    bounded.put(prefix(2), data, check=checked)
    assert bounded.get(prefix(1), check=checked) is None
    assert bounded.get(prefix(2), check=checked) == data
    assert bounded.stats()['entries'] == 1 and bounded.stats()['evictions'] == 1
    bounded.put(prefix(3), {'oversized': 'x' * (4 * one)}, check=checked)
    assert bounded.stats()['bypasses'] == 1
    assert bounded.get(prefix(2), check=checked) == data
    assert bounded.stats()['peakAccountedPayloadBytes'] <= bounded.max_bytes


def test_unknown_private_owner_bypasses_without_deepcopy_or_retention():
    class PrivateOwner:
        def __deepcopy__(self, memo):
            pytest.fail('Private owner deepcopy was invoked')
    owner = PrivateOwner(); reference = weakref.ref(owner)
    cache = ProgramPrefixCache()
    value = {'nested': [{'privateOwner': owner}]}
    cache.put(prefix(), value, check=checked)
    assert cache.stats()['bypasses'] == 1 and cache.stats()['entries'] == 0
    assert cache.get(prefix(), check=checked) is None
    del value, owner
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize('operation,boundary', [('put', 1), ('put', 2), ('get', 1), ('get', 2)])
def test_cancellation_cannot_publish_or_return_a_checkpoint(operation, boundary):
    cache = ProgramPrefixCache()
    if operation == 'get': cache.put(prefix(), payload(), check=checked)
    with pytest.raises(Cancelled):
        if operation == 'put': cache.put(prefix(), payload(), check=cancel_on(boundary))
        else: cache.get(prefix(), check=cancel_on(boundary))
    assert cache.stats()['entries'] == (1 if operation == 'get' else 0)
    if operation == 'get': assert cache.get(prefix(), check=checked) == payload()


def test_close_releases_all_payloads_and_refuses_further_use():
    cache = ProgramPrefixCache()
    cache.put(prefix(), payload(), check=checked)
    assert cache.stats()['accountedPayloadBytes'] > 0
    cache.close(); cache.close()
    assert cache.stats()['closed'] and cache.stats()['entries'] == 0
    assert cache.stats()['accountedPayloadBytes'] == 0
    with pytest.raises(ValueError, match='closed'): cache.get(prefix(), check=checked)
    with pytest.raises(ValueError, match='closed'): cache.put(prefix(), payload(), check=checked)


@pytest.mark.parametrize('limits', [{'max_bytes': 0}, {'max_entries': 0}])
def test_zero_capacity_is_an_exact_bypass(limits):
    cache = ProgramPrefixCache(**limits)
    cache.put(prefix(), payload(), check=checked)
    assert cache.stats()['entries'] == 0 and cache.stats()['bypasses'] == 1


@pytest.fixture(scope='module', params=['v1', 'context-v2'])
def recipe(request, qualified, comparison, context_case, context_report):
    if request.param == 'v1':
        engine, report = qualified[1], comparison
        context = None
    else:
        engine, report = context_case[1], context_report
        context, _ = project_context(engine, 'context')
    public = report['publicPlan']
    observation = planning.TwoToolObservation(project_observation(engine, admission(engine)['task'],
        report['sourceEvidence']['availableEvidence']), planning.task_spec(with_context=context is not None), context)
    controls = planning._normalize(observation, public['constraints'])
    chosen = public['chosenPair']
    return observation, controls, chosen['route'], chosen['secondary'], engine


def compile_variants(recipe, scope=None):
    observation, controls, route, secondary, _ = recipe
    short = planning._compile(observation, controls, route, secondary, controls['shortSeconds'], 'short', prefix_scope=scope)
    long = planning._compile(observation, controls, route, secondary, controls['longSeconds'], 'long',
        probe_pose=short['reassessment']['probeAction']['pose'], prefix_scope=scope)
    return short, long


def test_uncached_and_reused_rows_remain_exact_and_contact_durations_do_not_mix(recipe):
    observation, controls, _, _, native = recipe
    before = native.snapshot()
    expected = compile_variants(recipe)
    with planning._CompilationScope(observation, controls, kind='cache-test-equivalence') as scope:
        cached = compile_variants(recipe, scope)
        first_hits = scope.cache.stats()['hits']
        assert first_hits > 0
        assert semantic_digest(cached) == semantic_digest(expected)
        short, long = cached
        short_apply = next(a for a in short['actionProgram'] if a['operation'] == 'apply')
        long_apply = next(a for a in long['actionProgram'] if a['operation'] == 'apply')
        assert short_apply['duration'] == controls['shortSeconds'] < long_apply['duration'] == controls['longSeconds']
        assert short['summary']['cusaActivationSeconds'] < long['summary']['cusaActivationSeconds']
        short['summary']['additionalTargetMm3'] = -999
        short['actionProgram'][0]['duration'] = 99.
        if observation.context is not None:
            short['anatomicalEncounter']['nearSourceCells'].append(-1)
        assert semantic_digest(compile_variants(recipe, scope)) == semantic_digest(expected)
        assert scope.cache.stats()['hits'] > first_hits
    assert scope.root is None and scope.cache.stats()['closed'] and scope.cache.stats()['entries'] == 0
    assert native.snapshot() == before


@pytest.fixture(scope='module')
def guarded_recipe(context_case, context_report):
    engine = context_case[1]
    context, _ = project_context(engine, 'context')
    baseline = project_observation(engine, admission(engine)['task'], context_report['sourceEvidence']['availableEvidence'])
    fields = thaw_json(baseline.fields)
    fields['guard'] = {'clearanceMm': 0., 'required': ['brainstem'],
        'config': {'require_critical_coverage': False, 'critical_max_displacement_mm': 0.},
        'annotations': [{'id': 'brainstem', 'cells': [0], 'reviewStatus': 'unreviewed'}]}
    baseline = PlanningObservation(baseline.labels, baseline.affine, fields)
    observation = planning.TwoToolObservation(baseline, planning.task_spec(with_context=True), context)
    controls = planning._normalize(observation, context_report['publicPlan']['constraints'])
    return observation, controls


@pytest.mark.parametrize('changed', ['state', 'actions', 'history', 'protected-ids', 'task', 'remaining',
    'grid-owner', 'grid-affine', 'grid-inverse', 'cover-owner', 'guard-cells', 'guard-points',
    'guard-tree-owner', 'guard-tree-indices', 'guard-tree-bounds', 'guard-config', 'catalog', 'scene-owner',
    'occupancy-cache', 'by-label'])
def test_mutated_public_owner_cannot_authorize_restoration(guarded_recipe, monkeypatch, changed):
    from dataclasses import replace
    from resectionlab.surgical_scene import SurgicalScene
    observation, controls = guarded_recipe
    with planning._CompilationScope(observation, controls, kind='cache-test-owner') as scope:
        scope.model(observation, controls)
        root = scope.root
        if changed == 'state': root.state['elapsedSeconds'] += .1
        elif changed == 'actions': root.actions.append({'private-canary': True})
        elif changed == 'history': root.history.append({'private-canary': True})
        elif changed == 'protected-ids': root._protected_ids = frozenset({1})
        elif changed == 'task': root._task['maxActions'] += 1
        elif changed == 'remaining': root._remaining_actions -= 1
        elif changed == 'occupancy-cache': root._occupancy_cache = ('injected', object())
        elif changed == 'by-label': root.by_label = {}
        elif changed == 'grid-owner': root.grid = copy.copy(root.grid)
        elif changed == 'grid-affine':
            matrix = root.grid.affine.copy(); matrix[0, 3] += 1.
            object.__setattr__(root.grid, 'affine', matrix)
        elif changed == 'grid-inverse':
            matrix = root.grid._inverse.copy(); matrix[0, 3] += 1.
            object.__setattr__(root.grid, '_inverse', matrix)
        elif changed == 'cover-owner': root._cell_cover_cache = copy.copy(root._cell_cover_cache)
        elif changed == 'guard-cells': root.protection.cells = np.array([1], np.int64)
        elif changed == 'guard-points': root.protection._points = root.protection._points + 1.
        elif changed == 'guard-tree-owner': root.protection._tree = cKDTree(root.protection._points + 10.)
        elif changed == 'guard-tree-indices':
            with pytest.raises(ValueError, match='read-only'): root.protection._tree.indices[0] = 9
            root.protection._tree.indices.shape = (1, 1)
        elif changed == 'guard-tree-bounds':
            with pytest.raises(ValueError, match='read-only'): root.protection._tree.mins[0] += 1.
            root.protection._tree.mins.shape = (1, 3)
        elif changed == 'guard-config': root.guard_config = replace(root.guard_config, critical_max_displacement_mm=.1)
        elif changed == 'catalog':
            real = planning.catalog
            altered = real(); altered[0]['shaftRadiusMm'] += .1
            monkeypatch.setattr(planning, 'catalog', lambda: copy.deepcopy(altered))
        else:
            root.scene = SurgicalScene(root.scene.labels, root.scene.affine, root.scene.regions,
                                       {'kind': 'replacement-public-scene'}, root.scene.properties)
        with pytest.raises(ValueError, match='owner changed|source'):
            scope.require_owner()


@pytest.mark.parametrize('changed_binding', ['controls', 'context-mask'])
def test_changed_public_inputs_cannot_share_a_compilation_scope(guarded_recipe, changed_binding):
    observation, controls = guarded_recipe
    with planning._CompilationScope(observation, controls, kind='cache-test-binding') as scope:
        scope.model(observation, controls)
        changed = copy.deepcopy(controls)
        other = observation
        if changed_binding == 'controls':
            changed['encounter']['maxNearMm3'] = 0.
        else:
            mask = observation.context.mask.copy(); mask.flat[0] = not mask.flat[0]
            context = AnatomicalObservation(mask, observation.context.affine, observation.context.review_status)
            other = planning.TwoToolObservation(observation.baseline, observation.task, context)
        with pytest.raises(ValueError, match='observation or controls changed'):
            scope.model(other, changed)


@pytest.mark.parametrize('check_name', ['_check_burdens', '_check_action_clearance'])
def test_only_successful_fully_checked_prefixes_enter_cache(recipe, monkeypatch, check_name):
    observation, controls, route, secondary, _ = recipe
    real = getattr(planning, check_name)
    def refuse_second(engine, *args):
        if len(engine.actions) == 2:
            raise ValueError('injected checked-prefix refusal after native step')
        return real(engine, *args)
    monkeypatch.setattr(planning, check_name, refuse_second)
    with planning._CompilationScope(observation, controls, kind='cache-test-failure') as scope:
        with pytest.raises(ValueError, match='injected checked-prefix refusal'):
            planning._compile(observation, controls, route, secondary, controls['shortSeconds'], 'short', prefix_scope=scope)
        setup = [route['actionProgram'][0], {'slot': 'left', 'operation': 'inspect', 'duration': .01}]
        assert setup[0]['operation'] == 'exchange'
        assert scope.cache.get(setup[:1], check=checked) is not None
        assert scope.cache.get(setup, check=checked) is None
        assert scope.cache.stats()['stores'] == 1


def test_context_receipt_failure_cannot_publish_an_unchecked_prefix(guarded_recipe, context_report, monkeypatch):
    observation, controls = guarded_recipe
    pair = context_report['publicPlan']['chosenPair']
    real = planning.EncounterTracker.observe
    def refuse_second(tracker, engine, action, previous_slots):
        if len(engine.actions) == 2:
            raise ValueError('injected context receipt failure')
        return real(tracker, engine, action, previous_slots)
    monkeypatch.setattr(planning.EncounterTracker, 'observe', refuse_second)
    with planning._CompilationScope(observation, controls, kind='cache-test-context-failure') as scope:
        with pytest.raises(ValueError, match='injected context receipt failure'):
            planning._compile(observation, controls, pair['route'], pair['secondary'],
                              controls['shortSeconds'], 'short', prefix_scope=scope)
        assert scope.cache.stats()['stores'] == scope.cache.stats()['entries'] == 1


def test_exception_exit_releases_root_and_all_prefixes(guarded_recipe):
    observation, controls = guarded_recipe
    scope = planning._CompilationScope(observation, controls, kind='cache-test-exit')
    with pytest.raises(Cancelled):
        with scope:
            scope.model(observation, controls)
            root = weakref.ref(scope.root)
            scope.cache.put(prefix(), payload(), check=checked)
            raise Cancelled('stop this local computation')
    gc.collect()
    assert root() is None and scope.root is None and scope.owner is None
    assert scope.cache.stats()['closed'] and scope.cache.stats()['entries'] == 0
    assert scope.cache.stats()['accountedPayloadBytes'] == 0
    assert planning.prefix_cache_stats()['cache-test-exit']['completed'] is False
