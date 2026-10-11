"""Generated opening/model-state controls; no patient data, fitting or actor."""
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.native_cut_union import CutUnionSpec, EvacuatedVolumeSpec
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask, make_native_opening_task
from resectionlab import observed_search as search

MODE = 'return_plus_certified_evacuation_v1'
SPEC = CutUnionSpec(exposure_mode='same_step_parent_closure', closure_schedule='terminal_microstep_only')
ROOT = Path(__file__).resolve().parents[1]
BASE_REV = '6d7a142'
BASE_PINS = {
    'native_spatial_task': 'b8b05798015e4e93ab833598cc4b70b1e3f08a8837568ba701d0e3e272bdf57a',
    'observed_search': 'cb970256f027ac1a278e844e35cfcf1d54b8f81828f8cd165652784b454a417b',
}


def baseline(name):
    # Required committed baseline; never silently substitute newly promoted bytes.
    raw = subprocess.check_output(['git', 'show', f'{BASE_REV}:src/resectionlab/{name}.py'], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == BASE_PINS[name]
    qualified = 'resectionlab._opening_coverage_before_' + name
    module = ModuleType(qualified)
    module.__file__ = f'{BASE_REV}:src/resectionlab/{name}.py'
    module.__package__ = 'resectionlab'
    sys.modules[qualified] = module
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    return module


def task(horizon=2):
    return make_native_opening_task(max_steps=horizon, generated_cut_union=SPEC)


def first_action(t):
    inventory = t._prepare_inventory()
    # A real currently certified material cut; no changed geometry or fabricated ID.
    return max(inventory, key=lambda a: (len(inventory[a].removed_indices_native), a))


def clean(report):
    return {k: v for k, v in report.items() if k != 'planning_seconds'}


def test_default_source_model_inventory_history_and_independent_check_exact():
    old = baseline('native_spatial_task').make_native_opening_task(max_steps=2)
    new = make_native_opening_task(max_steps=2)
    assert old.case.source_hash == new.case.source_hash
    assert old.decision_model_hash == new.decision_model_hash
    assert old.candidate_inventory() == new.candidate_inventory()
    action = first_action(new)
    for a in (action, 'STOP'):
        assert old.step(a).info == new.step(a).info
    assert old.metrics() == new.metrics()
    assert old.independent_geometry_check().to_dict() == new.independent_geometry_check().to_dict()
    assert 'model_estimated_evacuated_progress' not in new.metrics()['history'][0]


@pytest.mark.parametrize('mode', ['return_only', 'return_plus_opening_depth_v1',
                                  'return_plus_opening_depth_volume_v1', 'return_plus_axial_layer_volume_v1'])
def test_old_retention_modes_exact_even_with_same_new_model_available(mode):
    before = baseline('observed_search')
    a, b = task(), task()
    opts = dict(max_calls=200, beam_width=2, seconds=20., transition_mode='lazy_planning',
                retention_mode=mode, retained_prefix_diagnostics=True)
    p, r = before.observed_beam_search(a, **opts)
    q, s = search.observed_beam_search(b, **opts)
    assert p == q and clean(r) == clean(s)
    assert a.metrics() == b.metrics()
    assert 'certified_evacuation_retention' not in s


def test_nominal_progress_is_post_commit_clone_reset_stop_and_replay_safe():
    original = task(3)
    model = original.planning_clone()
    initial = thaw_json(model.public_evacuated_progress())
    initial_state = model._engine.state_hash
    action = first_action(model)
    preview = model._inventory[action]
    assert model._engine.state_hash == initial_state
    assert thaw_json(model.public_evacuated_progress()) == initial
    outcome = model.advance_planning(action)
    progress = outcome.info['model_estimated_evacuated_progress']
    assert progress == thaw_json(model.public_evacuated_progress())
    assert progress['native_state_hash'] == model._engine.state_hash != initial_state
    assert progress['steps'] == 1 and progress['action_id'] == action
    assert progress['cumulative_evacuated_source_volume_mm3'] == preview.evacuated_volume['after']['evacuated_source_volume_mm3']
    assert progress['certified_leaf_count'] > 0
    assert not any('target' in k or 'reference' in k for k in progress)
    clone = model.clone()
    assert thaw_json(clone.public_evacuated_progress()) == progress
    clone.advance_planning('STOP')
    stopped = thaw_json(clone.public_evacuated_progress())
    assert stopped['action_id'] == 'STOP' and stopped['steps'] == 2
    assert stopped['cumulative_evacuated_source_volume_mm3'] == progress['cumulative_evacuated_source_volume_mm3']
    assert thaw_json(model.public_evacuated_progress()) == progress
    replay = original.fresh()
    for a in (action, 'STOP'):
        replay.step(a)
    assert replay.metrics()['history'][-1]['model_estimated_evacuated_progress'] == stopped
    assert replay.independent_geometry_check().feasible
    # A planning conversion of a completed STOP must retain the bound record too.
    assert thaw_json(replay.planning_clone().public_evacuated_progress()) == stopped
    clone.reset()
    assert thaw_json(clone.public_evacuated_progress()) == initial
    assert model._engine.state_hash != clone._engine.state_hash


def test_private_target_permutation_changes_no_public_model_input_or_plan():
    a = task()
    opposite = a.case.observed_support.astype(np.float32) - a.case.nominal_target
    b = NativeSpatialTask(replace(a.case, reference_target=opposite), max_steps=2)
    assert a.case.reference_hash != b.case.reference_hash
    assert a.case.source_hash == b.case.source_hash and a.decision_model_hash == b.decision_model_hash
    x, y = a.planning_clone(), b.planning_clone()
    ox, oy = x.observation(), y.observation()
    assert ox.action_ids == oy.action_ids
    assert np.array_equal(ox.image_channels, oy.image_channels)
    assert np.array_equal(ox.coverage, oy.coverage)
    action = first_action(x)
    assert x.advance_planning(action).info == y.advance_planning(action).info
    opts = dict(max_calls=200, beam_width=2, seconds=20., transition_mode='lazy_planning', retention_mode=MODE)
    p, r = search.observed_beam_search(a, **opts)
    q, s = search.observed_beam_search(b, **opts)
    # Verification time is cost, not a scientific equality field.
    for v in (r, s):
        v.pop('planning_seconds')
        v['certified_evacuation_retention'].pop('state_verification_seconds')
    assert p == q and r == s


@pytest.mark.parametrize('bad', ['missing_volume', 'acquired', 'private_mass', 'crop', 'untyped'])
def test_opt_in_cannot_admit_private_mass_or_hidden_crop_or_acquired_task(bad):
    case = task().case
    changes = {}
    if bad == 'missing_volume': changes['generated_evacuated_volume'] = None
    if bad == 'acquired': changes['track'] = 'annotation_assisted'
    if bad == 'private_mass':
        other = case.observed_support.astype(float) - case.nominal_target
        changes['generated_evacuated_volume'] = EvacuatedVolumeSpec(other, other, 'wrong mass')
    if bad == 'crop': changes['crop_shape'] = (3, 3, 3)
    if bad == 'untyped': changes['generated_cut_union'] = SPEC.to_record()
    with pytest.raises(ValueError, match='Generated evacuation'):
        replace(case, **changes)


@pytest.mark.parametrize('bad', ['volume', 'state', 'source', 'action', 'scope', 'history'])
def test_forged_progress_is_not_a_retention_certificate(bad):
    model = task().planning_clone()
    obs = model.observation()
    action = first_action(model)
    result = model.advance_planning(action)
    info = copy.deepcopy(result.info)
    if bad == 'volume': info['model_estimated_evacuated_progress']['cumulative_evacuated_source_volume_mm3'] += 1.
    if bad == 'state': info['model_estimated_evacuated_progress']['native_state_hash'] = 'stale'
    if bad == 'source': info['model_estimated_evacuated_progress']['source_hash'] = 'other'
    if bad == 'action': info['model_estimated_evacuated_progress']['action_id'] = 'other'
    if bad == 'scope': info['outcome_scope'] = 'separate_evaluator_reference'
    if bad == 'history': model._history[-1]['model_estimated_evacuated_progress']['certified_leaf_count'] += 1
    with pytest.raises((ValueError, RuntimeError)):
        search._public_certified_evacuation(obs, SimpleNamespace(info=info), action, model)


def test_progress_authentication_never_previews_observes_or_uses_outcome_target(monkeypatch):
    model = task().planning_clone()
    obs = model.observation()
    action = first_action(model)
    outcome = model.advance_planning(action)
    expected = outcome.info['model_estimated_evacuated_progress']['cumulative_evacuated_source_volume_mm3']
    def forbidden(*args, **kwargs): raise AssertionError('unexpected query')
    monkeypatch.setattr(model, 'observation', forbidden)
    monkeypatch.setattr(model._engine, 'preview_stroke', forbidden)
    # Irrelevant evaluator-like outcome annotations cannot drive the progress lane.
    outcome.info['target_removed_mm3'] = 1e100
    outcome.info['normal_removed_mm3'] = 1e100
    assert search._public_certified_evacuation(obs, outcome, action, model) == expected


def test_same_model_retention_native_full_replay_and_cap_accounting():
    root = task()
    initial = root.metrics()
    plans = []
    for mode in ('return_only', MODE):
        plan, report = search.observed_beam_search(root, max_calls=200, beam_width=2, seconds=20.,
            transition_mode='lazy_planning', retention_mode=mode)
        assert not report['call_cap_reached'] and not report['time_cap_reached']
        replay = root.fresh()
        for action in plan:
            replay.step(action)
        assert replay.terminated and replay.independent_geometry_check().feasible
        assert replay.metrics()['total_reward'] == report['estimated_incremental_return']
        if mode == MODE:
            evidence = report['certified_evacuation_retention']
            assert evidence['state_verification_calls'] == report['model_transition_calls']
            assert evidence['extra_observations_or_previews'] == 0
            assert not evidence['terminal_objective_changed']
        plans.append(plan)
    assert root.metrics() == initial
    _, capped = search.observed_beam_search(root, max_calls=1, beam_width=2, seconds=20.,
        transition_mode='lazy_planning', retention_mode=MODE)
    assert capped['call_cap_reached'] and capped['model_transition_calls'] == 1
    assert capped['certified_evacuation_retention']['state_verification_calls'] == 1


def test_negative_whole_cell_objective_still_prefers_stop_and_default_refuses_new_lane():
    base = task()
    costly = NativeSpatialTask(base.case, max_steps=2, reward=replace(base.reward_spec, action_cost=100.))
    plan, report = search.observed_beam_search(costly, max_calls=200, beam_width=2, seconds=20.,
        transition_mode='lazy_planning', retention_mode=MODE)
    assert plan == ('STOP',) and report['estimated_incremental_return'] == 0.
    replay = costly.fresh()
    replay.step('STOP')
    assert replay.terminated and replay.independent_geometry_check().feasible
    with pytest.raises(ValueError, match='explicit generated native model'):
        search.observed_beam_search(make_native_opening_task(), max_calls=20, beam_width=2, retention_mode=MODE)


def test_declared_model_or_fraction_storage_tamper_fails_closed():
    t = task()
    object.__setattr__(t.case, 'generated_cut_union', CutUnionSpec())
    with pytest.raises(RuntimeError, match='source content'):
        t.observation()


def test_progress_rejects_foreign_case_even_when_committed_engine_unchanged():
    model = task().planning_clone()
    action = first_action(model)
    model.advance_planning(action)
    state = model._engine.state_hash
    # Different valid generated source; a case swap alone leaves engine/history intact.
    model.case = replace(model.case, target_derivation=model.case.target_derivation + '; foreign declaration')
    assert model._engine.state_hash == state
    with pytest.raises(RuntimeError, match='intact committed model state'):
        model.public_evacuated_progress()
