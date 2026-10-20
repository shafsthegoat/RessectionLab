"""Shared-engine director contracts on explicit small engineering anatomy."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab.core import array_digest
from resectionlab import full_case_rehearsal as director
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def _pose(z):
    return {'position': [8., 8., z], 'quaternion': [0., 0., 0., 1.]}


def _engine(*, healthy=False, target=True):
    labels = np.zeros((20, 20, 20), np.uint16)
    regions = []
    if target:
        labels[7:10, 7:10, 8] = 2
        regions.append({'id': 2, 'name': 'target', 'material': 'tumor', 'target': True})
    if healthy:
        labels[7:10, 7:10, 9] = 1
        regions.append({'id': 1, 'name': 'brain', 'material': 'soft_tissue'})
    return SurgicalEngine(SurgicalScene(labels, np.eye(4), regions,
                                       {'kind': 'explicit-engineering-fixture'}, {}))


def _prepared(*, healthy=False):
    engine = _engine(healthy=healthy)
    for action in [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .2},
        {'slot': 'left', 'operation': 'place', 'pose': _pose(4.),
         'parameters': {'radiusMm': 1.5}, 'duration': .2},
    ]:
        assert engine.step(action)['status'] == 'accepted'
    return engine


def _removal_director(monkeypatch, engine):
    # Candidate scheduling is fixed here; all previews, geometry, effects and
    # commits still use the actual shared engine. This is not a patient route.
    access = {'id': 'fixture', 'outsidePose': _pose(4.), 'entryPose': _pose(7.),
              'drillRadiusMm': 3.}
    monkeypatch.setattr(director, 'access_candidates', lambda *_a, **_k: [access])
    stroke = {'slot': 'left', 'operation': 'apply', 'pose': _pose(7.), 'duration': 1.}
    monkeypatch.setattr(director, 'next_removal_candidates', lambda *_a, **_k: [stroke])
    meta = director.new_director(engine)
    meta.update(stage='remove', currentPhaseId='removal-fixture', cycleStartTargetMm3=0.,
                phases=[{'id': 'removal-fixture', 'label': 'Explicit fixture removal',
                         'startPosition': len(engine.actions), 'endPosition': len(engine.actions)}])
    return meta, stroke


def _without_wall(meta):
    return {key: value for key, value in meta.items() if key != 'wallSeconds'}


def test_slices_are_deterministic_source_preserving_and_finish_cleanup(monkeypatch):
    engine = _prepared()
    meta, _ = _removal_director(monkeypatch, engine)
    original, original_meta = engine.snapshot(), copy.deepcopy(meta)
    label_hash = array_digest(engine.scene.labels)
    whole, whole_meta = director.run_slice(engine, meta, max_actions=8)
    sliced, sliced_meta = engine, meta
    for _ in range(8):
        sliced, sliced_meta = director.run_slice(sliced, sliced_meta, max_actions=1)
        if sliced_meta['status'] in ('complete', 'stopped'):
            break
    assert whole_meta['status'] == sliced_meta['status'] == 'complete'
    assert whole_meta['stage'] == sliced_meta['stage'] == 'endpoint'
    assert whole.snapshot() == sliced.snapshot()
    assert _without_wall(whole_meta) == _without_wall(sliced_meta)
    assert not any(slot['inserted'] or slot['active'] or slot['grip']
                   for slot in whole.state['slots'].values())
    assert not whole.state['holds']
    assert engine.snapshot() == original and meta == original_meta
    assert array_digest(engine.scene.labels) == label_hash
    assert director.accounting(whole) == {
        'originalTargetMm3': 9., 'healthyRemovedMm3': 0., 'unknownRemovedMm3': 0.}


def test_director_commits_exact_public_preview_transitions(monkeypatch):
    engine = _prepared()
    meta, _ = _removal_director(monkeypatch, engine)
    result, _ = director.run_slice(engine, meta, max_actions=4)
    replay = engine.clone()
    for action in result.actions[len(engine.actions):]:
        before = replay.state_id
        preview = replay.preview(action)
        assert replay.state_id == before
        committed = replay.step(action)
        for field in ('status', 'reason', 'stateId', 'events', 'deltas', 'metrics'):
            assert committed.get(field) == preview.get(field)
    assert replay.snapshot() == result.snapshot()


def test_legal_cusa_stroke_removing_healthy_tissue_is_refused_before_commit(monkeypatch):
    engine = _prepared(healthy=True)
    meta, stroke = _removal_director(monkeypatch, engine)
    preview = engine.preview(stroke)
    assert preview['status'] == 'accepted'
    assert preview['metrics']['targetRemovedMm3'] == 9.
    assert preview['metrics']['nonTargetRemovedMm3'] > 0.
    before = engine.snapshot()
    candidate, result = director.run_slice(engine, meta, max_actions=1)
    assert engine.snapshot() == before
    assert candidate.state['removed'] == []
    assert any(row['reason'] == 'HEALTHY_OR_UNKNOWN_REMOVAL_REFUSED'
               for row in result['rejections'])


def test_cleanup_endpoint_reports_real_owned_residual_and_preserves_source(monkeypatch):
    engine = _prepared()
    assert engine.step({'slot': 'left', 'operation': 'move', 'pose': _pose(5.), 'duration': .2})['status'] == 'accepted'
    meta, _ = _removal_director(monkeypatch, engine)
    meta['stage'] = 'cleanup'
    before = engine.snapshot()
    candidate, result = director.run_slice(engine, meta, max_actions=8)
    assert result['status'] == 'stopped' and result['stage'] == 'endpoint'
    assert director.accounting(candidate)['originalTargetMm3'] == 9.
    assert candidate.metrics()['targetRemovedMm3'] == 0.
    assert not candidate.state['slots']['left']['inserted']
    assert engine.snapshot() == before


def test_director_rejects_a_stale_action_boundary(monkeypatch):
    engine = _prepared()
    meta, _ = _removal_director(monkeypatch, engine)
    assert engine.step({'slot': 'right', 'operation': 'inspect', 'duration': .2})['status'] == 'accepted'
    before = engine.snapshot()
    with pytest.raises(ValueError, match='state|boundary|checkpoint'):
        director.run_slice(engine, meta, max_actions=1)
    assert engine.snapshot() == before


def test_director_requires_an_owned_tumor_target():
    with pytest.raises(ValueError, match='target|tumor'):
        director.new_director(_engine(target=False))


def test_cancelled_slice_never_changes_the_input_engine_or_director(monkeypatch):
    engine = _prepared()
    meta, _ = _removal_director(monkeypatch, engine)
    before, before_meta = engine.snapshot(), copy.deepcopy(meta)
    checks = 0
    def check():
        nonlocal checks
        checks += 1
        if checks == 8:
            raise InterruptedError('explicit fixture cancellation')
    with pytest.raises(InterruptedError, match='fixture cancellation'):
        director.run_slice(engine, meta, max_actions=4, check=check)
    assert engine.snapshot() == before and meta == before_meta


def test_matching_strict_protection_replaces_only_an_intact_relaxed_engine(monkeypatch):
    from resectionlab import surgical_rl_protection as protection
    scene = _engine().scene
    annotation = protection.ProtectionAnnotation(
        'brainstem', (0,), protection.grid_binding(scene), 'explicit fixture', 'unreviewed', 'Fixture brainstem')
    supplied = protection.ProtectionSet(scene, [annotation], strict=True)
    monkeypatch.setattr(Path, 'exists', lambda _path: True)
    monkeypatch.setattr(protection, 'load_protection', lambda *_a, **_k: supplied)
    relaxed = SurgicalEngine(scene, guard_config=protection.GuardConfig(require_critical_coverage=False))
    bound = director.bind_case_protection(relaxed)
    assert bound.guard_config.require_critical_coverage
    assert bound._protected_ids == frozenset({0})
    assert relaxed._protected_ids == frozenset()
    assert relaxed.step({'slot': 'right', 'operation': 'inspect', 'duration': .2})['status'] == 'accepted'
    with pytest.raises(ValueError, match='intact|initial'):
        director.bind_case_protection(relaxed)


def _runner():
    script = Path(__file__).resolve().parents[1] / 'scripts/run_full_case_rehearsal.py'
    spec = importlib.util.spec_from_file_location('full_case_runner_fixture', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saved_workspace(path, *, case_id, branch_history=False, v2=False):
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.surgical_scene import from_case
    from resectionlab.workspace_bundle import save_workspace
    mask = np.zeros((12, 12, 12), bool)
    mask[4:8, 4:8, 4:8] = True
    metadata = {'anatomy_layers': {'target': {'kind': 'tumor'}}}
    if v2:
        from resectionlab.surgical_physics_v2 import default_profile
        metadata['surgical_properties'] = {'tissuePhysicsV2': default_profile()}
    case = CaseData(case_id=case_id, mri=np.zeros(mask.shape, np.float32),
                    compartments={'target': mask}, affine=np.eye(4),
                    source_refs=(SourceRef('fixture', 'synthetic://full-case-tests', provenance='simulated'),),
                    metadata=metadata)
    engine = SurgicalEngine(from_case(case))
    meta = director.new_director(engine, 'vascular' if branch_history else 'primary')
    record = {'mapping': engine.scene.regions, 'branches': {'main': engine.snapshot()},
              'branchId': 'main', 'position': 0, 'parents': {'main': None},
              'fullCase': {'main': meta}}
    if branch_history:
        archive = engine.clone()
        assert archive.step({'slot': 'right', 'operation': 'inspect', 'duration': .1})['status'] == 'accepted'
        record['branches']['full-primary'] = archive.snapshot()
        record['parents']['full-primary'] = {'branchId': 'main', 'position': 0}
    save_workspace(case, path, sources=[], imaging=None, episode=None, selection=None,
                   artifacts={'surgical_toolbox': record})
    return record


def test_cli_resume_rejects_an_unrelated_declared_original_source(tmp_path, monkeypatch):
    original, resume = tmp_path / 'original.ressectionlab', tmp_path / 'resume.ressectionlab'
    _saved_workspace(original, case_id='original-engineering-source')
    _saved_workspace(resume, case_id='different-engineering-source')
    before = original.read_bytes(), resume.read_bytes()
    runner = _runner()
    monkeypatch.setattr(sys, 'argv', [str(runner.__file__), '--case', str(original), '--resume', str(resume),
                                    '--output', str(tmp_path / 'out'), '--max-slices', '0'])
    with pytest.raises(ValueError, match='source|scene|original'):
        runner.main()
    assert (original.read_bytes(), resume.read_bytes()) == before


def test_cli_scenario_restart_preserves_existing_named_branches(tmp_path, monkeypatch):
    from resectionlab.workspace_bundle import load_workspace
    original = tmp_path / 'original.ressectionlab'
    record = _saved_workspace(original, case_id='branch-engineering-source', branch_history=True)
    before = original.read_bytes()
    runner = _runner()
    # This fixture exercises archive naming only. Patient binding has its own
    # source-bound guard test above; no protection or anatomy is invented here.
    monkeypatch.setattr(runner, 'bind_case_protection', lambda engine: engine)
    monkeypatch.setattr(sys, 'argv', [str(runner.__file__), '--case', str(original),
                                    '--output', str(tmp_path / 'out'), '--max-slices', '0'])
    runner.main()
    _, artifacts, _ = load_workspace(tmp_path / 'out/primary.ressectionlab')
    saved = artifacts['surgical_toolbox']
    assert saved['branches']['full-primary'] == record['branches']['full-primary']
    assert saved['branchId'] == 'full-primary-2'
    assert saved['parents']['full-primary-2'] == {'branchId': 'main', 'position': 0}
    assert original.read_bytes() == before


def test_cli_refuses_a_destination_that_would_overwrite_the_declared_source(tmp_path, monkeypatch):
    source = tmp_path / 'primary.ressectionlab'
    _saved_workspace(source, case_id='protected-original-engineering-source')
    before = source.read_bytes()
    runner = _runner()
    monkeypatch.setattr(sys, 'argv', [str(runner.__file__), '--case', str(source),
                                    '--output', str(tmp_path), '--max-slices', '1', '--slice-actions', '1'])
    with pytest.raises(ValueError, match='source|destination|original'):
        runner.main()
    assert source.read_bytes() == before


def test_cli_preserves_scheduling_progress_across_slices_with_only_refused_candidates(tmp_path, monkeypatch):
    from resectionlab.workspace_bundle import load_workspace, save_workspace
    source = tmp_path / 'original.ressectionlab'
    _saved_workspace(source, case_id='candidate-refusal-engineering-source', v2=True)
    case, artifacts, workspace = load_workspace(source)
    meta = artifacts['surgical_toolbox']['fullCase']['main']
    meta['stage'] = 'access_setup'
    # Deliberately invalid placement inside declared target. The real public
    # engine must refuse each proposal while the scheduler advances its cursor.
    meta['accesses'] = [{'id': f'blocked-{index}', 'outsidePose': _pose(6.),
                         'entryPose': _pose(6.), 'drillRadiusMm': 3.}
                        for index in range(20)]
    save_workspace(case, source, sources=workspace['sources'], imaging=workspace['imagingState'],
                   episode=None, selection=None, artifacts=artifacts)
    before = source.read_bytes()
    runner = _runner()
    monkeypatch.setattr(sys, 'argv', [str(runner.__file__), '--case', str(source),
                                    '--output', str(tmp_path / 'out'), '--max-slices', '2', '--slice-actions', '1'])
    runner.main()
    _, saved_artifacts, _ = load_workspace(tmp_path / 'out/primary.ressectionlab')
    saved = saved_artifacts['surgical_toolbox']
    assert saved['position'] == 0
    assert saved['fullCase']['main']['status'] == 'running'
    assert saved['fullCase']['main']['accessIndex'] == 16
    assert len(saved['fullCase']['main']['rejections']) == 16
    assert source.read_bytes() == before


def _opened_assistant_field():
    from resectionlab.surgical_physics_v2 import default_profile
    from resectionlab.full_case_assist import assistant_field_candidate
    labels = np.zeros((64, 64, 40), np.uint16)
    labels[41:44, 19:22, 18] = 3
    labels[41:44, 19:22, 15] = 2
    labels[19:22, 19:22, 15] = 2
    labels[40:45, 18:23, 14] = 1
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 3, 'name': 'bone', 'material': 'bone'},
    ], {'kind': 'explicit-assistant-engineering-fixture'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    def pose(x, z):
        return {'position': [float(x), 20., float(z)], 'quaternion': [1., 0., 0., 0.]}
    actions = [
        {'operation': 'place', 'pose': pose(42, 47), 'parameters': {'radiusMm': 1.5, 'forceN': 20., 'rpm': 80000., 'duty': 1.}},
        {'operation': 'apply', 'duration': 3., 'pose': pose(42, 17)},
        {'operation': 'withdraw'}, {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': pose(42, 47), 'parameters': {'radiusMm': 1.5, 'forceN': 0.}},
        {'operation': 'apply', 'duration': 1., 'pose': pose(42, 17)},
        {'operation': 'inspect', 'parameters': {'radiusMm': .5}}, {'operation': 'withdraw'},
        {'operation': 'place', 'pose': pose(20, 47), 'parameters': {'radiusMm': .6, 'forceN': 0.}},
    ]
    for action in actions:
        result = engine.step({'slot': 'left', 'duration': .2, **action})
        assert result['status'] in ('accepted', 'partial')
    assert engine.state['debrisMm3'] == 9.
    assert director.accounting(engine)['healthyRemovedMm3'] == 0.
    access = {'id': 'fixture', 'entryPose': pose(20, 15), 'outsidePose': pose(20, 47), 'drillRadiusMm': 3.}
    field = assistant_field_candidate(engine, access)
    assert field is not None and field['nearbyDebrisMm3'] == 9.
    meta = director.new_director(engine)
    meta.update(stage='assistant_setup', assistantStep=0, assistantField=field,
                coreAttempts=0, accesses=[access], accessIndex=0)
    return engine, meta, pose


def _active_assistant():
    engine, meta, pose = _opened_assistant_field()
    result, resumed = director.run_slice(engine, meta, max_actions=4)
    assert result.state['slots']['right']['instrument'] == 'frazier_suction'
    assert result.state['slots']['right']['active']
    return result, resumed, pose


def _conflicting_assistant_stroke(pose):
    return {'slot': 'left', 'operation': 'apply', 'duration': 2.,
            'parameters': {'radiusMm': 1.5, 'forceN': 0.},
            'path': [pose(20, 60), pose(42, 60), pose(42, 16.8)]}


def test_assistant_setup_clears_real_drill_debris_while_left_advances_shared_clock():
    engine, meta, _ = _opened_assistant_field()
    original = engine.snapshot()
    labels = array_digest(engine.scene.labels)
    result, _ = director.run_slice(engine, meta, max_actions=4)
    right = copy.deepcopy(result.state['slots']['right'])
    assert right['active'] and right['instrument'] == 'frazier_suction'
    before = result.state['debrisMm3']
    removed_before = result.state.get('debrisRemovedMm3', 0.)
    assert 0 < before < 9.
    assert result.step({'slot': 'left', 'operation': 'inspect', 'duration': .3})['status'] == 'accepted'
    assert result.state['slots']['right']['position'] == right['position']
    assert result.state['debrisMm3'] < before
    assert result.state['debrisRemovedMm3'] > removed_before
    assert result.state['debrisMm3'] + result.state['debrisRemovedMm3'] == pytest.approx(9.)
    assert director.accounting(result)['healthyRemovedMm3'] == 0.
    assert array_digest(engine.scene.labels) == labels and engine.snapshot() == original


def test_vascular_setup_cleans_both_hands_from_actual_removal_prefix_before_rescue(monkeypatch):
    from resectionlab import full_case_vascular
    engine, meta, _ = _active_assistant()
    initial, labels = engine.snapshot(), array_digest(engine.scene.labels)
    removed = engine.state['removed'][:]
    meta.update(scenario='vascular', stage='inspect')
    candidate = engine
    boundaries = []
    def fixture_plan(observed):
        boundaries.append(copy.deepcopy(observed.state['slots']))
        # This generic inspect only observes the handoff. No patient rescue
        # geometry or identity is substituted into this engineering fixture.
        action = {'slot': 'left', 'operation': 'inspect', 'duration': .2}
        return {'actions': [action], 'phases': [{'id': 'fixture-observation',
                'label': 'Engineering handoff observation', 'actions': [action]}]}
    monkeypatch.setattr(full_case_vascular, 'rescue_plan', fixture_plan)
    for _ in range(16):
        candidate, meta = director.run_slice(candidate, meta, max_actions=1)
        if meta['stage'] == 'vascular':
            break
        assert meta['status'] == 'running', meta.get('stopReason')
    assert meta['stage'] == 'vascular'
    assert len(boundaries) == 1
    assert not any(slot['active'] or slot['inserted'] or slot['grip']
                   for slot in boundaries[0].values())
    assert any(phase['id'] == 'vascular_setup' for phase in meta['phases'])
    assert not any(slot['active'] or slot['inserted'] or slot['grip']
                   for slot in candidate.state['slots'].values())
    assert not candidate.state['holds']
    assert candidate.state['removed'] == removed
    assert director.accounting(candidate)['healthyRemovedMm3'] == 0.
    assert engine.snapshot() == initial and array_digest(engine.scene.labels) == labels


def test_partial_assistant_collision_retracts_instead_of_discarding_access(monkeypatch):
    engine, meta, pose = _active_assistant()
    stroke = _conflicting_assistant_stroke(pose)
    preview = engine.preview(stroke, include_view=False)
    assert preview['status'] == 'partial' and preview['reason'] == 'TOOL_COLLISION:right'
    assert preview['deltas']['removedCells'] == []
    monkeypatch.setattr(director, 'next_removal_candidates', lambda *_a, **_k: [stroke])
    meta.update(stage='remove', cycleStartTargetMm3=engine.metrics()['targetRemovedMm3'])
    result, resumed = director.run_slice(engine, meta, max_actions=1)
    assert resumed['accessIndex'] == 0
    assert resumed.get('assistantResumeStage') == 'remove'
    assert resumed['currentPhaseId'] == 'assistant_retract'
    assert not result.state['slots']['right']['active']
    assert director.accounting(result)['healthyRemovedMm3'] == 0.


def test_assistant_retraction_parks_outside_following_full_instrument_envelope(monkeypatch):
    engine, meta, pose = _active_assistant()
    stroke = _conflicting_assistant_stroke(pose)
    monkeypatch.setattr(director, 'next_removal_candidates', lambda *_a, **_k: [stroke])
    meta.update(stage='assistant_retract', assistantResumeStage='remove',
                cycleStartTargetMm3=engine.metrics()['targetRemovedMm3'])
    result, _ = director.run_slice(engine, meta, max_actions=3)
    assert not result.state['slots']['right']['inserted'] and not result.state['slots']['right']['active']
    preview = result.preview(stroke, include_view=False)
    assert preview['status'] == 'accepted', preview.get('reason')
    assert preview['deltas']['removedCells']
    assert director.accounting(result)['healthyRemovedMm3'] == 0.


def test_checked_core_pocket_shrinks_before_checked_withdrawal():
    from resectionlab.surgical_physics_v2 import default_profile
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8:13, 8:13, 8:15] = 2
    labels[12, 10, 14] = 1
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True},
    ], {'kind': 'explicit-core-engineering-fixture'}, {'tissuePhysicsV2': default_profile()}))
    pose = lambda z: {'position': [10., 10., float(z)], 'quaternion': [0., 0., 0., 1.]}
    for action in [
        {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': pose(4), 'parameters': {'radiusMm': .6, 'forceN': 0.}},
        {'operation': 'apply', 'pose': pose(14), 'duration': 1.},
    ]:
        assert engine.step({'slot': 'left', 'duration': .2, **action})['status'] == 'accepted'
    before = engine.metrics()['targetRemovedMm3']
    from resectionlab.full_case_debulking import local_debulking_candidates
    defaults = local_debulking_candidates(engine)
    assert defaults and all(row['workingRadiusMm'] <= 1.5 for row in defaults)
    meta = director.new_director(engine)
    meta.update(stage='core', coreAttempts=0, corePositioning=0)
    first, meta = director.run_slice(engine, meta, max_actions=1)
    assert first.state['slots']['left']['position'] != engine.state['slots']['left']['position']
    assert first.parameters(first.state['slots']['left'])['radiusMm'] == .6
    assert first.metrics()['targetRemovedMm3'] == before
    second, _ = director.run_slice(first, meta, max_actions=1)
    assert .5 < second.parameters(second.state['slots']['left'])['radiusMm'] <= 6.
    assert second.metrics()['targetRemovedMm3'] > before
    assert director.accounting(second)['healthyRemovedMm3'] == 0.
    cleanup = director._prepare(second, 'cranial_drill', pose(4), {'radiusMm': 1.})
    assert cleanup['operation'] == 'inspect' and cleanup['parameters']['radiusMm'] == .5
    assert second.step(cleanup)['status'] == 'accepted'
    withdraw = director._prepare(second, 'cranial_drill', pose(4), {'radiusMm': 1.})
    assert withdraw['operation'] == 'withdraw'
    assert second.step(withdraw)['status'] == 'accepted'
    assert not second.state['slots']['left']['inserted']
    assert director.accounting(second)['healthyRemovedMm3'] == 0.


def test_selected_preview_is_reused_and_commit_matches_exact_public_state(monkeypatch):
    engine = _prepared()
    meta, _ = _removal_director(monkeypatch, engine)
    original = SurgicalEngine.preview
    calls = []
    def counted(self, action, **kwargs):
        result = original(self, action, **kwargs)
        calls.append((copy.deepcopy(action), result['stateId']))
        return result
    monkeypatch.setattr(SurgicalEngine, 'preview', counted)
    result, _ = director.run_slice(engine, meta, max_actions=1)
    assert len(calls) == 1
    assert result.state_id == calls[0][1]
    assert result.actions[-1] == calls[0][0]
    assert director.accounting(result)['healthyRemovedMm3'] == 0.


def test_two_frontier_batch_is_bounded_and_removes_only_actual_target():
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8:11, 8:11, 15] = labels[14:17, 8:11, 15] = 2
    labels[7:18, 7:12, 14] = 1
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True},
    ], {'kind': 'explicit-batching-engineering-fixture'}, {}))
    pose = lambda x, z: {'position': [float(x), 9., float(z)], 'quaternion': [1., 0., 0., 0.]}
    for action in [{'operation': 'exchange', 'instrument': 'cusa'},
                   {'operation': 'place', 'pose': pose(9, 35), 'parameters': {'radiusMm': 1.5}}]:
        assert engine.step({'slot': 'left', 'duration': .2, **action})['status'] == 'accepted'
    actions = [{'slot': 'left', 'operation': 'apply', 'duration': 1.5, 'parameters': {'radiusMm': 1.5},
                'path': [pose(9, 35), pose(x, 35), pose(x, 17)]} for x in (9, 15)]
    batched = director._batch_frontiers(actions)
    assert len(batched) == 3 and len(batched[0]['path']) <= 512
    before = engine.snapshot()
    preview = engine.preview(batched[0], include_view=False)
    committed = engine.step(batched[0])
    assert committed['status'] == 'accepted' and committed['stateId'] == preview['stateId']
    assert committed['metrics']['targetRemovedMm3'] == 2.
    assert director.accounting(engine)['healthyRemovedMm3'] == 0.
    assert len(before['actions']) == 2
    oversized = [{**action, 'path': [pose(x, 35)] * 300} for action, x in zip(actions, (9, 15))]
    assert director._batch_frontiers(oversized) == oversized


def test_partial_extra_parking_does_not_mark_the_assistant_as_parked():
    engine, previous, pose = _active_assistant()
    # A legally parked left instrument is clear of the initial right withdrawal
    # but obstructs the additional outward parking stroke. Real tool collision
    # must stop it without promoting a short endpoint into parked metadata.
    assert engine.step({'slot': 'left', 'operation': 'place', 'pose': pose(42, 137), 'duration': .2})['status'] == 'accepted'
    meta = director.new_director(engine)
    meta.update(stage='assistant_retract', assistantResumeStage='remove', assistantParked=False,
                assistantField=previous['assistantField'], accesses=previous['accesses'], accessIndex=0)
    result, resumed = director.run_slice(engine, meta, max_actions=3)
    assert resumed['lastOutcome']['status'] == 'partial'
    assert resumed['lastOutcome']['reason'] == 'TOOL_COLLISION:left'
    assert not resumed.get('assistantParked', False)
    assert director.accounting(result)['healthyRemovedMm3'] == 0.


@pytest.mark.parametrize('fine_steps', [150, 250])
def test_frontier_batch_preserves_fine_collar_return_or_declines_when_over_budget(fine_steps):
    labels = np.zeros((32, 24, 24), np.uint16)
    labels[5:22, 5:17, 18] = 3
    labels[9:13, 9:13, 18] = labels[15:19, 9:13, 18] = 0
    labels[10, 10, 15] = labels[16, 10, 15] = 2
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4), [
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 3, 'name': 'bone', 'material': 'bone'},
    ], {'kind': 'explicit-fine-collar-engineering-fixture'}, {}))
    pose = lambda x, z: {'position': [float(x), 10.08, float(z)], 'quaternion': [1., 0., 0., 0.]}
    for action in [{'operation': 'exchange', 'instrument': 'cusa'},
                   {'operation': 'place', 'pose': pose(10.08, 35), 'parameters': {'radiusMm': 1.5}}]:
        assert engine.step({'slot': 'left', 'duration': .2, **action})['status'] == 'accepted'
    actions = [{'slot': 'left', 'operation': 'apply', 'duration': 2., 'parameters': {'radiusMm': 1.5},
                'path': [pose(10.08, 35), pose(x, 35), pose(x, 22),
                         *[pose(x, z) for z in np.linspace(22, 17, fine_steps + 1)[1:]]]}
               for x in (10.08, 16.08)]
    original = copy.deepcopy(actions)
    source_hash = array_digest(engine.scene.labels)
    for action in actions:
        preview = engine.preview(action, include_view=False)
        assert preview['status'] == 'accepted' and preview['metrics']['targetRemovedMm3'] == 1.
    proposed = director._batch_frontiers(actions)
    assert actions == original
    if fine_steps == 250:
        assert proposed == actions, 'A fine return beyond512 poses must decline batching'
    else:
        assert len(proposed) == 3 and len(proposed[0]['path']) <= 512
        assert proposed[0]['duration'] > actions[0]['duration'] + actions[1]['duration']
        preview = engine.preview(proposed[0], include_view=False)
        assert preview['status'] == 'accepted', preview.get('reason')
        assert preview['metrics']['targetRemovedMm3'] == 2.
        assert preview['metrics']['nonTargetRemovedMm3'] == 0.
        committed = engine.step(proposed[0])
        assert committed['stateId'] == preview['stateId']
    assert array_digest(engine.scene.labels) == source_hash


@pytest.mark.parametrize('reservoir', [100., .01])
def test_vascular_endpoint_requires_actual_bleeding_then_source_control(reservoir):
    from resectionlab.surgical_physics_v2 import default_profile
    labels = np.zeros((24, 24, 24), np.uint16)
    # Cross the scissors' actual local-Y cut plane; axial contact alone need
    # not transect a vessel, even when the tip touches it.
    labels[10, 12:15, 12] = 1
    labels[18, 18, 18] = 2
    physics = default_profile()
    physics['vessel'].update(reservoirMm3=reservoir, refillConductanceMm3PerPaS=0.)
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'vessel', 'material': 'vessel'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True},
    ], {'kind': 'explicit-vascular-endpoint-engineering-fixture'}, {'tissuePhysicsV2': physics}))
    def endpoint(current):
        meta = director.new_director(current, 'vascular')
        meta['stage'] = 'endpoint'
        return director.run_slice(current, meta, max_actions=1)[1]
    assert endpoint(engine)['status'] == 'stopped'
    def step(operation, **fields):
        result = engine.step({'slot': 'left', 'operation': operation, 'duration': .2, **fields})
        assert result['status'] == 'accepted', result.get('reason')
    pose = lambda z: {'position': [10., 12., float(z)], 'quaternion': [0., 0., 0., 1.]}
    step('exchange', instrument='micro_scissors')
    step('place', pose=pose(8), parameters={'radiusMm': 1.5, 'jawOpeningMm': 0., 'forceN': 0.})
    step('apply', pose=pose(12), duration=1.)
    step('inspect', duration=2.)
    assert engine.metrics()['modeledVesselFluidGeneratedMm3'] > 0.
    assert endpoint(engine)['status'] == 'stopped'
    step('withdraw')
    step('exchange', instrument='bipolar_forceps')
    step('place', pose=pose(8), parameters={'radiusMm': 2., 'jawOpeningMm': 0., 'forceN': 0., 'power': 1.})
    step('apply', pose=pose(12), duration=1.)
    step('apply', duration=3.)
    step('deactivate')
    generated_after_energy = engine.state['vesselsV2']['generatedMm3']
    step('inspect', duration=5.)
    assert not engine.state['slots']['left']['active']
    assert engine.state['vesselsV2']['generatedMm3'] == generated_after_energy
    result = endpoint(engine)
    assert result['status'] == 'complete'
    assert 'does not claim target clearance' in result['stopReason']
    assert engine.metrics()['targetRemovedMm3'] == 0.
    assert all(not face['patent'] for face in engine.state['vesselsV2']['faces'].values())
