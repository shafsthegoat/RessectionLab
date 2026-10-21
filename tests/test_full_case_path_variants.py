"""New CUSA proposals use actual shared-engine previews and fine fallbacks."""
import copy
import sys

import numpy as np
import pytest

from resectionlab import full_case_debulking, full_case_rehearsal as director
from resectionlab.core import array_digest
from resectionlab.full_case_path_variants import cusa_path_variants
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def pose(x, z, y=10.08, q=(1., 0., 0., 0.)):
    return {'position': [float(x), float(y), float(z)], 'quaternion': list(q)}


def action(path):
    return {'slot': 'left', 'operation': 'apply', 'duration': 2.,
            'parameters': {'radiusMm': 1.5}, 'path': path}


@pytest.mark.parametrize('axis,sign', [(0, 1), (1, -1), (2, 1)])
def test_exact_axial_variants_preserve_endpoints_duration_and_untouched_fallback(axis, sign):
    path = []
    for value in range(5):
        p = pose(20., -10., y=3.)
        p['position'][axis] += sign * value / 4
        path.append(p)
    fine = action(path); original = copy.deepcopy(fine)
    coarse, fallback = cusa_path_variants(fine)
    assert coarse['path'] == [path[0], path[-1]] and fallback is fine
    assert fine == original and coarse['duration'] == fine['duration']
    assert coarse['parameters'] == fine['parameters']
    coarse['path'][0]['position'][0] += 1
    assert fine == original


def test_bends_reversals_duplicates_diagonals_and_changing_quaternions_are_retained():
    points = [pose(1, 4), pose(1, 3), pose(1, 2), pose(1, 2),
              pose(1, 3), pose(1, 4), pose(2, 4), pose(3, 4)]
    coarse, fine = cusa_path_variants(action(points))
    assert coarse['path'] == [points[i] for i in (0, 2, 3, 5, 7)]
    for path in ([pose(1, 1), pose(2, 2), pose(3, 3)],
                 [pose(1, 1), pose(1 + 1e-12, 2), pose(1, 3)],
                 [pose(1, 1), pose(1, 2, q=(-1., 0., 0., 0.)), pose(1, 3)]):
        proposed = action(path)
        assert cusa_path_variants(proposed) == [proposed]
    for operation in ('move', 'withdraw'):
        proposed = {**fine, 'operation': operation}
        assert cusa_path_variants(proposed) == [proposed]


def prepared(*, collar=False, healthy=False):
    labels = np.zeros((32, 24, 24), np.uint16)
    if collar:
        labels[5:22, 5:17, 18] = 3
        labels[9:13, 9:13, 18] = 0
    labels[10, 10, 15] = 2
    if healthy: labels[10, 10, 13] = 1
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'healthy', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 3, 'name': 'bone', 'material': 'bone'},
    ], {'kind': 'explicit-axial-path-engineering-fixture'}, {})
    engine = SurgicalEngine(scene)
    for a in [
        {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': pose(10.08, 35.), 'parameters': {'radiusMm': 1.5}},
    ]:
        assert engine.step({'slot': 'left', 'duration': .2, **a})['status'] == 'accepted'
    return engine


def stroke(end=17.):
    return action([pose(10.08, 35.), pose(10.08, 22.),
                   *[pose(10.08, z) for z in np.linspace(22., end, 151)[1:]]])


def meta_with_candidates(monkeypatch, engine, actions, stage):
    access = {'id': 'fixture', 'outsidePose': pose(10.08, 35.),
              'entryPose': pose(10.08, 17.), 'drillRadiusMm': 3.}
    monkeypatch.setattr(director, 'access_candidates', lambda *_a, **_k: [access])
    monkeypatch.setattr(director, '_frontier_candidates', lambda *_a, **_k: actions)
    monkeypatch.setattr(full_case_debulking, 'local_debulking_candidates',
        lambda *_a, **_k: [{'kind': 'core-removal', 'action': a} for a in actions])
    meta = director.new_director(engine)
    meta.update(stage=stage, cycleStartTargetMm3=0., coreAttempts=0, corePositioning=0)
    return meta


def preview_count(engine, proposed):
    calls = []
    previous = sys.getprofile()
    def profile(frame, event, _arg):
        if event == 'call' and frame.f_code is SurgicalEngine._substep.__code__: calls.append(1)
    sys.setprofile(profile)
    try: result = engine.preview(proposed, include_view=False)
    finally: sys.setprofile(previous)
    return result, len(calls)


@pytest.mark.parametrize('stage', ['core', 'remove'])
def test_safe_axial_coarse_removal_uses_own_preview_and_exact_commit(monkeypatch, stage):
    engine = prepared(); fine = stroke(); coarse = cusa_path_variants(fine)[0]
    before = engine.snapshot(); source_hash = array_digest(engine.scene.labels)
    preview, coarse_steps = preview_count(engine, coarse)
    fine_preview, fine_steps = preview_count(engine, fine)
    assert fine_preview['status'] == 'accepted' and 0 < coarse_steps < fine_steps
    assert preview['status'] == 'accepted' and preview['metrics']['targetRemovedMm3'] == 1.
    expected = engine.clone(); outcome = expected.step(coarse)
    assert outcome['stateId'] == preview['stateId'] and outcome['deltas'] == preview['deltas']
    meta = meta_with_candidates(monkeypatch, engine, [fine], stage)
    result, updated = director.run_slice(engine, meta, max_actions=1)
    assert result.actions[-1] == coarse and result.snapshot() == expected.snapshot()
    assert updated['attemptedActions'] == 1
    assert result.metrics()['nonTargetRemovedMm3'] == 0.
    assert engine.snapshot() == before and array_digest(engine.scene.labels) == source_hash


@pytest.mark.parametrize('stage', ['core', 'remove'])
def test_real_narrow_collar_coarse_fails_and_director_commits_fine(monkeypatch, stage):
    engine = prepared(collar=True); fine = stroke(); coarse, fallback = cusa_path_variants(fine)
    assert fallback is fine
    coarse_preview = engine.preview(coarse, include_view=False)
    assert coarse_preview['status'] != 'accepted'
    fine_preview = engine.preview(fine, include_view=False)
    assert fine_preview['status'] == 'accepted' and fine_preview['metrics']['targetRemovedMm3'] == 1.
    meta = meta_with_candidates(monkeypatch, engine, [fine], stage)
    result, updated = director.run_slice(engine, meta, max_actions=1)
    assert result.actions[-1] == fine and result.state_id == fine_preview['stateId']
    assert updated['attemptedActions'] == 2
    assert result.metrics()['nonTargetRemovedMm3'] == 0.
    assert any(row['action'] == coarse for row in updated['rejections'])


def test_core_refuses_actual_healthy_delta_in_coarse_and_fine_before_good_candidate(monkeypatch):
    engine = prepared(healthy=True); bad = stroke(12.); good = stroke()
    bad_coarse = cusa_path_variants(bad)[0]
    preview = engine.preview(bad_coarse, include_view=False)
    assert preview['status'] == 'accepted' and preview['metrics']['nonTargetRemovedMm3'] > 0.
    meta = meta_with_candidates(monkeypatch, engine, [bad, good], 'core')
    result, updated = director.run_slice(engine, meta, max_actions=1)
    assert result.actions[-1] == cusa_path_variants(good)[0]
    assert result.metrics()['targetRemovedMm3'] == 1. and result.metrics()['nonTargetRemovedMm3'] == 0.
    assert sum(r['reason'] == 'HEALTHY_OR_UNKNOWN_REMOVAL_REFUSED' for r in updated['rejections']) == 2


def test_zero_effect_coarse_positioning_falls_back_to_original_fine(monkeypatch):
    engine = prepared(); fine = stroke(24.)
    meta = meta_with_candidates(monkeypatch, engine, [fine], 'core')
    monkeypatch.setattr(full_case_debulking, 'local_debulking_candidates',
        lambda *_a, **_k: [{'kind': 'core-positioning', 'action': fine}])
    result, updated = director.run_slice(engine, meta, max_actions=1)
    assert result.actions[-1] == fine and updated['attemptedActions'] == 2
    assert result.metrics()['targetRemovedMm3'] == 0. and updated['corePositioning'] == 1
