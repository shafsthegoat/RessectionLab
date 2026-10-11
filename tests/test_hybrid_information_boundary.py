"""Adversarial generated API checks; no patient admission, actor or training.

These tests deliberately distinguish public planning feedback from the rich
evaluator task. They do not assert equal actor/search representation bandwidth.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache, inject_native_capsule_cache
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task, OPENING_TOOLS
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.observed_search import observed_beam_search
from resectionlab.patient_planning_evaluation import evaluate_sealed_annotation_plans
from resectionlab.persistent_passive import make_passive_development_task, MOVE, INSPECT


def pair():
    base = make_native_opening_task().case
    refs = [np.array(base.reference_target), np.zeros_like(base.reference_target)]
    return [NativeSpatialTask(replace(base, reference_target=ref,
        track='inference_only', target_source_kind='derived_from_scan',
        target_derivation='fixed generated public estimate; not an MRI estimator',
        proposal_mode='nominal_cavity_v1',
        proposal_config=NominalCavityProposalConfig(((0, 0), (1, 1)), 96)), max_steps=2)
        for ref in refs]


def public_snapshot(task):
    observation = task.observation()
    return dict(fingerprint=observation.fingerprint,
        ids=observation.action_ids, inventory=task.candidate_inventory(),
        arrays={name: (np.asarray(getattr(observation, name)).dtype.str,
                       np.asarray(getattr(observation, name)).shape,
                       np.asarray(getattr(observation, name)).tobytes())
                for name in ('image_channels', 'coverage', 'channel_available',
                    'affine_ras_mm', 'spacing_mm', 'action_geometry', 'action_mask', 'state_features')})


def opening(task, number):
    tool, voxel = ((OPENING_TOOLS[0].tool_id, [4, 4, 1]),
                   (OPENING_TOOLS[1].tool_id, [4, 4, 5]))[number]
    return next(row['action_id'] for row in task.candidate_inventory()['ledger']
                if row['feasible'] and row['tool_id'] == tool and row['voxel'] == voxel)


def stable_cost(cost):
    # Timing is not an information-equivalence quantity.
    return {key: value for key, value in cost.items()
            if 'seconds' not in key and 'timing' not in key}


def test_swap_private_reference_cannot_change_public_previews_cache_or_nominal_rewards(monkeypatch):
    tasks = pair()
    assert tasks[0].case.reference_hash != tasks[1].case.reference_hash
    assert tasks[0].case.source_hash == tasks[1].case.source_hash
    original = NativeResectionEngine.preview_stroke
    logs = []
    def preview(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        logs.append((args[0], tuple(np.asarray(args[1]).tolist()),
            tuple(np.asarray(kwargs['entry_mm']).tolist()), result.feasible, result.reason,
            result.removed_indices_native.tolist(), result.contact_indices_native.tolist(),
            [step.to_dict() for step in result.microsteps], result.geometry_unknowns))
        return result
    monkeypatch.setattr(NativeResectionEngine, 'preview_stroke', preview)
    evidence = []
    for rich in tasks:
        nominal = rich.planning_clone()
        cache = ExactCapsuleCoverCache(nominal._config, max_payload_bytes=1024**2)
        logs.clear()
        with inject_native_capsule_cache(cache):
            snapshots = [public_snapshot(nominal)]
            rewards = []
            for step in range(2):
                result = nominal.step(opening(nominal, step))
                rewards.append((result.reward, result.info))
                snapshots.append(public_snapshot(nominal))
        stats = {k: v for k, v in cache.stats().items() if not k.endswith('_seconds')}
        evidence.append((snapshots, rewards, list(logs), stats))
    assert evidence[0] == evidence[1]
    assert evidence[0][3]['hits'] > 0 and evidence[0][3]['calls'] > 0


def test_same_bounded_search_and_stop_under_private_swap():
    results = [observed_beam_search(task, max_calls=6, beam_width=2,
        seconds=10., transition_mode='lazy_planning') for task in pair()]
    assert results[0][0] == results[1][0]
    assert stable_cost(results[0][1]) == stable_cost(results[1][1])


def test_negative_control_rich_evaluator_reward_is_not_an_actor_input():
    tasks = pair()
    for step in range(2):
        assert public_snapshot(tasks[0]) == public_snapshot(tasks[1])
        outcomes = [task.step(opening(task, step)) for task in tasks]
    assert outcomes[0].reward != outcomes[1].reward
    assert all(row.info['outcome_scope'] == 'separate_evaluator_reference' for row in outcomes)
    # Neither past total return nor private reference reward enters the DTO.
    assert public_snapshot(tasks[0]) == public_snapshot(tasks[1])


def test_negative_control_hiding_only_the_target_image_does_not_hide_annotation_proposals():
    tasks = pair()
    swapped = []
    for task in tasks:
        # Intentionally violating hypothetical "hidden annotation" provenance:
        # the case truthfully declares it public, while this test hides one plane.
        source = replace(task.case, track='annotation_assisted',
            nominal_target=task.case.reference_target,
            target_source_kind='supplied_annotation', target_derivation='negative control')
        current = NativeSpatialTask(source, max_steps=2)
        obs = current.observation()
        images = np.array(obs.image_channels); images[2] = 0
        swapped.append((images, obs.action_ids, current.candidate_inventory()))
    np.testing.assert_array_equal(swapped[0][0], swapped[1][0])
    assert swapped[0][1] != swapped[1][1]
    assert swapped[0][2] != swapped[1][2]


def test_unknown_zero_cell_cannot_become_free_or_tool_clearance():
    tissue = np.zeros((9, 9, 9), bool); tissue[4, 4, 5] = True
    domain = np.ones_like(tissue); domain[4, 4, 1] = False
    tool = ToolGeometry('generated-shaft', .9, .3, 4., 35., 1.)
    config = NativeResectionConfig(tissue, np.zeros(tissue.shape, np.int16), np.eye(4),
        AccessWindow([4, 4, 4.5], [0, 0, 1], 3), (tool,),
        'generated-only', 'explicit knownness test', interaction_domain=domain)
    engine = NativeResectionEngine(config)
    assert not engine.connected_free_mask[4, 4, 1]
    before = engine.state_hash
    refused = engine.execute_stroke(tool.tool_id, [4, 4, 5])
    assert not refused.feasible and refused.reason.startswith('UNKNOWN_DOMAIN:')
    assert engine.state_hash == before and not engine.removed_mask.any()
    # This is a changed public domain, not an interchangeable hidden field.
    assert NativeResectionEngine(replace(config, interaction_domain=None)).preview_stroke(
        tool.tool_id, [4, 4, 5]).feasible


def bundle(tmp_path):
    task = make_native_opening_task()
    initial = task.observation().fingerprint
    for i in range(2): task.step(opening(task, i))
    metrics = task.metrics(); history = metrics['history']
    nominal = deepcopy(history)
    for row in nominal: row['outcome_scope'] = 'permitted_nominal_model'
    context = {'patient_group': 'generated-independent-boundary'}
    plan = dict(context_hash=semantic_digest(context), source_hash=task.case.source_hash,
        decision_model_hash=task.decision_model_hash, initial_observation_hash=initial,
        max_steps=2, actions=[r['action_id'] for r in history], history=nominal)
    audit = evaluate_native_spatial_episode(task)
    record = dict(version='patient-native-preflight-sealed-plans-v1', status='complete',
        patient_context=context, source_hash=task.case.source_hash,
        decision_model_hash=task.decision_model_hash, initial_observation_hash=initial,
        tools=[asdict(t) for t in task.case.tools], expected_methods=['SEARCH', 'IL', 'RL'],
        plans=[dict(method=m, plan=deepcopy(plan), plan_seal=semantic_digest(plan),
            replayed_history=deepcopy(history), independent_geometry=deepcopy(audit))
               for m in ('SEARCH', 'IL', 'RL')])
    path = tmp_path / 'complete-plans.json'
    reference = dict(patient_group=context['patient_group'], public_source_hash=task.case.source_hash,
        frame_correspondence='qualified_common_RAS_mm_frame',
        mask=np.zeros((9, 9, 7), bool), coverage=np.ones((9, 9, 7), bool),
        affine_ras_mm=np.eye(4), source_kind='generated_control', label_name='generated-only',
        source_sha256='a'*64)
    return path, record, reference


@pytest.mark.parametrize('mutation', ['last_seal', 'last_incomplete', 'last_pose', 'last_audit'])
def test_all_methods_complete_before_private_callback(tmp_path, mutation):
    path, record, reference = bundle(tmp_path)
    row = record['plans'][-1]
    if mutation == 'last_seal': row['plan_seal'] = 'wrong'
    elif mutation == 'last_incomplete':
        row['plan']['max_steps'] = 3
        row['plan_seal'] = semantic_digest(row['plan'])
    elif mutation == 'last_pose':
        pose = row['replayed_history'][0]['tip_mm']
        row['replayed_history'][0]['tip_mm'] = [pose[0] + 1, *pose[1:]]
    else: row['independent_geometry']['accepted'] = False
    path.write_text(json.dumps(record))
    calls = []
    with pytest.raises(ValueError):
        evaluate_sealed_annotation_plans(path, expected_sha256=sha256(path.read_bytes()).hexdigest(),
            load_reference=lambda: calls.append(1) or reference)
    assert calls == []


def test_private_annotation_swap_only_changes_postseal_scores(tmp_path, monkeypatch):
    path, record, reference = bundle(tmp_path)
    path.write_text(json.dumps(record)); digest = sha256(path.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs): raise AssertionError('planner called after private load')
    monkeypatch.setattr(NativeSpatialTask, 'planning_clone', forbidden)
    monkeypatch.setattr(NativeSpatialTask, 'step', forbidden)
    reports = []
    for positive in (False, True):
        reference['mask'][4, 4, 1] = positive
        reports.append(evaluate_sealed_annotation_plans(path, expected_sha256=digest,
            load_reference=lambda: reference))
    assert sha256(path.read_bytes()).hexdigest() == digest
    assert [r['plan_seal'] for r in reports[0]['methods']] == [r['plan_seal'] for r in reports[1]['methods']]
    assert all(not r['contacts']['whole_tool']['annotated_positive_encounter'] for r in reports[0]['methods'])
    assert all(r['contacts']['whole_tool']['annotated_positive_encounter'] for r in reports[1]['methods'])


def test_current_passive_protocol_hidden_status_only_revealed_on_acquire():
    tasks = [make_passive_development_task(inspection_status=s) for s in ('verified', 'unresolved')]
    assert tasks[0]._state_seal != tasks[1]._state_seal  # Internal integrity, not public state.
    assert tasks[0].observation().fingerprint == tasks[1].observation().fingerprint
    outcomes = [t.step(MOVE) for t in tasks]
    assert outcomes[0].info == outcomes[1].info and outcomes[0].reward == outcomes[1].reward
    assert tasks[0].observation().fingerprint == tasks[1].observation().fingerprint
    for t in tasks:
        with pytest.raises(ValueError): t.planning_clone()
        t.step(INSPECT)
    assert tasks[0].observation().fingerprint != tasks[1].observation().fingerprint
    # Existing v1 has only a status difference; new useful decision extension
    # needs its own generated adversary, not a retroactive v1 action claim.
    assert tasks[0].observation().action_ids == tasks[1].observation().action_ids
