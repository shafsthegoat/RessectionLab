"""Bounded saved-only review; no project, model, array or checkpoint decoding."""
import gzip
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGE = ROOT / 'build/train045-local-goal-ppo-v1'
RUN = STAGE / 'recovery-reverse1024-02'
PIN = {
    'report': 'dbbc0a67ca8b166cc3c22ffb2e27edbe01daaa052cbe89668d476161b6ed86a6',
    'launcher': 'ef62e3f6a0b29a2c227b5e2b921faa4b0a876032cffcc73f2cd56003303d1f7f',
    'index': '615ac1221618abcf5f0c34998a29941766b694c3dfb5f74f784b3033cac64e3c',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def semantic(value):
    return 'sha256:' + sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def raw(path, pin=None, cap=4 * 1024**2):
    path = Path(path)
    assert path.is_file() and path.stat().st_size <= cap, str(path)
    value = path.read_bytes()
    assert pin is None or sha(value) == pin, str(path)
    return value


def read(path, pin=None):
    return json.loads(raw(path, pin))


def gz(root, ref):
    data = raw(root / ref['path'], ref['sha256'])
    assert len(data) == ref['bytes']
    with gzip.GzipFile(fileobj=__import__('io').BytesIO(data)) as stream:
        decoded = stream.read(2 * 1024**2 + 1)
    assert len(decoded) <= 2 * 1024**2
    assert len(decoded) == ref['decompressed_bytes'] and sha(decoded) == ref['decompressed_sha256']
    return json.loads(decoded)


def close(a, b):
    assert abs(a - b) <= 1e-9 * max(1., abs(a), abs(b)), (a, b)


def episodes(directory):
    result = {'episodes': 0, 'decisions': 0, 'successes': 0, 'automatic_stops': 0,
              'by_start': {}, 'success_rows': [], 'immediate_STOP_by_start': {}}
    states, pairs = set(), set()
    for number, path in enumerate(sorted(directory.glob('episode-*.json.gz')), 1):
        with gzip.open(path, 'rb') as stream:
            data = stream.read(1024**2 + 1)
        assert len(data) <= 1024**2
        row = json.loads(data)
        assert row['episode'] == number and row['complete'] is True
        local, packets = row['local_rows'], row['action_records']
        auto = row['costs'].get('automatic_STOP_transitions', 0)
        assert len(packets) == len(local) + auto and auto in (0, 1)
        close(sum(x['reward'] for x in local), row['return'])
        assert sum(x['new_goal_removal'] for x in local) == int(row['success'])
        start = str(row['start_absolute_step'])
        group = result['by_start'].setdefault(start, {'episodes': 0, 'decisions': 0, 'successes': 0})
        group['episodes'] += 1
        group['decisions'] += len(local)
        group['successes'] += int(row['success'])
        if len(local) == 1 and local[0]['action_id'] == 'STOP':
            result['immediate_STOP_by_start'][start] = result['immediate_STOP_by_start'].get(start, 0) + 1
        for lr, packet in zip(local, packets):
            assert lr['action_id'] == packet['action_id'] and lr['source_state_hash'] == packet['source_state_hash']
            assert semantic({k: v for k, v in packet.items() if k != 'seal'}) == packet['seal']
            states.add(lr['source_state_hash'])
            pairs.add(semantic({'state': lr['source_state_hash'], 'action': packet['physical_action']}))
        result['episodes'] += 1
        result['decisions'] += len(local)
        result['successes'] += int(row['success'])
        result['automatic_stops'] += auto
        if row['success']:
            result['success_rows'].append({'episode': number, 'start': int(start), 'decisions': len(local),
                                           'return': row['return'], 'ending_new_decision': result['decisions']})
    result['unique_states'] = len(states)
    result['unique_state_actions'] = len(pairs)
    return result, states, pairs


def main():
    r = read(RUN / 'report.json', PIN['report'])
    parent = read(RUN / 'launcher.json', PIN['launcher'])
    index_path = STAGE / 'recovery-v2/source-index.json'
    idx = read(index_path, PIN['index'])
    assert r['source_index_sha256'] == parent['source_index_sha256'] == PIN['index']
    assert parent['report_sha256'] == PIN['report'] and parent['status'] == r['status'] == 'completed'
    life = parent['owned_lifecycle']
    assert life['verified'] and life['child_reaped'] and life['exit_code'] == parent['worker_exit_code'] == 0
    assert not life['errors'] and not life['remaining_owned_pids'] and not life['remaining_state_unknown']
    assert parent['stop_reason'] is None and r['source_unchanged'] and r['qualified']
    assert parent['sampled_peak_group_rss_bytes'] <= parent['caps']['worker_peak_rss_bytes']
    assert r['self_peak_rss_bytes'] <= parent['caps']['worker_peak_rss_bytes']
    for path, expected in idx['files'].items():
        assert Path(path).suffix not in ('.npy', '.npz', '.zip', '.pth', '.psckpt', '.qckpt')
        raw(ROOT / path, expected)
    assert r['subject'] == 'ReMIND-045' and r['role'] == 'TRAIN' and r['patients'] == 1
    assert not r['closed_roles_opened'] and len(r['source_visits']) == 1
    assert r['config']['absolute_horizon'] == 24 and r['config']['prefix_actions'] == 12
    assert r['config']['evaluation_start'] == 12 and r['config']['additional_decisions'] == 512
    assert r['context_continuity']['all_other_fields_exact']
    assert not r['recovery_admission']['original_campaign_completed']
    assert r['recovery_admission']['discarded_postcheckpoint_decisions_lower_bound'] == 320
    assert r['recovery_admission']['discarded_postcheckpoint_Adam_steps_lower_bound'] == 40
    arm = r['arms']['reverse-seed0']
    assert arm == read(RUN / 'reverse-seed0/curve.json') and arm['status'] == 'completed'
    c = arm['checkpoints'][0]
    prior = r['retained_reverse512']
    assert c['decisions'] == 1024 and c['additional_decisions'] == 512
    join = arm['optimizer_resume_join']
    assert join['all_tensors_and_optimizer_scalars_exact'] and join['prior_optimizer_completed'] == 353
    assert join['prior_decisions'] == 512 and arm['fresh_episode_and_RNG_at_resume']
    assert not arm['RNG_continuity_claim'] and not arm['discarded_postcheckpoint_work_enters_training']
    assert c['encoder_hash'] == prior['encoder_hash'] and not c['encoder_changed']
    assert c['optimizer_attempted'] == c['optimizer_completed'] == 415
    assert c['additional_optimizer_completed'] == c['optimizer_completed'] - 353 == 62
    assert len(c['gradient_updates']) == 415
    assert [x['completed'] for x in c['gradient_updates']] == list(range(1, 416))
    assert all(x['post_clip_encoder_gradient_norm'] == 0 for x in c['gradient_updates'])
    for ref, path in [(arm['restored_checkpoint'], ROOT / arm['restored_checkpoint']['path']),
                      (c['checkpoint'], RUN / c['checkpoint']['path'])]:
        assert len(raw(path, ref['sha256'], 2 * 1024**2)) == ref['bytes']
    e = c['evaluation']
    assert e == r['evaluations']['reverse-seed0-eval-1024']
    plan, history = gz(RUN, e['sealed_plan']), gz(RUN, e['history'])
    assert semantic({k: v for k, v in plan.items() if k != 'seal'}) == plan['seal'] == e['plan_seal']
    assert semantic(plan['packets']) == plan['packets_hash']
    assert semantic(history) == e['history_hash'] == e['audit']['committed_history_hash']
    assert plan['context_seal'] == r['local_context']['seal'] == semantic(r['local_context']['record'])
    assert e['complete'] and e['audit']['accepted'] and e['audit']['complete_episode']
    assert e['audit']['geometry']['feasible'] and e['audit']['geometry']['complete_tool_checked']
    assert not e['audit']['geometry']['failures'] and e['local_reward_audit_accepted']
    assert len(history) == len(plan['packets']) == 13 and len(plan['choices']) == len(plan['local_rows']) == 1
    assert history[-1]['action_id'] == plan['packets'][-1]['action_id'] == plan['choices'][0]['action_id'] == 'STOP'
    assert history[-1]['reward'] == history[-1]['normal_removed_mm3'] == history[-1]['target_removed_mm3'] == 0
    assert sum(x['reward'] for x in plan['local_rows']) == plan['return'] == e['return'] == 0
    assert not e['success'] and not any(x['new_goal_removal'] for x in plan['local_rows'])
    for packet in plan['packets']:
        assert semantic({k: v for k, v in packet.items() if k != 'seal'}) == packet['seal']
    uniform = r['retained_uniform1024']['evaluation']
    oldroot = STAGE / 'curriculum-1024-01'
    assert history == gz(oldroot, uniform['history'])
    assert plan['packets'] == gz(oldroot, uniform['sealed_plan'])['packets']
    assert not uniform['success'] and uniform['return'] == 0
    isolation = c['evaluation_isolation']
    assert isolation['unchanged_training_state'] and isolation['rng_restored'] and isolation['policy_mode_restored']

    ep, new_states, new_pairs = episodes(RUN / 'reverse-seed0')
    new = arm['segment_experience']
    assert ep['decisions'] == new['decisions'] == 512 and new['unfinished_episode_decisions'] == 0
    assert ep['episodes'] == new['completed_episodes'] and ep['successes'] == new['successful_episodes']
    assert new_states == set(new['source_state_hashes']) and new_pairs == set(new['state_physical_action_hashes'])
    for start in ('12', '13', '14', '15', '16'):
        group = ep['by_start'].get(start, {'decisions': 0, 'successes': 0})
        assert group['decisions'] == new['decisions_by_start'][start]
        assert group['successes'] == new['successful_episodes_by_start'][start]
    for reset in new['reset_records']:
        t = reset['recovery_decisions']
        assert reset['cumulative_decisions'] == 512 + t and reset['original_curriculum_decisions'] == 256 + t
        assert reset['start_step'] == (14 if t < 128 else 13 if t < 256 else 12)
    lineage = arm['experience']
    assert lineage['decisions'] == lineage['completed_episode_decisions'] + lineage['retired_unfinished_decisions_total'] + lineage['unfinished_episode_decisions'] == 1024
    assert lineage['retired_unfinished_decisions_total'] == 18
    assert set(lineage['source_state_hashes']) == set(prior['experience']['source_state_hashes']) | new_states
    assert set(lineage['state_physical_action_hashes']) == set(prior['experience']['state_physical_action_hashes']) | new_pairs

    old = read(oldroot / 'report.json', r['inherited_campaign']['report_sha256'])
    old_parent = read(oldroot / 'launcher.json', r['inherited_campaign']['launcher_sha256'])
    assert old['status'] == 'running' and old_parent['status'] == 'incomplete_or_failed' and old_parent['stop_reason'] == 'rss_cap'
    assert old_parent['last_saved_costs']['decisions'] == 832
    discarded, old_states, old_pairs = episodes(oldroot / 'reverse-seed0')
    assert discarded['decisions'] == 574 and 832 - 256 - discarded['decisions'] == 2
    campaign = read(ROOT / r['config']['campaign_report'])
    historical = campaign['arms']['trainable_encoder-seed0']['experience']
    us = old['arms']['uniform-seed0']['experience']
    known_states = set(lineage['source_state_hashes']) | set(us['source_state_hashes']) | set(historical['source_state_hashes'])
    known_pairs = set(lineage['state_physical_action_hashes']) | set(us['state_physical_action_hashes']) | set(historical['state_physical_action_hashes'])
    comp = r['comparison_experience']
    assert known_states == set(comp['source_state_hashes_known_union'])
    assert known_pairs == set(comp['state_physical_action_hashes_known_union'])
    # Count shared frozen256 once; the original trainable256 is separately paid.
    paid_decisions = 256 + 1024 + 1024 - 256 + 320
    assert comp['paid_PPO_decisions_lower_bound'] == paid_decisions == 2368
    known_states |= old_states
    known_pairs |= old_pairs

    costs = r['cost_segments']
    startup = r['config']['prior_startup_failure']
    sf = read(ROOT / startup['report']['path'], startup['report']['sha256'])
    sp = read(ROOT / startup['launcher']['path'], startup['launcher']['sha256'])
    assert not sf['arms'] and not sf['evaluations'] and sf['checkpoint_loads'] == 1
    assert sf['status'] == 'failed_or_incomplete' and sp['worker_exit_code'] == 1
    last = old_parent['last_saved_costs']
    assert r['native_budget']['counting_reliable'] and r['native_budget']['blocked_preview_attempts'] == 0
    assert r['native_transition_counts'] == {'attempted': 556, 'completed': 556, 'raised': 0}
    assert 28 + ep['decisions'] + ep['automatic_stops'] + 1 + 13 == 556
    assert r['native_budget']['native_preview_entries'] == 64192
    earlier = campaign['cost_segments']
    previews = earlier['cumulative_native_preview_entries'] + last['native_budget']['native_preview_entries'] + sf['native_budget']['native_preview_entries'] + 64192
    transitions = earlier['cumulative_native_transitions_attempted'] + last['transitions']['attempted'] + sf['native_transition_counts']['attempted'] + 556
    seconds = earlier['cumulative_worker_seconds'] + last['seconds'] + sf['worker_seconds'] + r['worker_seconds']
    assert previews == costs['cumulative_previews_lower_bound'] == 318705
    assert transitions == costs['cumulative_native_transitions_lower_bound'] == 2807
    close(seconds, costs['cumulative_worker_seconds_lower_bound'])
    pilot_parent = read(ROOT / r['config']['pilot_launcher'])
    campaign_parent = read(ROOT / r['config']['campaign_launcher'])
    paid_parent = sum(x['elapsed_seconds'] for x in (pilot_parent, campaign_parent, old_parent, sp, parent))
    search = read(STAGE / 'search-01/report.json', 'c5e5a8f61892ec9efa1c0ba25627df3249be2a0b16f384821852f8f69621f0c1')
    for key in ('goal', 'absolute_horizon', 'evaluation_start', 'prefix_actions', 'query', 'reward_policy', 'success'):
        assert search['config'][key] == r['config'][key]
    assert search['prefix_frontier'] == r['prefix_frontier']
    assert search['evaluation']['success'] and search['independent_audit']['accepted'] and search['local_reward_audit']['accepted']
    rolls = c['rollouts']
    steps = [(row['decisions'], (rolls[n+1]['optimizer_completed_before_update'] if n+1 < len(rolls) else 415) - row['optimizer_completed_before_update']) for n, row in enumerate(rolls)]
    return {
        'verdict': 'PASS_saved_evidence; completed recovery, goal failure; retain successful bounded search',
        'pins': {**PIN, 'curve': sha(raw(RUN / 'reverse-seed0/curve.json')), 'checkpoint': c['checkpoint']['sha256'],
                 'plan': e['sealed_plan']['sha256'], 'history': e['history']['sha256']},
        'source_files_sha_checked': len(idx['files']),
        'endpoint': {'goal_success': False, 'return': 0., 'decision': 'immediate STOP', 'full_history_rows': 13,
                     'fresh_geometry_accepted': True, 'identical_uniform_physical_history': True,
                     'isolation': isolation},
        'new_training': ep,
        'optimizer': {'lineage_completed': 415, 'restored': 353, 'new_completed': 62,
                      'new_steps_by_rollout': [x for x in steps if x[0] > 512],
                      'encoder_unchanged': True, 'final_logged_metrics': c['sb3_train_metrics'],
                      'paid_all_attempts_completed_lower_bound': 338 + 415 - 158 + 135 + 40},
        'critic_diagnostic': {
            'new_rollouts': [{'decisions': x['decisions'], 'values': x['rollout_values'],
                              'returns': x['returns'], 'advantages': x['advantages'],
                              'goal_bonus_events': x['goal_bonus_events']} for x in rolls[-8:]],
            'every_new_rollout_constant_saved_values': all(x['rollout_values']['std'] == 0 and
                x['rollout_values']['min'] == x['rollout_values']['max'] for x in rolls[-8:]),
            'interpretation': 'Constant sampled critic outputs within each rollout despite changing experience; not proof of missing inputs, dead activation, insufficient capacity or unique causal failure.'},
        'experience': {'recovered_lineage_decisions': 1024, 'completed_decisions': lineage['completed_episode_decisions'],
                       'retired_decisions': 18, 'live_decisions': 0,
                       'paid_all_branches_decisions_lower_bound': paid_decisions,
                       'known_all_paid_unique_states_lower_bound': len(known_states),
                       'known_all_paid_unique_state_actions_lower_bound': len(known_pairs),
                       'original_interrupted_reverse_completed_episode_decisions': discarded['decisions'],
                       'discarded_last_progress_unsaved_live_decisions': 2,
                       'unobserved_postprogress_work': None,
                       'overlap_scope': 'The 574 decisions include original interrupted reverse episodes before and after checkpoint512; unique-state/pair sets deduplicate retained overlap.'},
        'cost': {'current_worker_seconds': r['worker_seconds'], 'current_parent_seconds': parent['elapsed_seconds'],
                 'current_previews': 64192, 'current_transitions': 556,
                 'duration_scope': 'Runner/supervisor clock durations as reported; civil elapsed time and sleep effects not separately measured by this audit.',
                 'sampled_peak_bytes': parent['sampled_peak_group_rss_bytes'], 'self_peak_bytes': r['self_peak_rss_bytes'],
                 'all_training_attempts_parent_seconds': paid_parent,
                 'all_training_attempts_worker_seconds_lower_bound': seconds,
                 'all_training_attempts_previews_lower_bound': previews,
                 'all_training_attempts_transitions_lower_bound': transitions,
                 'endpoint_nominal': r['costs']['reverse-seed0-eval-1024.nominal'],
                 'endpoint_replay': r['costs']['reverse-seed0-eval-1024.fresh_replay_and_independent_geometry'],
                 'scope': 'Pilot,256,both curriculum branches including interrupted tail,startup failure,recovery02; qualification and search separate'},
        'search': {'success': True, 'return': search['evaluation']['return'],
                   'suffix_decisions': search['evaluation']['suffix_decisions'],
                   'claim': 'Retain search for this demonstrated task; paid model/reward/blocker queries differ from PPO tensor inputs, no global optimality or generalization claim'},
        'limits': ['No new geometry engine/replay: audited existing fresh independent full-tool certificates.',
                   'Training episodes are simulator experience, not independently replayed surgical observations.',
                   'Checkpoint bytes hashed only; no weights/cache/arrays decoded or model imports.',
                   'Reverse recovery restarts RNG/physical episode; it is not seamless continuation or a fully matched fresh timing comparison.',
                   'Sparse positive experience and STOP preference are observed; these data do not isolate a unique causal learning defect.'],
    }


if __name__ == '__main__':
    output = HERE / 'receipt.json'
    assert not output.exists(), 'Preserve previous audit receipt'
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('30s saved audit cap')))
    signal.alarm(30)
    try:
        result = main()
    except BaseException as error:
        result = {'verdict': 'FAILED_saved_reader', 'error': str(error), 'traceback': traceback.format_exc()}
    finally:
        signal.alarm(0)
    result.update(auditor_sha256=sha(Path(__file__).read_bytes()), elapsed_seconds=time.perf_counter()-started)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
    assert len(encoded) <= 65536
    with output.open('xb') as stream:
        stream.write(encoded)
    print(json.dumps({'verdict': result['verdict'], 'receipt_sha256': sha(encoded), 'bytes': len(encoded), 'seconds': result['elapsed_seconds']}))
    raise SystemExit(0 if result['verdict'].startswith('PASS') else 1)
