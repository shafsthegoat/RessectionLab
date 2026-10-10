"""Terminal-only saved JSON projection. No project/ML imports or payload reads.

Reuses the arithmetic of obstruction-opening-rl8-diagnosis-v1/derive.py, while
retaining every declared episode/endpoint slot when an attempt is unresolved.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = 'build/post-exposure-learning-rl8-recovery-v1/RL8'
OUTPUT = RUN + '/attempt-01'
TRAIN = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
STEPS = (1, 14, 1, 13)
RELEASE_SHA = '4da18b8bd21e6a54308d0b58b3d234c49c7351b56040e448ac7107a28a773e3e'
INDEX_SHA = '636f4b75ae95afa2a54c024882ca3b808246a6938db1fa1c5dd2a33443405589'
HEAD = 'e9a6ce8045a7c819c409d894573fdc5cef1980cd'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--result-sha256')
    args = parser.parse_args()
    inputs = {}; missing = []; issues = []

    def read(path, expected=None, optional=False):
        p = ROOT/path
        if optional and not p.exists():
            missing.append(path); return None
        if p.is_symlink() or not p.is_file() or not 0 < p.stat().st_size <= 8*1024**2:
            raise ValueError('Bounded regular JSON required: '+path)
        data = p.read_bytes(); digest = sha(data)
        if expected is not None and digest != expected:
            raise ValueError('Pinned JSON changed: '+path)
        value = json.loads(data)
        inputs[path] = digest
        return value

    # No attempt files are inspected before the root-pinned terminal receipt.
    receipt = read(OUTPUT+'.supervision/receipt.json', args.receipt_sha256)
    if (receipt.get('status') not in ('complete', 'failed_or_unresolved')
            or receipt.get('worker_termination_confirmed') is not True
            or type(receipt.get('exit_code')) is not int
            or receipt.get('final_owned_pids') != []
            or receipt.get('cleanup_errors') != []
            or receipt.get('release_sha256') != RELEASE_SHA
            or receipt.get('source_index') != {'path': RUN+'/source-index.json', 'sha256': INDEX_SHA}):
        raise ValueError('Exact terminal, reaped owned attempt required; no live projection')
    release = read(RUN+'/root-release.json', RELEASE_SHA)
    index = read(RUN+'/source-index.json', INDEX_SHA)
    if (release.get('expected_head') != HEAD or index.get('head') != HEAD
            or release.get('status') != 'released_one_attempt'
            or release.get('output') != OUTPUT or release.get('method') != 'RL'
            or release.get('TRAIN') != list(TRAIN)
            or release['execution_limits']['updates'] != 8):
        raise ValueError('Fixed four-TRAIN scratch RL8 release required')
    if args.result_sha256 is None:
        result_path = ROOT/OUTPUT/'result.json'
        if receipt.get('result_sha256') is not None or result_path.exists() or result_path.is_symlink():
            raise ValueError('A pinned or existing result requires its actual root-supplied hash')
        missing.append(OUTPUT+'/result.json'); result = {}
    else:
        result = read(OUTPUT+'/result.json', args.result_sha256)
        if receipt.get('result_sha256') not in (None, args.result_sha256):
            raise ValueError('Terminal parent and root result pins disagree')
        if result.get('method') != 'RL' or result.get('TRAIN') != list(TRAIN):
            raise ValueError('Result changed the declared endpoint or denominator')
        if result.get('SELECT_EVAL_opened') is not False or result.get('private_reference_reads') != 0:
            raise ValueError('Unexpected held-out/private use')

    episodes = []; positives = []; stops = []; decisions = []; updates = []
    previous_after = result.get('initial_parameter_hash')
    for update in range(1, 9):
        update_path = OUTPUT+f'/RL/update-{update:02d}'
        saved_update = read(update_path+'/update.json', optional=True)
        if saved_update is None:
            updates.append({'update': update, 'status': 'unavailable'})
        else:
            norms = saved_update['module_gradient_norms_before_clip']
            total = saved_update['gradient_norm_before_clip']
            if saved_update['completed_updates'] != update or (previous_after is not None and saved_update['before_parameter_hash'] != previous_after):
                issues.append(f'update-{update:02d}: parameter chain/count mismatch')
            previous_after = saved_update['after_parameter_hash']
            updates.append({'update': update, 'status': 'saved_shared_update', **saved_update,
                'clipped': total > release['learning_protocol']['max_gradient_norm'],
                'critic_head_squared_norm_fraction': norms['critic']**2/total**2 if total else None,
                'gradient_attribution_limit': 'Head norms do not separate actor/value contributions in the shared encoder.'})
        for subject in TRAIN:
            contribution = read(update_path+'/'+subject+'/gradient-contribution.json', optional=True)
            slot = {'update': update, 'subject': subject}
            if contribution is None:
                episodes.append({**slot, 'status': 'unavailable', 'return': None}); continue
            diag = contribution.get('rl_decision_diagnostics', {})
            ds = diag.get('decisions', [])
            local = []
            if not ds or len(ds) != contribution.get('steps') or len(ds) != contribution.get('loss_forward_calls'):
                local.append('decision/forward count mismatch')
            if saved_update is not None and diag.get('behavior_parameter_hash') != saved_update['before_parameter_hash']:
                local.append('behavior/update parameter mismatch')
            if diag.get('episodes_in_update') != 4:
                local.append('episode denominator mismatch')
            running = 0.; best = 0.; best_step = None
            for step, d in enumerate(ds):
                if (d['step'] != step or not math.isclose(sum(x['reward'] for x in ds[step:]), d['return_to_go'], rel_tol=1e-12, abs_tol=1e-10)
                        or not math.isclose(d['return_to_go_model_dtype']-d['value'], d['detached_advantage'], rel_tol=1e-6, abs_tol=1e-5)):
                    local.append(f'decision-{step}: RTG/advantage/order mismatch')
                running += d['reward']
                if running > best: best = running; best_step = step
                item = {**slot, **d, 'chosen_probability': math.exp(d['chosen_log_probability'])}
                decisions.append(item)
                if d['action_id'] != 'STOP' and d['reward'] > 0: positives.append(item)
                if d['action_id'] == 'STOP':
                    stops.append({**item, 'forced_by_legal_inventory': d['legal_action_count'] == 1})
            if not math.isclose(running, contribution['return'], rel_tol=1e-12, abs_tol=1e-10):
                local.append('episode reward sum mismatch')
            if ds and (not ds[-1]['terminated'] or any(d['terminated'] for d in ds[:-1])):
                local.append('incomplete/early-terminated trajectory')
            if local: issues.extend(f'{update}/{subject}: '+issue for issue in local)
            stop = next((d for d in ds if d['action_id'] == 'STOP'), None)
            root = ds[0] if ds else None
            episodes.append({**slot, 'status': 'scalar_checks_passed' if not local else 'inconsistent_saved_diagnostics',
                'steps': len(ds), 'return': contribution['return'],
                'positive_immediate_cuts': sum(d['action_id'] != 'STOP' and d['reward'] > 0 for d in ds),
                'best_observed_prefix_return_including_zero': best, 'best_prefix_last_step': best_step,
                'return_after_best_prefix': contribution['return']-best,
                'prefix_limit': 'Arithmetic of this sampled trajectory, not a newly replayed STOP policy.',
                'STOP_step': None if stop is None else stop['step'],
                'STOP_legal_count': None if stop is None else stop['legal_action_count'],
                'STOP_forced': None if stop is None else stop['legal_action_count'] == 1,
                'STOP_advantage': None if stop is None else stop['detached_advantage'],
                'STOP_probability_when_sampled': None if stop is None else math.exp(stop['chosen_log_probability']),
                'root': None if root is None else {k: root[k] for k in ('action_id','legal_action_count','value','entropy')},
                'root_sampled_action_probability': None if root is None else math.exp(root['chosen_log_probability']),
                'root_max_entropy_nats': None if root is None else math.log(root['legal_action_count'])})

    teacher_rows = []; endpoints = []
    for subject, count in zip(TRAIN, STEPS):
        for step in range(count):
            row = read(OUTPUT+f'/teacher-readout/{subject}/state-{step:02d}.json', optional=True)
            if row is None:
                teacher_rows.append({'subject': subject, 'step': step, 'status': 'unavailable'}); continue
            if row['subject'] != subject or row['step'] != step or row['endpoint'] != 'RL8':
                issues.append(f'{subject}/{step}: teacher-readout identity mismatch')
            legal = sum(row['action_mask']); scores = row['scores']
            teacher_rows.append({'subject': subject, 'step': step, 'status': 'saved_endpoint_readout',
                'teacher_action': row['teacher_action'], 'observation_hash': row['observation_hash'],
                'legal_actions': legal, 'forced_STOP': row['teacher_action'] == 'STOP' and legal == 1,
                **{k: v for k, v in scores.items() if k not in ('logits', 'probabilities')}})
        folder = OUTPUT+'/TRAIN-greedy/RL/'+subject
        plan = read(folder+'/plan.json', optional=True)
        replay = read(folder+'/native-replay.json', optional=True)
        saved = result.get('TRAIN_greedy', {}).get(subject)
        certificate = None if replay is None else replay['independent_geometry']
        history = [] if plan is None else plan['plan']['history']
        endpoints.append({'subject': subject, 'status': 'accepted_complete' if saved is not None and saved.get('complete') is True
            and certificate is not None and certificate.get('accepted') is True
            and certificate.get('complete_episode') is True and certificate.get('geometry', {}).get('feasible') is True
            else 'unresolved_or_unavailable',
            'result_row': saved, 'plan_seal': None if plan is None else plan['plan_seal'],
            'actions': None if plan is None else plan['plan']['actions'],
            'terminal_reason': None if plan is None else plan['plan']['terminal_reason'],
            'saved_plan_return': None if plan is None else sum(h['reward'] for h in history),
            'certificate_accepted': None if certificate is None else certificate.get('accepted'),
            'certificate_complete_episode': None if certificate is None else certificate.get('complete_episode'),
            'certificate_outcomes': None if certificate is None else certificate.get('outcomes'),
            'geometry_failures': None if certificate is None else certificate.get('geometry', {}).get('failures')})

    available = [r for r in episodes if r['return'] is not None]
    summary = {'status': 'terminal_saved_diagnostic_projection',
        'parent_status': receipt['status'], 'parent_exit_code': receipt['exit_code'], 'parent_stop_reason': receipt.get('stop_reason'),
        'parent_cleanup_errors': receipt.get('cleanup_errors'), 'result_status': result.get('status', 'unavailable'), 'result_failure': result.get('failure'),
        'result_available': bool(result), 'initial_parameter_pin_available': result.get('initial_parameter_hash') is not None,
        'episode_denominator': 32, 'available_episodes': len(available), 'decisions': len(decisions),
        'episode_returns': {key: sum(compare(r['return']) for r in available) for key, compare in
            [('positive', lambda x: x > 0), ('zero', lambda x: x == 0), ('negative', lambda x: x < 0)]},
        'positive_immediate_cuts': len(positives),
        'positive_cut_RTG_signs': {key: sum(compare(d['return_to_go']) for d in positives) for key, compare in
            [('positive', lambda x: x > 0), ('zero', lambda x: x == 0), ('negative', lambda x: x < 0)]},
        'positive_cut_advantage_signs': {key: sum(compare(d['detached_advantage']) for d in positives) for key, compare in
            [('positive', lambda x: x > 0), ('zero', lambda x: x == 0), ('negative', lambda x: x < 0)]},
        'STOP_decisions': len(stops), 'forced_STOP_decisions': sum(d['forced_by_legal_inventory'] for d in stops),
        'STOP_positive_advantage': sum(d['detached_advantage'] > 0 for d in stops),
        'STOP_negative_advantage': sum(d['detached_advantage'] < 0 for d in stops),
        'endpoint_denominator': 4, 'accepted_complete_endpoints': sum(r['status'] == 'accepted_complete' for r in endpoints),
        'teacher_readout_denominator': 29, 'optimizer_updates': result.get('optimizer_updates'),
        'checkpoint': result.get('checkpoints', {}).get('RL'),
        'parent_elapsed_seconds': receipt['elapsed_seconds'], 'sampled_peak_rss_bytes': receipt['sampled_peak_rss_bytes'],
        'missing_files': missing, 'arithmetic_or_identity_issues': issues,
        'scope': 'Saved scalar JSON only; no patient arrays, checkpoint payloads, model calls, replay or optimizer.',
        'interpretation_limits': ['Sampling entropy differs from greedy STOP.',
            'Negative full-continuation credit on a positive cut is not itself a bug.',
            'Module norms cannot attribute shared-encoder interference.',
            'No optimization, clinical, convergence or generalization claim is selected by this projection.']}
    # Recheck every file after reading, before publishing an immutable projection.
    if any(sha((ROOT/path).read_bytes()) != digest for path, digest in inputs.items()):
        raise ValueError('Terminal evidence changed during projection')
    destination = HERE/'projection-01'; destination.mkdir(exist_ok=False)
    for name, value in [('summary', summary), ('episodes', episodes), ('positive-cuts', positives), ('STOP-decisions', stops),
                        ('updates', updates), ('teacher-readouts', teacher_rows), ('endpoints', endpoints), ('input-index', inputs)]:
        with (destination/(name+'.json')).open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
