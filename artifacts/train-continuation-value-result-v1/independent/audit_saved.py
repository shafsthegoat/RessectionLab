"""Terminal-only stdlib saved-result audit. No model/native/array payload reads.

Run only after root supplies exact terminal report, launcher and source-index
hashes. Full saved historical traces are JSON; new compact route histories are
checked against their scalar certificates and joins, not a second native replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0, str(HERE))
from recorded_target_checks import SUBJECTS, semantic, validate_saved_episode, validate_corpus
from guarded_choice_checks import audit_choice

RL = 'build/post-exposure-learning-rl8-recovery-v1/RL8/attempt-01'
METHODS = ('GREEDY', 'IL64', 'RL8', 'Q_GUIDED')
CHECKS = 0
READS = {}


def need(value, label):
    global CHECKS
    CHECKS += 1
    if not value:
        raise ValueError(label)


def read_json(path, expected=None):
    path = Path(path)
    need(path.suffix == '.json' and not path.is_symlink(), 'Only regular JSON evidence')
    size = path.stat().st_size
    need(0 < size <= 8*1024**2, 'Bounded JSON bytes')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        need(digest == expected, 'Exact pinned JSON: '+str(path))
    READS[str(path.relative_to(ROOT))] = {'sha256': digest, 'bytes': len(raw)}
    return json.loads(raw)


def close(a, b, label, tol=1e-10):
    need(type(a) in (int, float) and type(b) in (int, float)
         and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, label)


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def check_target_rounding(actual, exact):
    # Independent direct summation can differ from the recurrence below double
    # precision. Permit at most one float32 representable step at the target,
    # never a fitted/performance tolerance. The stored label is explicitly f32.
    nearest = f32(exact)
    bits = struct.unpack('I', struct.pack('f', abs(nearest)))[0]
    step = struct.unpack('f', struct.pack('I', bits+1))[0]-abs(nearest)
    need(actual == f32(actual) and abs(actual-nearest) <= step+1e-12,
         'Float32 target encloses independent recorded-prefix arithmetic')


def valid_geometry(audit, label):
    need(audit['accepted'] is True and audit['complete_episode'] is True,
         label+' complete independent acceptance')
    geometry = audit['geometry']
    need(geometry['feasible'] is True and geometry['failures'] == []
         and geometry['complete_tool_checked'] and geometry['frontier_checked'],
         label+' full-tool/exposure geometry scope')


def audit_route(result, method, subject):
    need(result['complete'] is True and result['method'] == method,
         'Completed declared endpoint')
    actions, decisions, history = result['actions'], result['decisions'], result['history']
    n = len(actions)
    need(1 <= n <= 24 and n == len(decisions) == len(history), 'Complete bounded route counts')
    need(actions == [r['action_id'] for r in decisions] == [h['action_id'] for h in history],
         'Saved selected/action/history correspondence')
    need(decisions[-1]['terminated'] is True and all(not d['terminated'] for d in decisions[:-1]),
         'Only final route termination')
    need(('STOP' not in actions[:-1]) and
         ((actions[-1] == 'STOP' and result['terminal_reason'] == 'STOP') or
          ('STOP' not in actions and n == 24 and result['terminal_reason'] == 'HORIZON')),
         'Actual STOP or full horizon')
    world = result['world']
    need(world['subject'] == subject and world['max_steps'] == 24, 'Exact TRAIN subject/horizon')
    audit = result['audit']; valid_geometry(audit, subject+'/'+method)
    need(audit['committed_history_hash'] == result['history_hash'], 'Certificate full-history binding')
    for k in ('source_hash', 'decision_model_hash'):
        need(audit[k] == world[k], 'Same source/world for replay certificate')
    outcomes = audit['outcomes']
    total = 0.
    for i, (d, h) in enumerate(zip(decisions, history)):
        need(d['reward'] == h['reward'], 'Exact selected scalar reward')
        if i:
            need(decisions[i-1]['result_state_hash'] == d['source_state_hash'], 'Native state chain')
        if d['action_id'] == 'STOP':
            need(d['reward'] == 0 and d['source_state_hash'] == d['result_state_hash'], 'Unchanged STOP state/reward')
        else:
            need(h['outcome_scope'] == 'separate_evaluator_reference', 'Authoritative replay scope')
        total += d['reward']
    close(total, outcomes['total_reward'], 'Independent sum of complete route reward')
    close(sum(h['target_removed_mm3'] for h in history), outcomes['target_removed_mm3'], 'Target increment sum')
    close(sum(h['normal_removed_mm3'] for h in history), outcomes['normal_removed_mm3'], 'Outside increment sum')
    close(sum(h['complete_tool_path_length_mm'] for h in history), outcomes['complete_tool_path_length_mm'], 'Full path sum')
    need(sum(a != 'STOP' for a in actions) == outcomes['nonstop_actions'], 'NonSTOP denominator')
    plan = result['planning']
    need(plan['complete'] is True and result['planning_replay_join'] is True,
         'Complete planning and explicit replay join')
    need(len(plan['decisions']) == n and plan['model_transition_calls'] == n, 'Planning decision/transition counts')
    q_calls = 0; guards = []
    for old, new in zip(plan['decisions'], decisions):
        chosen = old.get('selected_action_id', old.get('action_id'))
        need(chosen == new['action_id'], 'Planning action equals executed action')
        for key in ('source_state_hash', 'result_state_hash', 'observation_hash', 'terminated'):
            if key in old:
                need(old[key] == new[key], 'Planning/replay '+key)
        if 'scores' in old:
            scores = old['scores']
            ids = [r['action_id'] for r in scores]
            rewards = [r['reward'] for r in scores]
            need(old['all_current_legal_actions_scored'] is True and
                 old['legal_nonstop_actions'] == len(scores)-1, 'Full certified inventory scored')
            predictions = None
            if method == 'Q_GUIDED' and old['actual_positive_candidates']:
                predictions = [r['predicted_q'] for r in scores]; q_calls += 1
            checked = audit_choice(ids, [True]*len(ids), rewards, predictions, chosen,
                                   method='guarded_Q' if method == 'Q_GUIDED' else 'greedy')
            need(checked['actual_selected_reward'] == new['reward'], 'Actual guard equals committed reward')
            guards.append(checked)
            if method == 'Q_GUIDED':
                need(old['actual_positive_candidates'] == checked['actual_positive_count'], 'Positive candidate count')
                reasons = {'no_actual_positive_action':'no_actual_positive_STOP',
                           'positive_q_among_actual_positive':'positive_Q_among_actual_positive',
                           'actual_greedy_fallback_no_positive_q':'no_positive_Q_greedy_fallback'}
                need(reasons[old['selection_reason']] == checked['reason'], 'Declared guarded choice reason')
        else:
            need(old['reward'] == new['reward'], 'Frozen actor replay reward')
    if 'estimated_incremental_return' in plan:
        need(total == plan['estimated_incremental_return'], 'Exact planning/replay reward accumulation')
    if method == 'Q_GUIDED':
        need(plan['nominal_history_hash'] == result['nominal_history_hash'], 'Full nominal history hash on replay')
        need(plan['source_hash'] == world['source_hash'] and plan['decision_model_hash'] == world['decision_model_hash'],
             'Q planning/replay source/model')
        need(plan['q_batches_started'] == plan['q_batches_completed'] == q_calls
             == plan['scorer_costs']['policy_forwards_completed']
             == plan['scorer_costs']['head_forwards'], 'One policy/head batch per productive state')
        need(plan['committed_transitions'] == n and plan['negative_preparation_allowed'] is False
             and plan['immediate_reward_added_to_Q'] is False and plan['successor_CNN_calls'] == 0,
             'Final restricted deployment/cost rule')
        need(plan['native_previews_started'] == plan['native_previews_completed'], 'Native preview started/returned join')
    return {'actions': n, 'motions': outcomes['nonstop_actions'],
            'target_mm3': outcomes['target_removed_mm3'], 'outside_mm3': outcomes['normal_removed_mm3'],
            'return': outcomes['total_reward'], 'path_mm': outcomes['complete_tool_path_length_mm'],
            'contact_upper_mm3': outcomes['cumulative_contacted_tissue_upper_bound_mm3'],
            'planning_previews': result['planning_costs']['native_preview_entries'],
            'planning_transitions': result['planning_costs'].get('native_transition_calls', 0),
            'planning_seconds': result['planning_costs']['complete_wall_seconds'],
            'replay_seconds': result['replay_costs']['complete_wall_seconds'],
            'q_batches': q_calls, 'choice_reasons': [r['reason'] for r in guards],
            'history_hash': result['history_hash'], 'nominal_history_hash': result['nominal_history_hash']}


def run(args):
    stage = HERE.parent; output = stage/'run-01'
    report = read_json(output/'report.json', args.report_sha256)
    launcher = read_json(output/'launcher.json', args.launcher_sha256)
    index = read_json(stage/'runner/source-index.json', args.index_sha256)
    need(index['execution_released'] is True, 'Exact root released source index')
    need(report['source_index_sha256'] == launcher['source_index_sha256'] == args.index_sha256,
         'Terminal source binding')
    need(launcher['report_sha256'] == args.report_sha256, 'Launcher binds exact final report')
    config = read_json(stage/'runner/config.json', index['files']['build/train-continuation-value-v1/runner/config.json'])
    need(report['config'] == config and config['methods'] == list(METHODS)
         and config['TRAIN'] == list(SUBJECTS), 'Exact modes and four TRAIN denominator')
    need(config['traces'] == 36 and config['states'] == 497 and config['head_updates'] == 64
         and config['max_steps'] == 24, 'Fixed learning endpoint')
    need(report['closed_roles_opened'] is False and report['new_training_rollouts'] == 0
         and report['original_collection_actor_reexecuted'] is False and report['base_optimizer_updates'] == 0,
         'No new collection, closed roles or base updates')
    reconstruction = report['reconstructed_traces']
    rows = []; labels = []
    manifest = read_json(output/'trace-manifest.json')
    need(semantic(manifest) == report['trace_manifest_hash'], 'Frozen trace-manifest binding')
    declarations = {r['key']:r for r in manifest}
    for item in reconstruction:
        subject, key = item['subject'], item['key']
        family = item['kind']
        update = None if family == 'teacher' else int(key.rsplit('RL', 1)[1])
        base = RL+'/teachers/'+subject if family == 'teacher' else RL+'/RL/update-%02d/'%update+subject
        evidence = [read_json(ROOT/(base+'/'+name), index['input_metadata'][base+'/'+name])
                    for name in ('complete-trace.json', 'plan.json', 'native-replay.json')]
        row = validate_saved_episode(*evidence, subject=subject, family=family, update=update)
        rows.append(row); labels.extend(row['labels'])
        need(item['states'] == row['steps'] and item['historical_trace_seal'] == row['trace_seal']
             and item['historical_plan_seal'] == row['plan_seal'], 'Reconstruction historical joins')
        need(item['descriptor_hash'] == semantic(declarations[key])
             and item['history_hash'] == declarations[key]['history_hash'], 'Actual reconstructed descriptor join')
        need(item['original_trace_sha256'] == index['input_metadata'][base+'/complete-trace.json']
             and item['original_replay_sha256'] == index['input_metadata'][base+'/native-replay.json'],
             'Original exact file references')
        valid_geometry(item['audit'], key)
        old_audit = evidence[2]['independent_geometry']
        need(item['audit']['committed_history_hash'] == old_audit['committed_history_hash'],
             'Reconstructed executed-history hash exactly historical')
    corpus_summary = validate_corpus(rows) if len(rows) == 36 else {'complete':False, 'reconstructed_traces':len(rows)}
    if 'corpus' in report:
        need(len(rows) == 36, 'A fit cannot use an incomplete corpus')
        logged = read_json(output/'logged-targets.json')
        need(logged['manifest_hash'] == report['trace_manifest_hash']
             and len(logged['targets']) == len(logged['weights']) == len(logged['STOP_rows']) == len(labels) == 497,
             'All497 labels, weights and STOP denominator')
        position = 0
        for row in rows:
            for label in row['labels']:
                check_target_rounding(logged['targets'][position], label['mandatory_logged_action_return'])
                need(logged['weights'][position] == f32(1./(4*9*row['steps'])), 'Exact float32 equal subject/trace/state weight')
                need(logged['STOP_rows'][position] is (label['action_id'] == 'STOP'), 'STOP rows retained')
                position += 1
        need(report['corpus']['states'] == 497 and report['corpus']['STOP_rows'] == sum(logged['STOP_rows'])
             and report['corpus']['costs']['policy_forwards_completed'] == 497,
             '497 reconstructed observation features')
    if 'head_fit' in report:
        fit = report['head_fit']; updates = fit['updates']
        need(len(updates) == 64 and [u['update'] for u in updates] == list(range(1,65))
             and report['head_optimizer_updates'] == 64, 'Exact64 linked head updates')
        need(fit['corpus_seal'] == report['corpus_seal'] and fit['base_parameter_hash'] == report['corpus']['base_parameter_hash'], 'Fit corpus/base binding')
        need(all(math.isfinite(u['weighted_mse_before_update']) and u['weighted_mse_before_update'] >= 0
                 and math.isfinite(u['gradient_norm_before_clip']) for u in updates), 'Finite complete scalar update series')
        need(fit['costs']['head_optimizer_steps'] == fit['costs']['head_backward_calls']
             == fit['costs']['head_training_forwards'] == 64
             and fit['costs']['base_optimizer_steps'] == fit['costs']['base_policy_forwards'] == 0,
             'Only the separate head fitted')
    outcomes = {s:{} for s in SUBJECTS}
    for subject, methods in report['cases'].items():
        need(subject in SUBJECTS and set(methods) <= set(METHODS), 'No excluded subjects/methods')
        for method, result in methods.items():
            if result.get('complete') is True:
                per_file = read_json(output/'evaluation'/subject/(method+'.json'))
                need(per_file == result, 'Per-arm file exactly agrees with final report')
                outcomes[subject][method] = audit_route(result, method, subject)
    comparison_complete = all(set(v) == set(METHODS) for v in outcomes.values())
    criterion = {'comparison_admissible':comparison_complete, 'useful_gain':False}
    if comparison_complete:
        checks = {}; tol = 1e-8
        for subject in SUBJECTS:
            a,b = outcomes[subject]['GREEDY'],outcomes[subject]['Q_GUIDED']
            worlds = report['cases'][subject]
            checks[subject] = {'same_world':worlds['GREEDY']['world'] == worlds['Q_GUIDED']['world'],
                'target_no_loss':b['target_mm3']+tol >= a['target_mm3'],
                'outside_no_increase':b['outside_mm3'] <= a['outside_mm3']+tol,
                'contact_no_increase':b['contact_upper_mm3'] <= a['contact_upper_mm3']+tol,
                'return_no_loss':b['return']+tol >= a['return']}
        def total(method,k): return sum(r[method][k] for r in outcomes.values())
        gains = {'fewer_planning_previews':total('Q_GUIDED','planning_previews') < total('GREEDY','planning_previews'),
                 'fewer_planning_transitions':total('Q_GUIDED','planning_transitions') < total('GREEDY','planning_transitions'),
                 'shorter_complete_tool_path':total('Q_GUIDED','path_mm') < total('GREEDY','path_mm')-tol,
                 'fewer_executed_nonstop_actions':total('Q_GUIDED','motions') < total('GREEDY','motions')}
        criterion.update(checks=checks,strict_gains=gains,useful_gain=all(all(x.values()) for x in checks.values()) and any(gains.values()))
        for key in ('comparison_admissible','useful_gain','checks','strict_gains'):
            need(report['useful_criterion'][key] == criterion[key], 'Independently recomputed criterion '+key)
    caps = config['caps']; budget = report['native_budget']
    need(report['total_frozen_policy_forwards'] <= caps['frozen_policy_forwards'], 'Total full-policy forward cap')
    need(budget['counting_reliable'] and budget['native_preview_entries'] <= caps['native_previews'], 'Native preview accounting/cap')
    lifecycle = launcher.get('owned_lifecycle') or {}
    clean = (launcher['status'] == 'completed' and launcher.get('stop_reason') is None
        and lifecycle.get('verified') and lifecycle.get('child_reaped')
        and lifecycle.get('exit_code') == 0 and not lifecycle.get('remaining_owned_pids')
        and not lifecycle.get('errors'))
    if launcher['status'] == 'completed':
        need(clean and launcher['worker_exit_code'] == 0 and report['status'] == 'completed', 'Clean completed supervision')
        need(launcher['sampled_peak_group_rss_bytes'] <= caps['worker_peak_rss_bytes']
             and report['self_peak_rss_bytes'] <= caps['worker_peak_rss_bytes']
             and launcher['elapsed_seconds'] < caps['parent_wall_seconds'], 'Recorded resource caps')
    return {'audit_scope':'saved JSON arithmetic and certificate/lineage joins; no native/model replay',
        'source_index_sha256':args.index_sha256,'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,
        'report_status':report['status'],'supervision_status':launcher['status'],'clean_supervision':bool(clean),
        'corpus':corpus_summary,'outcomes':outcomes,'independent_criterion':criterion,
        'total_native_previews':budget['native_preview_entries'],'total_policy_forwards':report['total_frozen_policy_forwards'],
        'head_updates':report['head_optimizer_updates'],'parent_seconds':launcher['elapsed_seconds'],
        'sampled_peak_rss_bytes':launcher['sampled_peak_group_rss_bytes'],
        'limits':['same four TRAIN worlds; no held-out evidence','only logged actions supervised; no Q* claim',
                  'actual-positive guard prevents negative preparation; not learned avoidance',
                  'compact new histories are hash/certificate joined, not independently re-executed',
                  'upfront reconstruction and fitting costs are included and unamortized'],
        'input_files':READS}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--report-sha256',required=True)
    p.add_argument('--launcher-sha256',required=True);p.add_argument('--index-sha256',required=True)
    p.add_argument('--output',required=True); args = p.parse_args()
    target = Path(args.output).resolve()
    need(target.parent == HERE and not target.exists(), 'Fresh audit output in owned scope')
    started = time.monotonic()
    def timeout(*unused): raise TimeoutError('Independent saved audit30s cap')
    signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,30.)
    try:
        result = run(args); result['audit_status'] = 'PASS_SAVED_EVIDENCE'; code = 0
    except BaseException as error:
        result = {'audit_status':'FAILED_REVIEWER_CHECK','failure':{'type':type(error).__name__,'message':str(error)},'input_files':READS};code=1
    signal.setitimer(signal.ITIMER_REAL,0)
    result.update(checks=CHECKS,seconds=time.monotonic()-started,model_calls=0,native_calls=0,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw = json.dumps(result,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    need(len(raw) <= 64*1024, 'Compact independent output64KiB cap')
    with target.open('xb') as stream: stream.write(raw)
    print(json.dumps({k:result[k] for k in ('audit_status','checks','seconds','model_calls','native_calls')}))
    return code


if __name__ == '__main__': raise SystemExit(main())
