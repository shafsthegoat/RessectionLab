"""One saved-only opening-pair audit; no task, model, numpy or native imports.

Run only after root supplies exact terminal report/launcher/index hashes.
The worker's independent executed-witness certificate supplies geometry validity;
this reader separately rebuilds leaf unions, rewards, costs and the useful gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
BASE = ROOT/'build/native-opening-coverage-v1'
OUT = Path(__file__).resolve().parent
MODES = ['return_only', 'return_plus_certified_evacuation_v1']
TOL = 1e-10
SUPPORT = {(4, 4, z) for z in range(1, 6)} | {(5, 5, 1)}
TARGET = {(4, 4, 4), (4, 4, 5)}
AFFINE = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
CONFIG_SHA = '19ed9907b82dce019212da33a3c5900c770637777410a8e237eba21c5ae85fb0'
COUNT = 0


def need(value, message):
    global COUNT
    COUNT += 1
    if not value:
        raise AssertionError(message)


def close(a, b, name):
    need(type(a) in (int, float) and type(b) in (int, float)
         and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= TOL, name)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path, expected=None):
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        need(sha == expected, 'exact bytes: '+str(path.relative_to(ROOT)))
    return json.loads(raw), {'path': str(path.relative_to(ROOT)), 'sha256': sha, 'bytes': len(raw)}


def complete_search(s):
    return (s.get('call_cap_reached') is False and s.get('time_cap_reached') is False
            and s.get('actor_forward_calls') == 0 and bool(s.get('layers'))
            and s.get('completed_layers') == len(s['layers'])
            and all(x.get('completed') is True for x in s['layers']))


def scalar_snapshot(snapshot, union, removed, leaves, name):
    count = sum(bits.bit_count() for bits in union.values())
    target = sum(bits.bit_count() for cell, bits in union.items() if cell in TARGET)/leaves
    source = count/leaves
    need(snapshot['certified_leaf_count'] == count, name+' leaf count')
    need(snapshot['whole_parent_clearance_count'] == len(removed), name+' whole count')
    need(snapshot['partial_certified_leaf_count'] == count-len(removed)*leaves, name+' partial count')
    close(snapshot['evacuated_source_volume_mm3'], source, name+' source')
    for bound in snapshot['evacuated_target_source_bounds_mm3']:
        close(bound, target, name+' binary-parent target')
    return source, target, source-target


def arm_readout(arm):
    mode, identity, metrics = arm['retention_mode'], arm['identity'], arm['metrics']
    history, actions, audit = arm['history'], arm['action_ids'], arm['audit']
    need(arm['status'] == 'completed' and arm['terminated'] and metrics['terminated'], mode+' complete')
    need([row['action_id'] for row in history] == actions, mode+' actions/history')
    need(1 <= len(actions) <= 8 and metrics['steps'] == len(actions), mode+' fixed horizon')
    need(actions[-1] == 'STOP' or len(actions) == 8, mode+' complete STOP/horizon')
    need('STOP' not in actions[:-1], mode+' no action after STOP')
    need(metrics['source_hash'] == identity['source_hash'] and metrics['decision_model_hash'] == identity['decision_model_hash'], mode+' identity')
    need(metrics['observation_track'] == 'synthetic_scan' and metrics['planning_estimator_only'] is False, mode+' authoritative generated scope')
    need(audit['feasible'] is True and audit['failures'] == [] and audit['complete_tool_checked'] and audit['frontier_checked'], mode+' independent full-tool certificate')
    need(audit['source_case_hash'] == identity['source_hash'], mode+' certificate source')
    need(audit['checker_version'] == 'independent-native-terminal-parent-closure-v1', mode+' declared closure checker')
    close(audit['source_voxel_volume_mm3'], 1., mode+' native voxel volume')
    model = identity['model']
    need(model['exposure_mode'] == 'same_step_parent_closure' and model['closure_schedule'] == 'terminal_microstep_only', mode+' model schedule')
    need(model['max_depth'] == 3, mode+' fixed leaf depth')
    leaves, full = 8**model['max_depth'], (1 << (8**model['max_depth']))-1
    union, removed, rows = {}, set(), []
    sums = {'target': 0., 'outside': 0., 'return': 0., 'path': 0.}
    previous_tool = None
    source = target = outside = 0.
    preparation = None
    last_snapshot = None
    for i, row in enumerate(history):
        aid = row['action_id']
        if aid == 'STOP':
            for key in ('reward', 'target_removed_mm3', 'normal_removed_mm3', 'insertion_distance_mm', 'complete_tool_path_length_mm'):
                close(row[key], 0., mode+' STOP '+key)
        else:
            need(row['native_affine'] == AFFINE and row['source_shape'] == [9, 9, 7], mode+' fixed native geometry')
            need(row['source_hash'] == identity['source_hash'], mode+' executed source')
            need(row['evacuated_volume']['spec'] == identity['volume'], mode+' volume spec')
            volume = row['evacuated_volume']
            before = scalar_snapshot(volume['before'], union, removed, leaves, mode+' before')
            if last_snapshot is not None:
                need(volume['before'] == last_snapshot, mode+' exact cumulative snapshot chain')
            native_removed = set()
            before_count = sum(v.bit_count() for v in union.values())
            for micro in row['microsteps']:
                proof = micro['cut_union']; added_count = 0
                for item in proof['cells']:
                    cell, bits = tuple(item['index']), int(item['added_bits_hex'], 16)
                    need(cell in SUPPORT and cell not in removed, mode+' declared occupied parent')
                    need(0 < bits <= full and not (union.get(cell, 0) & bits), mode+' disjoint new leaf bits')
                    union[cell] = union.get(cell, 0) | bits
                    added_count += bits.bit_count()
                need(proof['new_leaf_count'] == added_count, mode+' microstep leaf accounting')
                for value in micro['removed_indices_native']:
                    cell = tuple(value)
                    need(cell not in removed and union.get(cell) == full, mode+' whole removal requires complete certified union')
                    removed.add(cell); native_removed.add(cell)
            need(native_removed == {tuple(c) for c in row['removed_indices_native']}, mode+' action removal union')
            action_target = float(len(native_removed & TARGET))
            action_outside = float(len(native_removed-TARGET))
            close(row['removed_volume_mm3'], len(native_removed), mode+' removed volume')
            close(row['target_removed_mm3'], action_target, mode+' target reward input')
            close(row['normal_removed_mm3'], action_outside, mode+' outside reward input')
            distance = math.dist(row.get('physical_start_mm', row['entry_mm']), row['tip_mm'])
            close(row['insertion_distance_mm'], distance, mode+' physical insertion length')
            close(row['complete_tool_path_length_mm'], 2*distance, mode+' reverse path charged')
            weights = identity['reward']
            reward = (weights['target_per_mm3']*action_target-weights['normal_per_mm3']*action_outside
                      -weights['action_cost']-weights['motion_per_mm']*2*distance
                      -weights['tool_change_cost']*(previous_tool is not None and previous_tool != row['tool_id']))
            close(row['reward'], reward, mode+' exact unchanged reward')
            if preparation is None and action_target > 0:
                preparation = before[2]
            source, target, outside = scalar_snapshot(volume['after'], union, removed, leaves, mode+' after')
            new_count = sum(v.bit_count() for v in union.values())-before_count
            need(volume['delta_certified_leaf_count'] == new_count, mode+' action added bits')
            close(volume['delta_evacuated_source_volume_mm3'], new_count/leaves, mode+' delta volume')
            last_snapshot = volume['after']
            previous_tool = row['tool_id']
            sums['path'] += 2*distance
        progress = row['model_estimated_evacuated_progress']
        need(set(progress) == {'version', 'scope', 'source_hash', 'native_model_hash', 'native_state_hash', 'steps', 'action_id', 'certified_leaf_count', 'cumulative_evacuated_source_volume_mm3'}, mode+' public progress excludes reward/target/reference labels')
        need(progress['source_hash'] == identity['source_hash'] and progress['steps'] == i+1 and progress['action_id'] == aid, mode+' progress binding')
        need(progress['certified_leaf_count'] == sum(v.bit_count() for v in union.values()), mode+' public progress leaf count')
        close(progress['cumulative_evacuated_source_volume_mm3'], source, mode+' public source progress')
        for out, key in (('target', 'target_removed_mm3'), ('outside', 'normal_removed_mm3'), ('return', 'reward')):
            sums[out] += row[key]
        rows.append({'step': i+1, 'action_id': aid, 'tool_id': row.get('tool_id'), 'whole_target': row['target_removed_mm3'],
                     'whole_outside': row['normal_removed_mm3'], 'reward': row['reward'], 'certified_target': target, 'certified_outside': outside})
    if preparation is None:
        preparation = outside
    need(arm['chosen_nonstop_actions'] == sum(a != 'STOP' for a in actions) == audit['action_count'], mode+' selected action denominator')
    for k, mk in (('target', 'target_removed_mm3'), ('outside', 'normal_removed_mm3'), ('return', 'total_reward')):
        close(metrics[mk], sums[k], mode+' summed '+k)
    close(metrics['simulated_removed_volume_mm3'], len(removed), mode+' total removed')
    for k in ('claimed_source_tissue_volume_mm3', 'contained_source_tissue_volume_mm3'):
        close(audit[k], len(removed), mode+' certificate '+k)
    close(audit['unsupported_source_tissue_volume_mm3'], 0., mode+' no unsupported clearance')
    ev = audit['evacuated_volume']
    scalar_snapshot(ev, union, removed, leaves, mode+' independent final snapshot')
    if last_snapshot is not None:
        need(all(ev[k] == v for k, v in last_snapshot.items()), mode+' final executed/certificate snapshots')
    geometry = {'version': identity['volume']['version'], 'source_shape': [9, 9, 7], 'native_affine': AFFINE,
                'cut_union_spec': model, 'evacuated_volume_spec': identity['volume'], 'state_digest': ev['state_digest'],
                'whole_removed_indices': [list(c) for c in sorted(removed)],
                'partial_cells': [{'index': list(c), 'covered_bits_hex': format(bits, 'x')} for c, bits in sorted(union.items()) if c not in removed]}
    need('sha256:'+digest(geometry) == ev['geometry_digest'] == arm['physical']['geometry_digest'], mode+' independently rebuilt geometry digest')
    for k, value in (('source_mm3', source), ('target_mm3', target), ('outside_supplied_target_mm3', outside)):
        close(arm['physical'][k], value, mode+' independent '+k)
    close(arm['preparation_outside_mm3'], preparation, mode+' pre-first-target outside cost')
    search = arm['search']
    need(complete_search(search), mode+' complete search')
    need(search['beam_width'] == 2 and search['max_calls'] == 200 and search['transition_mode'] == 'lazy_planning', mode+' fixed search bounds')
    need(search['model_transition_calls'] <= 200 and search['completed_layers'] <= 8, mode+' search limits')
    need(sum(x['model_transition_calls'] for x in search['layers']) == search['model_transition_calls'], mode+' all layer costs')
    need(search['eager_transition_calls'] == 0 and search['lazy_planning_transition_calls'] == search['model_transition_calls'], mode+' lazy accounting')
    close(search['estimated_incremental_return'], sums['return'], mode+' planned/actual return')
    close(search['stop_baseline_return'], 0., mode+' unchanged STOP')
    return {'retention_mode': mode, 'whole_target_mm3': sums['target'], 'whole_outside_mm3': sums['outside'],
            'return': sums['return'], 'certified_target_mm3': target, 'certified_outside_mm3': outside,
            'preparation_outside_mm3': preparation, 'path_length_mm': sums['path'], 'motion_actions': arm['chosen_nonstop_actions'],
            'search_transitions': search['model_transition_calls'], 'completed_layers': search['completed_layers'],
            'planning_seconds': search['planning_seconds'], 'arm_seconds': arm['total_arm_seconds'], 'rows': rows,
            'geometry_digest': ev['geometry_digest']}


def main(args):
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('30s saved-only audit cap')))
    signal.alarm(30)
    output = {'schema': 'independent-generated-opening-pair-audit-v1', 'status': 'RUNNING', 'native_calls': 0, 'model_calls': 0, 'training_calls': 0}
    try:
        report, rb = read(BASE/'run-01/report.json', args.report_sha)
        launcher, lb = read(BASE/'run-01/launcher.json', args.launcher_sha)
        index, ib = read(BASE/'runner/source-index.json', args.index_sha)
        config, cb = read(BASE/'runner/config.json', CONFIG_SHA)
        output['bindings'] = {'report': rb, 'launcher': lb, 'source_index': ib, 'config': cb, 'canonical_commit': args.commit}
        need(index['execution_released'] is True, 'root released exact index')
        paths = list(index['files'])
        canonical = [p for p in paths if p.startswith('src/resectionlab/')]
        requests = ''.join(args.commit+':'+p+'\n' for p in canonical)
        proc = subprocess.run(['git', 'cat-file', '--batch'], input=requests.encode(), cwd=ROOT, capture_output=True, check=True, timeout=5)
        pos = 0
        for path in canonical:
            end = proc.stdout.index(b'\n', pos); header = proc.stdout[pos:end].split(); pos = end+1
            need(len(header) == 3 and header[1] == b'blob', 'committed source blob '+path)
            size = int(header[2]); raw = proc.stdout[pos:pos+size]; pos += size+1
            need(hashlib.sha256(raw).hexdigest() == index['files'][path], 'committed source hash '+path)
        for path in paths:
            if path not in canonical:
                need(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == index['files'][path], 'frozen noncanonical source '+path)
        need(report['config'] == config and report['source_index_sha256'] == launcher['source_index_sha256'] == args.index_sha, 'declaration/source joins')
        need(report['status'] == 'completed' and report['source_unchanged'] is True, 'saved scientific completion')
        need(launcher['report_sha256'] == args.report_sha and launcher['report_status'] == report['status'], 'launcher/report join')
        need([a['retention_mode'] for a in report['arms']] == MODES, 'all two arms in fixed execution order')
        need(config['arms'] == MODES and config['search'] == {'max_steps': 8, 'beam_width': 2, 'max_calls': 200, 'seconds': 120., 'transition_mode': 'lazy_planning'}, 'fixed predeclared treatment/search')
        a, b = report['arms']
        need(a['identity'] == b['identity'] and a['initial_inventory'] == b['initial_inventory'], 'exact same model/public source/mass/inputs/reward/candidates')
        for arm in report['arms']:
            inv = arm['initial_inventory']
            need(inv['complete'] and inv['omitted_count'] == 0 and digest(inv) == arm['identity']['initial_inventory_sha256'], 'complete bound initial inventory')
        rows = [arm_readout(arm) for arm in report['arms']]
        output['arms'] = rows
        c = report['positive_control']
        control_ok = c['passed'] is True and c['audit']['feasible'] is True and not c['audit']['failures'] and c['target_removed_mm3'] == 2. and c['total_reward'] > 0.
        need(control_ok, 'fixed positive geometric control')
        positive = all(r['whole_target_mm3'] > 0. and r['return'] > 0. for r in rows)
        x, y = rows
        independent = {'comparison_admissible': True, 'same_model_inputs_reward_candidates': True, 'both_selected_routes_positive': positive,
            'whole_target_no_loss': y['whole_target_mm3']+TOL >= x['whole_target_mm3'],
            'whole_outside_no_increase': y['whole_outside_mm3'] <= x['whole_outside_mm3']+TOL,
            'return_no_loss': y['return']+TOL >= x['return'],
            'certified_target_no_loss': y['certified_target_mm3']+TOL >= x['certified_target_mm3'],
            'final_certified_outside_no_increase': y['certified_outside_mm3'] <= x['certified_outside_mm3']+TOL,
            'strictly_fewer_search_transitions': y['search_transitions'] < x['search_transitions'],
            'strictly_lower_preparation_outside_volume': y['preparation_outside_mm3'] < x['preparation_outside_mm3']-TOL}
        independent['useful_gain'] = all(v for k, v in independent.items() if not k.startswith('strictly_')) and (independent['strictly_fewer_search_transitions'] or independent['strictly_lower_preparation_outside_volume'])
        for key, value in independent.items():
            need(report['useful_criterion'][key] == value, 'independent criterion '+key)
        need(report['useful_criterion']['tie_is_negative'] is True and report['useful_criterion']['numerical_allowance'] == TOL, 'fixed tolerance/tie semantics')
        need(launcher['useful_criterion'] == report['useful_criterion'], 'criterion receipt join')
        output['independent_criterion'] = independent
        costs = report['work_by_phase']
        for name, row in costs.items():
            need(row['previews_started'] == row['previews_returned'] == row['previews_feasible']+row['previews_refused'] and row['preview_exceptions'] == 0, name+' complete preview cost')
            need(row['transitions_started'] == row['transitions_returned'], name+' complete transitions')
            need(row['cached_inventory_requests'] <= row['inventory_requests'], name+' inventory scope')
            need(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in row.values()), name+' finite nonnegative costs')
        for mode, arm in zip(MODES, report['arms']):
            need(costs[mode+'.search']['transitions_started'] == arm['search']['model_transition_calls'], mode+' actual/search calls')
            need(costs[mode+'.replay']['transitions_started'] == len(arm['action_ids']), mode+' actual replay calls')
            need(costs[mode+'.independent_audit']['previews_started'] == 0 and costs[mode+'.independent_audit']['transitions_started'] == 0, mode+' independent checker not native rerun')
            need(arm['total_arm_seconds'] < 120., mode+' complete arm time cap')
        for output_key, phase_key in (('all_preview_calls_started', 'previews_started'), ('all_preview_calls_returned', 'previews_returned'), ('all_task_transitions_started', 'transitions_started')):
            need(report[output_key] == sum(r[phase_key] for r in costs.values()), output_key+' all phases charged')
        need(report['no_actor_or_optimizer_calls'] is True, 'zero actor/training declaration')
        progress = launcher['last_saved_costs']
        need(progress['active_preview'] is None and progress['phases'] == costs, 'final progress costs exact')
        evlane = b['search']['certified_evacuation_retention']
        need(evlane['terminal_objective_changed'] is False and evlane['extra_observations_or_previews'] == 0, 'one public retention delta only')
        need('certified_evacuation_retention' not in a['search'], 'baseline unchanged retention')
        clean = launcher.get('owned_lifecycle') or {}
        accepted = (launcher['status'] == 'completed' and launcher['stop_reason'] is None and launcher['worker_exit_code'] == 0
                    and clean.get('verified') is True and clean.get('child_reaped') is True and clean.get('exit_code') == 0
                    and not clean.get('remaining_owned_pids', [None]) and not clean.get('errors', [None]))
        need(launcher['caps'] == config['caps'] and launcher['automatic_retry'] is False, 'owned envelope/no retry')
        need(launcher['rss_samples'] > 0 and launcher['sampled_peak_group_rss_bytes'] <= config['caps']['worker_peak_rss_bytes'], 'sampled RSS cap')
        need(report['self_peak_rss_bytes'] <= config['caps']['worker_peak_rss_bytes'] and report['worker_seconds'] < 290 and launcher['elapsed_seconds'] < 300, 'reported RSS/wall caps')
        result_bytes = sum(p.stat().st_size for p in (BASE/'run-01').rglob('*') if p.is_file())
        need(result_bytes <= config['caps']['output_bytes'], 'saved output byte cap')
        output['supervision'] = {'accepted': accepted, 'status': launcher['status'], 'stop_reason': launcher['stop_reason'],
            'owned_lifecycle': clean, 'parent_seconds': launcher['elapsed_seconds'], 'worker_seconds': report['worker_seconds'],
            'sampled_peak_rss_bytes': launcher['sampled_peak_group_rss_bytes'], 'self_peak_rss_bytes': report['self_peak_rss_bytes'], 'output_bytes': result_bytes}
        output['work_by_phase'] = costs
        output['total_cost'] = {k: report[k] for k in ('all_preview_calls_started', 'all_preview_calls_returned', 'all_task_transitions_started')}
        output['limitations'] = ['One generated six-cell source, fixed order N=1; not patient, force, clinical, biological or learning evidence.',
            'The worker independently checked executed geometry; this audit rebuilt saved leaf unions/rewards and the gate without a native rerun.',
            'Binary generated public target equals the supplied evaluation target by fixed source construction; no unseen target labels enter retention.',
            'Certified partial evacuation is modeled progress; whole-parent occupancy and whole-cell reward stay unchanged.',
            'Search and nested setup/inventory/preview/integrity timings overlap and must not be summed. No general speed extrapolation.']
        output['status'] = 'PASS' if accepted else 'PASS_SAVED_SCIENCE_SUPERVISION_FAILED'
    except BaseException as error:
        output.update(status='FAIL', failure={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        signal.alarm(0)
        output['checks'] = COUNT
        output['audit_seconds'] = time.perf_counter()-started
        output['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        raw = (json.dumps(output, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        need(len(raw) <= 20*1024, 'compact audit output limit')
        with (OUT/'audit-result-01.json').open('xb') as stream:
            stream.write(raw)
    print(json.dumps({'status': output['status'], 'checks': COUNT, 'seconds': output['audit_seconds']}))
    return 0 if output['status'].startswith('PASS') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('report-sha', 'launcher-sha', 'index-sha', 'commit'):
        parser.add_argument('--'+key, required=True)
    raise SystemExit(main(parser.parse_args()))
