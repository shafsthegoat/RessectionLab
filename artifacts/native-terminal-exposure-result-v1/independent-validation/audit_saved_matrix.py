"""Terminal-only exposure-model matrix audit. No native or producer imports.

Reads exact root-supplied report/receipt/index hashes, independently clips the
checker-exported saved sparse unions with the unchanged plane evaluator, and
derives the predeclared criterion. Original failed attempts remain separate.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
import signal
import struct
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = ROOT/'build/native-partial-volume-v1/independent-review/plane_volume_evaluator.py'
EVALUATOR_SHA = '21df1470e7a65c62eaae4857c8b7c435b1ae448783934465d3cf3bdef2a82522'
BASELINE_SHA = 'c9265daaf186b025403ee499a333f0e2811feab66f65e10a98a3d97f17300ee9'
checks = 0
inputs = {}


def need(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def close(a, b, label, tolerance=1e-8):
    need(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, expected=None, limit=8*1024**2):
    path = ROOT/path if not Path(path).is_absolute() else Path(path)
    need(path.resolve().is_relative_to(ROOT) and path.suffix == '.json', 'local saved JSON')
    raw = path.read_bytes()
    need(len(raw) <= limit, 'bounded saved JSON')
    fingerprint = hashlib.sha256(raw).hexdigest()
    if expected:
        need(fingerprint == expected, 'exact input '+str(path.relative_to(ROOT)))
    inputs[str(path.relative_to(ROOT))] = {'sha256': fingerprint, 'bytes': len(raw)}
    return json.loads(raw)


def binding(record):
    return read(record['path'], record['sha256'])


def semantic(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def legacy_digest(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def equivalent(a, b, label, tolerance=1e-8):
    if isinstance(a, dict) and isinstance(b, dict):
        need(set(a) == set(b), label+' keys')
        for key in a:
            equivalent(a[key], b[key], label+'/'+key, tolerance)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        need(len(a) == len(b), label+' length')
        for i, (x, y) in enumerate(zip(a, b)):
            equivalent(x, y, label+'/'+str(i), tolerance)
    elif type(a) in (int, float) and type(b) in (int, float):
        close(a, b, label, tolerance)
    else:
        need(type(a) is type(b) and a == b, label)


def projection(row):
    ignored = {'source_state_hash', 'result_state_hash', 'native_seconds', 'partial_evacuation'}
    return {'native': {k: v for k, v in row['native'].items() if k != 'committed_history_hash'},
            'actions': [{k: v for k, v in a.items() if k not in ignored} for a in row['actions']]}


def equal_numeric(a, b, tolerance=1e-10):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equal_numeric(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal_numeric(x, y, tolerance) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return abs(a-b) <= tolerance
    return type(a) is type(b) and a == b


def determinant(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def committed_canonical_hashes(commit, names):
    """Authenticate the executed committed bytes despite authorized next edits."""
    resolved = subprocess.run(['git', 'rev-parse', commit+'^{commit}'], cwd=ROOT,
        check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    request = ''.join(resolved+':'+name+'\n' for name in names).encode()
    raw = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT, input=request,
        check=True, capture_output=True, timeout=5).stdout
    offset = 0
    result = {}
    for name in names:
        end = raw.index(b'\n', offset)
        fields = raw[offset:end].decode().split()
        need(len(fields) == 3 and fields[1] == 'blob', 'committed canonical blob '+name)
        size = int(fields[2]); begin = end+1
        blob = raw[begin:begin+size]
        need(len(blob) == size and raw[begin+size:begin+size+1] == b'\n', 'git blob framing')
        result[name] = hashlib.sha256(blob).hexdigest(); offset = begin+size+1
    need(offset == len(raw), 'no unparsed committed source output')
    return resolved, result


def audit(args):
    began = time.perf_counter()
    run = ROOT/args.run_directory
    r = read(run/'report.json', args.report_sha256)
    receipt = read(run/'launcher.json', args.launcher_sha256)
    index = read(args.source_index, args.source_index_sha256)
    cfg_path = ROOT/args.config
    cfg = read(cfg_path)
    need(index['execution_released'] is True, 'explicit released source index')
    need(receipt['source_index_sha256'] == args.source_index_sha256, 'receipt source index')
    need(index['files'][str(cfg_path.relative_to(ROOT))] == sha(cfg_path), 'config authenticated by closure')
    commit, canonical = committed_canonical_hashes(args.canonical_commit,
        [path for path in index['files'] if path.startswith('src/resectionlab/')])
    for path, fingerprint in index['files'].items():
        actual = canonical[path] if path in canonical else sha(ROOT/path)
        need(actual == fingerprint, 'executed source/reference '+path)
    failed_dir = Path('build/native-cut-residual-diagnostic-v1/comparison-run-01')
    failed_receipt = read(failed_dir/'launcher.json', index['files'][str(failed_dir/'launcher.json')])
    failed_progress = read(failed_dir/'progress.json', index['files'][str(failed_dir/'progress.json')])
    need(failed_receipt['status'] == 'incomplete_or_failed' and failed_receipt['child_reaped']
         and not failed_receipt['remaining_owned_pids'] and not failed_receipt['cleanup_errors'], 'original failed attempt retained/reaped')
    need('FileNotFoundError' in failed_receipt['stop_reason'] and 'report.json.tmp' in failed_receipt['stop_reason'],
         'original monitor race, not reclassified model failure')
    failed_calls = failed_progress['native_call_costs_including_incomplete_row']
    need(failed_calls['native_strokes_started'] == failed_calls['native_strokes_returned'] == 11,
         'original attempted native work retained')
    need(r['source']['files'] == index['files'], 'executed closure matches index')
    need(receipt['report_sha256'] == args.report_sha256, 'terminal report join')
    need(r['status'] == receipt['status'] == 'completed' and receipt['worker_exit_code'] == 0
         and receipt['stop_reason'] is None, 'successful owned result')
    need(receipt['child_reaped'] and not receipt['remaining_owned_pids'] and not receipt['cleanup_errors'], 'clean terminal reap')
    caps = cfg['caps']
    need(receipt['caps'] == caps and caps['threads'] == 1, 'same declared resource caps')
    need(receipt['elapsed_seconds'] < caps['parent_wall_seconds']
         and receipt['sampled_peak_group_rss_bytes'] <= caps['worker_peak_rss_bytes']
         and receipt['output_bytes_before_receipt'] < caps['output_bytes'], 'actual parent bounds')
    need(cfg['retained_baseline_report']['sha256'] == BASELINE_SHA, 'fixed partial-volume reference')
    prior = binding(cfg['retained_baseline_report'])
    decl = binding(cfg['retained_union_declaration'])
    correction = binding(cfg['retained_baseline_parity_correction'])
    need(correction['all_18_saved_projections_exactly_equal'] is True
         and correction['corrected_criterion']['meets_predeclared_useful_criterion'] is True,
         'separate saved parity correction retained')
    need(r['retained_baseline_report'] == cfg['retained_baseline_report']
         and r['retained_baseline_parity_correction'] == cfg['retained_baseline_parity_correction'], 'report baseline joins')
    need(r['declaration_hash'] == legacy_digest({**decl, 'version': cfg['version'], 'caps': caps}), 'fixed scientific declaration')
    need(r['source_unchanged'] and r['human_patients'] == r['training_updates'] == 0, 'no patient or training scope')
    need(r['model'] == cfg['model'] == 'same_capsule_parent_exposure_closure', 'single model intervention')
    expected = [g['name']+'-'+format(h, 'g')+'mm' for h in decl['spacings_mm'] for g in decl['grids']]
    need([x['row_id'] for x in r['rows']] == expected == r['planned_rows']
         and not r['missing_rows'] and len(expected) == 18, 'exact fixed row order')
    need(r['completed_rows'] == r['expected_rows'] == receipt['planned_rows'] == 18
         and r['planned_action_slots'] == receipt['planned_action_slots'] == 180, 'full denominators')
    need(sha(EVALUATOR) == EVALUATOR_SHA, 'frozen independent clipping source')
    spec = importlib.util.spec_from_file_location('independent_saved_planes', EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    old_rows = {x['row_id']: x for x in prior['rows']}
    counts, work = Counter(), Counter()
    summaries = []
    native_seconds = numerical_seconds = audit_seconds = clipping_seconds = 0.
    tolerance = cfg['physical_volume_tolerance_mm3']
    need(tolerance == 1e-8, 'predeclared physical tolerance')
    for row in r['rows']:
        rid = row['row_id']; old = old_rows[rid]; acts = row['actions']; native = row['native']; pe = row['partial_evacuation']
        need(len(acts) == row['planned_actions'] == 10, rid+' action denominator')
        for key in ('spacing_mm', 'grid', 'shape', 'affine', 'source_hash', 'fixed_tool', 'physical_actions_sha256',
                    'static_not_executable', 'exact_continuous_target_cap_mm3', 'public_parent_mass_spec'):
            need(row[key] == old[key], rid+' fixed '+key)
        cut = row['cut_union_spec']
        need(row['exposure_model'] == cut and cut['mode'] == 'cumulative_affine_parent_closure_cut_union_v3'
             and cut['exposure_mode'] == cfg['exposure_mode'] == 'same_step_parent_closure', rid+' explicit mode')
        for key in ('max_depth', 'max_tracked_cells', 'max_leaf_tests_per_preview'):
            need(cut[key] == old['cut_union_spec'][key], rid+' fixed '+key)
        need(row['retained_baseline_projection'] == projection(old), rid+' baseline projection intact')
        need(row['retained_baseline_projection_sha256'] == legacy_digest(projection(old))
             and row['current_action_result_projection_sha256'] == legacy_digest(projection(row)), rid+' projection hashes')
        same_projection = equal_numeric(projection(old), projection(row))
        need(row['action_result_projection_equal'] == same_projection, rid+' honest outcome changes')
        volume = abs(determinant(row['affine'])); leaves = 8**cut['max_depth']
        accepted = attempted = leaf_delta = 0
        delta_volume = total_reward = 0.
        state = evacuation = None
        refused = False
        for action, old_action in zip(acts, old['actions']):
            for key in ('index', 'kind', 'tool_id', 'entry_mm', 'tip_mm'):
                need(action[key] == old_action[key], rid+' unchanged action '+key)
            counts[action['status']] += 1
            if action['status'] == 'unexecuted_prior_refusal':
                need(refused, rid+' suffix retained after refusal')
                continue
            need(not refused, rid+' no implicit retry')
            attempted += 1; native_seconds += action['native_seconds']
            if state is not None:
                need(action['source_state_hash'] == state, rid+' native state chain')
            attempt = action['cut_union_attempt']
            for key in ('leaf_tests', 'new_leaf_count', 'witness_rows'):
                work[key] += attempt[key]
            work['peak_tracked_cells'] = max(work['peak_tracked_cells'], attempt['peak_tracked_cells'])
            work['budget_refusals'] += attempt['budget_refused']
            need(attempt['leaf_tests'] <= cut['max_leaf_tests_per_preview']
                 and attempt['peak_tracked_cells'] <= cut['max_tracked_cells'], rid+' per-preview proof bounds')
            if action['status'] == 'rejected':
                refused = True
                need(action['result_state_hash'] == action['source_state_hash']
                     and 'partial_evacuation' not in action, rid+' exact rejected rollback')
                continue
            need(action['status'] == 'accepted', rid+' known disposition')
            accepted += 1; state = action['result_state_hash']
            v = action['partial_evacuation']; before = v['before']; after = v['after']
            if evacuation is None:
                need(before['certified_leaf_count'] == 0 and before['evacuated_source_volume_mm3'] == 0, rid+' empty initial union')
            else:
                equivalent(before, evacuation, rid+' evacuation chain')
            need(v['spec'] == row['public_parent_mass_spec'], rid+' public parent interval contract')
            d = after['certified_leaf_count']-before['certified_leaf_count']
            need(d == v['delta_certified_leaf_count'] == action['cut_union_committed']['new_leaf_count'] and d >= 0, rid+' disjoint additions')
            close(v['delta_evacuated_source_volume_mm3'], d*volume/leaves, rid+' leaf-volume delta')
            close(after['evacuated_source_volume_mm3']-before['evacuated_source_volume_mm3'], v['delta_evacuated_source_volume_mm3'], rid+' cumulative volume')
            leaf_delta += d; delta_volume += v['delta_evacuated_source_volume_mm3']; evacuation = after
            target = action['nominal_target_source_credit_mm3']; normal = action['removed_source_cell_mm3']-target
            reward = decl['reward']
            value = (reward['target_per_mm3']*target-reward['normal_per_mm3']*normal-reward['action_cost']
                     -reward['motion_per_mm']*2*math.dist(action['entry_mm'], action['tip_mm']))
            close(value, action['frozen_reward'], rid+' unchanged reward coefficients and whole-cell basis')
            total_reward += action['frozen_reward']
        need(attempted == row['native_calls'] and accepted == native['accepted_actions']
             and native['complete_sequence'] == (accepted == 10), rid+' count/complete truth')
        close(total_reward, native['frozen_reward_total'], rid+' reward sum')
        counts['complete'] += native['complete_sequence']; counts['empty'] += accepted == 0
        radius = row['fixed_tool']['tip_radius_mm']
        need(all(a['tip_mm'][2]+radius < decl['target_start_z_mm'] for a in acts[:-1]), rid+' no prep target cap')
        height = acts[-1]['tip_mm'][2]+radius-decl['target_start_z_mm']
        cap_planned = math.pi*height**2*(radius-height/3)
        close(cap_planned, row['exact_continuous_target_cap_mm3'], rid+' exact cap')
        cap = cap_planned if acts[-1]['status'] == 'accepted' else 0.
        close(pe['accepted_target_cap_mm3'], cap, rid+' only committed support earns target cap')
        cert = row['independent_audit']
        if accepted:
            need(cert['status'] == 'completed' and cert['certificate']['feasible'] is True, rid+' independent prefix acceptance')
            c = cert['certificate']
            need(not c['failures'] and c['complete_tool_checked'] and c['frontier_checked']
                 and c['action_count'] == accepted and c['source_case_hash'] == row['source_hash'], rid+' full constraint scope')
            counts['certificates'] += 1; audit_seconds += cert['seconds']
            b = pe['certified_geometry']; sparse = read(run/b['path'], b['sha256'], 2*1024**2)
            need(inputs[str((run/b['path']).relative_to(ROOT))]['bytes'] == b['bytes'], rid+' sparse size')
            need(sparse['source_hash'] == row['source_hash'] and sparse['row_id'] == rid
                 and sparse['history_hash'] == native['committed_history_hash'], rid+' actual sparse history')
            g = sparse['geometry']
            need(g['native_affine'] == row['affine'] and g['source_shape'] == row['shape']
                 and g['cut_union_spec'] == cut and g['evacuated_volume_spec'] == row['public_parent_mass_spec'], rid+' sparse world/spec')
            need(g['geometry_digest'] == semantic({k: v for k, v in g.items() if k != 'geometry_digest'})
                 == pe['checker_geometry_digest'] == c['evacuated_volume']['geometry_digest'], rid+' accepted geometry seal')
            need(g['state_digest'] == evacuation['state_digest'] == c['evacuated_volume']['state_digest'], rid+' terminal union state')
            removed = g['whole_removed_indices']
            partials = {tuple(x['index']): int(x['covered_bits_hex'], 16) for x in g['partial_cells']}
            bounds = {tuple(x[0]): tuple(x[1:]) for x in sparse['parent_mass_bounds']}
            need(len(partials) == len(g['partial_cells']) and len(bounds) == len(sparse['parent_mass_bounds']), rid+' unique rows')
            for lo, hi in bounds.values():
                f = struct.unpack('!f', struct.pack('!f', (lo+hi)/2))[0]
                need((lo, hi) == evaluator.float32_fraction_enclosure(f), rid+' quantization enclosure')
            n = len(removed)*leaves+sum(bits.bit_count() for bits in partials.values())
            need(n == leaf_delta == evacuation['certified_leaf_count'] == c['evacuated_volume']['certified_leaf_count'], rid+' unique union leaf count')
            need(len(removed) == native['removed_cells'] and len(partials) == row['coverage']['partially_certified_occupied_cells'], rid+' full/partial counts')
            stamp = time.perf_counter()
            calc = evaluator.evaluate_certified_union(g['native_affine'], cut['max_depth'], removed, partials, bounds,
                tissue_plane_z_mm=0., target_plane_z_mm=decl['target_start_z_mm'])
            clipping_seconds += time.perf_counter()-stamp
            for key, value in calc.items():
                equivalent(value, pe[key], rid+' recalculated '+key)
            need(calc['physical_target_within_source_interval'] and calc['parent_target_mass_enclosure_pass'], rid+' mass premise and interval')
            close(delta_volume, calc['source_evacuated_mm3'], rid+' deltas to final geometry')
            equivalent(evacuation['evacuated_target_source_bounds_mm3'], calc['cumulative_target_interval_source_mm3'], rid+' cumulative interval')
            close(calc['whole_cell_physical_lower_mm3'], native['removed_physical_mm3'], rid+' actual whole-cell baseline')
            new = calc['physical_tissue_evacuated_lower_mm3']; target = calc['physical_target_evacuated_mm3']
            for key in ('full_parent_boxes', 'partial_parent_boxes', 'leaf_visits', 'face_clips'):
                work['evaluator_'+key] += calc['work'][key]
        else:
            need(cert['status'] == 'not_applicable_rejected_no_committed_history'
                 and pe['status'] == 'empty_no_committed_history' and 'certified_geometry' not in pe, rid+' empty N/A')
            need(pe['source_evacuated_mm3'] == 0, rid+' empty volume'); new = target = 0.
        lower = native['removed_physical_mm3']; upper = row['static_not_executable']['contact_physical_upper_mm3']
        need(lower-tolerance <= new <= upper+tolerance and 0 <= target <= cap+tolerance, rid+' physical bounds')
        for key, value in (('lower_bound_gain_mm3', new-lower), ('gap_before_mm3', upper-lower),
                           ('gap_after_mm3', upper-new), ('target_shortfall_mm3', cap-target)):
            close(pe[key], value, rid+' '+key)
        delta = target-old['partial_evacuation']['physical_target_evacuated_mm3']
        comp = row['baseline_comparison']
        expected_comp = {'physical_target_difference_mm3': delta,
            'physical_tissue_difference_mm3': new-old['partial_evacuation']['physical_tissue_evacuated_lower_mm3'],
            'source_evacuated_difference_mm3': pe['source_evacuated_mm3']-old['partial_evacuation']['source_evacuated_mm3'],
            'whole_cell_physical_target_difference_mm3': native['removed_physical_target_mm3']-old['native']['removed_physical_target_mm3'],
            'whole_cell_reward_difference': native['frozen_reward_total']-old['native']['frozen_reward_total'],
            'baseline_physical_target_mm3': old['partial_evacuation']['physical_target_evacuated_mm3'],
            'current_physical_target_mm3': target}
        for key, value in expected_comp.items():
            close(comp[key], value, rid+' comparison '+key)
        need(comp['baseline_complete'] == old['native']['complete_sequence']
             and comp['current_complete'] == native['complete_sequence']
             and comp['action_result_projection_equal'] == same_projection, rid+' comparison flags')
        numerical_seconds += row['timing_seconds']['numerical_row_total']
        summaries.append({'row_id': rid, 'spacing_mm': row['spacing_mm'], 'complete': native['complete_sequence'],
            'accepted_actions': accepted, 'first_refusal': next(({k: a[k] for k in ('index', 'reason', 'failure_tip_mm')} for a in acts if a['status'] == 'rejected'), None),
            'target_before_mm3': old['partial_evacuation']['physical_target_evacuated_mm3'], 'target_after_mm3': target,
            'target_gain_mm3': delta, 'actual_target_cap_mm3': cap, 'planned_target_cap_mm3': cap_planned,
            'actual_target_gap_mm3': cap-target, 'target_undercredit_fraction': (cap-target)/cap if cap else None,
            'physical_tissue_before_mm3': old['partial_evacuation']['physical_tissue_evacuated_lower_mm3'],
            'physical_tissue_after_mm3': new, 'source_evacuated_mm3': pe['source_evacuated_mm3'],
            'whole_cell_count_before': old['native']['removed_cells'], 'whole_cell_count_after': native['removed_cells'],
            'whole_cell_target_before_mm3': old['native']['removed_physical_target_mm3'],
            'whole_cell_target_after_mm3': native['removed_physical_target_mm3'],
            'reward_before': old['native']['frozen_reward_total'], 'reward_after': native['frozen_reward_total'],
            'action_result_projection_equal': same_projection, 'target_source_interval_mm3': pe.get('cumulative_target_interval_source_mm3', [0., 0.])})
    calls = counts['accepted']+counts['rejected']
    need(calls == r['native_call_costs_including_incomplete_row']['native_strokes_started']
         == r['native_call_costs_including_incomplete_row']['native_strokes_returned'] <= caps['max_native_strokes'], 'all native calls')
    need(calls+counts['unexecuted_prior_refusal'] == 180 and counts['certificates']+counts['empty'] == 18, 'all slots/certificate denominators')
    close(native_seconds, r['native_call_costs_including_incomplete_row']['native_seconds'], 'native time sum')
    for key in ('leaf_tests', 'new_leaf_count', 'witness_rows'):
        need(work[key] == r['returned_preview_coverage_work_including_incomplete_row'][key], 'attempted work '+key)
    iw = r['independent_work_including_incomplete_row']
    need(iw['history_audits_started'] == iw['history_audits_returned'] == iw['plane_evaluations_started']
         == iw['plane_evaluations_returned'] == counts['certificates'] <= caps['max_independent_audits'], 'all checker/clipping calls')
    old_complete = {x['row_id'] for x in prior['rows'] if x['native']['complete_sequence']}
    complete = {x['row_id'] for x in summaries if x['complete']}
    fine = [x for x in summaries if x['spacing_mm'] == .25]
    gains = sorted(x['row_id'] for x in fine if x['target_gain_mm3'] > tolerance)
    no_regression = len(fine) == 6 and all(x['target_gain_mm3'] >= -tolerance for x in fine)
    derived = (len(old_complete) == 13 and not old_complete-complete and no_regression and bool(gains) and work['budget_refusals'] == 0)
    c = r['useful_criterion']
    for key, value in {'meets_predeclared_useful_criterion': derived, 'baseline_completion_count': 13,
        'all_18_rows_present': True, 'all_nonempty_audits_accepted': True, 'all_physical_bounds_respected': True,
        'all_six_fine_target_volumes_nonregressing': no_regression, 'strict_fine_target_gain_rows': gains,
        'additional_baseline_completions': sorted(complete-old_complete), 'lost_baseline_completions': sorted(old_complete-complete),
        'no_preview_budget_refusal': work['budget_refusals'] == 0, 'action_result_equality_required': False}.items():
        need(c[key] == value, 'independently derived criterion '+key)
    need(native_seconds <= numerical_seconds and numerical_seconds+audit_seconds <= r['worker_seconds']+1e-6
         and r['worker_seconds'] < receipt['elapsed_seconds'], 'nested elapsed scopes')
    need(r['peak_worker_rss_bytes'] <= caps['worker_peak_rss_bytes'], 'worker memory cap')
    return {'schema': 'independent-saved-exposure-matrix-v1', 'status': 'PASS', 'checks': checks,
        'executed_canonical_commit': commit,
        'inputs': inputs, 'counts': dict(counts), 'work': dict(work), 'rows': summaries,
        'retained_monitor_failure': {'status': failed_receipt['status'], 'reason': failed_receipt['stop_reason'],
            'parent_seconds': failed_receipt['elapsed_seconds'], 'worker_exit_code': failed_receipt['worker_exit_code'],
            'recorded_native_work': failed_calls, 'clean_reap': True,
            'scope': 'Separate interrupted attempt; costs not silently discarded or included inside fresh run timings.'},
        'independently_meets_primary_criterion': derived, 'retained_original_baseline_reported_criterion': prior['useful_criterion'],
        'reported_current_criterion': c, 'fine_strict_gain_rows': gains,
        'costs': {'native_seconds': native_seconds, 'numerical_row_seconds': numerical_seconds,
            'reported_independent_checker_seconds': iw['history_audit_seconds'], 'reported_plane_seconds': iw['plane_evaluation_seconds'],
            'nested_independent_row_seconds': audit_seconds, 'worker_seconds': r['worker_seconds'],
            'parent_seconds': receipt['elapsed_seconds'], 'group_peak_rss_bytes': receipt['sampled_peak_group_rss_bytes'],
            'output_bytes_before_receipt': receipt['output_bytes_before_receipt'],
            'baseline_unpaired_worker_seconds': prior['worker_seconds'],
            'saved_reclipping_seconds': clipping_seconds, 'this_audit_seconds': time.perf_counter()-began},
        'limitations': ['Generated idealized evacuation/exposure model; no patient, force, clinical or learned-policy validation.',
            'Native history constraints are authenticated saved independent checker certificates; this audit independently reclips their final geometry, without native replay.',
            'The planned contact upper bound can include unexecuted suffixes. The actual target cap is zero unless action10 committed.',
            'Rewards use the same whole-cell rule; changed physical removals/rewards are reported rather than normalized away.',
            'Historical timing is not paired. The failed monitor attempt is separate retained provenance, not a model failure or free compute.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for field in ('run-directory', 'report-sha256', 'launcher-sha256', 'source-index', 'source-index-sha256', 'config', 'canonical-commit'):
        p.add_argument('--'+field, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('30 s saved audit bound')))
    signal.alarm(30)
    try:
        result = audit(args)
    except BaseException as error:
        result = {'status': 'AUDITOR_FAILURE', 'checks': checks, 'error': str(error),
                  'traceback': traceback.format_exc(), 'inputs': inputs}
    result['audit_source_sha256'] = sha(Path(__file__))
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    signal.alarm(0)
    print(json.dumps({'status': result['status'], 'checks': checks}))
    if result['status'] != 'PASS':
        raise SystemExit(1)
