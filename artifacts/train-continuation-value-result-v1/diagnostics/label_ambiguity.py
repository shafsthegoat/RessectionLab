"""Terminal-only weighted target ambiguity at exact public observation/actions.

Stdlib only. No model, native replay, feature or checkpoint decoding. This is an
optimistic deterministic-predictor MSE lower bound on the same fit corpus; it
cannot diagnose feature collisions or optimizer behavior by itself.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import time

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
SUBJECTS = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
READS = {}


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def semantic(value):
    return 'sha256:' + hashlib.sha256(canonical(value)).hexdigest()


def read(path, expected):
    need(path.is_file() and not path.is_symlink() and path.suffix == '.json', 'Pinned regular JSON only')
    need(0 < path.stat().st_size <= 16 * 1024**2, 'Bounded saved scalar JSON')
    data = path.read_bytes(); actual = hashlib.sha256(data).hexdigest()
    need(actual == expected, 'Exact terminal input hash differs: ' + str(path))
    READS[str(path.relative_to(STAGE))] = {'sha256': actual, 'bytes': len(data)}
    return json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def finite(value):
    need(type(value) in (float, int) and math.isfinite(value), 'Finite numeric label/weight')
    return value


def mandatory_prefix_oracle(rewards):
    # Deliberately direct enumeration, not the producer's reverse recurrence.
    # The first action is mandatory: do not clip a negative answer to STOP0.
    return [max(math.fsum(rewards[i:j+1]) for j in range(i, len(rewards)))
            for i in range(len(rewards))]


def check_rounding(saved, exact):
    nearest = f32(exact)
    bits = struct.unpack('<I', struct.pack('<f', abs(nearest)))[0]
    ulp = struct.unpack('<f', struct.pack('<I', bits+1))[0]-abs(nearest)
    need(saved == f32(saved) and abs(saved-nearest) <= ulp+1e-12,
         'Saved mandatory-action target differs beyond float32 summation rounding')


def group_stats(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['key']].append(row)
    details = []; kinds = Counter(); weight = math.fsum(r['weight'] for r in rows)
    for key, members in groups.items():
        mass = math.fsum(r['weight'] for r in members)
        mean = math.fsum(r['weight']*r['target'] for r in members)/mass
        contribution = math.fsum(r['weight']*(r['target']-mean)**2 for r in members)
        low, high = min(r['target'] for r in members), max(r['target'] for r in members)
        by_kind = {}
        for kind in ('teacher', 'RL'):
            subset = [r for r in members if r['kind'] == kind]
            if subset:
                submass = math.fsum(r['weight'] for r in subset)
                submean = math.fsum(r['weight']*r['target'] for r in subset)/submass
                by_kind[kind] = {'rows': len(subset), 'weight': submass, 'mean': submean,
                    'min': min(r['target'] for r in subset), 'max': max(r['target'] for r in subset),
                    'within_kind_contribution': math.fsum(r['weight']*(r['target']-submean)**2 for r in subset)}
        within_kind = math.fsum(v['within_kind_contribution'] for v in by_kind.values())
        between_kind = math.fsum(v['weight']*(v['mean']-mean)**2 for v in by_kind.values())
        need(abs(contribution-within_kind-between_kind) <= 1e-9 + 1e-12*contribution,
             'Weighted within/between-kind variance decomposition')
        category = 'mixed_teacher_RL' if len(by_kind) == 2 else next(iter(by_kind)) + '_only'
        kinds[category] += 1
        details.append({'key': key, 'subject': members[0]['subject'], 'rows': len(members),
            'weight': mass, 'mean': mean, 'min': low, 'max': high, 'range': high-low,
            'contribution': contribution, 'within_kind_contribution': within_kind,
            'between_kind_contribution': between_kind, 'kind': category, 'by_kind': by_kind})
    lower = math.fsum(g['contribution'] for g in details)
    duplicates = [g for g in details if g['rows'] > 1]
    conflicts = [g for g in details if g['range'] > 0.]
    return {'rows': len(rows), 'sum_weights': weight, 'distinct_groups': len(groups),
        'duplicate_groups': len(duplicates), 'rows_in_duplicate_groups': sum(g['rows'] for g in duplicates),
        'conflicting_groups': len(conflicts), 'rows_in_conflicting_groups': sum(g['rows'] for g in conflicts),
        'target_min': min((r['target'] for r in rows), default=None),
        'target_max': max((r['target'] for r in rows), default=None),
        'weighted_within_group_MSE_lower_bound': lower,
        'normalized_within_group_MSE_lower_bound': None if weight == 0 else lower/weight,
        'within_teacher_or_RL_contribution': math.fsum(g['within_kind_contribution'] for g in details),
        'between_teacher_RL_mean_contribution': math.fsum(g['between_kind_contribution'] for g in details),
        'group_kind_counts': dict(kinds)}, details


def comparison(lower_bound, fitted):
    if fitted is None:
        return {'available': False}
    finite(fitted)
    return {'available': True, 'reported_fit_MSE': fitted, 'optimistic_MSE_lower_bound': lower_bound,
        'fit_MSE_minus_bound': fitted-lower_bound,
        'fraction_of_fit_MSE_accounted_for_by_exact_group_variance': None if fitted == 0 else lower_bound/fitted,
        'roundoff_consistent_with_bound': fitted + 1e-6 + abs(fitted)*1e-5 >= lower_bound,
        'residual_interpretation': 'Remaining error may include feature collisions, head approximation, finite optimization or other factors; this projection does not separate them.'}


def run(args):
    started = time.perf_counter()
    launcher = read(STAGE/'run-01/launcher.json', args.launcher_sha256)
    need(launcher.get('status') != 'running' and launcher.get('worker_exit_code') is not None,
         'Actual terminal parent receipt required; never inspect a live run')
    report = read(STAGE/'run-01/report.json', args.report_sha256)
    index = read(STAGE/'runner/source-index.json', args.index_sha256)
    need(index['execution_released'] is True and
         launcher['source_index_sha256'] == report['source_index_sha256'] == args.index_sha256 and
         launcher['report_sha256'] == args.report_sha256, 'Exact terminal source/report binding')
    manifest = read(STAGE/'run-01/trace-manifest.json', args.manifest_sha256)
    labels = read(STAGE/'run-01/logged-targets.json', args.targets_sha256)
    need(semantic(manifest) == report['trace_manifest_hash'] == labels['manifest_hash'] ==
         report['corpus']['manifest_hash'], 'Exact original declared corpus manifest')
    need(tuple(report['config']['TRAIN']) == SUBJECTS and len(manifest) == 36 and
         report['corpus']['states'] == 497 and len(report['corpus']['trace_bindings']) == 36,
         'Fixed36/497 original four TRAIN denominator')
    need(labels['target_formula'] == report['config']['target'] and
         len(labels['targets']) == len(labels['weights']) == len(labels['STOP_rows']) == 497,
         'Exact target/weight/STOP row denominator')
    rows = []; counts = Counter(); position = 0; trace_keys = set(); teacher_states = 0
    negative_then_positive = 0
    for trace in manifest:
        world = trace['world']; subject = world['subject']; kind = trace['kind']; decisions = trace['decisions']
        need(subject in SUBJECTS and kind in ('teacher', 'RL') and world['max_steps'] == 24 and
             trace['key'] not in trace_keys, 'Unique declared TRAIN trace/world')
        trace_keys.add(trace['key']); counts[(subject, kind)] += 1
        need(1 <= len(decisions) <= 24 and all(d['terminated'] is (i == len(decisions)-1)
            for i, d in enumerate(decisions)), 'Completed STOP/horizon trajectory')
        need(all(d['action_id'] != 'STOP' for d in decisions[:-1]) and
             (decisions[-1]['action_id'] == 'STOP' or len(decisions) == 24), 'No future beyond actual terminal')
        rewards = [finite(d['reward']) for d in decisions]
        if decisions[-1]['action_id'] == 'STOP': need(rewards[-1] == 0., 'Unchanged STOP0')
        targets = mandatory_prefix_oracle(rewards)
        for i, (decision, target) in enumerate(zip(decisions, targets)):
            saved = finite(labels['targets'][position]); weight = finite(labels['weights'][position])
            check_rounding(saved, target)
            need(weight == f32(1./(4*9*len(decisions))) and weight > 0., 'Exact saved objective mass')
            stop = decision['action_id'] == 'STOP'
            need(labels['STOP_rows'][position] is stop and (not stop or saved == 0.), 'Separate fixed STOP0 row')
            key = (world['source_hash'], world['decision_model_hash'], decision['observation_hash'], decision['action_id'])
            rows.append({'key': key, 'subject': subject, 'kind': kind, 'target': saved,
                         'weight': weight, 'STOP': stop})
            negative_then_positive += int(rewards[i] < 0. and target > 0.)
            position += 1
        if kind == 'teacher': teacher_states += len(decisions)
    need(position == 497 and teacher_states == 29 and all(counts[(s, 'teacher')] == 1 and
         counts[(s, 'RL')] == 8 for s in SUBJECTS), 'No episode/state/source omissions')
    global_stats, groups = group_stats(rows)
    need(abs(global_stats['sum_weights']-1.) < 1e-6, 'Float32 objective mass sums to one')
    stop_stats, _ = group_stats([r for r in rows if r['STOP']])
    motion_stats, _ = group_stats([r for r in rows if not r['STOP']])
    need(stop_stats['weighted_within_group_MSE_lower_bound'] == 0., 'STOP ambiguity must be exactly zero')
    per_patient = {}
    for subject in SUBJECTS:
        subset = [r for r in rows if r['subject'] == subject]
        stats, _ = group_stats(subset)
        per_patient[subject] = {'all': stats,
            'by_behavior_kind': {k: group_stats([r for r in subset if r['kind'] == k])[0] for k in ('teacher', 'RL')},
            'fit_comparison': comparison(stats['normalized_within_group_MSE_lower_bound'],
                report.get('final_case_weighted_MSE', {}).get(subject))}
    conflicts = sorted((g for g in groups if g['range'] > 0.),
                       key=lambda g: (-g['contribution'], g['key']))
    top = []
    for g in conflicts[:16]:
        key = g['key']
        top.append({k: v for k, v in g.items() if k != 'key'} | {
            'source_hash': key[0], 'decision_model_hash': key[1], 'observation_hash': key[2], 'action_id': key[3]})
    fit = report.get('head_fit', {})
    result = {'schema': 'exact-public-state-action-label-ambiguity-v1',
        'terminal_parent_status': launcher['status'], 'report_status': report['status'],
        'group_key': ['world.source_hash', 'world.decision_model_hash', 'observation_hash', 'action_id'],
        'target': 'mandatory logged first action, then best recorded prefix; no nonSTOP clipping to zero',
        'arithmetic': 'Saved float32 targets/weights; float64 weighted means and squared residuals; independent direct-prefix oracle.',
        'all_rows': global_stats, 'STOP_rows': stop_stats, 'nonSTOP_rows': motion_stats,
        'by_behavior_kind': {k: group_stats([r for r in rows if r['kind'] == k])[0] for k in ('teacher', 'RL')},
        'per_patient': per_patient,
        'fit_comparison': comparison(global_stats['weighted_within_group_MSE_lower_bound'], fit.get('final_weighted_mse')),
        'negative_immediate_positive_recorded_continuation_examples': negative_then_positive,
        'top_conflicting_groups': top, 'top_groups_limit': 16,
        'all_conflicting_groups_in_aggregates': len(conflicts),
        'interpretation': [
            'Exact observation/action groups must receive the same output from any deterministic public-input predictor. Their weighted label variance is an optimistic irreducible squared-error lower bound on this fit corpus.',
            'Mixed teacher/RL mean differences quantify label disagreement; they do not by themselves prove that disagreement caused a deployed route regression.',
            'Distinct observations can collide after frozen feature extraction; this script never reads features and therefore cannot measure those additional collisions.',
            'Error above this bound does not prove insufficient optimization steps. Head capacity, lossy representation, optimization and other factors remain unresolved.',
            'STOP0 is a fixed action, not a fitted stopping signal. No costly-preparation or optimal-Q claim follows from recorded continuations.',
            'All aggregates retain36 trajectories and497 rows; top conflicts are a compact display, not a selection of the training denominator.'],
        'input_files': READS, 'diagnostic_seconds': time.perf_counter()-started}
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('report', 'launcher', 'index', 'manifest', 'targets'):
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output', default='terminal-label-ambiguity-01.json')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if key.endswith('_sha256'):
            need(len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Exact final SHA256 required')
    output = HERE/args.output
    need(output.parent.resolve() == HERE.resolve() and output.suffix == '.json' and not output.exists(),
         'Fresh bounded diagnostic output required')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('30-second saved-only cap')))
    signal.alarm(30)
    result = run(args); data = canonical(result) + b'\n'
    need(len(data) < 32*1024, 'Strict32KiB output cap')
    for name, pin in READS.items():
        need(hashlib.sha256((STAGE/name).read_bytes()).hexdigest() == pin['sha256'], 'Saved input changed while read')
    with output.open('xb') as stream: stream.write(data)
    signal.alarm(0)
    print(json.dumps({'output': str(output), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
        'conflicting_groups': result['all_rows']['conflicting_groups'],
        'fit_comparison': result['fit_comparison']}))


if __name__ == '__main__': main()
