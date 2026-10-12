"""Offline scalar-only plots; safe to rerun while atomic curriculum reports arrive.

No policy, checkpoint, source anatomy, array archive or native runtime is loaded.
Hashes identify the exact saved bytes read, not a new independent scientific audit.
"""
from pathlib import Path
import datetime
import gzip
import hashlib
import io
import json
import math
import os

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
RUN = STAGE / 'curriculum-1024-01'
os.environ['MPLCONFIGDIR'] = str(HERE / 'mpl-cache')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SOURCES = []
WARNINGS = []
LIMIT = 32 * 1024**2
COLORS = {'uniform': '#187a8d', 'reverse': '#b45a35'}


def read(path, optional=False):
    """One bounded byte snapshot; live partial writes are reported, never invented."""
    try:
        with path.open('rb') as stream:
            raw = stream.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError('Scalar input exceeds32MiB: ' + str(path))
        decoded = raw
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                decoded = stream.read(LIMIT + 1)
            if len(decoded) > LIMIT:
                raise ValueError('Decoded scalar input exceeds32MiB')
        value = json.loads(decoded, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (FileNotFoundError, json.JSONDecodeError, EOFError, gzip.BadGzipFile) as error:
        if not optional:
            raise
        if not isinstance(error, FileNotFoundError):
            WARNINGS.append({'path': str(path.relative_to(STAGE)), 'read_status': type(error).__name__})
        return None
    SOURCES.append({'path': str(path.relative_to(STAGE)), 'bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest()})
    return value


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Finite saved scalar required')
    return value


def evaluation(value, decisions):
    if not (value['complete'] and value['audit']['accepted'] and value['local_reward_audit_accepted']):
        raise ValueError('A full-start curve point requires accepted replay and local reward audit')
    return {'decisions': decisions, 'return': finite(value['return']),
            'success': value['success'], 'decision_count': value['decision_count'],
            'immediate_STOP': (not value['success'] and value['decision_count'] == 1
                               and value.get('rewards') == [0.0]),
            'history_hash': value['history_hash'], 'plan_seal': value['plan_seal'],
            'scope': 'Deterministic start12 evaluation; independently replayed geometry and task reward'}


def optimizer_curve(checkpoint):
    rows = checkpoint['rollouts']
    after = [r['optimizer_completed_before_update'] for r in rows[1:]]
    after.append(checkpoint['optimizer_completed'])
    points = [{'decisions': 0, 'optimizer_completed': 0}]
    points += [{'decisions': r['decisions'], 'optimizer_completed': count}
               for r, count in zip(rows, after)]
    if any(b['decisions'] <= a['decisions'] or b['optimizer_completed'] < a['optimizer_completed']
           for a, b in zip(points, points[1:])):
        raise ValueError('Nonmonotone saved optimizer accounting')
    return points


def episodes(directory, offset):
    rows = []
    consumed = offset
    for path in sorted(directory.glob('episode-*.json.gz')):
        e = read(path, optional=True)
        if e is None:
            continue
        if e['complete'] is not True:
            raise ValueError('Completed-episode file must be complete')
        count = e['costs']['policy_native_transitions']
        if type(count) is not int or count <= 0:
            raise ValueError('Actual policy decision count required')
        if e['episode'] != len(rows) + 1:
            WARNINGS.append({'path': str(directory.relative_to(STAGE)), 'read_status': 'episode_gap; later points omitted'})
            break
        consumed += count
        rows.append({'episode': e['episode'], 'end_training_decision': consumed, 'decisions': count,
                     'start': e['start_absolute_step'], 'return': finite(e['return']),
                     'success': e['success'], 'catalog_seconds': e['costs']['catalog_seconds'],
                     'native_history_hash': e['native_history_hash']})
    return rows


def phase_seconds(report, tag):
    return report.get('costs', {}).get(tag, {}).get('complete_wall_seconds')


def main():
    pilot = read(STAGE / 'pilot-01/report.json')
    parent = read(STAGE / 'campaign-256-01/report.json')
    search = read(STAGE / 'search-01/report.json')
    for report in (pilot, parent, search):
        if report['status'] != 'completed' or not report['source_unchanged']:
            raise ValueError('Completed retained reports required')
    if not search['independent_audit']['accepted'] or not search['local_reward_audit']['accepted']:
        raise ValueError('Search reference lacks accepted independent audits')
    report = read(RUN / 'report.json', optional=True) or {}
    progress = read(RUN / 'progress.json', optional=True) or {}
    launcher = read(RUN / 'launcher.json', optional=True)
    prior = parent['arms']['frozen_encoder-seed0']
    pilot_arm = pilot['arms']['frozen_encoder-seed0']
    shared = [evaluation(pilot_arm['initial_evaluation'], 0),
              evaluation(pilot_arm['checkpoints'][0]['evaluation'], 64),
              evaluation(prior['checkpoints'][0]['evaluation'], 256)]
    record = {'schema': 'offline-reset-curriculum-learning-curves-v1',
              'rendered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'One exposed TRAIN045 simulated goal; seed0; no held-out/generalization/clinical inference',
              'run_status': report.get('status', 'live_or_no_report_yet'),
              'launcher_status': None if launcher is None else launcher.get('status'),
              'shared_parent_evaluations': shared,
              'shared_parent_checkpoint': prior['checkpoints'][0]['checkpoint'],
              'shared_parent_nominal_inference_seconds': phase_seconds(parent, 'frozen_encoder-seed0-eval-256.nominal'),
              'shared_parent_experience': {k: v for k, v in prior['experience'].items()
                  if k not in ('source_state_hashes', 'state_physical_action_hashes', 'episode_returns')},
              'search': {'return': search['evaluation']['return'], 'success': search['evaluation']['success'],
                  'online_seconds': search['search']['seconds'], 'worker_seconds': search['worker_seconds'],
                  'online_native_previews': search['search']['native_usage_delta'],
                  'total_native_previews': search['native_budget']['native_preview_entries'],
                  'scope': 'Paid bounded search plus fresh audit; no training decisions'},
              'arms': {}, 'limitations': [
                  'Training successes are sampled trajectories, not independently replayed full-start evaluations.',
                  'Pending endpoints are missing observations, never zero-return failures.',
                  'Only one seed and one task; no confidence intervals or generalization claim.',
                  'Cumulative branch wall includes restoration, training, checkpointing and evaluation; pure training wall is not separately isolated.',
                  'Live progress has no arm identity; its optimizer count is reported separately, never assigned to an arm by inference.',
                  'Input snapshots are sequential reads, not an atomic worker snapshot; file hashes identify exact bytes.']}
    for method in ('uniform', 'reverse'):
        name = method + '-seed0'
        curve = read(RUN / name / 'curve.json', optional=True) or {}
        saved = report.get('arms', {}).get(name, {})
        candidates = [v for v in (curve, saved) if v]
        # Most advanced saved record wins; no live model or optimizer is accessed.
        arm = max(candidates, key=lambda a: (a.get('segment_experience', {}).get('decisions', 0),
                  len(a.get('checkpoints', [])), a.get('status') == 'completed'), default={})
        eps = episodes(RUN / name, 256)
        checkpoints = arm.get('checkpoints', [])
        endpoints = {q['decisions']: evaluation(q['evaluation'], q['decisions']) for q in checkpoints}
        # The report may contain an accepted endpoint before curve.json is committed.
        for total in (512, 1024):
            e = report.get('evaluations', {}).get(name + '-eval-' + str(total))
            if e is not None:
                endpoints[total] = evaluation(e, total)
        segment = arm.get('segment_experience', {})
        completed_decisions = sum(e['decisions'] for e in eps)
        observed = max(completed_decisions, segment.get('decisions', 0))
        starts = {str(s): {'episodes': sum(e['start'] == s for e in eps),
                  'successes': sum(e['start'] == s and e['success'] for e in eps),
                  'decisions_in_completed_episodes': sum(e['decisions'] for e in eps if e['start'] == s)}
                  for s in range(12, 17)}
        latest = checkpoints[-1] if checkpoints else None
        work = arm.get('cumulative_branch_seconds')
        if work is None and latest is not None:
            work = prior['cumulative_arm_seconds'] + latest['segment_seconds']
        cost_rows = []
        for total, endpoint in sorted(endpoints.items()):
            checkpoint = next((q for q in checkpoints if q['decisions'] == total), {})
            saved_eval = checkpoint.get('evaluation', report.get('evaluations', {}).get(name + '-eval-' + str(total), {}))
            cost_rows.append({'decisions': total,
                'nominal_inference_seconds': phase_seconds(report, name + '-eval-' + str(total) + '.nominal'),
                'fresh_replay_and_audit_seconds': phase_seconds(report, name + '-eval-' + str(total) + '.fresh_replay_and_independent_geometry'),
                'evaluation_plus_replay_audit_and_isolation_seconds': checkpoint.get('evaluation_isolation', {}).get('paid_seconds'),
                'catalog_construction_only_seconds': saved_eval.get('catalog_costs', {}).get('catalog_seconds'),
                'scope': 'Combined evaluation/isolation and catalog-only work are not pure inference latency.'})
        record['arms'][method] = {'status': arm.get('status', 'in_progress' if eps else 'pending'),
            'new_decisions_observed_lower_bound': observed,
            'branch_decisions_observed_lower_bound': 256 + observed,
            'full_start_evaluations': [endpoints[k] for k in sorted(endpoints)],
            'pending_endpoints': [s for s in (512, 1024) if s not in endpoints],
            'optimizer_curve': optimizer_curve(latest or prior['checkpoints'][0]),
            'actual_optimizer_completed_at_last_checkpoint': (latest or prior['checkpoints'][0])['optimizer_completed'],
            'saved_checkpoint_boundaries': [{'decisions': q['decisions'], 'checkpoint': q['checkpoint'],
                'optimizer_completed': q['optimizer_completed'],
                'additional_optimizer_completed': q['additional_optimizer_completed'],
                'segment_seconds_including_evaluation': q['segment_seconds'],
                'experience': {k: v for k, v in q['experience'].items()
                    if k not in ('source_state_hashes', 'state_physical_action_hashes', 'episode_returns')},
                'segment_experience': {k: v for k, v in q['segment_experience'].items()
                    if k not in ('source_state_hashes', 'state_physical_action_hashes', 'episode_returns')}} for q in checkpoints],
            'completed_training_episodes': eps, 'training_by_reset_start': starts,
            'segment_experience': {k: v for k, v in segment.items()
                if k not in ('source_state_hashes', 'state_physical_action_hashes', 'episode_returns')},
            'unfinished_decisions_at_saved_summary': segment.get('unfinished_episode_decisions'),
            'inherited_retired_segments': arm.get('experience', {}).get('retired_segments'),
            'cumulative_branch_wall_seconds_at_last_saved_boundary': work,
            'evaluation_costs': cost_rows,
            'completed_training_catalog_seconds_lower_bound': sum(e['catalog_seconds'] for e in eps)}
    new_count = sum(a['new_decisions_observed_lower_bound'] for a in record['arms'].values())
    record['experience_totals'] = {'patients': 1, 'shared_parent_decisions_counted_once': 256,
        'new_decisions_observed_lower_bound': new_count,
        'comparison_lineage_decisions_lower_bound': 256 + new_count,
        'all_paid_training_including_retained_trainable256_lower_bound': 512 + new_count,
        'exact_terminal_counts': report.get('comparison_experience', {}) and
            {k: v for k, v in report['comparison_experience'].items()
             if k not in ('source_state_hashes', 'state_physical_action_hashes')}}
    record['paid_worker_costs'] = {'retained_pilot_plus_256': parent['cost_segments'],
        'current_comparison_terminal': report.get('cost_segments'),
        'live_progress_unassigned_to_arm': {k: progress[k] for k in
            ('phase', 'seconds', 'decisions', 'optimizer_attempted', 'optimizer_completed', 'transitions') if k in progress},
        'live_preview_entries': progress.get('native_budget', {}).get('native_preview_entries'),
        'scope': 'Shared prior costs counted once; nested phase times are not summed.'}
    record['source_reports_and_episodes'] = SOURCES
    record['warnings'] = WARNINGS
    record['renderer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    render(record, prior)
    (HERE / 'learning-curves.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    output_bytes = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    if output_bytes > 5 * 1024**2:
        raise ValueError('Plot directory exceeds assigned5MiB')
    print(json.dumps({'run_status': record['run_status'], 'output_bytes': output_bytes,
        'endpoints': {k: [x['decisions'] for x in a['full_start_evaluations']] for k, a in record['arms'].items()},
        'new_decisions_observed_lower_bound': new_count, 'warnings': WARNINGS}))


def render(r, prior):
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    a, b, c, d, e, f = axes.flat
    shared = r['shared_parent_evaluations']
    a.plot([v['decisions'] for v in shared], [v['return'] for v in shared], 'o-',
           color='#555555', label='Shared frozen parent')
    a.axhline(r['search']['return'], color='#3a3a3a', ls='--', label='Matched search (audited)')
    parent_opt = optimizer_curve(prior['checkpoints'][0])
    b.step([v['decisions'] for v in parent_opt], [v['optimizer_completed'] for v in parent_opt],
           where='post', color='#555555', label='Shared parent')
    start_colors = {s: plt.get_cmap('viridis')((s - 12) / 4) for s in range(12, 17)}
    for i, (method, arm) in enumerate(r['arms'].items()):
        color = COLORS[method]
        points = [shared[-1], *arm['full_start_evaluations']]
        a.plot([p['decisions'] for p in points], [p['return'] for p in points], 'o-',
               color=color, label=method.capitalize())
        for p in arm['full_start_evaluations']:
            a.scatter([p['decisions']], [p['return']], marker='*' if p['success'] else 'x',
                      s=65, color=color if p['success'] else '#b22222', zorder=5)
            a.annotate('success' if p['success'] else ('FAIL: immediate STOP' if p['immediate_STOP'] else 'FAIL: no goal'), (p['decisions'], p['return']),
                       xytext=(-4 if p['decisions'] >= 1000 else 4, 9 + i * 11),
                       ha='right' if p['decisions'] >= 1000 else 'left',
                       textcoords='offset points', fontsize=8, color=color)
        if arm['pending_endpoints']:
            a.text(.02, .84 - i * .065, method.capitalize() + ' pending: ' +
                   ', '.join(map(str, arm['pending_endpoints'])), transform=a.transAxes, va='top', color=color)
        opt = [p for p in arm['optimizer_curve'] if p['decisions'] >= 256]
        b.step([p['decisions'] for p in opt], [p['optimizer_completed'] for p in opt],
               where='post', marker='o', color=color, label=method.capitalize())
        ax = (c, d)[i]
        eps = arm['completed_training_episodes']
        for start in range(12, 17):
            for success, marker in ((False, 'o'), (True, '^')):
                rows = [p for p in eps if p['start'] == start and p['success'] == success]
                ax.scatter([p['end_training_decision'] for p in rows], [p['return'] for p in rows],
                           color=start_colors[start], marker=marker, s=26 if success else 17, alpha=.85)
        ax.axhline(0, color='#bbbbbb', lw=.7)
        ax.set(title=method.capitalize() + ': sampled completed training episodes',
               ylabel='Episode return (no independent replay claim)', xlabel='Branch training decisions',
               xlim=(248, 1034), ylim=(-1.1, 11.5))
        if not eps:
            ax.text(.5, .45, 'No completed episodes saved yet', ha='center', transform=ax.transAxes)
        ax.text(.02, .98, 'successes / completed, by reset start: ' + ' · '.join(
            str(s) + ':' + str(arm['training_by_reset_start'][str(s)]['successes']) + '/' +
            str(arm['training_by_reset_start'][str(s)]['episodes']) for s in range(12, 17)),
            va='top', fontsize=8, transform=ax.transAxes)
        if method == 'reverse':
            for boundary in (384, 512, 640, 768):
                ax.axvline(boundary, color='#cccccc', ls=':', lw=.6)
        ax.legend(handles=[Line2D([], [], marker='o', ls='', color=start_colors[s], label='start ' + str(s))
                           for s in range(12, 17)], loc='center right', fontsize=7)
    a.set(title='A  Full-start deterministic endpoints (start 12)', xlabel='Branch training decisions',
          ylabel='Independently replayed task return', xlim=(-15, 1050), ylim=(-1.1, 11.5))
    b.set(title='B  Actual Adam steps at saved checkpoints', xlabel='Branch training decisions',
          ylabel='Cumulative optimizer steps (not epochs)', xlim=(-15, 1050))
    for ax in (a, b):
        ax.legend(fontsize=8, loc='center right')
    nominal = []
    cumulative = []
    names = ['Shared 256'] + [method.capitalize() + '\n' +
        (str(r['arms'][method]['saved_checkpoint_boundaries'][-1]['decisions'])
         if r['arms'][method]['saved_checkpoint_boundaries'] else '(pending)')
        for method in ('uniform', 'reverse')]
    nominal.append(r['shared_parent_nominal_inference_seconds'])
    cumulative.append(prior['cumulative_arm_seconds'])
    for method in ('uniform', 'reverse'):
        arm = r['arms'][method]
        nominal.append(arm['evaluation_costs'][-1]['nominal_inference_seconds'] if arm['evaluation_costs'] else None)
        cumulative.append(arm['cumulative_branch_wall_seconds_at_last_saved_boundary'])
    for i, (inf, work) in enumerate(zip(nominal, cumulative)):
        if inf is not None:
            e.bar(i - .18, inf, .32, color='#6da7c4', label='Latest nominal inference' if i == 0 else None)
        else:
            e.text(i - .18, .025, 'n/a', ha='center', transform=e.get_xaxis_transform(), color='#6da7c4')
        if work is not None:
            e.bar(i + .18, work, .32, color='#b89864')
        else:
            e.text(i + .18, .08, 'pending', ha='center', transform=e.get_xaxis_transform(), color='#777777')
    e.axhline(r['search']['online_seconds'], ls='--', color='#333333')
    e.set(yscale='log', xticks=range(3), xticklabels=names, xlim=(-.5, 2.6), ylabel='Measured seconds (log scale)',
          title='E  Inference versus cumulative branch work')
    e.legend(handles=[Line2D([], [], lw=6, color='#6da7c4', label='Latest nominal inference (when saved)'),
        Line2D([], [], lw=6, color='#b89864', label='Training + restore + saved evaluations'),
        Line2D([], [], ls='--', color='#333333', label='Search online: %.1f s' % r['search']['online_seconds'])], fontsize=8)
    f.axis('off')
    counts = r['experience_totals']
    status = 'Terminal' if r['run_status'] == 'completed' and r['launcher_status'] == 'completed' else 'Interim snapshot'
    text = [status + ' — ' + r['rendered_utc'][:19] + ' UTC',
        'Shared frozen 256-decision parent counted once; one patient, seed 0.',
        'Annotation-assisted geometric simulation; no clinical validation.',
        'New branch decisions observed: ≥' + str(counts['new_decisions_observed_lower_bound']),
        'Comparison lineage decisions: ≥' + str(counts['comparison_lineage_decisions_lower_bound']),
        'All paid PPO including prior trainable256: ≥' + str(counts['all_paid_training_including_retained_trainable256_lower_bound']),
        '', 'Search: return %.4f; online %.1fs; whole worker %.1fs' %
        (r['search']['return'], r['search']['online_seconds'], r['search']['worker_seconds']),
        'Retired and unfinished episodes are not counted as failures.',
        'Triangles: sampled successes; circles: other completed episodes.',
        'Pending curves are not failures. No smoothing or confidence bands.',
        'n/a: wall-time phase not closed/saved; never treated as zero.']
    for method, arm in r['arms'].items():
        if arm['full_start_evaluations']:
            latest = arm['full_start_evaluations'][-1]
            text += [method.capitalize() + ' ' + str(latest['decisions']) + ': ' +
                     ('SUCCESS' if latest['success'] else 'FAILURE') +
                     (' — immediate STOP; return0 is not success' if latest['immediate_STOP'] else
                      '; return %.4f' % latest['return'])]
    p = r['paid_worker_costs']['live_progress_unassigned_to_arm']
    if p:
        text += ['', 'Latest worker progress (arm name not recorded):',
                 'decisions=%s; actual Adam=%s; worker %.1fs' %
                 (p.get('decisions', 'n/a'), p.get('optimizer_completed', 'n/a'), p.get('seconds', 0))]
    f.text(0, 1, '\n'.join(text), va='top', fontsize=9, linespacing=1.65)
    for ax in (a, b, c, d, e):
        ax.grid(alpha=.16)
    fig.suptitle('TRAIN045 local-goal PPO: uniform vs reverse resets\n'
                 'Same frozen 256-decision weights + Adam; fixed task and reward; 768 new decisions per branch', fontsize=14)
    fig.savefig(HERE / 'learning-curves.png', dpi=135)
    fig.savefig(HERE / 'learning-curves.svg')
    svg = HERE / 'learning-curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
