"""Summarize checked saved JSON and draw a fixed six-TRAIN-case comparison.

No project imports, patient arrays, policy loading, simulation, or refitting.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'outputs/prepared-training-planner-comparison-v2'
METHODS = ('STOP', 'frozen_il', 'greedy_search')
SUBJECTS = tuple('sub-PAT' + n for n in ('05', '16', '20', '22', '25', '28'))
AUDIT_SHA = 'b98dd4141082c9a88b2caf0ecc8a224b318f0cca58568a4a48657eb60401710f'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [RAW / 'summary.json', RAW / 'declaration-input.json',
             HERE / 'independent-check.json', HERE / 'root-terminal.json']
    paths += [RAW / s / 'preparation.json' for s in SUBJECTS if s not in SUBJECTS[1:3]]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    summary = json.loads(paths[0].read_text())
    declaration = json.loads(paths[1].read_text())
    audit = json.loads(paths[2].read_text())
    terminal = json.loads(paths[3].read_text())
    assert sha(paths[2]) == AUDIT_SHA
    assert audit['attempts'][0]['status'] == 'saved_records_verified'
    assert terminal['exit_code'] == 0
    assert tuple(p['subject'] for p in summary['patients']) == SUBJECTS
    assert summary['patients_prescribed'] == 6 and summary['complete_comparisons'] == 4
    assert declaration['settings']['arm_seconds'] == 90.0
    assert declaration['settings']['arm_native_preview_entries'] == 468
    rows = []
    for source in summary['patients']:
        subject = source['subject']
        assert source['role'] == 'TRAIN'
        row = {'subject': subject, 'role': 'TRAIN', 'status': source['status'],
               'analysis_role': 'trained_case' if subject == SUBJECTS[0] else
                   ('historical_support_block' if subject in SUBJECTS[1:3] else 'development_transfer'),
               'arms': {}, 'shared_preparation': None, 'worker_elapsed_seconds': source.get('elapsed_seconds'),
               'worker_recorded_peak_rss_bytes': source.get('peak_rss_bytes'),
               'supervisor': source.get('supervisor'), 'policy_load_seconds': source.get('policy_load_seconds')}
        if source['status'] == 'historical_support_block':
            row['support_conflict'] = source['original_preparation']['coverage']
            row['failure_code'] = source['original_preparation']['failure_code']
            for method in METHODS:
                assert source['arms'][method]['outcomes'] is None
                row['arms'][method] = {'status': 'not_executed', 'outcomes': None,
                                       'online_seconds': None, 'native_previews': None}
        else:
            assert source['status'] == 'complete'
            preparation = json.loads((RAW / subject / 'preparation.json').read_text())
            access = preparation['access_preparation']
            row['shared_preparation'] = {
                'total_seconds': preparation['shared_preparation_seconds'],
                'original_loader_seconds': preparation['original_loader_seconds'],
                'access_preparation_seconds': access['elapsed_seconds'],
                'candidate_setup_seconds': access['candidate_setup_seconds'],
                'static_screening_seconds': access['screening_seconds'],
                'selected_initial_inventory_seconds': preparation['selected_initial_inventory_seconds'],
                'selected_exit': access['selected_exit'],
                'native_previews': sum(p['started'] for p in preparation['shared_native_preview_profile']['phases'].values()),
                'actor_nominal_target_fraction_visible': preparation['actor_coverage']['nominal_target_mass_fraction_visible'],
                'scope': 'Shared once across arms; outside their online guards. Subintervals are nested.'}
            for method in METHODS:
                arm = source['arms'][method]
                budget = arm['planning_budget']
                assert arm['status'] == 'complete' and arm['independent_evaluation_accepted']
                assert arm['initial_binding'] == source['initial_binding'] == arm['final_initial_binding']
                assert arm['initial_parameter_hash'] == arm['final_parameter_hash']
                assert arm['outcomes']['clinical_deficit_probability'] is None
                assert budget['native_preview_entries'] <= 468 and budget['elapsed_seconds'] < 90
                result = {key: arm[key] for key in (
                    'status', 'outcomes', 'actions', 'actor_decision_seconds', 'native_transition_seconds',
                    'independent_audit_seconds', 'full_arm_seconds_including_audit',
                    'independent_evaluation_accepted', 'optimizer_updates', 'initial_parameter_hash',
                    'final_parameter_hash', 'episode_sha256', 'terminal_history_sha256', 'terminal_metrics_hash')}
                result.update({'online_seconds': budget['elapsed_seconds'],
                               'native_previews': budget['native_preview_entries'],
                               'native_preview_profile': budget['native_preview_profile'],
                               'cache_hits': budget['cache_hits']})
                if method == 'greedy_search':
                    search = arm['search_accounting']
                    result['search'] = {k: search[k] for k in (
                        'planning_seconds', 'evaluated_nonstop_actions', 'model_transition_calls',
                        'complete', 'global_optimality_proven', 'objective_source', 'within_call_costs')}
                row['arms'][method] = result
            row['greedy_minus_il_reward'] = (row['arms']['greedy_search']['outcomes']['total_reward']
                                           - row['arms']['frozen_il']['outcomes']['total_reward'])
        rows.append(row)
    report = {
        'schema': 'prepared-training-planner-v2-saved-report-v1',
        'patients_prescribed': 6, 'complete_comparisons': 4, 'accepted_arms': 12,
        'optimizer_updates': 0, 'clinical_efficacy_measured': False,
        'settings': declaration['settings'], 'checkpoint': declaration['checkpoint'],
        'representation': declaration['representation'], 'fixed_arm_order': list(METHODS),
        'root_terminal': terminal, 'patients': rows, 'saved_input_hashes': before,
        'independent_audit': {'path': str(paths[2].relative_to(ROOT)), 'sha256': AUDIT_SHA,
                              'status': 'saved_records_verified', 'scope': 'Software/simulation records only; no physical or clinical validation.'},
        'interpretation': [
            'All six cases are TRAIN: PAT05 is the trained case; PAT22/25/28 are development transfer.',
            'Historical PAT16/20 support blocks remain null, not zero-valued episodes.',
            'Frozen imitation uses a fixed64 crop; greedy search uses full permitted nominal fields.',
            'Online cost includes clone/decisions/search/replay/successor inventories/export; shared preparation and audit are separate.',
            'Fixed-order single-run times are descriptive, not a randomized speed benchmark.',
            'Geometric normal removal is not neurological harm; clinical deficit probability remains null.',
            'Prior V1 durable-history failures are retained; V2 has exact matching saved behavior aside from timing/status.']}
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    (HERE / 'compact-results.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    plot(report)


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    colors = ('#77828D', '#007E87', '#B56D36')
    markers = ('o', 's', 'D')
    labels = ('STOP', 'Frozen imitation', 'Greedy search')
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 8.4), sharey=True)
    fig.subplots_adjust(left=.17, right=.965, bottom=.21, top=.78, wspace=.25)
    fig.suptitle('Prepared TRAIN planner comparison', x=.055, y=.965, ha='left', fontsize=20, weight='bold')
    fig.text(.055, .916, '4 complete comparisons / 6 prescribed cases · 12 accepted simulation histories · zero policy updates', fontsize=12, color='#465364')
    fig.legend([Line2D([], [], color=c, marker=m, linestyle='', markersize=7) for c, m in zip(colors, markers)],
               labels, loc='upper left', bbox_to_anchor=(.05, .88), ncol=3, frameon=False)
    axes[0].set_title('A  Frozen geometric objective', loc='left', fontsize=12, weight='bold', pad=12)
    axes[1].set_title('B  Online planning + execution cost', loc='left', fontsize=12, weight='bold', pad=12)
    for i, row in enumerate(report['patients']):
        if row['status'] != 'complete':
            for ax in axes:
                ax.axhspan(i-.42, i+.42, color='#F1F3F5', zorder=0)
                ax.text(.06, i, 'Not executed — support conflict', transform=ax.get_yaxis_transform(),
                        va='center', color='#66727E', fontsize=9)
            continue
        for j, method in enumerate(METHODS):
            arm = row['arms'][method]; y = i + (j-1)*.22
            values = (arm['outcomes']['total_reward'], arm['online_seconds'])
            for k, (ax, x) in enumerate(zip(axes, values)):
                ax.scatter(x, y, s=38, color=colors[j], marker=markers[j], zorder=3)
                label = f'{x:.1f}' if k == 0 else f'{x:.2f} s · {arm["native_previews"]} previews'
                ax.text(x + (14 if k == 0 else 1.7), y, label, va='center', fontsize=8.7, color=colors[j])
    for ax in axes:
        ax.set_ylim(5.6, -.6)
        ax.set_yticks(range(6))
        ax.set_yticklabels(['PAT05\ntrained case', 'PAT16\nsupport block', 'PAT20\nsupport block',
                            'PAT22\ndevelopment transfer', 'PAT25\ndevelopment transfer', 'PAT28\ndevelopment transfer'])
        ax.grid(axis='x', color='#E6EBEF', linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#CAD2DA')
        ax.tick_params(axis='y', length=0, pad=10)
    axes[0].set(xlim=(-30, 940), xlabel='Simulation reward (higher under this fixed objective)')
    axes[1].set(xlim=(-2, 108), xlabel='Guarded online seconds; labels include native preview calls')
    axes[1].axvline(90, color='#A4ADB6', linestyle='--', linewidth=1)
    axes[1].text(90, -.52, '90 s cap', ha='center', fontsize=8.7, color='#66727E')
    fig.text(.17, .14, 'Each arm: ≤90 s and ≤468 native previews. Shared preparation and independent audits are excluded here and reported separately.', fontsize=10, color='#465364')
    fig.text(.17, .095, 'Same prepared task; CNN sees its 64³ crop, search uses full nominal fields. Fixed-order times are descriptive.\nTRAIN development evidence only; geometric removal is not neurological harm. Clinical deficit probability remains null.',
             fontsize=10, color='#465364', linespacing=1.6)
    for extension in ('png', 'svg'):
        fig.savefig(HERE / f'comparison.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
