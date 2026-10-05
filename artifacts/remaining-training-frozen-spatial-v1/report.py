"""Read completed first-attempt receipts only; no task/model imports or replay."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary = json.loads((ROOT / 'summary.json').read_text())
declaration = json.loads((ROOT / 'declaration-input.json').read_text())
index = json.loads((ROOT / 'output-sha256.json').read_text())
for name, digest in index.items():
    assert sha(ROOT / name) == digest, name
rows = []
for patient in summary['patients']:
    name = patient['subject']
    preparation = json.loads((ROOT / name / 'preparation.json').read_text())
    coverage = preparation['coverage']
    actor = coverage.get('actor', {})
    proposals = coverage.get('proposals', {})
    rows.append({
        'subject': name, 'status': patient['status'],
        'coverage': {key: coverage.get(key) for key in (
            'full_target_source_cells', 'full_target_volume_mm3',
            'target_outside_support_source_cells', 'target_outside_actor_crop_source_cells')},
        'initial_legal_nonstop_actions': actor.get('legal_nonstop_actions'),
        'nominal_target_mass_fraction_visible': actor.get('nominal_target_mass_fraction_visible'),
        'initial_preview_dispositions': proposals.get('preview_dispositions'),
        'initial_accepted_action_aabb': proposals.get('accepted_envelope'),
        'preparation_seconds': patient['preparation_seconds'],
        'supervised_seconds': patient['supervisor']['seconds'],
        'sampled_peak_rss_bytes': patient['supervisor']['sampled_peak_rss_bytes'],
        'source_and_checkpoint_closure': patient['closure_check'],
        'methods': {method: {key: value.get(key) for key in (
            'status', 'failure', 'outcomes', 'actions', 'target_access_success',
            'independent_evaluation_accepted', 'initial_parameter_hash', 'final_parameter_hash',
            'total_method_seconds', 'planning_seconds', 'episode_seconds',
            'actor_decision_seconds', 'native_steps_including_inventory_seconds',
            'independent_audit_seconds', 'policy_forward_calls', 'invalid_actions')}
            for method, value in patient['methods'].items()},
    })
compact = {
    'scope': summary['scope'], 'source_commit': '5198947',
    'declaration_sha256': sha(ROOT / 'declaration-input.json'),
    'execution_index_sha256': sha(ROOT / 'output-sha256.json'),
    'patients_prescribed': 5, 'prepared_patients': 3, 'blocked_patients': 2,
    'complete_primary_pairs': summary['complete_pairs'],
    'nontrivial_complete_primary_pairs': 1,
    'accepted_episodes': 6, 'failed_episodes_retained_unassessed': 3,
    'optimizer_updates': 0, 'adaptation': False, 'automatic_retry': False,
    'checkpoint': declaration['checkpoint'],
    'random_seed': declaration['settings']['random_seed'],
    'method_order': declaration['methods'],
    'summed_supervised_patient_seconds': sum(r['supervised_seconds'] for r in rows),
    'maximum_sampled_worker_rss_bytes': max(r['sampled_peak_rss_bytes'] for r in rows),
    'paired_return_difference': summary['paired_return_difference'],
    'patients': rows,
}
(ROOT / 'compact-summary.json').write_text(json.dumps(compact, indent=2, sort_keys=True) + '\n')

lines = [
    '# Frozen PAT05 model on the five remaining TRAIN patients', '',
    'The first attempt produced two accepted primary pairs out of five prescribed patients; only PAT22 supplied an accepted primary comparison with non-STOP actions. The frozen model remained below greedy search on that case. No optimizer steps, adaptation, selection-patient access, or retries occurred.', '',
    '| Patient | Frozen policy return | Greedy return | Random return (one episode) | First-attempt disposition |',
    '|---|---:|---:|---:|---|',
]
notes = {
    'sub-PAT16': 'Blocked: 19 supplied target cells outside support',
    'sub-PAT20': 'Blocked: 125 supplied target cells outside support',
    'sub-PAT22': 'Primary pair accepted; random target-volume accounting failed',
    'sub-PAT25': 'Only STOP legal; all 78 previews failed shaft clearance',
    'sub-PAT28': 'Both primary normal-volume accounting checks failed; random accepted',
}
for row in rows:
    scores = [('—' if row['methods'][m]['outcomes'] is None else
               f"{row['methods'][m]['outcomes']['total_reward']:.3f}")
              for m in declaration['methods']]
    lines.append(f"| {row['subject'][4:]} | {' | '.join(scores)} | {notes[row['subject']]} |")
lines += [
    '', 'A dash means unassessed, never a zero. Failed method receipts retain the completed native history and failure reason. Their unaccepted simulator totals are excluded from comparison. The independent saved-record check confirmed all five cases, six accepted episodes, and three failed/null episodes; it did not rerun native geometry.', '',
    'On PAT22, the frozen model removed 697.998 mm³ of target and 127.000 mm³ of normal tissue, versus 819.998 and 153.000 mm³ for greedy. Target fractions were 5.016% and 5.893% of the complete 13,914.959 mm³ supplied annotation. The model’s return gap was −116.800. Both paths totaled 211.000 mm, with no tool changes. Partial contact has zero reward weight in this declared geometric task and is reported separately; it is not an injury probability.', '',
    'The three prepared cases included all supplied target cells in the 64³ actor crop. That does not imply proposal coverage: initially 10,186/13,915 PAT22 target centers and 7,509/10,269 PAT28 target centers lay outside the accepted proposal AABB. PAT25 had no accepted non-STOP proposal. The fixed access rule and generic tools therefore remain substantial task limitations. Support was neither expanded nor target-clipped for blocked cases.', '',
    'The 30,827-parameter spatial model is the fixed visited-imitation checkpoint trained only on PAT05. It ranks certified geometric candidates using a supplied annotation, scan, cavity and tool geometry. Greedy receives the same permitted annotation but scores the full nominal source fields, while the CNN uses its crop. This is development transfer from one training patient, not population pretraining, scan-only inference, or held-out clinical efficacy.', '',
    '| Patient | Preparation (s) | Frozen method (s) | Greedy plan + replay/audit (s) | Random method (s) | Whole supervised worker (s) |',
    '|---|---:|---:|---:|---:|---:|',
]
for row in rows:
    costs = [('—' if row['methods'][m]['total_method_seconds'] is None else
              f"{row['methods'][m]['total_method_seconds']:.3f}")
             for m in declaration['methods']]
    lines.append(f"| {row['subject'][4:]} | {row['preparation_seconds']:.3f} | {' | '.join(costs)} | {row['supervised_seconds']:.3f} |")
lines += [
    '', f"All five supervised workers finished within their 180 s / 6 GiB limits: summed supervision was {compact['summed_supervised_patient_seconds']:.3f} s, and the maximum sampled RSS was {compact['maximum_sampled_worker_rss_bytes']:,} bytes. Sampling can miss transient peaks. No timeout or automatic retry occurred. Methods ran frozen, greedy, then optional random, with random seed 200011. Failed-method time remains charged.", '',
    'PAT22 policy forwards used 0.934 s, but its full method took 20.331 s, including 12.795 s native transitions/inventory and 4.043 s independent audit, plus separately reported shared preparation of 7.587 s. Greedy planning took 15.174 s and its complete method 34.973 s. These measurements do not include prior PAT05 training costs and do not establish an amortized speed advantage.', '',
    'The source/checkpoint closure passed after every patient; the checkpoint parameter hash remained `74e90d9487dff6fe9db83c92e3887f15533e0499abaac386a73bbd7d4d566b4e`. The numerical source snapshot, exact declaration, copied checkpoint, all successful and failed receipts, and original output index are retained in `completed-run.tar.gz`, verified byte-for-byte against the raw files. `summary.json` and `output-sha256.json` remain original execution records. Later precision diagnosis or repair does not retroactively change this attempt.', '',
    'The pre-execution integrated check first reported 31 passed / 1 failed because the restricted Torch loader rejected NumPy scalar architecture metadata. A narrow hash-verified local-checkpoint loader correction yielded 32 passed in 2.10 s before this run. See `verification.json` for the provenance and exact targets.', '',
    '![Accepted first-attempt returns](comparison.png)', '',
]
(ROOT / 'RESULT.md').write_text('\n'.join(lines))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(10.8, 5.8))
colors = ['#176e85', '#cf7b30', '#777777']
labels = ['Frozen PAT05 model', 'Greedy search', 'Random legal (n=1)']
for i, row in enumerate(rows):
    for j, method in enumerate(declaration['methods']):
        outcome = row['methods'][method]['outcomes']; y = 4-i + (.21-j*.21)
        if outcome is not None:
            value = outcome['total_reward']
            ax.barh(y, value, height=.16, color=colors[j])
            ax.scatter([value], [y], color=colors[j], s=14, zorder=3)
            ax.text(value + 12, y, f'{value:.1f}', va='center', fontsize=9, color=colors[j])
        else:
            text = 'blocked' if row['status'] == 'preparation_blocked' else 'audit failed'
            ax.text(16, y, text, va='center', fontsize=9, color=colors[j])
ax.set_yticks(list(range(4,-1,-1)), [r['subject'][4:] for r in rows])
ax.set_xlim(-10, 880); ax.set_ylim(-.5, 4.5); ax.set_xlabel('Accepted geometric return; missing outcomes are not zero')
ax.set_title('Frozen transfer within TRAIN: 2/5 accepted primary pairs\nPAT25 is STOP-only; no updates, adaptation, or retries', loc='left', fontsize=13)
ax.spines[['top','right']].set_visible(False); ax.set_axisbelow(True);ax.grid(axis='x', alpha=.15)
ax.legend(handles=[Line2D([0],[0], color=c, lw=6, label=l) for c,l in zip(colors,labels)], loc='lower right', frameon=False)
fig.tight_layout();fig.savefig(ROOT/'comparison.png', dpi=150);fig.savefig(ROOT/'comparison.svg');plt.close(fig)
outputs=['compact-summary.json','RESULT.md','comparison.png','comparison.svg']
receipt={'report_source_sha256':sha(Path(__file__)), 'inputs_sha256':{k:index[k] for k in index if k=='summary.json' or k.endswith('/preparation.json')}, 'output_index_sha256':sha(ROOT/'output-sha256.json'), 'outputs_sha256':{k:sha(ROOT/k) for k in outputs}, 'new_patient_episodes':0,'policy_forwards':0,'gradients':0,'original_execution_records_modified':False}
(ROOT/'report-source.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in compact.items() if k not in ['patients','checkpoint']},indent=2))
