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
    'scope': summary['scope'], 'source_commit': '7ea85f5',
    'declaration_sha256': sha(ROOT / 'declaration-input.json'),
    'execution_index_sha256': sha(ROOT / 'output-sha256.json'),
    'patients_prescribed': 5, 'prepared_patients': 3, 'blocked_patients': 2,
    'complete_primary_pairs': summary['complete_pairs'],
    'nontrivial_complete_primary_pairs': 2,
    'accepted_episodes': 9, 'failed_episodes_retained_unassessed': 0,
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

comparison=json.loads((ROOT/'attempt-comparison.json').read_text())
assert comparison['status']=='passed'
lines=[
 '# Precision-only repeat: frozen PAT05 model on remaining TRAIN patients','',
 'All nine completed episodes across the three prepared patients passed independent native geometry and volume/reward accounting. PAT16 and PAT20 remained blocked. Greedy search exceeded the frozen model on both cases with legal non-STOP actions; PAT25 remained STOP-only. No training, adaptation, checkpoint selection, or further retry occurred.','',
 '| Patient | Frozen model return | Greedy return | Random return (one episode) | Disposition |',
 '|---|---:|---:|---:|---|',
]
notes={'sub-PAT16':'Blocked: 19 supplied target cells outside support','sub-PAT20':'Blocked: 125 supplied target cells outside support','sub-PAT22':'All three episodes accepted','sub-PAT25':'Only STOP legal; all 78 previews failed shaft clearance','sub-PAT28':'All three episodes accepted'}
for row in rows:
 scores=['—' if row['methods'][m]['outcomes'] is None else f"{row['methods'][m]['outcomes']['total_reward']:.3f}" for m in declaration['methods']]
 lines.append(f"| {row['subject'][4:]} | {' | '.join(scores)} | {notes[row['subject']]} |")
lines += ['', 'A dash means unassessed, never zero. There are three accepted primary pairs out of five prescribed patients, but only two pairs exercise a non-STOP policy choice. Random seed 200011 supplies one diagnostic trajectory per prepared case, not an estimate of typical random performance.','',
 '| Patient / method | Target removed (mm³) | Normal removed (mm³) | Fraction of complete target | Path (mm) |',
 '|---|---:|---:|---:|---:|']
for row in rows:
 if row['subject'] not in ('sub-PAT22','sub-PAT28'):continue
 for m,label in [('frozen_policy','frozen'),('greedy_search','greedy'),('random_legal','random')]:
  v=row['methods'][m]['outcomes'];lines.append(f"| {row['subject'][4:]} / {label} | {v['target_removed_mm3']:.3f} | {v['normal_removed_mm3']:.3f} | {100*v['reference_target_fraction_removed']:.3f}% | {v['complete_tool_path_length_mm']:.3f} |")
lines += ['', 'Frozen-minus-greedy return was −116.800 on PAT22 and −119.004 on PAT28. The PAT28 model removed less target and more normal tissue than greedy. Partial-contact cost remains zero in the frozen geometric reward and contact is reported separately; none of these scores is a clinical deficit estimate.','',
 'The repeat changed only the measured aggregation precision in `native_spatial_task.py` and `native_spatial_evaluation.py`. The original support acknowledgment timestamp, checkpoint bytes, patient order, roles, access rule, settings and random seed are identical. All five saved preparation records match exactly except timing. Across all nine completed trajectories, action IDs/order/masks, observed inputs, chosen actions, frozen logits/probabilities/values, removed/contact cells, microsteps, complete-tool geometry and provenance match the first attempt exactly. Only saved target/normal/reward arithmetic and measured durations changed. `attempt-comparison.json` records these assertions and input hashes.','',
 'This is a new execution after a precision correction, not a reinterpretation of the old attempt. Its three previously failed/null method outcomes remain failed/null in the original directory and original archive. The new nine native checks and the independent saved-record review pass separately. The largest per-episode change in the saved simulator return is ' + f"{max(abs(e['return_arithmetic_difference']) for p in comparison['patients'] for e in p['episodes']):.9f}" + ', with no change to behavior.','',
 'The model is still the 30,827-parameter PAT05 visited-imitation checkpoint, with parameter hash `74e90d9487dff6fe9db83c92e3887f15533e0499abaac386a73bbd7d4d566b4e`. This is development transfer from one training patient, not population pretraining or held-out efficacy. It ranks certified proposals from annotation-assisted input. The actor crop includes every supplied target cell on the three prepared cases; greedy still uses full nominal source fields. Initial accepted proposal envelopes exclude 10,186/13,915 PAT22 and 7,509/10,269 PAT28 target centers. PAT25 has no legal non-STOP proposal. Support was not expanded and targets were not clipped for the blocked cases.','',
 '| Patient | Shared preparation (s) | Frozen method (s) | Greedy plan + replay/audit (s) | Random method (s) | Whole supervised worker (s) |',
 '|---|---:|---:|---:|---:|---:|']
for row in rows:
 costs=['—' if row['methods'][m]['total_method_seconds'] is None else f"{row['methods'][m]['total_method_seconds']:.3f}" for m in declaration['methods']]
 lines.append(f"| {row['subject'][4:]} | {row['preparation_seconds']:.3f} | {' | '.join(costs)} | {row['supervised_seconds']:.3f} |")
lines += ['', f"Summed supervised patient time was {compact['summed_supervised_patient_seconds']:.3f} s, with maximum sampled worker RSS {compact['maximum_sampled_worker_rss_bytes']:,} bytes. Every worker finished within 180 s / 6 GiB. Methods ran frozen, greedy, then random. RSS sampling can miss transient peaks.",'',
 'PAT22/PAT28 actor forwards took 1.228/1.175 s, versus full frozen methods of 21.258/19.741 s; native transitions and successor inventories consumed 13.211/12.832 s, and independent audits 4.236/3.534 s. Shared preparation is charged separately above. Greedy planning took 15.889/15.030 s before its full replays. These are single-run costs and exclude prior PAT05 training; no amortized speed or clinical benefit is established.','',
 'Source commit `7ea85f5` and the exact declaration (`9981dd90217e57b1fd8c6ff62580e9c1c7c9cefe991537e8b056ec68ab4756a5`) were frozen before execution. All original execution files, the source snapshot and copied checkpoint are retained in `completed-run.tar.gz`, verified byte-for-byte. `summary.json` and `output-sha256.json` are unchanged execution records. The separate compact report and old/new comparison do not load patient arrays or perform policy forwards.','',
 '![Accepted precision-repeat returns](comparison.png)','']
(ROOT/'RESULT.md').write_text('\n'.join(lines))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
fig,ax=plt.subplots(figsize=(10.8,5.8));colors=['#176e85','#cf7b30','#777777'];labels=['Frozen PAT05 model','Greedy search','Random legal (n=1)']
for i,row in enumerate(rows):
 for j,method in enumerate(declaration['methods']):
  outcome=row['methods'][method]['outcomes'];y=4-i+(.21-j*.21)
  if outcome is not None:
   value=outcome['total_reward'];ax.barh(y,value,height=.16,color=colors[j]);ax.scatter([value],[y],color=colors[j],s=14,zorder=3);ax.text(value+12,y,f'{value:.1f}',va='center',fontsize=9,color=colors[j])
  else:ax.text(16,y,'blocked',va='center',fontsize=9,color=colors[j])
ax.set_yticks(list(range(4,-1,-1)),[r['subject'][4:] for r in rows]);ax.set_xlim(-10,880);ax.set_ylim(-.5,4.5);ax.set_xlabel('Accepted geometric return; blocked cases are not zero')
ax.set_title('Precision repeat: 3/5 accepted pairs; PAT25 is STOP-only\nSame nine action histories, frozen model, and task as first attempt',loc='left',fontsize=13)
ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15);ax.legend(handles=[Line2D([0],[0],color=c,lw=6,label=l) for c,l in zip(colors,labels)],loc='upper right',frameon=False)
fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);fig.savefig(ROOT/'comparison.svg');plt.close(fig)
outputs=['compact-summary.json','RESULT.md','comparison.png','comparison.svg']
receipt={'report_source_sha256':sha(Path(__file__)),'inputs_sha256':{k:index[k] for k in index if k=='summary.json' or k.endswith('/preparation.json')},'attempt_comparison_sha256':sha(ROOT/'attempt-comparison.json'),'output_index_sha256':sha(ROOT/'output-sha256.json'),'outputs_sha256':{k:sha(ROOT/k) for k in outputs},'new_patient_episodes':0,'policy_forwards':0,'gradients':0,'original_execution_records_modified':False}
(ROOT/'report-source.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in compact.items() if k not in ['patients','checkpoint']},indent=2))
