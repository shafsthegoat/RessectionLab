#!/usr/bin/env python3
"""Render this completed development attempt directly from its retained JSON."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

def source_bytes(name):
    path = HERE / name
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())

def read(name):
    return json.loads(source_bytes(name))

summary = read('summary.json')
status = read('experiment-status.json')
comparison = read('comparison/status.json')
assert summary['status'] == status['status'] == comparison['status'] == 'completed'
assert summary['final_worlds_used'] is False
assert len(comparison['completed_learning_runs']) == 6
assert summary['validation']['candidate_count'] == 13
assert not summary['validation']['rejected_candidate_ids']
assert all(summary['independent_geometry'].values())
audit_path = 'comparison/' + comparison['geometry_validation_run_id']
replays = read(audit_path + '/native-history-replay.json')
audits = read(audit_path + '/native-history-audit.json')
launch, worker = read('launch-source.json'), read('worker-source.json')
assert launch['numerical_runtime_content_hash'] == worker['numerical_runtime_content_hash'] == summary['source_hash']
shared = summary['frozen']['shared_checkpoint_hash']
learning = summary['learning']
for row in learning:
    assert row['actor_parameters_changed']
    assert row['gradient_steps'] <= 32 and row['optimization_environment_steps'] <= 256
    assert row['selected_selection_return'] == replays[row['optimizer_mode'] + ':' + str(row['seed'])]['metrics']['total_reward']
    if row['optimizer_mode'] == 'PROCEDURAL_PRETRAINED_ADAPTED':
        assert row['initial_checkpoint_hash'] == row['shared_checkpoint_hash'] == shared
assert summary['frozen']['gradient_steps'] == summary['frozen']['optimization_environment_steps'] == 0

sources = ['summary.json', 'experiment-status.json', 'comparison/status.json',
    'comparison/candidate-freeze.json', 'launch-source.json', 'worker-source.json',
    audit_path + '/native-history-replay.json', audit_path + '/native-history-audit.json']
source_hashes = {name: hashlib.sha256(source_bytes(name)).hexdigest() for name in sources}

def numbers(values):
    return ', '.join(f'{x:.2f}' for x in values)

search = next(x for x in summary['search'] if x['method'] == 'SEARCH')
greedy = next(x for x in summary['search'] if x['method'] == 'GREEDY')
frozen = summary['frozen']
offline = summary['offline_pretraining']
cost = offline['provenance']['offline_pretraining']
scratch = [row for row in learning if row['optimizer_mode'] == 'PATIENT_SCRATCH_RL']
adapted = [row for row in learning if row['optimizer_mode'] == 'PROCEDURAL_PRETRAINED_ADAPTED']
assert [row['seed'] for row in scratch] == [11, 23, 47] == [row['seed'] for row in adapted]
rows = []
for item in (greedy, search):
    rows.append((item['method'], None, item['nominal_score'], 0, item['environment_steps'], 0, item['elapsed_seconds']))
rows.append(('Procedural frozen', None, frozen['selection_return'], 0, 0, frozen['selection_environment_steps'], frozen['elapsed_seconds']))
for label, items in [('Scratch', scratch), ('Procedural adapted', adapted)]:
    for item in items:
        rows.append((label, item['seed'], item['selected_selection_return'], item['gradient_steps'],
            item['optimization_environment_steps'], item['selection_environment_steps'], item['elapsed_seconds']))
lines = ['# Procedural-to-patient development attempt v2', '',
    'The completed attempt demonstrates actual patient-specific updates and independently checked native removal. Search remains the strongest result in this declared action model; procedural pretraining does not establish an advantage over scratch learning.', '',
    'This is one previously explored patient-derived development case, two procedural source families, and zero human pretraining patients. The three seeds are optimizer repeats within this patient, not three independent patients. Motor and language evidence are absent; clinical deficit probabilities are null. All world perturbations are zero, and final/stress worlds remain unopened.', '',
    '| Method | Seed | Selected simulated return | Adam updates | Optimization/search transitions | Selection transitions | Online seconds |',
    '|---|---:|---:|---:|---:|---:|---:|']
for label, seed, score, updates, steps, select, seconds in rows:
    lines.append(f'| {label} | {seed if seed is not None else "—"} | {score:.2f} | {updates} | {steps} | {select} | {seconds:.3f} |')
lines += ['',
    'Online seconds are optimization plus selection (including initial selection); the frozen row measures its complete selection panel. Search and greedy terminated after exhausting/stopping their declared search, rather than consuming the whole allowance. Cold simulator setup, model/checkpoint initialization, extraction and validation are separate. Each scratch/adapted arm had a 30 s cooperative cap, 32-update cap, and 256-optimization-transition cap; selection transitions are additional. The wall-time allowance is matched, not the total transitions or executed Adam steps.', '',
    f'Scratch initial returns were [{numbers([x["initial_selection_return"] for x in scratch])}]. Seed 23 retained its initial checkpoint; seeds 11 and 47 improved. All adapted runs started from the identical shared policy at {frozen["selection_return"]:.2f} and selected updated checkpoints. All six actors changed, including the run whose selected checkpoint stayed initial. No learned arm exceeded search. The adapted-minus-scratch returns for seeds 11/23/47 are [{numbers([a["selected_selection_return"]-s["selected_selection_return"] for a,s in zip(adapted,scratch)])}]; these are descriptive optimizer differences in one case.', '',
    f'Fresh procedural pretraining used {cost["gradient_steps"]} Adam updates, {cost["optimization_environment_steps"]} optimization transitions and {cost["selection_environment_steps"]} selection transitions. Its total measured cost was {offline["total_offline_seconds"]:.3f} s, including {cost["elapsed_seconds"]:.3f} s optimization/selection, {cost["initialization_seconds"]:.3f} s initialization and {cost["preparation_seconds"]:.3f} s preparation. The full pretraining wrapper call was {summary["pretraining_call_seconds"]:.3f} s; the {offline["total_offline_seconds"]:.3f} s inner total excludes that small caller overhead. These costs are additional to patient adaptation.', '',
    f'Cooperative online wall overshoot ranged from {min(x["learner_wall_budget_overshoot_seconds"] for x in learning):.3f} to {max(x["learner_wall_budget_overshoot_seconds"] for x in learning):.3f} s. Cold setup took {min(x["preparation_seconds"] for x in learning):.3f}–{max(x["preparation_seconds"] for x in learning):.3f} s per learned arm. Scratch initialization took {min(x["initialization_seconds"] for x in scratch):.3f}–{max(x["initialization_seconds"] for x in scratch):.3f} s; adapted initialization took {min(x["initialization_seconds"] for x in adapted):.3f}–{max(x["initialization_seconds"] for x in adapted):.3f} s. Frozen shared validation/loading was separately measured at {frozen["shared_checkpoint_initialization_seconds"]:.3f} s. These phase times must not be described as end-to-end latency.', '',
    f'All {summary["validation"]["candidate_count"]} frozen candidates passed independent native containment, frontier and complete-tool checks, using {summary["validation"]["unique_audits"]} distinct sequence audits. Full validation took {summary["validation"]["full_validation_seconds"]:.3f} s. Search removes 249 mm³ of labeled target and 17 mm³ of modeled normal tissue; its cumulative partial normal contact is 32 mm³, recorded separately from removal. Its modeled residual target is 41,670 mm³: this bounded route slice is not a complete resection plan.', '',
    f'Other agents paused heavy tests and resampling during timed execution. Normal desktop and operating-system background activity remained uncontrolled; saved load averages and process peak RSS describe observed conditions, not isolated-machine timing. The worker peak RSS reached {max(x["execution_context_after"]["process_peak_rss_bytes"] for x in learning)/2**30:.3f} GiB. Validation spent about {summary["validation"]["full_validation_seconds"]-sum(summary["validation"]["candidate_seconds"].values()):.3f} s beyond measured candidate work in serialization and other overhead.', '',
    'The source is a public structural mirror of UCSF-PDGM-0004; equivalence to official TCIA bytes remains unverified. The working brain envelope is an unreviewed estimate, access is hypothetical, and the finite native action inventory is deliberately small. These results do not validate clinical safety, uncertainty robustness, population generalization or a complete surgery workflow.', '',
    f'The launcher completed in {status["total_seconds_including_source_freeze"]:.3f} s. Execution used committed source `68e4fde13b1fa13411e59af663bd17ae63885947`, numerical runtime `{summary["source_hash"]}`, the unchanged v1 scientific declaration, and the v2 implementation-repair attempt record. The failed v1 attempt is retained beside this directory and supplies no comparative result. Settings were not changed after this completed outcome.', '',
    '![Per-seed simulated returns and online optimization/selection time](comparison.png)', '',
    'The full native replay is retained locally and also versioned as lossless gzip with byte-for-byte roundtrip, SHA256 and size evidence in `replay-storage.json` beside the compressed file. No completed experiment record was changed.', '',
    'The plot and this report are generated from retained JSON by `report.py`; source checksums are in `report-source.json`. All native replays, selected checkpoints, complete selection records, source snapshots and phase costs remain available in this directory.']
(HERE/'RESULT.md').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
colors = {'Scratch':'#147D92','Procedural adapted':'#834CA5'}
for offset, label, items in [(-0.1,'Scratch',scratch),(0.1,'Procedural adapted',adapted)]:
    x = [i+offset for i in range(3)]
    y = [row['selected_selection_return'] for row in items]
    axes[0].plot(x,y,'o-',lw=1.5,markersize=7,color=colors[label],label=label)
axes[0].axhline(search['nominal_score'],color='#24354B',linestyle='--',lw=1.4,label='Search / greedy')
axes[0].axhline(frozen['selection_return'],color='#A17935',linestyle=':',lw=1.7,label='Procedural frozen')
axes[0].set(xticks=range(3),xticklabels=['11','23','47'],xlabel='Optimizer seed (same patient)',ylabel='Selected simulated return',ylim=(0,275),title='Selected policy performance')
axes[0].legend(loc='lower right',fontsize=9,frameon=False)
labels=['Greedy','Search','Frozen','Scratch 11','Scratch 23','Scratch 47','Adapted 11','Adapted 23','Adapted 47']
seconds=[greedy['elapsed_seconds'],search['elapsed_seconds'],frozen['elapsed_seconds']]+[r['elapsed_seconds'] for r in scratch+adapted]
axes[1].barh(range(len(labels)),seconds,color=['#657688','#24354B','#A17935']+['#147D92']*3+['#834CA5']*3)
axes[1].set(yticks=range(len(labels)),yticklabels=labels,xlabel='Online seconds (setup / audit excluded)',title='Actual optimization + selection time',xlim=(0,35))
axes[1].invert_yaxis();axes[1].axvline(30,color='#8A929E',lw=1,linestyle='--')
for i,value in enumerate(seconds): axes[1].text(value+.3,i,f'{value:.2f}',va='center',fontsize=8)
fig.suptitle('Procedural transfer · one development patient · deterministic worlds',fontsize=12)
fig.savefig(HERE/'comparison.png',dpi=180)
fig.savefig(HERE/'comparison.pdf')
plt.close(fig)
(HERE/'report-source.json').write_text(json.dumps({'source_file_sha256':source_hashes,'report_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numeric_values_copied_from_raw_json':True,'final_worlds_used':False,'independent_patient_count':1,'human_pretraining_patients':0},indent=2)+'\n')
print(HERE/'RESULT.md')
