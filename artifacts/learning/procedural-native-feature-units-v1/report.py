#!/usr/bin/env python3
"""Reproduce the completed feature-unit report from retained raw or gzipped JSON."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

def source_bytes(name):
    path = HERE / name
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes())

def read(name):
    return json.loads(source_bytes(name))

summary = read('study/summary.json')
launcher = read('experiment-status.json')
status = read('study/status.json')
manifest = read('study/manifest.json')
declaration = manifest['declaration']
freeze = read('study/candidate-freeze.json')
assert summary['status'] == launcher['status'] == status['status'] == 'completed'
assert len(summary['completed_learning_runs']) == 12
assert len(summary['completed_offline_runs']) == len(summary['completed_frozen_runs']) == 2
assert len(freeze['candidates']) == summary['validation']['candidate_count'] == 23
assert not summary['validation']['rejected_candidate_ids']
assert not summary['final_worlds_used']
audit_dir = 'study/' + summary['geometry_validation_run_id']
audits = read(audit_dir + '/native-history-audit.json')
replays = read(audit_dir + '/native-history-replay.json')
assert len(audits) == 23 and all(x['feasible'] for x in audits.values())
assert read('launch-source.json')['numerical_runtime_content_hash'] == summary['source_hash'] == read('study/source.json')['numerical_runtime_content_hash']
assert read('launch-source.json')['git_revision'] is None and read('study/source.json')['git_revision'] is None
rows = summary['learning']
profiles = ('RAW', 'FEATURE_UNITS')
seeds = (11, 23, 47)
by = {(row['phase'], row['input_profile'], row['seed']): row for row in rows}
for row in rows:
    assert row['actor_parameters_changed']
    assert row['gradient_steps'] <= 32 and row['optimization_environment_steps'] <= 256
    mode = 'PATIENT_SCRATCH_RL' if row['phase'] == 'scratch' else 'PROCEDURAL_PRETRAINED_ADAPTED'
    label = f"{row['input_profile']}:{mode}:{row['seed']}"
    assert row['selected_selection_return'] == replays[label]['metrics']['total_reward']
    assert row['input_profile_hash'] == declaration['profiles'][row['input_profile']]['profile_content_hash']
    if row['phase'] == 'adapted':
        assert row['initial_checkpoint_hash'] == summary['shared'][row['input_profile']]['policy_hash']
for seed in seeds:
    assert by['scratch', 'RAW', seed]['initial_trainable_parameter_hash'] == by['scratch', 'FEATURE_UNITS', seed]['initial_trainable_parameter_hash']
for profile in profiles:
    assert summary['frozen'][profile]['gradient_steps'] == 0
    assert summary['frozen'][profile]['optimization_environment_steps'] == 0
    assert summary['frozen'][profile]['selection_panel_complete']

search = next(row for row in summary['search'] if row['method'] == 'SEARCH')
greedy = next(row for row in summary['search'] if row['method'] == 'GREEDY')
physical = replays['SEARCH']['metrics']
saturation = {profile: [row['initial_policy_diagnostics']['actor_tanh_fraction_abs_above_0_95'] for row in rows if row['input_profile'] == profile] for profile in profiles}
paired = {phase: [by[phase, 'FEATURE_UNITS', seed]['selected_selection_return'] - by[phase, 'RAW', seed]['selected_selection_return'] for seed in seeds] for phase in ('scratch', 'adapted')}
raw_frozen = summary['frozen']['RAW']['selection_return']
scaled_frozen = summary['frozen']['FEATURE_UNITS']['selection_return']
assert all(by['adapted','FEATURE_UNITS',seed]['selected_is_initial'] for seed in seeds)
assert all(by['adapted','FEATURE_UNITS',seed]['selected_selection_return'] == scaled_frozen for seed in seeds)

def fmt(values):
    return ', '.join(f'{value:.2f}' for value in values)

lines = ['# Native actor feature-unit study', '',
    f'Fixed feature units raised this run’s procedural frozen-policy return from **{raw_frozen:.2f} to {scaled_frozen:.2f}**, matching SEARCH. Every FEATURE_UNITS adaptation run retained that initial frozen checkpoint: the higher selected starting policy did not yield demonstrated additional patient-specific adaptation gain. All twelve actors received real updates; no learned arm exceeded the search value of {search["nominal_score"]:.2f}.', '',
    'This is a completed, preregistered two-profile development comparison in one previously explored patient-derived structural case. Pretraining used two procedural families and **zero human patients**. Three optimizer seeds are repeats within this case, not independent patients. Geometry/actions/tools, physical reward definitions, critic architecture and input/output units, loss weights, clipping and optimizer settings were fixed; only actor input units differed.', '',
    '| Mode | Profile | Seed | Initial → selected return | Selected initial? | Adam updates | Optimization / selection transitions | Online seconds | Full arm seconds* |',
    '|---|---|---:|---:|---|---:|---:|---:|---:|']
for phase in ('scratch', 'adapted'):
    for seed in seeds:
        for profile in profiles:
            row = by[phase, profile, seed]
            lines.append(f'| {phase.title()} | {profile} | {seed} | {row["initial_selection_return"]:.2f} → {row["selected_selection_return"]:.2f} | {"Yes" if row["selected_is_initial"] else "No"} | {row["gradient_steps"]} | {row["optimization_environment_steps"]} / {row["selection_environment_steps"]} | {row["elapsed_seconds"]:.3f} | {row["complete_arm_seconds_before_independent_audit"]:.3f} |')
lines += ['', '*Full arm time includes cold simulator preparation, policy/checkpoint/optimizer initialization, training, selection, reporting and candidate extraction; independent geometry auditing and shared offline pretraining are separately charged below.', '',
    f'FEATURE_UNITS-minus-RAW selected returns for seeds 11/23/47 are **[{fmt(paired["scratch"])}] for scratch** and **[{fmt(paired["adapted"])}] for adaptation**. Scratch therefore shows only a 0.08 return difference in one seed at the selection endpoint. Initial behaviors differ substantially despite paired trainable tensors: scaled scratch seed 11 starts at the search value, while scaled seeds 23 and 47 improve from 33.40 and 0.00. The design tests the whole feature-unit pipeline; it cannot attribute the frozen-policy difference separately to initialization versus offline learning.', '',
    '| Offline pretraining | Adam updates | Optimization / selection transitions | Optimization + selection s | Initialization s | Preparation s | Inner total s | Full call s |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for profile in profiles:
    pre = summary['pretraining'][profile]
    cost = pre['provenance']['offline_pretraining']
    lines.append(f'| {profile} | {cost["gradient_steps"]} | {cost["optimization_environment_steps"]} / {cost["selection_environment_steps"]} | {cost["elapsed_seconds"]:.3f} | {cost["initialization_seconds"]:.3f} | {cost["preparation_seconds"]:.3f} | {pre["total_offline_seconds"]:.3f} | {pre["full_pretraining_call_seconds"]:.3f} |')
lines += ['', 'The inner total includes the module’s export; the full call includes caller overhead. Offline pretraining is additional to every patient-arm comparison and is not hidden inside an inference-only speed claim. Fixed order also leaves process/library warm-up effects observable in initialization costs.', '',
    '| Shared baseline / frozen policy | Return | Search / optimization transitions | Selection transitions | Algorithm seconds | Cold preparation s | Checkpoint initialization s | Full arm s* |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in (greedy, search):
    lines.append(f'| {row["method"]} | {row["nominal_score"]:.2f} | {row["environment_steps"]} | 0 | {row["elapsed_seconds"]:.3f} | {row["preparation_seconds"]:.3f} | — | {row["complete_arm_seconds_before_independent_audit"]:.3f} |')
for profile in profiles:
    row = summary['frozen'][profile]
    lines.append(f'| {profile} frozen | {row["selection_return"]:.2f} | 0 | {row["selection_environment_steps"]} | {row["elapsed_seconds"]:.3f} | {row["preparation_seconds"]:.3f} | {row["shared_checkpoint_initialization_seconds"]:.3f} | {row["complete_arm_seconds_before_independent_audit"]:.3f} |')
lines += ['',
    f'The primary comparison matches 30-second cooperative optimization-plus-selection allowances, not executed updates or total transitions. Actual online updates ranged from {min(row["gradient_steps"] for row in rows)} to {max(row["gradient_steps"] for row in rows)}; 32 was a cap. Selection transitions are additional to the 256-optimization-transition cap. Online overshoot was {min(row["learner_wall_budget_overshoot_seconds"] for row in rows):.3f}–{max(row["learner_wall_budget_overshoot_seconds"] for row in rows):.3f} s. Per-arm receipts separate initial selection, later/partial selection, final checkpoint export, setup and residual reporting costs. Shared geometry baselines were run once and reused explicitly across profiles.', '',
    f'All **23 candidate identities passed independent native checks**, using {summary["validation"]["unique_audits"]} distinct sequence audits. The denominator includes three shared geometry baselines, six scratch initial policies, twelve selected policies and two frozen policies; no failed or unfinished arm was omitted. Full validation took {summary["validation"]["full_validation_seconds"]:.3f} s, including approximately {summary["validation"]["full_validation_seconds"] - sum(summary["validation"]["candidate_seconds"].values()):.3f} s beyond measured candidate work in serialization and other overhead.', '',
    f'At the initial permitted optimization state, RAW actor tanh activations exceeded |0.95| for {min(saturation["RAW"]):.1%}–{max(saturation["RAW"]):.1%} of recorded hidden activations across all candidate rows including STOP; FEATURE_UNITS recorded {max(saturation["FEATURE_UNITS"]):.1%}. These diagnostics used the actual transformed input path and no extra transitions or gradients. Reduced saturation is observed, but the comparison does not establish saturation as the sole mechanism; critic state aliasing and reward-gradient balance were unchanged.', '',
    f'SEARCH removes {physical["simulated_removed_target_volume_mm3"]:.0f} mm³ labeled target and {physical["simulated_removed_normal_volume_mm3"]:.0f} mm³ modeled normal tissue. Its {physical["cumulative_partial_normal_contact_mm3"]:.0f} mm³ cumulative partial normal contact remains separate from removal. Modeled residual target is {physical["modeled_residual_target_volume_mm3"]:,.0f} mm³. The fixed proposal inventory supports only a bounded route slice, not a complete resection plan.', '',
    'Motor and language evidence remain absent and clinical deficit probabilities are null. The structural mirror’s official TCIA byte equivalence is unverified, the working brain envelope is an unreviewed estimate, and access is hypothetical. All perturbations were zero; final/stress worlds remained unopened. No clinical-safety, uncertainty-calibration, unseen-patient or population-generalization claim follows.', '',
    f'Total launcher time was {launcher["total_seconds_including_source_freeze"]:.3f} s; study time was {summary["total_seconds"]:.3f} s. Heavy parallel tests/resampling were paused, but normal desktop/OS background activity was uncontrolled. Peak recorded worker RSS was {max(row["execution_context_after"]["process_peak_rss_bytes"] for row in rows) / 2**30:.3f} GiB. These are observed local timings, not isolated-machine latency guarantees.', '',
    f'Execution used commit `0bffeaa33dc3120183d0c6fb124aff6e2d327a63`, archive SHA256 `f46356427bc270428984bd7f3e518e7946212c68ec0972111e4004fb10579cb9`, numerical runtime `{summary["source_hash"]}`, and declaration `{summary["declaration_hash"]}`. The tested archive passed 794 Python checks and actual public-case preflight before release. Both launcher and worker Git discovery were bounded to avoid mutable outer-repository metadata. Source snapshots, checkpoint/profile identities and raw records were retained without outcome-driven changes.', '',
    '![Initial and selected physical returns by profile and optimizer seed](comparison.png)', '',
    'This report and plot are generated by `report.py`; checksums are in `report-source.json`. Large native replay histories are preserved locally and may be read from the versioned lossless gzip, with byte/semantic roundtrip evidence in the adjacent `replay-storage.json`. No completed numerical record is modified by reporting.']
(HERE / 'RESULT.md').write_text('\n'.join(lines) + '\n')

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
colors = {'RAW':'#496078','FEATURE_UNITS':'#147D92'}
for axis, phase in zip(axes, ('scratch','adapted')):
    for seed_index, seed in enumerate(seeds):
        for profile, offset in (('RAW',-.12),('FEATURE_UNITS',.12)):
            row = by[phase, profile, seed]
            x = seed_index + offset
            initial, selected = row['initial_selection_return'], row['selected_selection_return']
            axis.plot([x,x],[initial,selected],color=colors[profile],alpha=.6,lw=2)
            axis.scatter([x],[initial],edgecolors=colors[profile],facecolors='white',s=90,lw=1.7,zorder=3)
            axis.scatter([x],[selected],color=colors[profile],s=25,zorder=4)
    axis.axhline(search['nominal_score'],color='#A57932',linestyle='--',lw=1.5)
    axis.set(xticks=range(3),xticklabels=[str(x) for x in seeds],xlabel='Optimizer seed (same patient)',
        ylabel='Physical simulated return',ylim=(-10,280),title=phase.title())
handles=[Line2D([0],[0],color=colors[p],marker='o',lw=0,label=p) for p in profiles]
handles += [Line2D([0],[0],marker='o',color='#777',markerfacecolor='white',lw=0,label='Initial (open)'),
    Line2D([0],[0],marker='o',color='#777',markersize=4,lw=0,label='Selected (dot)'),
    Line2D([0],[0],color='#A57932',linestyle='--',label='SEARCH / GREEDY')]
axes[1].legend(handles=handles,loc='lower center',fontsize=8.5,frameon=False)
fig.suptitle('Fixed actor units · one development patient · unchanged physical model',fontsize=12)
fig.savefig(HERE/'comparison.png',dpi=180)
fig.savefig(HERE/'comparison.pdf')
plt.close(fig)
names=['experiment-status.json','launch-source.json','study/source.json','study/summary.json','study/status.json',
    'study/manifest.json','study/candidate-freeze.json',audit_dir+'/native-history-audit.json',audit_dir+'/native-history-replay.json']
(HERE/'report-source.json').write_text(json.dumps({'source_file_sha256':{name:hashlib.sha256(source_bytes(name)).hexdigest() for name in names},
    'report_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'patient_derived_development_cases':1,
    'human_pretraining_patients':0,'optimizer_seeds_are_not_independent_patients':True,'final_worlds_used':False},indent=2)+'\n')
print(HERE/'RESULT.md')
