"""Scalar saved-record derivation only; never imports the project or opens arrays."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
pins = {}
def read(relative):
    raw = (ROOT / relative).read_bytes()
    pins[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

sel = 'build/obstruction-opening-select013-comparison-v1'
new = 'build/remind-post-exposure-feasibility-v1'
sc = read(sel + '/attempt-01/costs.json')
sr = read(sel + '/attempt-01/result.json')
sp = read(sel + '/attempt-01.supervision/receipt.json')
si = read(sel + '/source-index.json')
sl = read(sel + '/root-release.json')
nr = read(new + '/attempt-01/result.json')
np = read(new + '/attempt-01.supervision/receipt.json')
ni = read(new + '/source-index.json')
nl = read(new + '/root-release.json')
assert sr['status'] == 'complete_union_obstruction_SELECT013_comparison'
assert nr['status'] == 'post_exposure_batch_complete'
for receipt, result_path, release_path in [(sp,sel+'/attempt-01/result.json',sel+'/root-release.json'),
                                           (np,new+'/attempt-01/result.json',new+'/root-release.json')]:
    assert receipt['exit_code'] == 0
    assert receipt['result_sha256'] == pins[result_path]
    assert receipt['release_sha256'] == pins[release_path]
rows = []
for arm in ('IL64','RL8','observed_greedy','observed_beam'):
    construction = sc['phases'][arm+'.public_construction']
    run = sc['phases'][arm+'.planning_and_replay']
    row = {'arm':arm, 'construction_wall_seconds':construction['complete_wall_seconds'],
        'initial_preview_count':construction['native_preview_entries'],
        'initial_inventory_request_inclusive_seconds':construction['inventory_request_inclusive_seconds'],
        'collection_and_replay_wall_seconds':run['complete_wall_seconds'],
        'collection_and_replay_preview_count':run['native_preview_entries'],
        'collection_and_replay_inventory_seconds':run['inventory_request_inclusive_seconds'],
        'policy_forward_count':run.get('policy_forward_calls',0),
        'policy_forward_seconds':run.get('policy_forward_inclusive_seconds',0),
        'public_return':sr['arms'][arm]['public_return'],
        'constructor_plus_collection_replay_wall_seconds':construction['complete_wall_seconds']+run['complete_wall_seconds']}
    if arm.startswith('observed_'):
        selection = read(sel+'/attempt-01/'+arm+'/selection.json')
        row.update(selection_wall_seconds=selection['wall_seconds'],
                   selection_preview_count=selection['native_previews'],
                   constructor_plus_selection_wall_seconds=construction['complete_wall_seconds']+selection['wall_seconds'])
        if arm == 'observed_greedy':
            row['legal_actions_scored'] = selection['accounting']['evaluated_nonstop_actions']
    rows.append(row)

cases = []
for row in nr['cases']:
    a = row['greedy_accounting']; marks = {x['phase']:x['seconds'] for x in row['phase_marks']}
    initial = row['emitted_count']
    assert (row['native_preview_entries']-initial) % 2 == 0
    successor = (row['native_preview_entries']-initial)//2
    cases.append({'subject':row['patient_id'],'steps_including_STOP':row['steps_taken'],
        'initial_candidates_previewed':initial,'initial_legal':row['accepted_count'],
        'successor_previews_per_planning_or_execution':successor,
        'selection_total_previews_including_constructor':initial+successor,
        'all_phase_previews':row['native_preview_entries'],
        'before_initial_task_seconds':marks['before_admitted_task_initial_inventory'],
        'initial_task_inventory_observation_outputs_window_seconds':marks['case_inventory_retained']-marks['before_admitted_task_initial_inventory'],
        'greedy_seconds_excluding_initial_preparation':a['planning_seconds'],
        'nominal_actions_scored':a['evaluated_nonstop_actions'],
        'independent_audit_seconds':row['independent_episode_seconds'],
        'case_wall_seconds':row['elapsed_seconds'],
        'target_removed_mm3':row['full_route_outcomes']['target_removed_mm3'],
        'outside_target_removed_mm3':row['full_route_outcomes']['normal_removed_mm3']})
p = nr['native_budget']['native_preview_profile']['phases']
assert sum(c['initial_candidates_previewed'] for c in cases) == p['unclassified']['started'] == 361
assert sum(c['successor_previews_per_planning_or_execution'] for c in cases) == p['planning']['started'] == p['execution']['started'] == 2231
planning = sum(c['greedy_seconds_excluding_initial_preparation'] for c in cases)
preview = sum(x['seconds'] for x in p.values())
aggregates = {'initial_previews':361,'successor_planning_previews':2231,'authoritative_execution_previews':2231,
    'selection_previews_including_constructor':2592,'all_phase_previews':4823,
    'initial_native_preview_seconds':p['unclassified']['seconds'],
    'planning_native_preview_seconds':p['planning']['seconds'],
    'execution_native_preview_seconds':p['execution']['seconds'],
    'selection_native_preview_seconds_including_constructor':p['unclassified']['seconds']+p['planning']['seconds'],
    'greedy_seconds_excluding_initial_preparation':planning,
    'greedy_nonpreview_remainder_seconds':planning-p['planning']['seconds'],
    'planning_preview_fraction':p['planning']['seconds']/planning,
    'all_preview_seconds':preview,'worker_seconds':nr['elapsed_seconds'],
    'worker_nonpreview_remainder_seconds':nr['elapsed_seconds']-preview,
    'all_preview_fraction_of_worker':preview/nr['elapsed_seconds'],
    'initial_task_inventory_observation_outputs_window_seconds':sum(c['initial_task_inventory_observation_outputs_window_seconds'] for c in cases),
    'parent_seconds':np['elapsed_seconds'],'parent_sampled_peak_rss_bytes':np['sampled_peak_rss_bytes'],
    'nominal_actions_scored':sum(c['nominal_actions_scored'] for c in cases),
    'decisions_including_STOP':sum(c['steps_including_STOP'] for c in cases),
    'independent_audit_seconds':sum(c['independent_audit_seconds'] for c in cases)}
# Pin exactly the executed source snapshots; read Git objects, not live patient sources.
source_records = []
for index, release, key in ((si,sl,'source_files'),(ni,nl,'files')):
    head = release['expected_head']
    names = ['src/resectionlab/native_spatial_task.py','src/resectionlab/native_proposals.py',
             'src/resectionlab/spatial_policy.py','src/resectionlab/planning_budget.py',
             'src/resectionlab/patient_select_inference.py','src/resectionlab/patient_planning_preflight.py']
    for name in names:
        expected = index[key].get(name)
        if expected is None: continue
        raw = subprocess.run(['git','show',head+':'+name],cwd=ROOT,check=True,capture_output=True).stdout
        assert hashlib.sha256(raw).hexdigest() == expected
        source_records.append({'head':head,'path':name,'sha256':expected})
result = {'scope':'saved_scalar_and_source_analysis_only_no_array_model_or_native_execution',
    'SELECT013':{'executed_head':sl['expected_head'],'rows':rows,'worker_seconds':sr['wall_seconds'],
        'parent_seconds':sp['elapsed_seconds'],'all_arms_STOP':all(x['actions']==['STOP'] for x in sr['arms'].values()),
        'timing_warning':'Construction and collection/replay are separate outer scopes; inner inventory, transition and forward times must not all be summed. Cold/warm order is uncontrolled.'},
    'new_four':{'executed_head':nl['expected_head'],'cases':cases,'totals':aggregates,
        'timing_warning':'Preview phases are disjoint labels. Greedy time includes planning previews; its nonpreview remainder is an upper bound on pure nominal scoring, not its isolated duration. Per-case successor split follows deterministic identical full plan/execution and aggregate exact equality; no per-case native time profile exists.'},
    'executed_source_pins':source_records,'input_pins':pins}
(HERE/'scalars.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'SELECT':rows,'new_four':aggregates},indent=2))
