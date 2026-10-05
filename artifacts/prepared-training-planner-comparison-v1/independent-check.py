#!/usr/bin/env python3
"""Saved-only V1 audit. Never imports project/runtime/model code or decodes a case.

An authoritative terminal notification is required before --terminal-confirmed.
Failure/unknown method outcomes remain null, even if a simulator prefix exists.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/prepared-training-planner-comparison-v1'
COMMIT = '981f6c9c0d94291db5fc199b3ec145e5e9bb15d0'
ARCHIVE_SHA = '4476c1fc82015da9035511aafd41ad5857d1b115a50bb9467db77c422781b9b5'
DECL_SHA = '9e4447bbad9f6d77040bfcd51e1d446abb6f57c1821e1fdfa3b2096994e7fc21'
SUBJECTS = ['sub-PAT05','sub-PAT16','sub-PAT20','sub-PAT22','sub-PAT25','sub-PAT28']
ATTEMPTED = ['sub-PAT05','sub-PAT22','sub-PAT25','sub-PAT28']
BLOCKED = ['sub-PAT16','sub-PAT20']
METHODS = ['STOP','frozen_il','greedy_search']
SETTINGS = dict(arm_native_preview_entries=468, arm_seconds=90., device='cpu',
    max_rss_bytes=6442450944, max_steps=3, max_wall_seconds=600., optimizer_updates=0, torch_threads=1)
POLICY_HASH = 'sha256:74e90d9487dff6fe9db83c92e3887f15533e0499abaac386a73bbd7d4d566b4e'
COMMON = 'artifacts/pat05-real-geometric-learning-v1/declaration-input.json'
HISTORICAL = 'artifacts/remaining-training-frozen-spatial-float64-v1'
TRACKED = {}
CURRENT_SOURCE_DIFFS = {}


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    actual = hash_file(path)
    require(expected is None or actual == expected, 'byte binding: ' + str(path))
    TRACKED[str(path.relative_to(ROOT))] = actual
    return actual


def parse(raw):
    def reject(value):
        raise ValueError('nonfinite JSON ' + value)
    return json.loads(raw, parse_constant=reject)


def read(path, expected=None):
    bind(path, expected)
    return parse(Path(path).read_bytes())


def near(a, b, label, tolerance=1e-12):
    require(type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance), label)


def det3(matrix):
    a,b,c = [row[:3] for row in matrix[:3]]
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])


def original_records(declaration):
    originals = {}
    grid_receipt_path = 'artifacts/pat05-real-geometric-learning-v1/receipt.json'
    historical_files = declaration['input_closure']['pat05_historical_files']
    for path, sha in historical_files.items():
        bind(ROOT/path, sha)
    common = read(ROOT/COMMON, historical_files[COMMON])
    grid = read(ROOT/grid_receipt_path, historical_files[grid_receipt_path])['initial_task_metrics']['native_grid_reconciliation']
    expected = common['member']['expected_native_grid_binding']
    require({k:grid[k] for k in expected} == expected, 'PAT05 complete grid subset')
    originals['sub-PAT05'] = {**common, 'complete_native_grid_binding': {
        'schema':'pat05-complete-native-grid-binding-v1',
        'receipt':{'path':grid_receipt_path, 'sha256':historical_files[grid_receipt_path],
                   'json_pointer':'/initial_task_metrics/native_grid_reconciliation'},
        'grid_sha256':digest(grid), 'complete_grid':grid}}
    bind(ROOT/HISTORICAL/'completed-run.tar.gz')
    with tarfile.open(ROOT/HISTORICAL/'completed-run.tar.gz') as archive:
        for subject in SUBJECTS[1:]:
            path = ROOT/HISTORICAL/subject/'preparation.json'
            row = read(path) if path.exists() else parse(archive.extractfile(subject+'/preparation.json').read())
            require(row['role']=='TRAIN' and row['subject']==subject, 'original identity')
            require(row['binding_hash']=='sha256:'+digest(row['binding']), 'original binding digest')
            originals[subject] = row
    for subject, row in originals.items():
        require(digest(row)==declaration['input_closure']['original_records'][subject], subject+' frozen original')
        member = row['member'] if subject=='sub-PAT05' else row['binding']['member']
        bind(ROOT/member['case_bundle'], member['case_bundle_sha256'])  # opaque bytes only
    return originals


def source_checks(declaration):
    archive_record = read(ART/'source-archive.json')
    require(archive_record['source_commit']==COMMIT and archive_record['archive_sha256']==ARCHIVE_SHA, 'archive record')
    sources = declaration['source_sha256']
    require(len(sources)==70 and archive_record['files_sha256']==sources, 'exact 70 source inventory')
    bind(ART/'source.tar.gz', ARCHIVE_SHA)
    with tarfile.open(ART/'source.tar.gz') as archive:
        require(archive.pax_headers.get('comment')==COMMIT, 'archive commit PAX')
        files = [m for m in archive.getmembers() if m.isfile()]
        require(len(files)==70 and {m.name for m in files}==set(sources), 'closed archive membership')
        for member in files:
            raw = archive.extractfile(member).read()
            require(hashlib.sha256(raw).hexdigest()==sources[member.name], 'archive member '+member.name)
            git_raw = subprocess.run(['git','show',COMMIT+':'+member.name], cwd=ROOT,
                                     check=True, capture_output=True).stdout
            require(raw==git_raw, 'committed bytes '+member.name)
            # Root authorized V2 workspace repairs only after V1 terminated.
            current = hash_file(ROOT/member.name)
            if current != sources[member.name]:
                CURRENT_SOURCE_DIFFS[member.name] = {'executed':sources[member.name], 'current':current}
            bind(OUT/'source-snapshot'/member.name, sources[member.name])
    cp = declaration['checkpoint']
    require(cp['parameter_hash']==POLICY_HASH, 'frozen parameter identity')
    bind(ROOT/cp['path'], cp['sha256']); bind(OUT/'frozen-checkpoint.pt', cp['sha256'])
    anchor = read(ROOT/'artifacts/pat05-real-visited-imitation-v1/output-sha256.json')
    require(anchor['augmented_latest.pt']==cp['sha256'], 'checkpoint origin')


def profile_totals(profile):
    totals = Counter()
    for phase, row in profile['phases'].items():
        for key in ('started','returned','raised','feasible','rejected','returned_microsteps'):
            require(type(row[key]) is int and row[key]>=0, 'profile counter '+phase+key)
            totals[key] += row[key]
        require(row['started']==row['returned']+row['raised'], 'profile completion '+phase)
        require(row['returned']==row['feasible']+row['rejected'], 'profile disposition '+phase)
        require(sum(row['rejection_reasons'].values())==row['rejected'], 'rejection histogram')
        require(row['seconds']>=0, 'profile time')
    return dict(totals)


def preparation_checks(preparation, original, subject):
    require(preparation['subject']==subject and preparation['role']=='TRAIN', 'preparation identity')
    require(preparation['original_record_hash']==digest(original), 'original preparation record')
    if preparation['status']!='prepared':
        return {'status':preparation['status'], 'accepted_method_outcomes':None}
    access = preparation['access_preparation']
    original_binding = preparation['original_initial_binding']
    selected_binding = preparation['selected_initial_binding']
    if subject=='sub-PAT05':
        historical=read(ROOT/'artifacts/pat05-real-geometric-learning-v1/receipt.json')['initial_task_metrics']
        expected_source,expected_model=historical['source_hash'],historical['decision_model_hash']
        common,member=original,original['member']
    else:
        binding=original['binding'];common,member=binding['common_task'],binding['member']
        expected_source,expected_model=binding['native_source_hash'],binding['decision_model_hash']
    require(original_binding['source_hash']==expected_source and original_binding['model_hash']==expected_model,
        'original source/model matches independently pinned prior preparation')
    require(original_binding['reward']==common['objective'], 'original declared reward')
    require(access['original_access']==member['access'], 'original declared aperture')
    require(access['status']=='selected' and access['subject']==subject and access['role']=='TRAIN', 'access selected')
    require(access['original_source_hash']==original_binding['source_hash']
        and access['original_model_hash']==original_binding['model_hash']
        and access['selected_source_hash']==selected_binding['source_hash'], 'source/model access join')
    require(original_binding['reward']==selected_binding['reward']==access['reward']
        and original_binding['max_steps']==selected_binding['max_steps']==access['horizon']==3, 'common objective/horizon')
    require(access['native_preview_calls']==0 and access['selected_inventory_constructed'] is False
        and access['automatic_fallback'] is False and preparation['automatic_fallback'] is False, 'preparer scope')
    screen = access['screening']; exits = screen['exits']
    require(screen['complete'] is True and len(exits)==6, 'complete six exit screen')
    require({(r['axis'],r['outward_sign']) for r in exits}==set(itertools.product(range(3),(-1,1))), 'six exit identities')
    derived = {(r['axis'],r['outward_sign']):r for r in access['derived_exits']}
    count = 0
    for row in exits:
        require(row['status']=='screened' and row['distance_mm']==derived[(row['axis'],row['outward_sign'])]['distance_mm'], 'exit disposition/distance')
        require(row['eligible']==any(p['admissible'] for p in row['poses']), 'ANY entry eligibility')
        count += sum(p['duplicate_of'] is None for p in row['poses'])
    require(count==screen['screen_count'] and count<=468, 'static calls')
    winner = min((r for r in exits if r['eligible']), key=lambda r:(r['distance_mm'],r['axis'],r['outward_sign']))
    chosen = [winner['axis'],winner['outward_sign']]
    require(screen['selected_exit']==access['selected_exit']==chosen, 'deterministic selected access')
    require(access['selected_access']==derived[tuple(chosen)]['access'], 'selected aperture')
    inventory=preparation['selected_inventory']
    require(inventory['complete'] is True and inventory['declared_slots']==78, 'selected inventory bound')
    require(inventory['source_hash']==selected_binding['source_hash']
        and inventory['decision_model_hash']==selected_binding['model_hash']
        and inventory['steps_taken']==0 and inventory['remaining_steps']==3, 'selected inventory source/horizon')
    totals=profile_totals(preparation['shared_native_preview_profile'])
    return {'status':'prepared', 'selected_exit':chosen, 'static_checks':count,
        'shared_preview_totals':totals, 'initial_binding':selected_binding,
        'shared_preparation_seconds':preparation['shared_preparation_seconds']}


def episode_accounting(episode, initial_binding, original, subject):
    decisions=episode['decisions']; metrics=episode.get('metrics') or episode.get('failure_metrics')
    if not metrics:
        return {'durable_terminal':False, 'accepted_outcomes':None}
    history=metrics['history']
    member=original['member'] if subject=='sub-PAT05' else original['binding']['member']
    expected_grid=(original['complete_native_grid_binding']['complete_grid'] if subject=='sub-PAT05'
        else member['expected_native_grid_binding'])
    require(metrics['native_grid_reconciliation']==expected_grid, 'original complete frame preserved')
    require(metrics['support_provenance']['acknowledgment']==member['research_support_acknowledgment'],
        'original provisional support preserved')
    require(metrics['support_provenance']['cortical_access_permitted'] is False
        and metrics['support_provenance']['clinical_use_permitted'] is False, 'no support approval promotion')
    common=original if subject=='sub-PAT05' else original['binding']['common_task']
    tool_ids={r['tool_id'] for r in common['tools']}
    require(all(r['action_id']=='STOP' or r['tool_id'] in tool_ids for r in history), 'original tool IDs')
    require(metrics['source_hash']==initial_binding['source_hash'] and metrics['decision_model_hash']==initial_binding['model_hash'], 'episode source/model')
    require(episode['attempted_actions']==len(decisions), 'decision count')
    require(episode['committed_transitions']==len(history)==metrics['steps'], 'committed history count')
    require(episode['behavior_parameter_hash']==POLICY_HASH, 'recorded behavior identity')
    returned=[r for r in decisions if r['status']=='returned']
    require(len(returned)==len(history), 'all stored transitions returned')
    for decision, record in zip(returned, history):
        index=decision['selected_index']
        require(decision['action_ids'][index]==decision['action_id']==record['action_id'], 'selected action ID')
        require(decision['action_mask'][index] is True, 'selected action legal')
        require(decision['committed_info']==record, 'decision/full history join')
        near(decision['reward'], record['reward'], 'decision reward')
    require(not decisions or decisions[0]['observation_hash']==initial_binding['observation_hash'], 'initial observed task')
    require(not decisions or decisions[0]['action_ids']==initial_binding['action_ids'], 'initial inventory identical')
    if metrics['terminated']:
        require(1<=len(history)<=3 and (history[-1]['action_id']=='STOP' or len(history)==3), 'terminal completeness')
        require(all(r['action_id']!='STOP' for r in history[:-1]), 'STOP terminal only')
    reward=math.fsum(r['reward'] for r in history)
    near(metrics['total_reward'], reward, 'history return')
    if 'simulated_return' in episode: near(episode['simulated_return'], reward, 'simulator return retained')
    affine=metrics['native_grid_reconciliation']['derived_affine_ras_mm']
    volume=abs(det3(affine)); removed=set(); contact=set(); last_tool=None; changes=0; distance=0.; target=normal=0.
    weights=initial_binding['reward']
    for row in history:
        cells={tuple(cell) for cell in row.get('removed_indices_native',[])}
        require(len(cells)==len(row.get('removed_indices_native',[])) and not removed&cells, 'unique removal')
        if row['action_id']=='STOP':
            require(not cells and not row.get('microsteps'), 'STOP no removal')
            near(row['reward'],0.,'STOP reward'); continue
        contacts={tuple(cell) for cell in row['contact_indices_native']}
        microcontacts={tuple(cell) for step in row['microsteps'] for cell in step['contact_indices_native']}
        require(contacts==microcontacts, 'contact union')
        d=math.dist(row['entry_mm'],row['tip_mm'])
        change=int(last_tool is not None and last_tool!=row['tool_id'])
        reconstructed=(weights['target_per_mm3']*row['target_removed_mm3']-weights['normal_per_mm3']*row['normal_removed_mm3']
            -weights['action_cost']-2*weights['motion_per_mm']*d-weights['tool_change_cost']*change)
        near(row['reward'],reconstructed,'declared reward arithmetic',1e-7)
        near(row['target_removed_mm3']+row['normal_removed_mm3'],len(cells)*volume,'tissue volume arithmetic',1e-7)
        removed.update(cells);contact.update(contacts);target+=row['target_removed_mm3'];normal+=row['normal_removed_mm3']
        distance+=2*d;changes+=change;last_tool=row['tool_id']
    for key,value in [('simulated_removed_volume_mm3',len(removed)*volume),
        ('cumulative_contacted_tissue_upper_bound_mm3',len(contact)*volume),
        ('currently_retained_contacted_tissue_upper_bound_mm3',len(contact-removed)*volume)]:
        near(metrics[key],value,'source cell accounting '+key,1e-7)
    # Keep any float32 per-action versus aggregate target reduction residual; no label arrays are opened.
    return {'durable_terminal':metrics['terminated'], 'actions':[r['action_id'] for r in history],
        'recorded_simulator_return_unaudited':reward, 'removed_cell_union':len(removed),
        'contact_cell_union':len(contact), 'retained_contact_cells':len(contact-removed),
        'per_action_target_minus_aggregate_mm3':target-metrics['target_removed_mm3'],
        'per_action_normal_minus_aggregate_mm3':normal-metrics['normal_removed_mm3'],
        'path_length_mm':distance, 'tool_changes':changes, 'accepted_outcomes':None}


def arm_checks(row, directory, method, expected, original, subject):
    require(row==read(directory/(method+'-arm.json')), 'arm receipt copy')
    require(row['name']==method and row['optimizer_updates']==0 and row['initial_binding']==expected, 'arm initial binding')
    require(row['initial_parameter_hash']==POLICY_HASH, 'arm initial parameters')
    budget=row['planning_budget']
    require(budget['limits']=={'native_preview_entries':468,'planning_execution_seconds':90.}, 'arm frozen limits')
    require(budget['score'] is None and budget['cache_hits'] is None, 'no inferred score/cache count')
    totals=profile_totals(budget['native_preview_profile'])
    if budget['counting_reliable']:
        require(budget['native_preview_entries']==totals.get('started',0)<=468, 'arm actual entry counting')
    else:
        require(budget['native_preview_entries'] is None, 'unknown accounting stays unknown')
    result={'status':row['status'], 'failure':row.get('failure'), 'planning_budget_status':budget['status'],
        'native_preview_entries':budget['native_preview_entries'], 'elapsed_online_seconds':budget['elapsed_seconds'],
        'full_arm_seconds_including_audit':row['full_arm_seconds_including_audit'],
        'independent_audit_seconds':row.get('independent_audit_seconds'), 'accepted_outcomes':None,
        'final_runtime_parameter_hash_recorded':row.get('final_parameter_hash')}
    episode_path=directory/(method+'.json')
    episode=read(episode_path) if episode_path.exists() else None
    if episode:
        result['saved_episode_accounting']=episode_accounting(episode,expected,original,subject)
    if row['status']!='complete':
        require(row['outcomes'] is None and row.get('independent_evaluation_accepted') is not True, 'failed arm not promoted')
        return result
    require(episode and episode['status']=='complete', 'complete episode exists')
    require(row['closure_verified'] is True and row['final_initial_binding']==expected, 'unchanged base task')
    require(row['final_parameter_hash']==POLICY_HASH, 'final parameters unchanged')
    require(budget['status']=='complete_history_awaiting_independent_audit'
        and budget['history_complete_caller_attestation'] is True and budget['counting_reliable'] is True
        and budget['failure'] is None and budget['blocked_preview_attempts']==0
        and budget['time_overshoot_seconds']==0 and 0<=budget['elapsed_seconds']<=90., 'complete budget gate')
    terminal=read(directory/row['terminal_history_path'],row['terminal_history_sha256'])
    bind(episode_path,row['episode_sha256'])
    require(terminal['status']=='awaiting_independent_check' and terminal['metrics']==episode['metrics'], 'durable terminal binding')
    require(digest(terminal['metrics'])==row['terminal_metrics_hash'], 'terminal metrics hash')
    audit=episode['independent_evaluation']
    require(audit['accepted'] is True and audit['complete_episode'] is True and audit['geometry']['feasible'] is True, 'accepted full audit')
    require(audit['committed_history_hash']=='sha256:'+digest(episode['metrics']['history']), 'audit history')
    require(row['outcomes']==audit['outcomes'] and row['independent_evaluation_accepted'] is True, 'audited outcomes')
    require(row['actions']==[d['action_id'] for d in episode['decisions']], 'actions list')
    near(row['actor_decision_seconds'],sum(d['decision_seconds'] for d in episode['decisions']),'decision timings')
    near(row['native_transition_seconds'],sum(d['transition_seconds'] for d in episode['decisions']),'transition timings')
    near(row['outcomes']['total_reward'],episode['metrics']['total_reward'],'audited return',1e-7)
    result['accepted_outcomes']=row['outcomes']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-confirmed',required=True,help='Root session and terminal status; never inferred from files')
    args=parser.parse_args();started=time.perf_counter()
    result={'schema':'prepared-training-planner-saved-audit-v1','terminal_authority':args.terminal_confirmed,
        'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'status':'checking',
        'fixed_tolerances':{'json_hash_join':'exact','scalar_sum_relative_and_absolute':1e-12,
            'cell_volume_reward_relative_and_absolute':1e-7},
        'limits':['No patient arrays, checkpoint deserialization, model/native calls or refits.',
            'Saved certificates and source bindings authenticated; complete-tool geometry is not rerun.',
            'Stored target memberships are not independently reconstructed without dense labels.',
            'No runtime final-weight equality claim where failure omitted final_parameter_hash.',
            'Worker RSS is sampled, not hard OS enforcement; no total-process wall/RSS bound.',
            'Preparation, online phases and audit times overlap and must not be added indiscriminately.',
            'No clinical efficacy or learned adaptation is tested.']}
    try:
        declaration_path=ROOT/'manifests/experiments/prepared-training-planner-comparison-v1.json'
        declaration=read(declaration_path,DECL_SHA); release=read(ART/'release.json')
        require(declaration['subjects']==SUBJECTS and declaration['attempted']==ATTEMPTED
            and declaration['historical_blocks']==BLOCKED and declaration['methods']==METHODS
            and declaration['settings']==SETTINGS,'frozen design')
        require(release['released'] is True and release['source_commit']==COMMIT
            and release['source_archive_sha256']==ARCHIVE_SHA and release['declaration']['sha256']==DECL_SHA
            and release['settings']==SETTINGS,'release joins')
        bind(OUT/'declaration-input.json',DECL_SHA)
        source_checks(declaration); originals=original_records(declaration)
        cohort=read(ROOT/'manifests/experiments/btc-spatial-development-cohort-v1.json',declaration['cohort_sha256'])
        roles={r['subject']:r['development_role'] for key in ('existing_development_records','candidates') for r in cohort[key]}
        require(all(roles[s] in ('previously_consulted_training_and_method_development','population_training')
            for s in SUBJECTS),'all six original TRAIN roles')
        result['cohort_sha256']=declaration['cohort_sha256']
        index=read(OUT/'output-sha256.json')
        actual={str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()
            and 'source-snapshot' not in p.parts and p.name!='output-sha256.json'}
        require(set(index)==actual,'complete retained output index')
        for path,sha in index.items(): bind(OUT/path,sha)
        summary=read(OUT/'summary.json')
        require(summary['patients_prescribed']==6 and [r['subject'] for r in summary['patients']]==SUBJECTS,'six rows in order')
        require(summary['optimizer_updates']==0 and summary['adaptation'] is False
            and summary['clinical_efficacy_measured'] is False,'summary scope')
        cases=[];pairs={}
        for summary_row in summary['patients']:
            subject=summary_row['subject'];directory=OUT/subject;row=read(directory/'receipt.json')
            require(row=={k:v for k,v in summary_row.items() if k not in ('parent_closure','supervisor','comparison_eligible')},'summary receipt join')
            require(row['role']=='TRAIN' and row['optimizer_updates']==0 and set(row['arms'])==set(METHODS),'patient method/role')
            if subject in BLOCKED:
                require(row['status']=='historical_support_block' and row['patient_arrays_opened'] is False
                    and row['original_preparation']==originals[subject] and row['original_record_hash']==digest(originals[subject]),'historical support block')
                require(summary_row['supervisor'] is None and summary_row['parent_closure'] is None
                    and summary_row['comparison_eligible'] is False,'blocked no worker')
                require(all(a['status']=='not_executed' and a['outcomes'] is None for a in row['arms'].values()),'blocked null arms')
                cases.append({'subject':subject,'status':row['status'],'comparison_eligible':False});continue
            supervisor=read(directory/'supervisor.json');closure=read(directory/'closure-check.json')
            require(supervisor==summary_row['supervisor'] and closure==summary_row['parent_closure'],'parent joins')
            require(supervisor['declaration_sha256']==DECL_SHA and supervisor['automatic_retry'] is False,'supervisor identity/no retry')
            supervisor_ok=(supervisor['status']=='complete' and supervisor['returncode']==0
                and supervisor['termination_reason'] is None and supervisor['timed_out'] is False)
            if supervisor_ok:
                require(supervisor['seconds']<=600 and supervisor['sampled_peak_rss_bytes']<=6*1024**3,'worker reported resource bounds')
            preparation=read(directory/'preparation.json',row.get('preparation_sha256'))
            prep_result=preparation_checks(preparation,originals[subject],subject)
            expected=preparation.get('selected_initial_binding')
            arms={name:arm_checks(arm,directory,name,expected,originals[subject],subject) if arm['status'] not in ('not_executed','running')
                else {'status':arm['status'],'accepted_outcomes':None} for name,arm in row['arms'].items()}
            eligible=(supervisor_ok and row['status']=='complete' and row.get('closure_verified') is True
                and closure=={'sources_inputs_checkpoint_unchanged':True}
                and all(arm['accepted_outcomes'] is not None for arm in arms.values()))
            require(eligible==summary_row['comparison_eligible'],'independent eligibility')
            if eligible:
                pairs[subject]={left+' minus '+right:{key:arms[left]['accepted_outcomes'][key]-arms[right]['accepted_outcomes'][key]
                    for key in ('total_reward','target_removed_mm3','normal_removed_mm3')}
                    for left,right in itertools.combinations(METHODS,2)}
            cases.append({'subject':subject,'status':row['status'],'comparison_eligible':eligible,
                'preparation':prep_result,'arms':arms,'supervisor':supervisor,'closure':closure})
        require(summary['complete_comparisons']==len(pairs),'eligible denominator')
        result.update(status='saved_records_verified',study_complete_comparisons=len(pairs),
            patients_prescribed=6,historical_blocks=2,patients=cases,valid_complete_pair_differences=pairs,
            optimizer_updates_recorded=0,source_files_verified=70,source_commit=COMMIT,source_archive_sha256=ARCHIVE_SHA)
    except BaseException as error:
        result.update(status='independent_check_failed',failure={'type':type(error).__name__,'message':str(error)})
    result['bound_inputs_sha256']=dict(TRACKED)
    result['current_workspace_source_differences_after_authorized_v2_repair']=CURRENT_SOURCE_DIFFS
    result['bound_inputs_unchanged_after_check']=all(hash_file(ROOT/p)==h for p,h in TRACKED.items())
    result['checker_sha256']=hash_file(__file__);result['elapsed_seconds']=time.perf_counter()-started
    # Append attempts within the one assigned receipt; preserve the prospective contract and every failure.
    destination=ART/'independent-check.json'
    prior=parse(destination.read_bytes()) if destination.exists() else {'attempts':[]}
    prior.setdefault('attempts',[]).append(result)
    destination.write_text(json.dumps(prior,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','bound_inputs_unchanged_after_check')}
        | {'failure':result.get('failure'),'complete_comparisons':result.get('study_complete_comparisons')}))
    raise SystemExit(0 if result['status']=='saved_records_verified' and result['bound_inputs_unchanged_after_check'] else 1)


if __name__=='__main__':main()
