"""Prepared saved-only eight-update audit. Run only after root terminal notice.
No project imports, tensor/checkpoint loading, patient reconstruction or replay.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import time

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1'
BASE=ROOT/'build/cross-patient-planning-v1/fixed-four-train-pilot-v1'
OUT=Path(__file__).parent
SUBJECTS=['ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025']
EXPECTED_RELEASE='c1cde8eb92f9df5d85ccb91ed0e591adb1da70e011582f3be116d237fecf72db'
EXPECTED_HEAD='137bdfca654c2ad843385382efdc539044eb01b1'
READ_CAP=8*1024**2
seen={}; checks=0

def require(ok,message):
    global checks
    checks+=1
    if not ok: raise AssertionError(message)

def raw(path):
    path=Path(path)
    require(path.is_relative_to(ROOT) and '..' not in path.parts,'contained read path')
    require(path.suffix in ('.json','.py'),'JSON/source only; arrays/checkpoints refused')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        st=os.fstat(fd)
        require(stat.S_ISREG(st.st_mode) and st.st_size<=READ_CAP,'bounded regular file')
        with os.fdopen(fd,'rb',closefd=False) as stream: payload=stream.read(READ_CAP+1)
    finally: os.close(fd)
    require(len(payload)<=READ_CAP,'read cap')
    h=hashlib.sha256(payload).hexdigest(); relative=str(path.relative_to(ROOT))
    require(relative not in seen or seen[relative]['sha256']==h,'unchanged repeated input '+relative)
    seen[relative]={'sha256':h,'bytes':len(payload)}
    return payload

def read(path):
    return json.loads(raw(path),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def sha(path): return hashlib.sha256(raw(path)).hexdigest()
def digest(value): return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
def history_identity(rows): return [{k:v for k,v in r.items() if k!='outcome_scope'} for r in rows]
def saved_write(name,value):
    with (OUT/name).open('x') as stream: json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')

def inspect_visit(run,phase,subject,context,final_parameters,updates):
    directory=run/phase/subject
    sealed=read(directory/'plan.json'); plan=sealed['plan']
    trace=read(directory/'complete-trace.json'); replay=read(directory/'native-replay.json')
    metrics=replay['metrics']; audit=replay['independent_geometry']; history=metrics['history']; decisions=trace['decisions']
    actions=plan['actions']; count=len(actions)
    require(digest(plan)==sealed['plan_seal'],'exact plan seal '+phase+'/'+subject)
    require(plan['context_hash']==trace['context_hash']==digest(context),'complete context join')
    require(plan['history']==trace['metrics']['history'] and history_identity(plan['history'])==history_identity(history),'exact nominal/replay history except scope')
    require(digest(history)==audit['committed_history_hash'] and audit['accepted'] is True and audit['complete_episode'] and audit['geometry']['complete_tool_checked'] and audit['geometry']['failures']==[],'saved independent full-tool audit')
    require(count==len(history)==len(decisions)==metrics['steps'] and metrics['terminated'],'complete history count')
    require(audit['geometry']['action_count']==sum(a!='STOP' for a in actions),'geometry excludes STOP')
    require(actions==[x['action_id'] for x in history]==[x['action_id'] for x in decisions],'action identity')
    require(plan['terminal_reason']==('STOP' if actions[-1]=='STOP' else 'HORIZON') and (actions[-1]=='STOP' or count==24),'STOP/horizon completion')
    require(all(not x['terminated'] for x in decisions[:-1]) and decisions[-1]['terminated'],'only final transition terminates')
    behavior=decisions[0]['behavior_parameter_hash']
    require(all(x['behavior_parameter_hash']==behavior for x in decisions),'frozen collection behavior')
    require(digest({'context':trace['context_hash'],'observations':[x['observation_hash'] for x in decisions],'history':trace['metrics']['history'],'behavior_parameter_hash':behavior})==trace['trace_seal'],'complete trace seal')
    require(all(x['reward']==y['reward'] for x,y in zip(decisions,history)) and close(sum(x['reward'] for x in history),metrics['total_reward']),'saved reward sum')
    require(all(x['action_mask'][x['action_ids'].index(x['action_id'])] for x in decisions),'saved action legal mask')
    for field in ['target_removed_mm3','normal_removed_mm3']:
        require(close(sum(x.get(field,0) for x in history),metrics[field]) and close(metrics[field],audit['outcomes'][field]),'saved removal sums '+field)
    target=metrics['supplied_goal_region']
    require(target['fraction_denominator']=='entire_unchanged_supplied_region' and target['unsupported_region_removable'] is False and target['target_modified'] is False and target['occupancy_modified'] is False,'full unchanged target/support')
    require(close(target['fraction_of_full_region_removed'],metrics['target_removed_mm3']/target['full_region_membership_mm3']),'full target denominator')
    if phase.startswith('TRAIN-greedy/'):
        method=phase.split('/')[-1]
        require(plan['parameter_hash']==behavior==final_parameters[method] and plan['learning_updates']==updates,'fixed endpoint parameter identity')
    return {'phase':phase,'subject':subject,'steps':count,'actions':actions,'terminal':plan['terminal_reason'],'target_mm3':metrics['target_removed_mm3'],'outside_supplied_goal_mm3':metrics['normal_removed_mm3'],'public_return':metrics['total_reward'],'target_source_cells':audit['outcomes']['positive_target_source_cells_removed'],'target_fraction':target['fraction_of_full_region_removed'],'plan_seal':sealed['plan_seal'],'trace_seal':trace['trace_seal'],'behavior_parameter_hash':behavior,'plan_parameter_hash':plan['parameter_hash'],'learning_updates':plan['learning_updates']}

def main(expected_result):
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60 second saved audit cap')))
    signal.alarm(60)
    run=PREP/'attempt-01'; supervision=PREP/'attempt-01.supervision'
    require(len(expected_result)==64 and all(c in '0123456789abcdef' for c in expected_result),'terminal result SHA required')
    require(sha(run/'result.json')==expected_result and sha(PREP/'root-release.json')==EXPECTED_RELEASE,'root terminal identities')
    result=read(run/'result.json'); release=read(PREP/'root-release.json'); receipt=read(supervision/'receipt.json'); worker=read(supervision/'worker-final.json')
    require(release['expected_head']==EXPECTED_HEAD,'frozen run source commit')
    require(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] and not receipt['final_owned_pids'] and not receipt['cleanup_errors'] and receipt['stop_reason'] is None,'clean completed parent')
    require(receipt['release_sha256']==EXPECTED_RELEASE and receipt['result_sha256']==expected_result and worker['result_sha256']==worker['canonical_result_sha256']==expected_result,'parent/worker result joins')
    require(receipt['elapsed_seconds']<3600 and receipt['sampled_peak_rss_bytes']<3*1024**3 and receipt['output_bytes']<128*1024**2 and worker['wall_seconds']<3540,'actual resource caps')
    require(sha(ROOT/release['source_index']['path'])==release['source_index']['sha256'],'source inventory')
    index=read(ROOT/release['source_index']['path'])
    require(index['head']==EXPECTED_HEAD,'source index head')
    for group in ['source_files','metadata_files']:
        for path,h in index[group].items(): require(sha(ROOT/path)==h,'source/metadata pin '+path)
    baseline={key:read(ROOT/ref['path']) for key,ref in release['one_update_baseline'].items()}
    for key,ref in release['one_update_baseline'].items(): require(sha(ROOT/ref['path'])==ref['sha256'],'baseline pin '+key)
    require(result['status']=='complete_TRAIN_only_not_heldout_performance' and result['optimizer_updates']=={'IL':8,'RL':8} and result['TRAIN_subjects']==SUBJECTS,'complete eight-update four-TRAIN result')
    require(result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==0 and result['clinical_claim'] is False,'no held-out/private/clinical scope')
    require(result['initial_parameter_hash']==baseline['result']['initial_parameter_hash'],'same initial tensors')
    require(release['learning_protocol']=={**baseline['release']['learning_protocol'],'updates_per_method':8},'only protocol update count changes')
    require(release['cohort_limits']=={**baseline['release']['cohort_limits'],'max_optimizer_updates':16,'max_native_previews':3373440,'max_policy_forwards':2496},'only derived count limits change')
    config=read(run/'configuration.json'); costs=read(run/'costs.json'); ready=read(run/'teacher-readiness.json'); freeze=read(run/'checkpoint-freeze.json')
    require(config['learning_protocol']==release['learning_protocol'] and config['limits']==release['cohort_limits'] and digest(config['learning_protocol'])==release['learning_protocol_hash'],'configuration exact release')
    contexts={s:read(run/(s+'-context.json')) for s in SUBJECTS}
    require(all(contexts[s]['role']=='TRAIN' and contexts[s]['subject']==s for s in SUBJECTS),'fixed TRAIN contexts')
    final_parameters={m:result['checkpoints'][m]['parameter_hash'] for m in ['IL','RL']}
    visits={}; teachers=[]; endpoints=[]; curves={m:[] for m in ['IL','RL']}
    for s in SUBJECTS:
        row=inspect_visit(run,'teachers',s,contexts[s],final_parameters,8); visits[('teachers',s)]=row;teachers.append(row)
        search=read(run/'teachers'/s/'search.json')
        require(search['actions']==row['actions'] and not search['accounting']['call_cap_reached'] and not search['accounting']['time_cap_reached'],'complete uncapped teacher')
        require(ready[s]['status']=='complete_replayed' and ready[s]['plan_seal']==row['plan_seal'] and ready[s]['trace_seal']==row['trace_seal'] and close(ready[s]['public_return'],row['public_return']),'teacher readiness join')
        old=read(BASE/'attempt-01/teachers'/s/'native-replay.json')['metrics']
        require(row['actions']==[x['action_id'] for x in old['history']] and close(row['public_return'],old['total_reward']) and close(row['target_mm3'],old['target_removed_mm3']) and close(row['outside_supplied_goal_mm3'],old['normal_removed_mm3']),'same teacher physical outcome')
    require(any(row['public_return']>0 and any(a!='STOP' for a in row['actions']) for row in teachers),'retained declared teacher gate')
    for method in ['IL','RL']:
        before=result['initial_parameter_hash']
        for update in range(1,9):
            phase=f'{method}/update-{update:02d}'; u=read(run/phase/'update.json'); contributions=[]; exposures=[]
            for s in SUBJECTS:
                row=inspect_visit(run,phase,s,contexts[s],final_parameters,8);visits[(phase,s)]=row
                c=read(run/phase/s/'gradient-contribution.json');contributions.append(c);exposures.append({k:row[k] for k in ['subject','steps','public_return','target_mm3','outside_supplied_goal_mm3','target_source_cells']})
                require(c['context_hash']==digest(contexts[s]) and c['plan_seal']==row['plan_seal'] and c['trace_seal']==row['trace_seal'] and c['steps']==c['loss_forward_calls']==row['steps'],'complete per-case contribution')
                require(row['learning_updates']==update-1,'collection before current update')
                if method=='IL': require(row['trace_seal']==visits[('teachers',s)]['trace_seal'],'IL complete pinned teacher recollection')
                else: require(row['behavior_parameter_hash']==row['plan_parameter_hash']==before and close(c['return'],row['public_return']),'on-policy shared pre-update parameters and return')
            require(u['context_hashes']==[c['context_hash'] for c in contributions] and u['trace_seals']==[c['trace_seal'] for c in contributions],'all four ordered contributions')
            for field in ['loss','actor_loss','value_loss','entropy','loss_forward_calls']:require(close(u[field],sum(c[field] for c in contributions)),'contribution sum '+field)
            require(u['before_parameter_hash']==before and u['completed_updates']==update and u['optimizer_updates']==1 and u['patient_adaptation'] is False,'shared update chain')
            require(u['parameters_changed']==(u['after_parameter_hash']!=before),'parameter change record')
            if update==1:
                excluded={'context_hashes','trace_seals'}
                require({k:v for k,v in u.items() if k not in excluded}=={k:v for k,v in baseline[method+'_first'].items() if k not in excluded},'exact first-update numerical/tensor control '+method)
            before=u['after_parameter_hash']
            curves[method].append({'update':update,**{k:u[k] for k in ['loss','actor_loss','value_loss','entropy','loss_forward_calls','gradient_norm_before_clip','before_parameter_hash','after_parameter_hash']},'training_collection_exposure':exposures})
        require(before==final_parameters[method],'update8 fixed endpoint')
        f=freeze['checkpoints'][method]
        require(all(f[k]==v for k,v in result['checkpoints'][method].items()),'saved checkpoint metadata joins')
        require(f['metadata']['initial_parameter_hash']==result['initial_parameter_hash'] and f['metadata']['completed_updates']==8 and f['metadata']['TRAIN_subjects']==SUBJECTS and f['metadata']['parameter_hash']==before,'eight-update checkpoint lineage')
        for s in SUBJECTS:
            phase='TRAIN-greedy/'+method;row=inspect_visit(run,phase,s,contexts[s],final_parameters,8);visits[(phase,s)]=row
            endpoint=result['TRAIN_greedy'][method][s]
            require(endpoint['actions']==row['actions'] and endpoint['plan_seal']==row['plan_seal'] and endpoint['complete'] and endpoint['steps']==row['steps'] and close(endpoint['public_return'],row['public_return']),'final result endpoint join')
            old=read(BASE/'attempt-01'/phase/s/'native-replay.json')['metrics']
            row['one_update_endpoint']={'steps':old['steps'],'public_return':old['total_reward'],'target_mm3':old['target_removed_mm3'],'outside_supplied_goal_mm3':old['normal_removed_mm3']}
            row['eight_minus_one']={k:row[k]-row['one_update_endpoint'][k] for k in row['one_update_endpoint']}
            endpoints.append(row)
    control=read(supervision/'first-update-control.json')
    require(control['status']=='exact_first_update_control_passed' and sha(supervision/'first-update-control.json')==worker['first_update_control_sha256'],'worker first-update control binding')
    for method in ['IL','RL']:require(control['methods'][method]['exact_match'] and control['methods'][method]['update_record_sha256']==sha(run/method/'update-01/update.json'),'first-update exact record pin')
    require(len(visits)==76 and len(costs['completed_patient_visits'])==76 and all(x['source_released'] for x in costs['completed_patient_visits']),'76 complete released visits')
    forwards=sum(row['loss_forward_calls'] for method in curves for row in curves[method])+sum(row['steps'] for (phase,s),row in visits.items() if phase.startswith('RL/update-') or phase.startswith('TRAIN-greedy/'))
    require(forwards==costs['total_policy_forward_calls']<=2496,'forward arithmetic')
    previews=costs['native_budget']['native_preview_entries']
    require(previews<=3373440 and costs['native_budget']['blocked_preview_attempts']==0 and costs['native_budget']['failure'] is None,'preview accounting')
    for relative,meta in list(seen.items()): require(sha(ROOT/relative)==meta['sha256'],'post-read stability')
    summary={'status':'PASS_saved_metadata_consistency','checks':checks,'source_files_verified':len(index['source_files']),'metadata_files_verified':len(index['metadata_files']),'saved_records_and_sources_verified':len(seen),'teachers':teachers,'greedy_endpoints':endpoints,'learning_curves':curves,'parent':{k:receipt[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','output_bytes','final_owned_pids','cleanup_errors']},'costs':{'completed_replayed_visits':76,'shared_updates':16,'gradient_contributions':64,'native_previews':previews,'policy_forwards':forwards},'scope':'Saved metadata/arithmetic only. No checkpoint contents, model execution, patient arrays or geometry replay. In-memory endpoint inference, not checkpoint-reload validation. RL exposure is stochastic TRAIN collection at each pre-update policy, not a fixed greedy learning curve. All four TRAIN denominators retained; no held-out/clinical claim.','audit_elapsed_seconds':time.monotonic()-started,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    saved_write('audit-result.json',summary);saved_write('input-hashes.json',seen)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['teachers','greedy_endpoints','learning_curves']},sort_keys=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-result-sha256',required=True)
    args=parser.parse_args()
    main(args.terminal_result_sha256)
