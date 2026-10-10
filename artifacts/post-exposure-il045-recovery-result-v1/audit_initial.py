"""One bounded stdlib saved-result audit; no geometry/model/array execution."""
import hashlib,json,math,os,resource,signal,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RUN=ROOT/'build/post-exposure-IL045-evaluation-recovery-v1'
OLD=ROOT/'build/post-exposure-IL045-evaluation-diagnostic-v1'
IL=ROOT/'artifacts/post-exposure-il64-result-v1'
started=time.monotonic();inputs={};checks=0
signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('60s saved audit cap')))
signal.alarm(60)
def need(ok,label):
    global checks
    checks+=1
    if not ok:raise AssertionError(label)
def read(p,expected=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    need(p.suffix in ('.json','.py','.log'),'saved source/JSON only')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        s=os.fstat(fd);need(stat.S_ISREG(s.st_mode) and 0<=s.st_size<=4*1024**2,'bounded regular input')
        with os.fdopen(fd,'rb',closefd=False) as f:b=f.read(4*1024**2+1)
        need(len(b)==s.st_size and len(b)<=4*1024**2,'bounded stable input size')
    finally:os.close(fd)
    h=hashlib.sha256(b).hexdigest()
    if expected:need(h==expected,'exact input '+str(p))
    inputs[str(p.relative_to(ROOT))]={'sha256':h,'bytes':len(b)}
    return b
def load(p,h=None):return json.loads(read(p,h))
def semantic(x):return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
rh='432535613f699604a4b0f3b0aaa0fedebbe932e5ca6b7006beedd66ff823a7ce'
ph='c159b91ab2015dfd54dee677eb683b27e335a62657ba51838c398ed2d391a98e'
lh='e5dd29dc93460dc40c66a27dcf79986736797fd65ae9277e0e8026ea860d1797'
r=load(RUN/'attempt-01/result.json',rh);p=load(RUN/'attempt-01.supervision/receipt.json',ph)
l=load(RUN/'root-release.json',lh);idx=load(l['source_index']['path'],l['source_index']['sha256'])
c=load(RUN/'attempt-01/native-replay.json',r['native_replay_sha256'])
m=load(RUN/'attempt-01/committed-replay-metrics.json')
oldc=load(l['inputs']['prior_diagnostic_certificate']['path'],l['inputs']['prior_diagnostic_certificate']['sha256'])
oldr=load(l['inputs']['prior_diagnostic_result']['path'],l['inputs']['prior_diagnostic_result']['sha256'])
original=load(l['inputs']['failed_result']['path'],l['inputs']['failed_result']['sha256'])
originalreceipt=load(l['inputs']['failed_receipt']['path'],l['inputs']['failed_receipt']['sha256'])
originalidx=load(l['inputs']['original_source_index']['path'],l['inputs']['original_source_index']['sha256'])
plan_record=load(l['inputs']['plan']['path'],l['inputs']['plan']['sha256']);plan=plan_record['plan']
need(l['execution_released'] is True and l['expected_head']==idx['head'],'source release join')
need(p['source_index']==l['source_index'] and p['release_sha256']==r['release_sha256']==lh and p['result_sha256']==rh,'receipt joins')
need(p['status']=='complete' and p['exit_code']==0 and p['stop_reason'] is None,'owned terminal success')
need(p['cleanup_errors']==[] and p['remaining_owned_pids']==[] and p['worker_termination_confirmed'] is True and p['automatic_retry'] is False,'clean owned one attempt')
need(p['caps']==l['caps'] and p['elapsed_seconds']<p['caps']['parent_seconds'] and r['elapsed_seconds']<p['caps']['worker_seconds'],'wall caps')
need(p['sampled_peak_rss_bytes']<p['caps']['sampled_rss_bytes'],'sampled RSS cap')
need(sum(f.stat().st_size for f in (RUN/'attempt-01').rglob('*') if f.is_file())==p['output_bytes']<p['caps']['output_bytes'],'output size and cap')
need(r['source_visits']==1 and r['committed_actions']==24 and r['native_previews']==2184<=p['caps']['native_previews'],'exact visits/action/preview accounting')
need(all(r[k]==0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','teacher_search_calls')),'evaluation only')
need(r['training_admitted'] is False,'no new training authorization')
need(r['full_history_equal'] is True and r['independent_accepted'] is True and r['first_failing_action_indices_zero_based']==[],'accepted full replay')
need(c['metrics']==m==oldc['metrics'],'all committed metrics byte-value unchanged from negative diagnostic')
a=c['independent_geometry'];g=a['geometry']
need(a['accepted'] is True and a['complete_episode'] is True and g==r['independent_geometry'],'exact accepted certificate')
need(g['action_count']==24 and g['feasible'] is True and g['complete_tool_checked'] is True and g['frontier_checked'] is True and g['failures']==[] and g['first_failed_action'] is None,'complete independent full-tool/frontier checks')
need(g['unsupported_source_tissue_volume_mm3']==0 and g['claimed_source_tissue_volume_mm3']==g['contained_source_tissue_volume_mm3']==m['simulated_removed_volume_mm3'],'contained volume join')
need(a['committed_history_hash']==semantic(m['history']),'independent history seal')
need(semantic(plan)==plan_record['plan_seal']==r['saved_plan_seal'],'original plan seal')
need(semantic(plan['history'])==r['saved_history_hash'],'original history seal')
physical=lambda rows:[{k:v for k,v in x.items() if k!='outcome_scope'} for x in rows]
need(physical(plan['history'])==physical(m['history']),'complete original physical history unchanged')
need(plan['actions']==[x['action_id'] for x in m['history']] and len(plan['actions'])==24 and 'STOP' not in plan['actions'] and plan['terminal_reason']=='HORIZON','fixed horizon actions')
for key in ('source_hash','decision_model_hash','context_hash','parameter_hash'):
    need(r[key]==plan[key],'saved plan lineage '+key)
need(r['prior_learning_updates']==plan['learning_updates']==64 and original['optimizer_updates']=={'IL':64,'RL':0},'64 original updates, no new updates')
need(original['status']=='failed_or_unresolved' and originalreceipt['exit_code']!=0,'original negative preserved')
need(r['original_IL_result_sha256']==l['inputs']['failed_result']['sha256'] and r['original_IL_parent_receipt_sha256']==l['inputs']['failed_receipt']['sha256'],'original negative lineage')
need(r['checkpoint_sha256']==l['inputs']['checkpoint']['sha256']==original['checkpoints']['IL']['sha256'],'checkpoint identity without opening bytes')
need(r['parameter_hash']==original['checkpoints']['IL']['parameter_hash'],'parameter identity')
srcdelta={q:(originalidx['files'][q],h) for q,h in idx['files'].items() if q.startswith('src/') and q in originalidx['files'] and h!=originalidx['files'][q]}
expected={x['path']:(x['original_sha256'],x['current_sha256']) for x in l['permitted_source_delta']}
need(srcdelta==expected and set(srcdelta)=={'src/resectionlab/evaluation.py','src/resectionlab/patient_planning_preflight.py'},'only declared checker and logging source deltas')
need(idx['files']['src/resectionlab/evaluation.py']=='dbfac0b3d8acd3f8461581c96a5ae9421ed341a03559d3ab8ba0589e8a0d028a','reviewed stable-angle checker')
for name in ('batch_contract.py','initial_inventory_worker.py','run_owned.py'):
    read(RUN/name,idx['files'][str((RUN/name).relative_to(ROOT))])
cells=[tuple(v) for h in m['history'] for v in h['removed_indices_native']]
need(len(cells)==len(set(cells))==51,'51 distinct removed cells')
need(math.isclose(sum(x['target_removed_mm3'] for x in m['history']),m['target_removed_mm3'],rel_tol=0,abs_tol=1e-10),'target sum')
need(math.isclose(sum(x['reward'] for x in m['history']),m['total_reward'],rel_tol=0,abs_tol=1e-10),'reward sum')
need(m['normal_removed_mm3']==0,'outside-target removal zero')
need(a['outcomes']['clinical_deficit_probability'] is None and m['clinical_deficit_probability'] is None,'clinical claim absent')
oldsummary=load(IL/'route-summary.json');rows=[]
need([x['subject'] for x in oldsummary]==['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'],'all-four denominator/order')
for row in oldsummary:
    subject=row['subject'];ep=a if subject=='ReMIND-045' else load(IL/'files'/f'{subject}-independent-geometry.json')
    need(ep['accepted'] is True and ep['complete_episode'] is True and ep['geometry']['failures']==[],subject+' accepted endpoint')
    need(math.isclose(ep['outcomes']['target_removed_mm3'],row['target_removed_mm3'],abs_tol=1e-10,rel_tol=0),subject+' target unchanged')
    need(ep['outcomes']['total_reward']==row['public_return'],subject+' reward unchanged')
    rows.append({k:row[k] for k in ('subject','steps','terminal_reason','removed_cells','target_removed_mm3','outside_target_removed_mm3','public_return','same_removed_cell_union_as_teacher','teacher_target_removed_mm3','teacher_public_return')})
    rows[-1].update(independent_accepted=True,certificate_origin='corrected_evaluation_recovery' if subject=='ReMIND-045' else 'original_run',motion_count=sum(x!='STOP' for x in row['actions']),teacher_motion_count=sum(x!='STOP' for x in row['teacher_actions']))
need(rows[-1]['same_removed_cell_union_as_teacher'] and rows[-1]['motion_count']==24 and rows[-1]['teacher_motion_count']==12,'045 efficiency distinction retained')
summary={'status':'PASS_saved_accepted_recovery','checks':checks,'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'original_IL_attempt_status':original['status'],'recovery_geometric_acceptance':True,'new_training_or_clinical_authority':False,
 'native_previews':r['native_previews'],'owned_seconds':p['elapsed_seconds'],'owned_peak_rss_bytes':p['sampled_peak_rss_bytes'],
 'original_plan_and_full_metrics_unchanged':True,'source_delta':srcdelta,'four_TRAIN_outcomes':rows,
 '045_teacher_minus_IL_return':rows[-1]['teacher_public_return']-rows[-1]['public_return'],
 '045_retained_contact_upper_bound_mm3':m['currently_retained_contacted_tissue_upper_bound_mm3'],
 'physical_validation_pass':None,'scope':'saved JSON/source only; no acquired arrays, checkpoint bytes, geometry replay or model call'}
need(summary['peak_rss_bytes']<512*1024**2,'audit observed RSS cap');summary['checks']=checks
write('audit-result.json',summary);write('four-TRAIN-outcomes.json',rows)
write('accepted-certificate-summary.json',{k:v for k,v in a.items() if k!='metrics'})
for name,src in [('result.json',RUN/'attempt-01/result.json'),('receipt.json',RUN/'attempt-01.supervision/receipt.json'),('declaration.json',RUN/'attempt-01.supervision/declaration.json'),('worker.log',RUN/'attempt-01.supervision/worker.log'),('root-release.json',RUN/'root-release.json'),('source-index.json',RUN/'source-index.json'),('batch_contract.py',RUN/'batch_contract.py'),('initial_inventory_worker.py',RUN/'initial_inventory_worker.py'),('run_owned.py',RUN/'run_owned.py')]:
    (OUT/name).write_bytes(read(src))
write('input-index.json',{'files':inputs,'excluded_checkpoint_binding_not_opened':l['inputs']['checkpoint'],'full_native_replay_not_copied':{'path':str((RUN/'attempt-01/native-replay.json').relative_to(ROOT)),'sha256':r['native_replay_sha256']}})
signal.alarm(0)
print(json.dumps(summary,indent=2))
