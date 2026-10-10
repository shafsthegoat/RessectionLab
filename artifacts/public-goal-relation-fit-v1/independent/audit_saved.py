"""Prepared saved-JSON arithmetic; requires explicit terminal receipt binding."""
from pathlib import Path
import argparse,hashlib,json,math,os,resource,signal,stat,sys,time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--terminal-receipt-sha256',required=True)
args=p.parse_args()
if len(args.terminal_receipt_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.terminal_receipt_sha256):
    raise ValueError('Exact terminal hash required before result access')
ROOT=Path.cwd().resolve();OUT=ROOT/'build/goal-relation-fit-independent-v1'
PREP=ROOT/'build/goal-conditioned-policy-v1/goal-relation-policy-v1'
RUN=ROOT/'build/goal-conditioned-policy-v1/goal-relation-fit-run-v1';SUP=RUN.with_name(RUN.name+'.supervision')
OLD=ROOT/'build/goal-conditioned-policy-v1/full-teacher-refit-run-v1'
OLDART=ROOT/'artifacts/contact-full-teacher-refit-result-v1'
start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(RuntimeError('60second saved audit cap')));signal.alarm(60)
blocked=[];seen={}
def need(ok,reason):
    if not ok:raise ValueError(reason)
def guard(event,a):
    if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp','socket.connect','socket.bind','socket.getaddrinfo'}:
        blocked.append(event);raise RuntimeError('No process/network')
    if event=='import' and (str(a[0]).split('.')[0] in {'torch','numpy','scipy','resectionlab','scripts'}):
        blocked.append(str(a[0]));raise RuntimeError('No scientific/project imports')
    if event=='open' and isinstance(a[0],(str,bytes)):
        q=Path(os.fsdecode(a[0])).resolve()
        if (q.suffix in {'.gmckpt','.pt','.pth','.npy','.npz','.nii','.gz','.mat','.csv','.feb','.log'} or
            any(q.is_relative_to(ROOT/x) for x in ('data','sources','outputs'))):
            blocked.append(str(q));raise RuntimeError('No checkpoint/input/raw payload')
sys.addaudithook(guard)
def raw(path,expected=None):
    q=Path(path);q=q if q.is_absolute() else ROOT/q
    need(q.is_relative_to(ROOT) and '..' not in q.parts and not any(x.is_symlink() for x in (q,*q.parents)),'contained regular metadata')
    fd=os.open(q,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd);need(stat.S_ISREG(a.st_mode) and a.st_size<=1024**2,'1MiB file cap')
        with os.fdopen(fd,'rb',buffering=0,closefd=False) as f:data=f.read(1024**2+1)
        b=os.fstat(fd);ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(len(data)==a.st_size and ident(a)==ident(b)==ident(q.lstat()),'stable file')
    finally:os.close(fd)
    h=hashlib.sha256(data).hexdigest();need(expected is None or h==expected,'hash '+str(q))
    key=str(q.relative_to(ROOT));row={'sha256':h,'bytes':len(data)}
    need(key not in seen or seen[key]==row,'changed file');seen[key]=row
    return data
def read(path,expected=None):
    def pairs(items):
        r={}
        for k,v in items:need(k not in r,'duplicate JSON key');r[k]=v
        return r
    return json.loads(raw(path,expected),object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def digest(v):return 'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b):need(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),'saved arithmetic mismatch')
def hashof(path):return seen[str(path.relative_to(ROOT))]['sha256']

receipt=read(SUP/'receipt.json',args.terminal_receipt_sha256)
need(receipt['result_sha256']=='4eed507e1ceb9160714cfe3e64101415fa7d26ae62b80406633e8f93ab2e70db','root terminal result binding')
need(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] is True,'terminal completion')
need(receipt['final_owned_pids']==[] and receipt['cleanup_errors']==[] and receipt['stop_reason'] is None,'clean owned termination')
result=read(RUN/'result.json',receipt['result_sha256']);declaration=read(SUP/'declaration.json')
need(result['status']=='complete_fixed_full40_TRAIN_goal_relation_fit','complete fixed endpoint')
source=read(PREP/'source-index.json','c2feb9288dcd15f4feeb85cb999d86fa4c190600c80069ac11cde3352b39274c')
recipe=read(PREP/'run-recipe.json','e779bff3f289c9e034b36fc3c75ed3313e444adfb7c4678400cf746eead7d7a5')
need(declaration['source_index_sha256']==hashof(PREP/'source-index.json') and declaration['source_files']==source['source_files'],'exact admitted source')
for name,h in source['source_files'].items():raw(ROOT/name,h)
inputs=read(PREP/'input-index.json',source['input_index']['sha256'])
need(declaration['input_index_sha256']==source['input_index']['sha256'],'input index identity only')
experiment=read(RUN/'experiment-freeze.json');prospective=read(PREP/'prospective-experiment.json')
need(experiment==prospective and digest(experiment)==source['refit_experiment_hash']==result['experiment_hash']==recipe['experiment_hash'],'declared experiment')
need(experiment['architecture_hash']==source['architecture_hash'] and result['baseline_experiment_hash']==source['baseline_experiment_hash'],'architecture/default lineage')
corpus=experiment['teacher_states'];roles={r['layout_id']:r['role'] for r in experiment['family_manifest']['source_bindings']}
need(len(corpus)==40 and sum(r['step']==0 for r in corpus)==24 and sum(r['step']==1 for r in corpus)==16 and sum(r['action_id']=='STOP' for r in corpus)==8,'complete fixed denominators')
need(all(roles[r['layout_id']]=='TRAIN' for r in corpus),'TRAIN only')
roots=[r for r in corpus if r['step']==0];need(len({(r['layout_id'],r['goal_id']) for r in roots})==24,'24 unique tasks')
oldindex=read(OLDART/'external-evidence-index.json');oldpins={r['path']:r['sha256'] for r in oldindex['files']}
baseline=read(OLDART/'result.json','e68bdc9c0f030532544ee6aa1343fea76a1ef1a06cf47526e1702aa0a049ed51')
oldreceipt=read(OLDART/'receipt.json','ee30237f90565928aa4b05a7afb44dfce6f518f93e6603f6b5d4f070c1b5b441')
oldexperiment=read(OLD/'experiment-freeze.json',oldpins[str((OLD/'experiment-freeze.json').relative_to(ROOT))])
need(digest(oldexperiment)==baseline['experiment_hash']==source['baseline_full40_experiment_hash'] and oldexperiment['teacher_states']==corpus,'same ordered baseline40 corpus')
need(oldexperiment['family_manifest']==experiment['family_manifest'],'same family/source roles')
for key in ('updates','batch_size','loss_forward_calls','fixed_before_after_TRAIN_forwards','learning_rate','betas','eps','weight_decay','max_gradient_norm','seed'):
    need(oldexperiment['protocol'][key]==experiment['protocol'][key],'matched fixed optimization '+key)
counts={'optimizer_updates':32,'native_teacher_steps':40,'loss_forward_calls':1280,'readout_forward_calls':80,'new_checkpoint_loads':1,'prior_checkpoint_loads':0,'search_calls':0,'patient_reads':0,'SELECT_or_MEASUREMENT_reads':0}
need(all(result[k]==v and baseline[k]==v for k,v in counts.items()),'matched work and access counts')
need(result['geometry_previews']==baseline['geometry_previews']==760 and result['geometry_previews']<=2048,'unchanged reconstruction geometry work')
need(result['clinical_validation'] is False and result['greedy_native_rollout_executed'] is False,'no trajectory or clinical claim')
need(0<result['wall_seconds']<receipt['elapsed_seconds']<180 and 0<receipt['sampled_peak_rss_bytes']<=2**30,'resource caps')
need(receipt['caps']['threads']==1 and receipt['caps']['attempts']==1 and receipt['automatic_retry'] is False,'one thread/attempt')
initial=result['initial_checkpoint'];final=result['final_checkpoint'];init=result['shared_initialization']
need(init['shared_tensors_exact'] is True and init['added_input_columns_zero'] is True,'source-bound exact initialization receipt')
need(init['added_parameters']==init['expanded_parameters']-init['baseline_parameters']==704,'704 added zero parameters')
need(init['baseline_parameter_hash']==baseline['initial_checkpoint']['parameter_hash']==experiment['protocol']['baseline_shared_initial_parameter_hash'],'same baseline initialization')
need(init['expanded_parameter_hash']==initial['parameter_hash'] and init['expanded_parameter_hash']!=init['baseline_parameter_hash'],'separate expanded hash')
for item,updates in ((initial,0),(final,32)):
    lineage=item['lineage'];need(lineage['experiment_hash']==result['experiment_hash'] and lineage['optimizer_updates']==updates and lineage['real_patient_count']==0,'checkpoint metadata lineage')
    need(lineage['parameter_hash']==item['parameter_hash'],'checkpoint parameter metadata')
need(final['lineage']['initial_parameter_hash']==initial['parameter_hash'],'fresh expanded lineage')
reconstruction=[read(RUN/f'teacher-reconstruction-{i:02d}.json') for i in range(24)]
need(reconstruction==result['completed_teacher_reconstructions'],'complete24 reconstruction records')
for i,(r,c) in enumerate(zip(reconstruction,roots)):
    need(r['teacher_index']==i and r['original_binding_hash']==c['original_binding_hash'] and r['strategy_seal']==c['strategy_seal'],'fixed teacher source seal')
need({digest(x) for x in final['lineage']['training_bindings']}=={r['refit_binding_hash'] for r in reconstruction},'all24 training bindings')
need(all(x['role']=='TRAIN' and x['experiment_hash']==result['experiment_hash'] for x in final['lineage']['training_bindings']),'TRAIN lineage')
updates=[read(RUN/f'update-{i:02d}.json') for i in range(1,33)];previous=initial['parameter_hash'];norms=[]
for i,row in enumerate(updates,1):
    u=row['update_receipt'];need(row['update']==u['cumulative_updates']==i and u['optimizer_steps']==1,'32 ordered single updates')
    need(row['experiment_hash']==result['experiment_hash'] and row['all40_ordered_state_manifest_hash']==digest(corpus),'full40 loss provenance')
    need(row['loss']['supervised_actions']==row['loss']['loss_forward_calls']==40,'40 losses each update')
    need(u['initial_parameter_hash']==previous and u['updated_parameter_hash']!=previous and u['parameters_changed'] is True,'parameter transition chain')
    need(math.isfinite(u['gradient_norm_before_clip']) and u['gradient_norm_before_clip']>=0 and u['module_gradient_norms_before_clip']['critic']==0,'finite IL gradients')
    previous=u['updated_parameter_hash'];norms.append(u['gradient_norm_before_clip'])
need(previous==final['parameter_hash'] and sum(x['loss']['loss_forward_calls'] for x in updates)==1280,'final32 endpoint and1280 loss calls')

fixed=('teacher_index','layout_id','goal_id','step','binding_hash','teacher_action','observation_hash','action_ids','action_modes','legal_mask','teacher_action_index')
def check_row(row,c):
    for a,b in [('layout_id','layout_id'),('goal_id','goal_id'),('step','step'),('binding_hash','original_binding_hash'),('teacher_action','action_id'),('observation_hash','observation_hash')]:need(row[a]==c[b],'fixed state identity')
    ids,modes,mask,logits=(row[k] for k in ('action_ids','action_modes','legal_mask','logits'))
    need(len(ids)==len(modes)==len(mask)==len(logits) and len(set(ids))==len(ids),'action inventory')
    need(ids[0]=='STOP' and modes[0]=='stop' and mask[0] is True and all(type(v)is bool for v in mask),'STOP/mask semantics')
    legal=[i for i,m in enumerate(mask) if m];need(all(math.isfinite(logits[i]) for i in legal),'finite legal logits')
    need(all(logits[i] is None for i,m in enumerate(mask) if not m),'masked nulls')
    t=ids.index(row['teacher_action']);need(t==row['teacher_action_index'] and mask[t],'legal label')
    peak=max(logits[i] for i in legal);den=sum(math.exp(logits[i]-peak) for i in legal)
    probs={i:math.exp(logits[i]-peak)/den for i in legal};chosen=max(legal,key=lambda i:logits[i])
    need(row['greedy_action']==ids[chosen] and row['greedy_mode']==modes[chosen] and row['correct']==(chosen==t),'inventory-first argmax')
    close(row['teacher_probability'],probs[t]);close(row['STOP_probability'],probs[0]);close(row['cross_entropy'],peak+math.log(den)-logits[t]);close(row['entropy'],-sum(v*math.log(v) for v in probs.values() if v>0))
    movement=sorted((i for i in legal if modes[i]!='stop'),key=lambda i:(-logits[i],i))
    need(row['teacher_rank_among_legal_movements']==(None if t==0 else movement.index(t)+1),'movement rank')
    if movement:close(row['STOP_minus_best_movement_margin'],logits[0]-logits[movement[0]])
    else:need(row['STOP_minus_best_movement_margin'] is None,'empty movement margin')
    need(row['movement_inventory_counts']=={m:sum(mask[i] and modes[i]==m for i in range(len(ids))) for m in ('aspirate','probe')},'mode inventories')

def summarize(rows,claimed):
    groups={'all40':rows,'root24':[r for r in rows if r['step']==0],'second16':[r for r in rows if r['step']==1],
            'STOP8':[r for r in rows if r['teacher_action']=='STOP'],'movement32':[r for r in rows if r['teacher_action']!='STOP'],
            'aspiration_root16':[r for r in rows if r['step']==0 and r['teacher_action']!='STOP']}
    expected={'all40':40,'root24':24,'second16':16,'STOP8':8,'movement32':32,'aspiration_root16':16};out={}
    for k,g in groups.items():
        n=len(g);need(n==expected[k],'full denominator '+k)
        calc={'n':n,'mean_cross_entropy':sum(r['cross_entropy'] for r in g)/n,'accuracy':sum(r['correct'] for r in g)/n,
              'greedy_STOP':sum(r['greedy_mode']=='stop' for r in g),'teacher_movement_rank1':sum(r['teacher_rank_among_legal_movements']==1 for r in g),
              'mean_STOP_probability':sum(r['STOP_probability'] for r in g)/n}
        for name,v in calc.items():close(v,claimed[k][name])
        out[k]={**calc,'correct_count':sum(r['correct'] for r in g),'greedy_modes':{m:sum(r['greedy_mode']==m for r in g) for m in ('stop','aspirate','probe')}}
    return out
readouts={};baseline_rows={};summaries={};baseline_summaries={};init_errors=[];value_errors=[]
for label,params in (('before',initial['parameter_hash']),('after',final['parameter_hash'])):
    rows=[read(RUN/f'{label}-state-{i:02d}.json') for i in range(40)]
    oldrows=[read(OLD/f'{label}-state-{i:02d}.json',oldpins[str((OLD/f'{label}-state-{i:02d}.json').relative_to(ROOT))]) for i in range(40)]
    for i,(r,o,c) in enumerate(zip(rows,oldrows,corpus)):
        check_row(r,c);check_row(o,c);need(r['parameter_hash']==params,'readout parameter binding')
        need(all(r[k]==o[k] for k in fixed),'same baseline state/mask/inventory')
        if label=='before':
            errors=[abs(a-b) for a,b in zip(r['logits'],o['logits']) if a is not None];mx=max(errors);vd=abs(r['value']-o['value'])
            close(r['baseline_initial_max_abs_logit_difference'],mx);need(mx<=1e-6 and vd<=1e-6,'all40 initial functional parity')
            need(inputs['data'][f'baseline-before-state-{i:02d}.json']['sha256']==hashof(OLD/f'before-state-{i:02d}.json'),'admitted saved initial baseline')
            init_errors.append(mx);value_errors.append(vd)
    readouts[label]=rows;baseline_rows[label]=oldrows;summaries[label]=summarize(rows,result[label]);baseline_summaries[label]=summarize(oldrows,baseline[label])
need(all(all(a[k]==b[k] for k in fixed) for a,b in zip(readouts['before'],readouts['after'])),'unchanged before/after identities')
need(abs(updates[0]['loss']['loss']-summaries['before']['all40']['mean_cross_entropy'])<2e-7,'float32 initial CE mean')
costs=result['costs'];c=costs['refit.reconstruct40_native_teacher_states']
for k,v in {'geometry_preview_calls':760,'geometry_preview_returned_calls':760,'native_transition_calls':40,'native_transition_returned_calls':40,'native_nonstop_commit_calls':32,'native_nonstop_commit_returned_calls':32,'geometry_audit_calls':24,'geometry_audit_returned_calls':24,'task_clone_calls':24,'task_clone_returned_calls':24}.items():need(c[k]==v,'reconstruction costs')
for name,n in [('before.readout40',40),('loss40',1280),('after.readout40',40)]:
    need(costs['refit.'+name]['actor_forward_calls']==costs['refit.'+name]['actor_forward_returned_calls']==n,'forward costs')
pairs=[]
for layout in sorted({r['layout_id'] for r in roots}):
    new=[r for r in readouts['after'] if r['layout_id']==layout and r['step']==0]
    old=[r for r in baseline_rows['after'] if r['layout_id']==layout and r['step']==0]
    need(len(new)==len(old)==2,'all12 paired goals')
    common=set(new[0]['action_ids'])&set(new[1]['action_ids'])
    changes=[abs(new[0]['logits'][new[0]['action_ids'].index(a)]-new[1]['logits'][new[1]['action_ids'].index(a)]) for a in common
             if new[0]['legal_mask'][new[0]['action_ids'].index(a)] and new[1]['legal_mask'][new[1]['action_ids'].index(a)]]
    pairs.append({'layout_id':layout,'same_greedy_action_id':new[0]['greedy_action']==new[1]['greedy_action'],
         'baseline_same_greedy_action_id':old[0]['greedy_action']==old[1]['greedy_action'],
         'maximum_common_legal_score_difference':max(changes),'goals':[{'goal_id':n['goal_id'],'teacher_action':n['teacher_action'],
             'new_choice':n['greedy_action'],'new_correct':n['correct'],'new_STOP_margin':n['STOP_minus_best_movement_margin'],
             'baseline_choice':o['greedy_action'],'baseline_correct':o['correct']} for n,o in zip(new,old)]})
need(len(pairs)==12,'all12 layouts')
expected={'result.json','experiment-freeze.json','relation-initial.gmckpt','relation-IL-final.gmckpt'}|{f'{label}-state-{i:02d}.json' for label in ('before','after') for i in range(40)}|{f'update-{i:02d}.json' for i in range(1,33)}|{f'teacher-reconstruction-{i:02d}.json' for i in range(24)}
need({p.name for p in RUN.iterdir()}==expected,'exact completed output names')
snapshot=dict(seen)
for name,b in snapshot.items():raw(ROOT/name,b['sha256'])
need(snapshot==seen and not blocked,'unchanged saved evidence/access guards')
elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;need(elapsed<60 and rss<512*1024**2,'bounded saved audit')
audit={'status':'PASS_SAVED_JSON_ONLY','source_index_sha256':declaration['source_index_sha256'],'run_head':declaration['head'],
       'result_sha256':receipt['result_sha256'],'receipt_sha256':args.terminal_receipt_sha256,'experiment_hash':result['experiment_hash'],
       'counts':{**counts,'geometry_previews':760},'shared_initialization_receipt':init,
       'initial_function_max_logit_difference':max(init_errors),'initial_function_max_value_difference':max(value_errors),
       'tensor_recomparison_by_audit':False,'checkpoint_reads':0,'summaries':summaries,'baseline_summaries':baseline_summaries,
       'paired_roots':pairs,'all32_parameter_updates_chained':True,'ordered_manifest_hash':digest(corpus),
       'gradient_norm_range':[min(norms),max(norms)],'clipping_triggered_updates':sum(n>experiment['protocol']['max_gradient_norm'] for n in norms),
       'cost_comparison':{'same_updates':32,'same_loss_forwards':1280,'same_readout_forwards':80,'added_parameters':704,
         'new_parent_wall_seconds':receipt['elapsed_seconds'],'old_parent_wall_seconds':oldreceipt['elapsed_seconds'],
         'new_worker_wall_seconds':result['wall_seconds'],'old_worker_wall_seconds':baseline['wall_seconds'],
         'new_sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'old_sampled_peak_rss_bytes':oldreceipt['sampled_peak_rss_bytes'],
         'matched_total_compute_claim':False},
       'on_policy_trajectory_evidence':False,'clinical_validation':False,'audit_elapsed_seconds':elapsed,'audit_peak_self_rss_bytes':rss,
       'source_and_result_files_unchanged':len(snapshot),'blocked_access_attempts':blocked,
       'limitations':'Saved logits and metadata only; no checkpoint/tensor decode, native reconstruction, gradient or trajectory reexecution. Shared/zero tensor and gradient claims are source-bound producer records.'}
(OUT/'audit.json').write_text(json.dumps(audit,sort_keys=True,indent=2,allow_nan=False)+'\n')
(OUT/'source-and-output-hashes.json').write_text(json.dumps(snapshot,sort_keys=True,indent=2)+'\n')
signal.alarm(0);print(json.dumps({'status':audit['status'],'before':summaries['before'],'after':summaries['after'],'audit_seconds':elapsed,'rss_bytes':rss}))
