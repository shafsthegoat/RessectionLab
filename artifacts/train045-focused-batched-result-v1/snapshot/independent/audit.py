"""Independent bounded saved-result arithmetic/parity audit. Stdlib only.
Gzip inputs decoded one at a time (32MiB limit). No native/model/payload reads.
"""
import gc,gzip,hashlib,importlib.util,io,json,math,resource,signal,statistics,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent;OUT=STAGE/'run-01'
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
PINS={'report':'71a0806d9dee4a2497edeb47077844368618a9657fc70fa32f6ce7878310ca9d',
'launcher':'a463af29e84fec3c471d68fd645b10c9658986a7843d342633a58cac7e048594',
'index':'8bf1c3def8f773f5b9b4505392b995886d8cba87a96e3eb91af926d23ccb4df6'}
HELPER='build/train045-focused-policy-v1/independent-result2/audit_result.py'
HELPER_SHA='e51289e99d430453b5a564d0ae0aae872db8796ab677280a43eac33999848783'

def semantic(value):
    h=hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):h.update(part.encode())
    return 'sha256:'+h.hexdigest()

class Audit:
    def __init__(self):self.checks=0;self.files={}
    def need(self,condition,label):
        self.checks+=1
        if not condition:raise AssertionError('check %d: %s'%(self.checks,label))
    def close(self,x,y,label):
        self.need(type(x) in (int,float) and type(y) in (int,float) and math.isfinite(x) and math.isfinite(y)
            and math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-9),label)
    def raw(self,path,digest=None,limit=32*1024**2):
        path=Path(path);self.need(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'Bounded regular saved file '+path.name)
        raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
        if digest:self.need(actual==digest,'Pinned saved bytes '+path.name)
        self.files[str(path.relative_to(ROOT))]={'sha256':actual,'bytes':len(raw)};return raw
    def read(self,path,digest=None):return json.loads(self.raw(path,digest))
    def compressed(self,path,binding):
        raw=self.raw(path,binding['sha256'],3*1024**2)
        self.need(len(raw)==binding['bytes'] and binding['decompressed_limit']==32*1024**2,'Exact compressed size and decode limit')
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:decoded=stream.read(32*1024**2+1)
        del raw
        self.need(len(decoded)==binding['decompressed_bytes']<=32*1024**2
            and hashlib.sha256(decoded).hexdigest()==binding['decompressed_sha256'],'Exact bounded lossless decoded bytes')
        result=json.loads(decoded);del decoded;return result
    def eq(self,x,y,label):self.need(semantic(x)==semantic(y),label)

class VerifiedSide:
    def __init__(self,digest):self.digest=digest
    def __eq__(self,row):return semantic(row)==self.digest
class SideIndex:
    def __init__(self,records):self.digests=records
    def __getitem__(self,key):return VerifiedSide(self.digests[key])

def own_chain(a,directory,account,index_hash):
    context=a.read(directory/'ReMIND-045-context.json')
    binding=a.read(directory/'evaluation/source/admitted-public-bindings.json')
    raw=a.raw(directory/'world-admission.json');release=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    a.need(release['root_source_index_sha256']==index_hash and release['execution_released'] is True
        and binding['protocol']['runtime_release_sha256']==digest and context['protocol_sha256']==semantic(binding['protocol'])
        and account['context']['original_context_hash']==semantic(context)
        and account['context_seal']==semantic(account['context']),'Own release→protocol→context→focused-seal chain')
    return {'context':context,'binding':binding,'release':release,'release_sha':digest}

def chain_equal(a,old,new,oldaccount,newaccount):
    for x,y,key,label in ((old['release'],new['release'],'root_source_index_sha256','release'),
        (old['binding'],new['binding'],'protocol','full public bindings'),
        (old['binding']['protocol'],new['binding']['protocol'],'runtime_release_sha256','protocol'),
        (old['context'],new['context'],'protocol_sha256','original context'),
        (oldaccount['context'],newaccount['context'],'original_context_hash','focused context')):
        a.eq({k:v for k,v in x.items() if k!=key},{k:v for k,v in y.items() if k!=key},'Only authenticated release identity differs: '+label)

def projection(account):
    return {'context':account['context'],'context_seal':account['context_seal'],
        **{k:account[k] for k in ('controller','initial_focus','fixed_prefix_return','method','complete')},
        'packets':[semantic(p) for p in account['explored_action_records']],
        'history':[semantic(r) for r in account['explored_history']],
        'states':[semantic(r) for r in account['states'] if 'result_state_hash' in r.get('selected',{})],
        'prefixes':{str(n):{'packets':semantic(account['explored_action_records'][:n]),
            'history':semantic(account['explored_history'][:n]),'states':semantic(account['states'][:n-12])}
            for n in (22,23,24) if n<=len(account['explored_history'])}}

def diagnostics(account):
    rows=[];removed=set();contacts=set();prior=None
    for i,h in enumerate(account['explored_history']):
        removed.update(tuple(c) for c in h.get('removed_indices_native',()))
        contacts.update(tuple(c) for c in h.get('contact_indices_native',()))
        if i>=12:
            s=account['states'][i-12];c=s['candidates'];best=max(range(1,len(c)),key=lambda j:c[j]['immediate_reward'])
            win=next(j for j,v in enumerate(c) if v['action_id']==s['selected']['action_id'])
            rows.append({'absolute_action':i+1,'state':s['source_state_hash'],'query':s['query_catalog_digest'],
                'physical':s['selected']['packet']['physical_action'],'action':h['action_id'],'target_mm3':h['target_removed_mm3'],
                'outside_mm3':h['normal_removed_mm3'],'removed_cells':len(h.get('removed_indices_native',())),
                'switch':prior is not None and prior!=h['tool_id'],'path_mm':h['complete_tool_path_length_mm'],
                'reward':h['reward'],'selected_raw_tail':c[win]['raw_tail'],'selected_r_plus_tail':c[win]['ranking_value'],
                'best_immediate_action':c[best]['action_id'],'best_immediate_reward':c[best]['immediate_reward'],
                'best_immediate_raw_tail':c[best]['raw_tail'],'best_immediate_r_plus_tail':c[best]['ranking_value'],
                'immediate_regret':c[best]['immediate_reward']-c[win]['immediate_reward']})
        if h['action_id']!='STOP':prior=h['tool_id']
    return rows,removed,contacts

def inspect(a):
    index=a.read(STAGE/'runner/source-index.json',PINS['index']);report=a.read(OUT/'report.json',PINS['report']);parent=a.read(OUT/'launcher.json',PINS['launcher'])
    for path,digest in index['files'].items():a.raw(ROOT/path,digest)
    a.need(index['execution_released'] is True and report['source_index_sha256']==parent['source_index_sha256']==PINS['index']
        and parent['report_sha256']==PINS['report'] and report['source_unchanged'] is True,'Exact external terminal/source identity')
    life=parent['owned_lifecycle']
    a.need(report['status']==parent['status']=='completed' and parent['worker_exit_code']==life['exit_code']==0
        and life['verified'] is True and life['child_reaped'] is True and not life['errors'] and not life['remaining_owned_pids']
        and life['remaining_state_unknown'] is False and parent['stop_reason'] is None,'Clean verified terminal lifecycle')
    a.need(report['completion_scope']=='full_matched_comparison' and report['baseline_prefix_reproduced'] is True
        and report['frozen_models_unchanged'] is True and report['closed_roles_opened'] is False
        and report['cache_loads']==report['head_optimizer_updates']==0
        and report['checkpoint_loads']==report['checkpoint_loads_attempted']==2,'Inference only; same TRAIN045 source/model scope')
    a.eq(report['model_provenance']['checkpoints'],index['checkpoints'],'Externally pinned saved models; no model read')
    tailpath='build/continuation-tail-v1/run-01/report.json';trained=a.read(ROOT/tailpath,index['input_metadata'][tailpath])
    a.eq(report['model_provenance']['original_tail_fit'],trained['tail_fit'],'Frozen original fit retained')
    a.eq(report['model_provenance']['original_training_worlds'],trained['corpus']['worlds'],'Unchanged four TRAIN training worlds')
    a.need(set(trained['corpus']['worlds'])=={'ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'},'Fixed original fitted cohort')
    profile=report['catalog_profile'];a.eq(profile,a.read(OUT/'catalog-profile.json'),'Exact separately persisted engineering profile')
    orders=[['reference','batched'],['batched','reference']]*2
    a.need(profile['orders']==orders and len(profile['pairs'])==4 and profile['state_count']==1
        and profile['integrity_passed'] is True and profile['before_snapshot']==profile['after_snapshot'],
        'Four predeclared alternating same-catalog pairs, state/certificate snapshot unchanged')
    ref=[];bat=[]
    for n,row in enumerate(profile['pairs']):
        a.need(row['pair']==n+1 and row['order']==orders[n] and set(row['timings'])=={'reference','batched'},'Exact pair order/denominator')
        for mode,values in row['timings'].items():
            a.need(values['zero_new_preview_transition_forward'] is True and values['calls']['certificate']==values['calls']['score']==profile['distinct_actions']-1
                and values['wall_seconds']>0 and values['cpu_seconds']>0,'Every ray checked; zero extra scientific work')
            expected=profile['pages']+1+(2*profile['distinct_actions']-1 if mode=='reference' else 0)
            a.need(values['calls']['state']==expected,'Exact per-page versus repeated state authentication')
        ref.append(row['timings']['reference']['wall_seconds']);bat.append(row['timings']['batched']['wall_seconds'])
    rm,bm=statistics.median(ref),statistics.median(bat);save=statistics.median(x-y for x,y in zip(ref,bat));gate=bm<=.8*rm and save>=.25
    for key,value in [('reference_median_seconds',rm),('batched_median_seconds',bm),('median_ratio',bm/rm),('median_paired_saving_seconds',save)]:a.close(profile[key],value,'Recomputed engineering '+key)
    a.need(profile['maximum_median_ratio']==.8 and profile['minimum_median_paired_saving_seconds']==.25
        and profile['material_gate_passed'] is gate and gate,'Actual predeclared engineering gate passes')
    frontier=report['fixed_frontier'];a.need(frontier['seal']==semantic({k:v for k,v in frontier.items() if k!='seal'}),'Actual frontier seal')
    prefix_file=a.read(OUT/'fixed-prefix.json');prefix=prefix_file['focused_packets'];a.eq(frontier,prefix_file['focused_frontier'],'Actual fixed prefix')
    a.need(frontier['steps']==len(prefix)==len(prefix_file['joins'])==12,'Twelve fixed motions')
    deployment=a.read(OUT/'focused-deployment.json');a.need(deployment['seal']==semantic(deployment['record']),'New focused admission seal')
    planbinding=report['plans_sealed_before_replay'];plans=a.read(OUT/'plans.json',planbinding['sha256'])
    a.need(set(plans)==set(report['branches'])==set(report['selections'])=={'GREEDY','TAIL'},'All paired plans and endpoints retained')
    # Index saved sidecars once; retain hashes only while each full trace is processed.
    sides=a.compressed(OUT/report['sidecars']['path'],report['sidecars']);digests={}
    a.need(len(sides['records'])==report['sidecars']['unique_count']<=4096,'All bounded sidecars retained')
    for key,row in sides['records'].items():
        a.need(key==row['fingerprint']==semantic({k:v for k,v in row.items() if k!='fingerprint'}),'Authenticated native sidecar')
        digests[key]=semantic(row)
    del sides;gc.collect();side_index={'records':SideIndex(digests)}
    a.raw(ROOT/HELPER,HELPER_SHA)
    spec=importlib.util.spec_from_file_location('_previous_saved_scalar_audit',ROOT/HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    # Same saved arithmetic function, with streaming semantic hashing for bounded memory.
    helper.semantic=semantic
    result={};projections={};details={};cellsets={};chains={};profile_queries=None;tail_scoring=None
    for method in ('GREEDY','TAIL'):
        account=a.compressed(OUT/report['selections'][method]['path'],report['selections'][method])
        a.eq(account['context'],deployment['record'],'Same new admission for both arms')
        a.need(account['context_seal']==deployment['seal'] and report['selections'][method]['validation']=='accepted'
            and planbinding['seals'][method]==plans[method]['seal'],'Authenticated completed plan before replay')
        endpoint=a.read(OUT/'evaluation'/(method+'.json'));a.eq(endpoint,report['branches'][method],'Independent endpoint file/report join')
        result[method]=helper.audit_arm(a,method,account,endpoint,plans[method],prefix,frontier,side_index,deployment['record']['reward'])
        projections[method]=projection(account);chains[method]=own_chain(a,OUT,account,PINS['index'])
        details[method],cellsets[method],_=diagnostics(account)
        if method=='TAIL':tail_scoring=account['scorer_costs']
        if method=='GREEDY':
            initial=account['states'][0];profile_queries=initial['pages'][0]['base_emitted_count']+initial['pages'][0]['additional_emitted_count']+len(initial['goals'])
            a.need(len(initial['candidates'])==profile['distinct_actions'] and len(initial['pages'])==profile['pages'],'Profile corresponds to actual first same catalog')
        del account,endpoint;gc.collect()
    # Each historical gzip is likewise decoded and released separately.
    prior_specs=[('first_GREEDY11','GREEDY',11,False,0),('latest_GREEDY24','GREEDY',12,True,1),('latest_TAIL10','TAIL',10,False,2)]
    a.need(report['retained_trace_loads_attempted']==report['retained_trace_loads_completed']==3,'Exactly three post-plan trace reads')
    for label,method,n,complete,k in prior_specs:
        binding=report['retained_trace_transport'][k];a.need(binding['plans_json_sha256_before_decode']==planbinding['sha256'],'Old continuation consumed only after both plans sealed')
        path=ROOT/binding['path'];loaded=a.compressed(path,binding);account=loaded if complete else loaded['policy'];old=projection(account);new=projections[method]
        a.need(account['complete'] is complete and account['method']==method and len(old['history'])==12+n,'Exact retained trace scope')
        oldreport=a.read(path.parent/'report.json',index['input_metadata'][str((path.parent/'report.json').relative_to(ROOT))])
        oldchain=own_chain(a,path.parent,account,oldreport['source_index_sha256']);chain_equal(a,oldchain,chains[method],old,new)
        a.eq(old['packets'],new['packets'][:12+n],'Retained full physical packet prefix exact')
        a.eq(old['history'],new['history'][:12+n],'Retained full native history prefix exact')
        a.eq(old['states'],new['states'][:n],'Retained complete decision rows/order/scores/focus exact')
        claim=report['saved_trace_parity'][label]
        a.need(claim['accepted'] is True and claim['exact_full_prefix_actions']==12+n and claim['exact_committed_suffix_decisions']==n
            and claim['old_trace_used_by_planner'] is False,'Reported historical parity independently reproduced')
        for claimkey,local in [('packets_hash','packets'),('history_hash','history'),('decision_states_hash','states')]:
            a.need(claim[claimkey]==new['prefixes'][str(12+n)][local],'Exact reported prefix aggregate hash')
        del account,loaded,old,oldreport,oldchain;gc.collect()
    g=result['GREEDY']['outcomes'];t=result['TAIL']['outcomes'];delta={k:t[k]-g[k] for k in report['diagnosis']['deltas_tail_minus_greedy']}
    a.eq(delta,report['diagnosis']['deltas_tail_minus_greedy'],'Complete physical/return deltas')
    nonreg=delta['positive_target_source_cells_removed']>=0 and delta['target_removed_mm3']>=-1e-8 and delta['total_reward']>=-1e-8 and delta['normal_removed_mm3']<=1e-8
    strict=delta['positive_target_source_cells_removed']>0 or delta['target_removed_mm3']>1e-8 or delta['total_reward']>1e-8
    a.need(report['diagnosis']['comparison_admissible'] is True and report['diagnosis']['quality_nonregression'] is nonreg
        and report['diagnosis']['strict_quality_improvement'] is strict and report['diagnosis']['useful_gain'] is (nonreg and strict)
        and report['diagnosis']['tie_is_negative'] is True,'Independent original useful gate; negatives retained')
    a.close(g['total_reference_target_mm3'],t['total_reference_target_mm3'],'Unchanged full target denominator')
    reward=deployment['record']['reward']
    decomposition={'target_value':reward['target_per_mm3']*delta['target_removed_mm3'],
        'outside_cost':-reward['normal_per_mm3']*delta['normal_removed_mm3'],
        'action_cost':-reward['action_cost']*delta['nonstop_actions'],
        'switch_cost':-reward['tool_change_cost']*delta['tool_changes'],
        'travel_cost':-reward['motion_per_mm']*delta['complete_tool_path_length_mm']}
    a.close(sum(decomposition.values()),delta['total_reward'],'Exact return gap decomposition')
    costs=report['costs'];caps=report['config']['caps'];budget=report['native_budget']
    preview_total=sum(v.get('native_preview_entries',0) for k,v in costs.items() if k!='same045_source')
    a.need(preview_total==budget['native_preview_entries'] and budget['counting_reliable'] is True
        and budget['failure'] is None and budget['history_complete_caller_attestation'] is True,'Complete reliable query cost accounting')
    a.need(costs['first_suffix_catalog_profile']['native_preview_entries']==profile_queries
        and costs['first_suffix_catalog_profile'].get('native_transition_calls',0)==0
        and costs['first_suffix_catalog_profile'].get('policy_forward_calls',0)==0,'Profile creation fully charged; validation adds zero scientific calls')
    for method in result:a.need(costs[method+'_suffix_planning']['native_preview_entries']==result[method]['planning_preview_lower'],'Every discovered/catalog/goal preview counted')
    suffix=sum(r['suffix_transitions'] for r in result.values());transitions=36+2*suffix
    a.need(report['native_transition_counts']=={'attempted':transitions,'completed':transitions,'raised':0}
        and report['native_transition_attempts']==transitions and planbinding['native_transition_attempts_at_seal']==24+suffix,
        'Shared actual prefix and two independent suffixes charged once')
    a.need(report['total_frozen_policy_forwards']==result['TAIL']['tail_forwards']
        and report['shared_replay']['prefix_transitions']==12 and report['shared_replay']['suffix_transitions']=={'GREEDY':12,'TAIL':12},'Forward and full replay denominators')
    a.eq(report['focused_tail_scoring_costs'],tail_scoring,'Exact wrapper scoring report/full trace join')
    a.need(report['original_trained_world_scorer_costs']['policy_forwards_started']==0,'No unreported old-context scoring')
    priors=[]
    for key in ('preserved_failed_attempt','preserved_transport_failure','preserved_tail_timeout'):
        v=report[key];path=next(p for p,h in index['input_metadata'].items() if h==v['report_sha256']);r=a.read(ROOT/path,v['report_sha256'])
        a.need(r['status']=='failed_or_incomplete' and r['native_budget']['native_preview_entries']==v['native_previews']
            and r['native_transition_counts']==v['native_transition_counts'] and r['total_frozen_policy_forwards']==v['policy_forwards']
            and r['checkpoint_loads']==v['checkpoint_loads'],'Each original failed attempt and exact paid costs retained')
        a.close(r['worker_seconds'],v['worker_seconds'],'Prior worker time');priors.append(v)
    cumulative=report['failed_plus_recovery_costs']
    for key,actual in [('native_previews',preview_total+sum(v['native_previews'] for v in priors)),
        ('native_transition_attempts',transitions+sum(v['native_transition_counts']['attempted'] for v in priors)),
        ('frozen_policy_forwards',report['total_frozen_policy_forwards']+sum(v['policy_forwards'] for v in priors)),
        ('checkpoint_loads',2+sum(v['checkpoint_loads'] for v in priors))]:a.need(cumulative[key]==actual,'Cumulative '+key)
    a.need(cumulative['prior_failed_attempts']==3 and cumulative['prior_parent_seconds']==sum(v['parent_seconds'] for v in priors),'All three failures counted once')
    low=sum(v['worker_seconds'] for v in priors);a.need(low<=cumulative['worker_seconds']<=low+report['worker_seconds'],'Cumulative nested worker timestamp scope')
    a.need(report['worker_seconds']<caps['work_seconds'] and parent['elapsed_seconds']<caps['parent_wall_seconds']
        and max(report['self_peak_rss_bytes'],parent['sampled_peak_group_rss_bytes'])<=caps['worker_peak_rss_bytes']
        and preview_total<=caps['native_previews'] and transitions<=caps['native_transitions'],'Full unchanged resource caps')
    common=[]
    for x,y in zip(details['GREEDY'],details['TAIL']):
        if x['state']!=y['state'] or x['query']!=y['query']:break
        common.append({'absolute_action':x['absolute_action'],'same_action':x['action']==y['action'],'GREEDY':x,'TAIL':y})
        if x['action']!=y['action']:break
    sets={'common_removed_cells':len(cellsets['GREEDY']&cellsets['TAIL']),
        'only_GREEDY_cells':len(cellsets['GREEDY']-cellsets['TAIL']),'only_TAIL_cells':len(cellsets['TAIL']-cellsets['GREEDY']),
        'GREEDY_removed_set_hash':semantic(sorted(cellsets['GREEDY'])),'TAIL_removed_set_hash':semantic(sorted(cellsets['TAIL']))}
    return {'completed_comparison':True,'useful_gain':nonreg and strict,'outcomes':{m:result[m]['outcomes'] for m in result},
        'delta_tail_minus_greedy':delta,'return_gap_decomposition':decomposition,'removed_set_comparison':sets,
        'suffix_decisions':details,'common_states_until_first_divergence':common,
        'profile':{'reference_median_seconds':rm,'batched_median_seconds':bm,'ratio':bm/rm,'median_paired_saving_seconds':save,
            'gate':gate,'one_catalog_four_pairs':True,'validation_zero_scientific_calls':True,'creation_previews':profile_queries},
        'costs':{'current_previews':preview_total,'current_transitions':transitions,'current_forwards':report['total_frozen_policy_forwards'],
            'worker_seconds':report['worker_seconds'],'parent_seconds':parent['elapsed_seconds'],'peak_rss_bytes':parent['sampled_peak_group_rss_bytes'],
            'phases':costs,'scorer_costs':report['focused_tail_scoring_costs'],'cumulative_four_attempts':cumulative,
            'all_parent_seconds':parent['elapsed_seconds']+cumulative['prior_parent_seconds']},
        'claims':'Batched authentication improved one paired engineering microbenchmark and enabled this completed same-catalog comparison. Frozen TAIL did not improve learned planning quality. Same exposed TRAIN045 anatomy, no refit, no held-out/clinical/physical validation claim.',
        'limits':'Independent saved scalar/ranking/reward/hash arithmetic only; executed geometry certificates were produced by the worker independent full-history checker. No arrays, weights or neural predictions recomputed. State hashes/counter booleans authenticate the saved profile; this is one state, not generalized end-to-end speed evidence.'}

def main():
    output=HERE/'receipt.json'
    if output.exists():raise ValueError('Exclusive receipt; no retry overwrite')
    a=Audit();start=time.monotonic();receipt={'schema':'focused-batched-independent-result-v1','pins':PINS};code=0
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s saved audit')));signal.alarm(60)
    try:receipt.update(status='PASS_saved_result',findings=inspect(a))
    except BaseException as e:code=1;receipt.update(status='FAIL_reader_or_evidence',failure={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.alarm(0);source={p:v for p,v in a.files.items() if p.endswith('.py')}
        receipt.update(checks=a.checks,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checked_source_count=len(source),checked_source_digest=semantic(source),
            evidence={p:v for p,v in a.files.items() if p not in source},native_calls=0,model_calls=0,patient_payload_reads=0)
        raw=json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
        if len(raw)>65536:raise ValueError('64KiB receipt cap')
        with output.open('xb') as f:f.write(raw)
        print(json.dumps({'status':receipt['status'],'checks':a.checks,'seconds':receipt['elapsed_seconds'],'rss_bytes':receipt['peak_rss_bytes'],
            'receipt_sha256':hashlib.sha256(raw).hexdigest(),'receipt_bytes':len(raw)}))
    return code
if __name__=='__main__':raise SystemExit(main())
