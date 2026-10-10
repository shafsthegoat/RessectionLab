"""One bounded saved-only audit; no project imports, native chain, or stream replay."""
import ast, hashlib, json, math, os, resource, signal, stat, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
START=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60-second audit limit')))
signal.alarm(60)
FILES={}; ALLOWED={}; WRITES={str(OUT/'audit-result.json'),str(OUT/'result-summary.json')}
MAXRSS=256*1024**2

def check():
    assert time.monotonic()-START<60
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rss<=MAXRSS, ('observed RSS cap',rss)

def audit(event,args):
    if event=='open':
        p=args[0]
        if isinstance(p,int):return
        q=str(Path(p).absolute())
        flags=args[2]
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT):assert q in WRITES, ('write denied',q)
        else:assert q in ALLOWED, ('read denied',q)
    if event.startswith(('subprocess.','os.system','os.exec','os.spawn','socket.')):raise RuntimeError('process/network denied')
sys.addaudithook(audit)

def raw(path,sha=None,cap=8*1024**2,record=True):
    p=ROOT/path
    assert '..' not in p.parts and p.is_relative_to(ROOT)
    ALLOWED[str(p)]=True
    fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        st=os.fstat(f.fileno());assert stat.S_ISREG(st.st_mode) and st.st_size<=cap
        data=f.read(cap+1);assert len(data)==st.st_size and len(data)<=cap
    digest=hashlib.sha256(data).hexdigest()
    if sha:assert digest==sha,(path,digest,sha)
    if record:FILES[path]={'sha256':digest,'bytes':len(data)}
    check();return data

def obj(path,sha=None,cap=8*1024**2):return json.loads(raw(path,sha,cap),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
def bind(b):return obj(b['path'],b['sha256'])
def same(a,b,label=''):
    # Cross-language norm reduction can differ by <=4 ulps; this is comparison
    # tolerance ONLY. Every software pass/fail below uses exact frozen thresholds.
    if isinstance(a,float) and isinstance(b,(float,int)):
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=4*max(math.ulp(a),math.ulp(float(b))), (label,a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),(label,a.keys(),b.keys())
        for k in a:same(a[k],b[k],label+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),(label,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,label+'/'+str(i))
    else:assert a==b,(label,a,b)

def passed(m):return m['actual']<m['limit'] if m.get('comparison')=='lt' else m['actual']<=m['limit']
def run(b,n,s=60):return f'{b}:N{n}:S{s}:reference'

M='build/hbe-v5-native-manifest-preparation-v1/actual-input-manifest-accounting-v2.json'
m= obj(M,'4ba15123c7d0482a38576528b64a95d0b44bd7cccbd9851e8701fca1834fe69c')
old=obj('build/hbe-v5-native-manifest-preparation-v1/actual-input-manifest.json','ae9f8da5914b39e4f20f4f8b64f50cf9e6890250ac0e80cd9647580e07894449')
old['comparison_source_bindings']['scripts/mechanics_hbe_v5_native_comparison.py']='31fc3d66b401cf0c793e498292e8d44335e1e7d5ff11a3b1839238e86e6fc69e'
assert old==m
assert len(m['comparison_source_bindings'])==30
sources={p:raw(p,h).decode() for p,h in m['comparison_source_bindings'].items()}
raw('build/hbe-v5-native-manifest-preparation-v1/compare_worker.py','94e2e93cb439cf4602f95fa475c430f77f14428c37e2f4f07ae03d5953e2dacf')
base='build/hbe-v5-native-twelve-row-comparison-v1/attempt-02/'
r=obj(base+'receipt.json');c=obj(base+'comparison.json','6d3772eac38f8a1a210082a89cb00955f4a9b0d5304700fd0168711b9de99c12',16*1024**2)
raw(base+'readout-console.txt')
assert r['comparison_sha256']==FILES[base+'comparison.json']['sha256']
assert r['status']=='completed_numerical_comparison' and r['native_calls']==r['saved_stream_replays']==0
assert r['cleanup']==dict(contained=True,direct_child_reaped=True,errors=[],exit_code=0,fallback_used=False,remaining_members=[])
s=r['readout_stage'];assert s['status']=='completed_within_caps' and s['exit_code']==0 and s['kill_reason'] is None
assert s['elapsed_seconds']<=600 and s['peak_sampled_process_group_rss_bytes']<=3*1024**3
assert sum(FILES[base+x]['bytes'] for x in ('receipt.json','comparison.json','readout-console.txt'))<=16*1024**2
assert c['comparison_manifest']=={'path':M,'sha256':FILES[M]['sha256']}
assert c['comparison_source_bindings']==m['comparison_source_bindings']
for origins in (c['loaded_project_origins_before'],c['loaded_project_origins_after']):
    for name,p in origins.items():assert p in m['comparison_source_bindings'] or str(Path(p).relative_to(ROOT)) in m['comparison_source_bindings']
assert c['native_chain_verification_passes']==2
assert c['native_output_admitted'] and c['source_deck_bytes_authenticated'] and c['adapted_deck_authenticated']
for key in ('native_execution_released','calibration_released','measured_response_accessed','patient_data_accessed','continuum_error_bound'):assert c[key] is False
assert c['physical_validation_pass'] is None
study=obj('manifests/experiments/hbe-01-03-branch-calibration-v5.json','50c5dbc8c45279245a8dcd9b92d16e9499bad6bc61c6dfbd5b5e307af49a71ce')
prior=bind(study['previous_v4'])
ordered=[x['run_id'] for x in study['ordered_reference_runs']]
assert ordered==m['ordered_run_ids']==c['ordered_run_ids'] and len(ordered)==12
views={};row_summary={};receipts={}
for i,(spec,b) in enumerate(zip(study['ordered_reference_runs'],m['native_receipts'])):
    rid=spec['run_id'];entry=m['rows'][rid];receipt=bind(b);receipts[rid]=receipt
    d=receipt['saved_numerical_readout'] if i==0 else bind(entry['readout'])
    assert receipt['native_calls_attempted']==1 and receipt['run_id']==rid and receipt['no_retry'] is True
    assert d['run_id']==rid and d['steps']==spec['steps'] and d['frame_count']==spec['steps']+1
    half=spec['native_domain']=='lower_half_reconstructed'
    assert d['representation']==('reconstructed_full' if half else 'full_native_fixture')
    prov=d['provenance'];assert prov['source_binding_checked'] and prov['measured_response_accessed'] is False and prov['patient_data_accessed'] is False and prov['physical_validation_pass'] is None
    n=spec['steps'];endpoint=float(spec['endpoint_decimal_literal_m'])
    expected=[endpoint*(j/n) for j in range(n+1)];expected[-1]=endpoint
    assert d['load_coordinate_full_m']==expected
    f=d['applied_force_N'];p=d['probe_displacements_m']
    assert len(f)==len(p)==n+1 and all(math.isfinite(x) for x in f)
    assert all(len(state)==75 and all(len(v)==3 and all(math.isfinite(x) for x in v) for v in state) for state in p)
    criteria=d['criteria_max_ratio'];assert criteria and all(math.isfinite(x) and 0<=x<=1 for x in criteria.values())
    assert d['solver']['passed'] and d['solver']['normal_termination']
    assert len(d['solver']['states'])==n and all(q['actual_N2']<=q['limit_N2'] and q['passed'] for q in d['solver']['states'])
    for w in (d['full_energy_work'],d['native_energy_work']):
        if w is not None:assert w['passed'] and w['maximum_error_J']<=w['limit_J']
    assert (d['native_energy_work'] is not None)==half
    assert min(d['minimum_sampled_J'],d['minimum_logged_J'])>0 and d['numerical_passed'] is True and c['individual_stream_pass'][rid] is True
    coverage=c['coordinate_only_coverage'][rid];coords=prior['coordinates'][spec['branch']]['coordinates_m']
    assert len(coords)==30 and all(min(expected)<=x<=max(expected) for x in coords)
    assert coverage['row_count']==coverage['covered_row_count']==30 and coverage['outside_rows']==[] and coverage['coverage_tolerance_m']==0
    assert coverage['coordinate_sha256']==hashlib.sha256(json.dumps(coords,separators=(',',':')).encode()).hexdigest()
    views[rid]={'f':f[::n//60],'p':p[::n//60],'x':expected[::n//60]}
    row_summary[rid]={'native_domain':spec['native_domain'],'representation':d['representation'],'frame_count':n+1,'probe_count':75,'numerical_passed':True,'criteria_max_ratio':criteria,'minimum_sampled_J':d['minimum_sampled_J'],'minimum_logged_J':d['minimum_logged_J'],'full_work_maximum_error_J':d['full_energy_work']['maximum_error_J'],'full_work_limit_J':d['full_energy_work']['limit_J'],'maximum_solver_residual_ratio':max(q['actual_N2']/q['limit_N2'] for q in d['solver']['states'])}
    del d

def delta(a,b):
    x,y=views[a],views[b];assert x['x']==y['x'] and len(x['x'])==61
    df=[v-u for u,v in zip(x['f'],y['f'])]
    dp=[[[v-u for u,v in zip(p,q)] for p,q in zip(s,t)] for s,t in zip(x['p'],y['p'])]
    return df,dp,max(abs(x) for x in df),max(math.sqrt(sum(z*z for z in q)) for s in dp for q in s)

pair_summary={}
for branch,items in c['all_adjacent_signed_differences'].items():
    for key,pair in items.items():
        df,dp,fc,pc=delta(pair['coarse_run'],pair['fine_run']);kind=pair['kind']
        floor,rel,plim=(1.6e-7,.02,8e-6) if kind=='mesh' else (1.6e-8,.002,8e-7)
        flim=floor+rel*max(abs(x) for x in views[pair['fine_run']]['f'])
        assert df==pair['signed_force_difference_N'] and dp==pair['signed_probe_component_difference_m']
        for key2,val in [('maximum_force_change_N',fc),('maximum_probe_norm_change_m',pc),('force_limit_N',flim),('probe_norm_limit_m',plim)]:same(val,pair[key2],key2)
        assert pair['pairwise_limit_passed']==(fc<=flim and pc<=plim)
        pair_summary[branch+'/'+key]={k:v for k,v in pair.items() if not k.startswith('signed_')}
for k,pair in c['four_frozen_v4_pair_diagnostics'].items():
    df,dp,fc,pc=delta(pair['coarse_run'],pair['fine_run'])
    assert pair['signed_force_difference_N']==df and pair['signed_probe_component_difference_m']==dp
    same(fc,pair['maximum_force_change_N']);same(pc,pair['maximum_probe_norm_change_m'])
    assert pair['passed']==(fc<=pair['force_limit_N'] and pc<=pair['probe_norm_limit_m'])

def group(a,b,d,kind='mesh'):
    _,_,oldf,oldp=delta(a,b);_,_,fc,pc=delta(b,d)
    floor,rel,plim=(1e-5*(1000.*.004**2),.02,.002*.004) if kind=='mesh' else (1e-6*(1000.*.004**2),.002,.0002*.004)
    result={'reaction':{'actual':fc,'limit':floor+rel*max(abs(x) for x in views[d]['f']),'units':'N'},'motion':{'actual':pc,'limit':plim,'units':'m'}}
    if kind=='mesh':
        for name,new,old,small,unit in [('reaction_trend',fc,oldf,1e-5*(1000.*.004**2),'N'),('motion_trend',pc,oldp,plim,'m')]:
            result[name]={'actual':max(new,old),'limit':small,'units':unit} if max(new,old)<=small else {'actual':new,'limit':old,'units':unit,'comparison':'lt'}
    return result

def groups(branch,source,mixed=False):
    for label,saved in source.items():
        ns=[int(x[1:]) for x in label.split('-')];ids=[run(branch,n,120 if mixed and n==ns[-1] and n==(36 if branch=='compression' else 24) else 60) for n in ns]
        actual=group(*ids);same(actual,saved['metrics'],label)
        assert saved['passed']==all(passed(x) for x in actual.values())
for branch in ('compression','tension'):
    groups(branch,c[branch+'_spatial_groups']);groups(branch,c[branch+'_temporal']['mixed_spatial_groups'],True)
    n=36 if branch=='compression' else 24;metrics=group(run(branch,n),run(branch,n),run(branch,n,120),'step')
    same(metrics,c[branch+'_temporal']['metrics']);assert c[branch+'_temporal']['passed']==all(passed(x) for x in metrics.values())

# Reuse precisely the frozen scalar order solve, without importing its module.
# No I/O-capable statements or imports are compiled.
path='scripts/mechanics_hbe_halfheight_global_n36_readout.py'
tree=ast.parse(sources[path]);names={'order_ratio','unequal_order','state_diagnostic'}
functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];assert len(functions)==3
triplets=((8,12,16),(12,16,24),(16,24,32),(24,32,36))
ns={'math':math,'TRIPLETS':triplets,'FORCE_FLOOR_N':1.6e-7}
exec(compile(ast.Module(body=functions,type_ignores=[]),path,'exec'),ns)
conditionals=[]
for mixed,saved in [(False,c['compression_conditional']),(True,c['compression_temporal']['mixed_conditional'])]:
    rid=run('compression',36,120 if mixed else 60);allowance=1.6e-7+.02*max(abs(x) for x in views[rid]['f'])
    states=[ns['state_diagnostic']({n:views[run('compression',n,120 if mixed and n==36 else 60)]['f'][i] for n in (8,12,16,24,32,36)},index=i,force_limit_N=allowance) for i in range(61)]
    assert states==saved['states']
    unresolved=[q['state'] for q in states[1:] if q['status']!='resolved_conditional_model']
    exceed=[q['state'] for q in states[1:] if q['remaining_within_allowance'] is False]
    unstable=[q['state'] for q in states[1:] if q['increasing_order_drift'] is True or q['force_limit_instability'] is True]
    endpoint=states[-1]['orders']['N24-N32-N36']['status']=='eligible'
    for key,v in [('force_allowance_N',allowance),('unresolved_nonrest_states',unresolved),('remaining_exceedance_states',exceed),('instability_states',unstable),('endpoint_latest_triplet_eligible',endpoint),('passed_conditional_screen',endpoint and not unresolved and not exceed and not unstable)]:same(v,saved[key],key)
    conditionals.append({'mixed_S120':mixed,'force_allowance_N':allowance,'maximum_two_latest_limit_envelope_N':max(q['two_latest_limit_envelope_N'] or 0 for q in states),'maximum_limit_disagreement_N':max(q['latest_limit_disagreement_N'] or 0 for q in states),'unresolved_nonrest_states':unresolved,'remaining_exceedance_states':exceed,'instability_states':unstable,'endpoint':states[-1],'passed':saved['passed_conditional_screen']})
def cls(q):return {k:q[k] for k in ('status','increasing_order_drift','force_limit_instability','remaining_within_allowance')}|{'triplet_eligibility':{k:v['status'] for k,v in q['orders'].items()}}
changed=[i for i,(a,b) in enumerate(zip(c['compression_conditional']['states'],c['compression_temporal']['mixed_conditional']['states'])) if cls(a)!=cls(b)]
assert changed==c['compression_temporal']['state_classification_changes']
for branch in ('compression','tension'):
    before=c[branch+'_spatial_groups'];after=c[branch+'_temporal']['mixed_spatial_groups']
    changes=[g+'/'+k for g in before for k in before[g]['metrics'] if passed(before[g]['metrics'][k])!=passed(after[g]['metrics'][k])]
    assert changes==c[branch+'_temporal']['group_classification_changes']
    mixed=not changes and (not changed and c['compression_temporal']['mixed_conditional']['passed_conditional_screen'] if branch=='compression' else True)
    assert mixed==c[branch+'_temporal']['mixed_passed']
latest=c['compression_spatial_groups']['N24-N32-N36']['metrics'];req=c['compression_required_spatial']
assert req['reaction']==latest['reaction'] and req['motion']==latest['motion'] and req['passed']==all(passed(latest[k]) for k in ('reaction','motion'))
gates={'all_12_individual_streams':all(c['individual_stream_pass'].values()),'four_frozen_pairs':all(q['passed'] for q in c['four_frozen_v4_pair_diagnostics'].values()),'compression_latest_reaction_motion':req['passed'],'two_tension_triplets':all(q['passed'] for q in c['tension_spatial_groups'].values()),'compression_all_state_conditional':c['compression_conditional']['passed_conditional_screen'],'compression_temporal':c['compression_temporal']['passed'],'tension_temporal':c['tension_temporal']['passed'],'compression_mixed_classifications':c['compression_temporal']['mixed_passed'],'tension_mixed_classifications':c['tension_temporal']['mixed_passed']}
assert all(gates.values())==c['diagnostic_screen_passed']==c['numerical_qualification_passed']==r['numerical_qualification_passed']
# Aggregate counters from compact final receipt and four actual policy sidecars.
last=receipts[ordered[-1]];chain=c['original_chain_accounting'];closed=FILES[m['native_receipts'][-1]['path']]['bytes']+sum(v['bytes'] for v in last['output_bindings'].values())
for key,v in [('native_seconds',last['prior_native_wall_seconds']+last['native_stage']['elapsed_seconds']),('readout_seconds',last['prior_readout_wall_seconds']+last['readout_stage']['elapsed_seconds']),('prep_seconds',last['prior_prep_wall_seconds']+last['prep_elapsed_seconds']),('output_bytes',last['prior_active_output_bytes']+closed)]:assert chain[key]==v,(key,chain[key],v)
assert chain['native_calls']==12 and chain['supplement_replay_calls']==1
assert chain['sha256']==[x['sha256'] for x in m['native_receipts']]
extra_t=0.;extra_b=0
for rid in ordered[8:]:
    desc=m['rows'][rid]['policy_extension'];side=bind(desc['sidecar']);charge=side['extension_resource_charge'];native=receipts[rid]
    inc=charge['launcher_inclusive_prep_seconds_with_finalization_reserve']-native['prep_elapsed_seconds'];reserved=charge['sidecar_reserved_output_bytes']
    saved=c['extension_policy_accounting']['extensions'][rid]
    assert saved['incremental_prep_seconds']==inc and saved['incremental_output_bytes']==reserved
    assert saved['charged_aggregate']==charge['aggregate_with_extension']
    extra_t+=inc;extra_b+=reserved
assert extra_t==c['extension_policy_accounting']['incremental_prep_seconds'] and extra_b==c['extension_policy_accounting']['incremental_output_bytes']
combined=dict(chain)
for k in ('prep_seconds','combined_wall_seconds'):combined[k]+=extra_t
for k in ('output_bytes','combined_output_bytes'):combined[k]+=extra_b
assert combined==c['combined_policy_accounting']
assert combined['native_seconds']<=9600 and combined['readout_seconds']<=6600 and combined['prep_seconds']<=1800 and combined['combined_wall_seconds']<=18000 and combined['combined_output_bytes']<=6*1024**3
hist=c['historical_N32_negative'];assert hist['historical_only_not_v5_veto'] is True
assert c['exact_N12_exception']['original_receipt_sha256']==m['native_receipts'][1]['sha256'] and receipts[ordered[1]]['status']=='failed_or_incomplete'
assert c['known_timing_gap']
# Re-read only already opened compact files; never source paths listed inside a receipt.
for path,b in list(FILES.items()):raw(path,b['sha256'],max(b['bytes'],1),record=False)
summary={'schema':'hbe-v5-native-twelve-row-compact-numerical-summary-v1','comparison_sha256':FILES[base+'comparison.json']['sha256'],'manifest_sha256':FILES[M]['sha256'],'receipt_sha256':FILES[base+'receipt.json']['sha256'],'numerical_qualification_passed':c['numerical_qualification_passed'],'physical_validation_pass':None,'calibration_released':False,'measured_response_accessed':False,'patient_data_accessed':False,'continuum_error_bound':False,'scope':'Software numerical criteria for declared axial specimen problem; conditional compression model, not a continuum error bound, physical validation, calibration release or patient evidence.','gate_conjunction':gates,'rows':row_summary,'pair_metrics':pair_summary,'compression_spatial_groups':c['compression_spatial_groups'],'tension_spatial_groups':c['tension_spatial_groups'],'compression_required_spatial':req,'coarse_ungated_adjacent_exceedances':c['ungated_adjacent_limit_exceedances'],'compression_conditionals':conditionals,'temporal':{b:{k:v for k,v in c[b+'_temporal'].items() if k not in ('mixed_conditional',)} for b in ('compression','tension')},'original_chain_accounting':chain,'extension_policy_accounting':c['extension_policy_accounting'],'combined_policy_accounting':combined,'historical_N32_negative':hist,'exact_N12_exception':c['exact_N12_exception'],'known_timing_gap':c['known_timing_gap'],'comparison_execution':r['readout_stage'],'comparison_cleanup':r['cleanup'],'audit_method':'Independent stdlib pair/group formulas and gate conjunction; source-isolated unchanged order_ratio/unequal_order/state_diagnostic reproduce all122 compression diagnostic states exactly. No canonical comparator, source-chain authentication or saved-stream checker rerun.'}
for name,value in [('result-summary.json',summary),('audit-result.json',{'status':'PASS','files_pre_post_identical':FILES,'file_count':len(FILES),'canonical_comparator_calls':0,'native_calls':0,'stream_replays':0,'project_imports':0,'measured_response_reads':0,'patient_reads':0,'source_manifest_change_only':'native comparator a00d3f0e to31fc3d66','exact_scalar_state_count':122,'independently_checked_signed_pairs':10,'scope_limits':{'wall_seconds':60,'observed_peak_rss_bytes':MAXRSS},'elapsed_seconds':time.monotonic()-START,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cross_implementation_float_comparison':'at most4ULP for scalar norm results; no decision threshold tolerance','numerical_qualification_passed':all(gates.values())})]:
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with open(OUT/name,'xb') as f:f.write(data)
check();print(json.dumps({'status':'PASS','elapsed_seconds':time.monotonic()-START,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'files':len(FILES)}))
