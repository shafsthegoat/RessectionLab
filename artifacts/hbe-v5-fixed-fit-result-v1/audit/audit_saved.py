"""Independent stdlib-only saved JSON audit; never opens raw native/response data."""
from pathlib import Path
import hashlib,json,math,os,resource,signal,stat,sys,time

ROOT=Path.cwd().resolve();OUT=ROOT/'build/hbe-v5-fixed-fit-result-independent-v1'
RUN=ROOT/'outputs/mechanics/hbe-v5-fixed-fit-confirmation-v1/attempt-01'
REL={'path':'build/hbe-v5-fixed-fit-root-release-v1.json','sha256':'97c182618a90e746971b7375f758dbe5c8b13b7ef68f5c1e696522321146fc3d'}
FIT='b6b23a297f81b9481a6a0c7b584c8d1d84907baf86dd4ecb62b78474caf9666a'
MU=715.361571139082;SCALE=.715361571139082;R=.004;H=.00489159
started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(RuntimeError('60second audit limit')));signal.alarm(60)
def need(value,reason):
    if not value:raise ValueError(reason)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
blocked=[]
def guard(event,args):
    if event in {'subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp','socket.connect','socket.bind','socket.getaddrinfo'}:
        blocked.append(event);raise RuntimeError('No child/network in saved audit')
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).resolve()
        if (any(p.is_relative_to(ROOT/x) for x in ('data','sources')) or
            p.suffix.lower() in {'.log','.feb','.csv','.mat','.gz','.zip','.tar'} or
            p.name=='fit.json' or (p.name=='console.txt' and p.is_relative_to(RUN))):
            blocked.append(str(p));raise RuntimeError('Forbidden raw/measured/saved-fit payload')
sys.addaudithook(guard)
seen={};cache={}
def read(path,expected=None,maximum=4*1024**2):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    need(p.is_relative_to(ROOT) and '..' not in p.parts,'contained path')
    need(not any(x.is_symlink() for x in (p,*p.parents)),'no linked input')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd);need(stat.S_ISREG(a.st_mode) and a.st_size<=maximum,'bounded regular metadata')
        with os.fdopen(fd,'rb',buffering=0,closefd=False) as f:raw=f.read(maximum+1)
        b=os.fstat(fd)
        ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(ident(a)==ident(b)==ident(p.lstat()) and len(raw)==a.st_size,'stable metadata read')
    finally:os.close(fd)
    digest=sha(raw);need(expected is None or digest==expected,'binding '+str(p))
    key=str(p.relative_to(ROOT));item={'sha256':digest,'bytes':len(raw)}
    need(key not in seen or seen[key]==item,'metadata changed')
    seen[key]=item;cache[key]=raw
    return raw
def strict(raw):
    def pairs(rows):
        d={}
        for k,v in rows:need(k not in d,'duplicate key');d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def load(path,expected=None):return strict(read(path,expected))
def bound(binding):return load(binding['path'],binding['sha256'])
def binding(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p;k=str(p.relative_to(ROOT))
    if k not in seen:read(p)
    return {'path':k,'sha256':seen[k]['sha256']}
def near(a,b):need(math.isclose(a,b,rel_tol=5e-13,abs_tol=1e-25),'saved arithmetic disagreement')
def finite(v):
    if isinstance(v,float):need(math.isfinite(v),'nonfinite result')
    elif isinstance(v,dict):
        for x in v.values():finite(x)
    elif isinstance(v,list):
        for x in v:finite(x)

release=bound(REL);decl=bound(release['declaration']);caps=release['caps']
review=bound(release['independent_source_review'])
candidate={**release,'execution_released':False,'source_commit':None,'independent_source_review':None}
need(release['execution_released'] is True and release['source_commit']=='d09c78d8a90bbc1d6e507499c43c0db145b9f45e','root release/commit')
need(review['decision']=='GO' and review['release_candidate_sha256']==sha(canonical(candidate)),'exact prior source review')
need(caps==decl['caps'] and decl['authorized_measured_members']==[] and decl['donor_role']=='DEVELOPMENT','frozen no-measured scope')
for p,d in release['source_bindings'].items():read(p,d,1024**2)
need(len(release['source_bindings'])==44,'exact source closure count')
pub=load(RUN/'publication.json');terminal=load(RUN/'terminal.json');intent=load(RUN/'intent.json')
numeric=load(RUN/'numerical.json');pred=load(RUN/'predictions.json');freeze=load(RUN/'freeze.json')
ledger=load(RUN/'continuation-ledger.json');executions=load(RUN/'executions.json');prepared=load(RUN/'prepared.json')
for value in (pub,terminal,numeric,pred,freeze,executions,prepared):finite(value)
need(pub['accepted'] is True and terminal['status']=='completed_fitted_confirmation_and_prediction_freeze','accepted publication')
need(pub['release']==terminal['release']==intent['release']==freeze['release']==REL,'consistent release')
need(intent['source_commit']==release['source_commit'] and caps==terminal['caps']==intent['caps'],'caps/source')
need(pub['outputs']=={**terminal['output_inventory'],'terminal.json':seen[str((RUN/'terminal.json').relative_to(ROOT))]},'publication/terminal seal chain')
expected=set(pub['outputs'])|{'publication.json'}
need(len(expected)==25,'exact25 outputs')
stats={}
for p in RUN.rglob('*'):
    s=p.lstat();need(stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode),'no links/special files')
    if stat.S_ISREG(s.st_mode):stats[str(p.relative_to(RUN))]=s
need(set(stats)==expected and not (RUN/'finalization-failure.json').exists(),'complete inventory/no failure')
metadata_hashed=[];raw_not_opened=[]
for name,row in pub['outputs'].items():
    need(stats[name].st_size==row['bytes'],'published file size')
    if name.endswith('.json') or name.endswith('readout-console.txt'):
        read(RUN/name,row['sha256']);metadata_hashed.append(name)
    else:raw_not_opened.append(name)
total=sum(s.st_size for s in stats.values());need(total<=caps['new_output_bytes'],'final aggregate output cap')
need(0<terminal['elapsed_before_terminal_seconds']<pub['elapsed_seconds']<caps['total_seconds'],'lifecycle budget')
need(set(terminal['stages'])=={'prepare','compression','tension','readout'},'exact four stages')
stage_summary={}
for name,entry in terminal['stages'].items():
    kind='native' if name in ('compression','tension') else 'readout'
    record=entry['receipt'];c=entry['cleanup'];s=record[kind+'_stage']
    need(c=={'contained':True,'direct_child_reaped':True,'errors':[],'exit_code':0,'fallback_used':False,'remaining_members':[]},'clean owned child')
    need(record['release']==REL and record['phase']==name and record[kind+'_calls_attempted']==1,'one owned invocation')
    need(s['status']=='completed_within_caps' and s['exit_code']==0 and s['kill_reason'] is None,'successful owned child')
    limit=caps['native_seconds'][name] if kind=='native' else caps['preparation_seconds' if name=='prepare' else 'readout_seconds']
    need(0<s['elapsed_seconds']<s['wall_cap_seconds']<=limit,'stage wall limit')
    need(0<s['peak_sampled_process_group_rss_bytes']<=s['sampled_process_group_rss_cap_bytes']==caps['sampled_group_rss_bytes'],'RSS cap')
    need(0<=s['peak_sampled_active_output_bytes']<=s['active_output_cap_bytes'],'stage output cap')
    directory=name if kind=='native' else name+'-stage'
    need(load(RUN/directory/'receipt.json')==record,'persisted stage receipt')
    if kind=='native':
        need(s['active_output_cap_bytes']==caps['branch_output_bytes'][name],'branch output cap identity')
        need(sum(v.st_size for k,v in stats.items() if k.startswith(name+'/'))<=s['active_output_cap_bytes'],'closed branch output cap')
        need(s['command']==executions[name]['command'] and s['command'][1:]==['-noconfig','-no_title','-i','specimen.feb','-o','solver.log'],'exact native command')
        need(executions[name]['runtime_identity']==decl['runtime_identity'] and executions[name]['supervision']==entry,'runtime/supervision')
    else:
        command=s['command'];need(command[1:5]==['-I','-S','-B','-X'],'isolated worker')
        need(command[command.index('--release')+1]==REL['path'] and command[command.index('--release-sha256')+1]==REL['sha256'],'worker release')
    stage_summary[name]={k:s[k] for k in ('elapsed_seconds','wall_cap_seconds','peak_sampled_process_group_rss_bytes','sampled_process_group_rss_cap_bytes','peak_sampled_active_output_bytes','active_output_cap_bytes','pid')}
need(len({x['pid'] for x in stage_summary.values()})==4,'four distinct owned children')
need(sum(x['elapsed_seconds'] for x in stage_summary.values())<terminal['elapsed_before_terminal_seconds'],'elapsed stage accounting')
for key in ('fit_calls','mesher_calls','held_out_member_reads','measured_member_reads'):need(terminal[key]==0,'no unexpected work')
need(terminal['native_calls']==2 and terminal['native_calls_accounting']=='attempted','two native calls')
need(numeric['passed'] is True and numeric['fixed_fit_sha256']==FIT and numeric['mu_Pa']==MU and numeric['states_per_native_run']==121,'fixed paired confirmation')
need(numeric['physical_validation_pass'] is None and numeric['full_native_fine_equivalence_claim'] is False,'limited numerical claim')
criteria_names={'energy_scale_two','force_bottom_scale_one','force_midplane_scale_one','full_force_balance','full_force_sign_fixture','full_free_dof_reaction','full_moment_balance','full_prescribed_motion','full_work_energy','native_force_balance','native_free_dof_reaction','native_moment_balance','native_prescribed_motion','native_primitive_consistency','native_work_energy','solver_residual'}
branch_summary={}
for branch,row in numeric['branches'].items():
    need(branch in {'compression','tension'} and row['passed'] is True,'both axial checks')
    d=row['readout'];q=decl['references'][branch];ref=bound(q['readout']);e=executions[branch]
    need(row['reference']==q['readout'] and row['execution']==e and e['fixed_fit_sha256']==FIT,'reference/execution identity')
    need(d['run_id']==q['fitted_run_id']==e['run_id'] and d['steps']==120 and d['frame_count']==121 and d['representation']=='reconstructed_full','complete represented run')
    need(d['numerical_passed'] is True and d['minimum_sampled_J']>0 and d['minimum_logged_J']>0,'positive sampled/logged determinants')
    need(d['logged_sed_used_for_energy_gate'] is False and d['sampled_J_positivity_is_not_everywhere_proof'] is True,'honest positivity/work scope')
    need(set(d['criteria_max_ratio'])==criteria_names and all(0<=v<=1 for v in d['criteria_max_ratio'].values()),'unchanged16 criterion limits')
    provenance=d['provenance'];need(provenance['fixed_fit_sha256']==FIT and provenance['native_output_observed'] is True and provenance['generated_fixture_only'] is False and provenance['source_binding_checked'] is True,'actual fixed native provenance')
    for key in ('measured_response_accessed','patient_data_accessed'):need(provenance[key] is False,'no response/patient access')
    need(provenance['physical_validation_pass'] is None and provenance['output_origin']=='owned_fixed_fit_native','native-only claim')
    need(provenance['primitive_bindings']==e['primitive_bindings'],'primitive identity')
    for key,b in e['primitive_bindings'].items():need(b=={'path':str((RUN/branch/(key+'.log')).relative_to(ROOT)),'sha256':pub['outputs'][branch+'/'+key+'.log']['sha256']},'saved primitive binding')
    need(e['deck']==prepared['branches'][branch]['deck'] and provenance['adapted_deck_sha256']==e['deck']['sha256']==pub['outputs'][branch+'/specimen.feb']['sha256'],'deck identity')
    contract=prepared['branches'][branch]['contract'];need(contract['mu_Pa']==MU and contract['fixed_fit_sha256']==FIT,'fitted contract')
    x=d['load_coordinate_full_m'];need(x==ref['load_coordinate_full_m']==contract['full_coordinates_m'],'unchanged reference grid')
    need(len(x)==121 and x[0]==0 and x[-1]=={'compression':-.00073726,'tension':.0007360099999999}[branch],'exact endpoint')
    need(all((b-a)*x[-1]>0 for a,b in zip(x,x[1:])),'monotone load')
    for key in ('applied_force_N','energy_J','probe_displacements_m'):need(len(d[key])==len(ref[key])==121,'complete saved traces')
    need(all(len(p)==75 and all(len(v)==3 for v in p) for p in d['probe_displacements_m']),'75 three-component probes')
    solver=d['solver'];need(solver['normal_termination'] is True and solver['passed'] is True and len(solver['states'])==120,'solver residual completeness')
    ratios=[]
    for i,s in enumerate(solver['states'],1):
        need(s['fraction']==i/120 and s['declared_rtol']==1e-8 and all(s[k]>=0 for k in ('initial_N2','actual_N2','reported_required_N2')),'solver state schema')
        expected=max(1e-8*s['initial_N2'],(1e-10*MU*R**2)**2)*(1+2e-5)
        near(s['limit_N2'],expected)
        need(abs(s['reported_required_N2']-1e-8*s['initial_N2'])<=2e-5*max(s['reported_required_N2'],1e-8*s['initial_N2']),'required residual consistency')
        need(s['passed'] is True and s['actual_N2']<=expected,'residual pass');ratios.append(s['actual_N2']/expected)
    near(max(ratios),d['criteria_max_ratio']['solver_residual'])
    for which,height in (('full',H),('native',H/2)):
        w=d[which+'_energy_work'];work=w['work_J'];energy=w['energy_change_J']
        need(len(work)==len(energy)==121 and work[0]==energy[0]==0,'complete work traces')
        error=max(abs(a-b) for a,b in zip(work,energy));limit=1e-6*(MU*R*R*height)+.02*max(max(map(abs,work)),max(map(abs,energy)))
        near(w['maximum_error_J'],error);near(w['limit_J'],limit)
        need(w['passed'] is True and error<=limit,'work gate');near(error/limit,d['criteria_max_ratio'][which+'_work_energy'])
        if which=='full':
            integral=0.;force=d['applied_force_N'];energies=d['energy_J']
            for i in range(121):
                if i:integral+=(x[i]-x[i-1])*(force[i]+force[i-1])/2
                near(work[i],integral);near(energy[i],energies[i]-energies[0])
    need(set(row['scale'])=={'native','reconstructed_full'},'both scale representations')
    for representation,metrics in row['scale'].items():
        need(set(metrics)=={'motion','reaction','torque'},'scale metrics')
        for key,m in metrics.items():
            need(0<=m['actual']<=m['limit'] and m['limit']>0,'scale pass')
            if key!='motion':need(m['limit']==1 and m['units']=='ratio_to_declared_scale_tolerance','unchanged normalized scale tolerance')
            else:need(m['units']=='m' and m['limit']>=1e-6*R,'motion scale base floor')
        if representation=='reconstructed_full':near(metrics['motion']['limit'],1e-6*R+1e-5*abs(x[-1]))
    force_error=max(abs(a-SCALE*b) for a,b in zip(d['applied_force_N'],ref['applied_force_N']))
    energy_error=max(abs(a-SCALE*b) for a,b in zip(d['energy_J'],ref['energy_J']))
    probe_error=max(abs(a-b) for da,db in zip(d['probe_displacements_m'],ref['probe_displacements_m']) for va,vb in zip(da,db) for a,b in zip(va,vb))
    need(probe_error<=row['scale']['reconstructed_full']['motion']['limit'],'saved probes satisfy motion limit')
    branch_summary[branch]={'run_id':d['run_id'],'frame_count':121,'step_count':120,'numerical_passed':True,
        'minimum_sampled_J':d['minimum_sampled_J'],'minimum_logged_J':d['minimum_logged_J'],
        'criteria_max_ratio':d['criteria_max_ratio'],'scale':row['scale'],
        'full_work_error_J':d['full_energy_work']['maximum_error_J'],'full_work_limit_J':d['full_energy_work']['limit_J'],
        'native_work_error_J':d['native_energy_work']['maximum_error_J'],'native_work_limit_J':d['native_energy_work']['limit_J'],
        'saved_observable_scaling':{'maximum_force_error_N':force_error,'maximum_energy_error_J':energy_error,
                                    'maximum_probe_component_difference_m':probe_error,'new_tolerance_imposed':False}}
need(set(branch_summary)=={'compression','tension'},'exact two branches')
need(freeze['fixed_fit']==decl['fixed_fit'] and freeze['mu_Pa']==MU and freeze['scale']==SCALE,'fixed parameter freeze')
for k in ('refit_calls','mesher_calls','measured_member_reads','held_out_member_reads'):need(freeze[k]==0,'freeze no extra access')
need(freeze['native_calls']==2 and freeze['fitted_executions']==executions and freeze['held_out_access_released'] is False and freeze['patient_tool_mechanics_admitted'] is False and freeze['physical_validation_pass'] is None,'freeze scope')
need(freeze['numerical']==binding(RUN/'numerical.json') and freeze['predictions']==binding(RUN/'predictions.json') and terminal['freeze']==binding(RUN/'freeze.json'),'freeze content bindings')
need(freeze['roles']==decl['evidence']['roles'] and freeze['protocol']==decl['evidence']['protocol'],'same roles/protocol')
bound(freeze['roles']);bound(freeze['protocol'])
prior=bound(decl['fixed_fit']['publication']);need(prior['accepted'] is True and prior['outputs']['fit.json']['sha256']==FIT,'accepted prior fit')
oldledger=read(freeze['original_fit_ledger']['path'],freeze['original_fit_ledger']['sha256'])
need(prior['outputs']['access.jsonl']['sha256']==freeze['original_fit_ledger']['sha256'],'prior access ledger identity')
events=[strict(line) for line in oldledger.splitlines() if line]
need(all('torsion' not in str(e).lower() for e in events),'no prior heldout event')
need(ledger=={'phase':'freeze_saved','freeze':binding(RUN/'freeze.json'),'previous_fit_ledger':freeze['original_fit_ledger'],'measured_member_reads':0,'refit_calls':0},'chronological successor ledger')
for key in ('independent_audit','independent_report'):
    b=decl['fixed_fit'][key];read(b['path'],b['sha256'])
need(freeze['calibration_quality']=='unchanged_poor_within_specimen_fit_not_physical_validation','poor fit preserved')
v3=load('manifests/experiments/hbe-01-03-branch-calibration-v3.json')
registry={(x['path'],x['sha256']) for x in freeze['original_torsion_bindings']}
need(pred['schema']=='hbe-torsion-prediction-v1' and set(pred['reference'])==set(pred['fitted'])=={'torsion_neg','torsion_pos'},'exact original torsion branches')
prediction_summary={}
for branch in ('torsion_neg','torsion_pos'):
    b=v3['torsion_references'][branch]['readout'];need((b['path'],b['sha256']) in registry,'original torsion bound by freeze')
    original=bound(b);a=pred['reference'][branch];f=pred['fitted'][branch]
    need(a['coordinate']==original['load_coordinate'] and a['response']==original['applied_torque_Nm'],'original prediction source unchanged')
    need(f['coordinate']==a['coordinate'] and f['response']==[SCALE*v for v in a['response']] and f['response_unit']=='Nm','exact one-scale no-offset torsion prediction')
    need(len(f['coordinate'])==len(f['response'])==121 and f['coordinate'][0]==0,'complete frozen prediction')
    prediction_summary[branch]={'points':121,'reference_readout':b,'exact_saved_scaled_response':True,
       'coordinate_preserved':True,'response_unit':'Nm','measured_torque_read':False}
for first,second in (('numerical.json','freeze.json'),('predictions.json','freeze.json'),('freeze.json','continuation-ledger.json'),('continuation-ledger.json','terminal.json'),('terminal.json','publication.json')):
    need(stats[first].st_mtime_ns<=stats[second].st_mtime_ns,'filesystem chronology corroboration')
snap=dict(seen)
for p,row in snap.items():read(p,row['sha256'])
need(snap==seen,'audit input closure stable')
elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
need(elapsed<60 and peak<512*1024**2 and not blocked,'saved audit resource/access bound')
summary={'schema':'hbe-v5-fixed-fit-saved-result-summary-v1','decision':'PASS','release':REL,'source_commit':release['source_commit'],
 'publication':binding(RUN/'publication.json'),'terminal':binding(RUN/'terminal.json'),'numerical':binding(RUN/'numerical.json'),
 'freeze':binding(RUN/'freeze.json'),'predictions':binding(RUN/'predictions.json'),'continuation_ledger':binding(RUN/'continuation-ledger.json'),
 'fixed_fit_sha256':FIT,'mu_Pa':MU,'scale':SCALE,'branches':branch_summary,'frozen_torsion':prediction_summary,
 'stages':stage_summary,'publication_elapsed_seconds':pub['elapsed_seconds'],'lifecycle_limit_seconds':caps['total_seconds'],
 'final_output_bytes':total,'aggregate_output_limit_bytes':caps['new_output_bytes'],'output_file_count':25,
 'native_calls':2,'refit_calls':0,'measured_member_reads':0,'held_out_member_reads':0,
 'physical_validation_pass':None,'fine_full_native_equivalence_claim':False,
 'calibration_quality':'unchanged_poor_within_specimen_fit_not_physical_validation',
 'next_measured_access_released':False,
 'limits':'Saved JSON arithmetic/receipt audit only. No raw primitive rehash/replay; all-node motion/reaction/torque maxima and frame gates are authenticated producer results, not independently recomputed here.'}
(OUT/'result-summary.json').write_bytes(canonical(summary))
audit={'schema':'hbe-v5-fixed-fit-saved-audit-v1','decision':'PASS','elapsed_seconds':elapsed,'peak_self_rss_bytes':peak,
       'source_and_metadata_files_unchanged':len(snap),'input_bindings':snap,'output_stat_inventory':{k:{'bytes':v.st_size,'mtime_ns':v.st_mtime_ns} for k,v in stats.items()},
       'published_small_files_rehashed':metadata_hashed,'published_raw_files_stat_only_not_opened':raw_not_opened,
       'blocked_access_attempts':blocked,'native_calls_by_audit':0,'raw_primitive_reads':0,'measured_member_reads':0,'fit_calls':0,
       'current_runtime_binary_rehash':False,'result_summary_sha256':sha((OUT/'result-summary.json').read_bytes())}
(OUT/'audit-result.json').write_bytes(canonical(audit))
signal.alarm(0)
print(json.dumps({k:audit[k] for k in ('decision','elapsed_seconds','peak_self_rss_bytes','source_and_metadata_files_unchanged','blocked_access_attempts')}))
