"""Saved-fit arithmetic only. No estimator, native runtime, archive or measured-member reads."""
import hashlib,json,math,os,resource,signal,stat,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE='build/hbe-v5-axial-calibration-v1/attempt-01/'
RB={'path':'build/hbe-v5-axial-calibration-root-release-v1/root-release.json','sha256':'90e1fa69a75dfe5dec3fd205c46c0405266db40fd0be1c9a22611a12e468b958'}
start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('saved-fit audit30s')));signal.alarm(30)
allowed=set();snap={};allbytes={}
def guard(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=Path(args[0]).absolute();flags=args[2]
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT):assert path.is_relative_to(OUT)
        else:assert str(path) in allowed,('unlisted read',str(path))
    if event.startswith(('subprocess.','os.system','os.exec','os.spawn','socket.')):raise RuntimeError('process/network prohibited')
sys.addaudithook(guard)
def raw(path,sha=None,cap=1024**2):
    p=ROOT/path;assert '..' not in p.parts and p.is_relative_to(ROOT)
    assert not any(x.is_symlink() for x in (p,*p.parents) if x.is_relative_to(ROOT))
    allowed.add(str(p));fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        s=os.fstat(f.fileno());assert stat.S_ISREG(s.st_mode) and s.st_size<=cap
        b=f.read(cap+1);assert len(b)==s.st_size and len(b)<=cap
    h=hashlib.sha256(b).hexdigest()
    if sha:assert h==sha,(path,h,sha)
    snap[path]={'sha256':h,'bytes':len(b)};allbytes[path]=b
    assert time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<128*1024**2
    return b

def decode(b):return json.loads(b,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def obj(path,sha=None):return decode(raw(path,sha))
def bind(b):return obj(b['path'],b['sha256'])
def canon(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def close(a,b,name):
    assert type(a) in (float,int) and type(b) in (float,int) and math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15),(name,a,b)

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(value if isinstance(value,bytes) else canon(value))

names={'intent.json','intent-receipt.json','receipt.json','readout-console.txt','state.json','access.jsonl','fit.json','calibration-metrics.json','terminal.json','publication.json'}
assert {x.name for x in (ROOT/BASE).iterdir()}==names
for name in names:raw(BASE+name)
objects={n:decode(allbytes[BASE+n]) for n in names if n.endswith('.json')}
publication=objects['publication.json'];terminal=objects['terminal.json'];state=objects['state.json'];receipt=objects['receipt.json'];intent=objects['intent.json']
assert publication['accepted'] is True and publication['release']==RB and publication['physical_validation_pass'] is None
assert set(publication['outputs'])==names-{'publication.json'}
for n,b in publication['outputs'].items():assert b==snap[BASE+n]
assert set(terminal['output_inventory'])==names-{'publication.json','terminal.json'}
for n,b in terminal['output_inventory'].items():assert b==snap[BASE+n]
assert allbytes[BASE+'terminal.json']==canon(terminal) and allbytes[BASE+'publication.json']==canon(publication)
release=bind(RB);declaration=bind(release['declaration']);review=bind(release['independent_source_review'])
assert release['execution_released'] is True and release['source_commit']=='4384ea0ae681e46986de9505edf226dcaa2a6c8c'
false={**release,'execution_released':False,'source_commit':None,'independent_source_review':None}
assert hashlib.sha256(canon(false)).hexdigest()==review['release_candidate_sha256']=='0cc30a56af1b75e39625617b9975aa54fa00fd879a534a58f177bb695acea256'
assert review['schema']=='hbe-v5-axial-fit-source-review-v1' and review['decision']=='GO'
assert len(release['source_bindings'])==33
for path,h in release['source_bindings'].items():raw(path,h)
for record in (intent,terminal,state,publication):assert record['release']==RB
assert terminal['source_commit']==receipt['source_commit']==intent['source_commit']==release['source_commit']
assert terminal['caps']==release['caps']==declaration['caps']
assert terminal['cleanup']=={'contained':True,'direct_child_reaped':True,'errors':[],'exit_code':0,'fallback_used':False,'remaining_members':[]}
assert terminal['termination_requested'] is False
status='completed_axial_calibration_pending_fitted_confirmation'
assert terminal['status']==state['status']==status
for record in (terminal,state):
    assert record['fit_calls']==1 and record['native_calls']==record['mesher_calls']==record['held_out_member_reads']==0
    assert record['physical_validation_pass'] is None and record['freeze_saved'] is False and record['held_out_access_released'] is False
assert state['calibration_access_attempted'] is True and state['calibration_responses_accessed'] is True and state['patient_tool_mechanics_admitted'] is False
for key,n in [('fit','fit.json'),('calibration_metrics','calibration-metrics.json')]:
    assert state[key]==terminal[key]=={'path':BASE+n,'sha256':snap[BASE+n]['sha256']}
assert terminal['state']=={'path':BASE+'state.json','sha256':snap[BASE+'state.json']['sha256']}
assert terminal['access_ledger']=={'path':BASE+'access.jsonl','sha256':snap[BASE+'access.jsonl']['sha256']}
assert state['declaration']==release['declaration'] and state['qualification']==declaration['evidence']['comparison']
assert state['reference_readouts']=={b:v['readout'] for b,v in declaration['references'].items()}
for cache in (Path(terminal['parent_cache_prefix']),ROOT/BASE/'unused-worker-pycache'):assert not cache.exists() and not cache.is_symlink()
stage=terminal['readout_stage'];assert stage==receipt['readout_stage']
assert stage['status']=='completed_within_caps' and stage['exit_code']==0 and stage['kill_reason'] is None
assert stage['elapsed_seconds']<stage['wall_cap_seconds']<=60
assert stage['peak_sampled_process_group_rss_bytes']<=stage['sampled_process_group_rss_cap_bytes']==512*1024**2
assert stage['active_output_cap_bytes']==16*1024**2-131072
assert 0<=terminal['elapsed_before_terminal_publication_seconds']<=publication['elapsed_before_publication_seconds']<70
assert sum(snap[BASE+n]['bytes'] for n in names)<16*1024**2
assert terminal['readout_calls_attempted']==1
ledger=[decode(line) for line in allbytes[BASE+'access.jsonl'].splitlines()]
assert len(ledger)==2 and [e['phase'] for e in ledger]==['calibration_attempt','calibration_completed']
for i,e in enumerate(ledger):
    assert e['sequence']==i and e['members']==declaration['permitted_members']
    assert e['protocol_sha256']==declaration['evidence']['protocol']['sha256'] and e['release_sha256']==RB['sha256']
member_hashes={b:v['member']['sha256'] for b,v in declaration['references'].items()}
assert ledger[1]['member_sha256']==state['calibration_member_sha256']==member_hashes
assert sum(v['member']['bytes'] for v in declaration['references'].values())==2392
roles=bind(declaration['evidence']['roles'])
assert roles['selection']['donor_role']=='DEVELOPMENT' and roles['selection']['specimen_id']=='HBE_01_03'
assert [m['path'] for m in roles['calibration']['members']]==declaration['permitted_members']

# Independent arithmetic from SAVED fit records, without calling fit_scale,
# interpolation helpers, or reading the source archive/reference readouts.
fit=objects['fit.json'];metrics=objects['calibration-metrics.json']
assert fit['schema']=='hbe-positive-scale-fit-v1' and fit['fitted_parameters']==['positive_mu_scale_only']
assert fit['held_out_values_used'] is False and fit['offset_or_sign_fitted'] is False
assert fit['reference_mu_Pa']==1000 and fit['scale']>0 and fit['mu_Pa']>0
assert set(fit['modes'])==set(metrics['branches'])=={'compression','tension'}
assert metrics['physical_validation_pass'] is None and metrics['empirical_tolerance'] is None and metrics['independent_donor_validation'] is False
numerator=denominator=0.;summary={}
for branch in ('compression','tension'):
    mode=fit['modes'][branch];metric=metrics['branches'][branch]
    x,y,g,w=([float(t) for t in mode[k]] for k in ('coordinate','observed','reference_prediction','weights'))
    assert len(x)==len(y)==len(g)==len(w)==30 and all(math.isfinite(v) for a in (x,y,g,w) for v in a)
    assert mode['source_sha256']==member_hashes[branch]
    assert hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()==declaration['references'][branch]['coordinate_sha256']
    gaps=[abs(b-a) for a,b in zip(x,x[1:])];q=[gaps[0]/2]+[(a+b)/2 for a,b in zip(gaps,gaps[1:])]+[gaps[-1]/2]
    total=sum(q);q=[v/total for v in q]
    for got,want in zip(w,q):close(got,.5*want,'equal branch quadrature')
    numerator+=sum(a*b*c for a,b,c in zip(w,g,y));denominator+=sum(a*b*b for a,b in zip(w,g))
    assert metric['branch']==branch and metric['role']=='calibration_fit' and metric['row_count']==30 and metric['response_unit']=='N'
    predicted=[fit['scale']*v for v in g];residual=[p-o for p,o in zip(predicted,y)]
    for a,b in zip(predicted,metric['predicted_at_observed_coordinates']):close(a,b,'scaled prediction')
    for a,b in zip(residual,metric['residuals']):close(a,b,'signed residual')
    for a,b in zip(q,metric['weights']):close(a,b,'metric quadrature')
    rmse=math.sqrt(sum(a*b*b for a,b in zip(q,residual)));mae=sum(a*abs(b) for a,b in zip(q,residual));rms=math.sqrt(sum(a*b*b for a,b in zip(q,y)))
    endpoint=max(range(30),key=lambda i:abs(x[i]));close(rmse,metric['RMSE'],'branch RMSE');close(mae,metric['MAE'],'branch MAE');close(rms,metric['observed_RMS_denominator'],'normalization denominator')
    close(residual[endpoint],metric['signed_endpoint_bias'],'endpoint bias');close(rmse/rms,metric['normalized_RMSE'],'normalized RMSE')
    assert metric['physical_validation_pass'] is None and metric['row_IID_confidence_interval'] is None and metric['normalized_error_reason'] is None
    summary[branch]={k:metric[k] for k in ('role','row_count','response_unit','RMSE','MAE','observed_RMS_denominator','normalized_RMSE','signed_endpoint_bias','physical_validation_pass','row_IID_confidence_interval')}
close(numerator,fit['numerator'],'fit numerator');close(denominator,fit['denominator'],'fit denominator')
close(fit['denominator_minimum'],1e-12*(1000.*.004**2)**2,'fixed minimum');assert denominator>fit['denominator_minimum']
close(numerator/denominator,fit['scale'],'positive scale');close(fit['scale']*1000.,fit['mu_Pa'],'mu')
close(math.sqrt(sum(v['RMSE']**2 for v in summary.values())/2),metrics['equal_branch_RMSE'],'paired RMSE')
close(math.sqrt(sum(v['normalized_RMSE']**2 for v in summary.values())/2),metrics['equal_branch_normalized_RMSE'],'paired normalized RMSE')
for path,binding in list(snap.items()):raw(path,binding['sha256'])
assert {x.name for x in (ROOT/BASE).iterdir()}==names
result={'schema':'hbe-v5-axial-fit-saved-audit-v1','decision':'PASS_saved_fit_record_arithmetic_and_publication','source_commit':release['source_commit'],'release':RB,'fit':state['fit'],'calibration_metrics':state['calibration_metrics'],'publication_sha256':snap[BASE+'publication.json']['sha256'],'terminal_sha256':snap[BASE+'terminal.json']['sha256'],'scale':fit['scale'],'mu_Pa':fit['mu_Pa'],'reference_mu_Pa':1000.,'fitted_parameters':fit['fitted_parameters'],'numerator':fit['numerator'],'denominator':fit['denominator'],'denominator_minimum':fit['denominator_minimum'],'branch_metrics':summary,'equal_branch_RMSE_N':metrics['equal_branch_RMSE'],'equal_branch_normalized_RMSE':metrics['equal_branch_normalized_RMSE'],'interpretation':'Descriptive goodness of fit to two development curves from one specimen; no independent validation, physical tolerance or patient/tool property admission.','physical_validation_pass':None,'held_out_reads':0,'fit_calls_in_execution':1,'audit_estimator_calls':0,'audit_archive_reads':0,'audit_original_member_reads':0,'audit_saved_reference_reads':0,'audit_native_calls':0,'audit_project_imports':0,'audit_scope':'Saved-record arithmetic only; interpolated unscaled predictions checked from saved fit, not reconstructed from reference readouts. No archive or estimator rerun. Float agreement rel1e-12/abs1e-15 solely arithmetic audit, no fit-quality threshold.','execution':{'worker_seconds':stage['elapsed_seconds'],'elapsed_before_publication_seconds':publication['elapsed_before_publication_seconds'],'sampled_group_peak_rss_bytes':stage['peak_sampled_process_group_rss_bytes'],'output_bytes':sum(snap[BASE+n]['bytes'] for n in names),'cleanup':terminal['cleanup'],'ledger_events':2,'completed_members':2,'calibration_rows':60,'decoded_bytes':2392},'input_files_unchanged':snap,'audit_elapsed_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
save('audit-result.json',result)
# Exact compact execution evidence copies. These include authorized calibration
# samples in fit/metrics and must retain their DEVELOPMENT calibration scope.
for name in sorted(names):save('execution/'+name,allbytes[BASE+name])
save('root-release.json',allbytes[RB['path']]);save('declaration.json',allbytes[release['declaration']['path']])
print(json.dumps({k:result[k] for k in ('decision','scale','mu_Pa','branch_metrics','equal_branch_RMSE_N','equal_branch_normalized_RMSE','execution','audit_elapsed_seconds','audit_peak_rss_bytes')},indent=2))
