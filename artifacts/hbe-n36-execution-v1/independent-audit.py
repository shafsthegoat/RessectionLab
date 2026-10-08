"""Independent completed-output audit. Requires root-supplied terminal hashes.

Reads only saved numerical outputs. No native solve, mesher, measured curve,
release, calibration or fabricated response series is permitted.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
SOURCE=ROOT/'build/hbe-halfheight-global-n36-v1/source'
sys.path.insert(0,str(SOURCE))
from scripts import mechanics_hbe_halfheight_global_n36 as core
from scripts import mechanics_hbe_halfheight_global_n36_readout as reader
from scripts import mechanics_hbe_halfheight_global_n36_experiment as runner

def forbidden(*args,**kwargs):raise AssertionError('Native execution/mesh creation forbidden in saved-output audit')
runner.subprocess.Popen=forbidden;runner.runtime.supervise=forbidden;runner.old.solve=forbidden;core.generate_full_mesh=forbidden

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def load(p):return json.loads(Path(p).read_bytes())
def bind(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def tree(p):return {str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(p.rglob('*')) if f.is_file()}

# Independent scalar solver, separate from executable diagnostic. No test
# response generation: it receives only already completed saved force values.
FLOOR=1.6e-7
TRIPLETS=((8,12,16),(12,16,24),(16,24,32),(24,32,36))
def scalar(ns, fs):
 a=math.log(ns[1]/ns[0]);b=math.log(ns[2]/ns[1]);u=fs[1]-fs[0];v=fs[2]-fs[1]
 if not all(math.isfinite(x) for x in (*fs,u,v)):return {'status':'invalid'}
 if abs(u)<=FLOOR or abs(v)<=FLOOR:
  return {'status':'below_original_difference_floor' if abs(u)<=FLOOR and abs(v)<=FLOOR else 'insufficient_increment_resolution'}
 if (u>0)!=(v>0):return {'status':'opposite_signs'}
 q=u/v
 def quotient(p):return math.expm1(p*a)/(-math.expm1(-p*b)) if p else a/b
 if not math.isfinite(q):return {'status':'invalid'}
 if q<=a/b or q>quotient(16):return {'status':'no_admissible_positive_root'}
 lo,hi=0.,16.
 for _ in range(110):
  m=(lo+hi)/2
  if quotient(m)<q:lo=m
  else:hi=m
 p=(lo+hi)/2
 if p<1e-6:return {'status':'no_admissible_positive_root'}
 e=v/math.expm1(p*b);limit=fs[-1]+e
 if not all(math.isfinite(x) for x in (p,e,limit)):return {'status':'invalid'}
 return {'status':'eligible','order':p,'signed_remaining_indicator_N':e,'absolute_remaining_indicator_N':abs(e),'force_limit_N':limit}

def execute(expected_result,expected_publication):
 started=time.perf_counter();inputs={}
 P=ROOT/'outputs/mechanics/hbe-01-03-halfheight-global-n36-v1/experiment'
 assert sha(P/'result.json')==expected_result and sha(P/'publication-check.json')==expected_publication
 before=tree(P)
 def bound(b,json_value=True):
  p=(ROOT/b['path']).resolve();assert sha(p)==b['sha256'],str(p)
  inputs[str(p)]={'sha256':b['sha256'],'bytes':p.stat().st_size}
  return load(p) if json_value else p
 result=load(P/'result.json');pub=load(P/'publication-check.json')
 assert result['phase']=='solve' and result['status']=='completed_numerical_diagnostic_only'
 assert result['measured_data_accessed'] is False and result['aggregate_cap_seconds']==1800
 runner.require_publication(pub,result_binding=bind(P/'result.json'),phase='solve',cap=1800,output_cap=2*1024**3)
 assert pub['result_status']==result['status']
 base=bound(result['baseline']);state=bound(result['state']);study=bound(base['study_binding']);release=bound(base['release_binding'])
 assert base['release_binding']['sha256']=='119f21cf2a9b4d637784f8ea97ea641026737e7f4e804f908790e5d72df58b87'
 assert release['authorized'] is True and release['phase']=='solve' and release['study']==base['study_binding']
 assert release['source_commit']=='169efad4d2cb3b53a80091082c9845a2c45e8b99'
 assert release['source_bindings']==base['source_bindings'];runner.source_inventory(ROOT,release,study)
 for path,h in base['inputs'].items():bound({'path':path,'sha256':h},False)
 assert state['status']=='completed_numerical_diagnostic_only' and state['phase']=='solve'
 assert state['solver_invocations']==1 and state['gmsh_generation_calls']==0 and state['measured_data_accessed'] is False
 assert set(state['runs'])=={core.RUN_ID}
 sup=result['supervision'];assert sup==load(P/'supervision/supervision.json')
 assert sup['status']=='completed' and type(sup['exit_code']) is int and sup['exit_code']==0
 assert sup['kill_reason'] is sup['cleanup_error'] is sup['error'] is None
 assert sup['rss_cap_bytes']==3*1024**3 and 0<sup['sampled_peak_process_group_rss_bytes']<=3*1024**3
 assert 0<=sup['elapsed_seconds']<sup['wall_cap_seconds']<=1800
 assert 0<=result['phase_elapsed_before_result_publication_seconds']<=pub['elapsed_through_result_publication_seconds']<1800
 prep_result=bound(release['preparation']);prep_pub=bound(release['preparation_publication'])
 runner.require_publication(prep_pub,result_binding=release['preparation'],phase='prepare',cap=60,output_cap=2*1024**3)
 prep_review=bound(release['independent_preparation_review'])
 assert prep_review['status']=='actual_preparation_verified_for_separate_single_solve_release_review'
 assert prep_review['result']==release['preparation'] and prep_review['publication']==release['preparation_publication']
 row=state['runs'][core.RUN_ID];assert row['status']=='passed_individual_numerical_checks'
 saved=bound(row['readout']);execution=bound(saved['execution_binding'])
 assert execution['execution']==row['execution'];runner.require_completed_native(row['execution'],1500)
 assert 0<row['execution']['seconds_cap']<=1500
 assert execution['study']==base['study_binding'] and execution['runtime_identity']==base['runtime_identity']
 assert execution['backend_profile']==base['backend_profile']
 assert execution['primitive_bindings']==saved['primitive_bindings'] and execution['reconstruction']==saved['reconstruction']
 assert execution['run_id']==core.RUN_ID and execution['command']==row['command']
 expected_dir=P/'runs'/core.RUN_ID.replace(':','-')
 assert execution['command']==[base['executable'],'-noconfig','-no_title','-i','specimen.feb','-o','solver.log']
 assert execution['cwd']==str(expected_dir.relative_to(ROOT))
 for b in execution['primitive_bindings'].values():bound(b,False)
 for rel,receipt in row['retained_files'].items():
  p=expected_dir/rel;assert p.stat().st_size==receipt['bytes'] and sha(p)==receipt['sha256']
 # Independent reviewer invocation of the frozen complete native primitive reader.
 actual=reader.read_global_run(ROOT,half_bindings=execution['primitive_bindings'],reconstruction_binding=execution['reconstruction'],declaration_binding=base['study_binding'])
 actual['execution_binding']=saved['execution_binding']
 assert actual['passed'] is True and canonical(actual)==canonical(saved)
 runs={core.RUN_ID:actual};prior_checks={}
 assert set(state['reused_runs'])==set(study['prior_runs'])
 for key,b in state['reused_runs'].items():
  value=bound(b);original=bound(study['prior_runs'][key]['readout'])
  assert canonical(value)==canonical(original);runs[key]=value;prior_checks[key]={'canonical_complete_record_equal':True,'response_reparsed_by_this_review':False}
 for state_key,directory,sourcekey in [('baseline_replay','P1-replay','baseline_P1'),('N32_replay','N32-replay','baseline_N32')]:
  rec=bound(state[state_key]);prior=study[sourcekey]
  assert rec['status']=='completed_exact_original_context_replay' and rec['exit_code']==0 and rec['reaped'] is True
  assert rec['native_solver_calls']==0 and rec['complete_raw_replay_exact'] is True
  assert rec['original_source_archive']==prior['source_archive'] and rec['original_source_commit']==prior['source_commit']
  value=load(P/directory/'readout.json');accepted=bound(prior['readout'])
  assert canonical(value)==canonical(accepted);runs[prior['run_id']]=value
  prior_checks[prior['run_id']]={'canonical_complete_record_equal':True,'completed_original_context_replay_record':state[state_key],'response_reparsed_by_this_review':False}
 comparison=bound(state['comparison']);recomputed=reader.comparison_report(runs,study)
 assert canonical(comparison)==canonical(recomputed)
 # Separate scalar calculation of all 240 nonrest triplet estimates and decisions.
 forces={int(k.split(':')[1][1:]):v['applied_force_N'] for k,v in runs.items()}
 allowance=FLOOR+.02*max(map(abs,forces[36]));assert allowance==comparison['force_allowance_N']
 maxdiff=0.;exceed=[];unresolved=[];flags=[];envelopes=[]
 for i in range(1,61):
  row=comparison['states'][i];orders=[scalar(ns,[forces[n][i] for n in ns]) for ns in TRIPLETS]
  for ns,v in zip(TRIPLETS,orders):
   stored=row['orders']['-'.join('N'+str(n) for n in ns)];assert v['status']==stored['status']
   for field,value in v.items():
    if field=='status':continue
    error=abs(value-stored[field]);maxdiff=max(maxdiff,error)
    assert math.isclose(value,stored[field],rel_tol=3e-12,abs_tol=3e-14),(i,field,error)
  old,recent,current=orders[1:];drift=None
  if all(v['status']=='eligible' for v in orders[1:]):
   drift=abs(current['order']-recent['order'])>abs(recent['order']-old['order'])+1e-6
  assert row['increasing_order_drift'] is drift
  if recent['status']==current['status']=='eligible':
   e=max(abs(current['force_limit_N']-forces[36][i]),abs(recent['force_limit_N']-forces[36][i]))
   instability=abs(current['force_limit_N']-recent['force_limit_N'])>allowance
   assert row['status']=='resolved_conditional_model' and row['remaining_within_allowance'] is (e<=allowance)
   assert row['force_limit_instability'] is instability
   assert math.isclose(e,row['two_latest_limit_envelope_N'],rel_tol=3e-12,abs_tol=3e-14)
   envelopes.append((e,i))
   if e>allowance:exceed.append(i)
   if instability or drift:flags.append(i)
  else:
   unresolved.append(i)
   if drift:flags.append(i)
 assert exceed==comparison['remaining_exceedance_states'] and unresolved==comparison['unresolved_nonrest_states'] and flags==comparison['instability_states']
 F=np.asarray(forces[36]);F32=np.asarray(forces[32]);P36=np.asarray(actual['probe_displacements_m']);P32=np.asarray(runs[study['baseline_N32']['run_id']]['probe_displacements_m'])
 reaction=float(np.max(np.abs(F-F32)));motion=float(np.max(np.linalg.norm(P36-P32,axis=2)))
 # Independently recompute the same declared vector-norm motion statistic.
 metrics=comparison['original_style_N24_N32_N36_metrics'];assert math.isclose(reaction,metrics['reaction']['actual'],rel_tol=1e-13,abs_tol=1e-16)
 assert math.isclose(motion,metrics['motion']['actual'],rel_tol=1e-13,abs_tol=1e-18)
 assert metrics['motion']['limit']==8e-6
 adjacent=all(metrics[k]['actual']<=metrics[k]['limit'] for k in ('reaction','motion'))
 assert comparison['candidate_for_temporal_review'] is (adjacent and not unresolved and not exceed and not flags)
 for key in ['spatial_convergence_accepted','finest_level_step_convergence_evaluated','calibration_released','measured_data_accessed','automatic_next_run_permitted']:assert comparison[key] is False
 assert comparison['physical_validation_pass'] is None and comparison['terminal_uniform_family'] is True
 assert comparison['previous_N32_failed_comparison']==study['baseline_N32']['comparison']
 prior_negative=bound(study['baseline_N32']['comparison']);assert prior_negative['spatial_convergence_accepted'] is False
 watch=load(P/'output-watch.json');assert watch['status']=='watching' and watch.get('reason') is None
 assert watch['maximum_active_bytes']<=768*1024**2 and watch['maximum_total_bytes']<=2*1024**3
 all_bytes=sum(v['bytes'] for v in tree(P.parent).values());assert all_bytes==pub['retained_bytes_including_closeout']
 assert pub['retained_bytes_before_closeout']+len((P/'publication-check.json').read_bytes())==all_bytes
 assert tree(P)==before
 for path,rec in inputs.items():assert sha(path)==rec['sha256']
 (OUT/'input-hashes.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
 report={'schema':'hbe-n36-independent-completed-output-review-v1','created_at':datetime.now(timezone.utc).isoformat(),
  'status':'saved_execution_and_declared_diagnostic_verified','blocking_record_mismatches':[],
  'result':bind(P/'result.json'),'publication':bind(P/'publication-check.json'),'comparison':state['comparison'],
  'source_commit':release['source_commit'],'source_archive':release['source_archive'],
  'N36_complete_native_readout_canonical_equal':True,'comparison_canonical_equal':True,
  'prior_record_bindings':prior_checks,'prior_response_replay_basis':'Existing independently accepted complete prior readouts, with this execution replay records verified; no redundant prior raw replay by this reviewer.',
  'independent_scalar_estimates_checked':240,'maximum_scalar_field_absolute_difference':maxdiff,
  'conclusion':{'candidate_for_temporal_review':comparison['candidate_for_temporal_review'],'diagnostic_status':comparison['status'],
   'endpoint_force_N':forces[36][-1],'force_allowance_N':allowance,'endpoint':comparison['endpoint'],
   'remaining_exceedance_states':exceed,'unresolved_states':unresolved,'instability_states':flags,
   'native_N32_to_N36_max_reaction_change_N':reaction,'motion_vector_norm_max_m':motion,
   'spatial_convergence_accepted':False,'temporal_adequacy_established':False,'calibration_released':False,'physical_validation_pass':None},
  'execution_costs':{'native_elapsed_seconds':execution['execution']['elapsed_seconds'],
   'aggregate_through_result_publication_seconds':pub['elapsed_through_result_publication_seconds'],
   'sampled_worker_family_peak_rss_bytes':sup['sampled_peak_process_group_rss_bytes'],'output_watch':watch,'retained_bytes_including_closeout':all_bytes},
  'scope':{'review_native_calls':0,'review_mesher_calls':0,'new_meshes':0,'measured_curve_access':False,'tracked_source_edits':False,'release_created':False,
   'N36_saved_native_frames_reparsed':61,'previous_N32_negative_preserved':True,'original_inputs_and_saved_outputs_unchanged':True},
  'limits':['One terminal uniform spatial diagnostic only. Never temporal adequacy, physical validation or calibration.',
   'Memory/output peaks are sampled. Tiny closing receipt publication is outside its own recorded clock.',
   'No new solver/mesh/run or measured response was used. Replaying saved records is verification, not an independent physical measurement.'],
  'input_hash_inventory':bind(OUT/'input-hashes.json'),'audit_source':bind(Path(__file__)),
  'review_elapsed_seconds':time.perf_counter()-started}
 (OUT/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'verification':bind(OUT/'verification.json'),'status':report['status'],'scientific_status':comparison['status'],'review_seconds':report['review_elapsed_seconds']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result-sha256',required=True);p.add_argument('--publication-sha256',required=True);args=p.parse_args()
 execute(args.result_sha256,args.publication_sha256)
