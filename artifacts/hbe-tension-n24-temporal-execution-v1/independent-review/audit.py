"""Terminal-hash gated saved-output audit. No native call, mesher or curve access."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).parent
SOURCE=ROOT/'build/hbe-halfheight-tension-n24-temporal-v1/source'
sys.path.insert(0,str(SOURCE))
from scripts import mechanics_hbe_halfheight_tension_n24_temporal as case
from scripts import mechanics_hbe_halfheight_tension_n24_temporal_experiment as runner
COMMIT='2911d310a880c5bbdf121acb470f06c8d2bd58a8'
ARCHIVE='860527629a93f34f4a331c7b29dce0a845e7b16b997b47bdd269c37659e7c5b6'
RELEASE_SHA='f2c21af48d1200af2002a09578082f84d3d879debc5388a77e8790a57ca26b1d'
def forbidden(*a,**k):raise AssertionError('Native solve, new mesh, phase launch forbidden')
runner.common.subprocess.Popen=forbidden;runner.old.solve=forbidden;runner.runtime.supervise=forbidden
runner.launch=forbidden;runner.worker=forbidden;runner.prepare=forbidden
runner.receipts.core.generate_full_mesh=forbidden

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def binding(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def tree(p):return {str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(p.rglob('*')) if f.is_file()}
def close(a,b):assert math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-18),(a,b)
def metric_pass(m):return m['actual']<m['limit'] if m.get('comparison')=='lt' else m['actual']<=m['limit']

def independent_group(coarse,medium,fine,kind):
 """Direct scalar/vector arithmetic, separate from original refinement helper."""
 stride=fine['steps']//60
 f0,f1,f2=(np.asarray(r['applied_force_N'])[::r['steps']//60] for r in (coarse,medium,fine))
 p0,p1,p2=(np.asarray(r['probe_displacements_m'])[::r['steps']//60] for r in (coarse,medium,fine))
 prev_f=float(np.max(np.abs(f1-f0)));next_f=float(np.max(np.abs(f2-f1)))
 prev_u=float(np.max(np.linalg.norm(p1-p0,axis=2)));next_u=float(np.max(np.linalg.norm(p2-p1,axis=2)))
 floor,relative,u_limit=(1.6e-8,.002,8e-7) if kind=='step' else (1.6e-7,.02,8e-6)
 result={'reaction':{'actual':next_f,'limit':floor+relative*float(np.max(np.abs(f2))),'units':'N'},
         'motion':{'actual':next_u,'limit':u_limit,'units':'m'}}
 if kind=='mesh':
  for name,current,previous,absolute,units in [('reaction_trend',next_f,prev_f,floor,'N'),('motion_trend',next_u,prev_u,u_limit,'m')]:
   result[name]=({'actual':max(current,previous),'limit':absolute,'units':units} if max(current,previous)<=absolute else
                 {'actual':current,'limit':previous,'units':units,'comparison':'lt'})
 return result

def same_metrics(actual,saved):
 assert set(actual)==set(saved)
 for name,row in actual.items():
  other=saved[name];assert set(row)==set(other) and row['units']==other['units']
  assert row.get('comparison')==other.get('comparison')
  close(row['actual'],other['actual']);close(row['limit'],other['limit'])
  assert metric_pass(row)==metric_pass(other)

def execute(result_sha,publication_sha,comparison_sha):
 start=time.perf_counter();inputs={}
 P=ROOT/'outputs/mechanics/hbe-01-03-halfheight-tension-n24-temporal-v1/experiment'
 assert len(result_sha)==len(publication_sha)==len(comparison_sha)==64
 assert sha(P/'result.json')==result_sha and sha(P/'publication-check.json')==publication_sha
 assert sha(P/'comparison.json')==comparison_sha
 before=tree(P)
 def bound(b,*,json_value=True,maximum_bytes=2*1024**3):
  p=(ROOT/b['path']).resolve();assert p.stat().st_size<=maximum_bytes and sha(p)==b['sha256'],str(p)
  inputs[str(p)]={'bytes':p.stat().st_size,'sha256':b['sha256']}
  return load(p) if json_value else p
 result=load(P/'result.json');pub=load(P/'publication-check.json')
 assert result['schema']=='hbe-tension-n24-temporal-result-v1' and result['phase']=='solve'
 assert result['aggregate_cap_seconds']==600 and result['preparation_nested_in_aggregate'] is True and result['measured_data_accessed'] is False
 runner.receipts.require_publication(pub,result_binding=binding(P/'result.json'),phase='solve',cap=600,output_cap=1024**3)
 assert result['status']==pub['result_status']=='completed_numerical_diagnostic_only'
 sup=result['supervision'];assert sup['status']=='completed' and sup['exit_code']==0
 assert sup.get('kill_reason') is None and sup.get('cleanup_error') is None and sup['sampled_peak_process_group_rss_bytes']<=3*1024**3
 plan=bound(result['baseline']);state=bound(result['state']);study=bound(plan['study_binding']);release=bound(plan['release_binding'])
 assert release['schema']=='hbe-tension-n24-temporal-release-v1' and release['authorized'] is True and release['phase']=='solve'
 assert plan['release_binding']['sha256']==RELEASE_SHA and release['source_commit']==COMMIT and release['source_archive']['sha256']==ARCHIVE
 assert release['study']==plan['study_binding'] and release['declared_budgets']==study['budgets']
 assert release['source_bindings']==plan['source_bindings'] and release['automatic_retry'] is False
 review=bound(release['independent_implementation_review']);assert review['blocking_findings']==[]
 runner.source_inventory(ROOT,release,study,bound)
 assert Path(runner.__file__).resolve().is_relative_to(SOURCE) and Path(case.__file__).resolve().is_relative_to(SOURCE)
 for path,h in plan['inputs'].items():bound({'path':path,'sha256':h},json_value=False)
 # Previously accepted baseline is authenticated, not raw-response replayed.
 runner.baseline_metadata(ROOT,study,{'runtime_identity':plan['runtime_identity']},bound)
 assert state['schema']=='hbe-tension-n24-temporal-state-v1' and state['status']==result['status']
 assert state['run_id']==case.RUN_ID and state['solver_invocations']==1 and state['gmsh_generation_calls']==0 and state['measured_data_accessed'] is False
 prepared=bound(state['preparation']);assert prepared['schema']=='hbe-tension-n24-pure-deck-v1'
 assert prepared['study']==plan['study_binding'] and prepared['run_id']==case.RUN_ID and prepared['solver_calls']==prepared['gmsh_generation_calls']==0
 child=load(P/'preparation/child-execution.json')
 assert child['status']=='completed' and child['exit_code']==0 and child['cap_seconds']==60 and child['included_in_aggregate'] is True
 assert child['solver_calls']==child['gmsh_generation_calls']==0
 assert 0<=child['elapsed_seconds']<=state['preparation_elapsed_seconds']<60
 saved=bound(state['readout']);execution=bound(saved['execution_binding']);native=state['native_execution']
 assert execution['schema']=='hbe-tension-n24-native-execution-v1' and execution['run_id']==case.RUN_ID
 assert execution['execution']==native and execution['study']==plan['study_binding']
 runner.receipts.require_completed_native(native,420)
 assert 0<native['seconds_cap']<=420
 assert execution['runtime_identity']==plan['runtime_identity'] and execution['backend_profile']==plan['backend_profile']
 assert execution['primitive_bindings']==saved['primitive_bindings'] and execution['reconstruction']==saved['reconstruction']==prepared['case']['reconstruction']
 assert execution['backend_source_deck']==prepared['case']['backend_source_deck']
 assert execution['command']==native['command']==[plan['executable'],'-noconfig','-no_title','-i','specimen.feb','-o','solver.log']
 assert Path(native['cwd']).resolve()==(P/'run').resolve()
 for b in execution['primitive_bindings'].values():bound(b,json_value=False)
 bound(execution['executable'],json_value=False)
 actual=case.read_run(ROOT,plan['study_binding'],execution['primitive_bindings']);actual['execution_binding']=saved['execution_binding']
 assert actual['passed'] is True and actual['frame_count']==121 and canonical(actual)==canonical(saved)
 assert len(actual['half_native']['solver']['states'])==120
 base=study['baseline'];rows={int(n):bound(b) for n,b in base['older_tension_readouts_for_sensitivity'].items()};rows[24]=bound(base['baseline_readout'])
 old_comparison=bound(base['comparison']);assert old_comparison['passed'] is False
 comparison=bound(state['comparison']);assert state['comparison']['sha256']==comparison_sha
 recomputed=case.compare(rows,actual,study,old_comparison);assert canonical(comparison)==canonical(recomputed)
 temporal=independent_group(rows[24],rows[24],actual,'step');same_metrics(temporal,comparison['original_temporal_criteria'])
 independent_groups={};changes=[]
 for first in (12,8):
  name=f'tension:N{first}-N16-N24';prior=independent_group(rows[first],rows[16],rows[24],'mesh');new=independent_group(rows[first],rows[16],actual,'mesh')
  same_metrics(prior,comparison['mixed_step_groups'][name]['original']);same_metrics(new,comparison['mixed_step_groups'][name]['S120_substitution'])
  assert all(metric_pass(v) for v in prior.values())
  changes.extend(name+'/'+k for k in new if metric_pass(new[k])!=metric_pass(prior[k]))
  independent_groups[name]={'S60':prior,'S120_substitution':new}
 assert changes==comparison['mixed_step_classification_changes']
 F=np.asarray(actual['applied_force_N']);B=np.asarray(rows[24]['applied_force_N']);U=np.asarray(actual['probe_displacements_m']);V=np.asarray(rows[24]['probe_displacements_m'])
 assert F.shape==(121,) and B.shape==(61,) and U.shape==(121,75,3) and V.shape==(61,75,3)
 assert np.array_equal(np.asarray(actual['load_coordinate_m'])[::2],np.asarray(rows[24]['load_coordinate_m']))
 delta=F[::2]-B;motion=np.linalg.norm(U[::2]-V,axis=2).max(axis=1);maximum=int(np.argmax(np.abs(delta)))
 assert np.array_equal(delta,np.asarray(comparison['signed_S120_minus_S60_force_N'])) and maximum==comparison['maximum_force_shift_state']
 close(float(np.abs(delta).max()),temporal['reaction']['actual']);close(float(motion.max()),temporal['motion']['actual'])
 passed=all(metric_pass(v) for v in temporal.values());candidate=passed and not changes
 assert comparison['original_temporal_checks_passed'] is passed and comparison['tension_temporal_candidate'] is candidate
 status='original_temporal_criteria_failed' if not passed else 'temporal_pass_spatial_classification_fragile' if changes else 'tension_temporal_candidate'
 assert comparison['status']==status and comparison['historical_global_spatial_failure_preserved'] is True
 for k in ('calibration_released','spatial_convergence_accepted','measured_data_accessed','automatic_next_run_permitted'):assert comparison[k] is False
 assert comparison['physical_validation_pass'] is None
 for b in study['preserved_failures'].values():bound(b)
 watch=load(P/'output-watch.json');assert watch['status']=='watching' and watch.get('reason') is None
 assert watch['maximum_active_bytes']<=512*1024**2 and watch['maximum_total_bytes']<=1024**3
 retained=sum(p.stat().st_size for p in P.parent.rglob('*') if p.is_file())
 assert pub['retained_bytes_including_closeout']==retained and pub['retained_bytes_before_closeout']+(P/'publication-check.json').stat().st_size==retained
 assert tree(P)==before
 for p,b in inputs.items():assert sha(p)==b['sha256']
 (OUT/'input-hashes.json').write_text(json.dumps(inputs,sort_keys=True,indent=2)+'\n')
 report={'schema':'hbe-tension-n24-temporal-independent-output-review-v1','created_at':datetime.now(timezone.utc).isoformat(),
  'status':'saved_execution_and_temporal_diagnostic_verified','blocking_record_mismatches':[],
  'result':binding(P/'result.json'),'publication':binding(P/'publication-check.json'),'comparison':state['comparison'],
  'source_commit':COMMIT,'source_archive':release['source_archive'],'complete121_native_readout_canonical_equal':True,'complete_comparison_canonical_equal':True,
  'independent_temporal':{'criteria':temporal,'common_states':61,'probes':75,'signed_force_shift_N':delta.tolist(),'maximum_shift_state':maximum,
    'signed_endpoint_shift_N':float(delta[-1]),'common_probe_vector_shifts_m':motion.tolist()},
  'independent_mixed_groups':independent_groups,'mixed_classification_changes':changes,
  'conclusion':{'status':status,'tension_temporal_candidate':candidate,'original_global_failure_preserved':True,'calibration_released':False,
    'spatial_convergence_accepted':False,'physical_validation_pass':None,'automatic_next_run_permitted':False},
  'costs':{'preparation_seconds':state['preparation_elapsed_seconds'],'native_seconds':native['elapsed_seconds'],
    'publication_seconds':pub['elapsed_through_result_publication_seconds'],'sampled_rss_bytes':sup['sampled_peak_process_group_rss_bytes'],
    'output_watch':watch,'retained_bytes':retained},
  'scope':{'review_native_solver_calls':0,'review_mesher_calls':0,'measured_curve_reads':0,'old_raw_response_replays':0,
    'saved_native_frames_reconstructed':121,'all_inputs_and_outputs_unchanged':True,'tracked_edits':False},
  'limits':['Static tension increment check; not physical rate dependence or tissue validation.',
    'Only N24 is replaced by S120 in sensitivity groups, not a new homogeneous spatial refinement study.',
    'Full-domain quantities are reconstructed from half-native fields; no new native-full equivalence run.',
    'Memory and output peaks sampled; own closeout publication outside its recorded clock.'],
  'input_inventory':binding(OUT/'input-hashes.json'),'audit_source':binding(Path(__file__)),'review_seconds':time.perf_counter()-start}
 (OUT/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'verification':binding(OUT/'verification.json'),'conclusion':report['conclusion'],'review_seconds':report['review_seconds']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result-sha256',required=True);p.add_argument('--publication-sha256',required=True);p.add_argument('--comparison-sha256',required=True);args=p.parse_args()
 try:execute(args.result_sha256,args.publication_sha256,args.comparison_sha256)
 except BaseException as e:
  target=OUT/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json')
  target.write_text(json.dumps({'error_type':type(e).__name__,'message':str(e),'source':binding(Path(__file__)),'terminal_inputs':vars(args)},indent=2)+'\n');raise
