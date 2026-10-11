"""Terminal-only saved scalar audit. No project imports, solver, raw logs or datasets.

Run only after root provides terminal/comparison byte hashes. A failed terminal
without a produced comparison is recorded without reading exposed fit values.
"""
from pathlib import Path
import argparse,bisect,hashlib,json,math,traceback
ROOT=Path(__file__).resolve().parents[4]
ORIGINAL=ROOT/'outputs/mechanics/hbe-ogden-shape-comparison-v1/attempt-01'
OUT=ROOT/'outputs/mechanics/hbe-ogden-shape-comparison-v1/postprocess-01'
RUNS=(('compression',16,60),('compression',24,60),('tension',16,60),('tension',24,60),('compression',24,120),('tension',24,120))
PAIRS={'mesh':(1.6e-7,.02,8e-6),'time':(1.6e-8,.002,8e-7)}
checks=0;inputs={}
def need(x,label):
 global checks
 checks+=1
 if not x:raise AssertionError(label)
def close(a,b,label):need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=2e-10,abs_tol=2e-13),label)
def read_json(path,expected=None,maximum=8*1024**2):
 path=(ROOT/path if isinstance(path,str) else path).resolve()
 need(path.is_relative_to(ROOT) and path.suffix=='.json','JSON-only within repository')
 raw=path.read_bytes();need(len(raw)<=maximum,'compact JSON cap');h=hashlib.sha256(raw).hexdigest()
 if expected is not None:need(h==expected,'exact hash '+str(path.relative_to(ROOT)))
 inputs[str(path.relative_to(ROOT))]={'bytes':len(raw),'sha256':h}
 return json.loads(raw)
def bound(binding):return read_json(binding['path'],binding['sha256'])
def weights(x):
 need(len(x)>=2 and (all(b>a for a,b in zip(x,x[1:])) or all(b<a for a,b in zip(x,x[1:]))),'monotone coordinates')
 h=[abs(b-a) for a,b in zip(x,x[1:])];w=[h[0]/2]+[(a+b)/2 for a,b in zip(h,h[1:])]+[h[-1]/2];s=math.fsum(w)
 return [v/s for v in w]
def interp(xs,ys,q):
 if xs[0]>xs[-1]:xs=xs[::-1];ys=ys[::-1]
 need(xs[0]<=q<=xs[-1],'no extrapolation')
 if q==xs[-1]:return ys[-1]
 j=max(0,bisect.bisect_right(xs,q)-1);t=(q-xs[j])/(xs[j+1]-xs[j]);return ys[j]+t*(ys[j+1]-ys[j])
def named(row):return f'{row[0]}-N{row[1]}-S{row[2]}-alpha-minus8'
def profile(candidate,original,readouts):
 fit=candidate['fit'];n=d=0.;branches={}
 for b in ('compression','tension'):
  row=fit['modes'][b];prior=original['modes'][b];x=row['coordinate'];y=row['observed'];g=row['reference_prediction'];w=weights(x)
  need(len(x)==len(y)==len(g)==30,'30 complete values per branch')
  need(x==prior['coordinate'] and y==prior['observed'] and row['source_sha256']==prior['source_sha256'],'same exposed coordinates/values/source')
  need(row['weights']==prior['weights'],'same stored branch weights')
  for wa,wb in zip(row['weights'],w):close(wa,.5*wb,'equal-mode quadrature')
  ref=readouts[b]
  need(ref['steps']==60 and len(ref['load_coordinate_full_m'])==61,'matched N24 S60')
  for q,v in zip(x,g):close(interp(ref['load_coordinate_full_m'],ref['applied_force_N'],q),v,'independent signed interpolation')
  n+=math.fsum(.5*ww*gg*yy for ww,gg,yy in zip(w,g,y));d+=math.fsum(.5*ww*gg*gg for ww,gg in zip(w,g))
  branches[b]=(x,y,g,w)
 close(n,fit['numerator'],'profile numerator');close(d,fit['denominator'],'profile denominator')
 close(fit['denominator_minimum'],1e-12*(1000*.004*.004)**2,'identifiability floor')
 need(d>fit['denominator_minimum'] and n/d>0,'positive identified common scale')
 scale=n/d;close(scale,fit['scale'],'profile scale');close(1000*scale,fit['mu_Pa'],'profile mu')
 need(fit['reference_mu_Pa']==1000 and not fit['held_out_values_used'] and not fit['offset_or_sign_fitted'],'fit scope')
 rms_values=[];nrms_values=[];summary={}
 for b,(x,y,g,w) in branches.items():
  predicted=[scale*gg for gg in g];residual=[p-yy for p,yy in zip(predicted,y)]
  rms=math.sqrt(math.fsum(ww*ee*ee for ww,ee in zip(w,residual)));mae=math.fsum(ww*abs(ee) for ww,ee in zip(w,residual));den=math.sqrt(math.fsum(ww*yy*yy for ww,yy in zip(w,y)))
  metric=candidate['metrics']['branches'][b]
  need(metric['row_count']==30 and metric['response_unit']=='N','metric denominator/units')
  for key,value in [('RMSE',rms),('MAE',mae),('observed_RMS_denominator',den),('normalized_RMSE',rms/den),('signed_endpoint_bias',residual[max(range(30),key=lambda i:abs(x[i]))])]:close(metric[key],value,'metric '+b+'/'+key)
  for key,values in [('weights',w),('predicted_at_observed_coordinates',predicted),('residuals',residual)]:
   need(len(metric[key])==30,'metric vectors')
   for actual,value in zip(metric[key],values):close(actual,value,'per-value '+b+'/'+key)
  rms_values.append(rms);nrms_values.append(rms/den);summary[b]={'RMSE_N':rms,'normalized_RMSE':rms/den,'endpoint_bias_N':metric['signed_endpoint_bias'],'negative_observed_values':sum(v<0 for v in y)}
 rms=math.sqrt(math.fsum(v*v for v in rms_values)/2);nrms=math.sqrt(math.fsum(v*v for v in nrms_values)/2)
 close(candidate['metrics']['equal_branch_RMSE'],rms,'equal-branch RMS');close(candidate['metrics']['equal_branch_normalized_RMSE'],nrms,'equal-branch normalized RMS')
 need(candidate['scope']=='same_exposed_DEVELOPMENT_model_adequacy' and candidate['alpha_fit_optimizer_calls']==0 and candidate['physical_validation_pass'] is None and not candidate['new_independent_validation'],'candidate scope')
 return {'mu_Pa':1000*scale,'scale':scale,'equal_branch_RMSE_N':rms,'equal_branch_normalized_RMSE':nrms,'branches':summary}
def pair(coarse,fine,saved,kind):
 stride=fine['steps']//60;x=coarse['load_coordinate_full_m'];xx=fine['load_coordinate_full_m'][::stride]
 need(len(x)==61 and x==xx,'exact common-state coordinates')
 f0=coarse['applied_force_N'];f1=fine['applied_force_N'][::stride]
 p0=coarse['probe_displacements_m'];p1=fine['probe_displacements_m'][::stride]
 delta=[b-a for a,b in zip(f0,f1)];maxp=0.
 for i,(ca,fi) in enumerate(zip(p0,p1)):
  need(len(ca)==len(fi)==75,'75 fixed probes')
  for j,(a,b) in enumerate(zip(ca,fi)):
   need(len(a)==len(b)==3,'three-component probes');dp=[bb-aa for aa,bb in zip(a,b)];maxp=max(maxp,math.sqrt(math.fsum(v*v for v in dp)))
   for actual,value in zip(saved['signed_probe_component_difference_m'][i][j],dp):close(actual,value,'signed probe difference')
 for actual,value in zip(saved['signed_force_difference_N'],delta):close(actual,value,'signed force difference')
 floor,relative,limitp=PAIRS[kind];limitf=floor+relative*max(map(abs,f1));maxf=max(map(abs,delta))
 for key,value in [('force_limit_N',limitf),('maximum_force_change_N',maxf),('maximum_probe_norm_change_m',maxp),('probe_norm_limit_m',limitp)]:close(saved[key],value,'pair '+key)
 need(saved['pairwise_limit_passed']==(maxf<=limitf and maxp<=limitp),'independent pair predicate')
 return saved['pairwise_limit_passed']
def numerical(row,report):
 need(report['steps']==row[2] and report['frame_count']==row[2]+1,'every solved frame')
 x=report['load_coordinate_full_m'];f=report['applied_force_N'];energy=report['energy_J'];work=[0.]
 for i in range(1,len(x)):work.append(work[-1]+(x[i]-x[i-1])*(f[i]+f[i-1])/2)
 delta=[e-energy[0] for e in energy];maximum=max(abs(w-e) for w,e in zip(work,delta));limit=1e-6*1000*.004**2*.00489159+.02*max(max(map(abs,work)),max(map(abs,delta)))
 for key,value in [('maximum_error_J',maximum),('limit_J',limit)]:close(report['full_energy_work'][key],value,'independent energy/work '+key)
 for key,calculated in [('work_J',work),('energy_change_J',delta)]:
  for actual,value in zip(report['full_energy_work'][key],calculated):close(actual,value,'energy/work vector')
 need(report['full_energy_work']['passed']==(maximum<=limit),'energy/work predicate')
 values=report['criteria_max_ratio'];need(values and all(math.isfinite(x) and x>=0 for x in values.values()),'finite numerical ratios')
 passed=report['solver']['passed'] and report['full_energy_work']['passed'] and (report['native_energy_work'] is None or report['native_energy_work']['passed']) and all(v<=1 for v in values.values()) and report['minimum_sampled_J']>0 and report['minimum_logged_J']>0
 need(report['numerical_passed']==passed,'numerical predicate')
 need(report['logged_sed_used_for_energy_gate'] is False and report['sampled_J_positivity_is_not_everywhere_proof'] is True,'independent energy/J scope')
 return passed

def audit_recovery(args):
 terminal=read_json(OUT/'terminal.json',args.terminal_sha256)
 comparison=read_json(OUT/'comparison.json',args.comparison_sha256,512*1024)
 release=bound(terminal['release']);declaration=bound(release['declaration'])
 need(terminal['release']['sha256']==args.release_sha256,'root recovery release pin')
 need(release['execution_released'] and release['source_commit']=='a9ffe72b291be73d50b702962a77b162110bcd10','released committed recovery')
 need(release['output_directory']==str(OUT.relative_to(ROOT)) and release['caps']['native_calls']==release['caps']['numerical_replays']==0,'saved-only declared scope')
 need(terminal['schema']=='hbe-ogden-saved-recovery-terminal-v1' and terminal['status']=='completed_saved_readout_shape_comparison','recovery terminal complete')
 need(terminal['native_calls']==terminal['numerical_replays']==0 and terminal['physical_validation_pass'] is None,'no replay or physical promotion')
 recovery=comparison['saved_readout_recovery']
 need(recovery['upstream']==terminal['upstream']==release['upstream'],'exact upstream joins')
 upstream=terminal['upstream'];prior=bound(upstream['terminal']);request=bound(upstream['request']);oldrelease=bound(upstream['release'])
 need(upstream['terminal']['sha256']=='ca8127909803be8b64758504fc8a01f3d56040c8153c7df0b39fffcbf978a74e' and prior['status']=='failed_shape_comparison','failed original immutable')
 priorpub=read_json(ORIGINAL/'publication.json','ce25dff1dc3f31a5a1e0ec599719a04b6710530282803cd9f39b418288df3a6f')
 need(priorpub['accepted'] is False and not (ORIGINAL/'comparison.json').exists(),'failed publication retained')
 need(oldrelease['declaration']==release['declaration'] and declaration['fit']==comparison['source_fit'],'unchanged scientific declaration and exposed fit')
 need(recovery['original_attempt_preserved'] and not recovery['numerical_screen_recomputed'],'new arithmetic only')
 need(recovery['readouts']==request['reports'] and len(recovery['readouts'])==6,'same six readouts')
 accounting=terminal['accounting'];need(accounting==recovery['accounting'],'accounting joins')
 need(accounting['prior_attempt']=={'baseline_reproductions':1,'completed_reports':0,'positive_scale_profiles':2} and accounting['recovery_attempt']=={'baseline_reproductions':1,'positive_scale_profiles':2} and accounting['cumulative']=={'baseline_reproductions':2,'positive_scale_profiles':4},'truthful cumulative arithmetic')
 need(accounting['new_measured_member_reads']==accounting['new_native_calls']==accounting['numerical_replays']==accounting['alpha_optimizer_calls']==0,'zero new sources/solves/alpha optimization')
 publication=read_json(OUT/'publication.json');need(publication['accepted'] and publication['physical_validation_pass'] is None and publication['release']==terminal['release'],'accepted recovery only')
 need(set(publication['outputs'])==set(terminal['output_inventory'])|{'terminal.json'},'complete publication inventory')
 for name,b in publication['outputs'].items():
  raw=(OUT/name).read_bytes();need(len(raw)==b['bytes'] and hashlib.sha256(raw).hexdigest()==b['sha256'],'published output fixity '+name)
  if name!='terminal.json':need(b==terminal['output_inventory'][name],'terminal inventory join')
 need(publication['outputs']['terminal.json']['sha256']==args.terminal_sha256 and publication['outputs']['comparison.json']['sha256']==args.comparison_sha256,'root output pins')
 need(read_json(OUT/'arithmetic-accounting.json')==accounting,'saved accounting identity')
 need(read_json(OUT/'worker/receipt.json')==terminal['supervision']['receipt'],'owned worker receipt identity')
 c=terminal['supervision']['cleanup'];stage=terminal['supervision']['receipt']['readout_stage'];caps=release['caps']
 need(c['contained'] and c['direct_child_reaped'] and not c['remaining_members'] and not c['errors'] and not c['fallback_used'] and c['exit_code']==0,'clean recovery reap')
 need(stage['status']=='completed_within_caps' and stage['exit_code']==0 and stage['kill_reason'] is None,'owned recovery success')
 need(stage['elapsed_seconds']<terminal['elapsed_seconds']<caps['total_seconds'] and stage['elapsed_seconds']<stage['wall_cap_seconds'] and stage['peak_sampled_process_group_rss_bytes']<=caps['sampled_group_rss_bytes'] and sum(x['bytes'] for x in publication['outputs'].values())<=caps['new_output_bytes'],'recovery bounded resources')
 for path,sha in release['source_bindings'].items():
  raw=(ROOT/path).read_bytes();need(hashlib.sha256(raw).hexdigest()==sha,'source closure '+path)
 need(comparison['alpha_candidates']==[2.,-8.] and comparison['observed_samples']==60 and comparison['specimen']=='HBE_01_03' and comparison['specimen_role']=='DEVELOPMENT','two fixed candidates/same sixty values')
 need(comparison['grid_for_comparison']=='N24/S60' and comparison['fixed_K_over_mu']==149/3,'matched grid/common compressibility')
 for key in ('numerical_qualification_passed','continuum_error_bound','held_out_values_used','fitted_mu_native_confirmed','patient_properties_assigned'):need(comparison[key] is False,'no scope promotion '+key)
 need(comparison['physical_validation_pass'] is None and comparison['empirical_tolerance'] is None,'no physical threshold claim')
 original=bound(comparison['source_fit']);screen=bound(upstream['screen']);readouts={};numerical_summary={}
 for row in RUNS:
  record=bound(recovery['readouts'][named(row)]);need(numerical(row,record),'independent saved numerical predicate '+named(row));readouts[row]=record
  need(record['solver']['normal_termination'],'normal solver termination')
  states=record['solver']['states'];need(len(states)==row[2],'all noninitial solver states')
  for state in states:need(0<=state['actual_N2']<=state['limit_N2'],'every solver residual below limit')
  numerical_summary[named(row)]={'numerical_passed':True,'frames':record['frame_count'],'minimum_sampled_J':record['minimum_sampled_J'],'minimum_logged_J':record['minimum_logged_J'],'largest_saved_criterion_ratio':max(record['criteria_max_ratio'].values()),'independent_energy_work_ratio':record['full_energy_work']['maximum_error_J']/record['full_energy_work']['limit_J']}
 need(screen['passed'] and screen['diagnostic_only'] and not screen['old_v5_numerical_qualification_passed'] and not screen['continuum_error_bound'] and screen['physical_validation_pass'] is None,'original numerical screen scope')
 pair_summary={}
 for b in ('compression','tension'):
  for kind,coarse,fine in [('mesh',(b,16,60),(b,24,60)),('time',(b,24,60),(b,24,120))]:
   row=screen['pairs'][b+'/'+kind];need(pair(readouts[coarse],readouts[fine],row,kind),'every original new-alpha spatial/time array')
   pair_summary['-8/'+b+'/'+kind]={k:v for k,v in row.items() if not isinstance(v,(list,dict))}
 manifest=bound(declaration['evidence']['manifest']);old={};full_pair_cache={}
 for b in ('compression','tension'):
  old[b]={n:bound(manifest['rows'][f'{b}:N{n}:S60:reference']['readout']) for n in (16,24)}
  need(recovery['baseline_references'][b]==manifest['rows'][f'{b}:N24:S60:reference']['readout'],'baseline reference exact identity')
  compact=comparison['baseline_pair_screen'][b];pointer=compact['complete_signed_arrays'];bind=pointer['binding']
  need(bind==declaration['evidence']['comparison'] and compact['reference_identity_manifest']==declaration['evidence']['manifest'],'compact/full original array source')
  if bind['sha256'] not in full_pair_cache:full_pair_cache[bind['sha256']]=read_json(bind['path'],bind['sha256'],16*1024**2)
  full=full_pair_cache[bind['sha256']]
  for key in pointer['json_path']:full=full[key]
  need({k:v for k,v in full.items() if k not in ('signed_force_difference_N','signed_probe_component_difference_m')}=={k:v for k,v in compact.items() if k not in ('complete_signed_arrays','reference_identity_manifest')},'compact baseline scalars losslessly joined')
  need(pair(old[b][16],old[b][24],full,'mesh'),'baseline complete signed arrays')
  pair_summary['2/'+b+'/mesh']={k:v for k,v in full.items() if not isinstance(v,(list,dict))}
 summaries={'2':profile(comparison['candidates']['2'],original,{b:old[b][24] for b in old}),'-8':profile(comparison['candidates']['-8'],original,{b:readouts[b,24,60] for b in old})}
 # Independently reproduce retained original scale without native calls or a project fitter.
 numerator=denominator=0.
 for b in ('compression','tension'):
  row=original['modes'][b];w=weights(row['coordinate']);g=row['reference_prediction'];y=row['observed'];numerator+=math.fsum(.5*ww*gg*yy for ww,gg,yy in zip(w,g,y));denominator+=math.fsum(.5*ww*gg*gg for ww,gg in zip(w,g))
 close(numerator/denominator,original['scale'],'original retained fine-grid scale reproduction');close(1000*numerator/denominator,original['mu_Pa'],'original retained fine-grid mu reproduction')
 need(comparison['original_alpha2_baseline']['reproduced'] is True,'original scale reproduction reported');close(comparison['original_alpha2_baseline']['mu_Pa'],original['mu_Pa'],'original baseline report')
 for candidate in comparison['candidates'].values():
  need(candidate['fit']['fitted_parameters']==['mu_Pa'],'only mu fitted')
  need(candidate['metrics']['empirical_tolerance'] is None and not candidate['metrics']['independent_donor_validation'] and candidate['metrics']['physical_validation_pass'] is None,'metrics no physical validation')
 delta=summaries['-8']['equal_branch_RMSE_N']-summaries['2']['equal_branch_RMSE_N']
 return {'status':'PASS_RECOVERY_SAVED_ARITHMETIC','checks':checks,'inputs':inputs,'profiles':summaries,'delta_equal_branch_RMSE_N':delta,'relative_equal_branch_RMSE_reduction':-delta/summaries['2']['equal_branch_RMSE_N'],'old_fine_grid_reference_fit_mu_Pa':original['mu_Pa'],'numerical_readout_summaries':numerical_summary,'full_pair_checks':pair_summary,'arithmetic_accounting':accounting,'costs':{'recovery_parent_seconds':terminal['elapsed_seconds'],'recovery_child_seconds':stage['elapsed_seconds'],'peak_sampled_child_group_rss_bytes':stage['peak_sampled_process_group_rss_bytes'],'published_output_bytes':sum(x['bytes'] for x in publication['outputs'].values()),'new_native_calls':0,'numerical_replays':0},'failed_original_retained':True,'scope':'Independent arithmetic from the already exposed 60 signed DEVELOPMENT values and authenticated saved numerical JSONs. No FE, raw specimen archive, raw solver primitives, project fitting imports, tuning or alpha optimization.','limitations':['In-sample fit improvement is not independent physical validation.','Two fixed exponents and one positive common scale; no fitted-mu FE confirmation or continuum error bound.','Numerical scalar and signed spatial/time arrays were reconstructed, but primitive FE geometry/force fields were not replayed.','Tension residual may remain larger than the observed signal; improvements in a pooled metric do not establish model adequacy.']}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--terminal-sha256',required=True);p.add_argument('--comparison-sha256',required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--audit-output',type=Path,required=True);args=p.parse_args()
 try:result=audit_recovery(args)
 except Exception as error:result={'status':'AUDITOR_FAILURE','checks':checks,'error':str(error),'traceback':traceback.format_exc(),'inputs':inputs}
 with args.audit_output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'checks':result['checks']}))
 if result['status']=='AUDITOR_FAILURE':raise SystemExit(1)
