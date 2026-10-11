"""Terminal-only independent sparse-geometry/scalar audit; never runs native code.

Requires exact root terminal/report hashes. Imports only the frozen evaluator,
which operates on checked saved boxes/bitsets and analytic generated planes.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,math,struct,time,traceback
ROOT=Path(__file__).resolve().parents[3]
STAGE=ROOT/'build/native-partial-volume-v1';RUN=STAGE/'run-01'
checks=0;inputs={}
def need(value,label):
 global checks
 checks+=1
 if not value:raise AssertionError(label)
def close(a,b,label,tol=1e-8):need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,label)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path,expected=None,maximum=16*1024**2):
 path=ROOT/path if isinstance(path,str) else path
 need(path.resolve().is_relative_to(ROOT) and path.suffix=='.json','bounded local saved JSON')
 raw=path.read_bytes();need(len(raw)<=maximum,'JSON size cap');h=hashlib.sha256(raw).hexdigest()
 if expected:need(h==expected,'input bytes '+str(path.relative_to(ROOT)))
 inputs[str(path.relative_to(ROOT))]={'sha256':h,'bytes':len(raw)}
 return json.loads(raw)
def binding(b):return read(b['path'],b['sha256'])
def semantic(x):return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def legacy_digest(x):return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def projection(row):
 return {'native':{k:v for k,v in row['native'].items() if k!='committed_history_hash'},'actions':[{k:v for k,v in a.items() if k not in ('source_state_hash','result_state_hash','native_seconds','partial_evacuation')} for a in row['actions']]}
def equivalent(a,b,label,tol=1e-10):
 if isinstance(a,dict) and isinstance(b,dict):
  need(set(a)==set(b),label+' keys')
  for k in a:equivalent(a[k],b[k],label+'/'+k,tol)
 elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
  need(len(a)==len(b),label+' length')
  for i,(x,y) in enumerate(zip(a,b)):equivalent(x,y,label+'/'+str(i),tol)
 elif type(a) in (int,float) and type(b) in (int,float):close(a,b,label,tol)
 else:need(type(a)==type(b) and a==b,label)
def det(a):return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])

def audit(args):
 started=time.perf_counter();r=read(RUN/'report.json',args.report_sha256);launch=read(RUN/'launcher.json',args.launcher_sha256)
 index=read(STAGE/'runner/source-index.json','c625f7c69e1cfa2bb2dbd7fc4051c6fa20c3a24bc75c623006ebabf1fb952263');cfg=read(STAGE/'runner/config.json')
 need(index['execution_released'] and launch['source_index_sha256']==inputs[str((STAGE/'runner/source-index.json').relative_to(ROOT))]['sha256'],'root source release')
 for name,h in index['files'].items():need(sha(ROOT/name)==h,'source/reference unchanged '+name)
 need(r['source']['files']==index['files'],'report executed closure')
 need(launch['report_sha256']==args.report_sha256,'receipt report join')
 need(launch['status']==r['status']=='completed' and launch['worker_exit_code']==0 and launch['stop_reason'] is None,'owned terminal completion')
 need(launch['child_reaped'] and not launch['remaining_owned_pids'] and not launch['cleanup_errors'],'clean owned reap')
 need(launch['caps']==cfg['caps'] and launch['elapsed_seconds']<cfg['caps']['parent_wall_seconds'] and launch['sampled_peak_group_rss_bytes']<=cfg['caps']['worker_peak_rss_bytes'] and launch['output_bytes_before_receipt']<cfg['caps']['output_bytes'],'declared costs/caps')
 prior=binding(cfg['retained_union_report']);decl=binding(cfg['retained_union_declaration']);default=read(decl['retained_baseline']['report_path'],decl['retained_baseline']['report_sha256'])
 need(r['retained_union_report']==cfg['retained_union_report'] and r['retained_default_report']==decl['retained_baseline'],'retained references')
 spec={**decl,'version':cfg['version'],'caps':cfg['caps']};need(r['declaration_hash']==legacy_digest(spec),'same scientific declaration except explicit mode/caps')
 need(r['source_unchanged'] and r['human_patients']==r['training_updates']==0,'source fixity/no patients or learning')
 expected=[g['name']+'-'+format(h,'g')+'mm' for h in decl['spacings_mm'] for g in decl['grids']]
 need(r['planned_rows']==expected and [x['row_id'] for x in r['rows']]==expected and not r['missing_rows'],'fixed 18-row order')
 need(len(r['rows'])==r['completed_rows']==r['expected_rows']==launch['planned_rows']==18 and r['planned_action_slots']==launch['planned_action_slots']==180,'denominators')
 evaluator_path=STAGE/'independent-review/plane_volume_evaluator.py'
 need(sha(evaluator_path)=='21df1470e7a65c62eaae4857c8b7c435b1ae448783934465d3cf3bdef2a82522','frozen independent plane evaluator')
 module_spec=importlib.util.spec_from_file_location('saved_partial_independent',evaluator_path);E=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(E)
 before={x['row_id']:x for x in prior['rows']};default_rows={x['row_id']:x for x in default['rows']};summaries=[];counts=Counter();work=Counter();native_seconds=rows_seconds=audit_seconds=0.;clipping_seconds=0.;all_parity=True
 for row in r['rows']:
  rid=row['row_id'];old=before[rid];native=row['native'];acts=row['actions'];pe=row['partial_evacuation'];volume=abs(det(row['affine']));leaf_count=2**(3*row['cut_union_spec']['max_depth'])
  need(len(acts)==row['planned_actions']==10,rid+' ten slots')
  for k in ('spacing_mm','grid','shape','affine','source_hash','fixed_tool','physical_actions_sha256','cut_union_spec','static_not_executable','exact_continuous_target_cap_mm3'):need(row[k]==old[k],rid+' fixed '+k)
  equivalent(projection(row),projection(old),rid+' whole-cell/history projection')
  need(projection(row)==projection(old),rid+' EXACT serialized scientific parity')
  need(row['retained_union_projection_sha256']==legacy_digest(projection(old)) and row['current_legacy_projection_sha256']==legacy_digest(projection(row)) and row['current_legacy_projection_sha256']==row['retained_union_projection_sha256'],rid+' independently checked projection hashes')
  prior_state=None;prior_evac=None;delta_count=0;delta_volume=0.;reward_sum=0.;accepted=0;attempted=0;refused=False
  for a in acts:
   counts[a['status']]+=1
   if a['status']=='unexecuted_prior_refusal':need(refused,rid+' suffix after refusal');continue
   need(not refused,rid+' no implicit retry');attempted+=1;native_seconds+=a['native_seconds']
   if prior_state is not None:need(a['source_state_hash']==prior_state,rid+' native state chain')
   for k in ('leaf_tests','new_leaf_count','witness_rows'):work[k]+=a['cut_union_attempt'][k]
   if a['status']=='rejected':
    refused=True;need(a['source_state_hash']==a['result_state_hash'] and 'partial_evacuation' not in a,rid+' rejected state rollback');continue
   need(a['status']=='accepted',rid+' known status');accepted+=1;prior_state=a['result_state_hash'];v=a['partial_evacuation'];prev=v['before'];after=v['after']
   if prior_evac is None:need(prev['certified_leaf_count']==0 and prev['evacuated_source_volume_mm3']==0,rid+' initial empty evacuation')
   else:equivalent(prev,prior_evac,rid+' cumulative evacuation chain',1e-9)
   need(v['spec']==row['public_parent_mass_spec'],rid+' parent mass contract')
   d=after['certified_leaf_count']-prev['certified_leaf_count'];need(d==v['delta_certified_leaf_count']==a['cut_union_committed']['new_leaf_count'] and d>=0,rid+' unique committed leaf additions')
   close(v['delta_evacuated_source_volume_mm3'],d*volume/leaf_count,rid+' peraction source volume');close(after['evacuated_source_volume_mm3']-prev['evacuated_source_volume_mm3'],v['delta_evacuated_source_volume_mm3'],rid+' volume difference')
   delta_count+=d;delta_volume+=v['delta_evacuated_source_volume_mm3'];prior_evac=after
   target=a['nominal_target_source_credit_mm3'];normal=a['removed_source_cell_mm3']-target;reward=decl['reward']
   calculated=reward['target_per_mm3']*target-reward['normal_per_mm3']*normal-reward['action_cost']-reward['motion_per_mm']*2*math.dist(a['entry_mm'],a['tip_mm'])
   close(calculated,a['frozen_reward'],rid+' unchanged whole-cell reward');reward_sum+=a['frozen_reward']
  need(attempted==row['native_calls'] and accepted==native['accepted_actions'] and native['complete_sequence']==(accepted==10),rid+' truthful action/complete count')
  close(reward_sum,native['frozen_reward_total'],rid+' reward total');counts['complete']+=native['complete_sequence'];counts['empty']+=accepted==0
  cert=row['independent_audit'];upper=row['static_not_executable']['contact_physical_upper_mm3'];lower_old=old['native']['removed_physical_mm3'];cap=row['exact_continuous_target_cap_mm3']
  radius=row['fixed_tool']['tip_radius_mm'];height=decl['target_tip_z_mm']+radius-decl['target_start_z_mm'];close(math.pi*height*height*(radius-height/3),cap,rid+' exact target cap')
  if accepted:
   need(cert['status']=='completed' and cert['certificate']['feasible'] and not cert['certificate']['failures'] and cert['certificate']['complete_tool_checked'] and cert['certificate']['frontier_checked'],rid+' independent prefix geometry pass')
   c=cert['certificate'];need(c['source_case_hash']==row['source_hash'] and c['action_count']==accepted and c['checker_version']=='independent-native-partial-evacuation-v1',rid+' checker identity')
   counts['certificates']+=1;audit_seconds+=cert['seconds']
   b=pe['certified_geometry'];sparse=read(RUN/b['path'],b['sha256'],2*1024**2);need(inputs[str((RUN/b['path']).relative_to(ROOT))]['bytes']==b['bytes'],rid+' saved geometry size')
   need(sparse['row_id']==rid and sparse['source_hash']==row['source_hash'] and sparse['history_hash']==native['committed_history_hash'],rid+' saved union provenance')
   g=sparse['geometry'];need(g['native_affine']==row['affine'] and g['source_shape']==row['shape'] and g['cut_union_spec']==row['cut_union_spec'] and g['evacuated_volume_spec']==row['public_parent_mass_spec'],rid+' geometry physical contract')
   need(g['geometry_digest']==semantic({k:v for k,v in g.items() if k!='geometry_digest'})==pe['checker_geometry_digest']==c['evacuated_volume']['geometry_digest'],rid+' checker geometry digest')
   need(g['state_digest']==prior_evac['state_digest']==c['evacuated_volume']['state_digest'],rid+' final volume state')
   removed=g['whole_removed_indices'];partials={tuple(x['index']):int(x['covered_bits_hex'],16) for x in g['partial_cells']};bounds={tuple(x[0]):tuple(x[1:]) for x in sparse['parent_mass_bounds']}
   need(len(partials)==len(g['partial_cells']) and len(bounds)==len(sparse['parent_mass_bounds']),rid+' unique sparse rows')
   for key,(lo,hi) in bounds.items():
    # Recover exactly represented f from its rounding-bin center, then verify enclosure.
    f=struct.unpack('!f',struct.pack('!f',(lo+hi)/2))[0];expected_bounds=E.float32_fraction_enclosure(f)
    need((lo,hi)==expected_bounds,rid+' exact float32 parent quantization enclosure')
   count=len(removed)*leaf_count+sum(x.bit_count() for x in partials.values());need(count==delta_count==prior_evac['certified_leaf_count']==c['evacuated_volume']['certified_leaf_count'],rid+' unique final union count')
   need(len(removed)==native['removed_cells'] and len(partials)==row['coverage']['partially_certified_occupied_cells'],rid+' full and partial occupancy')
   stamp=time.perf_counter();calc=E.evaluate_certified_union(g['native_affine'],g['cut_union_spec']['max_depth'],removed,partials,bounds,tissue_plane_z_mm=0.,target_plane_z_mm=decl['target_start_z_mm']);clipping_seconds+=time.perf_counter()-stamp
   for k,v in calc.items():equivalent(v,pe[k],rid+' independent saved clipping/'+k,1e-8)
   need(calc['physical_target_within_source_interval'] and calc['parent_target_mass_enclosure_pass'],rid+' physical target interval premise and conclusion')
   close(delta_volume,calc['source_evacuated_mm3'],rid+' deltas sum to geometric union')
   equivalent(prior_evac['evacuated_target_source_bounds_mm3'],calc['cumulative_target_interval_source_mm3'],rid+' cumulative target interval',1e-8)
   close(calc['whole_cell_physical_lower_mm3'],lower_old,rid+' whole-cell physical baseline')
   new,target=calc['physical_tissue_evacuated_lower_mm3'],calc['physical_target_evacuated_mm3']
   for k in ('leaf_visits','face_clips','full_parent_boxes','partial_parent_boxes'):work['evaluator_'+k]+=calc['work'][k]
  else:
   need(cert['status']=='not_applicable_rejected_no_committed_history' and pe['status']=='empty_no_committed_history',rid+' empty certificate N/A')
   need('certified_geometry' not in pe and pe['source_evacuated_mm3']==0,rid+' no invented empty geometry');new=target=0.
  need(lower_old-1e-8<=new<=upper+1e-8 and 0<=target<=cap+1e-8,rid+' total and target physical bounds')
  for k,v in [('lower_bound_gain_mm3',new-lower_old),('gap_before_mm3',upper-lower_old),('gap_after_mm3',upper-new)]:close(pe[k],v,rid+' '+k)
  rows_seconds+=row['timing_seconds']['numerical_row_total']
  summaries.append({'row_id':rid,'spacing_mm':row['spacing_mm'],'complete':native['complete_sequence'],'accepted_actions':accepted,'first_refusal':next(({k:a[k] for k in ('index','reason','failure_tip_mm')} for a in acts if a['status']=='rejected'),None),'physical_lower_before_mm3':lower_old,'physical_lower_after_mm3':new,'planned_contact_upper_mm3':upper,'gap_before_mm3':upper-lower_old,'gap_after_mm3':upper-new,'gap_reduction_mm3':new-lower_old,'gap_reduction_fraction':0. if upper==lower_old else (new-lower_old)/(upper-lower_old),'physical_target_before_mm3':old['native']['removed_physical_target_mm3'],'physical_target_after_mm3':target,'exact_continuous_target_cap_mm3':cap,'remaining_target_shortfall_mm3':cap-target,'target_cap_undercredit_fraction':(cap-target)/cap,'source_evacuated_mm3':pe['source_evacuated_mm3'],'frozen_return':native['frozen_reward_total']})
 calls=counts['accepted']+counts['rejected'];need(calls==r['native_call_costs_including_incomplete_row']['native_strokes_started']==r['native_call_costs_including_incomplete_row']['native_strokes_returned'],'all native calls')
 need(counts['accepted']+counts['rejected']+counts['unexecuted_prior_refusal']==180 and counts['complete']==13 and counts['certificates']==15 and counts['empty']==3,'retained union denominators')
 close(native_seconds,r['native_call_costs_including_incomplete_row']['native_seconds'],'native timing sum')
 for k in ('leaf_tests','new_leaf_count','witness_rows'):need(work[k]==r['returned_preview_coverage_work_including_incomplete_row'][k],'all attempted '+k)
 old_complete={x['row_id'] for x in prior['rows'] if x['native']['complete_sequence']};default_complete={x['row_id'] for x in default['rows'] if x['native']['complete_sequence']};complete={x['row_id'] for x in summaries if x['complete']}
 fine=[x for x in summaries if x['spacing_mm']==.25];gains=sorted(complete-default_complete)
 independent_pass=(len(summaries)==18 and len(fine)==6 and all(x['gap_reduction_mm3']>1e-8 for x in fine) and complete==old_complete and not default_complete-complete and len(gains)>=1)
 reported_parity=all(x['legacy_projection_equal_within_tolerance'] for x in r['rows'])
 need(r['useful_criterion']['legacy_action_whole_cell_reward_projection_unchanged']==reported_parity and r['useful_criterion']['meets_predeclared_useful_criterion']==(independent_pass and reported_parity),'original criterion faithfully retained; separate exact saved parity')
 need(r['useful_criterion']['additional_default_completions']==gains and not r['useful_criterion']['lost_union_completions'] and not r['useful_criterion']['lost_default_completions'],'unchanged completion gains')
 need(native_seconds<=rows_seconds and rows_seconds+audit_seconds<=r['worker_seconds']+1e-6 and r['worker_seconds']<launch['elapsed_seconds'],'nested experiment timing')
 need(calls<=cfg['caps']['max_native_strokes'] and r['peak_worker_rss_bytes']<=cfg['caps']['worker_peak_rss_bytes'],'all calls/peak cap')
 return {'status':'PASS','checks':checks,'inputs':inputs,'counts':dict(counts),'meets_primary_criterion_using_exact_saved_parity':independent_pass,'original_reported_criterion':r['useful_criterion'],'original_false_parity_rows':[x['row_id'] for x in r['rows'] if not x['legacy_projection_equal_within_tolerance']],'all_whole_cell_rewards_and_projected_histories_unchanged':True,'rows':summaries,'fine_rows_summary':{'count':len(fine),'minimum_physical_lower_gain_mm3':min(x['gap_reduction_mm3'] for x in fine),'maximum_physical_lower_gain_mm3':max(x['gap_reduction_mm3'] for x in fine),'minimum_gap_reduction_fraction':min(x['gap_reduction_fraction'] for x in fine),'maximum_gap_reduction_fraction':max(x['gap_reduction_fraction'] for x in fine)},'work':dict(work),'costs':{'native_seconds':native_seconds,'numerical_rows_seconds':rows_seconds,'independent_audits_plus_clipping_seconds':audit_seconds,'worker_seconds':r['worker_seconds'],'parent_seconds':launch['elapsed_seconds'],'worker_peak_rss_bytes':r['peak_worker_rss_bytes'],'polled_group_peak_rss_bytes':launch['sampled_peak_group_rss_bytes'],'retained_union_worker_seconds':prior['worker_seconds'],'nonpaired_worker_ratio':r['worker_seconds']/prior['worker_seconds'],'this_saved_geometry_recalculation_seconds':clipping_seconds,'this_audit_elapsed_seconds':time.perf_counter()-started},'limitations':['Certified subcell evacuation is an idealized model assumption, not measured biological removal.','Independent capsule/exposure/shaft proof is the saved canonical checker certificate; this audit independently reclips its exported geometry, without native replay.','Planned contact upper bound includes prescribed but potentially unexecuted actions; it is unchanged and conservative, not exact cut-union volume.','Outer voxelization remains separated from physical volume; partial cells remain conservative whole-cell shaft obstacles.','Whole-cell rewards and prescribed histories remain unchanged; no policy, clinical, force, device or patient benefit is inferred.','Historical timing ratios are descriptive, not a paired speed experiment.']}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report-sha256',required=True);p.add_argument('--launcher-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 try:result=audit(args)
 except Exception as e:result={'status':'AUDITOR_FAILURE','checks':checks,'error':str(e),'traceback':traceback.format_exc(),'inputs':inputs}
 with args.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'checks':checks}))
 if result['status']!='PASS':raise SystemExit(1)
