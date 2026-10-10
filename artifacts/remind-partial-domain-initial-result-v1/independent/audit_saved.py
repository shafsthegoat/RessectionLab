"""Saved JSON/source audit only. Run after root confirms terminal; no array reads."""
import argparse, collections, hashlib, json, math, os, resource, signal, stat, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'build/remind-partial-domain-public-preparation-v1'
OUT=Path(__file__).parent
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']
seen={};checks=0
def need(ok,why):
 global checks
 checks+=1
 if not ok:raise AssertionError(why)
def raw(path):
 path=Path(path)
 need(path.is_relative_to(ROOT) and '..' not in path.parts and path.suffix in {'.json','.py','.log'},'only bounded saved metadata/source/logs')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  info=os.fstat(fd);need(stat.S_ISREG(info.st_mode) and info.st_size<=2*1024**2,'regular small file')
  with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(2*1024**2+1)
 finally:os.close(fd)
 need(len(data)<=2*1024**2,'bounded read')
 key=str(path.relative_to(ROOT));h=hashlib.sha256(data).hexdigest()
 need(key not in seen or seen[key]['sha256']==h,'unchanged input '+key)
 seen[key]={'sha256':h,'bytes':len(data)}
 return data
def read(p):return json.loads(raw(p),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def digest(value):return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main(result_sha,release_path,release_sha):
 started=time.monotonic()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60-second saved audit limit')));signal.alarm(60)
 run=PREP/'attempt-01';supervision=PREP/'attempt-01.supervision'
 need(sha(run/'result.json')==result_sha and sha(release_path)==release_sha,'root terminal/release pins')
 result=read(run/'result.json');release=read(release_path);receipt=read(supervision/'receipt.json')
 need(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] and receipt['stop_reason'] is None,'clean owned terminal')
 need(not receipt['remaining_owned_pids'] and not receipt['cleanup_errors'] and not receipt['cleanup_actions'],'clean owned cleanup')
 need(receipt['result_sha256']==result_sha and receipt['release_sha256']==result['release_sha256']==release_sha,'release/result/receipt joins')
 caps=release['caps'];need(receipt['caps']==caps and receipt['elapsed_seconds']<caps['parent_seconds'] and result['elapsed_seconds']<caps['worker_seconds'],'wall caps')
 need(receipt['sampled_peak_rss_bytes']<=caps['sampled_rss_bytes'] and receipt['output_bytes']<=caps['output_bytes'] and receipt['samples']>0,'sampled resource caps')
 need(release['execution_released'] is True and release['subjects']==SUBJECTS and result['all_four_cases_retained'] and not result['failed_cases_replaced'],'fixed complete denominator')
 need(receipt['source_index']==release['source_index'],'source index receipt join')
 need(sha(ROOT/release['source_index']['path'])==release['source_index']['sha256'],'source index pin')
 index=read(ROOT/release['source_index']['path'])
 for name,h in index['files'].items():need(sha(ROOT/name)==h,'source/metadata '+name)
 public=read(ROOT/release['public_index']['path']);need(sha(ROOT/release['public_index']['path'])==release['public_index']['sha256'],'public manifest index pin')
 need([r['patient_id'] for r in result['cases']]==SUBJECTS and [r['patient_id'] for r in public['cases']]==SUBJECTS,'four ordered cases')
 need(result['proposal_config']==release['proposal_config'] and result['max_steps']==24 and result['historical13axis120'] is True and result['retention_applied'] is False,'fixed initial proposal condition')
 for k in ('model_calls','optimizer_calls','checkpoint_loads','search_calls','transition_calls','blocked_file_or_external_calls','source_arrays_written'):need(result[k]==0,'zero '+k)
 need(result['training_admitted'] is False and result['methods_executed']==[] and result['clinical_claim'] is False,'no training/method/clinical inference')
 budget=result['native_budget'];need(budget['status']=='complete_history_awaiting_independent_audit' and budget['native_preview_entries']<=caps['max_native_previews'],'construction-only budget completed')
 need(read(run/'construction-ledger.json')==result['cases'],'construction ledger exact')
 summaries=[];expected_loads=[];previews=0
 for row,ref in zip(result['cases'],public['cases']):
  subject=row['patient_id'];base=run/subject;manifest=read(Path(ref['path']));need(sha(Path(ref['path']))==ref['sha256'],'case manifest pin')
  need(row==read(base/'construction-result.json'),'per-case result exact')
  need(row['live_source_or_task_after_cleanup']==0,'per-case source/task released')
  previews+=row['native_preview_entries']
  if row['status']=='case_failed_retained':
   summaries.append({'subject':subject,'status':row['status'],'error':row['error']});continue
  need(row['status']=='constructed_initial_inventory_only','completed case disposition')
  inv=read(base/'initial-inventory.json');coverage=read(base/'initial-public-proposal-coverage.json');context=read(base/'context.json')
  bindings=read(base/'admitted-public-bindings.json');assumed=read(base/'derived-occupancy-assumption.json')
  expected_loads.extend(str(Path(x['path']).resolve()) for x in manifest['input_files'].values())
  need(digest(context)==row['context_hash'] and context['source_hash']==inv['source_hash']==row['source_hash'],'case source/context join')
  need(context['decision_model_hash']==inv['decision_model_hash'] and context['public_source_binding_hash']==digest(bindings['source_binding']) and context['qc_receipt_hash']==digest(bindings['qc']),'model/source/QC joins')
  need(context['role']=='TRAIN' and context['subject']==subject and context['max_optimizer_updates']==0 and context['budgets']['max_policy_forwards']==0,'TRAIN zero model/update')
  need(context['execution_kind']=='search_only_no_policy' and not context['private_reference_in_task'] and not context['policy_comparison_permitted'],'public-only qualification')
  need(inv['steps_taken']==row['steps_taken']==0 and inv['max_steps']==24 and not inv['terminal'],'initial inventory only')
  emitted=[x for x in inv['ledger'] if x['proposal_reason']=='PROPOSED_UNCERTIFIED'];need(emitted==inv['emitted'],'exact emitted ledger subset')
  reasons=dict(collections.Counter(x['proposal_reason'] for x in inv['ledger']));preview_reasons=dict(collections.Counter(x['reason'] for x in emitted))
  accepted=[x for x in emitted if x['feasible']];omitted=reasons.get('CANDIDATE_CAP',0);duplicate=reasons.get('DUPLICATE_GEOMETRY',0)
  need(len(inv['ledger'])==inv['declared_slots'] and reasons==inv['disposition_counts']==row['proposal_dispositions']==coverage['proposal_dispositions'],'full declared-slot dispositions')
  need(len(emitted)==inv['emitted_count']==row['emitted_count']==inv['evaluated_slots']<=120 and len(accepted)==inv['accepted_count']==row['accepted_count'],'emission/legal arithmetic')
  need(inv['rejected_count']==len(emitted)-len(accepted) and inv['omitted_count']==omitted and inv['duplicate_count']==duplicate and inv['unavailable_count']==len(inv['ledger'])-len(emitted)-omitted-duplicate,'omitted distinct from rejected')
  need(inv['complete']==(omitted==0) and inv['ledger_complete'] and coverage['preview_dispositions']==preview_reasons,'catalog completeness scope')
  need(coverage['catalog_metadata']=={k:v for k,v in inv.items() if k not in {'ledger','emitted'}},'coverage metadata identity')
  need(len({x['action_id'] for x in emitted})==len(emitted),'unique emitted physical action records')
  total=manifest['public_target_support_consistency']['whole_tumor_positive_voxels'];counts=manifest['public_source_domain_counts'];domains=row['source_and_simulated_domains'];n=math.prod(manifest['shape_xyz'])
  need(sum(counts[k] for k in ('target_in_known_support_positive','target_in_known_support_zero','target_in_unknown_support_domain'))==total,'full placed target partition')
  need(domains['source_support_known_voxels']==counts['support_domain'] and domains['source_support_unknown_voxels']==n-counts['support_domain'],'Ds known/unknown denominator')
  unknown=counts['target_in_unknown_support_domain'];need(domains['target_in_unknown_support_voxels']==domains['assumed_material_beyond_source_domain_voxels']==unknown and domains['unavailable_interaction_voxels']==n-counts['support_domain']-unknown,'D equals Ds union positive T')
  need(domains==context['source_and_simulated_domains']==bindings['source_binding']['source_and_simulated_domains']==assumed['source_and_simulated_domains'] and domains['source_domain_extended'] is False,'same unextended Ds/domain metadata')
  need(total==row['target_denominator']==coverage['nominal_target_centers_total']==row['full_target_extent']['full_region_positive_voxels'],'unchanged full placed T')
  need(row['target_native_and_NN_sampling']==manifest['public_label_resampling'] and row['full_target_preserved'] and not row['source_domain_extended'],'native versus placed differences retained')
  need(row['fixed_access_outcome']==('legal_emitted_motion_exists' if accepted else 'no_legal_emitted_motion'),'fixed access classification')
  need(coverage['accepted_envelope']==row['accepted_envelope'] and coverage['emitted_envelope']==row['emitted_envelope'],'envelope summary join')
  summaries.append({'subject':subject,'status':row['status'],'emitted':len(emitted),'accepted':len(accepted),'rejected':len(emitted)-len(accepted),'cap_omitted':omitted,'dispositions':reasons,'preview_reasons':preview_reasons,'full_target_cells':total,'target_unknown_support_cells':unknown,'within_grid_unavailable_cells':domains['unavailable_interaction_voxels'],'native_preview_entries':row['native_preview_entries'],'target_depth_range_mm':row['target_depth_range_mm'],'accepted_envelope':row['accepted_envelope'],'source_hash':row['source_hash']})
 need(previews==budget['native_preview_entries'],'all four preview budget accounting')
 if all(x['status']=='constructed_initial_inventory_only' for x in result['cases']):need(collections.Counter(result['public_array_loads'])==collections.Counter(expected_loads) and len(expected_loads)==20,'exact twenty public array loads reported')
 for path in list(seen):raw(ROOT/path)
 elapsed=time.monotonic()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 need(elapsed<60 and peak<=512*1024**2,'independent audit resources')
 audit={'status':'PASS_saved_metadata_only','checks':checks,'elapsed_seconds':elapsed,'peak_rss_bytes':peak,'root_result_sha256':result_sha,'root_release_sha256':release_sha,'receipt_sha256':sha(supervision/'receipt.json'),'parent_seconds':receipt['elapsed_seconds'],'parent_peak_sampled_rss_bytes':receipt['sampled_peak_rss_bytes'],'native_preview_entries':previews,'cases':summaries,'limitation':'No patient array opens or geometry replay. Initial emitted feasibility only; cap omissions untested; no episode, global reachability, learning, anatomical or clinical success inferred.'}
 write('audit-result.json',audit);write('input-hashes.json',seen);print(json.dumps(audit,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result-sha256',required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);a=p.parse_args();main(a.result_sha256,a.release,a.release_sha256)
