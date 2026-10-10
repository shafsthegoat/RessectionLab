"""Metadata-only exact preflight for a saved public mask measurement."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUTPUT=HERE/'attempt-01'
SUBJECTS=('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045')
COUNTS=(52,52,10,52);KEYS=('supplied_support','supplied_whole_tumor','supplied_support_domain')
CAPS={'target_seconds':60,'worker_seconds':110,'parent_seconds':120,'sampled_rss_bytes':1536*1024**2,'output_bytes':16*1024**2,'supervision_bytes':8*1024**2,'log_bytes':4*1024**2,'threads':1,'max_geometric_queries':664,'attempts':1,'automatic_retry':False}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def read_bound(record):
 p=ROOT/record['path'];need(p.is_file() and not p.is_symlink() and p.stat().st_size<=2*1024**2,'bounded_regular_metadata')
 raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==record['sha256'],'bound_metadata_changed:'+str(p));return json.loads(raw)
def guard(path,expected,*,execute=False):
 r=read_bound({'path':str(path),'sha256':expected});template=json.loads((HERE/'prepared-release.json').read_bytes());copy=dict(r);copy.update(execution_released=False,expected_head=None)
 need(copy==template and type(r['execution_released']) is bool,'only_release_and_head_may_change')
 need(r['caps']==CAPS and r['subjects']==list(SUBJECTS),'scope_or_caps_changed')
 source=read_bound(r['source_index'])
 for name,digest in source['files'].items():need(sha(ROOT/name)==digest,'source_changed:'+name)
 index=read_bound(r['input_index']);need(tuple(x['patient_id'] for x in index['cases'])==SUBJECTS,'fixed_order')
 result=read_bound(index['original_result']);need(result['status']=='initial_inventory_batch_complete','original_terminal')
 original={x['patient_id']:x for x in result['cases']}
 for row,count in zip(index['cases'],COUNTS):
  manifest=read_bound(row['manifest']);inventory=read_bound(row['inventory']);derivation=read_bound(row['derivation']);bindings=read_bound(row['bindings'])
  need(manifest['role']=='TRAIN' and manifest['patient_id']==row['patient_id'] and not manifest['training_admitted'],'fixed_TRAIN_only')
  need(row['masks']=={k:manifest['input_files'][k] for k in KEYS},'exact_S_T_Ds_no_MRI')
  need(inventory['emitted_count']==count and len(inventory['emitted'])==count and inventory['accepted_count']==0 and inventory['omitted_count']==0,'saved_motion_denominator')
  need(inventory['source_hash']==original[row['patient_id']]['source_hash'],'saved_case_source_join')
  need(bindings['source_binding']['source_hash']==inventory['source_hash'] and bindings['source_binding']['subject']==row['patient_id'],'saved_native_frame_source_join')
  need(bindings['native_grid_reconciliation']['original_affine_ras_mm']==manifest['affine_ras_mm'] and bindings['native_grid_reconciliation']['shape']==manifest['shape_xyz'],'saved_native_frame_original_join')
  need(all(x['reason']=='UNKNOWN_DOMAIN:FORBIDDEN_COLLISION' and x['feasible'] is False for x in inventory['emitted']),'saved_reason_changed')
  need(derivation['rule']=='fixed_source_axis0_target_centroid_side_of_support_extent' and derivation['radius_mm']==6.0,'fixed_aperture')
 if execute:
  need(r['execution_released'] is True,'root_dispatch_required')
  need(r['expected_head']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'HEAD_changed')
 return r,index,True

def complete_result(result,release,expected):
 need(result.get('status')=='saved_mask_diagnostic_complete' and result.get('release_sha256')==expected,'terminal_result_binding')
 need(tuple(x['patient_id'] for x in result['cases'])==SUBJECTS,'all4_denominator')
 need(result['geometric_queries']==664 and result['mask_loads']==12,'exact_queries_and_loads')
 need(all(result[k]==0 for k in ('native_previews','transitions','search_calls','models','source_array_writes','blocked_calls')),'zero_forbidden_work')
 need(result['state_or_source_modified'] is False and result['training_admitted'] is False,'no_admission')
 for row,count in zip(result['cases'],COUNTS):
  need(row['status']=='measured_only' and row['saved_motions']==count and row['geometric_queries']==4*count,'case_complete')
  p=OUTPUT/row['result_file'];need(sha(p)==row['result_sha256'],'case_result_hash')
 return True
