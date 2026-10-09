"""Saved-metadata/header audit only: never reopen or decode original image payloads."""
from pathlib import Path
import ast,base64,collections,datetime,hashlib,importlib.metadata,itertools,json,math,os,platform,socket,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'data/annotations/resect-seg-v1/deferred-qc-v1/runs/remaining-train-pairs-02'
EXEC=ROOT/'build/resect-deferred-qc-execution-v2'
EXPECTED_COMMIT='446ea6dccf5a65a41c7f8f689586113a42290a95'
MANIFEST_SHA='9fd3ce43aa85805354091cc11bcf74705342726eafa45f453b8f3251afcafb83'
RUNNER_SHA='55328556f4719a9509effb10b4ef899a0dc1665dfa4ae61ee4c68bdd3c360fd8'
bindings={}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def data(p,expected=None):
 p=Path(p); assert p.is_file() and not p.is_symlink()
 assert not str(p).endswith(('.nii','.nii.gz','.mha','.mhd','.nrrd')),'Payload path prohibited'
 raw=p.read_bytes();h=sha(raw)
 if expected is not None:assert h==expected,(str(p),h,expected)
 bindings[str(p.relative_to(ROOT))]={'sha256':h,'bytes':len(raw)}
 return raw
def read(p,expected=None):return json.loads(data(p,expected))
def same(a,b):assert a==b,(a,b)
def forbid(*a,**kw):raise AssertionError('Scientific payload or network operation prohibited')
def hdr(prefix):
 raw=base64.b64decode(prefix['header_base64'],validate=True)
 assert len(raw)==348 and sha(raw)==prefix['header_sha256']
 return raw

def run():
 started=time.perf_counter()
 launch=read(EXEC/'launch.json');terminal=read(EXEC/'terminal.json')
 same(launch['source_commit'],EXPECTED_COMMIT);same(terminal['exit_code'],0)
 declaration=read(RUN/'declaration.json');batch=read(RUN/'batch.json')
 outer=json.loads(data(EXEC/'run.log',terminal['log_sha256']));same(outer,batch)
 assert 0<batch['elapsed_seconds']<=terminal['elapsed_seconds']<600
 same(declaration['execution']['files']['scripts/resect_deferred_qc.py'],RUNNER_SHA)
 same(declaration['execution']['files']['manifests/resect-deferred-qc-v1.json'],MANIFEST_SHA)
 for name,h in declaration['execution']['files'].items():
  saved=data(RUN/'source-snapshot'/name,h);same(data(ROOT/name,h),saved)
  committed=subprocess.run(['git','show',EXPECTED_COMMIT+':'+name],cwd=ROOT,capture_output=True,check=True).stdout
  same(sha(committed),h)
 same({str(p.relative_to(RUN/'source-snapshot')) for p in (RUN/'source-snapshot').rglob('*') if p.is_file()},set(declaration['execution']['files']))
 for name,h in launch['files_sha256'].items():data(ROOT/name,h)
 sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
 import resect_deferred_qc as q
 import nibabel as nib
 import numpy as np
 import resectionlab.imaging as imaging
 from resectionlab.critical_evidence import nifti1_header_record
 from nibabel.spatialimages import HeaderDataError
 socket.create_connection=forbid;q.verify_pair_files=forbid;q.inspect_pair=forbid;q.worker=forbid;q.batch=forbid
 q.gzip.open=forbid;nib.load=forbid;imaging.inspect_nifti=forbid
 m=q.preflight();same(q.execution_source(m),declaration['execution'])
 q.validate_execution(RUN,declaration,m,None);q.validate_inputs(RUN,m,None)
 proofs=q.input_proofs(m)
 same({p.name for p in (RUN/'input-snapshot').iterdir() if p.is_file()},{n+'.bin' for n in proofs})
 for name,p in proofs.items():
  raw=data(RUN/'input-snapshot'/(name+'.bin'),p['sha256']);same(len(raw),p['bytes'])
  same(data(ROOT/p['path'],p['sha256']),raw)
 same(batch['status'],'bounded_reviews_finished');same(declaration['max_seconds'],600.0)
 same(declaration['workers'],1);same(declaration['automatic_retries'],0)
 same(declaration['selected_pairs'],[r['pair']['id'] for r in m['pairs']])
 for x in (declaration,batch,batch['outcomes'][0]):assert q.claims_match(x)
 for k in ('denominator','missing_annotations','excluded_members','rights','historical_status_receipts'):same(batch[k],m[k])
 same(len(batch['outcomes']),25);same(batch['outcomes'][0],q.initial_outcomes(m)[0])
 same(dict(collections.Counter(x['status'] for x in batch['outcomes'])),batch['outcome_counts'])
 same(batch['outcome_counts'],{'inherited_structural_pass':1,'review_failed':24})
 same(len({r['pair']['patient_group'] for r in m['pairs']}|{m['inherited_pair']['patient_group']}),13)
 prior=read(ROOT/'build/resect-deferred-qc-continuity-fix-v1/review.json','c1efc55722f13dab0be3c847ad93f25f6351043bf73131a4210ccd56ba97daf2')
 for x in prior['preserved_actual_negative']:same(len(data(ROOT/x['path'],x['sha256'])),x['bytes'])
 rows=[];pids=[];deadlines=[];supervisors=set()
 for source,outcome in zip(m['pairs'],batch['outcomes'][1:]):
  pair=source['pair']['id'];same(outcome['pair_id'],pair)
  trial=RUN/'pairs'/pair;same(outcome['attempt'],str(trial.relative_to(ROOT)))
  intent=read(trial/'intent.json',outcome['intent_sha256']);start=read(trial/'started.json')
  r=read(trial/'receipt.json',outcome['receipt_sha256']);same(read(trial/'outcome.json'),outcome)
  same(json.loads(data(trial/'worker.log')),r)
  q.receipt_contract(r,source,intent,declaration)
  same(intent['declaration_sha256'],sha(q.encode(declaration)));same(intent['pair_id'],pair)
  same(start,{'pid':outcome['worker_pid'],'intent_sha256':outcome['intent_sha256']})
  same(outcome['supervision_status'],'completed');same(outcome['exit_code'],0)
  same(outcome['status'],r['status']);same(r['status'],'review_failed')
  same(outcome['review_status'],'review_failed');same(r['fixity_before'],'passed');same(r['fixity_after'],'passed')
  assert q.claims_match(r) and r['sources']==source and source['pair']['role']=='TRAIN'
  same(r['image_qc']['stage'],'raw_header_recording');same(r['image_qc']['error_type'],'HeaderDataError')
  same(r['image_qc']['reason'],'qfac (pixdim[0]) should be 1 or -1')
  same(r['image_qc']['classification'],'adapter_raw_header_recording_limitation')
  same(r['image_qc']['source_anatomical_validity'],'not_assessed')
  same(r['image_qc']['call_path'],['inspect_pair','inspect_original','nifti1_header_record','_header_summary','get_qform'])
  for name in ('header_qc','scalar_qc','geometry_qc'):same(r['image_qc'][name]['status'],'unknown_not_returned')
  same(r['pair_geometry'],{'status':'not_run'})
  elapsed=r['elapsed_seconds'];rss=r['peak_rss_bytes']
  assert math.isfinite(elapsed) and 0<elapsed<60 and type(rss) is int and rss>0
  same(r['rss_is_observed_not_hard_limit'],True);same(r['numerical_library_threads'],1)
  pids.append(start['pid']);supervisors.add(intent['supervisor_pid']);deadlines.append(intent['deadline'])
  # Recompute image-header-only framing/geometry; never assign scalar/content pass.
  iraw=hdr(r['image_prefix']);ih=nib.Nifti1Header(binaryblock=iraw,check=False)
  same(int(ih['sizeof_hdr']),348);same(bytes(ih['magic']),b'n+1\0')
  same(int(ih['qform_code']),0);same(float(ih['pixdim'][0]),0.);same(int(ih['sform_code']),1)
  same(ih.get_qform(coded=True),(None,0))
  try:ih.get_qform()
  except HeaderDataError as e:same(str(e),'qfac (pixdim[0]) should be 1 or -1')
  else:raise AssertionError('Expected original inactive qform interpretation failure')
  shape=[int(n) for n in ih.get_data_shape()];dtype=ih.get_data_dtype()
  assert dtype.kind in 'iuf' and int(ih['bitpix'])==dtype.itemsize*8
  image_budget=q.streaming.scalar_budget(shape,dtype.itemsize,m['bounds'])
  image_geometry=q.geometry_from_header(iraw,Path(source['image']['path']).name)
  raw_s,sc=ih.get_sform(coded=True)
  same(image_geometry['shape'],shape);same(sc,1)
  assert np.array_equal(np.array(image_geometry['affine_ras_mm']),raw_s) # all saved units are mm
  same(image_geometry['source_units'],'mm')
  # Mask: bind raw header, independently check all count arithmetic and declared budget.
  mask=r['mask_qc'];same(mask['status'],'passed');same(r['mask_geometry']['status'],'passed')
  mraw=hdr(r['mask_prefix']);same(mask['raw_grid'],nifti1_header_record(mraw,source['mask_sha256']))
  mh=nib.Nifti1Header(binaryblock=mraw,check=False);ms=[int(n) for n in mh.get_data_shape()];md=mh.get_data_dtype()
  same(mask['dtype'],md.str);slope,intercept=mh.get_slope_inter()
  same(mask['source_scaling'],[1.,0.] if slope is None else [float(slope),float(intercept)])
  same(mask['decoding_budget'],q.streaming.annotations.decoding_budget(ms,md.itemsize,m['bounds']))
  q.reported_header_contract(mraw,mask,r['mask_geometry']['header'],source,m['bounds'],'mask')
  vox=math.prod(ms);positive=mask['positive_voxels'];background=mask['background_voxels']
  assert type(positive) is int and type(background) is int and positive>0 and background>0
  same(positive+background,vox);same(mask['background_semantics'],'unknown')
  same(mask['extensions'],q.streaming.annotations.inspect_extensions(b'\0'*4,data_offset=352,endian=mh.endianness,maximum_offset=m['bounds']['max_nifti_data_offset']))
  same(float(mh['vox_offset']),352.)
  # Supplementary pair relation from saved headers only, independently enumerate corners.
  same(shape,ms)
  aa=np.asarray(image_geometry['affine_ras_mm']);bb=np.asarray(r['mask_geometry']['header']['affine_ras_mm'])
  corners=[np.array([*p,1.],dtype=float) for p in itertools.product(*[(0,n-1) for n in shape])]
  corner=max(float(np.linalg.norm((aa@p-bb@p)[:3])) for p in corners)
  pair_geometry=q.pair_geometry(image_geometry,r['mask_geometry']['header'])
  assert abs(corner-pair_geometry['maximum_corner_difference_mm'])<1e-12
  rows.append({'pair_id':pair,'patient_group':source['pair']['patient_group'],'status':r['status'],
    'image_content_review':'not_completed','image_header_only_review':'passed_unchanged_active_sform_rules',
    'image_raw_header_sha256':sha(iraw),'image_dtype':dtype.str,'image_shape':shape,
    'image_budget_from_saved_header':image_budget,'mask_raw_header_sha256':sha(mraw),
    'mask_positive_voxels_reported':positive,'mask_background_voxels_reported':background,
    'mask_count_sum_and_budget_recomputed':True,'mask_geometry_recomputed':True,
    'supplementary_header_pair_relation':pair_geometry,'independent_corner_difference_mm':corner,
    'pair_content_pass_or_admission':False,'elapsed_seconds':elapsed,'peak_rss_bytes_observed':rss,
    'fixity_before_and_after_reported':'passed','receipt_sha256':outcome['receipt_sha256']})
 same(len(set(pids)),24);same(len(supervisors),1);assert all(a<b for a,b in zip(deadlines,deadlines[1:]))
 assert deadlines[-1]-deadlines[0]<batch['elapsed_seconds']
 cleanup=[]
 for pid in pids:
  try:os.kill(pid,0)
  except ProcessLookupError:cleanup.append({'pid':pid,'present_at_audit':False})
  else:cleanup.append({'pid':pid,'present_at_audit':True,'interpretation':'pid existence alone cannot identify old worker'})
 for name,proof in list(bindings.items()):same(sha((ROOT/name).read_bytes()),proof['sha256'])
 result={'schema':'resect-deferred-qc-saved-output-audit-v1','status':'passed_saved_output_audit_of_completed_negative_QC',
  'source_commit':EXPECTED_COMMIT,'run_id':RUN.name,'source_files_verified':len(declaration['execution']['files']),
  'input_metadata_snapshots_verified':len(proofs),'binding_count':len(bindings),'denominator':m['denominator'],
  'outcome_counts':batch['outcome_counts'],'missing_annotations':m['missing_annotations'],
  'all_24_mask_binary_content_checks_reported_passed':True,
  'all_24_mask_count_arithmetic_budget_raw_header_geometry_rechecked':True,
  'all_24_image_header_geometry_from_saved_prefix_passed':True,
  'all_24_image_scalar_content_checks_unfinished':True,
  'supplementary_pair_header_status_counts':dict(collections.Counter(x['supplementary_header_pair_relation']['status'] for x in rows)),
  'maximum_supplementary_corner_difference_mm':max(x['independent_corner_difference_mm'] for x in rows),
  'mask_foreground_voxels_sum_across_24_correlated_timepoints':sum(x['mask_positive_voxels_reported'] for x in rows),
  'mask_voxels_total_across_24_correlated_timepoints':sum(x['mask_positive_voxels_reported']+x['mask_background_voxels_reported'] for x in rows),
  'resources':{'batch_elapsed_seconds':batch['elapsed_seconds'],'outer_elapsed_seconds':terminal['elapsed_seconds'],
    'worker_reported_seconds_sum':sum(x['elapsed_seconds'] for x in rows),
    'worker_reported_seconds_min':min(x['elapsed_seconds'] for x in rows),
    'worker_reported_seconds_max':max(x['elapsed_seconds'] for x in rows),
    'peak_single_worker_rss_bytes_observed':max(x['peak_rss_bytes_observed'] for x in rows),
    'rss_scope':'single worker observed high-water, not hard cap or aggregate process RSS',
    'max_mask_explicit_chunk_workspace_bytes':max(read(RUN/'pairs'/x['pair_id']/'receipt.json')['mask_qc']['decoding_budget']['chunk_working_bytes_bound'] for x in rows),
    'audit_elapsed_seconds':time.perf_counter()-started},
  'cleanup_pid_observations':cleanup,'prior_attempt01_metadata_hashes_unchanged':True,
  'claims':q.CLAIMS,'rows':rows,'bindings':bindings,
  'limitations':['Recomputed geometry uses only the already saved348-byte headers. These supplementary results do not change authoritative review_failed outcomes or authorize training/planning.',
   'Binary counts are authenticated recorded worker outputs with independent arithmetic, not independently recounted voxels; no original payload is reopened.',
   'All24 original-image scalar reviews remain unfinished; valid active sform does not establish source intensities, anatomy or label accuracy.',
   'Mask foreground is visible cavity; correlated during/after counts are not total removed tissue or independent patient outcomes.',
   'Invalid inactive qform recording is an adapter limitation, not a blanket image/anatomy rejection.'],
  'original_payload_reads':0,'image_or_mask_array_decodes':0,'new_QC_execution':0,'network_requests':0}
 (OUT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ['status','binding_count','outcome_counts','supplementary_pair_header_status_counts','maximum_supplementary_corner_difference_mm','resources']}))
if __name__=='__main__':
 try:run()
 except BaseException as error:
  (OUT/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},indent=2)+'\n')
  raise
