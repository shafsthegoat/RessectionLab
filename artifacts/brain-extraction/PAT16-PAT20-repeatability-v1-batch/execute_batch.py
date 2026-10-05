"""Execute one frozen local extraction declaration, preserving all attempts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,platform,shutil,subprocess,sys,time
import numpy as np
import nibabel as nib

root=Path.cwd();batch=Path(__file__).resolve().parent
manifest_path=root/'manifests/experiments/brain-extraction-pat16-pat20-repeatability-v1.json'
EXPECTED='5be3ade2eb6b383e796cdc34def48dae66a4fb68649b282723daa83e771477db'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
assert digest(manifest_path)==EXPECTED
m=json.loads(manifest_path.read_text())
shutil.copyfile(manifest_path,batch/'frozen_declaration.json')
shutil.copyfile(root/m['wrapper']['path'],batch/'brain_extraction_snapshot.py')
state={'schema_version':1,'declaration_sha256':EXPECTED,'started_at':utc(),'finished_at':None,'status':'running','attempts':[],
 'executor_sha256':digest(__file__),'wrapper_sha256':digest(batch/'brain_extraction_snapshot.py'),
 'host':{'system':platform.platform(),'architecture':platform.machine()},'working_anatomy_changed':False,
 'brain_reviewed':False,'cortical_access_permitted':False,'clinical_deficit_probability':None}
def save():
 (batch/'batch_record.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
def preflight():
 assert digest(manifest_path)==EXPECTED
 for item in [m['wrapper'],*m['model_assets'].values()]:assert digest(root/item['path'])==item['sha256'],item['path']
 for subject in m['subjects']:
  for key,expected in [('source_manifest','source_manifest_sha256'),('independent_source_qc','independent_source_qc_sha256'),('source_t1','source_t1_sha256'),('source_annotation','source_annotation_sha256')]:
   assert digest(root/subject[key])==subject[expected],subject[key]
  qc=json.loads((root/subject['independent_source_qc']).read_text())
  assert digest(root/subject['prepared_case'])==qc['bundle_sha256'],'Prepared source case changed'
 command=[str(root/m['inference_runtime']['interpreter']),'-c',"import importlib.metadata,json,sys; print(json.dumps({'python':sys.version.split()[0], **{name:importlib.metadata.version(name) for name in ('numpy','torch','surfa','scipy','nibabel')}}))"]
 observed=json.loads(subprocess.run(command,check=True,capture_output=True,text=True,timeout=30).stdout)
 for key,value in observed.items():assert m['inference_runtime'][key]==value,(key,value)
 return observed
save()
try:
 for declared in m['runs']:
  observed=preflight()
  out=root/declared['output_directory'];assert not out.exists(),str(out)
  attempt={'run_id':declared['run_id'],'subject':declared['subject'],'repetition':declared['repetition'],
   'output_directory':declared['output_directory'],'argv':declared['argv'],'runtime_metadata_preflight':observed,
   'started_at':utc(),'finished_at':None,'status':'running'}
  state['attempts'].append(attempt);save();print(json.dumps({'started':declared['run_id'],'at':attempt['started_at']}),flush=True)
  began=time.monotonic()
  with (batch/(declared['run_id']+'.log')).open('w') as log:
   process=subprocess.Popen(declared['argv'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
   try:exit_code=process.wait()
   except BaseException:
    process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    raise
  attempt.update({'exit_code':exit_code,'finished_at':utc(),'wall_seconds_including_parent_preparation':time.monotonic()-began,
                  'log_sha256':digest(batch/(declared['run_id']+'.log'))})
  if out.exists():shutil.copyfile(batch/'brain_extraction_snapshot.py',out/'implementation_snapshot.py')
  if exit_code:
   attempt['status']='failed';attempt['retained_child_records']={}
   for variant in declared['variant_order']:
    p=out/(variant+'_inference_record.json')
    if p.exists():attempt['retained_child_records'][variant]=json.loads(p.read_text())
   raise RuntimeError('Declared extraction CLI failed; no retries or further runs: '+declared['run_id'])
  report_path=out/'brain_extraction_report.json';report=json.loads(report_path.read_text())
  assert report['implementation_sha256']==m['wrapper']['sha256']
  assert report['brain_reviewed'] is False and report['cortical_access_permitted'] is False
  assert list(report['variants'])==declared['variant_order']
  children={}
  for variant,item in report['variants'].items():
   record=item['inference'];assert record['failure'] is None and record['exit_code']==0
   assert record['executed_runner_sha256']==m['expected_executed_MPS_runner_sha256']
   for key in ('device','cpu_threads','border_mm','timeout_seconds','maximum_rss_bytes'):
    assert record['configuration'][key]==m['frozen_configuration'][key],(variant,key)
   for key,value in record['runtime_versions'].items():assert value==m['inference_runtime'][key]
   for filename,expected in record['artifact_hashes'].items():assert digest(out/filename)==expected
   children[variant]={'elapsed_seconds':record['elapsed_seconds'],'sampled_process_peak_rss_bytes':record['sampled_process_peak_rss_bytes'],
    'mps_memory_samples':record.get('mps_memory_samples'),'memory_scope':record.get('mps_memory_measurement'),
    'artifact_hashes':record['artifact_hashes'],'source_annotation_outside_voxels':item['qc']['source_annotation_outside_voxels'],
    'estimated_envelope_volume_ml':item['qc']['mask_volume_ml'],'failure':None,'review_status':'review_required'}
  attempt.update({'status':'completed','report_sha256':digest(report_path),'children':children})
  save();print(json.dumps({'finished':declared['run_id'],'wall_seconds':attempt['wall_seconds_including_parent_preparation'],
    'omitted_source_annotation_voxels':{k:v['source_annotation_outside_voxels'] for k,v in children.items()}}),flush=True)
 state['repeat_comparisons']={}
 for subject in m['subjects']:
  comparison={}
  for variant in ('nocsf','main'):
   first=root/subject['output_root']/'r1';second=root/subject['output_root']/'r2'
   a=nib.load(first/(variant+'_mask.nii.gz')).get_fdata(dtype=np.float32)
   b=nib.load(second/(variant+'_mask.nii.gz')).get_fdata(dtype=np.float32)
   mismatch=int(np.count_nonzero(a!=b));del a,b
   a=nib.load(first/(variant+'_distance_mm.nii.gz')).get_fdata(dtype=np.float32)
   b=nib.load(second/(variant+'_distance_mm.nii.gz')).get_fdata(dtype=np.float32)
   comparison[variant]={'mask_differing_voxels':mismatch,'maximum_absolute_predicted_distance_difference_mm':float(np.max(np.abs(a-b))),
    'predicted_distance_arrays_identical':bool(np.array_equal(a,b)),'interpretation':'Within-patient fixed-runtime repeatability, not anatomical accuracy.'}
   del a,b
  state['repeat_comparisons'][subject['subject']]=comparison
 preflight()
 state['status']='completed';state['finished_at']=utc();save()
 print(json.dumps({'batch_status':'completed','repeat_comparisons':state['repeat_comparisons']}),flush=True)
except BaseException as error:
 state['status']='failed';state['failure']=f'{type(error).__name__}: {error}';state['finished_at']=utc()
 attempted={item['run_id'] for item in state['attempts']}
 state['not_run_after_failure']=[item['run_id'] for item in m['runs'] if item['run_id'] not in attempted]
 if state['attempts'] and state['attempts'][-1]['status']=='running':state['attempts'][-1]['status']='failed_or_interrupted'
 save();print(json.dumps({'batch_status':'failed','failure':state['failure']}),flush=True);raise
