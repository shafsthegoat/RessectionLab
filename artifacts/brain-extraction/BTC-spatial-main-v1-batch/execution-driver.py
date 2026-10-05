"""Fixed four-case main-model stage; never reads a tumor annotation or case array."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,sys,time,traceback
ROOT=Path(sys.argv[1]).resolve()
EVIDENCE=ROOT/'artifacts/brain-extraction/BTC-spatial-main-v1-batch'
RELEASE=EVIDENCE/'root-release.json'
def digest(path):
 with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
assert digest(RELEASE)=='3bc62f7a736f7d690df24754990b89075491fd7ae787404a264e559cb6b98934'
release=json.loads(RELEASE.read_text());snapshot=Path(release['snapshot'])
sys.path[:0]=[str(snapshot/'src')];sys.path.append(release['main_site_packages'])
from resectionlab import brain_extraction as extraction
import resectionlab.core,resectionlab.imaging
assert Path(extraction.__file__).resolve()==snapshot/'src/resectionlab/brain_extraction.py'
declaration=snapshot/'manifests/experiments/brain-extraction-btc-spatial-main-v1.json'
assert digest(declaration)==release['declaration_sha256']
d=json.loads(declaration.read_text());record_path=EVIDENCE/'inference-batch.json'
assert not record_path.exists(),'Existing run must be inspected, never repeated'
def utc():return datetime.now(timezone.utc).isoformat()
def save():
 temporary=record_path.with_suffix('.tmp')
 with temporary.open('w') as handle:
  json.dump(state,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
 temporary.replace(record_path)
state={'schema_version':1,'status':'running','started_at':utc(),'git_commit':release['git_commit'],'declaration_sha256':digest(declaration),'root_release_sha256':digest(RELEASE),'driver_path':str(Path(__file__).resolve()),'driver_sha256':digest(__file__),'attempts':[],'failures':[],'target_annotation_opened':False,'source_case_arrays_opened':False,'target_overlap_QC_performed':False,'model_training_performed':False,'brain_reviewed':False,'cortical_access_permitted':False,'working_brain_mask_attached':False,'project_module_origins':{name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items() if name.startswith('resectionlab') and getattr(module,'__file__',None)}}
assert all(Path(p).is_relative_to(snapshot/'src') for p in state['project_module_origins'].values())
save();started=time.perf_counter()
try:
 for item in d['subjects']:
  assert digest(ROOT/item['source_t1'])==item['source_t1_sha256']
  assert digest(ROOT/item['source_case'])==item['source_case_sha256']
  output=ROOT/item['output_directory'];assert not output.exists(),'Preserve prior artifacts'
  attempt={'subject':item['subject'],'development_role':item['development_role'],'patient_group':item['patient_group'],'status':'running','started_at':utc(),'output_directory':str(output),'source_t1_sha256':item['source_t1_sha256']}
  state['attempts'].append(attempt);save();print(json.dumps({'started':item['subject'],'at':utc()}),flush=True)
  child=extraction.run_synthstrip(ROOT/item['source_t1'],ROOT/d['model_cache'],output,model='main',allow_download=False,timeout_seconds=300,maximum_rss_bytes=6442450944,inference_python=Path(release['inference_python']),device='mps')
  assert child['executed_runner_sha256']==d['expected_executed_MPS_runner_sha256']
  for key in ('device','cpu_threads','border_mm','timeout_seconds','maximum_rss_bytes'):assert child['configuration'][key]==d['frozen_configuration'][key]
  for key,value in child['runtime_versions'].items():assert value==d['inference_runtime'][key]
  for name,info in d['model']['files'].items():assert child['model']['files'][name]['sha256']==info['sha256']
  # Basic source/grid validation only. Tumor overlap is forbidden until every
  # declared inference is complete and its output identities are frozen.
  report={'schema_version':1,'created_at':utc(),'implementation_sha256':digest(extraction.__file__),'source_t1_sha256':item['source_t1_sha256'],'source_annotation':None,'baseline':None,'variants':{'main':{'inference':child,'qc':{'status':'post_inference_QC_pending','brain_reviewed':False,'cortex_localized':False,'cortical_access_permitted':False}}},'clinical_use_status':'research_only','brain_reviewed':False,'cortical_access_permitted':False,'training_overlap_audit':'Exact released-checkpoint patient membership unverified; no disjointness claim.','declaration_sha256':digest(declaration),'development_role':item['development_role'],'variant_policy':'main fixed before all four new inferences; no target or reward based selection','all_four_outputs_before_annotation_overlap':True}
  report_path=output/'brain_extraction_report.json';report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  (output/'implementation_snapshot.py').write_bytes(Path(extraction.__file__).read_bytes())
  artifacts={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
  attempt.update(status='completed',finished_at=utc(),child_elapsed_seconds=child['elapsed_seconds'],sampled_child_RSS_bytes=child['sampled_process_peak_rss_bytes'],mps_memory_samples=child.get('mps_memory_samples'),artifact_files=artifacts)
  save();print(json.dumps({'finished':item['subject'],'child_seconds':child['elapsed_seconds'],'mask_sha256':child['artifact_hashes']['main_mask.nii.gz']}),flush=True)
 state['status']='all_four_inferences_completed_before_overlap_QC'
except BaseException as error:
 state['status']='failed';state['failures'].append({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
 if state['attempts'] and state['attempts'][-1]['status']=='running':state['attempts'][-1]['status']='failed_or_interrupted'
 attempted={a['subject'] for a in state['attempts']};state['not_run_after_failure']=[s for s in d['allowed_subjects'] if s not in attempted]
finally:
 state['finished_at']=utc();state['elapsed_seconds']=time.perf_counter()-started
 state['source_files_unchanged']=all(digest(ROOT/p)==h for p,h in release['source_files_sha256_before'].items())
 state['snapshot_files_unchanged']=all(digest(snapshot/p)==h for p,h in release['snapshot_files_sha256'].items())
 state['unopened_PAT29_PAT31_absent']=all(not (ROOT/'data/diffusion_source/ds001226-v5.0.1'/s).exists() for s in d['forbidden_subjects'])
 if not state['source_files_unchanged'] or not state['snapshot_files_unchanged'] or not state['unopened_PAT29_PAT31_absent']:state['status']='failed'
 save()
print(json.dumps({'status':state['status'],'elapsed_seconds':state['elapsed_seconds'],'receipt_sha256':digest(record_path)}),flush=True)
raise SystemExit(0 if state['status']=='all_four_inferences_completed_before_overlap_QC' else 1)
