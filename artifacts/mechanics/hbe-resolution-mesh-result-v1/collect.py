"""Saved-only collection; no imports of meshing, solver or response readers."""
from pathlib import Path
import hashlib,json,stat,time
root=Path(__file__).resolve().parents[3]
raw=root/'outputs/mechanics/hbe-01-03-resolution-v1/mesh-preparation'
out=Path(__file__).resolve().parent
start=time.monotonic();cache={}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def sha(p):
 p=Path(p).resolve()
 if p not in cache:cache[p]=digest(p)
 return cache[p]
def rel(p):return str(Path(p).resolve().relative_to(root))
def binding(p):return {'path':rel(p),'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def verify(row):
 p=root/row['path'];assert sha(p)==row['sha256'],p
 return p
def save(name,value):
 p=out/name
 with p.open('x') as f:f.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
 return binding(p)
result=load(raw/'result.json');state=load(raw/'state.json');base=load(raw/'baseline.json')
assert result['status']=='prepared_not_solved' and result['phase']=='prepare'
assert state['status']=='prepared_not_solved' and state['gmsh_generation_calls']==2 and state['mesh_preparation_invocations']==2
assert state['solver_invocations']==0 and state['measured_data_accessed'] is False
assert all(row['status']=='not_executed' for row in state['runs'].values()) and len(state['runs'])==4
for name in ('baseline','state'):verify(result[name])
supervision=load(raw/'supervision/supervision.json');assert result['supervision']==supervision
assert supervision['status']=='completed' and supervision['exit_code']==0 and supervision['elapsed_seconds']<120
assert supervision['kill_reason'] is None and supervision['cleanup_error'] is None
verified={}
for name,expected in base['inputs'].items():
 p=Path(name);assert sha(p)==expected,p
 verified[rel(p) if p.is_relative_to(root) else str(p)]={'sha256':expected,'unchanged':True}
release=load(verify(base['release_binding']));verify(release['source_archive'])
levels={}
for N,record_binding in state['levels'].items():
 p=verify(record_binding);v=load(p)
 assert v['status']=='prepared_not_solved' and v['quality']['passed'] and all(v['quality']['checks'].values())
 assert v['gmsh_generation_calls']==1 and v['solver_calls']==0 and v['curve_values_opened'] is False and v['runtime_unchanged'] is True
 assert v['source_sha256']==base['source_bindings']['mesh_deck']['sha256'] and v['resolution_declaration_sha256']==base['study_binding']['sha256']
 assert sha(p.parent/'mesh.json')==v['mesh_sha256'] and sha(p.parent/'specimen.msh')==v['native_mesh_sha256']
 for name,d in v['decks'].items():
  assert sha(p.parent/name/'specimen.feb')==d['deck_sha256'] and sha(p.parent/name/'loading.json')==d['loading_sha256']
 levels[N]={'receipt':record_binding,'quality':v['quality'],'elapsed_seconds':v['elapsed_seconds'],'generation_seconds':v['generation_seconds'],'mesh_sha256':v['mesh_sha256'],'native_mesh_sha256':v['native_mesh_sha256']}
for records in state['cases'].values():
 for rec in records.values():verify(rec)
old_index_path=root/'artifacts/mechanics/hbe-accelerate-experiment-result-v1/raw-output-index.json';old=load(old_index_path)
for name,rec in old['files'].items():
 p=root/old['root']/name;assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],p
assert len([p for p in (root/old['root']).rglob('*') if p.is_file()])==old['file_count']
files={}
for p in sorted(raw.rglob('*')):
 assert not p.is_symlink(),p
 if p.is_file():files[str(p.relative_to(raw))]={'bytes':p.stat().st_size,'sha256':sha(p)}
raw_index=save('raw-output-index.json',{'root':rel(raw),'file_count':len(files),'bytes':sum(v['bytes'] for v in files.values()),'files':files})
preservation=save('preservation.json',{'baseline':result['baseline'],'baseline_input_count':len(verified),'all_baseline_inputs_unchanged':True,'baseline_inputs':verified,
 'historical_failed_experiment':{'index':binding(old_index_path),'file_count':old['file_count'],'bytes':old['bytes'],'all_hashes_and_sizes_unchanged':True,'inventory_unchanged':True},
 'source_archive':release['source_archive'],'source_commit':release['source_commit'],'source_bindings':base['source_bindings'],
 'level_and_case_bindings_match_saved_state':True,'measured_archive_members_opened':0})
quality=save('mesh-quality-summary.json',{'method':'Copy of hash-verified actual generation quality receipts; no geometry recomputation or native execution by collector.','levels':levels,'cases':state['cases'],'solver_runs':state['runs'],'solver_calls':0})
parent_mode=stat.S_IMODE(raw.parent.stat().st_mode)
for p in raw.rglob('*'):
 if p.is_file():p.chmod(0o444)
for p in sorted([p for p in raw.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
raw.chmod(0o555)
assert stat.S_IMODE(raw.parent.stat().st_mode)==parent_mode and parent_mode&0o200
for name,row in files.items():
 p=raw/name
 assert digest(p)==row['sha256'] and stat.S_IMODE(p.stat().st_mode)==0o444
assert all(stat.S_IMODE(p.stat().st_mode)==0o555 for p in [raw]+[x for x in raw.rglob('*') if x.is_dir()])
freeze=save('readonly-freeze.json',{'raw_index':raw_index,'file_count':len(files),'file_mode':'0444','directory_mode':'0555','phase_directory_only':rel(raw),'study_root_mode_preserved':oct(parent_mode),'study_root_remains_owner_writable':True,'all_raw_bytes_rehashed_after_permission_freeze':True,'raw_content_edited':False,'compression_or_deletion':False})
watch=load(raw/'output-watch.json')
summary={'schema':'hbe-resolution-mesh-outcome-v1','status':'preparation_receipts_and_input_integrity_verified','actual_phase_status':'prepared_not_solved',
 'result':binding(raw/'result.json'),'state':result['state'],'baseline':result['baseline'],'release':base['release_binding'],'source_commit':release['source_commit'],'source_archive':release['source_archive'],
 'study':base['study_binding'],'runtime_identity':base['runtime_identity'],'backend_profile':base['backend_profile'],'gmsh_runtime':base['gmsh_runtime'],
 'gmsh_generation_calls':2,'solver_calls':0,'measured_data_accessed':False,'four_solver_cases_status':'not_executed','co_primary_convergence_evaluated':False,'material_fitted':False,'physical_validation_pass':None,
 'supervised_elapsed_seconds':supervision['elapsed_seconds'],'worker_body_elapsed_seconds':state['elapsed_seconds'],'sampled_peak_process_group_rss_bytes':supervision['sampled_peak_process_group_rss_bytes'],
 'wall_cap_seconds':supervision['wall_cap_seconds'],'rss_cap_bytes':supervision['rss_cap_bytes'],'output_watch':watch,'raw_index':raw_index,'preservation':preservation,'mesh_quality':quality,'readonly_freeze':freeze,
 'independent_saved_geometry_review':'separately_pending_at_collection','collector':binding(__file__),
 'collector_scope':'JSON, hash/size and permission verification only; no mesh generation, solver, primitive replay or measured member reads','collection_seconds':time.monotonic()-start}
save('outcome.json',summary)
print(json.dumps({'file_count':len(files),'bytes':sum(v['bytes'] for v in files.values()),'baseline_inputs':len(verified),'raw_index':raw_index,'outcome':binding(out/'outcome.json'),'collection_seconds':summary['collection_seconds']},indent=2))
