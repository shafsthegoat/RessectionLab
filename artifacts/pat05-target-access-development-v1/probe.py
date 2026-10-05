"""One TRAIN-only reachability/profile probe; no gradients or comparison claim."""
from pathlib import Path
import cProfile, hashlib, io, json, os, pstats, resource, signal, sys, time, traceback
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from compare_real_spatial_search import load_member
from resectionlab.real_patient_learning import read_development_cohort
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
OUT=Path(__file__).parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(row): (OUT/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
def timeout(*_): raise TimeoutError('Bounded development probe exceeded60s')
def peak(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
row={'task':'annotation-assisted single initial target-access motion then STOP','patient':'sub-PAT05','role':'TRAIN','success_definition':'independently accepted committed history with positive supplied-target volume; not whole-target removal','full_resection_claim':False,'optimizer_updates':0,'other_patients_opened':0,'cap_seconds':60,'rss_check_cap_bytes':3*1024**3,'rss_enforcement':'cooperative phase-boundary peak checks','status':'started'}
profile=cProfile.Profile();started=time.perf_counter();signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
paths=['src/resectionlab/native_spatial_task.py','src/resectionlab/native_resection.py','src/resectionlab/native_proposals.py','src/resectionlab/spatial_observations.py','src/resectionlab/evaluation.py']
row['source_sha256']={p:sha(ROOT/p) for p in paths}
row['input_definition']='manifests/experiments/pat05-real-spatial-search-comparison-v1.json'
row['input_definition_sha256']=sha(ROOT/row['input_definition'])
try:
 d=json.loads((ROOT/row['input_definition']).read_text());cohort=read_development_cohort(ROOT/'manifests/experiments/btc-spatial-development-cohort-v1.json')
 row['bundle_sha256_before']=sha(ROOT/d['member']['case_bundle'])
 with NativePreviewProfiler(NativeResectionEngine) as native:
  profile.enable();t=time.perf_counter()
  with native.phase('source_and_initial_inventory'): base=load_member(d,cohort)
  row['preparation_seconds']=time.perf_counter()-t
  if peak()>row['rss_check_cap_bytes']: raise MemoryError('Development probe exceeded RSS cap')
  t=time.perf_counter();observed=base.observation();inventory=base.candidate_inventory()
  row['initial_inventory']={k:v for k,v in inventory.items() if k!='emitted'}
  row['initial_action_ids']=list(observed.action_ids)
  model=base.planning_clone()
  scores=[]
  for identity,preview in model._prepare_inventory().items():
   record=model._score_record({**preview.to_history_record(),'action_id':identity},None)
   scores.append({k:record[k] for k in ('action_id','reward','target_removed_mm3','normal_removed_mm3','insertion_distance_mm')})
  row['initial_nominal_scores']=scores
  best=max(scores,key=lambda x:x['reward']);row['selected']=best
  row['initial_observation_inventory_scoring_seconds']=time.perf_counter()-t
  if best['target_removed_mm3']<=0: raise RuntimeError('No initial target-access route; coverage not demonstrated')
  t=time.perf_counter()
  with native.phase('actual_target_access_then_stop'):
   first=base.step(best['action_id']);row['committed_history']=[first.info];save(row)
   final=base.step('STOP');row['committed_history'].append(final.info)
  row['actual_episode_seconds']=time.perf_counter()-t
  row['metrics']=base.metrics()
  t=time.perf_counter();row['independent_audit']=asdict(base.independent_geometry_check());row['independent_audit_seconds']=time.perf_counter()-t
  if not row['independent_audit']['feasible']: raise RuntimeError('Independent native replay failed')
  total=float(base.case.reference_target.sum(dtype='float64')*base._config.voxel_volume_mm3)
  row['full_annotation_volume_mm3']=total;row['removed_target_fraction']=row['metrics']['target_removed_mm3']/total
  row['status']='completed_target_access_proof'
  row['native_preview_profile']=native.snapshot()
  row['bundle_sha256_after']=sha(ROOT/d['member']['case_bundle'])
except Exception as error:
 row.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
finally:
 profile.disable();signal.alarm(0);row['wall_seconds']=time.perf_counter()-started;row['peak_rss_bytes']=peak()
 row['source_unchanged']={p:sha(ROOT/p)==v for p,v in row['source_sha256'].items()}
 profile.dump_stats(str(OUT/'profile.pstats'))
 text=io.StringIO();pstats.Stats(profile,stream=text).sort_stats('cumulative').print_stats(55);(OUT/'profile.txt').write_text(text.getvalue())
 save(row);print(json.dumps({k:row[k] for k in ('status','wall_seconds','peak_rss_bytes')},indent=2))
