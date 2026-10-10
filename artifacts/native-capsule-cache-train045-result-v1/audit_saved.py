"""Bounded independent saved-JSON/source comparison; no scientific imports."""
import argparse,hashlib,json,math,os,resource,signal,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'build/native-capsule-cache-train045-pair-v1';OUT=RUN/'attempt-01'
HEAD='4cd1411fe68dc4125385cf0a842399ebe3e26c4b'
RELEASE='cc416924c1682d192e7671e3fad772925bca03a6d4cd14f72fdf334d1b5fa83b'
INDEX='20c1d6d4ac24e98efffaa9f3b3a22265ea7882da11f08bf5cf03ed0ca17e3491'
CACHE='7b47cf36493c4baeb305beb7f18d4f5acf5bd0fa08b0e80738e948a20bd58869'
seen={};checks=0
def need(ok,why):
 global checks
 checks+=1
 if not ok:raise ValueError(why)
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>256*1024**2:raise MemoryError('256MiB metadata audit')
def raw(path,pin=None):
 path=Path(path);path=path if path.is_absolute() else ROOT/path
 rel=path.relative_to(ROOT);need('..' not in rel.parts and path.suffix in ('.json','.py'),'metadata/source only')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  st=os.fstat(fd);need(stat.S_ISREG(st.st_mode) and 0<st.st_size<=8*1024**2,'bounded regular file')
  with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(8*1024**2+1)
 finally:os.close(fd)
 need(len(data)==st.st_size,'bounded stable length');sha=hashlib.sha256(data).hexdigest()
 need(pin is None or sha==pin,'exact hash '+str(rel));need(str(rel) not in seen or seen[str(rel)]['sha256']==sha,'unchanged input')
 seen[str(rel)]={'sha256':sha,'bytes':len(data)}
 need(sum(r['bytes'] for r in seen.values())<128*1024**2,'aggregate small evidence')
 return data
def read(path,pin=None):
 need(Path(path).suffix=='.json','JSON only')
 return json.loads(raw(path,pin),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def bound(ref):return read(ref['path'],ref['sha256'])
def digest(path):return seen[str(Path(path).relative_to(ROOT))]['sha256']
def semantic(v):return 'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def near(a,b,why):need(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),why)
def nontime(v):return {k:x for k,x in v.items() if k!='planning_seconds'}
def audit(receipt_sha,result_sha):
 for pin in (receipt_sha,result_sha):need(len(pin)==64 and all(c in '0123456789abcdef' for c in pin),'actual terminal pins required')
 parent=read(OUT.with_name('attempt-01.supervision')/'receipt.json',receipt_sha)
 need(parent['worker_termination_confirmed'] and not parent['remaining_owned_pids'] and not parent['cleanup_errors'],'terminal clean ownership before output reads')
 release=read(RUN/'root-release.json',RELEASE);index=bound(release['source_index']);result=read(OUT/'result.json',result_sha)
 need(release['source_index']['sha256']==INDEX and release['expected_head']==index['head']==HEAD,'exact current source/release')
 need(release['execution_released'] is True and release['arms']==['baseline','cached'] and release['subject']=='ReMIND-045','one fixed TRAIN pair')
 need(release['cache_scope']=='greedy_only_restore_before_replay_and_audit' and release['cache_limits']=={'max_payload_bytes':32*1024**2,'max_entries':16384},'fixed bounded coverage-only cache')
 need(index['files']=={**index['source_files'],**index['metadata_files']},'closure map consistency')
 need(index['files']['src/resectionlab/experimental_capsule_cache.py']==CACHE,'unchanged reviewed cache source')
 for path,pin in index['files'].items():raw(path,pin)
 need(parent['release_sha256']==result['release_sha256']==RELEASE and parent['source_index']==release['source_index'] and parent['result_sha256']==result_sha,'parent/result/release joins')
 need(parent['caps']==release['caps'] and parent['planned_subjects']==['ReMIND-045'] and parent['training_admitted'] is False and parent['automatic_retry'] is False,'declared scope')
 caps=release['caps'];need(caps=={'worker_seconds':120,'parent_seconds':150,'sampled_rss_bytes':3*1024**3,'output_bytes':32*1024**2,'supervision_bytes':4*1024**2,'log_bytes':2*1024**2,'native_previews':6000,'threads':1},'frozen resource limits')
 need(parent['elapsed_seconds']<150 and parent['sampled_peak_rss_bytes']<=3*1024**3 and parent['output_bytes']<=32*1024**2 and parent['samples']>0,'observed parent caps')
 originals={k:bound(v) for k,v in release['inputs'].items()}
 oldplan,oldmetrics,oldcert,case=(originals[k] for k in ('plan','metrics','audit','case'))
 need(oldcert['accepted'] and oldplan['accounting']['complete'] and len(oldplan['actions'])==13 and oldplan['actions'][-1]=='STOP','same fixed complete teacher')
 need(case['actions']==oldplan['actions']==[r['action_id'] for r in oldmetrics['history']] and case['full_route_outcomes']==oldcert['outcomes'],'historical result joins')
 if result['status']!='complete_paired_native_routes':
  need(parent['status']=='failed_or_unresolved' and parent['exit_code']!=0,'failed scientific attempt remains failed')
  return {'decision':'PASS_RETAINED_FAILED_PAIR','complete':False,'failure':result.get('failure'),'partial_arms':result['arms'],'parent':parent,'no_performance_claim':True}
 need(parent['status']=='complete' and parent['exit_code']==0 and parent['stop_reason'] is None,'complete parent')
 need(result['subject']=='ReMIND-045' and result['source_visits']==result['greedy_calls']==2 and result['committed_actions']==26 and result['live_sources_after_cleanup']==0,'two fresh complete sources')
 need(result['exact_pair_parity'] and result['native_previews']==4550 and result['elapsed_seconds']<120,'fixed work and worker cap')
 need(all(result[k]==0 for k in ('model_calls','optimizer_calls','checkpoint_loads','blocked_external_calls')),'zero external/model work')
 need([r['arm'] for r in result['arms']]==['baseline','cached'],'fixed ordering')
 summaries=[];world=None;metrics_seen=None;plan_seen=None
 for row in result['arms']:
  name=row['arm'];p=OUT/name
  need(read(p/'arm-result.json')==row,'exact terminal arm record')
  plan=read(p/'greedy-plan.json',row['files']['greedy-plan']);metrics=read(p/'episode-metrics.json',row['files']['episode-metrics']);cert=read(p/'independent-episode.json',row['files']['independent-episode'])
  need(row['status']=='complete' and row['committed_actions']==13 and row['native_previews']==2275 and row['exact_original_metrics'] and row['independent_accepted'] and row['cache_restored_before_replay'],'complete uncached replay per arm')
  need(plan['actions']==oldplan['actions'] and nontime(plan['accounting'])==nontime(oldplan['accounting']),'all actions and every non-time score/disposition exact')
  need(metrics==oldmetrics and (metrics_seen is None or metrics==metrics_seen),'full serialized metrics/history exact')
  need(plan_seen is None or nontime(plan['accounting'])==plan_seen,'pair planning-accounting exact')
  need(world is None or row['world']==world,'same complete source/context/model/observation/inventory')
  world=row['world'];metrics_seen=metrics;plan_seen=nontime(plan['accounting'])
  need(world['source_hash']==case['source_hash'] and world['context_hash']==case['context_hash'] and world['initial_observation_hash']==case['initial_observation_hash'],'original source/context/observation retained')
  need(metrics['source_hash']==cert['source_hash']==world['source_hash'] and metrics['decision_model_hash']==cert['decision_model_hash']==world['decision_model_hash'],'current evaluator source/model')
  need(cert['accepted'] and cert['complete_episode'] and cert['geometry']['feasible'] and not cert['geometry']['failures'] and cert['geometry']['complete_tool_checked'] and cert['geometry']['frontier_checked'],'current full-tool certificate')
  need(cert['committed_history_hash']==semantic(metrics['history']) and cert['outcomes']==oldcert['outcomes'],'current certificate full-history/outcome exact')
  need(metrics['terminated'] and metrics['steps']==13 and [h['action_id'] for h in metrics['history']]==plan['actions'],'complete route sequence')
  near(sum(h['reward'] for h in metrics['history']),metrics['total_reward'],'saved reward sum')
  near(sum(h.get('target_removed_mm3',0) for h in metrics['history']),cert['outcomes']['target_removed_mm3'],'saved target sum')
  near(sum(h.get('normal_removed_mm3',0) for h in metrics['history']),cert['outcomes']['normal_removed_mm3'],'saved outside sum')
  b=row['native_budget'];need(b['native_preview_entries']==2275 and b['counting_reliable'] and b['failure'] is None and b['blocked_preview_attempts']==0 and b['history_complete_caller_attestation'] is True,'per-arm uncapped accounting')
  for key in ('construction_seconds','cache_construction_seconds','greedy_wall_seconds','authoritative_replay_seconds','independent_audit_seconds','elapsed_seconds'):
   need(type(row[key]) in (float,int) and math.isfinite(row[key]) and row[key]>=0,'finite phase '+key)
  need(row['greedy_wall_seconds']>=plan['accounting']['planning_seconds'],'outer greedy timer includes own accounting')
  stats=row['cache_stats']
  if name=='baseline':need(stats is None and row['cache_construction_seconds']==0,'uncached baseline')
  else:
   need(stats['calls']>0 and stats['calls']==stats['hits']+stats['misses'] and stats['entries']==stats['misses']-stats['bypasses']-stats['evictions'],'cache hit/miss/retained entries arithmetic')
   need(stats['retained_payload_bytes']<=stats['max_payload_bytes']==32*1024**2 and stats['entries']<=stats['max_entries']==16384,'independent payload and entry caps')
   need(all(not stats[k] for k in ('feasibility_cached','cavity_cached','integrity_cached')) and row['cache_key_bookkeeping_bytes'] is None,'only cell-cover cached, bookkeeping not measured')
   need(stats['geometry_source_sha256']==index['files']['src/resectionlab/geometry.py'] and stats['source_guard_sha256']==index['files']['src/resectionlab/native_proposals.py'],'cache runtime source pins')
   need(stats['source_hash']==world['source_hash'],'cache source namespace')
  interval=parent['arm_interval_rss'][name];need(interval['samples']>0 and 0<interval['sampled_peak_rss_bytes']<=parent['sampled_peak_rss_bytes'],'whole-process arm interval samples')
  summaries.append({k:row[k] for k in ('arm','elapsed_seconds','construction_seconds','cache_construction_seconds','greedy_wall_seconds','authoritative_replay_seconds','independent_audit_seconds','native_previews','committed_actions','cache_stats','cache_restored_before_replay','process_cumulative_peak_rss_before_bytes','process_cumulative_peak_rss_after_bytes')}|{'parent_arm_interval_rss':interval,'outputs':row['files'],'outcomes':cert['outcomes']})
 one,two=summaries;ratio=two['greedy_wall_seconds']/one['greedy_wall_seconds'];difference=two['greedy_wall_seconds']-one['greedy_wall_seconds']
 near(ratio,result['greedy_wall_ratio_cached_over_baseline'],'reported paired greedy ratio');near(difference,result['greedy_seconds_difference_cached_minus_baseline'],'reported paired greedy difference')
 return {'decision':'PASS_SAVED_EXACT_CACHE_PAIR','complete':True,'subject':'ReMIND-045','actions':oldplan['actions'],'arms':summaries,'world':world,'native_previews':4550,'source_visits':2,'model_calls':0,'optimizer_calls':0,'checkpoint_loads':0,'parent_seconds':parent['elapsed_seconds'],'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],'output_bytes':parent['output_bytes'],'greedy_cached_over_baseline':ratio,'greedy_baseline_over_cached':1/ratio,'greedy_cached_minus_baseline_seconds':difference,'caveat':'One fixed baseline-first pair in one process; filesystem and allocator/order effects remain. Full-arm timers include uncached replay/current independent audit. Cache payload is not RSS; key bytes are unmeasured. No sustained, RL, beam, safety or clinical claim.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--receipt-sha256',required=True);p.add_argument('--result-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30s saved audit cap')));signal.alarm(30)
 try:
  result=audit(args.receipt_sha256,args.result_sha256)
  for path,entry in list(seen.items()):raw(path,entry['sha256'])
 except Exception as error:result={'decision':'HOLD_READER_OR_SAVED_EVIDENCE_MISMATCH','reason':type(error).__name__+': '+str(error)}
 finally:signal.alarm(0)
 result.update(checks=checks,audit_seconds=time.monotonic()-started,audit_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,evidence_files=seen,scope='stdlib saved JSON/source only; no arrays/weights/models/native calls',release_sha256=RELEASE,result_sha256=args.result_sha256,receipt_sha256=args.receipt_sha256)
 with args.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_files','world','actions')},indent=2))
 return 0 if result['decision'].startswith('PASS_') else 1
if __name__=='__main__':raise SystemExit(main())
