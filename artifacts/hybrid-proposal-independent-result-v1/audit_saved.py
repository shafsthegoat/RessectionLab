"""Stdlib-only saved JSON/source audit; no arrays, weights, imports or replay."""
import hashlib,json,math,os,signal,stat,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/hybrid-proposal-pilot-v1'; OUT=BASE/'attempt-01'; DEST=Path(__file__).resolve().parent
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']; MODES=['ridge_top8','sampled_target_top8','exhaustive']
PINS={'result.json':'77deca8508e6a69d78243870ef1dc362f4403e2e4480197052da10af196bad7c','source-index.json':'7ecb61085f8306a1b495190dd5ca5b07ef362c0c5df50f68b25e4aeb48b60c1f','root-release.json':'5a093996a88d9cddce08431280fb95fda351242d670d571255377eb054a798f3'}
seen={}; checks=0; began=time.monotonic()
def need(ok,why):
 global checks
 checks+=1
 if not ok: raise AssertionError(why)
def raw(path,expected=None):
 p=Path(path).resolve();need(p.is_relative_to(ROOT) and p.suffix in ('.json','.py'),'source/json only')
 need(p.name!='ridge.json' and not any(x in p.parts for x in ('data','sources')),'no payload or weights')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  info=os.fstat(fd);need(stat.S_ISREG(info.st_mode) and info.st_size<=8*1024**2,'bounded regular metadata')
  with os.fdopen(fd,'rb',closefd=False) as f:b=f.read(8*1024**2+1)
 finally:os.close(fd)
 need(len(b)<=8*1024**2,'bounded bytes'); h=hashlib.sha256(b).hexdigest()
 if expected:need(h==expected,'pin:'+str(p.relative_to(ROOT)))
 key=str(p.relative_to(ROOT))
 if key in seen:need(seen[key]['sha256']==h,'changed:'+key)
 seen[key]={'sha256':h,'bytes':len(b)}
 return b
def read(p,h=None):return json.loads(raw(p,h))
def semantic(v):return 'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b,why):need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-9),why)
def sumf(rows,k):return sum(r[k] for r in rows)
def main():
 r=read(OUT/'result.json',PINS['result.json']); release=read(BASE/'root-release.json',PINS['root-release.json']); index=read(BASE/'source-index.json',PINS['source-index.json']); receipt=read(BASE/'attempt-01.supervision/receipt.json')
 need(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'],'clean terminal')
 need(receipt['cleanup_errors']==receipt['remaining_owned_pids']==[] and receipt['stop_reason'] is None,'clean cleanup')
 need(receipt['result_sha256']==PINS['result.json'] and receipt['release_sha256']==PINS['root-release.json']==r['release_sha256'],'receipt joins')
 need(receipt['source_index']==release['source_index'] and release['source_index']['sha256']==PINS['source-index.json'],'index joins')
 need(index['head']==release['expected_head'],'HEAD metadata')
 for p,h in index['files'].items():raw(ROOT/p,h)
 need(receipt['caps']==release['caps'],'caps equality')
 for measured,limit in [('elapsed_seconds','parent_seconds'),('sampled_peak_rss_bytes','sampled_rss_bytes'),('output_bytes','output_bytes')]:need(receipt[measured]<release['caps'][limit],measured)
 need(r['elapsed_seconds']<release['caps']['worker_seconds'],'worker time')
 need(r['status']=='complete_fixed_TRAIN_ridge_proposals' and r['all_plans_sealed_before_evaluation'] and r['ridge_unchanged'] and r['cache_restored'],'completed sealed scope')
 need(not r['SELECT_EVAL_opened'] and all(r[k]==0 for k in ('model_calls','optimizer_calls','checkpoint_loads','blocked_external_calls')),'zero unauthorized scope')
 need((r['source_visits'],r['teacher_decisions'],r['fits'],r['fit_rows'])==(28,29,1,2592),'denominators')
 phases=['training']+['planning.'+m for m in MODES]+['replay.'+m for m in MODES]
 need([(v['phase'],v['subject']) for v in r['source_visits_detail']]==[(p,s) for p in phases for s in SUBJECTS],'all28 visits')
 for v in r['source_visits_detail']:need(v['status']=='complete' and v['source_hash']==release['source_world_hashes'][v['subject']],'world fixed')
 need([(a['method'],a['subject']) for a in r['arms']]==[(m,s) for m in MODES for s in SUBJECTS],'all12 denominator')
 labels=read(OUT/'training-labels.json');need(len(labels['states'])==29,'29 label states')
 need(sum(s['feature_rows'] for s in labels['states'])==r['fit_rows'],'label rows')
 need(labels['weighting']=='equal_state_then_uniform_raw_rows','fit weighting')
 summary=[]; plans={}; metrics={}; audits={}; training_cost=[]; disposition={}
 for subject,n in zip(SUBJECTS,[1,14,1,13]):
  old=ROOT/'build/remind-post-exposure-feasibility-v1/attempt-01'/subject
  prior=read(old/'episode-metrics.json'); t=read(OUT/'training'/subject/'teacher-metrics.json'); a=read(OUT/'training'/subject/'teacher-audit.json')
  need(t==prior,'exact full teacher metrics:'+subject);need(a['accepted'] and a['complete_episode'] and a['committed_history_hash']==semantic(t['history']),'teacher certificate')
  states=[s for s in labels['states'] if s['subject']==subject];need(len(states)==n and [s['teacher_action'] for s in states]==[h['action_id'] for h in t['history']],'label history join')
  for s in states:
   need(len(s['raw_ids'])==len(s['legal'])==len(s['priority_labels'])==s['feature_rows'],'raw label dimensions')
   need(all(math.isfinite(y) and y>=0 and (legal or y==0) for y,legal in zip(s['priority_labels'],s['legal'])),'priority label validity')
  training_cost.append(read(OUT/'training'/subject/'costs.json'))
 for row in r['arms']:
  m,s=row['method'],row['subject'];folder=OUT/m/s
  for p,h in row['files'].items():raw(OUT/p,h)
  plan=read(folder/'plan.json'); met=read(folder/'replay/episode-metrics.json'); cert=read(folder/'replay/independent-episode.json')
  seal=plan.pop('seal');need(semantic(plan)==seal,'plan seal');plan['seal']=seal
  need(row['complete'] and row['independent_accepted'] and cert['accepted'] and cert['complete_episode'],'accepted complete')
  need(cert['geometry']['feasible'] and cert['geometry']['failures']==[] and cert['geometry']['complete_tool_checked'],'complete geometry certificate')
  need(cert['committed_history_hash']==semantic(met['history']),'exact replay history binding')
  for k in ('source_hash','decision_model_hash'):need(plan[k]==met[k]==cert[k],k+' joins')
  need(plan['source_hash']==release['source_world_hashes'][s],'fixed source')
  history=met['history'];need(plan['actions']==[h['action_id'] for h in history],'complete action equality')
  need(met['terminated'] and met['steps']==len(history)<=24 and plan['actions'][-1]=='STOP','full STOP history')
  need(row['outcomes']==cert['outcomes'],'outcomes joined'); outcome=row['outcomes']
  motions=[h for h in history if h['action_id']!='STOP'];changes=sum(a['tool_id']!=b['tool_id'] for a,b in zip(motions,motions[1:]))
  close(sumf(history,'reward'),met['total_reward'],'return sum');close(sumf(history,'target_removed_mm3'),met['target_removed_mm3'],'target sum');close(sumf(history,'normal_removed_mm3'),met['normal_removed_mm3'],'outside sum')
  path=sumf(history,'complete_tool_path_length_mm');close(path,outcome['complete_tool_path_length_mm'],'path sum')
  need(changes==outcome['tool_changes'] and len(motions)==outcome['nonstop_actions']==cert['geometry']['action_count'],'actions and tool changes')
  close(met['total_reward'],met['target_removed_mm3']-.03*len(motions)-.001*path-.03*changes,'fixed reward scalar arithmetic')
  for k in ['target_removed_mm3','normal_removed_mm3','total_reward','simulated_removed_volume_mm3','cumulative_contacted_tissue_upper_bound_mm3','currently_retained_contacted_tissue_upper_bound_mm3']:close(met[k],outcome[k],k+' match')
  need(met['clinical_deficit_probability'] is None and outcome['clinical_deficit_probability'] is None,'no clinical claim')
  domain=met['source_and_simulated_domains'];goal=met['supplied_goal_region']
  need(not domain['source_domain_extended'] and not domain['anatomical_or_material_validation'] and not goal['target_modified'],'unknown/full target unchanged')
  close(goal['full_region_membership_mm3'],outcome['total_reference_target_mm3'],'full target denominator')
  close(outcome['target_removed_mm3']/outcome['total_reference_target_mm3'],outcome['reference_target_fraction_removed'],'full target fraction')
  need(plan['accounting']['complete'] and not plan['accounting']['global_optimality_proven'],'greedy qualification')
  need([d['selected_action_id'] for d in plan['accounting']['decisions']]==plan['actions'],'decision action join')
  events=sorted((folder/'inventories').glob('state-*.json'));need(len(events)==len(history),'all inventories')
  queries=0;counts={}
  for i,p in enumerate(events):
   inv=read(p);short=inv['shortlist'];need(inv['step']==i and short['queried_count']+short['not_queried_count']==short['raw_count'],'catalog partition')
   need(short['priority_is_not_legality'],'no priority legality conflation');queries+=short['queried_count']
   need(short['queried_count']==(min(8,short['raw_count']) if m!='exhaustive' else short['raw_count']),'query budget')
   for e in inv['ledger']:
    key=e.get('query_disposition','NO_QUERY_FIELD')+':'+e.get('reason','NO_REASON');counts[key]=counts.get(key,0)+1
  need(queries==row['online_previews']==row['online_costs']['counts']['native_previews'],'exact paid online queries')
  need(row['replay_previews']==queries and row['replay_costs']['counts']['native_previews']==queries,'replay queries')
  for costkey in ('online_costs','replay_costs'):
   secs=row[costkey]['seconds'];need(all(math.isfinite(v) and v>=0 for v in secs.values()),'nonnegative nested costs')
   need(sum(secs.get(k,0) for k in ('proposal_enumeration','public_features','ranking','exact_verification'))<=secs['inventory_inclusive']+1e-7,'cost nesting')
  c=row['cache_stats'];need(c['hits']+c['misses']==c['calls'] and not c['feasibility_cached'] and not c['cavity_cached'] and c['retained_payload_bytes']<=release['cache_limits']['max_payload_bytes'],'cache accounting')
  plans[m,s]=plan;metrics[m,s]=met;audits[m,s]=cert;disposition[m+':'+s]=counts
  summary.append({'method':m,'subject':s,'actions':len(history),'motions':len(motions),'target_cells':outcome['positive_target_source_cells_removed'],'target_mm3':outcome['target_removed_mm3'],'outside_mm3':outcome['normal_removed_mm3'],'return':outcome['total_reward'],'path_mm':path,'tool_changes':changes,'contact_upper_mm3':outcome['cumulative_contacted_tissue_upper_bound_mm3'],'retained_contact_upper_mm3':outcome['currently_retained_contacted_tissue_upper_bound_mm3'],'full_target_mm3':outcome['total_reference_target_mm3'],'unknown_support_cells':domain['source_support_unknown_voxels'],'online_previews':queries,'online_seconds':row['source_through_plan_seconds'],'replay_seconds':row['replay_and_audit_seconds']})
 for s in SUBJECTS:
  original=read(ROOT/'build/remind-post-exposure-feasibility-v1/attempt-01'/s/'episode-metrics.json')
  # The adapter explicitly adds sparse_proposal to its decision contract.
  # Current plan/metrics/certificate identity equality is checked above;
  # every other historical metric, including complete physical history, is exact.
  need({k:v for k,v in metrics['exhaustive',s].items() if k!='decision_model_hash'}=={k:v for k,v in original.items() if k!='decision_model_hash'},'original exhaustive metrics except explicit selector identity:'+s)
  for m in MODES:
   for k in ['source_hash','reference_hash','native_grid_reconciliation','post_exposure','source_and_simulated_domains','intensity_normalization']:
    need(metrics[m,s][k]==metrics['exhaustive',s][k],'matched world:'+k)
 totals={m:{k:sum(a[k] for a in r['arms'] if a['method']==m) for k in ['online_previews','source_through_plan_seconds','replay_previews','replay_and_audit_seconds']} for m in MODES}
 need(totals==r['comparison']['totals'],'exact cost totals')
 def worse(a,b):
  x,y=audits[a]['outcomes'],audits[b]['outcomes'];tol=release['gates']['numeric_tolerance']
  return [k for k,sign in [('target_removed_mm3',1),('total_reward',1),('normal_removed_mm3',-1),('cumulative_contacted_tissue_upper_bound_mm3',-1)] if sign*x[k]<sign*y[k]-tol]
 losses={s:worse(('ridge_top8',s),('exhaustive',s)) for s in SUBJECTS};hloss={s:worse(('ridge_top8',s),('sampled_target_top8',s)) for s in SUBJECTS}
 quality=not any(losses.values());hq=not any(hloss.values());hi=hq and any(worse(('sampled_target_top8',s),('ridge_top8',s)) for s in SUBJECTS)
 ratios={'preview_ratio':totals['ridge_top8']['online_previews']/totals['exhaustive']['online_previews'],'online_wall_ratio':totals['ridge_top8']['source_through_plan_seconds']/totals['exhaustive']['source_through_plan_seconds'],'heuristic_wall_ratio':totals['ridge_top8']['source_through_plan_seconds']/totals['sampled_target_top8']['source_through_plan_seconds']}
 for k,v in ratios.items():close(v,r['comparison'][k],k)
 verdict=quality and ratios['preview_ratio']<=.5 and ratios['online_wall_ratio']<=.8 and (hi or hq and ratios['heuristic_wall_ratio']<=.9)
 need((quality,hq,hi,verdict)==tuple(r['comparison'][k] for k in ['no_quality_loss_vs_exhaustive','heuristic_no_quality_loss','heuristic_quality_improvement','TRAIN_go']),'all frozen gates')
 requests=r['native_budget']['requests'];need(sum(v['preview_admissions'] for v in requests.values())==r['native_previews']==10896,'all requests')
 need(r['native_transitions']==2*29+2*sum(len(p['actions']) for p in plans.values())==182,'all transitions')
 need(r['ridge_calls']==2*sum(len(plans['ridge_top8',s]['actions']) for s in SUBJECTS)==58,'all ridge calls')
 for m in MODES:
  for s in SUBJECTS:
   a=next(x for x in r['arms'] if x['method']==m and x['subject']==s)
   for phase,key in [('planning','online_previews'),('replay','replay_previews')]:need(requests[phase+'.'+m+'.'+s]['preview_admissions']==a[key],'request arm join')
 overhead={m:{k:sum(a['online_costs']['seconds'].get(k,0) for a in r['arms'] if a['method']==m) for k in ['public_features','ranking','proposal_enumeration','exact_verification','inventory_inclusive']} for m in MODES}
 firstdiv={}
 for s in SUBJECTS:
  aa,bb=plans['ridge_top8',s]['actions'],plans['exhaustive',s]['actions'];firstdiv[s]=next((i for i,(a,b) in enumerate(zip(aa,bb)) if a!=b),None)
 for p,h in list(seen.items()):raw(ROOT/p,h['sha256'])
 return {'status':'PASS_saved_evidence_consistency','checks':checks,'source_index_entries':len(index['files']),'source_and_input_files_unchanged':len(seen),'runtime_seconds':time.monotonic()-began,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'stdlib saved source/JSON only; no weights/arrays/native/model/refit','parent':{k:receipt[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','output_bytes','remaining_owned_pids','cleanup_errors']},'worker_seconds':r['elapsed_seconds'],'fit_seconds':r['fit_seconds'],'teacher_state_count':29,'fit_rows':r['fit_rows'],'source_visits':28,'native_previews':r['native_previews'],'native_transitions':r['native_transitions'],'ridge_calls':r['ridge_calls'],'arms':summary,'totals':totals,'ratios':ratios,'TRAIN_go':verdict,'loss_fields_vs_exhaustive':losses,'loss_fields_vs_heuristic':hloss,'online_nested_costs':overhead,'training_costs':[{k:v for k,v in t.items() if k!='cache_stats'} for t in training_cost],'first_action_divergence_zero_based':firstdiv,'catalog_dispositions':disposition,'weights_not_opened':{'path':str((OUT/'ridge.json').relative_to(ROOT)),'parent_recorded_sha256':r['ridge_file_sha256']},'limitations':['Saved certificate consistency; no new geometry replay.','All12-before-evaluation follows reviewed control flow plus saved attestation, not independent execution tracing.','All targets are disclosed TRAIN annotations; no private evaluator or transfer outcome.','Single fixed order; exhaustive computes unused features and writes instrumentation; no clean-baseline latency claim.','Nested timing fields must not be summed as independent costs.','Full target/unknown-domain assumptions and clinical nulls persist.']}
if __name__=='__main__':
 signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('30-second saved audit cap')));signal.alarm(30)
 try:
  result=main();(DEST/'audit-result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');(DEST/'input-index.json').write_text(json.dumps(seen,indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k] for k in ['status','checks','runtime_seconds','source_and_input_files_unchanged','TRAIN_go']}))
 except BaseException as e:
  (DEST/'audit-failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'checks':checks},indent=2)+'\n');raise
 finally:signal.alarm(0)
