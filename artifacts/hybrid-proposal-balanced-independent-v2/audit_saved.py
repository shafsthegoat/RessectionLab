"""Stdlib-only saved JSON/source audit; no arrays, weights, imports or replay."""
import hashlib,json,math,os,signal,stat,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/hybrid-proposal-balanced-v2'; OUT=BASE/'attempt-01'; DEST=Path(__file__).resolve().parent
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']; MODES=['old_ridge_top8','balanced_ridge_top8','exhaustive']
seen={}; checks=0; began=time.monotonic()
def need(ok,why):
 global checks
 checks+=1
 if not ok: raise AssertionError(why)
def raw(path,expected=None):
 p=Path(path).resolve();need(p.is_relative_to(ROOT) and p.suffix in ('.json','.py'),'source/json only')
 need(p.name not in ('ridge.json','old_ridge_top8.json','balanced_ridge_top8.json') and not any(x in p.parts for x in ('data','sources')),'no payload or weights')
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
def main(args):
 r=read(OUT/'result.json',args.result_sha);receipt=read(BASE/'attempt-01.supervision/receipt.json',args.receipt_sha)
 release=read(BASE/'root-release.json',args.release_sha);index=read(BASE/'source-index.json',args.index_sha)
 need(receipt['result_sha256']==args.result_sha and receipt['release_sha256']==r['release_sha256']==args.release_sha,'receipt joins')
 need(receipt['source_index']==release['source_index'] and release['source_index']['sha256']==args.index_sha,'index joins')
 need(index['head']==release['expected_head'],'HEAD metadata')
 for p,h in index['files'].items():raw(ROOT/p,h)
 need(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'],'clean terminal')
 need(receipt['cleanup_errors']==receipt['remaining_owned_pids']==[] and receipt['stop_reason'] is None,'clean cleanup')
 need(receipt['caps']==release['caps'],'caps')
 for measured,limit in [('elapsed_seconds','parent_seconds'),('sampled_peak_rss_bytes','sampled_rss_bytes'),('output_bytes','output_bytes')]:need(receipt[measured]<release['caps'][limit],measured)
 need(r['elapsed_seconds']<release['caps']['worker_seconds'],'worker time')
 need(r['status']=='complete_fixed_TRAIN_balanced_ridge_contrast' and r['all_plans_sealed_before_evaluation'] and r['ridge_unchanged'] and r['cache_restored'],'complete scope')
 need(not r['SELECT_EVAL_opened'] and all(r[k]==0 for k in ('model_calls','optimizer_calls','checkpoint_loads','blocked_external_calls')),'no expanded scope')
 need((r['source_visits'],r['teacher_decisions'],r['fits'])==(28,29,2),'fixed denominator')
 need(r['old_ridge_parameter_identity'] and r['teacher_labels_identity'],'original controls')
 oldbase=ROOT/'build/hybrid-proposal-pilot-v1/attempt-01'
 oldr=read(oldbase/'result.json','77deca8508e6a69d78243870ef1dc362f4403e2e4480197052da10af196bad7c')
 oldlabels=read(oldbase/'training-labels.json','437c2620cceeb191ddf38e3cc932c9e6e7f918c8481110e5319d29b963b248de')
 labels=read(OUT/'training-labels.json');need(labels['data_hash']==oldlabels['data_hash'] and labels['states']==oldlabels['states'],'all teacher labels exact')
 need(len(labels['states'])==29 and sum(s['feature_rows'] for s in labels['states'])==r['fit_rows']==2592,'all training rows')
 phases=['training']+['planning.'+m for m in MODES]+['replay.'+m for m in MODES]
 need([(v['phase'],v['subject']) for v in r['source_visits_detail']]==[(p,s) for p in phases for s in SUBJECTS],'all28 visits')
 for v in r['source_visits_detail']:need(v['status']=='complete' and v['source_hash']==release['source_world_hashes'][v['subject']],'public world')
 need([(a['method'],a['subject']) for a in r['arms']]==[(m,s) for m in MODES for s in SUBJECTS],'all12 arms')
 readouts=read(OUT/'teacher-readouts.json');weights=read(OUT/'regression-weights.json');state_summary=[]
 need(len(readouts['states'])==len(weights['groups'])==29 and readouts['calls']==r['teacher_score_calls']==58 and readouts['no_native_queries'],'teacher readout denominator')
 for label,rec,group in zip(labels['states'],readouts['states'],weights['groups']):
  for k in ('subject','step','teacher_action','raw_ids','public_input_hash','priority_labels'):need(label[k]==rec[k],'teacher readout join:'+k)
  n=label['feature_rows'];y=label['priority_labels'];pos=sum(v>0 for v in y);zero=n-pos
  need(group['subject']==label['subject'] and group['step']==label['step'] and group['rows']==n and group['positive_rows']==pos,'group join')
  close(group['old_mass'],1/29,'old state mass');close(group['balanced_mass'],1/29,'balanced state mass')
  close(group['old_positive_mass'],pos/(29*n),'old positive mass')
  pmass=1/58 if pos and zero else (1/29 if pos else 0.)
  close(group['balanced_positive_mass'],pmass,'balanced positive mass');close(group['balanced_zero_mass'],1/29-pmass,'balanced zero mass')
  summary={'subject':label['subject'],'step':label['step'],'teacher_action':label['teacher_action'],'raw_rows':n,'positive_rows':pos,'best_positive_reward_label':max(y,default=0.),'models':{}}
  for mode in MODES[:2]:
   model=rec['models'][mode];scores=model['scores'];need(len(scores)==n and all(math.isfinite(v) for v in scores),'saved scalar scores')
   order=sorted(range(n),key=lambda i:(-scores[i],i));rank=[order.index(i)+1 for i in range(n)]
   need(rank==model['ranks'] and model['top8_ids']==[label['raw_ids'][i] for i in order[:8]],'stable raw-order ranks')
   teacher_rank=None if label['teacher_action']=='STOP' else rank[label['raw_ids'].index(label['teacher_action'])]
   need(teacher_rank==model['teacher_rank'],'teacher rank')
   selected=order[:8];best=max(y,default=0.);got=max((y[i] for i in selected),default=0.)
   old_mse=sum((v-t)**2/(29*n) for v,t in zip(scores,y))
   balanced_mse=sum((v-t)**2*(1/(58*(pos if t>0 else zero)) if pos and zero else 1/(29*n)) for v,t in zip(scores,y))
   summary['models'][mode]={'teacher_rank':teacher_rank,'top8_positive_count':sum(y[i]>0 for i in selected),'top8_best_label':got,'positive_priority_regret':best-got,'best_label_rank':min((rank[i] for i,v in enumerate(y) if v==best),default=None),'old_weighted_squared_error':old_mse,'balanced_weighted_squared_error':balanced_mse}
  state_summary.append(summary)
 plans={};metrics={};audits={};summary=[];training_cost=[]
 for s in SUBJECTS:
  t=read(OUT/'training'/s/'teacher-metrics.json');old=read(oldbase/'training'/s/'teacher-metrics.json');cert=read(OUT/'training'/s/'teacher-audit.json')
  need(t==old and cert['accepted'] and cert['committed_history_hash']==semantic(t['history']),'teacher full replay')
  training_cost.append({k:v for k,v in read(OUT/'training'/s/'costs.json').items() if k!='cache_stats'})
 for row in r['arms']:
  m,s=row['method'],row['subject'];folder=OUT/m/s
  for p,h in row['files'].items():raw(OUT/p,h)
  plan=read(folder/'plan.json');met=read(folder/'replay/episode-metrics.json');cert=read(folder/'replay/independent-episode.json')
  seal=plan.pop('seal');need(semantic(plan)==seal,'seal');plan['seal']=seal
  need(row['complete'] and row['independent_accepted'] and cert['accepted'] and cert['complete_episode'],'complete accepted')
  need(cert['geometry']['feasible'] and cert['geometry']['failures']==[] and cert['geometry']['complete_tool_checked'],'full geometry')
  hist=met['history'];outcome=row['outcomes'];need(cert['committed_history_hash']==semantic(hist) and outcome==cert['outcomes'],'exact certificate')
  for k in ('source_hash','decision_model_hash'):need(plan[k]==met[k]==cert[k],k)
  need(plan['source_hash']==release['source_world_hashes'][s],'same source')
  need(plan['actions']==[h['action_id'] for h in hist] and met['terminated'] and met['steps']==len(hist)<=24,'complete history')
  need(plan['actions'][-1]=='STOP' or len(hist)==24,'honest endpoint')
  motions=[h for h in hist if h['action_id']!='STOP'];changes=sum(a['tool_id']!=b['tool_id'] for a,b in zip(motions,motions[1:]));path=sumf(hist,'complete_tool_path_length_mm')
  close(sumf(hist,'reward'),met['total_reward'],'reward sum');close(sumf(hist,'target_removed_mm3'),met['target_removed_mm3'],'target sum');close(sumf(hist,'normal_removed_mm3'),met['normal_removed_mm3'],'outside sum')
  need(changes==outcome['tool_changes'] and len(motions)==outcome['nonstop_actions']==cert['geometry']['action_count'],'actions/changes')
  close(met['total_reward'],met['target_removed_mm3']-.2*met['normal_removed_mm3']-.03*len(motions)-.001*path-.03*changes,'fixed reward')
  for k in ('target_removed_mm3','normal_removed_mm3','total_reward','cumulative_contacted_tissue_upper_bound_mm3'):close(met[k],outcome[k],k)
  domain=met['source_and_simulated_domains'];goal=met['supplied_goal_region'];need(not domain['source_domain_extended'] and not domain['anatomical_or_material_validation'] and not goal['target_modified'],'domain/full target')
  close(goal['full_region_membership_mm3'],outcome['total_reference_target_mm3'],'full denominator');close(outcome['target_removed_mm3']/outcome['total_reference_target_mm3'],outcome['reference_target_fraction_removed'],'fraction')
  need(outcome['clinical_deficit_probability'] is None,'no clinical claim')
  need(plan['accounting']['complete'] and not plan['accounting']['global_optimality_proven'],'greedy scope')
  need([d['selected_action_id'] for d in plan['accounting']['decisions']]==plan['actions'],'selection actions')
  events=sorted((folder/'inventories').glob('state-*.json'));need(len(events)==len(plan['accounting']['decisions']),'inventory count');queries=0
  for p in events:
   inv=read(p);short=inv['shortlist'];queries+=short['queried_count'];need(short['queried_count']+short['not_queried_count']==short['raw_count'],'query partition')
   need(short['priority_is_not_legality'],'legality separate');need(short['queried_count']==(short['raw_count'] if m=='exhaustive' else min(8,short['raw_count'])),'top8')
   if m=='exhaustive':need(short['priority_scores'] is None,'no exhaustive scores')
   else:
    scores=short['priority_scores'];ids=short['raw_ids'];need(len(scores)==len(ids)==short['raw_count'],'aligned raw priorities')
    order=sorted(range(len(ids)),key=lambda i:(-scores[i],i));need(short['selected_ids']==[ids[i] for i in order[:8]],'exact top8')
  need(queries==row['online_previews']==row['online_costs']['counts']['native_previews'],'charged queries')
  for key in ('online_costs','replay_costs'):
   secs=row[key]['seconds'];need(all(math.isfinite(v) and v>=0 for v in secs.values()),'finite costs')
   need(sum(secs.get(k,0) for k in ('proposal_enumeration','public_features','ranking','exact_verification','input_binding','inventory_logging'))<=secs['inventory_inclusive']+1e-7,'nested cost containment')
   if m=='exhaustive':need(row[key]['counts'].get('feature_rows',0)==row[key]['counts'].get('ridge_calls',0)==0 and secs.get('public_features',0)==secs.get('ranking',0)==0,'lean exhaustive paid work')
  c=row['cache_stats'];need(c['hits']+c['misses']==c['calls'] and not c['feasibility_cached'] and not c['cavity_cached'],'cache semantics')
  plans[m,s]=plan;metrics[m,s]=met;audits[m,s]=cert
  summary.append({'method':m,'subject':s,'motions':len(motions),'endpoint':row['endpoint'],'target_mm3':outcome['target_removed_mm3'],'target_cells':outcome['positive_target_source_cells_removed'],'outside_mm3':outcome['normal_removed_mm3'],'return':outcome['total_reward'],'path_mm':path,'tool_changes':changes,'contact_upper_mm3':outcome['cumulative_contacted_tissue_upper_bound_mm3'],'full_target_mm3':outcome['total_reference_target_mm3'],'online_previews':row['online_previews'],'online_seconds':row['source_through_plan_seconds']})
 for s in SUBJECTS:
  for m,oldm in [('old_ridge_top8','ridge_top8'),('exhaustive','exhaustive')]:
   old=read(oldbase/oldm/s/'replay/episode-metrics.json')
   # AdmittedSparseTask._contract_record binds PilotAdmission.fingerprint,
   # whose explicit pilot version/release changes in v2. Exclude only this
   # top-level identity from historical comparison, never history/world/math.
   # Current plan/metrics/certificate identities remain exact above.
   need({k:v for k,v in metrics[m,s].items() if k!='decision_model_hash'}=={k:v for k,v in old.items() if k!='decision_model_hash'},'exact historical metrics except explicit pilot contract identity')
  oldp=read(oldbase/'ridge_top8'/s/'plan.json');need(plans['old_ridge_top8',s]['ridge_hash']==oldp['ridge_hash'] and plans['old_ridge_top8',s]['actions']==oldp['actions'],'old parameter/actions')
  for m in MODES:
   for k in ('source_hash','reference_hash','native_grid_reconciliation','post_exposure','source_and_simulated_domains','intensity_normalization'):need(metrics[m,s][k]==metrics['exhaustive',s][k],'matched world:'+k)
 totals={m:{k:sum(a[k] for a in r['arms'] if a['method']==m) for k in ('online_previews','source_through_plan_seconds','replay_previews','replay_and_audit_seconds')} for m in MODES};need(totals==r['comparison']['totals'],'totals')
 def losses(a,b):
  x,y=audits[a]['outcomes'],audits[b]['outcomes'];t=release['gates']['numeric_tolerance']
  return [k for k,sgn in [('target_removed_mm3',1),('total_reward',1),('normal_removed_mm3',-1),('cumulative_contacted_tissue_upper_bound_mm3',-1)] if sgn*x[k]<sgn*y[k]-t]
 loss={m:{s:losses(('balanced_ridge_top8',s),(m,s)) for s in SUBJECTS} for m in ('old_ridge_top8','exhaustive')}
 quality=not any(loss['exhaustive'].values());oldquality=not any(loss['old_ridge_top8'].values());improved=any(audits['balanced_ridge_top8',s]['outcomes']['total_reward']>audits['old_ridge_top8',s]['outcomes']['total_reward']+release['gates']['numeric_tolerance'] for s in SUBJECTS)
 ratios={'preview_ratio':totals['balanced_ridge_top8']['online_previews']/max(1,totals['exhaustive']['online_previews']),'online_wall_ratio':totals['balanced_ridge_top8']['source_through_plan_seconds']/totals['exhaustive']['source_through_plan_seconds']}
 for k,v in ratios.items():close(v,r['comparison'][k],k)
 verdict=quality and oldquality and improved and ratios['preview_ratio']<=.5 and ratios['online_wall_ratio']<=.8
 need((quality,oldquality,improved,verdict)==tuple(r['comparison'][k] for k in ('balanced_no_quality_loss_vs_exhaustive','balanced_no_quality_loss_vs_old_ridge','balanced_strict_return_improvement_vs_old_ridge','loss_contrast_go')),'prospective gates')
 need(r['comparison']['historical_v1_comparison']==oldr['comparison'] and not r['comparison']['v1_retroactively_promoted'] and not oldr['comparison']['TRAIN_go'],'original negative preserved')
 need(r['native_transitions']==2*29+2*sum(len(p['actions']) for p in plans.values())<=634,'transition accounting')
 need(sum(q['preview_admissions'] for q in r['native_budget']['requests'].values())==r['native_previews']<=40000,'all preview requests')
 calls=sum(a['online_costs']['counts'].get('ridge_calls',0)+a['replay_costs']['counts'].get('ridge_calls',0) for a in r['arms']);need(r['ridge_calls']==calls+58<=442 and calls==r['online_replay_ridge_calls'],'linear calls')
 for p,h in list(seen.items()):raw(ROOT/p,h['sha256'])
 return {'status':'PASS_saved_evidence_consistency','checks':checks,'runtime_seconds':time.monotonic()-began,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'files_unchanged':len(seen),'arms':summary,'states':state_summary,'totals':totals,'ratios':ratios,'loss_fields':loss,'loss_contrast_go':verdict,'historical_v1_TRAIN_go':False,'native_previews':r['native_previews'],'native_transitions':r['native_transitions'],'source_visits':28,'fits':2,'fit_seconds':r['fit_seconds'],'ridge_calls':r['ridge_calls'],'training_costs':training_cost,'parent':{k:receipt[k] for k in ('elapsed_seconds','sampled_peak_rss_bytes','output_bytes','remaining_owned_pids','cleanup_errors')},'limits':'saved consistency only; no arrays/weights/model or new native replay; all-plan chronology is source order plus attestation; TRAIN single ordered pair only; no physical/transfer validation'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--result-sha',required=True);p.add_argument('--receipt-sha',required=True);p.add_argument('--release-sha',required=True);p.add_argument('--index-sha',required=True);args=p.parse_args()
 signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError('30-second saved audit cap')));signal.alarm(30)
 try:
  result=main(args);(DEST/'audit-result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');(DEST/'input-index.json').write_text(json.dumps(seen,indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k] for k in ('status','checks','runtime_seconds','loss_contrast_go')}))
 except BaseException as e:
  (DEST/'audit-failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'checks':checks},indent=2)+'\n');raise
 finally:signal.alarm(0)
