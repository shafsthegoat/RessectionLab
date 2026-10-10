"""Bounded saved JSON/source audit only. No application, model, or engine imports."""
from pathlib import Path
import hashlib,json,math,re,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
LIVE=ROOT/'build/train-refit-desktop-live-v1';PREP=ROOT/'build/goal-conditioned-policy-v1/train-refit-integration-v1'
OLD=ROOT/'build/goal-conditioned-policy-v1/full-teacher-rollout-run-v1'
ATTEMPTS=Path.home()/'Library/Application Support/ressectionlab-desktop/research-runs/generated-contact-family-attempts'
start=time.monotonic();files={}
def raw(p,limit=2*1024*1024):
 p=Path(p);assert p.suffix not in ('.gmckpt','.npz','.npy','.nii','.gz','.pt','.pth'),str(p)
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<=limit,str(p)
 b=p.read_bytes();assert len(b)<=limit;files[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return b
def read(p,limit=2*1024*1024):return json.loads(raw(p,limit))
def sha(p):raw(p);return files[str(p)]['sha256']
def digest(v):return 'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def cells(v):
 assert all(len(p)==3 and all(type(x) is int for x in p) for p in v);return {tuple(p) for p in v}
def maskhash(points,shape):
 b=bytearray(math.prod(shape))
 for x,y,z in points:
  assert 0<=x<shape[0] and 0<=y<shape[1] and 0<=z<shape[2];b[(x*shape[1]+y)*shape[2]+z]=1
 return 'sha256:'+hashlib.sha256(json.dumps({'dtype':'|b1','shape':shape},sort_keys=True).encode()+b).hexdigest()
launch=read(LIVE/'launch.json');terminal=read(LIVE/'terminal.json');raw(LIVE/'electron.log')
assert terminal['exit_code']==0 and terminal['stop_reason'] is None
assert terminal['elapsed_seconds']<launch['wall_cap_seconds'] and terminal['peak_sampled_process_group_rss_bytes']<launch['sampled_group_rss_cap_bytes']
window=( (LIVE/'launch.json').stat().st_mtime, (LIVE/'terminal.json').stat().st_mtime )
entries=list(ATTEMPTS.iterdir());assert len(entries)<=100
candidates=[]
for d in entries:
 p=d/'attempt.json'
 if d.is_dir() and p.is_file() and window[0]<=p.stat().st_mtime<=window[1]:
  a=read(p,4096);candidates.append((d,a))
assert len(candidates)==1,[(str(d),a) for d,a in candidates]
run,attempt=candidates[0]
assert (attempt['layoutId'],attempt['goalId'],attempt['selector'])==('pcf-14','surface','IL_TRAIN_REFIT')
assert attempt['automaticRetry'] is False and attempt['maxWallSeconds']==20 and attempt['maxSampledWorkerRssBytes']==2**30
r=read(run/'result.json');receipt=read(run/'supervision.json');completion=read(run/'completion.json');read(run/'progress.json');raw(run/'worker.log')
assert sha(run/'result.json')==completion['resultSha256']=='e006aef48c63a30eb323d9622176f48951363bc4ce4c94577d4a2508feeb1c49'
assert sha(run/'supervision.json')==completion['supervisionSha256']
assert receipt['status']==completion['status']=='complete' and receipt['workerReturnCode']==0 and receipt['stopReason'] is None
assert receipt['cleanupErrors']==[] and receipt['unresolvedWorkerPid'] is None and receipt['workerTerminationConfirmed'] is True and receipt['automaticRetry'] is False
assert receipt['elapsedSeconds']<20 and receipt['sampledPeakRssBytes']<2**30
assert r['status']=='complete' and r['scope']=='generated_contact_family_v3' and r['newOptimizerUpdates']==0 and r['clinicalValidation'] is False
assert r['selector']==r['policyVariant']=='IL_TRAIN_REFIT' and r['splitRole']=='TRAIN'
e=r['episode'];old=read(OLD/'TRAIN-10-episode.json');oldresult=read(OLD/'result.json');row=oldresult['rows'][10]
assert sha(OLD/'result.json')=='32dd1bdcbd61f9ba41dd6654aaf33c9c446cf1ba35dc222e88c4734aec0782d8'
assert (row['layout_id'],row['goal_id'],row['role'])==('pcf-14','surface','TRAIN')
assert e==old['episode'] and r['episodeCanonicalJson']==old['episodeCanonicalJson']
assert json.loads(r['episodeCanonicalJson'])=={k:v for k,v in e.items() if k!='episodeId'}
assert 'sha256:'+hashlib.sha256(r['episodeCanonicalJson'].encode()).hexdigest()==e['episodeId']==row['episode_id']
assert digest(e['planning']['strategy'])==e['planning']['strategySeal']==row['strategy_seal']
assert e['planning']['strategy']['history']==e['history']==e['metrics']['history']
assert digest(e['history'])==row['history_hash']
assert e['schema']=='resectionlab.shared-native-contact-learning-episode.v3' and e['selector']=='IL'
assert e['splitRole']=='TRAIN' and e['layoutId']==r['layoutId']=='pcf-14' and e['publicGoal']['goalId']==r['goalId']=='surface'
assert e['evidenceKind']=='generated_software_fixture' and e['patientAdmission'] is False and e['clinicalValidation'] is False
assert e['sourceHash']==row['source_hash']==e['sourceBinding']['native_source_hash']
assert e['caseHash']==r['caseHash']==row['case_hash']==e['sourceBinding']['display_case_hash'] and e['sourceBinding']['private_reference_published'] is False
assert e['decisionModelHash']==row['decision_model_hash']
publication=read(ROOT/'artifacts/public-contact-train-refit-v1/desktop-release.json');assert sha(ROOT/'artifacts/public-contact-train-refit-v1/desktop-release.json')=='68091a27acf63715f23e5a649de1f06dee56b2391e6bb6919fc8bbb36a2d8887'
expected={'releaseManifestSha256':'68091a27acf63715f23e5a649de1f06dee56b2391e6bb6919fc8bbb36a2d8887',**{out:publication['files'][slot]['sha256'] for out,slot in [('checkpointFileSha256','checkpoint'),('fitResultSha256','fitResult'),('rolloutResultSha256','rolloutResult'),('independentAuditSha256','independentAudit')]}}
assert r['releaseEvidence']==expected
assert expected['checkpointFileSha256']=='5d7151397141c92ff82d8684814b9a0caed111f1809268bd448b8c1ea26d6bf7'
a=e['learnedAuthorship'];assert a['method']=='IL' and a['checkpointFileSha256']==expected['checkpointFileSha256']
assert a['parameterHash']==publication['parameterHash'] and a['experimentHash']==r['experimentHash']==publication['experimentHash']
assert a['familyHash']==r['familyHash']==e['familyHash']==publication['familyHash']
assert a['completedUpdates']==32 and a['inferenceOptimizerUpdates']==0
assert digest(e['taskContract']['checkpoint'])==a['trainingLineageHash']
assert e['planning']['actor_forward_calls']==e['planning']['model_transition_calls']==2 and e['planning']['optimizer_updates']==0
assert e['planning']['sealedBeforeExecution'] is True and e['planning']['referenceScoringPerformed'] is False
assert e['planning']['observation_ids']==row['planning']['observation_ids'] and e['planning']['parameter_hash']==a['parameterHash']
assert e['taskContract']['proposalMode']==e['taskContract']['sourceCandidateVersion']=='fixed_lattice_access_centerline_v1'
assert e['taskContract']['declaration']['role']=='TRAIN' and e['taskContract']['declaration']['experiment_hash']==r['experimentHash']
assert e['geometryAudit']==row['geometry_audit'] and e['geometryAudit']['feasible'] is True and e['geometryAudit']['failures']==[]
assert e['geometryAudit']['complete_tool_checked'] is True and e['geometryAudit']['frontier_checked'] is True
shape=e['shape'];assert shape==[15,15,14];goal=tuple(e['publicGoal']['nativeIndex']);tools={t['tool_id']:t for t in e['tools']}
objective=e['taskContract']['objective'];assert digest(objective)==e['publicGoal']['objectiveHash']
removed=set();contacts=set();probed=set();reward=0.;cost=0.;distance=0.;previous_tool=None;engine_state=None
frames=[(-1,'initial',None,None,[],[],[])];actions=[]
assert [h['interaction_mode'] for h in e['history']]==['aspirate','probe']
for i,h in enumerate(e['history']):
 if engine_state is not None:assert h['source_state_hash']==engine_state
 engine_state=h['result_state_hash'];tool=tools[h['tool_id']];assert tool['interactionMode']==h['interaction_mode']
 assert h['source_hash']==e['sourceHash'] and h['native_affine']==e['affine'] and h['source_shape']==shape
 before=int(goal in probed and goal not in removed);assert h['goal_potential_before']==before
 close(h['insertion_distance_mm'],math.dist(h['entry_mm'],h['tip_mm']));close(h['complete_tool_path_length_mm'],2*h['insertion_distance_mm'])
 assert h['tool_change_count']==int(previous_tool is not None and previous_tool!=h['tool_id']);previous_tool=h['tool_id']
 rm=set();ct=set();pr=set();last=h['entry_mm']
 for m in h['microsteps']:
  assert m['tip_start_mm']==last;last=m['tip_end_mm'];assert m['active_radius_mm']==tool['tip_radius_mm']
  for k in range(3):close(m['active_stroke_start_mm'][k],m['tip_start_mm'][k]-tool['tip_length_mm']*h['axis_unit'][k])
  assert m['active_stroke_end_mm']==m['tip_end_mm']
  mr=cells(m['removed_indices_native']);mc=cells(m['contact_indices_native']);mp=mc-removed if h['interaction_mode']=='probe' else set()
  if h['interaction_mode']=='probe':assert not mr
  assert not mr&removed;removed|=mr;contacts|=mc;probed|=mp;rm|=mr;ct|=mc;pr|=mp
  frames.append((i,'insertion',h,m['tip_end_mm'],m['removed_indices_native'],m['contact_indices_native'],[list(p) for p in sorted(mp)]))
 assert last==h['tip_mm'] and rm==cells(h['removed_indices_native']) and ct==cells(h['contact_indices_native']) and pr==cells(h['probe_contact_indices_native'])
 for m in reversed(h['microsteps']):frames.append((i,'withdrawal',h,m['tip_start_mm'],[],[],[]))
 after=int(goal in probed and goal not in removed);assert h['goal_potential_after']==after
 c=objective['costs'];expectedcost=len(rm)*c['normal_per_mm3']+c['action_cost']+h['complete_tool_path_length_mm']*c['motion_per_mm']+h['tool_change_count']*c['tool_change_cost']
 close(h['removed_volume_mm3'],len(rm));close(h['effort_and_removal_cost'],expectedcost);close(h['reward'],objective['completion_value']*(after-before)-expectedcost)
 cost+=expectedcost;reward+=h['reward'];distance+=h['complete_tool_path_length_mm']
 actions.append({'action_id':h['action_id'],'mode':h['interaction_mode'],'removed_cells':len(rm),'microsteps':len(h['microsteps']),'entry_ras_mm':h['entry_mm'],'tip_ras_mm':h['tip_mm'],'round_trip_path_mm':h['complete_tool_path_length_mm'],'cost':h['effort_and_removal_cost'],'reward':h['reward']})
assert len(frames)==len(e['replayFrames'])==77 and cells(e['finalRemovedIndicesNative'])==removed
fr=set();fc=set();fp=set();prev=None;prevremaining=None
for n,(f,x) in enumerate(zip(e['replayFrames'],frames)):
 i,phase,h,tip,rm,ct,pr=x
 assert f['frameIndex']==n and f['actionIndex']==i and f['phase']==phase and f['tipRasMm']==tip
 assert f['toolId']==(None if h is None else h['tool_id']) and f['mode']==(None if h is None else h['interaction_mode'])
 assert f['axis']==(None if h is None else h['axis_unit'])
 assert cells(f['removedIndicesNative'])==cells(rm) and cells(f['contactIndicesNative'])==cells(ct) and cells(f['probeContactIndicesNative'])==cells(pr)
 fr|=cells(rm);fc|=cells(ct);fp|=cells(pr)
 assert f['cavityHash']==maskhash(fr,shape) and f['contactHash']==maskhash(fc,shape) and f['probeContactHash']==maskhash(fp,shape)
 if prevremaining is not None and not rm:assert f['remainingHash']==prevremaining
 prevremaining=f['remainingHash']
 state=digest({k:v for k,v in f.items() if k not in ('removedIndicesNative','contactIndicesNative','probeContactIndicesNative','stateBefore','stateAfter')})
 assert f['stateAfter']==state and f['stateBefore']==(state if n==0 else prev);prev=state
assert e['initialStateId']==e['replayFrames'][0]['stateAfter'] and e['finalStateId']==prev and e['nativeEngineFinalStateId']==engine_state
assert fr==removed and fc==contacts and fp==probed and goal in probed and goal not in removed
close(reward,.292);close(cost,.708);close(distance,18);assert len(removed)==3
close(e['metrics']['total_reward'],reward);assert e['metrics']['goal_contacted_and_retained'] is True
assert e['replayFrames'][39]['phase']=='insertion' and e['replayFrames'][39]['mode']=='probe'
assert e['replayFrames'][76]['phase']=='withdrawal'
# Exact promoted source and root validation evidence, no imports/build execution.
index=read(PREP/'promotion-index.json');assert sha(PREP/'promotion-index.json')=='905568afe3b7ff573de6c2f522149c8938874a8e1eed4ca94ce79af187afbb6f'
assert sha(PREP/'promotion.patch')=='35a057f58ba7fd540d55c649e61e16c5b6197c9e86e70b771c02882931e47ec1'
for item in index['files']:assert sha(ROOT/item['destination'])==item['source_sha256']
closure=read(PREP/'source-closure.json')
for rel,h in closure['files'].items():assert sha(ROOT/('src/resectionlab/'+rel[6:] if rel.startswith('local:') else rel))==h
rootindex=read(PREP/'root-index.json');assert rootindex['tree']=='27d8bf22171d87749bb3ef0093fe43bf396cef42'
logs={name:raw(PREP/name).decode() for name in ('root-python-tests.log','root-desktop-tests.log','root-index-tests.log','root-working-build.log','root-index-build.log')}
assert '33 passed' in logs['root-python-tests.log']
counts={name:[int(x) for x in re.findall(r'ℹ pass (\d+)',logs[name])] for name in ('root-desktop-tests.log','root-index-tests.log')}
assert counts=={'root-desktop-tests.log':[97,341,79],'root-index-tests.log':[97,331,79]}
assert all(not re.search(r'ℹ fail [1-9]',logs[name]) for name in counts)
assert 'built in' in logs['root-working-build.log'] and 'built in' in logs['root-index-build.log']
summary={'status':'PASS_SAVED_ONLY_LIVE_TRAIN_INTEGRATION','attempt_path':str(run),'new_attempts_during_live_window':1,'result_sha256':sha(run/'result.json'),'episode_id':e['episodeId'],'identical_to_original_TRAIN_10_episode_and_canonical_json':True,'strategy_seal':e['planning']['strategySeal'],'history_hash':digest(e['history']),'request':{k:attempt[k] for k in ('layoutId','goalId','selector')},'episode_algorithm':'IL','policy_variant':'IL_TRAIN_REFIT','release_evidence':expected,'checkpoint_payload_read':False,'actions':actions,'frames':77,'microsteps':38,'final_outcome':{'goal_retained_and_contacted':True,'removed_mm3':3,'round_trip_path_mm':distance,'cost':cost,'return':reward},'saved_SEARCH_same_goal':row['saved_search_metrics'],'difference_from_saved_SEARCH':row['reward_difference_from_saved_search'],'inference_accounting':{'actor_forwards':2,'planning_transitions':2,'authoritative_actions':2,'optimizer_updates':0,'inventory_previews':'not instrumented in desktop worker','fresh_auditor_model_or_native_calls':0},'owned_child':receipt,'desktop_session':terminal,'canonical_tree':rootindex['tree'],'validation':{'python':33,'working_desktop_groups':counts['root-desktop-tests.log'],'index_desktop_groups':counts['root-index-tests.log'],'both_saved_build_logs_pass':True},'scope_limits':['exact saved native-history/geometry certificate corroboration; no fresh collision or tissue-geometry computation','remaining tissue hashes corroborated by exact prior envelope and unchanged-delta rules, not reconstructed from source arrays','child result stores release evidence; final outer executionProvenance transport/render state was not separately persisted','root reports frame39/frame76 visual inspection, save refusal and SELECT Execute disabled; auditor did not reopen UI','one previously observed generated TRAIN case; not new performance, generalization, heldout or clinical evidence'],'elapsed_audit_seconds':time.monotonic()-start}
(OUT/'audit.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n');(OUT/'evidence-hashes.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ('status','new_attempts_during_live_window','identical_to_original_TRAIN_10_episode_and_canonical_json','frames','microsteps','final_outcome','elapsed_audit_seconds')},indent=2))
