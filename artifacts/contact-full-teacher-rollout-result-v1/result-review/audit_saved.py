"""Saved JSON/opaque byte audit; no project, model or native execution."""
from pathlib import Path
import hashlib, json, math, statistics, time
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'build/goal-conditioned-policy-v1/full-teacher-rollout-run-v1'
SUP=RUN.with_name(RUN.name+'.supervision')
PREP=ROOT/'build/goal-conditioned-policy-v1/full-teacher-rollout-v1'
OUT=Path(__file__).resolve().parent
start=time.monotonic();files={}
def sha(p):
    value=hashlib.sha256(p.read_bytes()).hexdigest();files[str(p.relative_to(ROOT))]={'sha256':value,'bytes':p.stat().st_size};return value
def read(p):sha(p);return json.loads(p.read_text())
def digest(v):return 'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def cells(v):
    assert all(len(p)==3 and all(type(x) is int for x in p) for p in v)
    return {tuple(p) for p in v}
def maskhash(occupied,shape):
    raw=bytearray(math.prod(shape))
    for x,y,z in occupied:
        assert 0<=x<shape[0] and 0<=y<shape[1] and 0<=z<shape[2]
        raw[(x*shape[1]+y)*shape[2]+z]=1
    header=json.dumps({'dtype':'|b1','shape':shape},sort_keys=True).encode()
    return 'sha256:'+hashlib.sha256(header+raw).hexdigest()
result=read(RUN/'result.json');receipt=read(SUP/'receipt.json');declaration=read(SUP/'declaration.json');local=read(RUN/'declaration.json')
read(SUP/'progress.json');sha(SUP/'worker.log')
assert result['status']=='complete_fixed24_TRAIN_greedy_pass' and receipt['status']=='complete' and receipt['exit_code']==0
assert receipt['result_sha256']==sha(RUN/'result.json')=='32dd1bdcbd61f9ba41dd6654aaf33c9c446cf1ba35dc222e88c4734aec0782d8'
assert receipt['stop_reason'] is None and not receipt['final_owned_pids'] and not receipt['cleanup_errors'] and receipt['worker_termination_confirmed']
assert receipt['elapsed_seconds']<45 and receipt['sampled_peak_rss_bytes']<2**30
source=read(PREP/'source-index.json');inputs=read(PREP/'input-index.json')
assert sha(PREP/'source-index.json')==declaration['source_index_sha256']=='f3cf38fbcb3b0d5a743272fca7e081e263f0d462a6b7101c528d28f0b56bed76'
assert sha(PREP/'input-index.json')==declaration['input_index_sha256']==source['input_index']['sha256']
assert source['source_files']==declaration['source_files']
for p,h in source['source_files'].items():assert sha(ROOT/p)==h,p
data={}
for k,v in inputs['data'].items():
    assert sha(ROOT/v['path'])==v['sha256'];data[k]=read(ROOT/v['path'])
exp=data['refit-experiment.json'];fit=data['refit-result.json']
assert digest(exp)==source['experiment_hash']==inputs['experiment_hash']==result['experiment_hash']==local['experiment_hash']
keys=[(r['layout_id'],g) for r in exp['family_manifest']['source_bindings'] if r['role']=='TRAIN' for g in ('surface','deep')]
assert len(keys)==24 and local['keys']==[list(k) for k in keys]
checkpoint=result['checkpoint'];assert {k:checkpoint[k] for k in inputs['checkpoint']}==inputs['checkpoint']
assert checkpoint['sha256']==sha(ROOT/checkpoint['path'])==fit['final_checkpoint']['sha256']
assert checkpoint['parameter_hash']==fit['final_checkpoint']['parameter_hash']==result['parameter_hash_after']
assert checkpoint['lineage_hash']==digest(fit['final_checkpoint']['lineage'])
assert result['partial_plans']==list(range(24)) and len(result['rows'])==24
counts={'frames':0,'microsteps':0,'actions':0,'nonstop_actions':0,'removed_cells':0}
table=[]
for i,(layout,goal) in enumerate(keys):
    row=read(RUN/f'TRAIN-{i:02d}-result.json');p=read(RUN/f'TRAIN-{i:02d}-plan.json');env=read(RUN/f'TRAIN-{i:02d}-episode.json');e=env['episode']
    teacher=data[f'teacher-{i:02d}.json'];saved=teacher['strategy'];plan=p['plan']
    assert row==result['rows'][i] and row['status']=='complete' and row['role']==teacher['role']=='TRAIN'
    assert row['index']==i and (row['layout_id'],row['goal_id'])==(layout,goal)==(teacher['layout_id'],teacher['goal_id'])
    assert digest(saved['strategy'])==row['saved_search_strategy_seal']==saved['strategySeal']
    assert row['saved_search_actions']==saved['strategy']['actions'] and row['saved_search_accounting']==teacher['search']
    assert all(v==saved['metrics'][k] for k,v in row['saved_search_metrics'].items())
    assert digest(plan)==p['seal']==row['strategy_seal']
    assert row['actions']==plan['actions']==[h['action_id'] for h in plan['history']]
    assert plan['history']==e['history']==e['metrics']['history'] and digest(e['history'])==row['history_hash']
    assert json.loads(env['episodeCanonicalJson'])=={k:v for k,v in e.items() if k!='episodeId'}
    assert 'sha256:'+hashlib.sha256(env['episodeCanonicalJson'].encode()).hexdigest()==e['episodeId']==row['episode_id']
    assert digest({k:v for k,v in e.items() if k!='episodeId'})==e['episodeId']
    assert e['schema']=='resectionlab.shared-native-contact-learning-episode.v3' and e['selector']=='IL'
    assert e['evidenceKind']=='generated_software_fixture' and e['splitRole']=='TRAIN' and not e['patientAdmission'] and not e['clinicalValidation']
    assert e['layoutId']==layout and e['publicGoal']['goalId']==goal
    assert e['sourceHash']==plan['source_hash']==row['source_hash']==e['sourceBinding']['native_source_hash']
    assert e['decisionModelHash']==plan['decision_model_hash']==row['decision_model_hash']
    assert e['caseHash']==row['case_hash']==e['sourceBinding']['display_case_hash'] and not e['sourceBinding']['private_reference_published']
    a=e['learnedAuthorship'];assert a['parameterHash']==checkpoint['parameter_hash'] and a['checkpointFileSha256']==checkpoint['sha256']
    assert a['experimentHash']==result['experiment_hash'] and a['trainingLineageHash']==checkpoint['lineage_hash'] and a['completedUpdates']==32 and a['inferenceOptimizerUpdates']==0
    contract=e['taskContract'];assert contract['checkpoint']==fit['final_checkpoint']['lineage']
    assert contract['maxSteps']==plan['max_steps']==2 and contract['proposalMode']==contract['sourceCandidateVersion']=='fixed_lattice_access_centerline_v1'
    assert contract['declaration']['role']=='TRAIN' and contract['declaration']['experiment_hash']==result['experiment_hash']
    assert e['planning']['sealedBeforeExecution'] and not e['planning']['referenceScoringPerformed']
    assert p['accounting']==row['planning'] and row['planning']['parameter_hash']==checkpoint['parameter_hash']
    assert row['planning']['actor_forward_calls']==row['planning']['model_transition_calls']==len(plan['actions'])<=2
    assert row['planning']['optimizer_updates']==0
    root=next(x for x in exp['teacher_states'] if x['layout_id']==layout and x['goal_id']==goal and x['step']==0)
    assert row['planning']['observation_ids'][0]==root['observation_hash']==e['initialObservationBinding']['observationHash']
    assert all(row['metrics'][k]==e['metrics'][k] for k in row['metrics'])
    audit=e['geometryAudit'];assert audit==row['geometry_audit'] and audit['feasible'] and not audit['failures']
    assert audit['complete_tool_checked'] and audit['frontier_checked'] and audit['source_case_hash']==e['sourceHash']
    assert audit['unsupported_source_tissue_volume_mm3']==0 and audit['action_count']==sum(a!='STOP' for a in row['actions'])
    shape=e['shape'];tools={t['tool_id']:t for t in e['tools']};objective=contract['objective'];goalcell=tuple(e['publicGoal']['nativeIndex'])
    assert digest(objective)==e['publicGoal']['objectiveHash']
    assert objective['native_index']==e['publicGoal']['nativeIndex']
    assert e['affine']==[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]  # these exact generated layouts
    assert e['publicGoal']['rasMm']==list(goalcell)
    removed=set();touched=set();probed=set();reward=0.;previous_tool=None;state=None;expected_frames=[(-1,'initial',None,None,[],[],[])]
    for j,h in enumerate(e['history']):
        if state is not None:assert state==h['source_state_hash']
        state=h['result_state_hash'];before=int(goalcell in probed and goalcell not in removed)
        assert h['goal_potential_before']==before and h['public_objective_hash']==e['publicGoal']['objectiveHash']
        if h['action_id']=='STOP':
            assert h['interaction_mode']=='stop' and not h['microsteps'] and not h['removed_indices_native'] and not h['contact_indices_native'] and not h['probe_contact_indices_native']
            assert h['source_state_hash']==state and h['reward']==0 and h['effort_and_removal_cost']==0
            expected_frames.append((j,'stop',h,None,[],[],[]));continue
        counts['nonstop_actions']+=1;tool=tools[h['tool_id']]
        assert tool['interactionMode']==h['interaction_mode'] and h['source_hash']==e['sourceHash']
        assert h['source_shape']==shape and h['native_affine']==e['affine']
        assert h['retraction']=='reverse_identical_insertion_path_after_removal_no_in_brain_reorientation'
        close(sum(x*x for x in h['axis_unit']),1.)
        close(h['insertion_distance_mm'],math.dist(h['tip_mm'],h['entry_mm']))
        close(h['complete_tool_path_length_mm'],2*h['insertion_distance_mm'])
        assert h['tool_change_count']==int(previous_tool is not None and previous_tool!=h['tool_id']);previous_tool=h['tool_id']
        rm=set();ct=set();pr=set();last=h['entry_mm']
        for m in h['microsteps']:
            counts['microsteps']+=1
            assert m['tip_start_mm']==last;last=m['tip_end_mm']
            assert m['active_radius_mm']==tool['tip_radius_mm']
            mr=cells(m['removed_indices_native']);mc=cells(m['contact_indices_native'])
            mp=mc-removed if h['interaction_mode']=='probe' else set()
            if h['interaction_mode']=='probe':assert not mr
            assert not mr & removed;removed|=mr;touched|=mc;probed|=mp;rm|=mr;ct|=mc;pr|=mp
            expected_frames.append((j,'insertion',h,m['tip_end_mm'],m['removed_indices_native'],m['contact_indices_native'],[list(x) for x in sorted(mp)]))
        assert last==h['tip_mm'] and rm==cells(h['removed_indices_native']) and ct==cells(h['contact_indices_native']) and pr==cells(h['probe_contact_indices_native'])
        for m in reversed(h['microsteps']):expected_frames.append((j,'withdrawal',h,m['tip_start_mm'],[],[],[]))
        close(h['removed_volume_mm3'],len(rm));close(h['removal_cost_volume_mm3'],len(rm))
        after=int(goalcell in probed and goalcell not in removed);assert h['goal_potential_after']==after
        costs=objective['costs'];cost=len(rm)*costs['normal_per_mm3']+costs['action_cost']+h['complete_tool_path_length_mm']*costs['motion_per_mm']+h['tool_change_count']*costs['tool_change_cost']
        close(h['effort_and_removal_cost'],cost);close(h['reward'],objective['completion_value']*(after-before)-cost);reward+=h['reward']
    assert state==e['nativeEngineFinalStateId'];close(reward,row['metrics']['total_reward'])
    assert cells(e['finalRemovedIndicesNative'])==removed
    close(len(removed),row['metrics']['removed_volume_mm3']);close(audit['claimed_source_tissue_volume_mm3'],len(removed));close(audit['contained_source_tissue_volume_mm3'],len(removed))
    assert row['metrics']['goal_retained']==(goalcell not in removed)
    assert row['metrics']['goal_contacted_and_retained']==(goalcell in probed and goalcell not in removed)
    assert len(expected_frames)==len(e['replayFrames']);fr_removed=set();fr_touch=set();fr_probe=set();previous=None;prev_remaining=None
    for n,(f,x) in enumerate(zip(e['replayFrames'],expected_frames)):
        j,phase,h,tip,rm,ct,pr=x
        assert f['frameIndex']==n and f['actionIndex']==j and f['phase']==phase and f['tipRasMm']==tip
        assert f['toolId']==(None if h is None else h.get('tool_id')) and f['mode']==(None if h is None else h['interaction_mode']) and f['axis']==(None if h is None else h.get('axis_unit'))
        assert cells(f['removedIndicesNative'])==cells(rm) and cells(f['contactIndicesNative'])==cells(ct) and cells(f['probeContactIndicesNative'])==cells(pr)
        fr_removed|=cells(rm);fr_touch|=cells(ct);fr_probe|=cells(pr)
        assert f['cavityHash']==maskhash(fr_removed,shape) and f['contactHash']==maskhash(fr_touch,shape) and f['probeContactHash']==maskhash(fr_probe,shape)
        if prev_remaining is not None and not rm:assert f['remainingHash']==prev_remaining
        prev_remaining=f['remainingHash']
        identity=digest({k:v for k,v in f.items() if k not in ('removedIndicesNative','contactIndicesNative','probeContactIndicesNative','stateBefore','stateAfter')})
        assert f['stateAfter']==identity and f['stateBefore']==(identity if n==0 else previous);previous=identity
    assert e['initialStateId']==e['replayFrames'][0]['stateAfter'] and e['finalStateId']==previous
    assert e['replayFrames'][0]['remainingHash']==e['sourceBinding']['support_hash']
    assert fr_removed==removed and fr_touch==touched and fr_probe==probed
    counts['frames']+=len(expected_frames);counts['actions']+=len(row['actions']);counts['removed_cells']+=len(removed)
    close(row['reward_difference_from_saved_search'],reward-row['saved_search_metrics']['total_reward'])
    table.append({'index':i,'layout_id':layout,'goal_id':goal,'actions':row['actions'],'modes':[h['interaction_mode'] for h in e['history']],'contact':row['metrics']['goal_contacted_and_retained'],'return':reward,'saved_search_contact':row['saved_search_metrics']['goal_contacted_and_retained'],'saved_search_return':row['saved_search_metrics']['total_reward'],'removed_mm3':len(removed),'saved_search_removed_mm3':row['saved_search_metrics']['removed_volume_mm3'],'same_sequence':row['actions']==row['saved_search_actions'],'episode_id':e['episodeId'],'seal':p['seal'],'frames':len(expected_frames)})

summaries={}
for name,rows in [('all24',result['rows']),('surface',[r for r in result['rows'] if r['goal_id']=='surface']),('deep',[r for r in result['rows'] if r['goal_id']=='deep'])]:
    calc={'n':len(rows),'goal_contacts':sum(r['metrics']['goal_contacted_and_retained'] for r in rows),'saved_search_goal_contacts':sum(r['saved_search_metrics']['goal_contacted_and_retained'] for r in rows),'mean_return':statistics.mean(r['metrics']['total_reward'] for r in rows),'saved_search_mean_return':statistics.mean(r['saved_search_metrics']['total_reward'] for r in rows),'STOP_only':sum(r['actions']==['STOP'] for r in rows),'same_action_sequence_as_saved_search':sum(r['actions']==r['saved_search_actions'] for r in rows),'lower_return_than_saved_search':sum(r['reward_difference_from_saved_search']<0 for r in rows),'same_return_as_saved_search':sum(r['reward_difference_from_saved_search']==0 for r in rows),'higher_return_than_saved_search':sum(r['reward_difference_from_saved_search']>0 for r in rows),'mean_removed_volume_mm3':statistics.mean(r['metrics']['removed_volume_mm3'] for r in rows)}
    for k,v in calc.items():close(v,result['summary'][name][k])
    summaries[name]=calc
expected={'checkpoint_loads':1,'actor_forward_calls':30,'geometry_previews':704,'optimizer_updates':0,'new_search_calls':0,'new_teacher_calls':0,'patient_reads':0,'SELECT_or_MEASUREMENT_reads':0,'planning_transition_calls':30,'authoritative_execution_transition_calls':30,'observed_native_transition_calls':60}
assert all(result[k]==v for k,v in expected.items()) and counts['actions']==30 and counts['nonstop_actions']==12
costs=result['costs'];assert sum(r.get('geometry_preview_calls',0) for r in costs.values())==704
assert costs['greedy_planning']['actor_forward_calls']==costs['greedy_planning']['native_transition_calls']==30
assert costs['authoritative_native_execution_and_v3_replay']['native_transition_calls']==30
assert costs['authoritative_native_execution_and_v3_replay']['geometry_audit_calls']==24
assert not result['clinical_validation'] and not result['checkpoint_selection'] and not result['desktop_publication']
assert result['wall_seconds']<receipt['elapsed_seconds']
names={'result.json','declaration.json'}|{f'TRAIN-{i:02d}-{suffix}.json' for i in range(24) for suffix in ('result','episode','plan')}
assert {p.name for p in RUN.iterdir() if p.is_file()}==names
assert {p.name for p in SUP.iterdir() if p.is_file()}=={'receipt.json','declaration.json','progress.json','worker.log'}
audit={'status':'PASS_SAVED_ONLY','summary':summaries,'counters':expected,'history_counts':counts,'saved_search_transition_calls':sum(r['saved_search_accounting']['model_transition_calls'] for r in result['rows']),'saved_search_planning_seconds':sum(r['saved_search_accounting']['planning_seconds'] for r in result['rows']),'parent_seconds':receipt['elapsed_seconds'],'worker_seconds':result['wall_seconds'],'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'parameter_hash':checkpoint['parameter_hash'],'checkpoint_sha256':checkpoint['sha256'],'run_head':declaration['head'],'result_sha256':sha(RUN/'result.json'),'rows':table,'limits':'No rerun: verifies saved engine/audit receipts, seals, exact frame/microstep correspondence, independent delta-mask hashes and reward arithmetic. Does not recompute collision/containment geometry or remaining tissue masks from source arrays. No project/ML import, model/native task, checkpoint decode or held-out access.','elapsed_seconds':time.monotonic()-start}
(OUT/'audit.json').write_text(json.dumps(audit,sort_keys=True,indent=2)+'\n')
(OUT/'evidence-hashes.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='rows'},indent=2))
