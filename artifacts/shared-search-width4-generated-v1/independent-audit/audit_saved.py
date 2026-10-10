"""Stdlib audit of the completed width-4 generated search; no execution."""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BASE=ROOT/'build/integration-first-search-width4-v1';ATT=BASE/'attempt-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda v:'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
read=lambda n:json.loads((ATT/n).read_text())
a,c,r,s=[read(n) for n in ('attempt.json','saved-comparison.json','search.json','supervisor.json')]
old_path=ROOT/'build/legacy-RL256-shared-executed-v3/comparison.json';old=json.loads(old_path.read_text())
assert c['v3_comparison_sha256']==sha(old_path)=='07d61374b4b97271e1f5c6f5e6eab395725afdcef953acdd2799c5a178e917c8'
assert sha(BASE/'run.py')==a['runner_sha256']==s['declaration_sha256']=='04cc89d9e84788c6298c0cdc38cb08d49c24c8eadf751826fb00c40422c604a6'
for path,expected in a['source_sha256'].items():assert sha(ROOT/path)==expected,path
assert s['status']==r['status']==c['status']=='complete' and s['returncode']==0
assert s['worker_termination_confirmed'] and s['unresolved_worker_pid'] is None and not s['cleanup_errors']
assert not s['automatic_retry'] and not s['timed_out']
assert 0<s['seconds']<a['max_wall_seconds']==30 and 0<s['sampled_peak_rss_bytes']<a['max_sampled_worker_rss_bytes']==1024**3
assert r['checkpoint_loads']==r['policy_forward_calls']==r['new_training_updates']==0
record=r['search'];rows=record['decisions'];account=record['search_accounting']
assert record['method']=='SEARCH' and record['policy_identity']['checkpoint_sha256'] is None
assert record['source_hash']==old['policy']['source_hash'] and record['environment_contract_hash']==old['policy']['environment_contract_hash']
assert rows[0]['observation_before']==old['policy']['decisions'][0]['observation_before']
assert rows[0]['physical_transition']['source_state_hash']==old['policy']['decisions'][0]['physical_transition']['source_state_hash']
assert record['action_ids']==[row['action_id'] for row in rows] and len(rows)==2
assert record['physical_history_hash']==digest([row['physical_transition'] for row in rows])
removed=set();path_mm=0.;microsteps=0
for i,row in enumerate(rows):
    ph=row['physical_transition'];assert row['action_mode']==ph['interaction_mode']=='aspirate'
    assert ph['action_id']==row['action_id'] and row['terminated'] is (i==1)
    if i:
        assert row['observation_before']==rows[i-1]['observation_after']
        assert ph['source_state_hash']==rows[i-1]['post_model_state_hash']
    macro=set(map(tuple,ph['removed_indices_native']));mr=set();mc=set();previous=ph['entry_mm']
    for step in ph['microsteps']:
        assert step['tip_start_mm']==previous;previous=step['tip_end_mm']
        delta=set(map(tuple,step['removed_indices_native']));assert not mr&delta;mr|=delta
        mc|=set(map(tuple,step['contact_indices_native']))
    assert previous==ph['tip_mm'] and mr==macro and mc==set(map(tuple,ph['contact_indices_native']))
    assert not removed&macro;removed|=macro
    assert math.isclose(ph['complete_tool_path_length_mm'],2*math.dist(ph['entry_mm'],ph['tip_mm']),abs_tol=1e-12)
    path_mm+=ph['complete_tool_path_length_mm'];microsteps+=len(ph['microsteps'])
assert record['terminal_model_state_hash']==rows[-1]['post_model_state_hash']
assert removed=={(4,4,k) for k in range(1,6)}|{(5,5,1)}
outcome=r['generated_modeled_outcome'];assert outcome['target_removed_mm3']==2 and outcome['normal_removed_mm3']==4 and outcome['simulated_removed_volume_mm3']==6
assert path_mm==10. and math.isclose(2-.2*4-.03*2-.001*path_mm-.03,outcome['total_reward'],abs_tol=1e-12)
assert outcome['total_reward']==c['width4']['modeled_return']==1.1
assert account['model_transition_calls']==account['eager_transition_calls']==15 and account['max_calls']==24 and account['beam_width']==4
assert sum(layer['model_transition_calls'] for layer in account['layers'])==15
assert account['beam_pruned_prefixes']==account['actor_forward_calls']==0
assert not account['call_cap_reached'] and not account['time_cap_reached']
assert account['objective_source']=='observed_scan_estimator_only'
assert c['width2']['action_ids']==old['search']['action_ids']==['STOP'] and c['width2']['model_transition_calls']==9
assert c['saved_rl256_modeled_return']==old['generated_modeled_outcomes']['policy']['total_reward']
assert r['clinical_validation'] is None and outcome['clinical_deficit_probability'] is None
inventory={str(p.relative_to(ATT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ATT.rglob('*')) if p.is_file()}
receipt={'status':'PASS_saved_only','files':inventory,'file_count':len(inventory),'total_bytes':sum(v['bytes'] for v in inventory.values()),
    'runner_sha256':sha(BASE/'run.py'),'same_source_environment_observation_and_initial_engine_state':True,
    'history_digest_state_chain_microstep_unions_and_return_arithmetic_verified':True,'microsteps':microsteps,
    'outcome':outcome,'complete_tool_path_mm':10.,'accounting':account,'resources':s,
    'counterfactual_claim_limit':'Saved results support width2 pruning as the cause of this smoke STOP: width4 retained all four root prefixes and found1.100 using15calls. No general search/learning performance, patient, private-outcome or clinical claim. Prior width2 and RL records remain unchanged.',
    'replay_scope':'Reviewed worker source requires exact shared-task replay and independent geometry pass before completion. This audit checks saved records only and performs no new replay.',
    'new_audit_checkpoint_loads':0,'new_audit_model_forwards':0,'new_audit_simulator_runs':0}
(OUT/'audit.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'sha256':sha(OUT/'audit.json'),'files':len(inventory),'bytes':receipt['total_bytes']}))
