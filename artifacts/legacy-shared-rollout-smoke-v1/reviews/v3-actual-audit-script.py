"""Stdlib saved-record audit: no checkpoint decoding, forward, or simulator run."""
from pathlib import Path
import hashlib, json, math
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ATTEMPT=ROOT/'build/legacy-RL256-shared-executed-v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda value:'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
read=lambda name:json.loads((ATTEMPT/name).read_text())
attempt,comparison,completion,supervisor=[read(name) for name in ('attempt.json','comparison.json','completion.json','supervisor.json')]
inventory={str(p.relative_to(ATTEMPT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ATTEMPT.rglob('*')) if p.is_file()}
assert completion['comparison_sha256']==sha(ATTEMPT/'comparison.json')
assert completion['supervision']==supervisor
assert comparison['status']==completion['status']==supervisor['status']=='complete'
assert supervisor['returncode']==0 and supervisor['worker_termination_confirmed'] is True
assert supervisor['cleanup_errors']==[] and supervisor['unresolved_worker_pid'] is None
assert not supervisor['automatic_retry'] and not supervisor['timed_out']
assert 0<supervisor['seconds']<attempt['max_wall_seconds']==90
assert 0<supervisor['sampled_peak_rss_bytes']<attempt['max_sampled_worker_rss_bytes']==1024**3
assert hashlib.sha256(json.dumps(attempt,sort_keys=True).encode()).hexdigest()==supervisor['declaration_sha256']
for relative,expected in attempt['source_sha256'].items():
    assert sha(ROOT/relative)==expected,relative
assert sha(ROOT/'build/integration-first-policy-v3/supervise_legacy_rollout.py')=='02b7b5ca7b0a93fa4bae55694a342d48dfbe8fab024b42cfdbaec3096616bdb8'
policy,search=comparison['policy'],comparison['search']
assert comparison['new_training_updates']==0 and comparison['training_updates_preexisting']==256
assert comparison['policy_forward_calls']==len(policy['decisions'])==2
assert comparison['clinical_validation'] is None
assert all(comparison[key] is True for key in ('same_source','same_environment','same_initial_observation'))
assert policy['source_hash']==search['source_hash']==comparison['checkpoint_validation']['source_hash']
assert policy['environment_contract_hash']==search['environment_contract_hash']==comparison['checkpoint_validation']['decision_model_hash']
assert policy['decisions'][0]['observation_before']==search['decisions'][0]['observation_before']=='sha256:c9458a99068bfaae90ba69062073a674c55ee31ad66d11a5c86461ef2c834eba'
validation=comparison['checkpoint_validation']
identity=policy['policy_identity']
assert identity['checkpoint_validation_receipt_sha256']==digest(validation)
assert identity['checkpoint_sha256']==validation['checkpoint_file_sha256']=='sha256:1d391665f66cdd0c1fa1db150261b853820a9d30d62fc20c99cf021cf0229fbd'
assert identity['parameter_hash']==validation['policy_parameter_hash']=='sha256:f52e14097ea6a35436eaf30f0ece5658e24987e289ca30ff6a529816545b5721'
assert identity['architecture_hash']==validation['architecture_hash']=='sha256:a3b6740ab188f01016610c566e2009c57ba422c8f9be24adca014f695ae9c917'
assert identity['training_status']=='trained_checkpoint' and validation['updates']==256
assert validation['strict_state_dict'] is True and validation['checkpoint_bytes']==394009
assert validation['safe_loader']=='torch.load(weights_only=True,map_location=cpu)'
support={(4,4,k) for k in range(1,6)}|{(5,5,1)}
target={(4,4,4),(4,4,5)}
per_method={}
for method,record in (('policy',policy),('search',search)):
    assert record['observation_track']=='synthetic_scan' and record['evidence_level']=='generated_software_development'
    rows=record['decisions']
    assert record['action_ids']==[row['action_id'] for row in rows]
    assert record['physical_history_hash']==digest([row['physical_transition'] for row in rows])
    removed=set();microstep_count=0;path_mm=0.;nonstop=0;changes=0;prior_tool=None
    for index,row in enumerate(rows):
        physical=row['physical_transition']
        assert physical['action_id']==row['action_id']
        assert physical['interaction_mode']==row['action_mode']
        assert row['terminated'] is (index==len(rows)-1)
        if index:
            assert row['observation_before']==rows[index-1]['observation_after']
            assert physical['source_state_hash']==rows[index-1]['post_model_state_hash']
        if row['action_id']=='STOP':
            assert row['action_mode']=='stop' and not physical.get('removed_indices_native',[])
            continue
        assert row['action_mode']=='aspirate' and physical['tool_id']==row['tool_id']
        assert physical['source_hash']==record['source_hash'] and physical['source_shape']==[9,9,7]
        assert physical['native_affine']==[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
        macro_removed=set(map(tuple,physical['removed_indices_native']))
        macro_contact=set(map(tuple,physical['contact_indices_native']))
        assert not removed&macro_removed and macro_removed<=support and macro_removed<=macro_contact
        micro_removed=set();micro_contact=set();previous=physical['entry_mm']
        for step in physical['microsteps']:
            assert step['tip_start_mm']==previous
            previous=step['tip_end_mm']
            delta=set(map(tuple,step['removed_indices_native']))
            assert not micro_removed&delta
            micro_removed|=delta;micro_contact|=set(map(tuple,step['contact_indices_native']))
        assert previous==physical['tip_mm'] and micro_removed==macro_removed and micro_contact==macro_contact
        assert math.isclose(physical['complete_tool_path_length_mm'],2*math.dist(physical['entry_mm'],physical['tip_mm']),abs_tol=1e-12)
        assert physical['retraction']=='reverse_identical_insertion_path_after_removal_no_in_brain_reorientation'
        removed|=macro_removed;microstep_count+=len(physical['microsteps']);path_mm+=physical['complete_tool_path_length_mm'];nonstop+=1
        changes+=prior_tool is not None and prior_tool!=row['tool_id'];prior_tool=row['tool_id']
    assert record['terminal_model_state_hash']==rows[-1]['post_model_state_hash']
    outcomes=comparison['generated_modeled_outcomes'][method]
    target_count=len(removed&target);normal_count=len(removed-target)
    reconstructed=target_count-.2*normal_count-.03*nonstop-.001*path_mm-.03*changes
    assert math.isclose(reconstructed,outcomes['total_reward'],abs_tol=1e-12)
    assert outcomes['target_removed_mm3']==target_count and outcomes['normal_removed_mm3']==normal_count
    assert outcomes['simulated_removed_volume_mm3']==len(removed) and outcomes['clinical_deficit_probability'] is None
    per_method[method]={'actions':record['action_ids'],'microsteps':microstep_count,'removed_cells':len(removed),
        'target_mm3':target_count,'normal_mm3':normal_count,'complete_tool_path_mm':path_mm,'tool_changes':changes,
        'return':outcomes['total_reward'],'history_digest_verified':True,'state_and_observation_chain_verified':True}
assert search['action_ids']==['STOP']
assert search['terminal_model_state_hash']==policy['decisions'][0]['physical_transition']['source_state_hash']
engine_config=policy['decisions'][0]['physical_transition']['decision_model_hash']
assert 'sha256:'+hashlib.sha256((engine_config+':initial').encode()).hexdigest()==search['terminal_model_state_hash']
accounting=search['search_accounting']
assert accounting['max_calls']==24 and accounting['beam_width']==2
assert accounting['model_transition_calls']==accounting['eager_transition_calls']==9
assert sum(layer['model_transition_calls'] for layer in accounting['layers'])==9
assert accounting['actor_forward_calls']==0 and accounting['negative_prefixes_pruned_by_beam']==2
assert not accounting['call_cap_reached'] and not accounting['time_cap_reached']
historical=ROOT/'artifacts/native-opening-rl-capacity-v1/independent-verification.json'
prior=json.loads(historical.read_text())
assert prior['search_gap']['search_return']==1.1
assert prior['search_gap']['fixed_final_return']==comparison['generated_modeled_outcomes']['policy']['total_reward']
assert inventory=={str(p.relative_to(ATTEMPT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ATTEMPT.rglob('*')) if p.is_file()}
report={'status':'PASS_saved_record_audit','comparison_sha256':sha(ATTEMPT/'comparison.json'),
    'attempt_inventory':inventory,'attempt_file_count_at_audit':len(inventory),'attempt_bytes_at_audit':sum(x['bytes'] for x in inventory.values()),
    'source_bindings_verified':attempt['source_sha256'],'per_method':per_method,'resources':supervisor,
    'same_source_environment_initial_observation_and_engine_state':True,'actual_actor_forwards_recorded':2,
    'new_training_updates':0,'checkpoint_validation_digest_verified':True,'search_accounting':accounting,
    'replay_scope':'Saved motion/delta/history/state-chain arithmetic verified independently. Reviewed worker completion path requires exact NativeSpatialTask replay and independent geometry checks for both arms; no new simulator replay run by this audit.',
    'historical_stronger_search':{'path':str(historical.relative_to(ROOT)),'sha256':sha(historical),'return':1.1,'gap_above_RL':.002},
    'claim':'Actual frozen RL checkpoint executes two steps through the shared generated legacy aspiration pathway. Narrow beam-2 smoke search is not a competitive baseline; no RL advantage, mixed-mode learning, patient transfer, or physical/clinical validation follows.',
    'review_activity':{'actual_checkpoint_loads':0,'model_forwards':0,'simulator_replays':0,'patient_reads':0,'network_calls':0}}
(OUT/'audit.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'audit_sha256':sha(OUT/'audit.json'),'files':len(inventory),'bytes':report['attempt_bytes_at_audit']}))
