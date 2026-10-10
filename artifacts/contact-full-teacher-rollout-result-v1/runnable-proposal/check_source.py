"""Bounded stdlib source/index and final-metric summary controls; no ML/native."""
import ast,hashlib,json,runpy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).parent;start=time.perf_counter();checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((HERE/'source-index.json').read_text());inputs=json.loads((HERE/'input-index.json').read_text())
for relative,expected in index['source_files'].items():assert sha(ROOT/relative)==expected,relative
assert sha(HERE/'input-index.json')==index['input_index']['sha256'];checks.append('exact_runtime_source_and_input_index_pins')
assert len(inputs['data'])==26
for name,row in inputs['data'].items():
    assert name in ('refit-experiment.json','refit-result.json') or name.startswith('teacher-')
    assert sha(ROOT/row['path'])==row['sha256']
assert inputs['checkpoint']['sha256']=='5d7151397141c92ff82d8684814b9a0caed111f1809268bd448b8c1ea26d6bf7'
assert inputs['experiment_hash']=='sha256:fd211e00dbe6dd1c9ed50a58d6c3b9f9f8896a0acd3a7815b2c52e4a8dc1014f'
checks.append('26_exact_TRAIN_reference_inputs_one_fixed_final_artifact_metadata')
worker=(HERE/'rollout_worker.py').read_text();monitor=(HERE/'run_rollout_owned.py').read_text()
for source in (worker,monitor):ast.parse(source)
assert 'observed_beam_search(' not in worker and '.backward(' not in worker and 'torch.optim' not in worker
assert "keys=experiment.keys('TRAIN')" in worker and 'task.max_steps!=2' in worker
assert 'plan_goal_mode_strategy(policy,task' in worker and 'export_family_strategy(experiment,task' in worker
assert "'hard_total_wall_seconds':45." in monitor and "if final_elapsed>=caps['hard_total_wall_seconds']" in monitor
assert 'cleanup_owned' in monitor and 'FastDarwinSampler' in monitor and '_require_parent_lease()' in worker
assert "raise FileExistsError('One rollout attempt only; no retry or overwrite')" in monitor
checks.append('same_shared_planner_exporter_fixed_role_horizon_and_owned45s_caps')
summary=runpy.run_path(str(HERE/'rollout_summary.py'))['summarize'];rows=[]
for i in range(24):
    earned=.5 if i%3==0 else 0.;teacher=.5;actions=['ASPIRATE','PROBE'] if earned else ['STOP']
    rows.append({'role':'TRAIN','status':'complete','goal_id':'surface' if i%2==0 else 'deep',
        'metrics':{'total_reward':earned,'goal_contacted_and_retained':bool(earned),'removed_volume_mm3':2. if earned else 0.},
        'saved_search_metrics':{'total_reward':teacher,'goal_contacted_and_retained':True},
        'actions':actions,'saved_search_actions':['ASPIRATE','PROBE'],'reward_difference_from_saved_search':earned-teacher})
r=summary(rows);assert r['all24']['goal_contacts']==8 and r['all24']['STOP_only']==16
assert r['all24']['mean_return']==1/6 and r['all24']['saved_search_mean_return']==.5
assert r['all24']['lower_return_than_saved_search']==16 and r['surface']['n']==r['deep']['n']==12
assert r['all24']['same_action_sequence_as_saved_search']==8;checks.append('actual_final_metric_summary_preserves_negative_outcomes')
for bad in (rows[:-1],[{**r,'role':'SELECT'} for r in rows],[{**r,'status':'failed_or_capped'} for r in rows]):
    try:summary(bad)
    except ValueError:pass
    else:raise AssertionError('Incomplete or non-TRAIN final metrics admitted')
checks.append('incomplete_failed_or_nonTRAIN_summary_refused')
assert not (ROOT/'build/goal-conditioned-policy-v1/full-teacher-rollout-run-v1').exists()
receipt={'status':'PASS','checks':checks,'count':len(checks),'wall_seconds':time.perf_counter()-start,
    'project_or_ML_imports':0,'model_forwards':0,'checkpoint_decodes':0,'native_steps':0,'optimizer_updates':0,
    'source_index_sha256':sha(HERE/'source-index.json'),'control_sha256':sha(Path(__file__))}
with (HERE/'source-controls.json').open('x') as file:json.dump(receipt,file,indent=2);file.write('\n')
print(json.dumps(receipt,indent=2))
