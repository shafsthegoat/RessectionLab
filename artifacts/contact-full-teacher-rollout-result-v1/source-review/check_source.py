"""Source, saved metadata and summary arithmetic only; no ML/project imports."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/goal-conditioned-policy-v1/full-teacher-rollout-v1'
OUT=Path(__file__).resolve().parent
start=time.monotonic()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
digest=lambda x:'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
checks=[]
source=read(BASE/'source-index.json');inputs=read(BASE/'input-index.json');recipe=read(BASE/'run-recipe.json')
assert sha(BASE/'source-index.json')=='f3cf38fbcb3b0d5a743272fca7e081e263f0d462a6b7101c528d28f0b56bed76'
assert sha(BASE/'run-recipe.json')=='2d2994322c492cff6a57d8fdf072abc9a2ef526667a7ac4c3c8bea493c3f5d9b'
assert sha(BASE/'input-index.json')==source['input_index']['sha256']==recipe['input_index_sha256']
assert sha(BASE/'PROTOCOL.txt')==recipe['protocol_sha256']
for name,value in source['source_files'].items():assert sha(ROOT/name)==value,name
checks.append('exact_executing_source_input_protocol_recipe_pins')
assert set(inputs['data'])=={*(f'teacher-{i:02d}.json' for i in range(24)),'refit-experiment.json','refit-result.json'}
assert inputs['caps']==source['caps']==recipe['caps']=={'hard_total_wall_seconds':45,'sampled_owned_tree_rss_bytes':1073741824,'geometry_previews':2048,'actor_forwards':48,'planning_transitions':48,'authoritative_execution_transitions':48,'threads':1,'attempts':1}
data={}
for name,row in inputs['data'].items():
    assert sha(ROOT/row['path'])==row['sha256'];data[name]=read(ROOT/row['path'])
experiment=data['refit-experiment.json']
assert digest(experiment)==source['experiment_hash']==inputs['experiment_hash']
keys=[(r['layout_id'],g) for r in experiment['family_manifest']['source_bindings'] if r['role']=='TRAIN' for g in ('surface','deep')]
teachers=[data[f'teacher-{i:02d}.json'] for i in range(24)]
assert [(t['layout_id'],t['goal_id']) for t in teachers]==keys and len(set(keys))==24
for i,t in enumerate(teachers):
    assert t['teacher_index']==i and t['role']==t['binding']['role']=='TRAIN'
    assert digest(t['strategy']['strategy'])==t['strategy']['strategySeal']
    assert all(k in t['strategy']['metrics'] for k in ('total_reward','goal_contacted_and_retained','goal_retained','removed_volume_mm3','steps','terminated'))
checkpoint=inputs['checkpoint'];fit=data['refit-result.json']
assert checkpoint['sha256']==fit['final_checkpoint']['sha256']=='5d7151397141c92ff82d8684814b9a0caed111f1809268bd448b8c1ea26d6bf7'
assert checkpoint['parameter_hash']==fit['final_checkpoint']['parameter_hash']
assert Path(fit['final_checkpoint']['path'])==ROOT/checkpoint['path']
assert checkpoint==recipe['checkpoint'] and fit['optimizer_updates']==32
checks.append('fixed24_TRAIN_source_order_saved_search_metrics_and_single_refit_identity_no_payload_read')

worker=ast.parse((BASE/'rollout_worker.py').read_text())
calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call)]
names=[n.func.id for n in calls if isinstance(n.func,ast.Name)]
assert names.count('load_contact_checkpoint')==names.count('plan_goal_mode_strategy')==names.count('export_family_strategy')==1
assert names.count('make_family_task')==1
assert not ({'contact_gradient_step','contact_imitation_loss','contact_reinforce_loss','bounded_beam_search','freeze_final_checkpoints','make_frozen_evaluation_task'} & set(names))
text=(BASE/'rollout_worker.py').read_text()
assert text.index("write(f'TRAIN-{i:02d}-plan.json'")<text.index('display,episode=export_family_strategy')
assert text.index("rows.extend(")<text.index('for row,teacher in zip(rows,teachers):')
assert "result['actor_forward_calls']>=48" in text and "result['geometry_previews']>=2048" in text
assert "if task.max_steps!=2" in text and "task.observation().fingerprint!=root_state['observation_hash']" in text
assert "if parameter_hash(policy)!=initial_weights" in text
policy=ast.parse((ROOT/'src/resectionlab/goal_mode_spatial_policy.py').read_text())
act=next(n for n in ast.walk(policy) if isinstance(n,ast.FunctionDef) and n.name=='act')
assert any(ast.unparse(d)=='torch.no_grad()' for d in act.decorator_list)
assert 'stochastic: bool=False' in ast.unparse(act).replace(' = ', '=')
checks.append('single_bounded_eval_path_uses_shared_sealed_exporter_and_retains_fixed_failure_slots')

namespace={}
summary_tree=ast.parse((BASE/'rollout_summary.py').read_text())
assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(summary_tree))
exec(compile(summary_tree,'pure_rollout_summary','exec'),namespace)
rows=[]
for i in range(24):
    # Deliberately poor completed rollouts must remain in denominators.
    same=i%3==0
    rows.append({'status':'complete','role':'TRAIN','goal_id':'surface' if i<12 else 'deep',
        'metrics':{'goal_contacted_and_retained':same,'total_reward':0. if same else -2.,'removed_volume_mm3':float(i)},
        'saved_search_metrics':{'goal_contacted_and_retained':True,'total_reward':1.},
        'actions':['STOP'] if same else ['a','b'],'saved_search_actions':['a','b'],
        'reward_difference_from_saved_search':-1. if same else -3.})
summary=namespace['summarize'](rows)
assert summary['all24']['n']==24 and summary['all24']['goal_contacts']==8
assert summary['all24']['lower_return_than_saved_search']==24 and summary['all24']['STOP_only']==8
assert summary['all24']['same_action_sequence_as_saved_search']==16
assert all(summary[g]['n']==12 for g in ('surface','deep'))
for variant in ('missing','failure','heldout'):
    bad=copy.deepcopy(rows)
    if variant=='missing':bad.pop()
    elif variant=='failure':bad[0]['status']='failed_or_capped'
    else:bad[0]['role']='SELECT'
    try:namespace['summarize'](bad)
    except ValueError:pass
    else:raise AssertionError(variant)
checks.append('pure_summary_retains_bad_outcomes_refuses_incomplete_or_nonTRAIN_pass')

monitor=(BASE/'run_rollout_owned.py').read_text()
assert monitor.index("final_elapsed=time.monotonic()-started")<monitor.index("if final_elapsed>=caps['hard_total_wall_seconds']")<monitor.index('complete=reason is None')
assert "'elapsed_seconds':final_elapsed" in monitor
assert "supervision.mkdir(exist_ok=False)" in monitor and "if output.exists():raise FileExistsError" in monitor
assert 'cleanup_owned(process,sampler,detached_seen,cleanup_actions,errors)' in monitor
assert "os.close(lease_write)" in monitor
assert not (ROOT/recipe['output']).exists()
checks.append('existing_owned_monitor_final_elapsed_and_single_attempt_refusals_intact')
receipt={'status':'PASS_SOURCE_ONLY','checks':checks,'control_groups':len(checks),'elapsed_seconds':time.monotonic()-start,'source_index_sha256':sha(BASE/'source-index.json'),'input_index_sha256':sha(BASE/'input-index.json'),'recipe_sha256':sha(BASE/'run-recipe.json'),'model_constructions':0,'checkpoint_payload_reads':0,'forwards':0,'native_tasks':0,'optimizer_steps':0,'searches':0,'imports':'stdlib only'}
(OUT/'source-controls.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2))
