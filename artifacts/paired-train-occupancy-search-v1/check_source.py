"""Small stdlib-only contract/AST controls; no factory/native/model execution."""
import ast
import copy
import json
from pathlib import Path
import runpy
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pilot_contract as c
started=time.monotonic();checks=[]
records,_=c.inputs();raw=c.condition_admission(c.CONDITIONS[0]);union=c.condition_admission(c.CONDITIONS[1])
assert raw['limits']==records['release']['limits']
assert raw['learning_protocol_hash']==records['release']['learning_protocol_hash']
assert union['limits']=={**raw['limits'],'max_policy_forwards':0,'max_optimizer_updates':0}
assert union['learning_protocol_hash']!=raw['learning_protocol_hash']
assert c.EXECUTION['max_search_commits']+c.EXECUTION['max_execution_replay_commits']==576
checks.append('Exact_raw_admission_envelope_derived_zero_budget_and_separate_protocol_transition_accounting')
release=c.release_template('0'*40,'1'*64);pending=copy.deepcopy(release)
release['status']='released_one_attempt';c.validate_release(release)
for name,mutation in (
    ('pending',lambda v:v.update(status='pending_root_release')),
    ('bool_attempt',lambda v:v.update(attempts=True)),
    ('changed_arms',lambda v:v['planned_arms'].reverse()),
    ('model_budget',lambda v:v['execution_limits'].update(policy_forwards=1)),
    ('changed_access_envelope',lambda v:v['condition_admission'][c.CONDITIONS[0]]['limits'].update(max_steps=6))):
    bad=copy.deepcopy(release);mutation(bad)
    try:c.validate_release(bad)
    except ValueError:pass
    else:raise AssertionError(name)
checks.append('Pending_release_changed_order_type_model_budget_and_horizon_refused')
result={'status':'complete_paired_TRAIN_search','arms':[{**r,'status':'complete','source_released':True,'stop_only':True} for r in c.ARMS],
    'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'private_reference_reads':0,
    'SELECT_EVAL_opened':False,'global_resource_failure':None}
assert c.complete_result(result)
for mutation in (lambda r:r['arms'][2].update(status='failed_or_unresolved'),
    lambda r:r['arms'][3].update(source_released=False),lambda r:r.update(policy_forwards=True),
    lambda r:r['arms'].pop(),lambda r:r.update(SELECT_EVAL_opened=True)):
    bad=copy.deepcopy(result);mutation(bad);assert not c.complete_result(bad)
checks.append('Eight_complete_negative_STOP_rows_valid_but_missing_failed_leaked_or_prohibited_rows_refused')
def named(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
worker=ast.parse((HERE/'search_worker.py').read_text());main=named(worker,'main')
lines={}
for n in ast.walk(main):
    if isinstance(n,ast.Call):lines.setdefault(ast.unparse(n.func),[]).append(n.lineno)
    if isinstance(n,ast.Import) and n.names[0].name=='torch':torch_line=n.lineno
assert min(lines['_require_parent_lease'])<min(lines['source_guard'])<torch_line<min(lines['execute'])
source=(HERE/'search_worker.py').read_text()
for absent in ('PatientTrainSession','_collect(','load_cohort_checkpoint','_save_checkpoint','torch.load(','Adam('):assert absent not in source
calls=[ast.unparse(n.func) for n in ast.walk(worker) if isinstance(n,ast.Call)]
assert calls.count('task.observed_greedy_search')==1 and calls.count('evaluate_native_spatial_episode')==1
assert 'replay.step' in calls and 'task.step' in calls and 'task.fresh' in calls
assert 'history_complete=False' not in source
assert "budget.complete(history_complete=True)" in source and "Unresolved arm: all eight planned rows retained" in source
checks.append('Owned_lease_before_scientific_import_no_learner_calls_existing_greedy_full_rollout_seal_fresh_replay_audit_and_strict_failure')
parent=ast.parse((HERE/'run_owned.py').read_text())
old=ast.parse((c.ROOT/'build/balanced-teacher-il64-cached-v1/run_owned.py').read_text())
assert ast.dump(next(n for n in ast.walk(parent) if isinstance(n,ast.While)))==ast.dump(next(n for n in ast.walk(old) if isinstance(n,ast.While)))
for name in ('tree_bytes','small_json'):assert ast.dump(named(parent,name))==ast.dump(named(old,name))
p_cleanup=next(n for n in named(parent,'main').body if isinstance(n,ast.Try)).finalbody
old_cleanup=next(n for n in named(old,'main').body if isinstance(n,ast.Try)).finalbody
assert [ast.dump(n) for n in p_cleanup[:3]]==[ast.dump(n) for n in old_cleanup[:3]]
assert "result['checkpoints']" not in (HERE/'run_owned.py').read_text()
checks.append('Reused_owned_supervisor_loop_cleanup_and_metadata_checks_no_checkpoint_payload_hashing')
freezer=runpy.run_path(str(HERE/'freeze-runtime.py'),run_name='source_check')
paths=freezer['source_closure']();names={str(p.relative_to(c.ROOT)) for p in paths}
assert {'src/resectionlab/public_patient_factory.py','src/resectionlab/patient_planning_admission.py',
    'src/resectionlab/native_spatial_task.py','src/resectionlab/native_spatial_evaluation.py',
    'src/resectionlab/native_resection.py','src/resectionlab/native_proposals.py'}<=names
assert not any(n in sys.modules for n in ('numpy','torch','resectionlab'))
assert not (HERE/'source-index.json').exists() and not (HERE/'release-template.json').exists()
checks.append('Static_source_closure_includes_factory_admission_geometry_with_no_scientific_import_or_freeze')
receipt={'status':'pass','groups':checks,'elapsed_seconds':time.monotonic()-started,
    'scope':'metadata and AST only; not a patient/engine/model run',
    'patient_array_reads':0,'native_steps':0,'model_forwards':0,'optimizer_updates':0,'checkpoint_payload_reads':0,
    'runtime_source_count':len(paths),'files':{name:c.sha(HERE/name) for name in ('pilot_contract.py','search_worker.py','run_owned.py','freeze-runtime.py','check_source.py','input-index.json')}}
(HERE/'source-controls.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
