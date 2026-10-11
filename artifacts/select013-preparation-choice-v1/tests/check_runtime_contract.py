"""Stdlib-only serialized terminal guard checks; no imports of scientific code."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import time
HERE=Path(__file__).resolve().parents[1]

def main():
 started=time.perf_counter();source=(HERE/'comparison_contract.py').read_text()
 tree=ast.parse(source)
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='complete_result')
 scope={'ARMS':('narrow','wide')};exec(compile(ast.Module(body=[function],type_ignores=[]),'completion-guard','exec'),scope)
 good={'status':'complete_SELECT013_preparation_choice_comparison',
  'all_action_plans_sealed_before_evaluation':True,'world_equal':True,'focused_plan_context_replayed':True,
  'source_physics_equal':True,'exploration_geometry_feasible':True,
  'narrow_historical_control':dict(exact_history=True,exact_action_records=True,exact_selected_records=True),
  'held_inputs_opened':False,'EVAL_opened':False,
  **{k:0 for k in ('checkpoint_loads','model_forwards','optimizer_attempts','gradient_attempts','updates')},
  'arms':{a:{'status':'complete_replayed','independent_accepted':True,'full_history_equal':True,
       'selector_accounting':dict(complete=True,prefix_verified_count=4,intervention_applied=True),
       'exploration_geometry_feasible':True,'focus_context_replayed':True,
       'outcomes':dict(total_reward=0,target_removed_mm3=0)} for a in ('wide','narrow')}}
 results=[]
 def check(name,row,expected):
  actual=scope['complete_result'](json.loads(json.dumps(row,sort_keys=True)))
  if actual is not expected:raise AssertionError(name)
  results.append(name)
 check('sorted_JSON_complete_negative_passes',good,True)
 for key in good['narrow_historical_control']:
  bad=copy.deepcopy(good);bad['narrow_historical_control'][key]=False;check('refuse_'+key,bad,False)
 for field,value in [('complete',False),('prefix_verified_count',3),('intervention_applied',False)]:
  bad=copy.deepcopy(good);bad['arms']['wide']['selector_accounting'][field]=value;check('refuse_'+field,bad,False)
 for field in ('exploration_geometry_feasible','focus_context_replayed','independent_accepted','full_history_equal'):
  bad=copy.deepcopy(good);bad['arms']['wide'][field]=False;check('refuse_'+field,bad,False)
 bad=copy.deepcopy(good);bad['arms'].pop('wide');check('refuse_missing_arm',bad,False)
 bad=copy.deepcopy(good);bad['arms']['extra']=bad['arms']['wide'];check('refuse_extra_arm',bad,False)
 bad=copy.deepcopy(good);bad['arms']['wide']['status']='search_unresolved';check('refuse_unresolved_arm',bad,False)
 bad=copy.deepcopy(good);bad['all_action_plans_sealed_before_evaluation']=False;check('refuse_unsealed',bad,False)
 output={'status':'passed','checks':results,'checks_count':len(results),'wall_seconds':time.perf_counter()-started,
   'scope':'stdlib AST-isolated real terminal guard; no task/native/model/patient execution',
   'contract_sha256':hashlib.sha256(source.encode()).hexdigest(),
   'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with (HERE/'runtime-controls-01.json').open('x') as f:json.dump(output,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(output))
if __name__=='__main__':main()
