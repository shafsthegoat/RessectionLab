"""Execute the actual pure completion predicate from its AST; no historical imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import signal
source=Path(__file__).with_name('comparison_contract.py')
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30-second control limit')))
signal.alarm(30)
data=source.read_bytes();tree=ast.parse(data)
nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='complete_result') or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ARMS' for t in n.targets))]
assert len(nodes)==2
namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
check=namespace['complete_result'];arms=namespace['ARMS'];assert arms==('fresh_STOP','focused_corridor')
r={'status':'complete_SELECT013_focused_corridor_comparison','all_action_plans_sealed_before_evaluation':True,'world_equal':True,'focused_plan_context_replayed':True,'source_physics_equal':True,'exploration_geometry_feasible':True,'held_inputs_opened':False,'EVAL_opened':False,'checkpoint_loads':0,'model_forwards':0,'optimizer_attempts':0,'gradient_attempts':0,'updates':0,'arms':{a:{'status':'complete_replayed','independent_accepted':True,'full_history_equal':True,'selector_accounting':{'complete':True}} for a in arms}}
sorted_result=json.loads(json.dumps(r,sort_keys=True));assert list(sorted_result['arms'])!=list(arms);assert check(sorted_result)
missing=copy.deepcopy(sorted_result);missing['arms'].pop(arms[0]);assert not check(missing)
extra=copy.deepcopy(sorted_result);extra['arms']['undeclared']=copy.deepcopy(extra['arms'][arms[0]]);assert not check(extra)
incomplete=copy.deepcopy(sorted_result);incomplete['arms'][arms[1]]['selector_accounting']['complete']=False;assert not check(incomplete)
invalid=copy.deepcopy(sorted_result);invalid['focused_plan_context_replayed']=False;assert not check(invalid)
assert source.read_bytes()==data
print(json.dumps({'status':'PASS','checks':5,'source_sha256':hashlib.sha256(data).hexdigest(),'execution_order':list(arms),'serialized_key_order':list(sorted_result['arms']),'scope':'actual pure contract predicate AST; synthetic terminal records only, no dependency imports, acquired data, native calls or model'}))
