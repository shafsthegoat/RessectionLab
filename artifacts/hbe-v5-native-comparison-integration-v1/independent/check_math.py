"""Independent generated-array parity and syntax check; no native/response reads."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import ast, hashlib, importlib.util, json, sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CAND=ROOT/'build/hbe-v5-native-comparison-preparation-v1'
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import scripts
scripts.__path__ = [str(CAND/'stage/scripts'), *scripts.__path__]
from scripts import mechanics_hbe_v5_comparison as candidate
spec=importlib.util.spec_from_file_location('independent_original_comparison',ROOT/'scripts/mechanics_hbe_v5_comparison.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
from test_mechanics_hbe_v5_comparison import generated_rows

def forbid(event,args):
 if event in ('subprocess.Popen','os.system','os.posix_spawn'): raise AssertionError('process forbidden')
 if event=='open' and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0])).resolve()
  if any(p.is_relative_to(ROOT/x) for x in ('data/mechanics','outputs/mechanics')):
   raise AssertionError('native/measured payload forbidden')
sys.addaudithook(forbid)
old_raw=(ROOT/'scripts/mechanics_hbe_v5_comparison.py').read_bytes()
assert hashlib.sha256(old_raw).hexdigest()=='a8c6a985ba5db35547cb2a1204396cbdd20af6122d15c4cf846f21532bd777b1'
new_raw=Path(candidate.__file__).read_bytes()
a,b=ast.parse(old_raw),ast.parse(new_raw)
old={n.name:n for n in a.body if isinstance(n,ast.FunctionDef)}
new={n.name:n for n in b.body if isinstance(n,ast.FunctionDef)}
def syntax(x): return ast.dump(x,include_attributes=False)
assert [syntax(n) for n in a.body if isinstance(n,ast.Assign)]==[syntax(n) for n in b.body if isinstance(n,ast.Assign)]
for name in old:
 if name not in ('_view','compare_generated_rows'): assert syntax(old[name])==syntax(new[name]),name
body=old['compare_generated_rows'].body
start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='coverage')
assert [syntax(n) for n in body[start:-1]]==[syntax(n) for n in new['_compare_views'].body[1:-1]]
body=old['_view'].body
start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='coordinates')
assert [syntax(n) for n in body[start:]]==[syntax(n) for n in new['_numerical_view'].body[3:]]
results=[]
for mode in ('baseline','coarse_sign','odd_step_spike','probe_error','criteria_contradiction','native_provenance'):
 rows=generated_rows()
 if mode=='coarse_sign': rows['tension:N8:S60:reference']['applied_force_N']=[-x for x in rows['tension:N8:S60:reference']['applied_force_N']]
 if mode=='odd_step_spike': rows['compression:N36:S120:reference']['applied_force_N'][119]=100000.
 if mode=='probe_error': rows['tension:N24:S60:reference']['probe_displacements_m'][40][37][1]=0.002
 if mode=='criteria_contradiction': rows['compression:N24:S60:reference']['criteria_max_ratio']['fixture_only']=1.2
 if mode=='native_provenance': rows['compression:N8:S60:reference']['provenance']['native_output_observed']=True
 outcomes=[]
 for module in (original,candidate):
  try: outcomes.append(('returned',module.compare_generated_rows(ROOT,deepcopy(rows))))
  except ValueError as error: outcomes.append(('refused',str(error)))
 assert outcomes[0]==outcomes[1],mode
 results.append({'case':mode,'exact_original_candidate_equal':True,'outcome':outcomes[0][0]})
print(json.dumps({'original_sha256':hashlib.sha256(old_raw).hexdigest(),'candidate_sha256':hashlib.sha256(new_raw).hexdigest(),'constant_and_math_ast_unchanged':True,'generated_cases':results,'native_or_replay_calls':0,'native_or_measured_payload_reads':0},indent=2))
