"""Metadata/source-only preparation check; no scientific imports or payloads."""
import ast
import json
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import axis_contract as c
start=time.perf_counter();names=('axis_contract.py','axis_worker.py','run_owned.py','freeze-runtime.py')
for name in names:ast.parse((HERE/name).read_text())
records,refs,target=c.inputs()
assert len(refs)==12 and c.EXTRA_AXIS==(-1,4) and c.CLEARANCE_VOXEL==(121,44,72)
r=c.release_template('a'*40,'b'*64)
try:c.validate_release(r)
except ValueError:pass
else:raise AssertionError('Pending release admitted')
c.validate_release({**r,'status':'released_one_attempt'})
for field,value in (('worker_seconds',61),('attempts',2)):
    try:c.validate_release({**r,'status':'released_one_attempt',field:value})
    except ValueError:pass
    else:raise AssertionError('Changed release admitted')
assert c.EXECUTION['max_native_transition_calls']==2*(1+2+3)
assert c.EXECUTION['max_diagnostic_previews']==2
assert max(range(3),key=lambda i:[0.,-.1,0.][i])==0
raw=records['union_inventory']['ledger'];changed=json.loads(c.canonical(raw))
changed[0]['action_id']='new-source-id';changed[0]['proposal_id']='new-source-id'
assert c.without_ids(raw)==c.without_ids(changed)
changed[0]['feasible']=not changed[0]['feasible'];assert c.without_ids(raw)!=c.without_ids(changed)
assert not {'torch','numpy','resectionlab'} & sys.modules.keys()
result={'status':'PASS','groups':['four AST parses','twelve metadata joins and exact declared axis/voxel',
    'pending/widened release refusal','conditional12 transition/2 diagnostic bound and STOP ties',
    'identity-only row changes permitted; changed feasibility refused','no scientific imports'],
    'seconds':time.perf_counter()-start,'source_hashes':{n:c.sha(HERE/n) for n in names},
    'input_index_sha256':c.INPUT_SHA,'patient_array_reads':0,'native_calls':0,'model_calls':0}
(HERE/'source-controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
