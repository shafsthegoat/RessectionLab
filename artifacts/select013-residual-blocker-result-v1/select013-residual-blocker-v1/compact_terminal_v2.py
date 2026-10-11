"""Losslessly remove repeated preparation records and the redundant full projection."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];POST=HERE/'postrun'
def pin(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def save(p,x):
 with p.open('x') as f:json.dump(x,f,sort_keys=True,separators=(',',':'));f.write('\n')
x=json.loads((POST/'causal-evidence.json').read_text())
for row in x['branches']:
 if row['full_native_history_encoding']['append']!=[row['prep']]:raise ValueError('Native prep cannot losslessly deduplicate')
 row['full_native_history_encoding']={'prefix':'original_native_history','append_record':'this_branch.prep'}
 nominal=row.pop('nominal_prep');base=row['prep']
 patch={k:v for k,v in nominal.items() if k not in base or base[k]!=v};removed=[k for k in base if k not in nominal]
 rebuilt={k:v for k,v in base.items() if k not in removed};rebuilt.update(patch)
 if rebuilt!=nominal:raise ValueError('Nominal score cannot losslessly reconstruct')
 row['nominal_prep_encoding']={'base_record':'this_branch.prep','remove_keys':removed,'replace_or_add':patch}
x['format']='saved-residual-diagnostic-prefix-and-prep-dedup-v2'
payload=POST/'causal-evidence-v2.json';save(payload,x)
old=json.loads((POST/'lean-archive-list.json').read_text())
omit={str((POST/'causal-evidence.json').relative_to(ROOT)),str((POST/'terminal-projection.json').relative_to(ROOT))}
files=[r for r in old['files'] if r['path'] not in omit]+[pin(payload),pin(HERE/'compact_terminal_v2.py')]
size=sum(r['bytes'] for r in files)
value={'files':files,'bytes':size,'file_count':len(files),'requested_ceiling_bytes':2*1024**2,
 'within_requested_ceiling':size<=2*1024**2,'previous_oversize_list':pin(POST/'lean-archive-list.json'),
 'history_encoding':'original47 native history once; each branch exact prep record once; native tail appends prep, nominal score reconstructs prep plus explicit patch',
 'no_history_loss':True,'original_runtime_untouched':True}
archive=POST/'lean-archive-list-v2.json';save(archive,value)
print(json.dumps({'archive':pin(archive),'payload':pin(payload),'bytes':size,'files':len(files),'within_2MiB':size<=2*1024**2}))
