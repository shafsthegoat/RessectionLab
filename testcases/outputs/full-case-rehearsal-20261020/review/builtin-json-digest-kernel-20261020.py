"""One guarded digest candidate on saved JSON/cornercases; no physics or GUI."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from hashlib import sha256
from types import MappingProxyType
import json,math,statistics,timeit
import numpy as np
from resectionlab.surgical_archive_codec import decode_artifacts
from resectionlab.core import freeze_json,thaw_json
B=Path('testcases/outputs/full-case-rehearsal-20261020');V=B/'review'
p=B/'gui/vascular-observability-28962e66.ressectionlab';raw=p.read_bytes();h=sha256(raw).hexdigest()
with ZipFile(BytesIO(raw)) as z:t=decode_artifacts(json.loads(z.read('manifest.json'))['artifacts'])['surgical_toolbox']
s=t['branches']['full-vascular'];payload={'scene':s['sceneId'],'state':s['checkpoint']['state']}
if 'protectionContract' in s:payload['protectionContract']=s['protectionContract']['contentHash']

def legacy(value):
    text=json.dumps(thaw_json(freeze_json(value)),sort_keys=True,separators=(',',':'),allow_nan=False)
    return 'sha256:'+sha256(text.encode('utf-8')).hexdigest()

def exact_builtin_json(value,depth=0):
    if depth>=32:return False
    kind=type(value)
    if kind is int or kind is str or kind is bool or value is None:return True
    if kind is float:return math.isfinite(value)
    if kind is list:return all(exact_builtin_json(item,depth+1) for item in value)
    if kind is dict:return all(type(key) is str and exact_builtin_json(item,depth+1) for key,item in value.items())
    return False

def candidate(value):
    try:direct=exact_builtin_json(value)
    except RecursionError:direct=False
    text=json.dumps(value if direct else thaw_json(freeze_json(value)),sort_keys=True,separators=(',',':'),allow_nan=False)
    return 'sha256:'+sha256(text.encode('utf-8')).hexdigest()

class DictSubclass(dict):pass
class ListSubclass(list):pass
class IntSubclass(int):pass
class FloatSubclass(float):pass
class StrSubclass(str):pass
shared={'leaf':[1,-0.,math.nextafter(0.,1.)]}
deep=0
for _ in range(70):deep=[deep]
cycle=[];cycle.append(cycle)
cases=[('payload',payload),('finiteedges',{'unicode':'é😀\u0000','signedZero':-0.,'tiny':math.nextafter(0.,1.),'largeFloat':1.7976931348623157e308,'bigInt':10**200,'bool':True,'none':None}),
 ('aliases',{'a':shared,'b':shared}),('tuple',{'tuple':(1,2.)}),('mappingproxy',MappingProxyType({'a':1})),
 ('numpy-scalars',{'a':np.int64(2),'b':np.float64(-0.)}),('dict-subclass',DictSubclass({'a':1})),('list-subclass',ListSubclass([1,2])),
 ('scalar-subclasses',{'a':IntSubclass(2),'b':FloatSubclass(.5),'c':StrSubclass('é')}),('key-subclass',{StrSubclass('a'):1}),
 ('non-str-key',{1:2}),('nan',float('nan')),('inf',float('inf')),('numpy-array',np.array([1.])),('deep',deep),('cycle',cycle),
 ('unsupported',object()),('too-large-int',10**5000)]
rows=[]
for name,value in cases:
 outcomes=[]
 for fn in (legacy,candidate):
  try:outcomes.append(('value',fn(value)))
  except (ValueError,TypeError,RecursionError) as error:outcomes.append(('error',type(error).__name__,str(error)))
 assert outcomes[0]==outcomes[1],(name,outcomes)
 rows.append({'case':name,'exactSameHashOrException':True,'directPath':exact_builtin_json(value),'outcome':outcomes[0]})
assert legacy(payload)==candidate(payload)==s['stateId']
initial=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
measure={}
for name,fn in [('frozenLegacyFreezeThawDigest',legacy),('guardedBuiltinWalkDigest',candidate)]:
 values=timeit.repeat(lambda:fn(payload),number=3,repeat=3)
 measure[name]={'secondsFor3Hashes':values,'medianMsPerHash':statistics.median(values)*1000/3}
assert initial==json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False) and sha256(p.read_bytes()).hexdigest()==h
out={'scope':'ONE guarded hash kernel on saved73actionendpoint and18cornercases; no engine construction/replay/cachedseek/GUI or production edit.',
 'nativeSha256':h,'stateId':s['stateId'],'payloadJsonBytes':len(initial.encode()),'parityCases':rows,'timings':measure,
 'speedup':measure['frozenLegacyFreezeThawDigest']['medianMsPerHash']/measure['guardedBuiltinWalkDigest']['medianMsPerHash'],
 'contract':'Exact builtin JSON graph only, fresh validation eachcall. Depth>=32 and guardRecursionError fall back to literal legacy normalization; no cachedIDs or SHA removal. Requires stable input during call, as existing mutable metadata hashing does. No adversarial concurrent mutation equivalence claim. Full source/cache fingerprints would change if production adopted and cannot be bypassed.'}
with (V/'builtin-json-digest-kernel-20261020.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps({'cases':len(rows),'timings':measure,'speedup':out['speedup'],'stateId':out['stateId']},indent=2))
