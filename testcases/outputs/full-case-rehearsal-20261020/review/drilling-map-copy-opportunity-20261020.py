"""One detached-state copy comparison; no engine import or physics execution."""
import ast
import copy
from datetime import datetime,timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import statistics
import struct
import timeit
from zipfile import ZipFile
import numpy as np
from resectionlab.surgical_archive_codec import decode_artifacts

BASE=Path('testcases/outputs/full-case-rehearsal-20261020')
NATIVE=BASE/'primary-v2/checkpoints/primary-over-eighty-percent-target.ressectionlab'
ENGINE_SOURCE=Path('src/resectionlab/surgical_engine.py')
text=ENGINE_SOURCE.read_text(); node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_microstep_checkpoint')
frozen_source=ast.get_source_segment(text,node)
namespace={'copy':copy}; exec(frozen_source,namespace); current=namespace['_microstep_checkpoint']
# Exactly one alternative: preserve the existing memo, then guard only the
# three named flat dictionaries inside an exact built-in drillingV2 dict.
candidate_source=frozen_source.replace('    return copy.deepcopy(state, memo)', '''    drilling = state.get('drillingV2')
    if type(drilling) is dict:
        for name in ('workJ', 'temperatureC', 'thermalDoseCSeconds'):
            value = drilling.get(name)
            if id(value) in memo:
                continue
            if type(value) is dict and all(type(key) is str and type(amount) in (int, float)
                                           for key, amount in value.items()):
                memo[id(value)] = value.copy()
    return copy.deepcopy(state, memo)''')
assert candidate_source!=frozen_source
namespace2={'copy':copy}; exec(candidate_source,namespace2); candidate=namespace2['_microstep_checkpoint']
raw=NATIVE.read_bytes(); native_sha=sha256(raw).hexdigest()
assert sha256(NATIVE.read_bytes()).hexdigest()==native_sha
with ZipFile(BytesIO(raw)) as archive:
    toolbox=decode_artifacts(json.loads(archive.read('manifest.json'))['artifacts'])['surgical_toolbox']
snapshot=toolbox['branches']['full-primary']; state=snapshot['checkpoint']['state']
assert len(snapshot['actions'])==1954
encode=lambda x:json.dumps(x,separators=(',',':'),allow_nan=False).encode()

def signature(value):
    leaves=[]; aliases=[]; seen={}
    def visit(x,path):
        if isinstance(x,(dict,list)):
            if id(x) in seen: aliases.append((path,seen[id(x)])); return
            seen[id(x)]=path
            leaves.append((path,type(x).__module__,type(x).__qualname__,len(x)))
            if isinstance(x,dict):
                for key,item in x.items():
                    leaves.append((path+('key',),type(key).__module__,type(key).__qualname__,repr(key)))
                    visit(item,path+(key,))
            else:
                for i,item in enumerate(x): visit(item,path+(i,))
        elif type(x) is float: leaves.append((path,'float',struct.pack('!d',x).hex()))
        else: leaves.append((path,type(x).__module__,type(x).__qualname__,repr(x)))
    visit(value,())
    return leaves,aliases

def check(value,serializable=True):
    before=encode(value) if serializable else None
    legacy=current(value); proposed=candidate(value)
    assert signature(legacy)==signature(proposed)
    if serializable: assert before==encode(legacy)==encode(proposed)==encode(value)
    assert legacy['flapComponents'] is proposed['flapComponents'] is value['flapComponents']
    for name in ('removed','debris'):
        if value.get(name) is not value['flapComponents']:
            assert proposed[name] is not value[name]
    outer=value.get('drillingV2')
    if type(outer) is dict:
        assert proposed['drillingV2'] is not outer
        for name in ('workJ','temperatureC','thermalDoseCSeconds'):
            if isinstance(outer.get(name),dict) and outer[name] is not value['flapComponents']:
                assert proposed['drillingV2'][name] is not outer[name]
    return legacy,proposed

initial=encode(state); legacy,proposed=check(state)
for name in ('workJ','temperatureC','thermalDoseCSeconds'):
    copied=proposed['drillingV2'][name]; copied['audit-only']=-0.
    assert 'audit-only' not in state['drillingV2'][name]
assert encode(state)==initial

# Small explicit data-graph/fallback checks, with no engine/model fixtures.
shared={'0':-0.,'1':float.fromhex('0x1.0000000000001p+0')}; shared2={'2':3.}
probe={'aliasBefore':shared2,'removed':[],'debris':shared,'flapComponents':[],
       'drillingV2':{'workJ':shared,'temperatureC':shared2,'thermalDoseCSeconds':shared2}}
check(probe)
probe2={'removed':[],'debris':{},'flapComponents':shared,
        'drillingV2':{'workJ':shared,'temperatureC':{},'thermalDoseCSeconds':{}}}
check(probe2)
class DictSubclass(dict): pass
fallbacks=[]
for name,value in [('nested',{'1':[1.]}),('numpy-int',{'1':np.int64(2)}),('dict-subclass',DictSubclass({'1':2.})),('outer-subclass',None)]:
    s={'removed':[],'debris':{},'flapComponents':[],
       'drillingV2':{'workJ':{},'temperatureC':{},'thermalDoseCSeconds':{}}}
    if name=='outer-subclass': s['drillingV2']=DictSubclass({'workJ':{},'temperatureC':{},'thermalDoseCSeconds':{}})
    else: s['drillingV2']['temperatureC']=value
    check(s,serializable=name!='numpy-int'); fallbacks.append(name)

results={}
for name,function in [('frozenCurrentRemovedDebrisMemo',current),('proposedPlusThreeDrillingMaps',candidate)]:
    samples=timeit.repeat(lambda:function(state),number=5,repeat=3)
    results[name]={'secondsForFiveCopies':samples,'medianMillisecondsPerCopy':statistics.median(samples)*1000/5}
assert encode(state)==initial and sha256(ENGINE_SOURCE.read_bytes()).hexdigest()==sha256(text.encode()).hexdigest()
result={'scope':'Copy kernel on detached1954checkpoint JSON only; no engine import/construction/replay/action/episode or live-process interaction.',
        'native':str(NATIVE),'nativeSha256':native_sha,'actions':1954,'stateId':snapshot['stateId'],
        'stateJsonSha256':sha256(initial).hexdigest(),'exactJsonOrderTypesFloatBitsAliases':True,
        'independentMutableCopiesAndOriginalUnchanged':True,'existingComponentExemptionAndMemoPrecedence':True,
        'counts':{'removed':len(state['removed']),'debris':len(state['debris']),**{k:len(state['drillingV2'][k]) for k in ('workJ','temperatureC','thermalDoseCSeconds')}},
        'fallbackProbes':fallbacks,'frozenCurrentSourceSha256':sha256(frozen_source.encode()).hexdigest(),
        'proposedSourceSha256':sha256(candidate_source.encode()).hexdigest(),'frozenCurrentSource':frozen_source,'proposedSource':candidate_source,
        'timings':results,'kernelSpeedup':results['frozenCurrentRemovedDebrisMemo']['medianMillisecondsPerCopy']/results['proposedPlusThreeDrillingMaps']['medianMillisecondsPerCopy'],
        'qualification':'One alternative; guard scans/copies/deallocation included. Not per-action speedup. Only exact built-in str-key numeric-value maps inside exact built-in outerdict qualify. Custom memo-introspecting __deepcopy__ parity is not promised; JSON graph contract only.'}
out=BASE/'review/drilling-map-copy-opportunity-20261020.json'
with out.open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('frozenCurrentSource','proposedSource')},indent=2))
