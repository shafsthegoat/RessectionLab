"""One exact hashing kernel on saved JSON only; no engine or replay calls."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from hashlib import sha256
import json,statistics,timeit
from resectionlab.surgical_archive_codec import decode_artifacts
from resectionlab.core import semantic_digest,freeze_json,thaw_json
base=Path('testcases/outputs/full-case-rehearsal-20261020')
native=base/'gui/vascular-observability-28962e66.ressectionlab'
raw=native.read_bytes();h=sha256(raw).hexdigest()
assert h==sha256(native.read_bytes()).hexdigest()
with ZipFile(BytesIO(raw)) as z:toolbox=decode_artifacts(json.loads(z.read('manifest.json'))['artifacts'])['surgical_toolbox']
snapshot=toolbox['branches']['full-vascular'];state=snapshot['checkpoint']['state']
payload={'scene':snapshot['sceneId'],'state':state}
if 'protectionContract' in snapshot:payload['protectionContract']=snapshot['protectionContract']['contentHash']
def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def existing():return semantic_digest(payload)
def candidate():return 'sha256:'+sha256(encoded(payload)).hexdigest()
initial=encoded(payload)
assert encoded(thaw_json(freeze_json(payload)))==initial
assert existing()==candidate()==snapshot['stateId']
measure={}
for name,fn in [('genericSemanticDigest',existing),('directBuiltinJsonDigest',candidate)]:
 values=timeit.repeat(fn,number=3,repeat=3)
 measure[name]={'secondsFor3Hashes':values,'medianMsPerHash':statistics.median(values)*1000/3}
assert encoded(payload)==initial and sha256(native.read_bytes()).hexdigest()==h
result={'scope':'Saved endpoint JSON kernel only; no engine/scene construction, cached-frame seek, replay, GUI or live-process modification.',
 'native':str(native),'nativeSha256':h,'branch':'full-vascular','actions':len(snapshot['actions']),'stateId':snapshot['stateId'],
 'payloadJsonBytes':len(initial),'flapComponentsJsonBytes':len(encoded(state['flapComponents'])),
 'exactCanonicalJsonAndStateId':True,'originalUnchanged':True,'timings':measure,
 'kernelSpeedup':measure['genericSemanticDigest']['medianMsPerHash']/measure['directBuiltinJsonDigest']['medianMsPerHash'],
 'qualification':'One alternative only. Direct encoding is justified ONLY for fresh exactbuiltin JSON-decoder output and known builtin scene/protection identities, not arbitrary public mutable engine states, MappingProxy/custom metadata or NumPy values. A complete hash comparison remains mandatory; no trusted stored-ID substitution. Not actual per-frame/action/episode timing.'}
out=base/'review/frame-state-hash-kernel-20261020.json'
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
