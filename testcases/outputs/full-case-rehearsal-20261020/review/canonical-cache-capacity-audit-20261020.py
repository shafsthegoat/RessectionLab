"""Saved JSON compression audit only; no scene/engine/replay construction."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
from resectionlab.surgical_archive_codec import decode_artifacts
from resectionlab.surgical_verified_cache import _encode_sized, _encode, STATE_SCHEMA

BASE=Path('testcases/outputs/full-case-rehearsal-20261020')
paths=sorted((BASE/'primary-v2/checkpoints').glob('*.ressectionlab'))+[BASE/'primary-v2/primary.ressectionlab', BASE/'vascular-v2/vascular.ressectionlab']
rows=[]; unique_parts={}; contents={}
for path in paths:
    raw=path.read_bytes(); digest=sha256(raw).hexdigest()
    if sha256(path.read_bytes()).hexdigest()!=digest: raise RuntimeError(f'Unstable saved native: {path}')
    if digest in contents:
        rows.append({'path':str(path),'sha256':digest,'identicalTo':contents[digest]}); continue
    contents[digest]=str(path)
    with ZipFile(BytesIO(raw)) as archive:
        toolbox=decode_artifacts(json.loads(archive.read('manifest.json'))['artifacts'])['surgical_toolbox']
    branch='vascular-rescue' if path.parent.name=='vascular-v2' else 'full-primary'
    if branch not in toolbox['branches']:
        branch=toolbox['branchId']
    snapshot=toolbox['branches'][branch]; state=snapshot['checkpoint']['state']
    fields={}; descriptor={}
    for name,value in state.items():
        compressed,size=_encode_sized(value); h=sha256(compressed).hexdigest(); unique_parts[h]=len(compressed)
        fields[name]={'gzip1Bytes':len(compressed),'jsonBytes':size,'sha256':h}
        descriptor[name]={'file':'part-'+h+'.json.gz','sha256':h,'uncompressedBytes':size}
    row={'path':str(path),'sha256':digest,'branch':branch,'actions':len(snapshot['actions']),'stateId':snapshot['stateId'],
         'sceneId':snapshot['sceneId'],'checkpointPosition':snapshot['checkpoint']['position'],
         'gzip1AllFieldBytes':sum(x['gzip1Bytes'] for x in fields.values()),
         'stateDescriptorGzip1Bytes':len(_encode({'schema':STATE_SCHEMA,'fields':descriptor})),
         'fields':fields,'removedCells':len(state['removed']), 'debrisEntries':len(state['debris']),
         'actionOperations':dict(Counter(a['operation'] for a in snapshot['actions'])),
         'zeroDurationCommands':sum(a.get('duration',0)==0 for a in snapshot['actions'])}
    rows.append(row)
primary=sorted([r for r in rows if r.get('branch')=='full-primary'],key=lambda r:r['actions'])
latest=max(primary,key=lambda r:r['actions']); n=latest['actions']+1
scenarios=[]
for names in [('removed','drillingV2'),('removed','drillingV2','debris'),tuple(latest['fields'])]:
    endpoint=sum(latest['fields'][name]['gzip1Bytes'] for name in names)
    # A scenario, not an inferred physical trajectory: charging the measured
    # endpoint field payload at every boundary and assuming all are unique.
    scenarios.append({'fields':list(names),'measuredEndpointGzip1Bytes':endpoint,'assumedBoundaries':n,
                      'hypotheticalUniqueEveryBoundaryBytes':endpoint*n})
changes={name:len({r['fields'][name]['sha256'] for r in primary if name in r['fields']}) for name in latest['fields']}
# Interpolation is explicitly only a storage planning scenario; no missing
# states are fabricated or passed to physics/cache APIs.
trapezoid=0.; variable=('removed','drillingV2','debris')
for a,b in zip(primary,primary[1:]):
    if a['actions']==b['actions']: continue
    va=sum(a['fields'][k]['gzip1Bytes'] for k in variable); vb=sum(b['fields'][k]['gzip1Bytes'] for k in variable)
    trapezoid+=(b['actions']-a['actions'])*(va+vb)/2
result={'utc':datetime.now(timezone.utc).isoformat(),'method':'Stable saved native JSON only; exact canonical gzip1 encoder per top-level field and descriptor. No engine/scene or physics calls.',
        'rows':rows,'exactObservedCheckpointUniquePartBytesAcrossAllRows':sum(unique_parts.values()),
        'primaryDistinctCheckpointCount':len(primary),'primaryObservedDistinctFieldHashes':changes,
        'scenarios':scenarios,'interpolatedThreeFieldUniqueEveryBoundaryScenarioBytes':trapezoid,
        'interpolationCoversActionPositions':[primary[0]['actions'],latest['actions']],
        'limits':'Only listed checkpoint payloads are exact. No complete per-action cache has been produced; distinct intermediate field count, canonical history/manifest bytes and completed action count remain unknown. Scenarios assume uniqueness and are not measured cache size.'}
out=BASE/'review/canonical-cache-capacity-audit-20261020.json'
with out.open('x') as stream: json.dump(result,stream,indent=2)
print(out)
for r in primary: print(r['actions'],r['gzip1AllFieldBytes'],{k:r['fields'][k]['gzip1Bytes'] for k in ['removed','drillingV2','debris','flapComponents']})
print('latest',latest['actions'],'scenarios',scenarios,'interpolation',trapezoid)
for r in rows:
 if r.get('branch')!='full-primary' and 'fields' in r: print('other',r['branch'],r['actions'],r['gzip1AllFieldBytes'])
