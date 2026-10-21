"""Saved commands only; exact-rational collinearity; no engine imports/calls."""
from fractions import Fraction
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
from resectionlab.surgical_archive_codec import decode_artifacts
BASE=Path('testcases/outputs/full-case-rehearsal-20261020')
p=BASE/'primary-v2/primary.ressectionlab'
for _ in range(3):
 raw=p.read_bytes(); h=sha256(raw).hexdigest()
 if sha256(p.read_bytes()).hexdigest()==h:break
else: raise ValueError('Unstable saved native')
with ZipFile(BytesIO(raw)) as z:toolbox=decode_artifacts(json.loads(z.read('manifest.json'))['artifacts'])['surgical_toolbox']
snapshot=toolbox['branches']['full-primary']; actions=snapshot['actions']
def can_drop(a,b,c):
 if not (a['quaternion']==b['quaternion']==c['quaternion']): return False
 points=[[Fraction(v) for v in x['position']] for x in (a,b,c)]
 ab=[v-u for u,v in zip(*points[:2])];bc=[v-u for u,v in zip(*points[1:])]
 return sum(u*v for u,v in zip(ab,bc))>0 and all(ab[i]*bc[j]==ab[j]*bc[i] for i,j in ((0,1),(0,2),(1,2)))
def compressed(path):
 kept=[]
 for pose in path:
  while len(kept)>=2 and can_drop(kept[-2],kept[-1],pose): kept.pop()
  kept.append(pose)
 return kept
rows=[]
for i,a in enumerate(actions[-96:],start=max(0,len(actions)-96)):
 if 'path' not in a:continue
 path=a['path'];short=compressed(path)
 rows.append({'actionIndex1Based':i+1,'operation':a['operation'],'originalPoses':len(path),'retainedPosesExactMonotone':len(short),
              'constantQuaternion':all(pose['quaternion']==path[0]['quaternion'] for pose in path),
              'duration':a['duration']})
r={'scope':'Saved command geometry only; exact rational represented-float collinearity; no physics admission/timing or oldaction equivalence.',
 'nativeSha256':h,'actions':len(actions),'stateId':snapshot['stateId'],'examinedLastActions':96,'paths':len(rows),
 'totalOriginalPoses':sum(x['originalPoses'] for x in rows),'totalRetainedPosesExactMonotone':sum(x['retainedPosesExactMonotone'] for x in rows),
 'constantQuaternionPaths':sum(x['constantQuaternion'] for x in rows),'rows':rows,
 'qualifications':['Retains first/last point, every reversal/zero-length leg and changing quaternion; no initial priorstate required because first point is retained.',
 'Pose reduction is not substep reduction: engine independently samples coarse segments by space+angle+duration.',
 'All path-bearing operations counted, not exclusively core proposals; actionhistory lacks director provenance per individual candidate.']}
out=BASE/'review/collinear-path-readonly-audit-20261020.json'
with out.open('x') as f:json.dump(r,f,indent=2)
print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2));print(Counter((x['operation'],x['constantQuaternion']) for x in rows))
print([(x['actionIndex1Based'],x['operation'],x['originalPoses'],x['retainedPosesExactMonotone']) for x in rows])
