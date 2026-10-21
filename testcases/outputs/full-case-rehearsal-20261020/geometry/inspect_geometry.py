import sys,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import load_protection,GuardConfig
from resectionlab.core import array_digest,thaw_json
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');p=Path('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab')
t=time.perf_counter();case,artifacts,w=load_workspace(p);scene=load_environment(p)
protection=load_protection(scene,'testcases/environments/meningioma-sub-06/rl-protection.json',strict=True)
engine=SurgicalEngine(scene,protection=protection,guard_config=GuardConfig())
def stats(m):
 ids=np.flatnonzero(m); ijk=np.column_stack(np.unravel_index(ids,m.shape)); xyz=scene.points(ids)
 return {'cells':len(ids),'volumeMm3':len(ids)*scene.voxel_mm3,'boundsIndex':[ijk.min(0).tolist(),ijk.max(0).tolist()] if len(ids) else None,'boundsRAS':[xyz.min(0).tolist(),xyz.max(0).tolist()] if len(ids) else None,'centroidRAS':xyz.mean(0).tolist() if len(ids) else None}
masks={'brain':case.brain_mask,**case.compartments}; layers=thaw_json(case.metadata).get('anatomy_layers',{})
result={'source':{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'caseHash':case.semantic_hash,'planningHash':case.planning_hash,'sceneId':scene.identity,'labelArrayHash':array_digest(scene.labels),'mriHash':array_digest(case.mri)},'environment':scene.descriptor(),'annotations':{name:{**stats(m),'arrayHash':array_digest(m),'metadata':layers.get(name)} for name,m in masks.items()},'owned':{r['name']:{**r,**stats(scene.labels==r['id'])} for r in scene.regions},'overlaps':{a+' / '+b:int(np.count_nonzero(ma&mb)) for a,ma in masks.items() for b,mb in masks.items() if a<b},'properties':scene.properties,'guard':engine.guard_report(),'protectionBounds':{a.structure_id:stats(np.isin(np.arange(scene.labels.size).reshape(scene.shape),a.cells)) for a in protection.annotations},'artifactKeys':list(artifacts),'workspace':None if w is None else {'originals':len(w['originals']),'sources':len(w['sources']),'sessionHash':w['sessionHash']},'toolbox':{k:v for k,v in artifacts.get('surgical_toolbox',{}).items() if k!='branches'},'priorBranches':{k:{'keys':list(v),'actions':len(v.get('actions',[])),'stateRemoved':len(v.get('state',{}).get('removed',[])),'guard':v.get('protectionContract')} for k,v in artifacts.get('surgical_toolbox',{}).get('branches',{}).items()},'freshMetrics':engine.metrics(),'loadWallSeconds':time.perf_counter()-t}
(out/'source_geometry.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['source','owned','priorBranches','freshMetrics','loadWallSeconds']},indent=2))
