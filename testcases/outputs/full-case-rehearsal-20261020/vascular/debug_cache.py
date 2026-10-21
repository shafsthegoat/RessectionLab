import numpy as np,tempfile,json
from pathlib import Path
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache,implementation_fingerprint,_encode
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.core import semantic_digest
l=np.zeros((24,24,20),np.uint16);l[8,7:10,8]=1;e=SurgicalEngine(SurgicalScene(l,np.eye(4),[{'id':1,'name':'fixture','material':'vessel'}],{},{'tissuePhysicsV2':default_profile()}))
for a in [{'slot':'left','operation':'exchange','instrument':'micro_scissors','duration':.2},{'slot':'left','operation':'place','duration':.2,'pose':{'position':[8.,8.,4.],'quaternion':[0.,0.,0.,1.]},'parameters':{'radiusMm':1.5,'jawOpeningMm':0.}},{'slot':'left','operation':'apply','duration':1.,'pose':{'position':[8.,8.,7.],'quaternion':[0.,0.,0.,1.]}},{'slot':'left','operation':'inspect','duration':.4}]:e.step(a)
r=e.snapshot()
with tempfile.TemporaryDirectory()as d:
 c=VerifiedSurgicalCache(Path(d)/'c');out=e.fresh();w=c._begin_replay(out,r);print('initial',w.fingerprint,implementation_fingerprint(),w.closed)
 w.capture(out); print('capture',w.closed)
 for a in r['actions']:
  out.step(a);w.capture(out);print('act',w.closed,implementation_fingerprint()==w.fingerprint)
 print('conditions',semantic_digest(r)==w.snapshot_hash,len(w.states),len(out.actions)+1,out.state_id==r['stateId'])
 out.scene.require_cache_source();print('sourceok');print('manifestbytes',len(_encode({'schema':'s','sceneId':out.scene.identity,'implementation':w.fingerprint,'snapshotHash':w.snapshot_hash,'states':w.states,'history':out.history})))
 w.publish(out,r);print('pub',w.closed,c.statistics)

with tempfile.TemporaryDirectory() as d:
 c=VerifiedSurgicalCache(Path(d)/'c');out=SurgicalEngine.restore(e.scene,r,use_cache=True,verified_cache=c);print('RESTORE',c.statistics,c.last_skip,getattr(c,'last_components',None))
