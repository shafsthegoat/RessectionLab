import numpy as np
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.full_case_assist import assistant_field_candidate
l=np.zeros((64,64,40),np.uint16);l[41:44,19:22,18]=3;l[41:44,19:22,15]=2;l[19:22,19:22,15]=2;l[40:45,18:23,14]=1
s=SurgicalScene(l,np.eye(4),[{'id':1,'name':'brain','material':'soft_tissue'},{'id':2,'name':'target','material':'tumor','target':True},{'id':3,'name':'bone','material':'bone'}],{}, {'tissuePhysicsV2':default_profile()});e=SurgicalEngine(s)
def pose(x,z):return{'position':[x,20.,z],'quaternion':[1.,0.,0.,0.]}
def step(op,**kw):
 a={'slot':'left','operation':op,'duration':.2,**kw};r=e.step(a);print(op,r['status'],r.get('reason'),r['metrics']['targetRemovedMm3'],r['metrics']['nonTargetRemovedMm3'],e.state['slots']['left']['position']);return r
step('place',pose=pose(42,47),parameters={'radiusMm':1.5,'forceN':20.,'rpm':80000.,'duty':1.})
step('apply',duration=3.,pose=pose(42,17))
step('withdraw');step('exchange',instrument='cusa');step('place',pose=pose(42,47),parameters={'radiusMm':1.5,'forceN':0.});step('apply',duration=1.,pose=pose(42,17));step('inspect',parameters={'radiusMm':.5});step('withdraw');step('place',pose=pose(20,47),parameters={'radiusMm':.6,'forceN':0.})
print('DEBRIS',e.state['debrisMm3']);print('FIELD',assistant_field_candidate(e,{'entryPose':pose(20,15)}))
