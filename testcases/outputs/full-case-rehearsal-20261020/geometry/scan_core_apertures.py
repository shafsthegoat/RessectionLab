import sys,json,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.surgical_motion import envelopes
from resectionlab.surgical_tools import TOOLS
path='testcases/outputs/full-case-rehearsal-20261020/geometry/subvoxel-frontier-516.ressectionlab';c,a,w=load_workspace(path);tb=a['surgical_toolbox'];sn=tb['branches'][tb['branchId']];s=load_environment(path);p,g=restore_protection_contract(s,sn['protectionContract']);e=SurgicalEngine(s,protection=p,guard_config=g);e.state=copy.deepcopy(sn['checkpoint']['state']);e.actions=copy.deepcopy(sn['actions']);owner=e.occupancy();state=e.state['slots']['left'];start=np.array(state['position']);rows=[]
for retreat in [2,4,8,12,16,20]:
 point=start+[0,0,retreat]
 for radius in [1.5,2.5,4,6]:
  parts=envelopes(TOOLS['cusa'],{**e.parameters(state),'radiusMm':radius},point,state['quaternion']);work=set();shaft=set()
  for x,y,r,part in parts:(shaft if part=='shaft' else work).update(owner.material_ids(e.cells(x,y,r)))
  ids=owner.material_ids(e.cells(point,point,radius));rows.append({'retreatMm':retreat,'radiusMm':radius,'sourceLabelsWorking':{str(label):int(sum(s.labels.ravel()[i]==label for i in work)) for label in [1,2,3,4]},'shaftCells':len(shaft),'tipCells':len(ids)})
print(json.dumps(rows,indent=2));Path('testcases/outputs/full-case-rehearsal-20261020/geometry/core_aperture_scan.json').write_text(json.dumps({'geometryOnly':True,'notExecuted':True,'sourceState':sn['stateId'],'rows':rows},indent=2))
