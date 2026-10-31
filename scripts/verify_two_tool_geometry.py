"""Independent complete-geometry audit of exported two-tool action programs.

Consumes actionProgram, never the strategy compiler. Replays exact actual
microsteps and checks whole capsules through the separate geometry.check_route.
Active CUSA working contact may touch declared target; shafts/non-target cells
and other tools remain excluded. A post-contact axial retreat has a separate
whole-cell half-space proof. No patient validity or optical/suction benefit.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.core import semantic_digest
from resectionlab.geometry import GeometryScene,ToolGeometry,ToolPose,SphereObstacle,check_route
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace

def require(ok,reason):
 if not ok:raise ValueError('INDEPENDENT_TWO_TOOL: '+reason)
def physical_hash(e):return semantic_digest({k:v for k,v in e.state.items() if k!='observations'})
def pose(s):return ToolPose(s['position'],Rotation.from_quat(s['quaternion']).apply([0.,0.,1.]))
def geometry(tool,shaft=False):
 require(tool in ('cusa','frazier_suction'),'unsupported tool geometry')
 return ToolGeometry('independent-'+tool+('-shaft' if shaft else ''),.5 if shaft else 1.5,.5,40.,tip_length_mm=1e-10 if tool=='cusa' else 2.)

class CaptureEngine(SurgicalEngine):
 def _substep(self,key,op,p,q,active,dt):
  require(not self.state['holds'] and not self.state.get('deformation') and not self.state.get('detachedFragments'),'geometry-changing state outside static audit')
  before=copy.deepcopy(self.state['slots']);removed=sorted(set(self.state['removed'])-self._review_baseline_removed)
  events=super()._substep(key,op,p,q,active,dt)
  self._review_trace.append({'hand':key,'operation':op,'active':active,'before':before,'after':copy.deepcopy(self.state['slots']),'removedBefore':removed,'dt':dt})
  return events

def check_geometry(trace,scene,base_removed,protection=None):
 target_ids={r['id'] for r in scene.regions if r.get('target') and r['material']=='tumor'}
 points=[]
 for frame in trace:
  for when in ('before','after'):
   for s in frame[when].values():
    a=pose(s);points.extend([a.tip_mm,a.tip_mm-40*a.axis_unit])
 if not points:return {'passed':True,'checks':0,'microsteps':0,'failures':[],'scope':'empty STOP program'}
 vox=scene.inverse[:3,:3]@np.asarray(points).T+scene.inverse[:3,3,None]
 padding=np.linalg.norm(scene.inverse[:3,:3],axis=1)*5.
 lo=np.maximum(0,np.floor(vox.min(axis=1)-padding).astype(int));hi=np.minimum(np.array(scene.shape)-1,np.ceil(vox.max(axis=1)+padding).astype(int))
 # 5mm padding exceeds complete 1.5mm working radii + native per-step inflation
 # + voxel support. Every non-placement rotation is explicitly refused below.
 slices=tuple(slice(int(a),int(b)+1) for a,b in zip(lo,hi));labels=scene.labels[slices]
 retained=labels!=0
 for cell in base_removed:
  v=np.asarray(np.unravel_index(cell,scene.shape))-lo
  if np.all(v>=0) and np.all(v<retained.shape):retained[tuple(v)]=False
 affine=scene.affine.copy();affine[:3,3]=(scene.affine@np.r_[lo,1.])[:3]
 spheres=()
 if protection is not None:
  corners=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
  cell_radius=float(np.max(np.linalg.norm(corners@scene.affine[:3,:3].T,axis=1)))
  spheres=tuple(SphereObstacle(point,cell_radius+protection.clearance_mm,'declared-protection-'+str(int(cell))) for point,cell in zip(scene.points(protection.cells),protection.cells))
 cache={};failures=[];checked=0;seen_active=set();exit_origin={};exit_axis={};exit_cells=set();min_margin=None
 def world(removed,allow_target):
  key=(tuple(removed),allow_target)
  if key not in cache:
   mask=retained.copy()
   for cell in removed:
    v=np.asarray(np.unravel_index(cell,scene.shape))-lo
    if np.all(v>=0) and np.all(v<mask.shape):mask[tuple(v)]=False
   if allow_target:mask &= ~np.isin(labels,list(target_ids))
   cache[key]=GeometryScene(mask,affine,sphere_obstacles=spheres,enforce_tip_in_bounds=False)
  return cache[key]
 for i,frame in enumerate(trace):
  acting=frame['hand'];bs=frame['before'][acting];a=frame['after'][acting];other=next(h for h in frame['before'] if h!=acting);os=frame['before'][other]
  active_cusa=frame['active'] and a['instrument']=='cusa'
  if active_cusa:seen_active.add(acting)
  exit_contact=acting in seen_active and a['instrument']=='cusa' and frame['operation'] in ('deactivate','withdraw')
  allowed_target=active_cusa or exit_contact
  if frame['operation']!='place':
   require(float(pose(bs).axis_unit@pose(a).axis_unit)>1-1e-12,'this bounded auditor requires fixed-axis trajectories')
  for hand,s in frame['after'].items():
   require(s['instrument'] in ('cusa','frazier_suction'),'unknown instrument')
   require(s.get('parameters',{}).get('radiusMm',1.5)==1.5,'catalog radius changed')
   if s['instrument']=='frazier_suction':require(s.get('parameters',{}).get('depthMm',2.)==2.,'suction working length changed')
  checks=[('full',geometry(a['instrument']),(pose(bs),pose(a)),world(frame['removedBefore'],allowed_target),[(geometry(os['instrument']),pose(os))])]
  if allowed_target:checks.append(('shaft',geometry('cusa',True),(pose(bs),pose(a)),world(frame['removedBefore'],False),()))
  # Static other hand is independently checked too; only previously active
  # CUSA working target contact is retained, never shaft/non-target overlap.
  other_controlled=other in seen_active and os['instrument']=='cusa'
  checks.append(('stationary-other-full',geometry(os['instrument']),(pose(os),),world(frame['removedBefore'],other_controlled),()))
  if other_controlled:checks.append(('stationary-other-shaft',geometry('cusa',True),(pose(os),),world(frame['removedBefore'],False),()))
  for name,tool,trajectory,ws,others in checks:
   result=check_route(tool,trajectory,ws,other_tools=others);checked+=1
   if not result.feasible:failures.append({'frame':i,'check':name,'operation':frame['operation'],'failures':[vars(x) for x in result.failures]})
   if exit_contact and name=='full':
    if acting not in exit_origin:exit_origin[acting]=pose(bs).tip_mm;exit_axis[acting]=pose(bs).axis_unit
    delta=pose(a).tip_mm-pose(bs).tip_mm;axis=exit_axis[acting]
    if np.linalg.norm(np.cross(delta,axis))>1e-8 or float(delta@axis)>1e-12:failures.append({'frame':i,'check':'nonaxial-or-forward-contact-exit'})
    support=.5*float(np.sum(np.abs(scene.affine[:3,:3].T@axis)))
    for v in result.swept_voxel_indices:
     source=int(np.ravel_multi_index(v+lo,scene.shape))
     if int(labels[tuple(v)]) not in target_ids or not retained[tuple(v)] or source in frame['removedBefore']:continue
     exit_cells.add(source);center=(scene.affine@np.r_[v+lo,1.])[:3];margin=float((center-exit_origin[acting])@axis)-support
     min_margin=margin if min_margin is None else min(min_margin,margin)
     if margin<0.:failures.append({'frame':i,'check':'exit-target-cell-not-wholly-ahead','sourceCell':source,'marginMm':margin})
 return {'passed':not failures,'checks':checked,'microsteps':len(trace),'failures':failures[:30],'failureCount':len(failures),'cropLo':lo.tolist(),'cropHi':hi.tolist(),'cropPaddingMm':5.,'exitTargetContactSourceCells':sorted(exit_cells),'minimumExitCellHalfspaceMarginMm':min_margin,
 'proof':'Fixed-axis exit moves away from every retained contacted target cell lying wholly in its forward half-space. Whole-cell shaft/non-target/other-tool checks remain active. Whole tool is checked against all retained tissue during inactive access/suction movement.',
 'outsideImageAnatomy':'unknown; original source field only'}

def verify(report,source,*,branch=None,revision=1):
 source=Path(source).resolve();require(source.is_relative_to(ROOT/'testcases'),'source must stay in ignored testcases')
 if not isinstance(report,dict):
  report_path=Path(report)
  if report_path.suffix=='.ressectionlab':
   _,saved,_=load_workspace(report_path);history=saved['surgical_toolbox'].get('strategyComparisons',[])
   require(type(revision) is int and 1<=revision<=len(history),'choose a saved study revision')
   item=history[revision-1];report=item['report']
   if report_path.resolve()==source and branch is None:branch=item['sourceBranchId']
  else:report=json.loads(report_path.read_text())
 sha=hashlib.sha256(source.read_bytes()).hexdigest();_,art,_=load_workspace(source);record=art['surgical_toolbox'];scene=load_environment(source)
 snap=record['branches'][branch or record['branchId']];count=report['prefixActions']
 require(len(snap['actions'])>=count,'source prefix is too short')
 protection=config=None
 if snap.get('protectionContract'):
  from resectionlab.surgical_rl_protection import restore_protection_contract
  protection,config=restore_protection_contract(scene,snap['protectionContract'])
 baseline=SurgicalEngine(scene,protection=protection,guard_config=config)
 for action in snap['actions'][:count]:require(baseline.step(copy.deepcopy(action))['status']!='rejected','source replay refused')
 require(scene.identity==report['sourceSceneId'] and baseline.state_id==report['startStateId'],'report source/baseline differs')
 require(set(scene.properties) <= {'tissuePhysicsV2','cusaV3'} and 'tissuePhysicsV2' in scene.properties,'only the declared voxel V2/CUSA profiles are supported')
 require(baseline.state.get('toolMechanicsRevision')==2 and baseline.state.get('physicalClockRevision')==1,'mechanics2/shared clock1 required')
 require(not baseline.state['holds'] and not baseline.state.get('deformation') and not baseline.state.get('detachedFragments') and 'patientPosition' not in baseline.state,'released undeformed baseline required')
 require(all(not s['active'] and not s['inserted'] and not s['grip'] for s in baseline.state['slots'].values()),'both baseline tools must be inactive/withdrawn')
 model=baseline.state.get('deformationModel',{})
 require(not any(model.get(k) for k in ('active','anchors','velocities')),'pending mechanical state outside static audit')
 plan=report['publicPlan'];require(semantic_digest({k:v for k,v in plan.items() if k!='publicPlanSeal'})==plan['publicPlanSeal'],'public plan seal differs')
 target_ids={r['id'] for r in scene.regions if r.get('target') and r['material']=='tumor'};base_removed=set(baseline.state['removed']);rows=[]
 for alt in plan['alternatives']:
  start=time.monotonic();trial=baseline.clone();trial.__class__=CaptureEngine;trial._review_baseline_removed=base_removed;trial._review_trace=[];receipts=[];state_matches=[]
  for index,action in enumerate(alt['actionProgram']):
   result=trial.step(copy.deepcopy(action));finite_partial=action['slot']=='left' and action['operation']=='apply' and result['status']=='partial' and result.get('reasonCode')=='TISSUE_ASPIRATION_IN_PROGRESS' and abs(result['elapsedSeconds']-action['duration'])<=1e-8
   require(result['status']=='accepted' or finite_partial,'exported action refused or stopped before its declared interval')
   expected=alt.get('physicalStateHashes',[]);match=index<len(expected) and physical_hash(trial)==expected[index];state_matches.append(match)
   receipts.append({'status':result['status'],'reason':result.get('reasonCode'),'elapsedSeconds':result['elapsedSeconds']})
  geometry_result=check_geometry(trial._review_trace,scene,base_removed,protection)
  new=sorted(set(trial.state['removed'])-base_removed);non_target=[c for c in new if int(scene.labels.flat[c]) not in target_ids]
  cold=SurgicalEngine.restore(scene,trial.snapshot());replay=cold.snapshot()==trial.snapshot()
  released=not trial.state['holds'] and all(not s['active'] and not s['inserted'] and not s['grip'] for s in trial.state['slots'].values())
  row={'id':alt['id'],'passed':geometry_result['passed'] and all(state_matches) and not non_target and replay and released,'independentGeometry':geometry_result,'nativePhysicalStatesMatchExport':all(state_matches),'nativeActionReceipts':receipts,'newRemovedSourceCells':new,'newNonTargetSourceCells':non_target,'exactColdReplay':replay,'finalReleased':released,'finalStateId':trial.state_id,'wallSeconds':time.monotonic()-start}
  rows.append(row);print(alt['id'],'passed',row['passed'],'checks',geometry_result['checks'],'failureCount',geometry_result.get('failureCount',0),flush=True)
 unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==sha
 return {'schema':'resectionlab.two-tool-independent-geometry.v1','source':str(source),'sourceSha256':sha,'sourceUnchanged':unchanged,'baselineStateId':baseline.state_id,'publicPlanSeal':plan['publicPlanSeal'],'alternatives':rows,'passed':unchanged and all(x['passed'] for x in rows),'scope':'Exact exported native programs; independent complete capsules and controlled target retreat; no clinical, visibility, liquid-suction-benefit or complete-anatomy qualification'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('source',type=Path);p.add_argument('--branch');p.add_argument('--revision',type=int,default=1);p.add_argument('--output',required=True,type=Path);a=p.parse_args();require(not a.output.exists(),'preserve earlier evidence')
 require(a.output.resolve().is_relative_to(ROOT/'testcases'),'keep audit outputs in ignored testcases')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 value=verify(a.report,a.source,branch=a.branch,revision=a.revision);a.output.write_text(json.dumps(value,indent=2)+'\n')

 if not value['passed']:raise SystemExit(1)
