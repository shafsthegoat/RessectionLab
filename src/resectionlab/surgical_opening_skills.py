"""Promotable stock-source proposal/macro functions; no private physics/search.

Outer controller owns frozen task/eligibility, frontier, budget, selection and
branch publication. These functions reuse authoritative checked native skills.
A refused macro's accepted internal prefix must never be published as a plan.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .surgical_motion import swept_envelopes
from .surgical_tools import TOOLS
from .surgical_staged_skills import SkillAction,checked_actions,checked_kerf_motion,_finite_pose


def stock_kerf_options(scene_id,route,*,max_depth_mm):
    """Fixed20-entry roll/offset family from a visible source-defined ray.

    This is a proposal inventory, never a source removal or feasibility proof.
    The dimensions are the existing native catalog dimensions.
    """
    if (type(scene_id) is not str or not scene_id or type(max_depth_mm) not in (int,float)
            or not math.isfinite(max_depth_mm) or not .12 <= max_depth_mm <= 24.):
        raise ValueError('source binding and bounded opening depth required')
    if (type(route) is not dict or set(route) != {'A','quaternion'}
            or np.asarray(route['A']).shape != (3,) or np.asarray(route['quaternion']).shape != (4,)
            or not np.isfinite([*route['A'],*route['quaternion']]).all()
            or abs(np.linalg.norm(route['quaternion'])-1.) > 1e-6):
        raise ValueError('Explicit finite source placement and normalized orientation required')
    base=Rotation.from_quat(route['quaternion']);axis=base.apply([0,0,1]);u,v=base.apply([[1,0,0],[0,1,0]]);a0=np.asarray(route['A']);intervals=int(max_depth_mm/.12);rows=[]
    for roll in (0.,45.,90.,135.):
        for ox,oy in ((0.,0.),(.35,0.),(-.35,0.),(0.,.35),(0.,-.35)):
            q=(base*Rotation.from_rotvec([0,0,np.deg2rad(roll)])).as_quat().tolist();a=a0+ox*u+oy*v
            rows.append({'id':f'roll{roll:g}-offset{ox:g},{oy:g}','sourceSceneId':scene_id,'outsidePose':{'position':a.tolist(),'quaternion':q},'path':[{'position':(a+axis*.12*j).tolist(),'quaternion':q} for j in range(1,intervals+1)],'microIntervalMm':.12,'secondsPerInterval':.09,'rollClearDeg':90.,'rollMicroIntervalDeg':.1,'secondsPerRollInterval':.01,'maxRollPosesPerAction':300})
    return rows


def _validate_option(engine, option):
    """The declared finite inventory is not a way to resize the stock tool."""
    expected = {'id','sourceSceneId','outsidePose','path','microIntervalMm','secondsPerInterval',
                'rollClearDeg','rollMicroIntervalDeg','secondsPerRollInterval','maxRollPosesPerAction'}
    if type(option) is not dict or set(option) != expected or option['sourceSceneId'] != engine.scene.identity:
        raise ValueError('Exact source-bound stock opening option required')
    if (type(option['path']) is not list or not 1 <= len(option['path']) <= 200
            or option['microIntervalMm'] != .12 or option['secondsPerInterval'] != .09
            or option['rollClearDeg'] != 90. or option['rollMicroIntervalDeg'] != .1
            or option['secondsPerRollInterval'] != .01 or option['maxRollPosesPerAction'] != 300):
        raise ValueError('Stock opening option exceeds the declared finite family')
    outside = _finite_pose(option['outsidePose'])
    quaternion = np.asarray(outside['quaternion'])
    if abs(np.linalg.norm(quaternion)-1.) > 1e-6:
        raise ValueError('Stock opening requires a normalized source orientation')
    axis = Rotation.from_quat(quaternion).apply([0., 0., 1.])
    for interval, pose in enumerate(option['path'], 1):
        pose = _finite_pose(pose)
        expected_position = np.asarray(outside['position'])+axis*.12*interval
        if (not np.allclose(pose['position'], expected_position, rtol=0., atol=1e-8)
                or not np.allclose(pose['quaternion'], quaternion, rtol=0., atol=1e-8)):
            raise ValueError('Stock opening path exceeds the declared straight .12 mm family')
    slot = engine.state['slots']['right']; params = engine.parameters(slot)
    if (slot['instrument'] != 'craniotome' or params['radiusMm'] != TOOLS['craniotome']['radiusMm']
            or params['depthMm'] != 2. or engine.state.get('toolMechanicsRevision') != 2):
        raise ValueError('STOCK_KERF_GEOMETRY_REQUIRED')


def source_material_shortlist(engine,option):
    """Refresh retained working-envelope contacts; native still decides safety.

    Stop BEFORE unsupported current source material, rather than clipping a
    material/protected/domain refusal. Shaft and other-tool checks stay native.
    """
    _validate_option(engine, option)
    path=[];previous=option['outsidePose'];parameters=engine.parameters(engine.state['slots']['right'])
    for endpoint in option['path']:
        hit=set()
        for a,b,radius,part in swept_envelopes(TOOLS['craniotome'],parameters,previous,endpoint):
            if part!='shaft':hit.update(engine.touched(a,b,radius))
        if any(engine.by_label[int(engine.scene.labels.flat[cell])]['material']!='bone' for cell in hit):break
        path.append(copy.deepcopy(endpoint));previous=endpoint
    return path


def checked_stock_opening_macro(engine,option,*,limits,meter,check=lambda:None,phase='patient.stock_opening'):
    """Shallow kerf -> gradual roll -> active bone return -> full withdrawal.

    Exact public preview/step, full source anatomy/domain, stock geometry and
    caller caps govern every part. Active return is ADDITIONAL tissue cutting,
    charged to the ledger. Only a complete accepted macro may be retained.
    Tool exchange remains a separately checked outside-only outer skill.
    """
    _validate_option(engine, option)
    if (engine.state['holds'] or any(slot['inserted'] or slot['active'] or slot['grip']
                                    for slot in engine.state['slots'].values())):
        raise ValueError('COMPLETE_CLEANUP_REQUIRED_BEFORE_STOCK_OPENING')
    bone = tuple(row['id'] for row in engine.scene.regions if row['material'] == 'bone')
    if limits.allowed_removed_label_ids is not None:
        bone = tuple(sorted(set(bone) & set(limits.allowed_removed_label_ids)))
    if not bone: raise ValueError('Stock opening macro has no permitted bone label')
    from dataclasses import replace
    limits = replace(limits, allowed_removed_label_ids=bone)
    initial=engine.metrics();source_state=engine.state_id;parts=[];calls=meter.calls
    def action(op,duration=.1,**kw):return {'slot':'right','operation':op,'duration':duration,**kw}
    def finish(candidate,status,reason=None):
        return candidate,{'status':status,'reason':reason,'sourceStateId':source_state,'finalStateId':candidate.state_id,'optionId':option['id'],'parts':parts,'physicalCalls':meter.calls-calls,'completeMacroAdmitted':status=='accepted','retentionRole':'accepted_native_skill' if status=='accepted' else 'stopped_native_prefix_not_admitted','targetIncrementMm3':candidate.metrics()['targetRemovedMm3']-initial['targetRemovedMm3'],'nonTargetIncrementMm3':candidate.metrics()['nonTargetRemovedMm3']-initial['nonTargetRemovedMm3'],'activeReturnIsTreatment':True}
    outside=copy.deepcopy(option['outsidePose'])
    candidate,receipt=checked_actions(engine,[SkillAction(action('place',pose=outside),require_endpoint=True)],limits=limits,meter=meter,check=check,phase=phase+'.place');parts.append(receipt)
    if receipt['status']!='accepted':return finish(candidate,receipt['status'],receipt['reason'])
    path=source_material_shortlist(candidate,option)
    if not path:return finish(candidate,'refused','SOURCE_MATERIAL_SHORTLIST_EMPTY')
    forward=action('apply',min(30.,len(path)*option['secondsPerInterval']),path=path)
    candidate,receipt=checked_kerf_motion(candidate,SkillAction(forward),limits=limits,meter=meter,check=check,phase=phase+'.forward');parts.append(receipt)
    if receipt['status']!='accepted':return finish(candidate,receipt['status'],receipt['reason'])
    # Use only the exact newly accepted forward path, never requested endpoints.
    path=copy.deepcopy(candidate.actions[-1]['path']);endpoint=candidate.state['slots']['right'];position=list(endpoint['position']);base=Rotation.from_quat(endpoint['quaternion']);steps=int(option['rollClearDeg']/option['rollMicroIntervalDeg'])
    rollpath=[{'position':position,'quaternion':(base*Rotation.from_rotvec([0,0,np.deg2rad(option['rollClearDeg']*j/steps)])).as_quat().tolist()} for j in range(1,steps+1)]
    chunk=option['maxRollPosesPerAction'];actions=[SkillAction(action('apply',len(rollpath[j:j+chunk])*option['secondsPerRollInterval'],path=rollpath[j:j+chunk]),require_endpoint=True) for j in range(0,len(rollpath),chunk)]
    candidate,receipt=checked_actions(candidate,actions,limits=limits,meter=meter,check=check,phase=phase+'.roll');parts.append(receipt)
    if receipt['status']!='accepted':return finish(candidate,receipt['status'],receipt['reason'])
    q=list(candidate.state['slots']['right']['quaternion']);outside['quaternion']=q;returnpath=[{**pose,'quaternion':q} for pose in path[-2::-1]]+[outside]
    candidate,receipt=checked_actions(candidate,[SkillAction(action('apply',min(30.,len(path)*option['secondsPerInterval']),path=returnpath),require_endpoint=True)],limits=limits,meter=meter,check=check,phase=phase+'.active_return');parts.append(receipt)
    if receipt['status']!='accepted':return finish(candidate,receipt['status'],receipt['reason'])
    candidate,receipt=checked_actions(candidate,[SkillAction(action('withdraw',.1,path=[outside]),require_endpoint=True)],limits=limits,meter=meter,check=check,phase=phase+'.exit');parts.append(receipt)
    if receipt['status']!='accepted':return finish(candidate,receipt['status'],receipt['reason'])
    if any(slot['inserted'] or slot['active'] or slot['grip'] for slot in candidate.state['slots'].values()) or candidate.state['holds']:
        return finish(candidate,'refused','COMPLETE_MACRO_WITHDRAWAL_REQUIRED')
    return finish(candidate,'accepted')
